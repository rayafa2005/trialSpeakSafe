"""
Tatvaani - TatvaNet Training Script
=====================================
Run:
  From scratch: py -3.10 training/train.py --reset_best
  Resume run:   py -3.10 training/train.py --resume checkpoints/last_checkpoint.pt
  Fine-tune:    py -3.10 training/train.py --finetune checkpoints/best_model.pt --lr 0.0001 --epochs 10
"""

import sys
import os
import argparse
import subprocess
import time
import traceback
from pathlib import Path

# Force unbuffered UTF-8 output so Windows PowerShell and Kaggle print every update immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torch.amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_curve
from torch.utils.tensorboard import SummaryWriter

# Path setup
SCRIPT_DIR   = Path(__file__).resolve().parent
ML_ROOT      = SCRIPT_DIR.parent
PROJECT_ROOT = ML_ROOT.parent

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from model.tatvanet import TatvaNet
from training.focal_loss import FocalLoss
from training.augment import AudioAugmenter

CONFIG = {
    "train_manifest":      "data/train_manifest.csv",
    "val_manifest":        "data/val_manifest.csv",
    "checkpoint_dir":      "checkpoints",
    "log_dir":             "runs/tatvanet",
    "sample_rate":         16000,
    "min_duration":        0.5,
    "max_duration":        2.0,
    "target_samples":      32000,  # 2.0 seconds at 16kHz
    "epochs":              50,
    "batch_size":          64,     # Optimized for T4 / RTX 4060 VRAM
    "lr":                  1e-3,
    "lr_min":              1e-5,
    "warmup_epochs":       3,
    "early_stop_patience": 7,
    "grad_clip":           1.0,
    "num_workers":         0,      # Safe default for Windows; override with --num_workers 4 on Linux/Kaggle
    "pin_memory":          False,
    "num_classes":         2,      # 0=Safe/Real, 1=Danger/Fake
    "augment_train":       True,
    "log_interval":        20,     # Print progress every 20 batches
}


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    try:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        return float((fpr[idx] + fnr[idx]) / 2.0)
    except Exception:
        return 0.5


def load_audio_robust(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Loads any audio container (.wav, .flac, .mp3, .m4a, .mp4, .opus, .ogg) into a
    mono float32 tensor [1, N] at target_sr.
    Tries soundfile first, then torchaudio, then ffmpeg pipe fallback.
    """
    ext = os.path.splitext(audio_path)[1].lower()

    # Fast path for standard uncompressed/lossless formats
    if ext not in (".m4a", ".mp4", ".opus", ".aac", ".webm"):
        try:
            samples, sr = sf.read(audio_path, dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            waveform = torch.from_numpy(samples).unsqueeze(0)
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            return waveform
        except Exception:
            pass

        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            return waveform
        except Exception:
            pass

    # Universal ffmpeg fallback for .m4a, .opus, .mp4, or tricky containers
    cmd = [
        "ffmpeg", "-v", "error", "-i", audio_path,
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(target_sr), "pipe:1"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode == 0 and len(res.stdout) >= 4:
        arr = np.frombuffer(res.stdout, dtype=np.float32).copy()
        return torch.from_numpy(arr).unsqueeze(0)

    raise RuntimeError(f"Failed to decode audio file: {audio_path}")


def extract_active_2s_window(
    waveform: torch.Tensor,
    target_n: int = 32000,
    random_crop: bool = False
) -> torch.Tensor:
    """
    1. Removes DC offset.
    2. Strips leading and trailing dead air using 10ms (160-sample) frame RMS
       (threshold = 5% of peak frame RMS, i.e., ~-26 dB) if remaining voiced span >= 0.5s.
    3. Selects a random 2.0s window during training (`random_crop=True`) or the
       highest-energy 2.0s window during validation/inference (`random_crop=False`).
    4. Peak-normalizes the selected 2.0s window.
    """
    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    n = waveform.shape[-1]

    # Strip leading/trailing silence on clips > 0.6s
    frame_len = 160
    if n >= frame_len * 6:
        n_frames = n // frame_len
        frames = waveform[0, :n_frames * frame_len].view(n_frames, frame_len)
        frame_rms = frames.pow(2).mean(dim=1).sqrt()
        peak_rms = frame_rms.max()
        if peak_rms > 1e-6:
            active_idx = torch.nonzero(frame_rms > (peak_rms * 0.05), as_tuple=False).squeeze(-1)
            if active_idx.numel() > 0:
                start_sample = max(0, int(active_idx[0].item()) * frame_len - frame_len)
                end_sample = min(n, (int(active_idx[-1].item()) + 2) * frame_len)
                if (end_sample - start_sample) >= 8000:  # Keep if at least 0.5s of active speech
                    waveform = waveform[:, start_sample:end_sample]
                    n = waveform.shape[-1]

    # Pad or crop to target_n (32,000 samples)
    if n < target_n:
        waveform = F.pad(waveform, (0, target_n - n))
    elif n > target_n:
        if random_crop:
            start = torch.randint(0, n - target_n + 1, (1,)).item()
            waveform = waveform[:, start:start + target_n]
        else:
            # Deterministic max-energy 2.0s window (stride = 4000 samples = 0.25s)
            stride = 4000
            windows = waveform.unfold(dimension=-1, size=target_n, step=stride)  # (1, n_win, target_n)
            if windows.shape[1] > 0:
                win_energy = windows.pow(2).mean(dim=-1).squeeze(0)
                best_idx = int(torch.argmax(win_energy).item())
                start = best_idx * stride
                waveform = waveform[:, start:start + target_n]
            else:
                waveform = waveform[:, :target_n]

    # Peak-normalize the final 2.0s window
    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    peak = waveform.abs().max()
    if peak > 1e-6:
        waveform = waveform / peak

    return waveform


class TatvaaniDataset(Dataset):
    def __init__(self, manifest_path: str, config: dict, augment: bool = False):
        self.config    = config
        self.augment   = augment
        self.sr        = config["sample_rate"]
        self.target_n  = config["target_samples"]
        self.augmenter = AudioAugmenter(sample_rate=self.sr) if augment else None

        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = ML_ROOT / manifest_path

        print(f"[Dataset] Loading manifest: {manifest_path}", flush=True)
        df = pd.read_csv(manifest_path)

        # Fast path resolution without slow synchronous disk syscalls
        root_str = str(ML_ROOT)
        df["abs_path"] = df["path"].apply(
            lambda p: os.path.join(root_str, p) if not os.path.isabs(p) else p
        )

        # Map labels: 0 -> 0 (Safe/Real), anything > 0 -> 1 (Danger/Fake)
        df["binary_label"] = df["label"].apply(lambda l: 1 if int(l) > 0 else 0)
        self.df = df
        print(f"[Dataset] Loaded {len(self.df):,} samples from {manifest_path.name}", flush=True)
        print(f"[Dataset] Binary label distribution: {dict(self.df['binary_label'].value_counts().sort_index())}", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        label      = int(row["binary_label"])
        audio_path = str(row["abs_path"])

        try:
            waveform = load_audio_robust(audio_path, target_sr=self.sr)
        except Exception as e:
            print(f"[Dataset WARNING] Could not decode {audio_path}: {e}", flush=True)
            return torch.zeros(1, self.target_n), label

        waveform = extract_active_2s_window(
            waveform,
            target_n=self.target_n,
            random_crop=self.augment
        )

        if self.augment and self.augmenter is not None:
            waveform = self.augmenter(waveform)

        return waveform, label


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    return torch.stack(waveforms), torch.tensor(labels, dtype=torch.long)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


@torch.no_grad()
def validate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_scores = []
    all_labels = []
    n_batches = len(loader)
    use_amp = device.type == "cuda"

    print("\n  [Validation] Evaluating on unseen speakers...", end="", flush=True)

    for val_idx, (waveforms, labels) in enumerate(loader):
        waveforms = waveforms.to(device)
        labels    = labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out    = model(waveforms)
            logits = out["logits"]
            loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        probs       = torch.softmax(logits, dim=-1)
        danger_prob = probs[:, 1].cpu().numpy()  # Binary fake probability
        all_scores.extend(danger_prob.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        if (val_idx + 1) % 50 == 0 or (val_idx + 1) == n_batches:
            pct = 100.0 * (val_idx + 1) / n_batches
            print(f"\r  [Validation] Progress: [{pct:5.1f}%] ({val_idx+1}/{n_batches} batches)", end="", flush=True)

    print()  # newline
    avg_loss = total_loss / max(len(loader.dataset), 1)
    eer      = compute_eer(np.array(all_scores), np.array(all_labels))
    model.train()
    return avg_loss, eer


def train(
    config: dict,
    resume_path: str = None,
    finetune_path: str = None,
    reset_best: bool = False,
    debug: bool = False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    checkpoint_dir = Path(config["checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ML_ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = ML_ROOT / config["log_dir"]
    writer  = SummaryWriter(log_dir=str(log_dir))
    print(f"[Train] TensorBoard: tensorboard --logdir {log_dir}", flush=True)

    # Datasets
    print("\n[Train] Loading datasets...", flush=True)
    train_dataset = TatvaaniDataset(config["train_manifest"], config, augment=config["augment_train"])
    val_dataset   = TatvaaniDataset(config["val_manifest"],   config, augment=False)

    if debug:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, list(range(min(128, len(train_dataset)))))
        val_dataset   = Subset(val_dataset,   list(range(min(64, len(val_dataset)))))
        config["epochs"] = 2

    batch_size  = config["batch_size"]
    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2,
        shuffle=False, num_workers=num_workers, pin_memory=False,
        collate_fn=collate_fn,
    )

    # Verify first batch
    print("[Train] Verifying first batch...", flush=True)
    try:
        wb, lb = next(iter(train_loader))
        print(f"[Train] First batch OK — waveforms: {wb.shape}, labels: {lb.shape}", flush=True)
        print(f"[Train] Label values: {lb.unique().tolist()}", flush=True)
    except Exception as e:
        print(f"[Train] FIRST BATCH FAILED: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    # Model
    model = TatvaNet(sample_rate=config["sample_rate"], n_classes=config["num_classes"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {n_params:,}", flush=True)
    print(f"[Train] Epochs: {config['epochs']}, Batch: {batch_size}, LR: {config['lr']}", flush=True)

    # Compute dynamic class weights to balance Real vs Fake audio
    if hasattr(train_dataset, "df"):
        ds_df = train_dataset.df
    elif hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "df"):
        ds_df = train_dataset.dataset.df
    else:
        ds_df = None

    if ds_df is not None:
        counts = dict(ds_df["binary_label"].value_counts())
    else:
        counts = {0: 1, 1: 1}
    n_real = max(counts.get(0, 1), 1)
    n_fake = max(counts.get(1, 1), 1)
    weight_real = n_fake / (n_real + n_fake) * 2.0
    weight_fake = n_real / (n_real + n_fake) * 2.0
    alpha_tensor = torch.tensor([weight_real, weight_fake], dtype=torch.float32).to(device)
    print(f"[Train] Balanced FocalLoss alpha weights: Real={weight_real:.2f}, Fake={weight_fake:.2f}", flush=True)
    criterion = FocalLoss(gamma=2.0, alpha=alpha_tensor)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(config["epochs"] - config["warmup_epochs"], 1),
        eta_min=config["lr_min"],
    )

    # GradScaler — using torch.amp
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"[Train] Optimizer, Scheduler & GradScaler OK (AMP enabled: {use_amp})", flush=True)

    # Resume / Fine-tune / Baseline EER Gate
    start_epoch = 0
    best_eer    = float("inf")
    no_improve  = 0
    existing_best_path = checkpoint_dir / "best_model.pt"

    if finetune_path is not None:
        fp = Path(finetune_path)
        if not fp.is_absolute() and not fp.exists():
            fp = checkpoint_dir / fp
        if fp.exists():
            print(f"[Train] Fine-tuning from weights in {fp} (fresh optimizer & schedule, epoch 0)", flush=True)
            ckpt = torch.load(fp, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=False)
            if not reset_best:
                print("[Train] Measuring baseline EER of loaded checkpoint on current validation set...", flush=True)
                _, baseline_eer = validate(model, val_loader, criterion, device, config)
                best_eer = baseline_eer
                print(
                    f"[Train] Baseline Val EER on current dataset: {100*best_eer:.2f}% "
                    f"(best_model.pt will ONLY be overwritten if Val EER < {100*best_eer:.2f}%)",
                    flush=True
                )
    elif resume_path is not None:
        rp = Path(resume_path)
        if not rp.is_absolute() and not rp.exists():
            rp = checkpoint_dir / rp
        if rp.exists():
            print(f"[Train] Resuming from {rp}", flush=True)
            ckpt = torch.load(rp, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            no_improve  = 0 if reset_best else ckpt.get("no_improve", 0)
            if reset_best:
                best_eer = float("inf")
                print(f"[Train] Resumed at epoch {start_epoch} with --reset_best (best_eer = inf)", flush=True)
            else:
                print("[Train] Measuring baseline EER of resumed checkpoint on current validation set...", flush=True)
                _, baseline_eer = validate(model, val_loader, criterion, device, config)
                best_eer = baseline_eer
                print(f"[Train] Resumed at epoch {start_epoch}, current dataset baseline EER: {100*best_eer:.2f}%", flush=True)
    elif existing_best_path.exists() and not reset_best and not debug:
        # Even when training from scratch, if an existing best_model.pt is present and --reset_best is not passed,
        # evaluate the existing best_model.pt on the current validation set so we never overwrite it with a worse model!
        try:
            print(f"[Train] Found existing {existing_best_path}. Evaluating its baseline EER on current validation set...", flush=True)
            ref_model = TatvaNet(sample_rate=config["sample_rate"], n_classes=config["num_classes"]).to(device)
            ref_ckpt  = torch.load(existing_best_path, map_location=device)
            ref_model.load_state_dict(ref_ckpt["model_state"], strict=False)
            _, baseline_eer = validate(ref_model, val_loader, criterion, device, config)
            best_eer = baseline_eer
            del ref_model
            print(
                f"[Train] Existing best_model.pt EER on this validation set: {100*best_eer:.2f}% "
                f"(will only overwrite if new model beats {100*best_eer:.2f}%)",
                flush=True
            )
        except Exception as e:
            print(f"[Train] Could not evaluate existing best_model.pt ({e}); starting with best_eer=inf.", flush=True)
            best_eer = float("inf")

    # Training loop
    total_batches = len(train_loader)
    log_interval  = config.get("log_interval", 20)

    print(f"\n{'='*65}", flush=True)
    print(f"STARTING TRAINING — Epochs {start_epoch+1} to {config['epochs']} ({total_batches} batches/epoch)", flush=True)
    print(f"{'='*65}\n", flush=True)

    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        model.train()

        if finetune_path is None and epoch < config["warmup_epochs"]:
            warmup_lr(optimizer, epoch, config["warmup_epochs"], config["lr"])

        epoch_loss = 0.0
        n_correct  = 0
        n_total    = 0

        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels    = labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out    = model(waveforms)
                logits = out["logits"]
                loss   = criterion(logits, labels)

            if torch.isnan(loss):
                print(f"[Train] NaN loss at batch {batch_idx}, skipping.", flush=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * len(labels)
            preds       = logits.argmax(dim=-1)
            n_correct  += (preds == labels).sum().item()
            n_total    += len(labels)

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                speed = (batch_idx + 1) / max(time.time() - epoch_start, 1e-3)
                remaining_batches = total_batches - (batch_idx + 1)
                eta_seconds = remaining_batches / max(speed, 1e-3)
                pct = 100.0 * (batch_idx + 1) / total_batches

                cur_loss = epoch_loss / max(n_total, 1)
                cur_acc  = 100.0 * n_correct / max(n_total, 1)

                print(
                    f"  [Epoch {epoch+1:02d}/{config['epochs']}] [{pct:5.1f}%] "
                    f"({batch_idx+1:4d}/{total_batches}) | "
                    f"Loss: {cur_loss:.4f} | Acc: {cur_acc:5.1f}% | "
                    f"Speed: {speed:4.1f} it/s | ETA: {format_time(eta_seconds)}",
                    flush=True
                )

            if debug and batch_idx >= 1:
                break

        train_loss = epoch_loss / max(n_total, 1)
        train_acc  = n_correct  / max(n_total, 1)

        val_loss, val_eer = validate(model, val_loader, criterion, device, config)

        if finetune_path is not None or epoch >= config["warmup_epochs"]:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        print(
            f"\n>>> Epoch {epoch+1:02d}/{config['epochs']} Summary: "
            f"Train Loss={train_loss:.4f} | Train Acc={100*train_acc:.2f}% | "
            f"Val Loss={val_loss:.4f} | Val EER={100*val_eer:.2f}% | "
            f"LR={get_lr(optimizer):.2e} | Time={format_time(epoch_time)}\n",
            flush=True
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("EER/val",    val_eer,    epoch)
        writer.add_scalar("LR",         get_lr(optimizer), epoch)

        # Update best_eer BEFORE constructing ckpt so best_model.pt stores the exact new best_eer
        is_best = val_eer < best_eer
        if is_best:
            best_eer   = val_eer
            no_improve = 0
        else:
            no_improve += 1

        ckpt = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_eer":        best_eer,
            "no_improve":      no_improve,
            "val_eer":         val_eer,
            "config":          config,
        }
        torch.save(ckpt, checkpoint_dir / "last_checkpoint.pt")

        if is_best:
            torch.save(ckpt, checkpoint_dir / "best_model.pt")
            print(f"  [BEST] *** New Best Model Saved! Val EER: {100*best_eer:.2f}% ***\n", flush=True)
        else:
            print(
                f"  No improvement over best EER {100*best_eer:.2f}% "
                f"({no_improve}/{config['early_stop_patience']})\n",
                flush=True
            )

        if no_improve >= config["early_stop_patience"] and not debug:
            print(f"[Train] Early stopping triggered at epoch {epoch+1}.", flush=True)
            break

    print(f"\n{'='*65}", flush=True)
    print("TRAINING COMPLETE")
    print(f"Best validation EER: {100*best_eer:.2f}%")
    print(f"Best model saved at: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*65}\n", flush=True)
    print("Next step: py -3.10 export/export_tflite.py --checkpoint checkpoints/best_model.pt --clip_duration 2.0", flush=True)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TatvaNet")
    parser.add_argument("--resume",         type=str, default=None, help="Resume interrupted training from checkpoint")
    parser.add_argument("--finetune",       type=str, default=None, help="Fine-tune weights from checkpoint with fresh optimizer & baseline EER gate")
    parser.add_argument("--reset_best",     action="store_true",    help="Ignore any prior checkpoint EER and start best_eer at inf")
    parser.add_argument("--debug",          action="store_true")
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest",   type=str, default=None)
    parser.add_argument("--epochs",         type=int, default=None)
    parser.add_argument("--batch_size",     type=int, default=None)
    parser.add_argument("--lr",             type=float, default=None)
    parser.add_argument("--num_workers",    type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    config = CONFIG.copy()
    if args.train_manifest: config["train_manifest"] = args.train_manifest
    if args.val_manifest:   config["val_manifest"]   = args.val_manifest
    if args.epochs:         config["epochs"]         = args.epochs
    if args.batch_size:     config["batch_size"]     = args.batch_size
    if args.lr:             config["lr"]             = args.lr
    if args.num_workers is not None: config["num_workers"] = args.num_workers
    if args.checkpoint_dir: config["checkpoint_dir"] = args.checkpoint_dir

    train(
        config,
        resume_path=args.resume,
        finetune_path=args.finetune,
        reset_best=args.reset_best,
        debug=args.debug
    )

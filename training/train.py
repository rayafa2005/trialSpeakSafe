"""
Tatvaani - TatvaNet Training Script
=====================================
Run: py -3.11 training/train.py
     py -3.11 training/train.py --resume checkpoints/last_checkpoint.pt
"""

import sys
import os
import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
# PyTorch 2.x AMP API — torch.amp replaces deprecated torch.cuda.amp
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
    "min_duration":        1.0,
    "max_duration":        4.0,
    "target_samples":      48000,
    "epochs":              50,
    "batch_size":          32,
    "lr":                  1e-3,
    "lr_min":              1e-5,
    "warmup_epochs":       3,
    "early_stop_patience": 7,
    "grad_clip":           1.0,
    "num_workers":         0,      # must be 0 on Windows
    "pin_memory":          False,  # False is safer on Windows with num_workers=0
    "num_classes":         3,
    "augment_train":       True,
}


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    try:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        return float((fpr[idx] + fnr[idx]) / 2.0)
    except Exception:
        return 0.5


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

        print(f"[Dataset] Loading manifest: {manifest_path}")
        df = pd.read_csv(manifest_path)

        def resolve_path(p: str) -> Path:
            p = Path(p)
            if p.is_absolute():
                return p
            c = ML_ROOT / p
            if c.exists():
                return c
            c2 = PROJECT_ROOT / p
            if c2.exists():
                return c2
            return c

        df["abs_path"]  = df["path"].apply(resolve_path)
        exists_mask     = df["abs_path"].apply(lambda p: p.exists())
        n_missing       = (~exists_mask).sum()
        if n_missing > 0:
            print(f"[Dataset] WARNING: {n_missing}/{len(df)} files missing, skipping them.")
            df = df[exists_mask].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError("[Dataset] No audio files found on disk.")

        self.df = df
        print(f"[Dataset] Loaded {len(self.df)} samples from {manifest_path.name}")
        print(f"[Dataset] Label distribution: {dict(self.df['label'].value_counts().sort_index())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        label     = int(row["label"])
        audio_path = row["abs_path"]

        try:
            waveform, sr = torchaudio.load(str(audio_path))
        except Exception:
            return torch.zeros(1, self.target_n), label

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sr:
            waveform = T.Resample(sr, self.sr)(waveform)

        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        n = waveform.shape[1]
        if n < self.target_n:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_n - n))
        elif n > self.target_n:
            if self.augment:
                start    = torch.randint(0, n - self.target_n + 1, (1,)).item()
                waveform = waveform[:, start:start + self.target_n]
            else:
                waveform = waveform[:, :self.target_n]

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


@torch.no_grad()
def validate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_scores = []
    all_labels = []

    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels    = labels.to(device)

        # Use torch.amp.autocast with device_type string
        with torch.amp.autocast(device_type=device.type):
            logits = model(waveforms)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        probs       = torch.softmax(logits, dim=-1)
        danger_prob = probs[:, 2].cpu().numpy()
        binary_lbl  = np.where(labels.cpu().numpy() > 0, 1, 0)
        all_scores.extend(danger_prob.tolist())
        all_labels.extend(binary_lbl.tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    eer      = compute_eer(np.array(all_scores), np.array(all_labels))
    model.train()
    return avg_loss, eer


def train(config: dict, resume_path: str = None, debug: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    checkpoint_dir = ML_ROOT / config["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ML_ROOT / config["log_dir"]
    writer  = SummaryWriter(log_dir=str(log_dir))
    print(f"[Train] TensorBoard: tensorboard --logdir {log_dir}")

    # Datasets
    print("\n[Train] Loading datasets...")
    train_dataset = TatvaaniDataset(config["train_manifest"], config, augment=config["augment_train"])
    val_dataset   = TatvaaniDataset(config["val_manifest"],   config, augment=False)

    if debug:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, list(range(min(64, len(train_dataset)))))
        val_dataset   = Subset(val_dataset,   list(range(min(32, len(val_dataset)))))
        config["epochs"] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"] * 2,
        shuffle=False, num_workers=0, pin_memory=False,
        collate_fn=collate_fn,
    )

    # Verify first batch
    print("[Train] Verifying first batch...")
    try:
        wb, lb = next(iter(train_loader))
        print(f"[Train] First batch OK — waveforms: {wb.shape}, labels: {lb.shape}")
        print(f"[Train] Label values: {lb.unique().tolist()}")
    except Exception as e:
        print(f"[Train] FIRST BATCH FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Model
    model = TatvaNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {n_params:,}")
    print(f"[Train] Epochs: {config['epochs']}, Batch: {config['batch_size']}, LR: {config['lr']}")

    # Fixed alpha weights — Safe=3x, Caution=2x, Danger=1x
    # Fixed because Caution class has 0 training samples (binary dataset)
    # Dynamic computation causes div-by-zero and inf weights
    alpha_tensor = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32).to(device)
    print(f"[Train] FocalLoss alpha weights: {alpha_tensor.tolist()} (fixed)")
    criterion = FocalLoss(gamma=2.0, alpha=alpha_tensor)
    print("[Train] FocalLoss OK")

    # Optimizer
    print("[Train] Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    print("[Train] Optimizer OK")

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(config["epochs"] - config["warmup_epochs"], 1),
        eta_min=config["lr_min"],
    )
    print("[Train] Scheduler OK")

    # GradScaler — using torch.amp (PyTorch 2.x API, not deprecated torch.cuda.amp)
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"[Train] GradScaler OK (AMP enabled: {use_amp})")

    # Resume
    start_epoch = 0
    best_eer    = float("inf")
    no_improve  = 0

    if resume_path is not None:
        rp = Path(resume_path)
        if not rp.is_absolute():
            rp = checkpoint_dir / rp
        if rp.exists():
            print(f"[Train] Resuming from {rp}")
            ckpt        = torch.load(rp, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_eer    = ckpt.get("best_eer", float("inf"))
            no_improve  = ckpt.get("no_improve", 0)
            print(f"[Train] Resumed at epoch {start_epoch}, best EER: {best_eer:.4f}")

    # Training loop
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING — {config['epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        model.train()

        if epoch < config["warmup_epochs"]:
            warmup_lr(optimizer, epoch, config["warmup_epochs"], config["lr"])

        epoch_loss = 0.0
        n_correct  = 0
        n_total    = 0

        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels    = labels.to(device)
            optimizer.zero_grad()

            # torch.amp.autocast with explicit device_type (PyTorch 2.x API)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(waveforms)
                loss   = criterion(logits, labels)

            if torch.isnan(loss):
                print(f"[Train] NaN loss at batch {batch_idx}, skipping.")
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

            if (batch_idx + 1) % 100 == 0:
                pct = 100.0 * (batch_idx + 1) / len(train_loader)
                print(
                    f"  Epoch {epoch+1:02d} [{pct:5.1f}%] "
                    f"loss={epoch_loss/n_total:.4f} "
                    f"acc={100*n_correct/n_total:.1f}% "
                    f"lr={get_lr(optimizer):.2e}"
                )

            if debug and batch_idx >= 1:
                break

        train_loss = epoch_loss / max(n_total, 1)
        train_acc  = n_correct  / max(n_total, 1)

        val_loss, val_eer = validate(model, val_loader, criterion, device, config)

        if epoch >= config["warmup_epochs"]:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        print(
            f"\nEpoch {epoch+1:02d}/{config['epochs']} "
            f"train_loss={train_loss:.4f} train_acc={100*train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_EER={100*val_eer:.2f}% "
            f"lr={get_lr(optimizer):.2e} time={epoch_time:.0f}s"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("EER/val",    val_eer,    epoch)
        writer.add_scalar("LR",         get_lr(optimizer), epoch)

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

        if val_eer < best_eer:
            best_eer   = val_eer
            no_improve = 0
            torch.save(ckpt, checkpoint_dir / "best_model.pt")
            print(f"  [BEST] New best EER: {100*best_eer:.2f}% -- saved best_model.pt")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{config['early_stop_patience']})")

        if no_improve >= config["early_stop_patience"] and not debug:
            print(f"[Train] Early stopping at epoch {epoch+1}.")
            break

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best validation EER: {100*best_eer:.2f}%")
    print(f"Best model: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*60}")
    print(f"Next: py -3.11 eval/evaluate.py --checkpoint checkpoints/best_model.pt")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TatvaNet")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()
    train(CONFIG, resume_path=args.resume, debug=args.debug)
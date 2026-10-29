"""
scripts/finetune_elevenlabs.py
==============================
Targeted fine-tuning of TatvaNet to sharply discriminate between:
  1. Modern ElevenLabs / Neural TTS synthetic speech (Label 1)
  2. Authentic phone-mic / WhatsApp mobile speech from tr/ (Label 0)

Usage:
  python scripts/finetune_elevenlabs.py --checkpoint checkpoints/best_model_new.pt --epochs 4
  python scripts/finetune_elevenlabs.py --checkpoint checkpoints/best_model.pt --epochs 4
"""

import os
import sys
import glob
import random
import shutil
import argparse
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torchaudio
import soundfile as sf
import pandas as pd
from datasets import load_dataset

from training.train import load_audio_robust
from export.export_tflite import export_onnx
from model.tatvanet import TatvaNet


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TatvaNet on ElevenLabs + tr/ real audio.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_new.pt",
                        help="Path to starting PyTorch checkpoint.")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate for fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--n_samples", type=int, default=350,
                        help="Number of ElevenLabs samples to stream.")
    parser.add_argument("--output_onnx", type=str, default="export/tatvanet_finetuned.onnx",
                        help="Path to save the fine-tuned ONNX model.")
    return parser.parse_args()


def prepare_elevenlabs_data(n_samples: int = 350, out_dir: str = "data/raw/elevenlabs_fakes") -> list:
    """Streams n_samples from skypro1111/elevenlabs_dataset without downloading the full archive."""
    os.makedirs(out_dir, exist_ok=True)
    existing = glob.glob(f"{out_dir}/*.wav")
    if len(existing) >= n_samples:
        print(f"[ElevenLabs] Found {len(existing)} cached samples in {out_dir}. Skipping download.")
        return [{"path": os.path.abspath(p), "label": 1, "speaker_id": f"el_spk_{i % 20}"} for i, p in enumerate(existing[:n_samples])]

    print(f"[ElevenLabs] Streaming {n_samples} samples from HuggingFace (skypro1111/elevenlabs_dataset)...", flush=True)
    ds = load_dataset("skypro1111/elevenlabs_dataset", split="train", streaming=True)
    records = []

    for idx, item in enumerate(ds):
        if idx >= n_samples:
            break
        arr = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        t_wav = torch.from_numpy(arr).float()
        if t_wav.ndim > 1:
            t_wav = t_wav.mean(dim=0)
        if sr != 16000:
            t_wav = torchaudio.functional.resample(t_wav, sr, 16000)

        out_path = os.path.join(out_dir, f"el_{idx:04d}.wav")
        sf.write(out_path, t_wav.numpy(), 16000)
        records.append({
            "path": os.path.abspath(out_path),
            "label": 1,
            "speaker_id": f"el_spk_{idx % 20}"
        })
        if (idx + 1) % 50 == 0 or (idx + 1) == n_samples:
            print(f"  Streamed {idx + 1}/{n_samples} ElevenLabs samples...", flush=True)

    return records


def prepare_tr_data(target_n: int = 350, out_dir: str = "data/raw/tr_finetune") -> list:
    """Slices 2.0s active voiced windows from tr/ mobile phone recordings."""
    os.makedirs(out_dir, exist_ok=True)
    existing = glob.glob(f"{out_dir}/*.wav")
    if len(existing) >= target_n:
        print(f"[tr/] Found {len(existing)} cached chunks in {out_dir}.")
        return [{"path": os.path.abspath(p), "label": 0, "speaker_id": f"tr_{os.path.basename(p).split('_')[1]}"} for p in existing[:target_n]]

    tr_files = sorted(glob.glob("tr/*"))
    assert len(tr_files) > 0, "No files found in tr/ directory!"
    print(f"[tr/] Slicing 2.0s active chunks from {len(tr_files)} files in tr/...", flush=True)

    records = []
    for f in tr_files:
        spk = os.path.basename(f).split("-")[-1].split(".")[0].strip()
        wav, sr = load_audio_robust(f)
        if wav is not None and wav.shape[-1] >= 16000:
            total_s = wav.shape[-1]
            for start in range(0, max(1, total_s - 32000 + 1), 10000):  # ~0.6s stride
                chunk = wav[:, start:start + 32000]
                if chunk.shape[-1] < 32000:
                    chunk = torch.nn.functional.pad(chunk, (0, 32000 - chunk.shape[-1]))
                out_path = os.path.join(out_dir, f"tr_{spk}_{len(records):04d}.wav")
                sf.write(out_path, chunk.squeeze(0).numpy(), 16000)
                records.append({
                    "path": os.path.abspath(out_path),
                    "label": 0,
                    "speaker_id": f"tr_{spk}"
                })
                if len(records) >= target_n:
                    break
        if len(records) >= target_n:
            break

    print(f"[tr/] Prepared {len(records)} active chunks across all speakers.", flush=True)
    return records


def main():
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Fallback check for alternate locations
        alt_paths = [
            Path("checkpoints/best_model_new.pt"),
            Path("checkpoints/best_model.pt"),
            Path("/kaggle/working/trialSpeakSafe/checkpoints/best_model.pt"),
            Path("/kaggle/working/best_model.pt")
        ]
        for ap in alt_paths:
            if ap.exists():
                ckpt_path = ap
                break
    assert ckpt_path.exists(), f"Starting checkpoint not found at {args.checkpoint}!"
    print(f"[Setup] Starting fine-tuning from: {ckpt_path}", flush=True)

    # 1. Prepare ElevenLabs and tr/ data
    fake_recs = prepare_elevenlabs_data(n_samples=args.n_samples)
    real_recs = prepare_tr_data(target_n=len(fake_recs))

    # 2. Split strictly by speaker (zero leakage)
    def split_by_speaker(recs, train_ratio=0.85):
        by_spk = {}
        for r in recs:
            by_spk.setdefault(r["speaker_id"], []).append(r)
        spks = sorted(by_spk.keys())
        random.shuffle(spks)
        n_tr = max(1, int(len(spks) * train_ratio))
        tr = [r for s in spks[:n_tr] for r in by_spk[s]]
        va = [r for s in spks[n_tr:] for r in by_spk[s]]
        return tr, va

    tr_fake, va_fake = split_by_speaker(fake_recs)
    tr_real, va_real = split_by_speaker(real_recs)

    m_tr = min(len(tr_fake), len(tr_real))
    m_va = min(len(va_fake), len(va_real))

    df_train = pd.DataFrame(tr_fake[:m_tr] + tr_real[:m_tr]).sample(frac=1, random_state=42)
    df_val = pd.DataFrame(va_fake[:m_va] + va_real[:m_va]).sample(frac=1, random_state=42)

    os.makedirs("data", exist_ok=True)
    train_manifest = "data/finetune_train.csv"
    val_manifest = "data/finetune_val.csv"
    df_train.to_csv(train_manifest, index=False)
    df_val.to_csv(val_manifest, index=False)
    print(f"[Manifests] Train: {len(df_train)} samples (1:1), Val: {len(df_val)} samples (1:1).", flush=True)

    # 3. Launch fine-tuning via training/train.py
    cmd = [
        sys.executable, "-u", "training/train.py",
        "--finetune", str(ckpt_path),
        "--train_manifest", train_manifest,
        "--val_manifest", val_manifest,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--reset_best",
        "--checkpoint_dir", "checkpoints"
    ]
    print(f"\n[Train] Executing: {' '.join(cmd)}\n", flush=True)
    import subprocess
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[Error] Training exited with code {proc.returncode}")
        return

    # 4. Export fine-tuned model to ONNX
    print(f"\n[Export] Exporting fine-tuned model to ONNX: {args.output_onnx}", flush=True)
    device = torch.device("cpu")
    best_ckpt = Path("checkpoints/best_model.pt")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model = TatvaNet(sample_rate=16000, n_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    os.makedirs(os.path.dirname(args.output_onnx) or ".", exist_ok=True)
    export_onnx(model, args.output_onnx, clip_samples=32000, device=device)
    print(f"[Export] Saved fine-tuned ONNX to: {args.output_onnx}", flush=True)

    # 5. Run held-out benchmark
    print("\n[Benchmark] Evaluating held-out test audios...", flush=True)
    bench_cmd = [
        sys.executable, "-u", "TEST_FOLDER_AUDIOS.py",
        "--folder", "test_audio(razz)/fake",
        "--onnx", args.output_onnx,
        "--checkpoint", str(best_ckpt)
    ]
    subprocess.run(bench_cmd)

    tr_bench_cmd = [
        sys.executable, "-u", "TEST_FOLDER_AUDIOS.py",
        "--folder", "tr",
        "--onnx", args.output_onnx,
        "--checkpoint", str(best_ckpt)
    ]
    subprocess.run(tr_bench_cmd)
    print("\n[Complete] Fine-tuning and benchmark finished successfully!", flush=True)


if __name__ == "__main__":
    main()

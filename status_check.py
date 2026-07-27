"""
=============================================================================
Tatvaani — status_check.py
=============================================================================
Run: py -3.11 status_check.py

Tells you exactly:
  1. How many audio files are processed per dataset
  2. Manifest stats (train/val/test split status)
  3. Whether training has started / what checkpoint exists
  4. What to run next
=============================================================================
"""

import sys
from pathlib import Path
import csv

ML_ROOT = Path(__file__).resolve().parent

print("=" * 60)
print("TATVAANI PROJECT STATUS")
print("=" * 60)

# ── 1. Processed audio files ──────────────────────────────────────────────
print("\n[1] PROCESSED AUDIO FILES")
processed_root = ML_ROOT / "data" / "processed"
audio_exts = {".wav", ".flac"}

if not processed_root.exists():
    print("  data/processed/ does not exist yet")
else:
    total_files = 0
    for dataset_dir in sorted(processed_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        files = [f for f in dataset_dir.rglob("*") if f.suffix.lower() in audio_exts]
        size_mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"  {dataset_dir.name:20s}  {len(files):8,} files  ({size_mb:.0f} MB)")
        total_files += len(files)
    print(f"  {'TOTAL':20s}  {total_files:8,} files")

# ── 2. Manifest status ────────────────────────────────────────────────────
print("\n[2] MANIFEST FILES")
for csv_name in ["manifest.csv", "train_manifest.csv", "val_manifest.csv", "test_manifest.csv"]:
    p = ML_ROOT / "data" / csv_name
    if p.exists():
        with open(p, "r") as f:
            n = sum(1 for _ in f) - 1  # subtract header
        # Get label distribution
        with open(p, "r") as f:
            reader = csv.DictReader(f)
            labels = {}
            datasets = {}
            for row in reader:
                lbl = row.get("label", "?")
                ds  = row.get("dataset", "?")
                labels[lbl]   = labels.get(lbl, 0) + 1
                datasets[ds]  = datasets.get(ds, 0) + 1
        label_str   = "  ".join(f"label{k}={v:,}" for k, v in sorted(labels.items()))
        dataset_str = "  ".join(f"{k}={v:,}" for k, v in sorted(datasets.items()))
        print(f"  {csv_name:25s}  {n:8,} rows")
        print(f"    Labels:   {label_str}")
        print(f"    Datasets: {dataset_str}")
    else:
        print(f"  {csv_name:25s}  NOT FOUND")

# ── 3. Checkpoints ────────────────────────────────────────────────────────
print("\n[3] TRAINING CHECKPOINTS")
ckpt_dir = ML_ROOT / "checkpoints"
if not ckpt_dir.exists() or not list(ckpt_dir.glob("*.pt")):
    print("  No checkpoints found — training has not completed an epoch yet")
else:
    for ckpt in sorted(ckpt_dir.glob("*.pt")):
        size_mb = ckpt.stat().st_size / 1e6
        import os, time
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(ckpt.stat().st_mtime))
        # Try to read epoch from checkpoint
        try:
            import torch
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
            epoch   = data.get("epoch", "?")
            val_eer = data.get("val_eer", None)
            eer_str = f"  val_EER={100*val_eer:.2f}%" if val_eer is not None else ""
            print(f"  {ckpt.name:30s}  {size_mb:.1f}MB  epoch={epoch}{eer_str}  ({mtime})")
        except Exception:
            print(f"  {ckpt.name:30s}  {size_mb:.1f}MB  ({mtime})")

# ── 4. TensorBoard logs ───────────────────────────────────────────────────
print("\n[4] TENSORBOARD LOGS")
runs_dir = ML_ROOT / "runs"
if runs_dir.exists():
    event_files = list(runs_dir.rglob("events.out.*"))
    if event_files:
        latest = max(event_files, key=lambda f: f.stat().st_mtime)
        import time
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(latest.stat().st_mtime))
        print(f"  {len(event_files)} event file(s), latest: {mtime}")
        print(f"  View: tensorboard --logdir {runs_dir}")
    else:
        print("  No event files — training hasn't logged yet")
else:
    print("  runs/ directory not found")

# ── 5. What to do next ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("WHAT TO DO NEXT")
print("=" * 60)

manifest_path = ML_ROOT / "data" / "manifest.csv"
train_path    = ML_ROOT / "data" / "train_manifest.csv"
ckpt_best     = ML_ROOT / "checkpoints" / "best_model.pt"
ckpt_last     = ML_ROOT / "checkpoints" / "last_checkpoint.pt"

if not manifest_path.exists():
    print("  1. Process datasets first:")
    print("     py -3.11 process_dataset.py --dataset asvspoof --input data/raw/ASVspoof2019_LA")
    print("     py -3.11 process_dataset.py --dataset wavefake --input data/raw/WaveFake")
elif not train_path.exists():
    print("  1. Split the manifest:")
    print("     py -3.11 split_dataset.py")
elif ckpt_best.exists():
    print("  ✅ Training has a best checkpoint!")
    print("  Next: evaluate and export:")
    print("     py -3.11 eval/evaluate.py --checkpoint checkpoints/best_model.pt")
    print("     py -3.11 export/export_tflite.py --checkpoint checkpoints/best_model.pt")
elif ckpt_last.exists():
    print("  Training was interrupted. Resume with:")
    print("     py -3.11 training/train.py --resume checkpoints/last_checkpoint.pt")
else:
    print("  ✅ Ready to train! Run in a STANDALONE PowerShell window:")
    print("     (Win+R → powershell → Enter — so it survives VS Code closing)")
    print()
    print("     cd C:\\Users\\reema\\Tatvaani\\tatvaani_ml")
    print("     py -3.11 training/train.py 2>&1 | Tee-Object train.log")
    print()
    print("  This saves all output to train.log so you can check progress anytime:")
    print("     Get-Content train.log -Tail 20")

print()
print("  To check training progress anytime (even after VS Code restart):")
print("     Get-Content C:\\Users\\reema\\Tatvaani\\tatvaani_ml\\train.log -Tail 30")
print("=" * 60)
"""
=============================================================================
Tatvaani — process_librispeech.py
=============================================================================
Converts LibriSpeech train-clean-100 FLAC files to 16kHz mono WAV
and appends them to manifest.csv as label 0 (bonafide/Safe).

Run: py -3.11 process_librispeech.py

Structure expected:
  data/raw/LibriSpeech/train-clean-100/
    103/         ← speaker ID
      1240/      ← chapter ID
        103-1240-0000.flac
        103-1240-0000.trans.txt
        ...
=============================================================================
"""

import csv
import sys
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T

ML_ROOT       = Path(__file__).resolve().parent
SRC_DIR       = ML_ROOT / "data" / "raw" / "LibriSpeech" / "train-clean-100"
OUT_DIR       = ML_ROOT / "data" / "processed" / "librispeech"
MANIFEST_PATH = ML_ROOT / "data" / "manifest.csv"
COLUMNS       = ["path", "label", "dataset", "speaker_id", "duration", "language"]

TARGET_SR   = 16000
MIN_DUR     = 1.0
MAX_DUR     = 4.0
OVERLAP     = 0.5


def process():
    print("=" * 60)
    print("LIBRISPEECH PROCESSOR")
    print("=" * 60)

    if not SRC_DIR.exists():
        print(f"ERROR: {SRC_DIR} not found")
        print("Check extraction completed correctly.")
        sys.exit(1)

    flac_files = list(SRC_DIR.rglob("*.flac"))
    print(f"Found {len(flac_files):,} FLAC files")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing manifest to avoid duplicates
    existing_rows  = []
    existing_paths = set()
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        existing_paths = {r["path"] for r in existing_rows}
        print(f"Manifest: {len(existing_rows):,} existing entries")

    new_rows  = []
    n_ok      = 0
    n_skip    = 0
    n_seg     = 0

    for i, flac in enumerate(flac_files):
        if i % 5000 == 0 and i > 0:
            print(f"  {i:,}/{len(flac_files):,}  ({n_seg:,} segments so far)...")

        # Speaker ID is the grandparent folder name
        speaker_id = f"libri_{flac.parent.parent.name}"

        try:
            waveform, sr = torchaudio.load(str(flac))
        except Exception:
            n_skip += 1
            continue

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz
        if sr != TARGET_SR:
            waveform = T.Resample(sr, TARGET_SR)(waveform)

        # Normalise
        peak = waveform.abs().max()
        if peak < 1e-6:
            n_skip += 1
            continue
        waveform = waveform / peak

        dur = waveform.shape[1] / TARGET_SR
        if dur < MIN_DUR:
            n_skip += 1
            continue

        # Split into segments if longer than MAX_DUR
        target_n  = int(MAX_DUR  * TARGET_SR)
        overlap_n = int(OVERLAP  * TARGET_SR)

        segments = []
        if dur <= MAX_DUR:
            segments.append(waveform)
        else:
            n_samp = waveform.shape[1]
            start  = 0
            while start < n_samp:
                end = min(start + target_n, n_samp)
                seg = waveform[:, start:end]
                if seg.shape[1] / TARGET_SR >= MIN_DUR:
                    segments.append(seg)
                start += target_n - overlap_n
                if end == n_samp:
                    break

        # Save each segment
        for seg_idx, seg in enumerate(segments):
            out_name = f"libri_{flac.stem}_s{seg_idx:02d}.wav"
            out_path = OUT_DIR / out_name
            rel_path = str(out_path.relative_to(ML_ROOT)).replace("\\", "/")

            if rel_path in existing_paths:
                continue

            torchaudio.save(str(out_path), seg, TARGET_SR)
            seg_dur = seg.shape[1] / TARGET_SR

            new_rows.append({
                "path":       rel_path,
                "label":      0,           # bonafide = Safe
                "dataset":    "librispeech",
                "speaker_id": speaker_id,
                "duration":   f"{seg_dur:.3f}",
                "language":   "english",
            })
            n_seg += 1

        n_ok += 1

    print(f"\nDone: {n_ok:,} files → {n_seg:,} segments  ({n_skip:,} skipped)")

    if not new_rows:
        print("No new rows to add.")
        return

    # Append to manifest
    all_rows = existing_rows + new_rows
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    # Final stats
    all_labels = [int(r["label"]) for r in all_rows]
    n_safe   = all_labels.count(0)
    n_danger = all_labels.count(2)

    print(f"\n{'='*60}")
    print(f"MANIFEST UPDATED")
    print(f"Total rows:       {len(all_rows):,}")
    print(f"Label 0 (Safe):   {n_safe:,}  ({100*n_safe/len(all_rows):.1f}%)")
    print(f"Label 2 (Danger): {n_danger:,}  ({100*n_danger/len(all_rows):.1f}%)")
    print(f"\nNext:")
    print(f"  py -3.11 split_dataset.py")
    print(f"  Then training in standalone PowerShell:")
    print(f"  py -3.11 training\\train.py 2>&1 | Tee-Object train.log")
    print(f"{'='*60}")


if __name__ == "__main__":
    process()
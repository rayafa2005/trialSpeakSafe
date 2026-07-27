"""
=============================================================================
Tatvaani — Targeted Fix Script
=============================================================================
File: patch2.py
Run:  py -3.11 patch2.py

Fixes:
  1. tatvanet.py line 61 — 'num_classes: int = 3' inserted before non-default
     args causing SyntaxError. Fix: move num_classes to END of params, or
     make ALL params have defaults. We make channels and reduction keep their
     defaults and put num_classes at the end.

  2. IndicSynth reality check — 170GB of parquet, extracting all is impractical.
     Strategy: extract ONE parquet file per language (≈600MB sample) to get
     Indian language coverage without filling disk. This gives ~5,000-10,000
     clips per language — enough for fine-tuning.
=============================================================================
"""

import sys
import re
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Fix 1: tatvanet.py — fix the __init__ signature
# ---------------------------------------------------------------------------
tatvanet_path = ML_ROOT / "model" / "tatvanet.py"

if not tatvanet_path.exists():
    print(f"ERROR: {tatvanet_path} not found")
    sys.exit(1)

text  = tatvanet_path.read_text(encoding="utf-8")
lines = text.splitlines()

# Find the __init__ line
init_idx = None
for i, line in enumerate(lines):
    if "def __init__(self" in line and "TatvaNet" not in line:
        # Get the class context — find the class definition above this line
        for j in range(i - 1, max(0, i - 20), -1):
            if "class TatvaNet" in lines[j]:
                init_idx = i
                break
    if init_idx is not None:
        break

# Also try finding it directly
if init_idx is None:
    for i, line in enumerate(lines):
        if "def __init__(self" in line:
            init_idx = i
            break

if init_idx is None:
    print("ERROR: Could not find __init__ in tatvanet.py")
    print("Manual fix: open model/tatvanet.py and find the __init__ line.")
    print("Remove 'num_classes: int = 3,' from the beginning of the params.")
    sys.exit(1)

current_init = lines[init_idx]
print(f"[Fix 1] Found __init__ at line {init_idx + 1}:")
print(f"        {current_init.strip()}")

# The patch.py added 'num_classes: int = 3,' at the front.
# We need to:
# (a) Remove it from the front
# (b) Add it at the end (before the closing paren) with a default value

# Strategy: parse out what's there and reconstruct cleanly.
# Handle multi-line __init__ by collecting continuation lines too
init_lines = [current_init]
j = init_idx + 1
while ")" not in "".join(init_lines) and j < len(lines):
    init_lines.append(lines[j])
    j += 1

full_init = " ".join(l.strip() for l in init_lines)
print(f"[Fix 1] Full signature: {full_init[:120]}...")

# Remove the incorrectly placed num_classes default arg
# Pattern: 'num_classes: int = 3,' or ', num_classes: int = 3'
cleaned = re.sub(r'\s*num_classes\s*:\s*int\s*=\s*\d+\s*,\s*', '', full_init)
cleaned = re.sub(r',\s*num_classes\s*:\s*int\s*=\s*\d+', '', cleaned)
cleaned = re.sub(r'\(\s*num_classes\s*:\s*int\s*=\s*\d+\s*\)', '()', cleaned)

print(f"[Fix 1] After removing bad placement: {cleaned[:120]}")

# Now check if num_classes is still there (correctly placed) or needs adding
if "num_classes" not in cleaned:
    # Add it at the end before the closing paren, as a keyword arg with default
    # Find the last ')' and insert before it
    last_paren = cleaned.rfind(")")
    if last_paren != -1:
        # Check if there are already params
        inside = cleaned[cleaned.find("(") + 1 : last_paren].strip()
        if inside and not inside.endswith(","):
            cleaned = cleaned[:last_paren] + ", num_classes: int = 3" + cleaned[last_paren:]
        elif inside:
            cleaned = cleaned[:last_paren] + "num_classes: int = 3" + cleaned[last_paren:]
        else:
            cleaned = cleaned[:last_paren] + "num_classes: int = 3" + cleaned[last_paren:]
    print(f"[Fix 1] Added num_classes at end: {cleaned[:150]}")

# Replace the original lines with the fixed single line
# (preserve indentation from original)
indent = len(current_init) - len(current_init.lstrip())
indented = " " * indent + cleaned.strip()

# Replace original init lines in the text
new_lines = lines[:init_idx] + [indented] + lines[j:]
new_text = "\n".join(new_lines)

# Verify it's valid Python syntax before writing
import ast
try:
    ast.parse(new_text)
    tatvanet_path.write_text(new_text, encoding="utf-8")
    print(f"[Fix 1] ✅ tatvanet.py syntax OK — saved")
except SyntaxError as e:
    print(f"[Fix 1] ⚠️  Auto-fix produced invalid syntax: {e}")
    print(f"[Fix 1] The fixed line would be:")
    print(f"        {indented}")
    print()
    print("MANUAL FIX REQUIRED:")
    print("  Open model/tatvanet.py")
    print(f"  Find line {init_idx + 1} which currently reads:")
    print(f"    {current_init.strip()}")
    print("  Change it to remove 'num_classes: int = 3,' from the FRONT")
    print("  and add 'num_classes: int = 3' at the END of the params list.")
    print("  Example:")
    print("    def __init__(self, channels: int = 64, reduction: int = 8, num_classes: int = 3):")
    print()
    print("Then run: py -3.11 training/train.py --debug")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Fix 2: Print IndicSynth strategy (extract 1 parquet per language)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("INDICSYNTH STRATEGY")
print("=" * 60)
print()
print("IndicSynth is 170GB of parquet — extracting all is impractical.")
print("Plan: Extract ONE parquet file per language for fine-tuning coverage.")
print("Each parquet ≈ 300-600MB → gives ~5,000 clips per language.")
print()
print("Run this AFTER training starts (in a separate terminal):")
print()
print("  py -3.11 extract_indicsynth_sample.py")
print()
print("This script is being written now...")

# Write the targeted extractor
extractor_content = '''\
"""
Tatvaani - IndicSynth Sample Extractor
Extracts first parquet file from each language for fine-tuning coverage.
Run in a separate terminal while training is ongoing.
Usage: py -3.11 extract_indicsynth_sample.py
"""
import sys
import csv
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T

try:
    import pyarrow.parquet as pq
except ImportError:
    print("Install pyarrow: py -3.11 -m pip install pyarrow")
    sys.exit(1)

ML_ROOT      = Path(__file__).resolve().parent
# IndicSynth location - update this path if yours is different
INDICSYNTH_ROOT = ML_ROOT / "data" / "raw" / "IndicSynth"
OUT_DIR      = ML_ROOT / "data" / "processed" / "indicsynth"
MANIFEST_PATH = ML_ROOT / "data" / "manifest.csv"
TARGET_SR    = 16000
MIN_DUR      = 1.0
MAX_DUR      = 4.0
COLUMNS      = ["path", "label", "dataset", "speaker_id", "duration", "language"]

LANGS = ["Bengali", "Hindi", "Kannada", "Telugu"]

def extract_one_parquet(lang_dir: Path, lang: str, out_dir: Path) -> list:
    """Extract audio from the first parquet file of a language."""
    parquet_files = sorted(lang_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"  [{lang}] No parquet files found in {lang_dir}")
        return []

    # Take first parquet only — enough for coverage without filling disk
    pf = parquet_files[0]
    size_mb = pf.stat().st_size / 1e6
    print(f"  [{lang}] Reading {pf.name} ({size_mb:.0f} MB)...")

    try:
        table = pq.read_table(str(pf))
    except Exception as e:
        print(f"  [{lang}] Failed to read parquet: {e}")
        return []

    cols = table.schema.names
    print(f"  [{lang}] Columns: {cols}")

    # Find audio column
    audio_col = None
    for c in ["audio", "speech", "waveform", "audio_array", "wav"]:
        if c in cols:
            audio_col = c
            break

    if audio_col is None:
        print(f"  [{lang}] No audio column found. Columns: {cols}")
        print(f"  [{lang}] Skipping - inspect parquet manually.")
        return []

    rows = table.to_pydict()[audio_col]
    print(f"  [{lang}] {len(rows)} rows in parquet")

    out_lang_dir = out_dir / lang
    out_lang_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    n_ok = 0
    n_skip = 0

    for i, row in enumerate(rows):
        if i % 500 == 0:
            print(f"    {i}/{len(rows)} ({n_ok} saved)...")
        try:
            # HuggingFace audio column format
            if isinstance(row, dict):
                arr = row.get("array") or row.get("waveform")
                sr  = row.get("sampling_rate", TARGET_SR)
                if arr is None:
                    n_skip += 1
                    continue
                waveform = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            elif isinstance(row, (list, bytes)):
                if isinstance(row, bytes):
                    n_skip += 1
                    continue
                waveform = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
                sr = TARGET_SR
            else:
                n_skip += 1
                continue

            # Resample
            if sr != TARGET_SR:
                waveform = T.Resample(sr, TARGET_SR)(waveform)

            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

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

            # Truncate to max duration
            max_n = int(MAX_DUR * TARGET_SR)
            if waveform.shape[1] > max_n:
                waveform = waveform[:, :max_n]
                dur = MAX_DUR

            out_path = out_lang_dir / f"{lang.lower()}_{i:07d}.wav"
            torchaudio.save(str(out_path), waveform, TARGET_SR)

            rel_path = str(out_path.relative_to(ML_ROOT)).replace("\\\\", "/")
            new_rows.append({
                "path": rel_path,
                "label": 2,
                "dataset": "indicsynth",
                "speaker_id": f"indicsynth_{lang}_{i}",
                "duration": f"{dur:.3f}",
                "language": lang,
            })
            n_ok += 1

        except Exception as e:
            n_skip += 1
            continue

    print(f"  [{lang}] Done: {n_ok} saved, {n_skip} skipped")
    return new_rows


def main():
    print("IndicSynth Sample Extractor")
    print("Extracting first parquet per language for fine-tuning coverage")
    print()

    if not INDICSYNTH_ROOT.exists():
        print(f"ERROR: IndicSynth not found at {INDICSYNTH_ROOT}")
        print("Update INDICSYNTH_ROOT in this script to your actual path.")
        sys.exit(1)

    # Load existing manifest
    existing_rows = []
    existing_paths = set()
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        existing_paths = {r["path"] for r in existing_rows}
        print(f"Manifest: {len(existing_rows)} existing entries")

    all_new_rows = []
    for lang in LANGS:
        lang_dir = INDICSYNTH_ROOT / lang
        if not lang_dir.exists():
            # Try lowercase
            lang_dir = INDICSYNTH_ROOT / lang.lower()
        if not lang_dir.exists():
            print(f"[{lang}] Directory not found, skipping")
            continue
        print(f"\\nProcessing {lang}...")
        new_rows = extract_one_parquet(lang_dir, lang, OUT_DIR)
        all_new_rows.extend(new_rows)

    if all_new_rows:
        all_rows = existing_rows + all_new_rows
        with open(MANIFEST_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\\nManifest updated: {len(all_rows)} total entries")
        print(f"Added {len(all_new_rows)} IndicSynth segments")
        print()
        print("Next: re-run split then fine-tune:")
        print("  py -3.11 split_dataset.py")
        print("  py -3.11 training/train.py --resume checkpoints/best_model.pt")
    else:
        print("\\nNo new segments added.")
        print("Check the column names printed above and update the script if needed.")


if __name__ == "__main__":
    main()
'''

extractor_path = ML_ROOT / "extract_indicsynth_sample.py"
extractor_path.write_text(extractor_content, encoding="utf-8")
print(f"[Written] extract_indicsynth_sample.py → {extractor_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("ALL FIXES APPLIED")
print("=" * 60)
print()
print("TERMINAL 1 — Start training now:")
print("  py -3.11 training/train.py --debug")
print("  (if debug passes 2 epochs, run full: py -3.11 training/train.py)")
print()
print("TERMINAL 2 — Extract IndicSynth sample while training runs:")
print("  py -3.11 extract_indicsynth_sample.py")
print()
print("After training + extraction both complete:")
print("  py -3.11 split_dataset.py")
print("  py -3.11 training/train.py --resume checkpoints/best_model.pt")
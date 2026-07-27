"""
=============================================================================
Tatvaani — patch4.py
=============================================================================
Fixes:
  1. tatvanet.py forward() — waveform arrives as [B, 1, 1, N] (4D) but
     sinc_conv expects [B, 1, N] (3D). Add .squeeze(1) or reshape at the
     top of forward() in path_a before sinc conv.

  2. extract_indicsynth_sample.py — audio column is a dict but the actual
     audio array key is NOT 'array'. Print the dict keys so we can see
     what key to use, then try all common keys.

Run: py -3.11 patch4.py
=============================================================================
"""

import sys
import ast
import re
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# ── Fix 1: tatvanet.py — squeeze extra dim in forward() ──────────────────

tatvanet_path = ML_ROOT / "model" / "tatvanet.py"
text = tatvanet_path.read_text(encoding="utf-8")
lines = text.splitlines()

# Find the forward() method and the line that calls path_a
# We need to insert a squeeze before path_a is called
# The traceback shows: tatvanet.py line 498: emb_a = self.path_a(x)
# and line 214: x = self.sinc(x)
# So x enters path_a as [B, 1, 1, N] — we need to squeeze in forward()

# Strategy: find "emb_a = self.path_a(x)" and insert squeeze before it
forward_fixed = False
for i, line in enumerate(lines):
    stripped = line.strip()
    # Find the line that passes x to path_a
    if "emb_a = self.path_a" in stripped or "self.path_a(x)" in stripped:
        indent = len(line) - len(line.lstrip())
        # Insert a squeeze line before this line
        squeeze_line = " " * indent + "x = x.view(x.shape[0], 1, -1)  # ensure [B, 1, N] for sinc_conv (fixes 4D input bug)"
        # Check if already fixed
        if i > 0 and "view(x.shape[0]" in lines[i-1]:
            print("[Fix 1] ✅ Squeeze already present in tatvanet.py forward()")
            forward_fixed = True
            break
        lines.insert(i, squeeze_line)
        print(f"[Fix 1] Inserted shape fix at line {i+1} (before path_a call):")
        print(f"        {squeeze_line.strip()}")
        forward_fixed = True
        break

if not forward_fixed:
    # Alternative: find the forward method and insert at the top
    in_forward = False
    for i, line in enumerate(lines):
        if "def forward(self" in line and in_forward is False:
            in_forward = True
            # Find first non-comment, non-docstring line after def forward
            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if s and not s.startswith("#") and not s.startswith('"""') and not s.startswith("'''"):
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    squeeze = " " * indent + "# Ensure waveform is [B, 1, N] regardless of how Dataset returned it\n" + \
                              " " * indent + "if x.dim() == 4: x = x.squeeze(1)  # [B,1,1,N] -> [B,1,N]"
                    lines.insert(j, squeeze)
                    print(f"[Fix 1] Inserted dim check at top of forward() (line {j+1})")
                    forward_fixed = True
                    break
                j += 1
            break

if not forward_fixed:
    print("[Fix 1] ⚠️  Could not auto-patch tatvanet.py forward()")
    print("         MANUAL FIX: Open model/tatvanet.py, find 'def forward(self'")
    print("         Add this as the FIRST line of the method body:")
    print("         if x.dim() == 4: x = x.squeeze(1)  # [B,1,1,N] -> [B,1,N]")

# Also fix sinc_conv.py directly — squeeze before conv1d as a safety net
sinc_path = ML_ROOT / "features" / "sinc_conv.py"
if sinc_path.exists():
    sinc_text = sinc_path.read_text(encoding="utf-8")
    sinc_lines = sinc_text.splitlines()
    sinc_fixed = False
    for i, line in enumerate(sinc_lines):
        if "F.conv1d(x," in line or "conv1d(x," in line:
            indent = len(line) - len(line.lstrip())
            # Insert squeeze before conv1d call
            if i > 0 and "squeeze" in sinc_lines[i-1]:
                print("[Fix 1b] ✅ sinc_conv.py already has squeeze before conv1d")
                sinc_fixed = True
                break
            squeeze = " " * indent + "if x.dim() == 4: x = x.squeeze(1)  # safety: ensure 3D for conv1d"
            sinc_lines.insert(i, squeeze)
            print(f"[Fix 1b] Inserted squeeze in sinc_conv.py before conv1d (line {i+1})")
            sinc_fixed = True
            break
    if sinc_fixed:
        sinc_text = "\n".join(sinc_lines)
        try:
            ast.parse(sinc_text)
            sinc_path.write_text(sinc_text, encoding="utf-8")
            print("[Fix 1b] ✅ sinc_conv.py saved")
        except SyntaxError as e:
            print(f"[Fix 1b] ❌ Syntax error in sinc_conv.py: {e}")

# Save tatvanet.py
new_text = "\n".join(lines)
try:
    ast.parse(new_text)
    tatvanet_path.write_text(new_text, encoding="utf-8")
    print("[Fix 1] ✅ tatvanet.py saved")
except SyntaxError as e:
    print(f"[Fix 1] ❌ Syntax error: {e}")
    print("         Manual fix required — see instructions above")


# ── Fix 2: extract_indicsynth_sample.py — handle actual audio dict keys ──

extractor_path = ML_ROOT / "extract_indicsynth_sample.py"

new_extractor = '''\
"""
Tatvaani - IndicSynth Sample Extractor (fixed audio key handling)
The MLADDC parquet stores audio as a dict — this version tries all known
key names and prints what it finds so you can debug if needed.
Run: py -3.11 extract_indicsynth_sample.py
"""
import sys
import csv
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
import io

try:
    import pyarrow.parquet as pq
except ImportError:
    print("Install pyarrow: py -3.11 -m pip install pyarrow")
    sys.exit(1)

ML_ROOT       = Path(__file__).resolve().parent
# IndicSynth is inside tatvaani_ml/data/raw/IndicSynth
INDICSYNTH_ROOT = ML_ROOT / "data" / "raw" / "IndicSynth"
OUT_DIR       = ML_ROOT / "data" / "processed" / "indicsynth"
MANIFEST_PATH = ML_ROOT / "data" / "manifest.csv"
TARGET_SR     = 16000
MIN_DUR       = 1.0
MAX_DUR       = 4.0
COLUMNS       = ["path", "label", "dataset", "speaker_id", "duration", "language"]
LANGS         = ["Bengali", "Hindi", "Kannada", "Telugu"]


def try_load_audio_from_row(row):
    """
    Try every known format that HuggingFace parquet audio columns use.
    Returns (waveform [1,N], sample_rate) or None if can\'t decode.
    """
    # Format 1: dict with \'array\' key (most common HF format)
    if isinstance(row, dict):
        # Print keys on first call (debugging)
        arr = None
        sr  = TARGET_SR

        # Try all known array key names
        for key in ["array", "waveform", "data", "samples", "audio_array", "wav"]:
            if key in row and row[key] is not None:
                arr = row[key]
                break

        # Try all known sample rate key names
        for key in ["sampling_rate", "sample_rate", "sr", "rate"]:
            if key in row and row[key] is not None:
                sr = int(row[key])
                break

        if arr is None:
            # Maybe the dict itself has numeric keys or is nested differently
            # Try bytes value (encoded audio file)
            for key, val in row.items():
                if isinstance(val, (bytes, bytearray)) and len(val) > 100:
                    # Try decoding as audio file
                    try:
                        buf = io.BytesIO(val)
                        waveform, sr2 = torchaudio.load(buf)
                        return waveform, sr2
                    except Exception:
                        continue
                elif isinstance(val, (list, tuple)) and len(val) > 100:
                    arr = val
                    break
            if arr is None:
                return None

        waveform = torch.tensor(arr, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, sr

    # Format 2: raw list/array
    elif isinstance(row, (list, tuple)) and len(row) > 100:
        waveform = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
        return waveform, TARGET_SR

    # Format 3: bytes (encoded audio file like MP3/WAV/FLAC)
    elif isinstance(row, (bytes, bytearray)) and len(row) > 100:
        try:
            buf = io.BytesIO(row)
            waveform, sr = torchaudio.load(buf)
            return waveform, sr
        except Exception:
            return None

    return None


def extract_one_parquet(lang_dir: Path, lang: str, out_dir: Path) -> list:
    parquet_files = sorted(lang_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"  [{lang}] No parquet files found")
        return []

    pf = parquet_files[0]
    size_mb = pf.stat().st_size / 1e6
    print(f"  [{lang}] Reading {pf.name} ({size_mb:.0f} MB)...")

    try:
        table = pq.read_table(str(pf))
    except Exception as e:
        print(f"  [{lang}] Failed to read: {e}")
        return []

    cols = table.schema.names
    print(f"  [{lang}] Columns: {cols}")

    # Find the audio column
    audio_col = None
    for c in ["audio", "speech", "waveform", "Target Reference Audio",
              "Source Reference Audio", "wav", "audio_array"]:
        if c in cols:
            audio_col = c
            break

    if audio_col is None:
        print(f"  [{lang}] No audio column. Skipping.")
        return []

    print(f"  [{lang}] Using column: \'{audio_col}\'")
    rows = table.to_pydict()[audio_col]
    print(f"  [{lang}] {len(rows)} rows")

    # Inspect first row to understand format
    if rows:
        first = rows[0]
        print(f"  [{lang}] First row type: {type(first).__name__}")
        if isinstance(first, dict):
            print(f"  [{lang}] Dict keys: {list(first.keys())}")
            # Show value types
            for k, v in first.items():
                if isinstance(v, (list, tuple)):
                    print(f"           {k}: list of {len(v)} elements, first={v[0] if v else None}")
                elif isinstance(v, (bytes, bytearray)):
                    print(f"           {k}: bytes ({len(v)} bytes)")
                else:
                    print(f"           {k}: {type(v).__name__} = {str(v)[:60]}")

    out_lang_dir = out_dir / lang
    out_lang_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    n_ok = 0
    n_skip = 0

    for i, row in enumerate(rows):
        if i % 300 == 0:
            print(f"    {i}/{len(rows)} ({n_ok} saved)...")

        result = try_load_audio_from_row(row)
        if result is None:
            n_skip += 1
            continue

        waveform, sr = result

        try:
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample
            if sr != TARGET_SR:
                waveform = T.Resample(sr, TARGET_SR)(waveform)
            # Normalise
            peak = waveform.abs().max()
            if peak < 1e-6:
                n_skip += 1
                continue
            waveform = waveform / peak
            # Duration check
            dur = waveform.shape[1] / TARGET_SR
            if dur < MIN_DUR:
                n_skip += 1
                continue
            # Truncate
            max_n = int(MAX_DUR * TARGET_SR)
            if waveform.shape[1] > max_n:
                waveform = waveform[:, :max_n]
                dur = MAX_DUR

            out_path = out_lang_dir / f"{lang.lower()}_{i:07d}.wav"
            torchaudio.save(str(out_path), waveform, TARGET_SR)
            rel_path = str(out_path.relative_to(ML_ROOT)).replace("\\\\", "/")
            new_rows.append({
                "path": rel_path, "label": 2,
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
    print("IndicSynth Sample Extractor v2")
    print(f"Looking in: {INDICSYNTH_ROOT}")

    if not INDICSYNTH_ROOT.exists():
        print(f"ERROR: Not found: {INDICSYNTH_ROOT}")
        sys.exit(1)

    existing_rows = []
    existing_paths = set()
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        existing_paths = {r["path"] for r in existing_rows}
        print(f"Manifest: {len(existing_rows)} existing entries")

    all_new = []
    for lang in LANGS:
        lang_dir = INDICSYNTH_ROOT / lang
        if not lang_dir.exists():
            print(f"[{lang}] Not found, skipping")
            continue
        print(f"\\nProcessing {lang}...")
        all_new.extend(extract_one_parquet(lang_dir, lang, OUT_DIR))

    if all_new:
        all_rows = existing_rows + all_new
        with open(MANIFEST_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\\nManifest updated: {len(all_rows)} total ({len(all_new)} new)")
        print("Next: py -3.11 split_dataset.py")
        print("      py -3.11 training/train.py --resume checkpoints/best_model.pt")
    else:
        print("\\nNo segments saved.")
        print("Check the dict keys printed above — the audio data may be")
        print("under \'Source Reference Audio\' or \'Target Reference Audio\'.")
        print("Update the audio_col list in extract_one_parquet() if needed.")


if __name__ == "__main__":
    main()
'''

extractor_path.write_text(new_extractor, encoding="utf-8")
print("[Fix 2] ✅ extract_indicsynth_sample.py rewritten with full audio key handling")


# ── Verify both files ─────────────────────────────────────────────────────
print()
print("=" * 55)
print("Verifying syntax...")
for path in [tatvanet_path, sinc_path, extractor_path]:
    if path and path.exists():
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            print(f"  ✅ {path.name}")
        except SyntaxError as e:
            print(f"  ❌ {path.name}: {e}")

print()
print("=" * 55)
print("DONE. Run:")
print()
print("  Terminal 1 — debug check:")
print("    py -3.11 diagnose_forward.py")
print()
print("  If forward pass passes, start training:")
print("    py -3.11 training/train.py --debug")
print("    py -3.11 training/train.py")
print()
print("  Terminal 2 — IndicSynth extraction:")
print("    py -3.11 extract_indicsynth_sample.py")
print("=" * 55)
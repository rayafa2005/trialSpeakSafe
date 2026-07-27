"""
=============================================================================
Tatvaani — Quick Patch Script
=============================================================================
File: patch.py
Run:  py -3.11 patch.py

Fixes two issues found during debug run:
  1. train.py passes num_classes to TatvaNet but tatvanet.py doesn't accept it
     → Patch: hardcode num_classes=3 inside train.py, don't pass via kwarg
  2. fix_indicsynth.py has unicode escape error in docstring (Windows paths)
     → Rewrite the file cleanly

Run this ONCE, then: py -3.11 training/train.py --debug
=============================================================================
"""

import sys
import re
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Fix 1: train.py — remove num_classes kwarg from TatvaNet() call
# ---------------------------------------------------------------------------
train_path = ML_ROOT / "training" / "train.py"

if not train_path.exists():
    print(f"ERROR: {train_path} not found. Are you in tatvaani_ml/?")
    sys.exit(1)

text = train_path.read_text(encoding="utf-8")

# Replace the TatvaNet instantiation line to not pass num_classes
old_line = 'model = TatvaNet(num_classes=config["num_classes"]).to(device)'
new_line = 'model = TatvaNet().to(device)  # num_classes=3 hardcoded in tatvanet.py'

if old_line in text:
    text = text.replace(old_line, new_line)
    train_path.write_text(text, encoding="utf-8")
    print(f"[Fix 1] ✅ train.py patched — TatvaNet() called without num_classes kwarg")
else:
    # Already patched or different wording — check if it's the variant
    if "TatvaNet().to(device)" in text:
        print(f"[Fix 1] ✅ train.py already patched")
    else:
        # Find the actual line and report it
        for i, line in enumerate(text.splitlines(), 1):
            if "TatvaNet(" in line:
                print(f"[Fix 1] Found TatvaNet call at line {i}: {line.strip()}")
        print("[Fix 1] ⚠️  Could not auto-patch. See MANUAL FIX below.")
        print("         Find the line with TatvaNet( and change it to: TatvaNet()")

# ---------------------------------------------------------------------------
# Fix 2: Also check tatvanet.py — add num_classes param gracefully so future
#         calls with num_classes= don't break
# ---------------------------------------------------------------------------
tatvanet_path = ML_ROOT / "model" / "tatvanet.py"

if tatvanet_path.exists():
    tnet_text = tatvanet_path.read_text(encoding="utf-8")
    
    # Look for __init__ signature
    lines = tnet_text.splitlines()
    init_line_idx = None
    for i, line in enumerate(lines):
        if "def __init__(self" in line and init_line_idx is None:
            init_line_idx = i
            print(f"\n[Info] TatvaNet.__init__ at line {i+1}: {line.strip()}")
    
    if init_line_idx is not None:
        init_line = lines[init_line_idx]
        
        # Check if num_classes is already there
        if "num_classes" in init_line:
            print("[Fix 2] ✅ TatvaNet already accepts num_classes — no patch needed")
            print("         The issue was something else. Check the full traceback.")
        else:
            # Add num_classes=3 as a default parameter
            # Handle: def __init__(self): or def __init__(self, other_params):
            if "def __init__(self):" in init_line:
                new_init = init_line.replace(
                    "def __init__(self):",
                    "def __init__(self, num_classes: int = 3):"
                )
            elif "def __init__(self," in init_line:
                new_init = init_line.replace(
                    "def __init__(self,",
                    "def __init__(self, num_classes: int = 3,"
                )
            else:
                new_init = None
            
            if new_init:
                lines[init_line_idx] = new_init
                new_tnet_text = "\n".join(lines)
                tatvanet_path.write_text(new_tnet_text, encoding="utf-8")
                print(f"[Fix 2] ✅ tatvanet.py patched — __init__ now accepts num_classes=3 default")
            else:
                print(f"[Fix 2] ⚠️  Could not auto-patch tatvanet.py __init__")
                print(f"         Manually add 'num_classes: int = 3' to its __init__ params")
else:
    print(f"[Fix 2] ⚠️  tatvanet.py not found at {tatvanet_path}")

# ---------------------------------------------------------------------------
# Fix 3: Rewrite fix_indicsynth.py without unicode escape issues
# ---------------------------------------------------------------------------
indicsynth_path = ML_ROOT / "fix_indicsynth.py"

indicsynth_content = (
    'import sys\n'
    'import csv\n'
    'import argparse\n'
    'from pathlib import Path\n'
    'import torch\n'
    'import torchaudio\n'
    'import torchaudio.transforms as T\n'
    '\n'
    'ML_ROOT      = Path(__file__).resolve().parent\n'
    'PROJECT_ROOT = ML_ROOT.parent\n'
    'INDICSYNTH_ROOT = PROJECT_ROOT / "data" / "raw" / "IndicSynth"\n'
    'PROCESSED_DIR   = ML_ROOT / "data" / "processed" / "indicsynth"\n'
    'MANIFEST_PATH   = ML_ROOT / "data" / "manifest.csv"\n'
    'TARGET_SR       = 16000\n'
    'MIN_DURATION_S  = 1.0\n'
    'MAX_DURATION_S  = 4.0\n'
    'OVERLAP_S       = 0.5\n'
    'EXPECTED_LANGS  = ["Bengali", "Hindi", "Kannada", "Telugu"]\n'
    'AUDIO_EXTS      = {".wav", ".flac", ".mp3", ".ogg"}\n'
    '\n'
    '\n'
    'def diagnose():\n'
    '    print("\\n" + "="*60)\n'
    '    print("INDICSYNTH DIAGNOSTIC")\n'
    '    print("="*60)\n'
    '    print(f"Looking in: {INDICSYNTH_ROOT}")\n'
    '\n'
    '    if not INDICSYNTH_ROOT.exists():\n'
    '        print(f"ERROR: Directory does not exist: {INDICSYNTH_ROOT}")\n'
    '        return\n'
    '\n'
    '    all_files   = list(INDICSYNTH_ROOT.rglob("*"))\n'
    '    audio_files = [f for f in all_files if f.suffix.lower() in AUDIO_EXTS]\n'
    '    parquet_files = list(INDICSYNTH_ROOT.rglob("*.parquet"))\n'
    '\n'
    '    print(f"Total items:   {len(all_files)}")\n'
    '    print(f"Audio files:   {len(audio_files)}")\n'
    '    print(f"Parquet files: {len(parquet_files)}")\n'
    '    print()\n'
    '\n'
    '    print("Top-level contents:")\n'
    '    for item in sorted(INDICSYNTH_ROOT.iterdir()):\n'
    '        if item.name.startswith("."):\n'
    '            continue\n'
    '        if item.is_dir():\n'
    '            sub_audio = [f for f in item.rglob("*") if f.suffix.lower() in AUDIO_EXTS]\n'
    '            sub_parquet = list(item.rglob("*.parquet"))\n'
    '            total_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / 1e6\n'
    '            print(f"  {item.name}/  ({len(sub_audio)} audio, {len(sub_parquet)} parquet, {total_mb:.0f} MB)")\n'
    '            # Show first few files inside\n'
    '            for sub in sorted(item.iterdir())[:3]:\n'
    '                print(f"    {sub.name}")\n'
    '        else:\n'
    '            print(f"  {item.name}  ({item.stat().st_size/1024:.0f} KB)")\n'
    '\n'
    '    print()\n'
    '    if audio_files:\n'
    '        print(f"First 5 audio files:")\n'
    '        for f in audio_files[:5]:\n'
    '            print(f"  {f.relative_to(INDICSYNTH_ROOT)}")\n'
    '    elif parquet_files:\n'
    '        print("No WAV/FLAC found. Parquet files detected.")\n'
    '        print("Run: py -3.11 fix_indicsynth.py --extract-parquet")\n'
    '    else:\n'
    '        print("No audio or parquet. Download may be incomplete.")\n'
    '\n'
    '\n'
    'def extract_from_parquet():\n'
    '    try:\n'
    '        import pyarrow.parquet as pq\n'
    '        import numpy as np\n'
    '    except ImportError:\n'
    '        print("Install pyarrow: py -3.11 -m pip install pyarrow")\n'
    '        sys.exit(1)\n'
    '\n'
    '    parquet_files = list(INDICSYNTH_ROOT.rglob("*.parquet"))\n'
    '    if not parquet_files:\n'
    '        print("No parquet files found. Run --diagnose first.")\n'
    '        return\n'
    '\n'
    '    print(f"Found {len(parquet_files)} parquet files.")\n'
    '    total = 0\n'
    '    for pf_path in parquet_files:\n'
    '        lang_name = pf_path.parent.parent.name\n'
    '        out_dir = PROCESSED_DIR / lang_name\n'
    '        out_dir.mkdir(parents=True, exist_ok=True)\n'
    '        table = pq.read_table(str(pf_path))\n'
    '        cols  = table.schema.names\n'
    '        print(f"  {pf_path.name} columns: {cols}")\n'
    '        audio_col = next((c for c in ["audio","speech","waveform"] if c in cols), None)\n'
    '        if not audio_col:\n'
    '            print(f"  No audio column found. Skipping.")\n'
    '            continue\n'
    '        for row in table.to_pydict()[audio_col]:\n'
    '            try:\n'
    '                if isinstance(row, dict):\n'
    '                    arr = torch.tensor(row["array"], dtype=torch.float32).unsqueeze(0)\n'
    '                    sr  = row.get("sampling_rate", 16000)\n'
    '                else:\n'
    '                    arr = torch.tensor(row, dtype=torch.float32).unsqueeze(0)\n'
    '                    sr  = 16000\n'
    '                if sr != TARGET_SR:\n'
    '                    arr = T.Resample(sr, TARGET_SR)(arr)\n'
    '                out_path = out_dir / f"{lang_name}_{total:07d}.wav"\n'
    '                torchaudio.save(str(out_path), arr, TARGET_SR)\n'
    '                total += 1\n'
    '                if total % 1000 == 0:\n'
    '                    print(f"  Extracted {total}...")\n'
    '            except Exception:\n'
    '                continue\n'
    '    print(f"Done. {total} files extracted to {PROCESSED_DIR}")\n'
    '\n'
    '\n'
    'def process():\n'
    '    print("\\nSearching for audio files...")\n'
    '    search_roots = [INDICSYNTH_ROOT, PROCESSED_DIR]\n'
    '    audio_files = []\n'
    '    for root in search_roots:\n'
    '        if root.exists():\n'
    '            audio_files += [f for f in root.rglob("*") if f.suffix.lower() in AUDIO_EXTS]\n'
    '    audio_files = list(set(audio_files))\n'
    '\n'
    '    if not audio_files:\n'
    '        print("No audio found. Run --diagnose first.")\n'
    '        sys.exit(1)\n'
    '    print(f"Found {len(audio_files)} audio files.")\n'
    '\n'
    '    COLUMNS = ["path", "label", "dataset", "speaker_id", "duration", "language"]\n'
    '    existing_rows = []\n'
    '    existing_paths = set()\n'
    '    if MANIFEST_PATH.exists():\n'
    '        with open(MANIFEST_PATH, "r", newline="") as f:\n'
    '            reader = csv.DictReader(f)\n'
    '            existing_rows = list(reader)\n'
    '        existing_paths = {r["path"] for r in existing_rows}\n'
    '        print(f"Manifest has {len(existing_rows)} existing entries.")\n'
    '\n'
    '    new_rows = []\n'
    '    n_proc = 0\n'
    '    n_skip = 0\n'
    '\n'
    '    for idx, audio_path in enumerate(audio_files):\n'
    '        if idx % 1000 == 0:\n'
    '            print(f"  {idx}/{len(audio_files)}  ({len(new_rows)} new segments)")\n'
    '        lang = "unknown"\n'
    '        for part in audio_path.parts:\n'
    '            for kl in EXPECTED_LANGS:\n'
    '                if kl.lower() in part.lower():\n'
    '                    lang = kl\n'
    '                    break\n'
    '        try:\n'
    '            waveform, sr = torchaudio.load(str(audio_path))\n'
    '        except Exception:\n'
    '            n_skip += 1\n'
    '            continue\n'
    '        if waveform.shape[0] > 1:\n'
    '            waveform = waveform.mean(dim=0, keepdim=True)\n'
    '        if sr != TARGET_SR:\n'
    '            waveform = T.Resample(sr, TARGET_SR)(waveform)\n'
    '        peak = waveform.abs().max()\n'
    '        if peak < 1e-6:\n'
    '            n_skip += 1\n'
    '            continue\n'
    '        waveform = waveform / peak\n'
    '        duration_s = waveform.shape[1] / TARGET_SR\n'
    '        if duration_s < MIN_DURATION_S:\n'
    '            n_skip += 1\n'
    '            continue\n'
    '\n'
    '        target_n  = int(MAX_DURATION_S * TARGET_SR)\n'
    '        overlap_n = int(OVERLAP_S * TARGET_SR)\n'
    '        segments = []\n'
    '        if duration_s <= MAX_DURATION_S:\n'
    '            segments.append(waveform)\n'
    '        else:\n'
    '            n_samp = waveform.shape[1]\n'
    '            start  = 0\n'
    '            while start < n_samp:\n'
    '                end = min(start + target_n, n_samp)\n'
    '                seg = waveform[:, start:end]\n'
    '                if seg.shape[1] / TARGET_SR >= MIN_DURATION_S:\n'
    '                    segments.append(seg)\n'
    '                start += target_n - overlap_n\n'
    '                if end == n_samp:\n'
    '                    break\n'
    '\n'
    '        out_dir_lang = ML_ROOT / "data" / "processed" / "indicsynth" / lang\n'
    '        out_dir_lang.mkdir(parents=True, exist_ok=True)\n'
    '        speaker_id = f"indicsynth_{audio_path.stem}"\n'
    '\n'
    '        for seg_idx, seg in enumerate(segments):\n'
    '            seg_name = f"{audio_path.stem}_seg{seg_idx:03d}.wav"\n'
    '            seg_path = out_dir_lang / seg_name\n'
    '            rel_path = str(seg_path.relative_to(ML_ROOT)).replace("\\\\", "/")\n'
    '            if rel_path in existing_paths:\n'
    '                continue\n'
    '            torchaudio.save(str(seg_path), seg, TARGET_SR)\n'
    '            new_rows.append({\n'
    '                "path": rel_path, "label": 2,\n'
    '                "dataset": "indicsynth", "speaker_id": speaker_id,\n'
    '                "duration": f"{seg.shape[1]/TARGET_SR:.3f}", "language": lang\n'
    '            })\n'
    '        n_proc += 1\n'
    '\n'
    '    print(f"Processed: {n_proc}, Skipped: {n_skip}, New segments: {len(new_rows)}")\n'
    '    if new_rows:\n'
    '        all_rows = existing_rows + new_rows\n'
    '        with open(MANIFEST_PATH, "w", newline="") as f:\n'
    '            writer = csv.DictWriter(f, fieldnames=COLUMNS)\n'
    '            writer.writeheader()\n'
    '            writer.writerows(all_rows)\n'
    '        print(f"Manifest updated: {len(all_rows)} total entries.")\n'
    '        print("Next: py -3.11 split_dataset.py  then  py -3.11 training/train.py")\n'
    '\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    parser = argparse.ArgumentParser()\n'
    '    parser.add_argument("--diagnose",        action="store_true")\n'
    '    parser.add_argument("--extract-parquet", action="store_true")\n'
    '    parser.add_argument("--process",         action="store_true")\n'
    '    args = parser.parse_args()\n'
    '    if args.diagnose:\n'
    '        diagnose()\n'
    '    elif getattr(args, "extract_parquet", False):\n'
    '        extract_from_parquet()\n'
    '    elif args.process:\n'
    '        process()\n'
    '    else:\n'
    '        diagnose()\n'
)

indicsynth_path.write_text(indicsynth_content, encoding="utf-8")
print(f"[Fix 3] ✅ fix_indicsynth.py rewritten (no unicode escape issues)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PATCH COMPLETE")
print("="*60)
print("\nRun now:")
print("  py -3.11 training/train.py --debug")
print("\nIf debug passes (2 epochs complete), run full training:")
print("  py -3.11 training/train.py")
print("\nFor IndicSynth (separate terminal):")
print("  py -3.11 fix_indicsynth.py --diagnose")
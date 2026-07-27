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
    Returns (waveform [1,N], sample_rate) or None if can't decode.
    """
    # Format 1: dict with 'array' key (most common HF format)
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

    print(f"  [{lang}] Using column: '{audio_col}'")
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
            rel_path = str(out_path.relative_to(ML_ROOT)).replace("\\", "/")
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
        print(f"\nProcessing {lang}...")
        all_new.extend(extract_one_parquet(lang_dir, lang, OUT_DIR))

    if all_new:
        all_rows = existing_rows + all_new
        with open(MANIFEST_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            all_rows = [{k: r.get(k, '') for k in COLUMNS} for r in all_rows]  # strip unknown keys (e.g. gender)
            writer.writerows(all_rows)
        print(f"\nManifest updated: {len(all_rows)} total ({len(all_new)} new)")
        print("Next: py -3.11 split_dataset.py")
        print("      py -3.11 training/train.py --resume checkpoints/best_model.pt")
    else:
        print("\nNo segments saved.")
        print("Check the dict keys printed above — the audio data may be")
        print("under 'Source Reference Audio' or 'Target Reference Audio'.")
        print("Update the audio_col list in extract_one_parquet() if needed.")


if __name__ == "__main__":
    main()

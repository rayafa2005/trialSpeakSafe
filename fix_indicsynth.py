import sys
import csv
import argparse
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T

ML_ROOT      = Path(__file__).resolve().parent
PROJECT_ROOT = ML_ROOT.parent
INDICSYNTH_ROOT = PROJECT_ROOT / "tatvaani_ml" /"data"/ "raw" / "IndicSynth"
PROCESSED_DIR   = ML_ROOT / "data" / "processed" / "indicsynth"
MANIFEST_PATH   = ML_ROOT / "data" / "manifest.csv"
TARGET_SR       = 16000
MIN_DURATION_S  = 1.0
MAX_DURATION_S  = 4.0
OVERLAP_S       = 0.5
EXPECTED_LANGS  = ["Bengali", "Hindi", "Kannada", "Telugu"]
AUDIO_EXTS      = {".wav", ".flac", ".mp3", ".ogg"}


def diagnose():
    print("\n" + "="*60)
    print("INDICSYNTH DIAGNOSTIC")
    print("="*60)
    print(f"Looking in: {INDICSYNTH_ROOT}")

    if not INDICSYNTH_ROOT.exists():
        print(f"ERROR: Directory does not exist: {INDICSYNTH_ROOT}")
        return

    all_files   = list(INDICSYNTH_ROOT.rglob("*"))
    audio_files = [f for f in all_files if f.suffix.lower() in AUDIO_EXTS]
    parquet_files = list(INDICSYNTH_ROOT.rglob("*.parquet"))

    print(f"Total items:   {len(all_files)}")
    print(f"Audio files:   {len(audio_files)}")
    print(f"Parquet files: {len(parquet_files)}")
    print()

    print("Top-level contents:")
    for item in sorted(INDICSYNTH_ROOT.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            sub_audio = [f for f in item.rglob("*") if f.suffix.lower() in AUDIO_EXTS]
            sub_parquet = list(item.rglob("*.parquet"))
            total_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / 1e6
            print(f"  {item.name}/  ({len(sub_audio)} audio, {len(sub_parquet)} parquet, {total_mb:.0f} MB)")
            # Show first few files inside
            for sub in sorted(item.iterdir())[:3]:
                print(f"    {sub.name}")
        else:
            print(f"  {item.name}  ({item.stat().st_size/1024:.0f} KB)")

    print()
    if audio_files:
        print(f"First 5 audio files:")
        for f in audio_files[:5]:
            print(f"  {f.relative_to(INDICSYNTH_ROOT)}")
    elif parquet_files:
        print("No WAV/FLAC found. Parquet files detected.")
        print("Run: py -3.11 fix_indicsynth.py --extract-parquet")
    else:
        print("No audio or parquet. Download may be incomplete.")


def extract_from_parquet():
    try:
        import pyarrow.parquet as pq
        import numpy as np
    except ImportError:
        print("Install pyarrow: py -3.11 -m pip install pyarrow")
        sys.exit(1)

    parquet_files = list(INDICSYNTH_ROOT.rglob("*.parquet"))
    if not parquet_files:
        print("No parquet files found. Run --diagnose first.")
        return

    print(f"Found {len(parquet_files)} parquet files.")
    total = 0
    for pf_path in parquet_files:
        lang_name = pf_path.parent.parent.name
        out_dir = PROCESSED_DIR / lang_name
        out_dir.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(str(pf_path))
        cols  = table.schema.names
        print(f"  {pf_path.name} columns: {cols}")
        audio_col = next((c for c in ["audio","speech","waveform"] if c in cols), None)
        if not audio_col:
            print(f"  No audio column found. Skipping.")
            continue
        for row in table.to_pydict()[audio_col]:
            try:
                if isinstance(row, dict):
                    arr = torch.tensor(row["array"], dtype=torch.float32).unsqueeze(0)
                    sr  = row.get("sampling_rate", 16000)
                else:
                    arr = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
                    sr  = 16000
                if sr != TARGET_SR:
                    arr = T.Resample(sr, TARGET_SR)(arr)
                out_path = out_dir / f"{lang_name}_{total:07d}.wav"
                torchaudio.save(str(out_path), arr, TARGET_SR)
                total += 1
                if total % 1000 == 0:
                    print(f"  Extracted {total}...")
            except Exception:
                continue
    print(f"Done. {total} files extracted to {PROCESSED_DIR}")


def process():
    print("\nSearching for audio files...")
    search_roots = [INDICSYNTH_ROOT, PROCESSED_DIR]
    audio_files = []
    for root in search_roots:
        if root.exists():
            audio_files += [f for f in root.rglob("*") if f.suffix.lower() in AUDIO_EXTS]
    audio_files = list(set(audio_files))

    if not audio_files:
        print("No audio found. Run --diagnose first.")
        sys.exit(1)
    print(f"Found {len(audio_files)} audio files.")

    COLUMNS = ["path", "label", "dataset", "speaker_id", "duration", "language"]
    existing_rows = []
    existing_paths = set()
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        existing_paths = {r["path"] for r in existing_rows}
        print(f"Manifest has {len(existing_rows)} existing entries.")

    new_rows = []
    n_proc = 0
    n_skip = 0

    for idx, audio_path in enumerate(audio_files):
        if idx % 1000 == 0:
            print(f"  {idx}/{len(audio_files)}  ({len(new_rows)} new segments)")
        lang = "unknown"
        for part in audio_path.parts:
            for kl in EXPECTED_LANGS:
                if kl.lower() in part.lower():
                    lang = kl
                    break
        try:
            waveform, sr = torchaudio.load(str(audio_path))
        except Exception:
            n_skip += 1
            continue
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = T.Resample(sr, TARGET_SR)(waveform)
        peak = waveform.abs().max()
        if peak < 1e-6:
            n_skip += 1
            continue
        waveform = waveform / peak
        duration_s = waveform.shape[1] / TARGET_SR
        if duration_s < MIN_DURATION_S:
            n_skip += 1
            continue

        target_n  = int(MAX_DURATION_S * TARGET_SR)
        overlap_n = int(OVERLAP_S * TARGET_SR)
        segments = []
        if duration_s <= MAX_DURATION_S:
            segments.append(waveform)
        else:
            n_samp = waveform.shape[1]
            start  = 0
            while start < n_samp:
                end = min(start + target_n, n_samp)
                seg = waveform[:, start:end]
                if seg.shape[1] / TARGET_SR >= MIN_DURATION_S:
                    segments.append(seg)
                start += target_n - overlap_n
                if end == n_samp:
                    break

        out_dir_lang = ML_ROOT / "data" / "processed" / "indicsynth" / lang
        out_dir_lang.mkdir(parents=True, exist_ok=True)
        speaker_id = f"indicsynth_{audio_path.stem}"

        for seg_idx, seg in enumerate(segments):
            seg_name = f"{audio_path.stem}_seg{seg_idx:03d}.wav"
            seg_path = out_dir_lang / seg_name
            rel_path = str(seg_path.relative_to(ML_ROOT)).replace("\\", "/")
            if rel_path in existing_paths:
                continue
            torchaudio.save(str(seg_path), seg, TARGET_SR)
            new_rows.append({
                "path": rel_path, "label": 2,
                "dataset": "indicsynth", "speaker_id": speaker_id,
                "duration": f"{seg.shape[1]/TARGET_SR:.3f}", "language": lang
            })
        n_proc += 1

    print(f"Processed: {n_proc}, Skipped: {n_skip}, New segments: {len(new_rows)}")
    if new_rows:
        all_rows = existing_rows + new_rows
        with open(MANIFEST_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Manifest updated: {len(all_rows)} total entries.")
        print("Next: py -3.11 split_dataset.py  then  py -3.11 training/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose",        action="store_true")
    parser.add_argument("--extract-parquet", action="store_true")
    parser.add_argument("--process",         action="store_true")
    args = parser.parse_args()
    if args.diagnose:
        diagnose()
    elif getattr(args, "extract_parquet", False):
        extract_from_parquet()
    elif args.process:
        process()
    else:
        diagnose()

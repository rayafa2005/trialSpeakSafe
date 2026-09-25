"""
process_dataset.py
==================
Tatvaani ML Pipeline — Step 1: Dataset Ingestion & Preprocessing

Handles all datasets:
  - ASVspoof 2019 LA  (FLAC, 16kHz, already mono)
  - WaveFake          (WAV, 22.05kHz → needs resample)
  - IndicSynth        (WAV, 16kHz, mono)
  - MLADDC            (WAV, mixed rates)
  - InDeepFake        (MP4/WAV, mixed)

Output: canonical WAV files at 16kHz mono float32, saved to
        tatvaani_ml/data/processed/
        + master manifest.csv

Usage:
  py -3.11 process_dataset.py --dataset asvspoof --input data/raw/ASVspoof2019_LA
  py -3.11 process_dataset.py --dataset wavefake  --input data/raw/WaveFake
  py -3.11 process_dataset.py --dataset indicsynth --input data/raw/IndicSynth
  py -3.11 process_dataset.py --all
"""

import os
import csv
import argparse
import warnings
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_SR       = 16000       # All audio resampled to 16kHz
TARGET_CHANNELS = 1           # Mono
MIN_DURATION    = 1.0         # Discard clips shorter than 1 second
MAX_DURATION    = 5.0         # Split clips longer than 5 seconds
SPLIT_HOP       = 2.5         # Hop size when splitting long clips (seconds)
SPLIT_LEN       = 5.0         # Length of each split segment (seconds)
OUTPUT_DIR      = Path("data/processed")
MANIFEST_PATH   = Path("data/manifest.csv")

# ─── Manifest CSV columns ─────────────────────────────────────────────────────
MANIFEST_COLS = [
    "path",          # relative path to processed WAV
    "label",         # 0=bonafide(safe), 1=caution, 2=spoof(danger)
    "dataset",       # source dataset name
    "speaker_id",    # speaker identifier (for disjoint splitting)
    "language",      # language code (en, kn, hi, te, etc.)
    "duration",      # clip duration in seconds
    "gender",        # m/f/unknown
]

# ─── Install check helper ─────────────────────────────────────────────────────
def check_dependencies():
    """Check required packages are available under py -3.11."""
    missing = []
    try:
        import torchaudio
    except ImportError:
        missing.append("torchaudio")
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    if missing:
        print(f"[ERROR] Missing packages: {missing}")
        print("Run: py -3.11 -m pip install torchaudio torch soundfile librosa")
        exit(1)
    print("[OK] All dependencies found.")

# ─── Core audio utilities ─────────────────────────────────────────────────────

def load_audio(path: Path) -> tuple:
    """
    Load any audio file to float32 numpy array at TARGET_SR mono.
    Handles: WAV, FLAC, MP3, M4A, FLAC via torchaudio.
    Falls back to librosa for edge cases.
    Returns: (samples: np.ndarray float32, sample_rate: int)
    """
    import torch
    import torchaudio

    path = str(path)
    try:
        # Primary: torchaudio (fast, handles FLAC/WAV natively)
        waveform, sr = torchaudio.load(path)

        # Convert to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=TARGET_SR
            )
            waveform = resampler(waveform)

        # Convert to float32 numpy, shape: (samples,)
        samples = waveform.squeeze(0).numpy().astype(np.float32)
        return samples, TARGET_SR

    except Exception as e:
        # Fallback: librosa (handles MP3, M4A, and broken headers)
        try:
            import librosa
            samples, sr = librosa.load(path, sr=TARGET_SR, mono=True)
            return samples.astype(np.float32), TARGET_SR
        except Exception as e2:
            print(f"[SKIP] Cannot load {path}: {e2}")
            return None, None


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """
    Peak-normalize audio to [-1, 1].
    If the clip is silent (all zeros), return None to discard it.
    """
    peak = np.abs(samples).max()
    if peak < 1e-6:
        return None  # Silent clip — discard
    return samples / peak


def vad_trim(samples: np.ndarray, sr: int, top_db: float = 30.0) -> np.ndarray:
    """
    Voice Activity Detection trim — removes leading/trailing silence.
    Uses librosa's trim with top_db threshold.
    Returns trimmed samples or None if too short after trimming.
    """
    import librosa
    trimmed, _ = librosa.effects.trim(samples, top_db=top_db)
    duration = len(trimmed) / sr
    if duration < MIN_DURATION:
        return None  # Too short after trimming
    return trimmed


def split_long_clip(samples: np.ndarray, sr: int) -> list:
    """
    If clip is longer than MAX_DURATION, split into SPLIT_LEN segments
    with SPLIT_HOP overlap. Returns list of numpy arrays.
    If clip is within range, returns [samples] as single-element list.
    """
    duration = len(samples) / sr
    if duration <= MAX_DURATION:
        return [samples]

    segments = []
    hop_samples = int(SPLIT_HOP * sr)
    seg_samples = int(SPLIT_LEN * sr)
    start = 0

    while start + seg_samples <= len(samples):
        segment = samples[start : start + seg_samples]
        segments.append(segment)
        start += hop_samples

    return segments if segments else [samples[:int(MAX_DURATION * sr)]]


def save_processed(samples: np.ndarray, out_path: Path):
    """Save float32 numpy array as 16kHz mono WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), samples, TARGET_SR, subtype="FLOAT")


def process_single_file(
    in_path: Path,
    out_path: Path,
    label: int,
    dataset: str,
    speaker_id: str,
    language: str,
    gender: str = "unknown"
) -> list:
    """
    Full pipeline for a single audio file:
    load → mono → resample → normalize → VAD → split → save
    Returns list of manifest row dicts (one per output segment).
    """
    samples, sr = load_audio(in_path)
    if samples is None:
        return []

    samples = normalize_audio(samples)
    if samples is None:
        return []

    samples = vad_trim(samples, sr)
    if samples is None:
        return []

    segments = split_long_clip(samples, sr)
    rows = []

    for i, seg in enumerate(segments):
        # Build output filename: dataset_speakerid_originalname_segN.wav
        seg_name = f"{dataset}_{speaker_id}_{out_path.stem}_s{i:02d}.wav"
        seg_path = OUTPUT_DIR / dataset / seg_name

        save_processed(seg, seg_path)

        rows.append({
            "path": str(seg_path),
            "label": label,
            "dataset": dataset,
            "speaker_id": speaker_id,
            "language": language,
            "duration": round(len(seg) / TARGET_SR, 3),
            "gender": gender,
        })

    return rows


# ─── Dataset-specific processors ──────────────────────────────────────────────

def process_asvspoof(input_dir: Path, manifest_rows: list):
    """
    ASVspoof 2019 LA dataset processor.
    Structure:
      LA/ASVspoof2019_LA_train/flac/*.flac
      LA/ASVspoof2019_LA_train/ASVspoof2019.LA.cm.train.trn.txt  (protocol file)
    Label file format: speaker_id filename - system_id label
    """
    print("\n[ASVspoof 2019 LA] Processing...")

    # Find protocol files (label files)
    proto_files = list(input_dir.rglob("*.txt"))
    label_map = {}  # filename → (label, speaker_id, system_id)

    for proto in proto_files:
        # Only process countermeasure protocol files
        if "cm." not in proto.name:
            continue
        with open(proto, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # Format: SPEAKER_ID FILENAME - SYSTEM_ID LABEL
                spk_id   = parts[0]
                filename = parts[1]
                label_str = parts[4]  # "bonafide" or "spoof"
                label = 0 if label_str == "bonafide" else 2
                label_map[filename] = (label, spk_id)

    # Process FLAC files
    flac_files = list(input_dir.rglob("*.flac"))
    print(f"  Found {len(flac_files)} FLAC files")
    processed = 0

    for flac_path in flac_files:
        stem = flac_path.stem
        if stem not in label_map:
            continue  # Skip files not in protocol

        label, spk_id = label_map[stem]
        rows = process_single_file(
            in_path=flac_path,
            out_path=flac_path,  # used only for naming
            label=label,
            dataset="asvspoof",
            speaker_id=spk_id,
            language="en",
            gender="unknown"
        )
        manifest_rows.extend(rows)
        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed}/{len(flac_files)}")

    print(f"  [ASVspoof] Done: {processed} files → {len(manifest_rows)} segments")


def process_wavefake(input_dir: Path, manifest_rows: list):
    """
    WaveFake dataset processor.
    Structure:
      WaveFake/
        LJSpeech-1.1/wavs/*.wav          (bonafide, real)
        generated_audio/
          ljspeech_*/                     (spoof, per-vocoder folder)
            *.wav
    WaveFake is 22.05kHz — torchaudio resampler handles this automatically.
    """
    print("\n[WaveFake] Processing...")
    count = 0

    # Bonafide: LJSpeech real audio
    real_dirs = list(input_dir.rglob("wavs"))
    for real_dir in real_dirs:
        for wav_path in real_dir.glob("*.wav"):
            # LJSpeech speaker ID: LJ (single speaker dataset)
            rows = process_single_file(
                in_path=wav_path,
                out_path=wav_path,
                label=0,  # bonafide
                dataset="wavefake",
                speaker_id="LJ",
                language="en",
                gender="f"  # LJSpeech is female speaker
            )
            manifest_rows.extend(rows)
            count += 1

    # Spoof: generated_audio subfolders (one per vocoder)
    gen_dir = input_dir / "generated_audio"
    if gen_dir.exists():
        for vocoder_dir in gen_dir.iterdir():
            if not vocoder_dir.is_dir():
                continue
            vocoder_name = vocoder_dir.name  # e.g. "ljspeech_melgan"
            for wav_path in vocoder_dir.rglob("*.wav"):
                rows = process_single_file(
                    in_path=wav_path,
                    out_path=wav_path,
                    label=2,  # spoof
                    dataset="wavefake",
                    speaker_id=f"LJ_{vocoder_name}",
                    language="en",
                    gender="f"
                )
                manifest_rows.extend(rows)
                count += 1

    print(f"  [WaveFake] Done: {count} files → segments added to manifest")


def process_indicsynth(input_dir: Path, manifest_rows: list):
    """
    IndicSynth dataset processor.
    Structure:
      IndicSynth/
        Kannada/
          audio/*.wav          (synthetic — label=2 spoof)
          metadata.csv         (columns: filename, speaker_id, gender, ...)
        Hindi/
          audio/*.wav
          metadata.csv
        ...
    Bonafide for IndicSynth comes from the IndicSuperb source dataset
    (referenced in metadata). We label all IndicSynth audio as spoof (2).
    """
    print("\n[IndicSynth] Processing...")

    # Language code mapping
    lang_map = {
        "Kannada": "kn", "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
        "Bengali": "bn", "Gujarati": "gu", "Malayalam": "ml",
        "Marathi": "mr", "Odia": "or", "Punjabi": "pa",
        "Sanskrit": "sa", "Urdu": "ur"
    }

    count = 0
    for lang_dir in input_dir.iterdir():
        if not lang_dir.is_dir():
            continue

        lang_name = lang_dir.name
        lang_code = lang_map.get(lang_name, "unknown")
        audio_dir = lang_dir / "audio"
        meta_path = lang_dir / "metadata.csv"

        if not audio_dir.exists():
            print(f"  [SKIP] No audio dir in {lang_dir}")
            continue

        # Load metadata for speaker IDs and gender
        speaker_meta = {}  # filename_stem → (speaker_id, gender)
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try common column names in IndicSynth metadata
                    fname = row.get("filename", row.get("file", "")).replace(".wav", "")
                    spk   = row.get("speaker_id", row.get("target_speaker", "unknown"))
                    gen   = row.get("gender", "unknown").lower()
                    if gen not in ("m", "f"):
                        gen = "unknown"
                    speaker_meta[fname] = (spk, gen)

        for wav_path in audio_dir.glob("*.wav"):
            stem = wav_path.stem
            spk_id, gender = speaker_meta.get(stem, (f"{lang_code}_unknown", "unknown"))

            rows = process_single_file(
                in_path=wav_path,
                out_path=wav_path,
                label=2,  # all IndicSynth = synthetic = spoof
                dataset="indicsynth",
                speaker_id=f"{lang_code}_{spk_id}",
                language=lang_code,
                gender=gender
            )
            manifest_rows.extend(rows)
            count += 1

        print(f"  [{lang_name}] processed {count} files so far...")

    print(f"  [IndicSynth] Done: {count} files → segments added to manifest")


def process_public_collected(input_dir: Path, manifest_rows: list):
    """
    Public-collected consented audio processor.
    Expected structure:
      public_collected/
        UUID_xxxx.wav   (already renamed to UUID before processing)
        collection_log.csv  (UUID, language, gender, age_group)

    All public-collected audio is bonafide (label=0, Safe).
    CRITICAL: speaker UUIDs ensure no identity leakage across splits.
    """
    print("\n[Public Collected] Processing...")

    log_path = input_dir / "collection_log.csv"
    uuid_meta = {}

    if log_path.exists():
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uuid = row.get("uuid", "")
                uuid_meta[uuid] = {
                    "language": row.get("language", "unknown"),
                    "gender":   row.get("gender", "unknown").lower(),
                }

    count = 0
    for wav_path in input_dir.glob("*.wav"):
        uuid = wav_path.stem
        meta = uuid_meta.get(uuid, {"language": "unknown", "gender": "unknown"})

        rows = process_single_file(
            in_path=wav_path,
            out_path=wav_path,
            label=0,  # bonafide
            dataset="public_collected",
            speaker_id=uuid,  # UUID as speaker ID — fully anonymous
            language=meta["language"],
            gender=meta["gender"]
        )
        manifest_rows.extend(rows)
        count += 1

    print(f"  [Public Collected] Done: {count} files processed")


def process_mlaad(input_dir: Path, manifest_rows: list):
    """
    MLAAD / MLAAD-tiny dataset processor.
    Structure:
      MLAAD-tiny/
        original/
          de/*.wav (bonafide)
          en/*.wav (bonafide)
        fake/
          de/<model_name>/*.wav (spoof)
          en/<model_name>/*.wav (spoof)
    """
    print("\n[MLAAD] Processing...")
    count = 0
    existing_paths = {r["path"] for r in manifest_rows}

    # 1. Process original (bonafide, Safe, label=0)
    orig_dir = input_dir / "original"
    if orig_dir.exists():
        for lang_dir in orig_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name
            for wav_path in lang_dir.rglob("*.wav"):
                stem = wav_path.stem
                spk = stem.split("_")[0] if "_" in stem else "spk"
                rows = process_single_file(
                    in_path=wav_path,
                    out_path=wav_path,
                    label=0,  # bonafide / Safe
                    dataset="mlaad",
                    speaker_id=f"mlaad_{lang}_{spk}",
                    language=lang,
                    gender="unknown"
                )
                for r in rows:
                    if r["path"] not in existing_paths:
                        manifest_rows.append(r)
                        existing_paths.add(r["path"])
                count += 1

    # 2. Process fake (synthetic, Danger, label=2)
    fake_dir = input_dir / "fake"
    if fake_dir.exists():
        for lang_dir in fake_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name
            for model_dir in lang_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name
                for wav_path in model_dir.rglob("*.wav"):
                    stem = wav_path.stem
                    spk = stem.split("_")[0] if "_" in stem else "fake"
                    rows = process_single_file(
                        in_path=wav_path,
                        out_path=wav_path,
                        label=2,  # spoof / Danger
                        dataset="mlaad",
                        speaker_id=f"mlaad_{model_name}_{lang}_{spk}",
                        language=lang,
                        gender="unknown"
                    )
                    for r in rows:
                        if r["path"] not in existing_paths:
                            manifest_rows.append(r)
                            existing_paths.add(r["path"])
                    count += 1

    print(f"  [MLAAD] Done: {count} files processed → segments added to manifest")


def process_librispeech_dataset(input_dir: Path, manifest_rows: list):
    """
    LibriSpeech train-clean-100 processor.
    Structure:
      LibriSpeech/
        train-clean-100/
          <speaker_id>/<chapter_id>/*.flac
    """
    print("\n[LibriSpeech] Processing...")
    src = input_dir / "train-clean-100" if (input_dir / "train-clean-100").exists() else input_dir
    flac_files = list(src.rglob("*.flac"))
    print(f"  Found {len(flac_files):,} FLAC files in LibriSpeech")

    existing_paths = {r["path"] for r in manifest_rows}
    count = 0
    for flac in flac_files:
        spk_id = f"libri_{flac.parent.parent.name}" if flac.parent.parent.name.isdigit() else f"libri_{flac.parent.name}"
        rows = process_single_file(
            in_path=flac,
            out_path=flac,
            label=0,  # bonafide / Safe
            dataset="librispeech",
            speaker_id=spk_id,
            language="en",
            gender="unknown"
        )
        for r in rows:
            if r["path"] not in existing_paths:
                manifest_rows.append(r)
                existing_paths.add(r["path"])
        count += 1
        if count % 2000 == 0:
            print(f"  [LibriSpeech] Processed {count}/{len(flac_files)} files...")

    print(f"  [LibriSpeech] Done: {count} files processed")


def find_dir(candidates: list) -> Path:
    """Find the first existing directory among candidate paths."""
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


# ─── Manifest writer ──────────────────────────────────────────────────────────

def write_manifest(rows: list, path: Path):
    """Write all processed file metadata to master manifest CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Manifest] Written: {len(rows)} entries → {path}")


def print_stats(rows: list):
    """Print dataset statistics for bias auditing."""
    from collections import Counter

    labels   = Counter(r["label"] for r in rows)
    datasets = Counter(r["dataset"] for r in rows)
    langs    = Counter(r["language"] for r in rows)
    genders  = Counter(r["gender"] for r in rows)

    print("\n" + "="*50)
    print("DATASET STATISTICS (Bias Audit)")
    print("="*50)
    print(f"Total segments: {len(rows)}")
    print(f"\nLabel distribution:")
    for k, v in sorted(labels.items()):
        name = {0: "Safe/bonafide", 1: "Caution/half-truth", 2: "Danger/spoof"}.get(int(k), str(k))
        print(f"  {name}: {v} ({100*v/len(rows):.1f}%)")
    print(f"\nDataset distribution:")
    for k, v in datasets.most_common():
        print(f"  {k}: {v}")
    print(f"\nLanguage distribution (top 10):")
    for k, v in langs.most_common(10):
        print(f"  {k}: {v}")
    print(f"\nGender distribution:")
    for k, v in genders.most_common():
        print(f"  {k}: {v}")
    print("="*50)


# ─── Main entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tatvaani Dataset Processor")
    parser.add_argument("--all", action="store_true",
        help="Process all available datasets found in data/raw/")
    parser.add_argument("--dataset", choices=[
        "asvspoof", "mlaad", "librispeech", "wavefake", "indicsynth", "public_collected", "all"
    ], default="all")
    parser.add_argument("--input",  type=str, default=None,
        help="Input directory for single dataset mode")
    parser.add_argument("--output", type=str, default="data/processed",
        help="Output directory for processed WAV files")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    args = parser.parse_args()

    global OUTPUT_DIR, MANIFEST_PATH
    OUTPUT_DIR    = Path(args.output)
    MANIFEST_PATH = Path(args.manifest)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    check_dependencies()

    manifest_rows = []

    # Load existing manifest if it exists (allows incremental processing)
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_rows = list(reader)
        print(f"[Manifest] Loaded {len(manifest_rows)} existing entries.")

    process_all = args.all or args.dataset == "all"

    # 1. ASVspoof 2019 LA
    if process_all or args.dataset == "asvspoof":
        p = Path(args.input) if args.input else find_dir([
            "data/raw/LA",
            "data/raw/ASVspoof2019_LA",
            "data/raw/ASV2019",
        ])
        if p and p.exists():
            process_asvspoof(p, manifest_rows)
        else:
            if not process_all:
                print(f"[SKIP] ASVspoof not found at {p}")

    # 2. MLAAD
    if process_all or args.dataset == "mlaad":
        p = Path(args.input) if args.input else find_dir([
            "data/raw/MLAAD-tiny",
            "data/raw/MLAAD",
            "data/raw/mlaad-tiny",
            "data/raw/mlaad",
        ])
        if p and p.exists():
            process_mlaad(p, manifest_rows)
        else:
            if not process_all:
                print(f"[SKIP] MLAAD not found at {p}")

    # 3. LibriSpeech
    if process_all or args.dataset == "librispeech":
        p = Path(args.input) if args.input else find_dir([
            "data/raw/LibriSpeech/train-clean-100",
            "data/raw/LibriSpeech",
        ])
        if p and p.exists():
            process_librispeech_dataset(p, manifest_rows)
        else:
            if not process_all:
                print(f"[SKIP] LibriSpeech not found at {p}")

    # 4. WaveFake
    if process_all or args.dataset == "wavefake":
        p = Path(args.input) if args.input else find_dir(["data/raw/WaveFake"])
        if p and p.exists():
            process_wavefake(p, manifest_rows)
        else:
            if not process_all:
                print(f"[SKIP] WaveFake not found at {p}")

    # 5. IndicSynth
    if process_all or args.dataset == "indicsynth":
        p = Path(args.input) if args.input else find_dir(["data/raw/IndicSynth"])
        if p and p.exists():
            process_indicsynth(p, manifest_rows)
        else:
            if not process_all:
                print(f"[SKIP] IndicSynth not found at {p}")

    # 6. Public Collected
    if process_all or args.dataset == "public_collected":
        p = Path(args.input) if args.input else find_dir(["data/raw/public_collected"])
        if p and p.exists():
            process_public_collected(p, manifest_rows)
        else:
            if not process_all:
                print(f"[SKIP] public_collected not found at {p}")

    write_manifest(manifest_rows, MANIFEST_PATH)
    print_stats(manifest_rows)
    print("\n[DONE] Dataset processing complete.")
    print("Next step: py -3.10 split_dataset.py --balance")


if __name__ == "__main__":
    main()
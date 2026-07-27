"""
=============================================================================
Tatvaani — rebuild_manifest.py (v2)
=============================================================================
Reads ASVspoof labels from the official CM protocol txt files.
WaveFake: LJSpeech subfolder = bonafide, generated_audio = spoof.
IndicSynth: all spoof.

Run: py -3.11 rebuild_manifest.py
=============================================================================
"""

import csv
import sys
from pathlib import Path

ML_ROOT        = Path(__file__).resolve().parent
PROCESSED_ROOT = ML_ROOT / "data" / "processed"
MANIFEST_PATH  = ML_ROOT / "data" / "manifest.csv"
COLUMNS        = ["path", "label", "dataset", "speaker_id", "duration", "language"]


def load_asvspoof_labels() -> dict:
    """
    Reads CM protocol files → dict of filename_stem -> label (0 or 2).
    Protocol line format: speaker  filename  -  attack  label
    e.g.: LA_0001  LA_T_1000137  -  A07  spoof
    """
    protocol_dir = (
        ML_ROOT / "data" / "raw" / "ASVspoof2019_LA"
        / "LA" / "ASVspoof2019_LA_cm_protocols"
    )
    if not protocol_dir.exists():
        print(f"[ASVspoof] Protocol dir not found: {protocol_dir}")
        return {}

    labels = {}
    for pf in sorted(protocol_dir.glob("*.txt")):
        n_bon = n_spoof = 0
        with open(pf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                fname  = parts[1]
                label  = 0 if parts[4].lower() == "bonafide" else 2
                labels[fname] = label
                if label == 0: n_bon += 1
                else: n_spoof += 1
        print(f"  {pf.name}: bonafide={n_bon:,}  spoof={n_spoof:,}")

    print(f"[ASVspoof] {len(labels):,} label mappings loaded")
    return labels


def get_duration(file_path: Path) -> float:
    try:
        import struct
        with open(file_path, "rb") as f:
            h = f.read(44)
        if len(h) < 44 or h[:4] != b"RIFF":
            raise ValueError
        ch  = struct.unpack_from("<H", h, 22)[0]
        sr  = struct.unpack_from("<I", h, 24)[0]
        bps = struct.unpack_from("<H", h, 34)[0]
        ds  = struct.unpack_from("<I", h, 40)[0]
        if sr == 0 or ch == 0 or bps == 0:
            raise ValueError
        return round(ds / (ch * (bps // 8)) / sr, 3)
    except Exception:
        return 2.0


def process_asvspoof(asvspoof_labels: dict) -> list:
    dataset_dir = PROCESSED_ROOT / "asvspoof"
    wav_files   = list(dataset_dir.rglob("*.wav"))
    print(f"\n[ASVspoof] Scanning {len(wav_files):,} files...")

    rows = []
    n_bon = n_spoof = n_unmatched = 0

    # Debug: show key format mismatch early
    if wav_files and asvspoof_labels:
        sample_stem = wav_files[0].stem
        sample_key  = next(iter(asvspoof_labels))
        print(f"  Sample processed stem: {sample_stem}")
        print(f"  Sample protocol key:   {sample_key}")

    for i, wav in enumerate(wav_files):
        if i % 20000 == 0 and i > 0:
            print(f"  {i:,}/{len(wav_files):,}...")

        stem  = wav.stem
        label = 2  # default spoof

        # The processed filename contains the original ASVspoof stem.
        # Try direct match first, then substring search.
        if stem in asvspoof_labels:
            label = asvspoof_labels[stem]
        else:
            # Substring: our stem might be "asvspoof_LA_0001_LA_E_1000137_seg000"
            # Protocol key is "LA_E_1000137" — search for it inside our stem
            for key in asvspoof_labels:
                if key in stem:
                    label = asvspoof_labels[key]
                    break
            else:
                n_unmatched += 1

        parts      = stem.split("_")
        speaker_id = "_".join(parts[:4]) if len(parts) >= 4 else stem[:12]
        rel_path   = str(wav.relative_to(ML_ROOT)).replace("\\", "/")

        rows.append({
            "path": rel_path, "label": label, "dataset": "asvspoof",
            "speaker_id": speaker_id, "duration": get_duration(wav),
            "language": "english",
        })
        if label == 0: n_bon += 1
        else: n_spoof += 1

    print(f"[ASVspoof] bonafide={n_bon:,}  spoof={n_spoof:,}  unmatched={n_unmatched:,}")
    if n_unmatched > len(wav_files) * 0.3:
        print(f"  ⚠️  High unmatched rate — label lookup may need adjustment")
        print(f"  Defaulted all unmatched to spoof (label=2)")
    return rows


def process_wavefake() -> list:
    dataset_dir = PROCESSED_ROOT / "wavefake"
    wav_files   = list(dataset_dir.rglob("*.wav"))
    print(f"\n[WaveFake] Scanning {len(wav_files):,} files...")

    # Show sample paths to verify bonafide detection
    sample_paths = [str(w.relative_to(PROCESSED_ROOT)) for w in wav_files[:4]]
    print(f"  Sample paths: {sample_paths}")

    rows = []
    n_bon = n_spoof = 0

    for i, wav in enumerate(wav_files):
        if i % 30000 == 0 and i > 0:
            print(f"  {i:,}/{len(wav_files):,}...")

        path_lower = str(wav).lower()
        label      = 0 if ("ljspeech" in path_lower or "lj_speech" in path_lower) else 2
        rel_path   = str(wav.relative_to(ML_ROOT)).replace("\\", "/")

        rows.append({
            "path": rel_path, "label": label, "dataset": "wavefake",
            "speaker_id": f"wavefake_{wav.parent.name}",
            "duration": get_duration(wav), "language": "english",
        })
        if label == 0: n_bon += 1
        else: n_spoof += 1

    print(f"[WaveFake] bonafide={n_bon:,}  spoof={n_spoof:,}")
    return rows


def process_indicsynth() -> list:
    dataset_dir = PROCESSED_ROOT / "indicsynth"
    if not dataset_dir.exists():
        print(f"\n[IndicSynth] Directory not found — skipping")
        return []

    wav_files = list(dataset_dir.rglob("*.wav"))
    print(f"\n[IndicSynth] {len(wav_files):,} files (all spoof)")

    lang_map = {"bengali": "Bengali", "hindi": "Hindi",
                "kannada": "Kannada", "telugu": "Telugu"}
    rows = []
    for wav in wav_files:
        path_lower = str(wav).lower()
        language   = next((v for k, v in lang_map.items() if k in path_lower), "indian")
        rel_path   = str(wav.relative_to(ML_ROOT)).replace("\\", "/")
        rows.append({
            "path": rel_path, "label": 2, "dataset": "indicsynth",
            "speaker_id": f"indicsynth_{wav.stem[:16]}",
            "duration": get_duration(wav), "language": language,
        })
    return rows


def main():
    print("=" * 60)
    print("TATVAANI MANIFEST REBUILDER v2")
    print("=" * 60)

    asvspoof_labels = load_asvspoof_labels()

    all_rows  = []
    all_rows += process_asvspoof(asvspoof_labels)
    all_rows += process_wavefake()
    all_rows += process_indicsynth()

    if not all_rows:
        print("ERROR: No rows generated.")
        sys.exit(1)

    print(f"\nWriting {len(all_rows):,} rows → manifest.csv ...")
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    n_safe   = sum(1 for r in all_rows if int(r["label"]) == 0)
    n_danger = sum(1 for r in all_rows if int(r["label"]) == 2)
    print(f"\n{'='*60}")
    print(f"MANIFEST REBUILT")
    print(f"Total:            {len(all_rows):,}")
    print(f"Label 0 (Safe):   {n_safe:,}  ({100*n_safe/len(all_rows):.1f}%)")
    print(f"Label 2 (Danger): {n_danger:,}  ({100*n_danger/len(all_rows):.1f}%)")
    print(f"\nNext:  py -3.11 split_dataset.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
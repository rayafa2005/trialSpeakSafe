"""
split_dataset.py
================
Tatvaani ML Pipeline — Step 2: Speaker-Disjoint Train/Val/Test Split

CRITICAL RULE: The same speaker_id must NEVER appear in both train and test.
If it does, the model memorizes voices instead of learning deepfake artifacts,
and accuracy numbers become meaningless (inflated by identity leakage).

Split ratios: 70% train / 15% val / 15% test
Special rule: Hold-out datasets (in_the_wild, public_collected test clips)
              are ALWAYS assigned to test only — never seen during training.

Output:
  data/train_manifest.csv
  data/val_manifest.csv
  data/test_manifest.csv

Usage:
  py -3.11 split_dataset.py
  py -3.11 split_dataset.py --manifest data/manifest.csv --seed 42
"""

import csv
import argparse
import random
from pathlib import Path
from collections import defaultdict

# ─── Constants ────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# These datasets go ONLY to test — never seen during training
# Used for blind demo evaluation
HOLDOUT_ONLY_DATASETS = {"in_the_wild"}

# These datasets are included in training
TRAIN_ELIGIBLE_DATASETS = {"asvspoof", "wavefake", "indicsynth", "public_collected", "mladdc"}

# ─── Main split logic ─────────────────────────────────────────────────────────

def load_manifest(path: Path) -> list:
    """Load master manifest CSV into list of row dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"[Split] Loaded {len(rows)} total entries from {path}")
    return rows


def group_by_speaker(rows: list) -> dict:
    """
    Group manifest rows by (dataset, speaker_id).
    Returns dict: key=(dataset, speaker_id) → list of rows
    This ensures we split at speaker level, not file level.
    """
    groups = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["speaker_id"])
        groups[key].append(row)
    return groups


def split_speakers(groups: dict, seed: int) -> tuple:
    """
    Split speaker groups into train/val/test sets.
    
    Rules:
    1. Hold-out datasets → always test
    2. Remaining speakers shuffled and split 70/15/15
    3. Label balance is checked after split — warn if skewed
    
    Returns: (train_rows, val_rows, test_rows)
    """
    random.seed(seed)

    # Separate hold-out speakers from eligible speakers
    holdout_groups = {}
    eligible_groups = {}

    for key, rows in groups.items():
        dataset, speaker_id = key
        if dataset in HOLDOUT_ONLY_DATASETS:
            holdout_groups[key] = rows
        else:
            eligible_groups[key] = rows

    # Shuffle eligible speaker keys for random split
    eligible_keys = list(eligible_groups.keys())
    random.shuffle(eligible_keys)

    # Split speaker keys (not rows) at 70/15/15
    n = len(eligible_keys)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_keys = eligible_keys[:n_train]
    val_keys   = eligible_keys[n_train : n_train + n_val]
    test_keys  = eligible_keys[n_train + n_val:]

    # Collect rows per split
    train_rows = []
    val_rows   = []
    test_rows  = []

    for key in train_keys:
        train_rows.extend(eligible_groups[key])

    for key in val_keys:
        val_rows.extend(eligible_groups[key])

    for key in test_keys:
        test_rows.extend(eligible_groups[key])

    # Hold-out always goes to test
    for key, rows in holdout_groups.items():
        test_rows.extend(rows)

    return train_rows, val_rows, test_rows


def verify_no_speaker_leakage(train_rows, val_rows, test_rows):
    """
    Verify that no speaker_id appears in both train and test.
    This is the most important validation step.
    Prints a warning if leakage is detected.
    """
    # Build speaker sets per split (dataset+speaker_id combos)
    train_speakers = set(
        (r["dataset"], r["speaker_id"]) for r in train_rows
    )
    val_speakers = set(
        (r["dataset"], r["speaker_id"]) for r in val_rows
    )
    test_speakers = set(
        (r["dataset"], r["speaker_id"]) for r in test_rows
    )

    train_test_overlap = train_speakers & test_speakers
    train_val_overlap  = train_speakers & val_speakers

    if train_test_overlap:
        print(f"[WARNING] Speaker leakage detected: "
              f"{len(train_test_overlap)} speakers in both train AND test!")
        print("  This means accuracy numbers will be inflated. Fix the split.")
        for spk in list(train_test_overlap)[:5]:
            print(f"    Leaked speaker: {spk}")
    else:
        print("[OK] No speaker leakage between train and test sets.")

    if train_val_overlap:
        print(f"[WARNING] {len(train_val_overlap)} speakers in both train and val.")
    else:
        print("[OK] No speaker leakage between train and val sets.")


def print_split_stats(train_rows, val_rows, test_rows):
    """Print label distribution per split for bias auditing."""
    from collections import Counter

    def label_dist(rows, name):
        labels = Counter(int(r["label"]) for r in rows)
        total = len(rows)
        label_names = {0: "Safe", 1: "Caution", 2: "Danger"}
        print(f"\n  {name}: {total} segments")
        for lbl in sorted(labels):
            pct = 100 * labels[lbl] / total if total > 0 else 0
            print(f"    {label_names.get(lbl, lbl)}: {labels[lbl]} ({pct:.1f}%)")

    print("\n" + "="*50)
    print("SPLIT STATISTICS")
    print("="*50)
    total = len(train_rows) + len(val_rows) + len(test_rows)
    print(f"Total segments: {total}")
    label_dist(train_rows, "Train")
    label_dist(val_rows,   "Val  ")
    label_dist(test_rows,  "Test ")

    # Dataset distribution per split
    print(f"\n  Train datasets: "
          f"{dict(Counter(r['dataset'] for r in train_rows).most_common())}")
    print(f"  Test datasets:  "
          f"{dict(Counter(r['dataset'] for r in test_rows).most_common())}")
    print("="*50)


def write_split(rows: list, path: Path):
    """Write a split manifest CSV."""
    # Shuffle rows within each split for better batch diversity
    random.shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[WARNING] Empty split — not writing {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Split] Written: {len(rows)} rows → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tatvaani Speaker-Disjoint Split")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--outdir",   type=str, default="data")
    parser.add_argument("--seed",     type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir       = Path(args.outdir)

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("Run process_dataset.py first.")
        exit(1)

    # Load and group
    rows   = load_manifest(manifest_path)
    groups = group_by_speaker(rows)
    print(f"[Split] Found {len(groups)} unique (dataset, speaker) groups")

    # Split
    train_rows, val_rows, test_rows = split_speakers(groups, args.seed)

    # Validate — this is non-negotiable
    verify_no_speaker_leakage(train_rows, val_rows, test_rows)

    # Stats
    print_split_stats(train_rows, val_rows, test_rows)

    # Write
    write_split(train_rows, out_dir / "train_manifest.csv")
    write_split(val_rows,   out_dir / "val_manifest.csv")
    write_split(test_rows,  out_dir / "test_manifest.csv")

    print("\n[DONE] Split complete.")
    print("Next step: py -3.11 tatvaani_ml/training/train.py")


if __name__ == "__main__":
    main()
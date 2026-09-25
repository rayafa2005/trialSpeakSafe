"""
create_balanced_subset.py
=========================
Creates a balanced train and val manifest subset (e.g. 10,000 Real + 10,000 Fake)
from data/train_manifest.csv and data/val_manifest.csv for fast, robust training.

Usage:
  python create_balanced_subset.py --n_per_class 10000
"""

import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create balanced subset manifest")
    parser.add_argument("--train_in", default="data/train_manifest.csv")
    parser.add_argument("--val_in", default="data/val_manifest.csv")
    parser.add_argument("--train_out", default="data/train_manifest_20k.csv")
    parser.add_argument("--val_out", default="data/val_manifest_4k.csv")
    parser.add_argument("--n_per_class", type=int, default=10000, help="Number of samples per class for training")
    parser.add_argument("--n_val_per_class", type=int, default=2000, help="Number of samples per class for validation")
    args = parser.parse_args()

    train_path = Path(args.train_in)
    if not train_path.exists():
        print(f"Error: {train_path} not found.")
        return

    df_train = pd.read_csv(train_path)
    # Map any fake label > 0 to 2 (or 1 depending on format)
    df_train["bin_label"] = df_train["label"].apply(lambda l: 1 if int(l) > 0 else 0)

    real_train = df_train[df_train["bin_label"] == 0]
    fake_train = df_train[df_train["bin_label"] == 1]

    n_real = min(len(real_train), args.n_per_class)
    n_fake = min(len(fake_train), args.n_per_class)

    balanced_train = pd.concat([
        real_train.sample(n=n_real, random_state=42),
        fake_train.sample(n=n_fake, random_state=42)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    balanced_train.to_csv(args.train_out, index=False)
    print(f"[SUCCESS] Balanced train manifest saved: {args.train_out} ({len(balanced_train)} rows: {n_real} Real, {n_fake} Fake)")

    # Validation
    val_path = Path(args.val_in)
    if val_path.exists():
        df_val = pd.read_csv(val_path)
        df_val["bin_label"] = df_val["label"].apply(lambda l: 1 if int(l) > 0 else 0)
        real_val = df_val[df_val["bin_label"] == 0]
        fake_val = df_val[df_val["bin_label"] == 1]

        n_v_real = min(len(real_val), args.n_val_per_class)
        n_v_fake = min(len(fake_val), args.n_val_per_class)

        balanced_val = pd.concat([
            real_val.sample(n=n_v_real, random_state=42),
            fake_val.sample(n=n_v_fake, random_state=42)
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        balanced_val.to_csv(args.val_out, index=False)
        print(f"[SUCCESS] Balanced val manifest saved: {args.val_out} ({len(balanced_val)} rows: {n_v_real} Real, {n_v_fake} Fake)")

if __name__ == "__main__":
    main()

"""
=============================================================================
Tatvaani — Final Fix (patch3.py)
=============================================================================
Run: py -3.11 patch3.py

Does exactly two things:
  1. Fixes tatvanet.py line 61 to: def __init__(self, channels=64, reduction=8, num_classes=3):
     (all defaults, correct order — no SyntaxError possible)
  2. Fixes train.py to call TatvaNet() with no args (uses defaults)
  3. Fixes the divide-by-zero in FocalLoss alpha when a class is missing
     from the debug subset
=============================================================================
"""

import sys
import ast
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent


# ── 1. Fix tatvanet.py ────────────────────────────────────────────────────
tatvanet_path = ML_ROOT / "model" / "tatvanet.py"
text  = tatvanet_path.read_text(encoding="utf-8")
lines = text.splitlines()

# Find the __init__ line (whichever form it's currently in)
fixed = False
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("def __init__(self") and "channels" in stripped:
        # Replace entirely with a clean, unambiguous signature
        indent = len(line) - len(line.lstrip())
        lines[i] = " " * indent + "def __init__(self, channels: int = 64, reduction: int = 8, num_classes: int = 3):"
        print(f"[Fix 1] Replaced line {i+1}:")
        print(f"        OLD: {stripped}")
        print(f"        NEW: {lines[i].strip()}")
        fixed = True
        break

if not fixed:
    print("[Fix 1] WARNING: Could not find __init__ with 'channels' in tatvanet.py")
    print("        Open model/tatvanet.py and manually set line 61 to:")
    print("        def __init__(self, channels: int = 64, reduction: int = 8, num_classes: int = 3):")
else:
    new_text = "\n".join(lines)
    try:
        ast.parse(new_text)
        tatvanet_path.write_text(new_text, encoding="utf-8")
        print("[Fix 1] ✅ tatvanet.py saved — syntax valid")
    except SyntaxError as e:
        print(f"[Fix 1] ❌ Still a syntax error after fix: {e}")
        print("        Please edit model/tatvanet.py line 61 manually.")
        sys.exit(1)


# ── 2. Fix train.py ───────────────────────────────────────────────────────
train_path = ML_ROOT / "training" / "train.py"
train_text = train_path.read_text(encoding="utf-8")

# Remove any num_classes kwarg from TatvaNet() call
import re
# Match: TatvaNet(anything with num_classes=...).to(device)
# Replace with: TatvaNet().to(device)
new_train = re.sub(
    r'TatvaNet\([^)]*\)\.to\(device\)',
    'TatvaNet().to(device)',
    train_text
)
if new_train != train_text:
    train_path.write_text(new_train, encoding="utf-8")
    print("[Fix 2] ✅ train.py — TatvaNet() call cleaned (no kwargs)")
else:
    # Check if it's already clean
    if "TatvaNet().to(device)" in train_text:
        print("[Fix 2] ✅ train.py already calls TatvaNet() with no args")
    else:
        print("[Fix 2] ⚠️  Could not find TatvaNet() call in train.py")
        print("         Search for 'TatvaNet(' in training/train.py and change to TatvaNet()")


# ── 3. Fix divide-by-zero in train.py alpha computation ──────────────────
# The debug subset had 0 samples of class 1 (Caution), making counts_arr[1]=0
# and inv_freq[1]=inf. Fix: use max(count, 1) so zero-count classes get
# weight 1.0 instead of inf.
train_text2 = train_path.read_text(encoding="utf-8")

old_counts = "    counts_arr  = np.array([class_counts.get(c, 1) for c in range(config[\"num_classes\"])], dtype=float)"
new_counts  = "    counts_arr  = np.array([max(class_counts.get(c, 0), 1) for c in range(config[\"num_classes\"])], dtype=float)  # max(...,1) prevents div-by-zero when a class is absent from subset"

if old_counts in train_text2:
    train_text2 = train_text2.replace(old_counts, new_counts)
    train_path.write_text(train_text2, encoding="utf-8")
    print("[Fix 3] ✅ train.py — divide-by-zero in alpha weights fixed")
elif "max(class_counts.get(c, 0), 1)" in train_text2:
    print("[Fix 3] ✅ divide-by-zero fix already applied")
else:
    # Fallback: find and fix by pattern
    train_text2 = re.sub(
        r'class_counts\.get\(c,\s*1\)',
        'max(class_counts.get(c, 0), 1)',
        train_text2
    )
    train_path.write_text(train_text2, encoding="utf-8")
    print("[Fix 3] ✅ train.py — divide-by-zero patched via regex")


# ── Final verification ────────────────────────────────────────────────────
print()
print("=" * 55)
print("Verifying both files parse cleanly...")
for path in [tatvanet_path, train_path]:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        print(f"  ✅ {path.name}")
    except SyntaxError as e:
        print(f"  ❌ {path.name} — SyntaxError: {e}")

print()
print("=" * 55)
print("ALL DONE. Run in order:")
print()
print("  Step 1 — Verify pipeline (fast, 2 epochs on 64 samples):")
print("    py -3.11 training/train.py --debug")
print()
print("  Step 2 — If debug prints 2 epoch summaries, start real training:")
print("    py -3.11 training/train.py")
print()
print("  Step 3 — In a SEPARATE terminal while training runs:")
print("    py -3.11 extract_indicsynth_sample.py")
print("=" * 55)
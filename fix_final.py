"""
Tatvaani — fix_final.py
=======================
1. Fixes diagnose_forward.py — output is a dict, not a tensor
2. Reports class balance in your manifests (all-Danger is a training problem)
3. Shows next steps

Run: py -3.11 fix_final.py
"""
import re
import csv
from pathlib import Path
from collections import Counter

ML_ROOT = Path(__file__).resolve().parent

# ── Fix 1: diagnose_forward.py ─────────────────────────────────────────────
diag_path = ML_ROOT / "diagnose_forward.py"
text = diag_path.read_text(encoding="utf-8")

# Replace any line that does output.shape with output['logits'].shape
# Common patterns:
#   print(f"... {output.shape}")
#   assert output.shape == ...
fixes = [
    (r'output\.shape', "output['logits'].shape"),
    (r'\boutput\b(?![\[\.])', "output['logits']"),   # bare `output` used as tensor
]

new_text = text
new_text = re.sub(r'output\.shape', "output['logits'].shape", new_text)

# Also fix any line that passes output directly to shape checks
# e.g.  assert output.shape[1] == 3
new_text = re.sub(r'output\.shape\[', "output['logits'].shape[", new_text)

if new_text != text:
    diag_path.write_text(new_text, encoding="utf-8")
    print("✅ diagnose_forward.py fixed (output.shape → output['logits'].shape)")
else:
    print("⚠️  diagnose_forward.py — no automatic fix applied")
    print("   Open it, find line 68, change:  output.shape")
    print("   to:                              output['logits'].shape")

# ── Fix 2: Class balance report ────────────────────────────────────────────
print()
print("=" * 55)
print("CLASS BALANCE REPORT")
print("=" * 55)

label_map = {"0": "SAFE", "1": "CAUTION", "2": "DANGER"}

for fname in ["train_manifest.csv", "val_manifest.csv", "test_manifest.csv"]:
    fpath = ML_ROOT / "data" / fname
    if not fpath.exists():
        print(f"  {fname}: not found")
        continue
    with open(fpath, newline="") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r.get("label", "?") for r in rows)
    total = len(rows)
    print(f"\n  {fname} ({total} rows):")
    for lbl in sorted(counts):
        name = label_map.get(str(lbl), f"class_{lbl}")
        pct = 100 * counts[lbl] / total
        bar = "█" * int(pct / 2)
        print(f"    {name:8s} (label={lbl}): {counts[lbl]:5d}  {pct:5.1f}%  {bar}")

print()
print("=" * 55)
print("⚠️  If all labels are DANGER (2), you need bonafide (SAFE=0) data.")
print("   Your ASVspoof + WaveFake downloads contain bonafide clips.")
print("   Run process_dataset.py on them first, THEN split_dataset.py.")
print()
print("   ASVspoof bonafide clips are in:")
print("     LA/ASVspoof2019_LA_train/flac/  (label: bonafide in .txt)")
print("   WaveFake real clips are in:")
print("     WaveFake/ljspeech/  (the real LJSpeech wavs)")
print("=" * 55)

# ── Fix 3: Verify the forward pass will now work ────────────────────────────
print()
print("Run these now:")
print()
print("  py -3.11 diagnose_forward.py")
print()
print("  If it shows all green → start training:")
print("  py -3.11 training/train.py")
print()
print("  If training crashes with class imbalance / NaN loss,")
print("  process ASVspoof first:")
print("  py -3.11 process_dataset.py")
print("  py -3.11 split_dataset.py")
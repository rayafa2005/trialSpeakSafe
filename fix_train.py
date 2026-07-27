"""
Quick fix for train.py — run once then start training.
py -3.11 fix_train.py
"""
from pathlib import Path

path = Path("training/train.py")
text = path.read_text(encoding="utf-8")

# Fix 1: remove all emojis that crash Windows terminal
replacements = [
    ("\u2705", "[OK]"),
    ("\u274c", "[FAIL]"),
    ("\u26a0", "[WARN]"),
    ("\u2714", "[OK]"),
    ("\u2716", "[FAIL]"),
    ("\u26a0\ufe0f", "[WARN]"),
]
for old, new in replacements:
    text = text.replace(old, new)

# Fix 2: divide by zero when class 1 (Caution) has 0 samples
# class_counts.get(c, 1) returns 1 for missing classes which is wrong
# max(..., 1) ensures minimum count of 1 to prevent 1/0 = inf
old = 'counts_arr  = np.array([class_counts.get(c, 1) for c in range(config["num_classes"])], dtype=float)'
new = 'counts_arr  = np.array([max(class_counts.get(c, 0), 1) for c in range(config["num_classes"])], dtype=float)'
if old in text:
    text = text.replace(old, new)
    print("[OK] Fixed divide-by-zero in alpha weights")
elif "max(class_counts.get(c, 0), 1)" in text:
    print("[OK] Divide-by-zero already fixed")
else:
    print("[WARN] Could not find alpha weights line - check manually")

path.write_text(text, encoding="utf-8")
print("[OK] train.py saved - all emojis removed")
print("")
print("Now run:")
print("  py -3.11 training\\train.py")
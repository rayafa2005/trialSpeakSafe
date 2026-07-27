"""
Tatvaani — fix_focal_loss.py
=============================
FIX: focal_loss.py forward() receives the raw model output dict
     but calls F.log_softmax(logits) on it — dict has no log_softmax.

TWO places to fix:
  1. focal_loss.py forward() — extract logits from dict if needed
  2. diagnose_forward.py line ~93 — passes `output` (dict) to loss_fn
     instead of output['logits']

Run: py -3.11 fix_focal_loss.py
"""
import re
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# ── Fix 1: focal_loss.py — accept dict OR tensor ──────────────────────────
fl_path = ML_ROOT / "training" / "focal_loss.py"
text = fl_path.read_text(encoding="utf-8")

# Find the forward() method and insert a dict-unwrap guard at the top
guard = "        # Accept raw logits tensor OR TatvaNet output dict\n        if isinstance(logits, dict): logits = logits['logits']\n"

if "isinstance(logits, dict)" in text:
    print("✅ focal_loss.py already has dict guard")
else:
    # Insert after the def forward(self, logits line
    new_text = re.sub(
        r'(def forward\(self,\s*logits[^\)]*\):[ \t]*\n)',
        r'\1' + guard,
        text
    )
    if new_text == text:
        # Try alternate pattern — forward with different arg name
        new_text = re.sub(
            r'(def forward\(self,[^\)]+\):[ \t]*\n)',
            r'\1' + guard,
            text, count=1
        )
    if new_text != text:
        fl_path.write_text(new_text, encoding="utf-8")
        print("✅ focal_loss.py patched — will unwrap dict automatically")
    else:
        print("⚠️  Could not auto-patch focal_loss.py")
        print("   Open training/focal_loss.py, find def forward(self, logits...")
        print("   Add as FIRST line of the method body:")
        print("   if isinstance(logits, dict): logits = logits['logits']")

# ── Fix 2: diagnose_forward.py — pass output['logits'] to loss_fn ─────────
diag_path = ML_ROOT / "diagnose_forward.py"
text2 = diag_path.read_text(encoding="utf-8")

# Fix:  loss = loss_fn(output, labels)  →  loss = loss_fn(output['logits'], labels)
new_text2 = re.sub(
    r'loss\s*=\s*loss_fn\(\s*output\s*,',
    "loss = loss_fn(output['logits'],",
    text2
)
if new_text2 != text2:
    diag_path.write_text(new_text2, encoding="utf-8")
    print("✅ diagnose_forward.py fixed — passes output['logits'] to loss_fn")
else:
    if "output['logits']" in text2:
        print("✅ diagnose_forward.py already correct")
    else:
        print("⚠️  Could not auto-patch diagnose_forward.py line ~93")
        print("   Change:  loss = loss_fn(output, labels)")
        print("   To:      loss = loss_fn(output['logits'], labels)")

print()
print("=" * 55)
print("Run: py -3.11 diagnose_forward.py")
print("Expect all steps green → then: py -3.11 training/train.py")
print("=" * 55)
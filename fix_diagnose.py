"""
Tatvaani — fix_diagnose.py
Fixes the last broken line in diagnose_forward.py:
  output[0]  →  output['probs'][0]
Run: py -3.11 fix_diagnose.py
"""
import re
from pathlib import Path

path = Path(__file__).resolve().parent / "diagnose_forward.py"
text = path.read_text(encoding="utf-8")

# Fix output[0] → output['probs'][0]
new_text = re.sub(r'\boutput\[0\]', "output['probs'][0]", text)
# Fix any other bare output[N] patterns
new_text = re.sub(r'\boutput\[(\d+)\]', r"output['probs'][\1]", new_text)

if new_text != text:
    path.write_text(new_text, encoding="utf-8")
    print("✅ diagnose_forward.py fixed")
else:
    print("Pattern not found — open diagnose_forward.py line 69 manually")
    print("Change:  output[0]")
    print("To:      output['probs'][0]")

print()
print("Run: py -3.11 diagnose_forward.py")
print()
print("The forward pass IS working (shape [4,3] confirmed).")
print("Once diagnose goes all-green, start training:")
print("  py -3.11 training/train.py")
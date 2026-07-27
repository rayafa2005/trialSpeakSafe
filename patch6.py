"""
=============================================================================
Tatvaani — patch6.py
=============================================================================
PROBLEM:
  spectrogram.py → STFT submodule forward() → line ~109:
      out = F.interpolate(out, size=(TARGET_HEIGHT, out.shape[-1]), ...)

  torch.stft() returns [B, freq, T] — a 3D tensor.
  F.interpolate for 2D spatial resize needs 4D [B, C, H, W].
  Fix: unsqueeze(1) before interpolate, squeeze(1) after — for every
  F.interpolate call in the file that targets a 2D (H, W) size.

THIS PATCH:
  - Reads features/spectrogram.py
  - Finds every line containing F.interpolate( with a 2-element size tuple
  - Wraps each with unsqueeze/squeeze guards
  - Validates syntax
  - Saves in place

Run:  py -3.11 patch6.py
Then: py -3.11 diagnose_forward.py
=============================================================================
"""

import ast
import re
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# ── Find spectrogram.py ───────────────────────────────────────────────────────
candidates = [
    ML_ROOT / "features" / "spectrogram.py",
    ML_ROOT / "model" / ".." / "features" / "spectrogram.py",
]
spec_path = None
for c in candidates:
    if c.exists():
        spec_path = c.resolve()
        break

if spec_path is None:
    print("❌ Cannot find spectrogram.py")
    print("   Expected at: tatvaani_ml/features/spectrogram.py")
    raise SystemExit(1)

print(f"Found: {spec_path}")
original = spec_path.read_text(encoding="utf-8")

# ── Show the broken area for confirmation ─────────────────────────────────────
lines = original.splitlines()
print(f"\nTotal lines: {len(lines)}")
print("\nLines containing 'interpolate':")
for i, line in enumerate(lines, 1):
    if "interpolate" in line:
        print(f"  {i:4d}: {line}")

# ── Core fix: process line by line ───────────────────────────────────────────
# For each line that calls F.interpolate(..., size=(X, Y)...) where size has 2 elements,
# replace it with a 3-line unsqueeze → interpolate → squeeze block.

new_lines = []
i = 0
fixes_applied = 0

while i < len(lines):
    line = lines[i]

    # Detect: assignment to F.interpolate with a 2-element size tuple
    # Matches:   out = F.interpolate(out, size=(TARGET_HEIGHT, out.shape[-1]), ...)
    # Also multiline calls (collect continuation lines)
    if "F.interpolate(" in line and "size=(" in line:
        indent = len(line) - len(line.lstrip())
        ind = " " * indent

        # Collect full call (may span multiple lines if there's a trailing comma + next line)
        full_call = line.rstrip()
        j = i
        # Count parens to find end of call
        open_p = full_call.count("(") - full_call.count(")")
        while open_p > 0 and j + 1 < len(lines):
            j += 1
            full_call += " " + lines[j].strip()
            open_p += lines[j].count("(") - lines[j].count(")")

        # Parse: LHS = F.interpolate(VAR, ...)
        m = re.match(r'\s*(\w+)\s*=\s*F\.interpolate\(\s*(\w+)\s*,(.+)\)', full_call.strip())
        if m:
            lhs     = m.group(1)
            src_var = m.group(2)
            args    = m.group(3).strip()

            # Check if _needs_sq guard already there (already patched)
            if i > 0 and "_needs_sq" in lines[i - 1]:
                new_lines.append(line)
                i += 1
                continue

            # Build the replacement block
            replacement = [
                f"{ind}_needs_sq = {src_var}.dim() == 3",
                f"{ind}if _needs_sq: {src_var} = {src_var}.unsqueeze(1)",
                f"{ind}{lhs} = F.interpolate({src_var}, {args})",
                f"{ind}if _needs_sq: {lhs} = {lhs}.squeeze(1)",
            ]

            # Make sure mode and align_corners are present (required for bilinear)
            interp_line = replacement[2]
            if "mode=" not in interp_line:
                # Insert before closing paren
                interp_line = interp_line.rstrip(")")
                interp_line += ", mode='bilinear', align_corners=False)"
                replacement[2] = interp_line

            new_lines.extend(replacement)
            fixes_applied += 1
            print(f"\n✅ Fixed interpolate at original line {i+1}:")
            for r in replacement:
                print(f"   {r}")

            # Skip all lines consumed by this call
            i = j + 1
            continue

    new_lines.append(line)
    i += 1

print(f"\nTotal fixes applied: {fixes_applied}")

if fixes_applied == 0:
    if "_needs_sq" in original:
        print("⚠️  No changes made — file appears already patched (_needs_sq found).")
        print("   If the error persists, the issue is NOT in F.interpolate.")
        print("   Run: py -3.11 diagnose_spectrogram.py  (created below)")
    else:
        print("⚠️  Pattern not matched. Printing all lines with 'interpolate' for manual fix:")
        for i2, l in enumerate(lines, 1):
            if "interpolate" in l.lower():
                print(f"  line {i2}: {l}")
        print()
        print("Manual fix: for each F.interpolate line that targets (H, W), wrap as:")
        print("   _needs_sq = out.dim() == 3")
        print("   if _needs_sq: out = out.unsqueeze(1)")
        print("   out = F.interpolate(out, size=(TARGET_HEIGHT, out.shape[-1]),")
        print("                       mode='bilinear', align_corners=False)")
        print("   if _needs_sq: out = out.squeeze(1)")

# ── Validate & save ───────────────────────────────────────────────────────────
new_text = "\n".join(new_lines)

try:
    ast.parse(new_text)
    print(f"\n✅ Syntax OK")
except SyntaxError as e:
    print(f"\n❌ Syntax error after patch: {e}")
    print("   spectrogram.py NOT saved. Fix manually.")
    raise SystemExit(1)

spec_path.write_text(new_text, encoding="utf-8")
print(f"✅ Saved: {spec_path}")


# ── Write a mini diagnostic script in case we need to dig further ─────────────
diag = ML_ROOT / "diagnose_spectrogram.py"
diag.write_text('''\
"""
Quick diagnostic — prints actual shapes through spectrogram pipeline.
Run: py -3.11 diagnose_spectrogram.py
"""
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(4, 48000).to(device)
print(f"Input: {x.shape}")

from features.spectrogram import MultiStreamSpectrogram
spec = MultiStreamSpectrogram().to(device)

# Hook every forward to print shapes
hooks = []
def make_hook(name):
    def hook(mod, inp, out):
        in_s  = inp[0].shape if isinstance(inp, tuple) else "?"
        out_s = out.shape if hasattr(out, "shape") else type(out)
        print(f"  {name:40s}  in={in_s}  out={out_s}")
    return hook

for name, mod in spec.named_modules():
    hooks.append(mod.register_forward_hook(make_hook(name or "root")))

with torch.no_grad():
    out = spec(x)

for h in hooks:
    h.remove()

print(f"\\nFinal spectrogram output: {out.shape}  (expected [4, 3, 128, T])")
''', encoding="utf-8")
print(f"✅ Created: {diag.name}  (run if forward still fails after this patch)")


# ── Final instructions ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Done. Run:")
print()
print("  py -3.11 diagnose_forward.py")
print()
print("If it's still red, run:")
print("  py -3.11 diagnose_spectrogram.py")
print("and paste the output here.")
print("=" * 60)
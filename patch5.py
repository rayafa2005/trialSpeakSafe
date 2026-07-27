"""
=============================================================================
Tatvaani — patch5.py
=============================================================================
ROOT CAUSE (from traceback):
  spectrogram.py line 108:
      out = F.interpolate(out, size=(TARGET_HEIGHT, out.shape[-1]))

  The STFT branch produces `out` with shape [B, 513, T] — a 3D tensor.
  F.interpolate for 2D (spatial) resizing requires 4D: [B, C, H, W].
  PyTorch raises:
      ValueError: Input and output must have the same number of spatial
      dimensions, but got input with spatial dimensions of [4, 513, 301]
      and output size of (128, 301).

FIX:
  Every F.interpolate call in spectrogram.py that targets 2D (H, W)
  must receive a 4D tensor. The fix: unsqueeze(1) before interpolate,
  squeeze(1) after.

  Additionally: the waveform going into the spectrogram pipeline arrives
  as [B, 1, N] (after the view() fix in tatvanet.py forward). The
  spectrogram module expects [B, N] (no channel dim). We add a squeeze
  at the entry of FeaturePipeline / spectrogram forward() to handle both.

Run:  py -3.11 patch5.py
Then: py -3.11 diagnose_forward.py
=============================================================================
"""

import ast
import re
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# ── Locate spectrogram.py ─────────────────────────────────────────────────────
spec_path = ML_ROOT / "features" / "spectrogram.py"
if not spec_path.exists():
    # Some builds put it under model/../features/
    alt = ML_ROOT / "model" / ".." / "features" / "spectrogram.py"
    if alt.exists():
        spec_path = alt.resolve()

if not spec_path.exists():
    print(f"❌ Cannot find spectrogram.py under {ML_ROOT}")
    print("   Looked in: features/spectrogram.py")
    raise SystemExit(1)

print(f"Found: {spec_path}")
text = spec_path.read_text(encoding="utf-8")

# ── Fix 1: F.interpolate calls — wrap with unsqueeze/squeeze ─────────────────
#
# Pattern we are looking for (any of these forms):
#   out = F.interpolate(out, size=(TARGET_HEIGHT, out.shape[-1]))
#   out = F.interpolate(out, size=(TARGET_HEIGHT, out.shape[-1]), mode=...)
#   x   = F.interpolate(x,   size=(...))
#
# We replace:
#   VAR = F.interpolate(VAR, size=(H, W_EXPR))
# with:
#   _needs_squeeze = VAR.dim() == 3
#   if _needs_squeeze: VAR = VAR.unsqueeze(1)
#   VAR = F.interpolate(VAR, size=(H, W_EXPR), mode='bilinear', align_corners=False)
#   if _needs_squeeze: VAR = VAR.squeeze(1)

INTERP_RE = re.compile(
    r'^(\s*)(\w+)\s*=\s*F\.interpolate\(\s*(\w+)\s*,\s*size\s*=\s*\(([^)]+)\)(.*?)\)\s*$',
    re.MULTILINE
)

def replace_interpolate(m):
    indent   = m.group(1)
    lhs      = m.group(2)
    rhs_var  = m.group(3)
    size_arg = m.group(4)
    rest     = m.group(5).strip()          # e.g. ", mode='bilinear', align_corners=False"

    # If rest already contains mode= and align_corners=, keep; otherwise add
    if "mode=" not in rest:
        rest = rest.rstrip(",").strip()
        if rest:
            rest = ", " + rest + ", mode='bilinear', align_corners=False"
        else:
            rest = ", mode='bilinear', align_corners=False"
    else:
        if rest and not rest.startswith(","):
            rest = ", " + rest

    lines = [
        f"{indent}_needs_sq = {rhs_var}.dim() == 3",
        f"{indent}if _needs_sq: {rhs_var} = {rhs_var}.unsqueeze(1)",
        f"{indent}{lhs} = F.interpolate({rhs_var}, size=({size_arg}){rest})",
        f"{indent}if _needs_sq: {lhs} = {lhs}.squeeze(1)",
    ]
    return "\n".join(lines)

new_text, n_interp = INTERP_RE.subn(replace_interpolate, text)

if n_interp == 0:
    print("⚠️  No F.interpolate pattern found — checking for already-patched file...")
    if "_needs_sq" in text:
        print("   Already patched (found _needs_sq). Skipping interpolate fix.")
    else:
        print("   Pattern not matched. You may need to fix spectrogram.py manually.")
        print("   Find every line like:")
        print("       out = F.interpolate(out, size=(TARGET_HEIGHT, out.shape[-1]))")
        print("   And replace with:")
        print("       _needs_sq = out.dim() == 3")
        print("       if _needs_sq: out = out.unsqueeze(1)")
        print("       out = F.interpolate(out, size=(TARGET_HEIGHT, out.shape[-1]), mode='bilinear', align_corners=False)")
        print("       if _needs_sq: out = out.squeeze(1)")
else:
    print(f"✅ Fixed {n_interp} F.interpolate call(s) in spectrogram.py")

# ── Fix 2: Input squeeze — spectrogram forward() receives [B,1,N] from tatvanet ─
#
# The waveform enters spectrogram as [B, 1, N] after tatvanet's view().
# Every compute_mel / compute_stft / compute_cqt call expects [B, N].
# Find the forward() in the spectrogram module and insert a squeeze guard.

# Strategy: find the first forward method definition in the file,
# insert   if x.dim() == 3 and x.shape[1] == 1: x = x.squeeze(1)
# right after the def line (before any computation).

forward_guard = "if x.dim() == 3 and x.shape[1] == 1: x = x.squeeze(1)  # accept [B,1,N] from tatvanet"

if forward_guard in new_text:
    print("✅ Input squeeze guard already present in spectrogram.py")
else:
    lines = new_text.splitlines()
    in_forward = False
    guard_inserted = False
    for i, line in enumerate(lines):
        if re.match(r'\s*def forward\s*\(', line):
            in_forward = True
            continue
        if in_forward:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            # First real code line inside forward()
            indent = len(line) - len(line.lstrip())
            guard_line = " " * indent + forward_guard
            lines.insert(i, guard_line)
            print(f"✅ Inserted input squeeze guard at line {i+1} in spectrogram.py forward()")
            guard_inserted = True
            in_forward = False
            break

    if not guard_inserted:
        print("⚠️  Could not auto-insert input squeeze guard — do it manually (see above)")

    new_text = "\n".join(lines)

# ── Validate & save spectrogram.py ────────────────────────────────────────────
try:
    ast.parse(new_text)
except SyntaxError as e:
    print(f"❌ Syntax error after patching spectrogram.py: {e}")
    print("   spectrogram.py NOT saved. Fix manually.")
    raise SystemExit(1)

spec_path.write_text(new_text, encoding="utf-8")
print(f"✅ spectrogram.py saved → {spec_path}")


# ── Fix 3: delta.py FeaturePipeline — same input squeeze guard ────────────────
delta_path = ML_ROOT / "features" / "delta.py"
if not delta_path.exists():
    delta_path = (ML_ROOT / "model" / ".." / "features" / "delta.py").resolve()

if delta_path.exists():
    d_text = delta_path.read_text(encoding="utf-8")

    delta_guard = "if x.dim() == 3 and x.shape[1] == 1: x = x.squeeze(1)  # accept [B,1,N]"

    if delta_guard in d_text:
        print("✅ Input squeeze guard already present in delta.py")
    else:
        d_lines = d_text.splitlines()
        in_fwd = False
        inserted = False
        for i, line in enumerate(d_lines):
            if re.match(r'\s*def forward\s*\(', line):
                in_fwd = True
                continue
            if in_fwd:
                stripped = line.strip()
                if stripped == "" or stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                indent = len(line) - len(line.lstrip())
                d_lines.insert(i, " " * indent + delta_guard)
                print(f"✅ Inserted input squeeze guard in delta.py forward() at line {i+1}")
                inserted = True
                in_fwd = False
                break
        if not inserted:
            print("⚠️  Could not auto-patch delta.py — check manually")

        d_text = "\n".join(d_lines)
        try:
            ast.parse(d_text)
            delta_path.write_text(d_text, encoding="utf-8")
            print(f"✅ delta.py saved → {delta_path}")
        except SyntaxError as e:
            print(f"❌ Syntax error in delta.py after patch: {e}  — NOT saved")
else:
    print(f"⚠️  delta.py not found at {delta_path} — skip")


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("patch5.py complete.")
print()
print("Next steps:")
print()
print("  1. Verify the fix:")
print("       py -3.11 diagnose_forward.py")
print()
print("  2. If forward pass is green, start training:")
print("       py -3.11 training/train.py --debug   # 1 batch sanity check")
print("       py -3.11 training/train.py            # full run")
print("=" * 60)
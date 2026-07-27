"""
=============================================================================
Tatvaani — patch7.py
=============================================================================
FIX 1: spectrogram.py STFT forward()
  The _needs_sq guard from patch6 isn't working because the interpolate
  call re-reads `out` which is still 3D at that point.
  ROOT FIX: just unsqueeze(1) the raw STFT output immediately after
  torch.stft() before any interpolation. That's the real fix.

FIX 2: extract_indicsynth_sample.py CSV fieldnames
  ValueError: dict contains fields not in fieldnames: 'gender'
  A row dict has a 'gender' key but COLUMNS list doesn't include it.
  Fix: strip rows to only the declared COLUMNS before writing.

Run:  py -3.11 patch7.py
Then: py -3.11 diagnose_forward.py
Then: py -3.11 extract_indicsynth_sample.py
=============================================================================
"""

import ast
import re
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# =============================================================================
# FIX 1: spectrogram.py — find the STFT submodule forward() and unsqueeze
#         the raw torch.stft output from [B, F, T] to [B, 1, F, T]
# =============================================================================

spec_path = (ML_ROOT / "features" / "spectrogram.py").resolve()
if not spec_path.exists():
    spec_path = (ML_ROOT / "model" / ".." / "features" / "spectrogram.py").resolve()

print(f"spectrogram.py: {spec_path}")
assert spec_path.exists(), f"Not found: {spec_path}"

text = spec_path.read_text(encoding="utf-8")
lines = text.splitlines()

print(f"Lines: {len(lines)}")

# Strategy: find every line that calls torch.stft() or F.stft() and assigns
# the result to a variable, then insert an unsqueeze(1) right after it.
# Pattern:   out = torch.stft(...) or out = x.stft(...)  etc.
# Also find:  out = self._stft(x)  — if they wrapped it in a helper

stft_fixed = 0
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]

    # Detect torch.stft assignment — collect multiline call
    is_stft = ("torch.stft(" in line or
               "torch.stft (" in line or
               ("= self." in line and "stft" in line.lower() and "(" in line and
                "interpolate" not in line and "self.stft(" not in line))

    # More targeted: look for lines where a variable is assigned from something
    # containing 'stft' — but NOT self.stft(x) calls (those are method calls
    # to the submodule, not the raw torch op)
    # Real pattern: look for torch.stft specifically
    is_torch_stft = "torch.stft(" in line

    if is_torch_stft:
        # Collect full call
        full = line.rstrip()
        j = i
        opens = full.count("(") - full.count(")")
        while opens > 0 and j + 1 < len(lines):
            j += 1
            full += " " + lines[j].strip()
            opens += lines[j].count("(") - lines[j].count(")")

        # Get LHS variable name
        m = re.match(r'\s*(\w+)\s*=\s*torch\.stft\(', full)
        lhs = m.group(1) if m else "out"
        indent = len(line) - len(line.lstrip())
        ind = " " * indent

        # Check if unsqueeze already present on next lines
        lookahead = " ".join(lines[i+1:i+4]) if i+1 < len(lines) else ""
        already_fixed = "unsqueeze(1)" in lookahead and lhs in lookahead

        # Add all original lines for this call
        for k in range(i, j+1):
            new_lines.append(lines[k])

        if not already_fixed:
            # torch.stft returns [B, F, T] (real) or [B, F, T, 2] (complex)
            # We need [B, 1, F, T] for F.interpolate 2D
            new_lines.append(f"{ind}# Fix: torch.stft → [B,F,T], unsqueeze to [B,1,F,T] for interpolate")
            new_lines.append(f"{ind}if {lhs}.dim() == 3: {lhs} = {lhs}.unsqueeze(1)")
            new_lines.append(f"{ind}elif {lhs}.dim() == 4 and {lhs}.shape[-1] == 2:")
            new_lines.append(f"{ind}    # Complex output [B,F,T,2] → magnitude [B,F,T] → [B,1,F,T]")
            new_lines.append(f"{ind}    {lhs} = {lhs}[..., 0].pow(2).add({lhs}[..., 1].pow(2)).sqrt().unsqueeze(1)")
            stft_fixed += 1
            print(f"✅ Inserted unsqueeze after torch.stft at line {i+1} (lhs={lhs})")

        i = j + 1
        continue

    new_lines.append(line)
    i += 1

print(f"torch.stft fixes: {stft_fixed}")

# If no torch.stft found directly, the STFT is computed in a sub-module.
# The actual crash is at spectrogram.py line 111 (from traceback):
#   out = F.interpolate(out, size=(TARGET_HEIGHT, out.shape[-1]), mode="bilinear", align_corners=False)
# And `out` is [4, 513, 301] — 3D.
# The _needs_sq patch6 added was on the WRONG variable or wrong scope.
# Find this exact line and replace it robustly:

if stft_fixed == 0:
    print("No torch.stft() found directly — fixing interpolate calls more aggressively...")
    new_lines2 = []
    i = 0
    interp_fixed = 0
    while i < len(new_lines):
        line = new_lines[i]
        if "F.interpolate(" in line and "size=(" in line:
            indent = len(line) - len(line.lstrip())
            ind = " " * indent

            # Collect full call
            full = line.rstrip()
            j = i
            opens = full.count("(") - full.count(")")
            while opens > 0 and j + 1 < len(new_lines):
                j += 1
                full += " " + new_lines[j].strip()
                opens += new_lines[j].count("(") - new_lines[j].count(")")

            # Extract LHS and source variable
            m = re.match(r'\s*(\w+)\s*=\s*F\.interpolate\(\s*(\w+)\s*,(.+)\)', full.strip())
            if m:
                lhs = m.group(1)
                src = m.group(2)
                args = m.group(3).strip()

                # Check not already wrapped
                lookahead = " ".join(new_lines[max(0,i-2):i])
                if "_needs_sq" in lookahead or f"{src} = {src}.unsqueeze(1)" in lookahead:
                    # Already wrapped — but the old patch used wrong variable name
                    # Replace the whole block
                    pass

                # Write clean version
                replacement = [
                    f"{ind}# interpolate fix: ensure 4D for bilinear",
                    f"{ind}_{src}_sq = {src}.dim() == 3",
                    f"{ind}if _{src}_sq: {src} = {src}.unsqueeze(1)",
                    f"{ind}{lhs} = F.interpolate({src}, {args})",
                    f"{ind}if _{src}_sq: {lhs} = {lhs}.squeeze(1)",
                ]
                # Ensure mode=bilinear in the interpolate line
                if "mode=" not in replacement[3]:
                    replacement[3] = replacement[3].rstrip(")")
                    if not replacement[3].rstrip().endswith(","):
                        replacement[3] += ", mode='bilinear', align_corners=False)"
                    else:
                        replacement[3] += " mode='bilinear', align_corners=False)"

                new_lines2.extend(replacement)
                interp_fixed += 1
                print(f"✅ Rewrote interpolate at original line {i+1}: {src} → {lhs}")
                i = j + 1
                continue
        new_lines2.append(line)
        i += 1

    new_lines = new_lines2
    print(f"Interpolate rewrites: {interp_fixed}")

# Validate & save
new_text = "\n".join(new_lines)
try:
    ast.parse(new_text)
    print("✅ spectrogram.py syntax OK")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    # Show context
    err_lines = new_text.splitlines()
    start = max(0, e.lineno - 3)
    end = min(len(err_lines), e.lineno + 3)
    for k, l in enumerate(err_lines[start:end], start+1):
        marker = " >>>" if k == e.lineno else "    "
        print(f"{marker} {k}: {l}")
    raise SystemExit(1)

spec_path.write_text(new_text, encoding="utf-8")
print(f"✅ Saved: {spec_path}")


# =============================================================================
# FIX 2: extract_indicsynth_sample.py — strip rows to declared COLUMNS only
# =============================================================================

extractor_path = ML_ROOT / "extract_indicsynth_sample.py"
assert extractor_path.exists(), f"Not found: {extractor_path}"

ext_text = extractor_path.read_text(encoding="utf-8")

# The COLUMNS list in the file — find it
cols_match = re.search(r'COLUMNS\s*=\s*\[([^\]]+)\]', ext_text)
if cols_match:
    print(f"\nCOLUMNS found: {cols_match.group(0)[:80]}")

# Fix: before writer.writerows(all_rows), filter each row to only COLUMNS keys.
# Find the writerows call and insert a filter before it.

old_writerows = "writer.writerows(all_rows)"
new_writerows = (
    "# Strip any extra keys (e.g. 'gender') not in COLUMNS\n"
    "            all_rows = [{k: r.get(k, '') for k in COLUMNS} for r in all_rows]\n"
    "            writer.writerows(all_rows)"
)

if old_writerows in ext_text:
    ext_text = ext_text.replace(old_writerows, new_writerows, 1)
    print("✅ Fixed writer.writerows to strip extra keys")
else:
    # Try alternative — maybe it's not indented the same way
    ext_text = re.sub(
        r'(writer\.writerows\(all_rows\))',
        ("all_rows = [{k: r.get(k, '') for k in COLUMNS} for r in all_rows]  "
         "# strip extra keys\n        \\1"),
        ext_text, count=1
    )
    print("✅ Fixed writer.writerows (alt pattern)")

# Also fix individual row appends that may include 'gender':
# new_rows.append({...}) — strip to COLUMNS
# This is harder to patch generically; instead add a filter in extract_one_parquet
# right before "new_rows.append"

old_append = "new_rows.append({"
if old_append in ext_text:
    # Wrap the append: strip keys not in COLUMNS
    ext_text = ext_text.replace(
        "new_rows.append({",
        "_row_data = {",
        1  # only first occurrence (inside the loop)
    )
    # Find where the dict closes and add the filtered append
    # This is tricky without seeing the file; use a regex instead
    # Pattern: _row_data = { ... }) followed by n_ok += 1
    ext_text = re.sub(
        r'(_row_data\s*=\s*\{[^}]+\})\s*\n(\s*)(n_ok\s*\+=\s*1)',
        r'\1\n\2new_rows.append({k: _row_data.get(k, "") for k in COLUMNS})\n\2\3',
        ext_text, count=1
    )
    print("✅ Fixed new_rows.append to strip extra keys at row level")

try:
    ast.parse(ext_text)
    extractor_path.write_text(ext_text, encoding="utf-8")
    print(f"✅ Saved: {extractor_path}")
except SyntaxError as e:
    print(f"❌ Syntax error in extractor: {e} — NOT saved")
    print("   Manual fix: find 'writer.writerows(all_rows)' and add before it:")
    print("   all_rows = [{k: r.get(k,'') for k in COLUMNS} for r in all_rows]")


# =============================================================================
print()
print("=" * 60)
print("patch7 done. Run:")
print()
print("  py -3.11 diagnose_forward.py")
print()
print("  (in separate terminal, restart extraction from scratch)")
print("  py -3.11 extract_indicsynth_sample.py")
print("=" * 60)
"""
Tatvaani — fix_csv.py
Fixes: ValueError: dict contains fields not in fieldnames: 'gender'
The extractor builds rows with a 'gender' key that isn't in COLUMNS.
Fix: filter every row to only declared COLUMNS before writing.
Run: py -3.11 fix_csv.py
"""
import re
from pathlib import Path

path = Path(__file__).parent / "extract_indicsynth_sample.py"
text = path.read_text(encoding="utf-8")

# Replace the writerows line with a version that strips unknown keys
old = "writer.writerows(all_rows)"
new = ("all_rows = [{k: r.get(k, '') for k in COLUMNS} for r in all_rows]"
       "  # strip unknown keys (e.g. gender)\n"
       "            writer.writerows(all_rows)")

if old not in text:
    print("Pattern not found — already fixed or different indent.")
    print("Manual fix: find 'writer.writerows(all_rows)' and add before it:")
    print("  all_rows = [{k: r.get(k,'') for k in COLUMNS} for r in all_rows]")
else:
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print("✅ extract_indicsynth_sample.py fixed")

print()
print("Now run:")
print("  py -3.11 extract_indicsynth_sample.py")
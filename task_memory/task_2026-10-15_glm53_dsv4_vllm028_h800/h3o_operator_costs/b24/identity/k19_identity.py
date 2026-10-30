"""Byte-compare the metrics and collectives of two A10 run roots.

    python k19_identity.py <root a> <root b>
"""
import filecmp
import sys
from pathlib import Path

a, b = Path(sys.argv[1]), Path(sys.argv[2])
files_a = sorted(p.relative_to(a) for p in (a / "metrics").rglob("*") if p.is_file())
files_b = sorted(p.relative_to(b) for p in (b / "metrics").rglob("*") if p.is_file())
print("file lists equal" if files_a == files_b else f"file lists differ: {set(files_a) ^ set(files_b)}", len(files_a))
for rel in files_a + [Path("collectives.json")]:
    same = (b / rel).exists() and filecmp.cmp(a / rel, b / rel, shallow=False)
    print("identical" if same else "DIFFERS", rel, (a / rel).stat().st_size)

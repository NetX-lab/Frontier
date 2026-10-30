"""List what differs between two A10 run roots that k19_identity.py flags.

    python k19_identity_detail.py <root a> <root b>

Prints the differing config.json keys, the differing collectives.json leaves
and whether the decompressed forwards.jsonl.gz files are equal.
"""
import gzip
import json
import sys
from pathlib import Path


def leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def differing(path_a, path_b):
    a = dict(leaves(json.loads(path_a.read_text())))
    b = dict(leaves(json.loads(path_b.read_text())))
    keys = sorted(set(a) | set(b))
    return [(key, a.get(key), b.get(key)) for key in keys if a.get(key) != b.get(key)], len(keys)


root_a, root_b = Path(sys.argv[1]), Path(sys.argv[2])
config_a = next((root_a / "metrics").rglob("config.json"))
config_b = root_b / config_a.relative_to(root_a)
rows, total = differing(config_a, config_b)
print(f"config.json differing keys {len(rows)} of {total}")
for key, a, b in rows:
    print(f"   {key} | {a} | {b}")
rows, total = differing(root_a / "collectives.json", root_b / "collectives.json")
print(f"collectives.json differing leaves {len(rows)} of {total}")
for key, a, b in rows:
    print(f"   {key} | {a} | {b}")
with gzip.open(root_a / "forwards.jsonl.gz") as fa, gzip.open(root_b / "forwards.jsonl.gz") as fb:
    print("forwards.jsonl decompressed equal:", fa.read() == fb.read())

"""Per-scope difference of the first real prefill forward between two A10 runs.

    python scope_diff.py <base run name> <candidate run name>
"""
import gzip
import json
import sys

S = "/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10"


def first_prefill(name):
    with gzip.open(f"{S}/{name}/forwards.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if record["role"] == "prefill" and record["real"] and record["local_tokens"] == 6144:
                return record


b, c = (first_prefill(name) for name in sys.argv[1:3])
print("composed_ms", round(b["composed_ms"], 3), round(c["composed_ms"], 3), round(c["composed_ms"] - b["composed_ms"], 3))
for key in sorted(set(b["segment_ms"]) | set(c["segment_ms"])):
    vb, vc = b["segment_ms"].get(key, 0.0), c["segment_ms"].get(key, 0.0)
    print("segment", key, round(vb, 3), round(vc, 3), round(vc - vb, 3))
for key in sorted(set(b["scope_ms"]) | set(c["scope_ms"])):
    vb, vc = b["scope_ms"].get(key, 0.0), c["scope_ms"].get(key, 0.0)
    if abs(vc - vb) > 1e-6:
        print("scope", key, round(vb, 3), round(vc, 3), round(vc - vb, 3))

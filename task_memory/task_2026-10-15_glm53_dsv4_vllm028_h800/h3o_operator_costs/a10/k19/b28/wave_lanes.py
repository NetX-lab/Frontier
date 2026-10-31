"""Lanes of the n-th prefill wave with a 6144-token lane in an A10 run.

    python wave_lanes.py <run name> <n>
"""
import gzip
import json
import sys
from collections import defaultdict

S = "/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10"
name, n = sys.argv[1], int(sys.argv[2])
groups = defaultdict(list)
with gzip.open(f"{S}/{name}/forwards.jsonl.gz", "rt") as stream:
    for line in stream:
        record = json.loads(line)
        if record["role"] == "prefill":
            groups[record["start_s"]].append(record)
waves = [start for start in sorted(groups) if any(r["local_tokens"] == 6144 for r in groups[start])]
start = waves[n]
for r in sorted(groups[start], key=lambda r: r["dp"]):
    print(round(start, 6), r["dp"], r["real"], r["local_tokens"], r["step_class"], round(r["composed_ms"], 3),
          {k: round(v, 3) for k, v in r["segment_ms"].items()},
          {k: round(v, 3) for k, v in r["scope_ms"].items() if "_ep_" in k})

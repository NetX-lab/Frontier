"""Mean composed forward time per prefill wave shape in two A10 runs (reads partial gzip streams).

    python wave_prices.py <case> <base run name> <candidate run name>

A wave is the set of prefill-role forwards with one start time; its shape is the
sorted tuple of local token counts of its lanes.
"""
import gzip
import json
import statistics
import sys
import zlib
from collections import defaultdict

S = "/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10"
case, base, cand = sys.argv[1:4]


def waves(name):
    groups = defaultdict(list)
    try:
        with gzip.open(f"{S}/{name}/forwards.jsonl.gz", "rt") as stream:
            for line in stream:
                record = json.loads(line)
                if record["role"] == "prefill":
                    groups[record["start_s"]].append(record)
    except (EOFError, zlib.error, json.JSONDecodeError):
        pass  # the candidate stream is still being written; its tail is incomplete
    shapes = defaultdict(list)
    for start in sorted(groups)[:-1]:
        forwards = groups[start]
        shape = tuple(sorted((r["local_tokens"] for r in forwards), reverse=True))
        shapes[shape].append(max(r["composed_ms"] for r in forwards))
    return shapes


b, c = waves(base), waves(cand)
print(case, "shape (local tokens per lane) | base n, mean ms | candidate n, mean ms | difference")
for shape in sorted(set(b) | set(c), key=lambda s: (-len(b.get(s, [])), s)):
    if max(shape) < 64:
        continue
    nb, nc = len(b.get(shape, [])), len(c.get(shape, []))
    mb = statistics.mean(b[shape]) if nb else float("nan")
    mc = statistics.mean(c[shape]) if nc else float("nan")
    print(f"  {shape} | {nb:5d} {mb:9.3f} | {nc:5d} {mc:9.3f} | {mc - mb:+8.3f}")

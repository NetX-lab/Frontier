"""EP scope change per prefill wave with a 6144-token lane, paired by wave order, grouped by the DP index of that lane.

    python wave_ep_by_dp.py <base run name> <candidate run name>
"""
import gzip
import json
import statistics
import sys
from collections import defaultdict

S = "/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10"


def waves(name):
    groups = defaultdict(list)
    with gzip.open(f"{S}/{name}/forwards.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if record["role"] == "prefill":
                groups[record["start_s"]].append(record)
    rows = []
    for start in sorted(groups):
        big = [r for r in groups[start] if r["local_tokens"] == 6144]
        if big:
            r = big[0]
            rows.append((r["dp"], r["composed_ms"], r["scope_ms"]["dsv4_ep_dispatch_allgather"],
                         r["scope_ms"]["dsv4_ep_output_reduce_scatter"]))
    return rows


base, cand = waves(sys.argv[1]), waves(sys.argv[2])
assert len(base) == len(cand)
by_dp = defaultdict(list)
for b, c in zip(base, cand):
    assert b[0] == c[0]
    by_dp[b[0]].append((c[1] - b[1], c[2] - b[2], c[3] - b[3], b[2], b[3], c[2], c[3]))
print("waves", len(base))
for dp in sorted(by_dp):
    rows = by_dp[dp]
    cols = list(zip(*rows))
    print(f"dp {dp} n {len(rows):4d} composed {statistics.mean(cols[0]):+7.3f} dispatch {statistics.mean(cols[1]):+7.3f}"
          f" combine {statistics.mean(cols[2]):+7.3f} | base dispatch {min(cols[3]):.3f}-{max(cols[3]):.3f}"
          f" combine {min(cols[4]):.3f}-{max(cols[4]):.3f} | cand dispatch {min(cols[5]):.3f}-{max(cols[5]):.3f}"
          f" combine {min(cols[6]):.3f}-{max(cols[6]):.3f}")
all_rows = [row for rows in by_dp.values() for row in rows]
print(f"all composed {statistics.mean(r[0] for r in all_rows):+7.3f} dispatch {statistics.mean(r[1] for r in all_rows):+7.3f}"
      f" combine {statistics.mean(r[2] for r in all_rows):+7.3f}")

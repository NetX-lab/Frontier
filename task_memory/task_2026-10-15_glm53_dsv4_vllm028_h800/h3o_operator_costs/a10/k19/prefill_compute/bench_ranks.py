"""Per-call start and end of the EP combine Reduce on the 8 ranks of node 0 in one benchmark trace directory.

    python bench_ranks.py <benchmark traces dir of the call>
"""
import json
import statistics
import sys

R = "ncclDevKernel_Reduce_Sum_bf16_RING_LL"
bench = {}
for rank in range(8):
    events = json.load(open(f"{sys.argv[1]}/rank{rank}.json"))
    events = events["traceEvents"] if isinstance(events, dict) else events
    bench[rank] = sorted([(e["ts"], e["dur"]) for e in events if e.get("cat") == "kernel" and e["name"].startswith(R)])
calls = len(bench[0])
print(f"{calls} calls; start and end of each rank relative to rank 0's start (us); durations")
for k in range(calls):
    print(f"  call {k}: start", [round(bench[r][k][0] - bench[0][k][0], 1) for r in range(8)],
          "end", [round(bench[r][k][0] + bench[r][k][1] - bench[0][k][0], 1) for r in range(8)],
          "dur", [round(bench[r][k][1], 1) for r in range(8)])

"""Compare the EP combine Reduce on the 8 ranks of node 0 (same host clock): in the native forward
(all ranks arrive together) against the back-to-back benchmark (G5h-32 _01, eager_t6144_1_1_1).

    python b2b_vs_model.py <benchmark traces dir of the call>
"""
import glob
import json
import statistics
import sys

R = "ncclDevKernel_Reduce_Sum_bf16_RING_LL"
model = {}
for p in sorted(glob.glob("calls_dp0_pp0_tp*_dcp0_ep*_rank*.json")):
    model[int(p.split("_tp")[1].split("_")[0])] = json.load(open(p))[0]["calls"][R]
layers = len(model[0])
print("native forward 64: kernel duration and end offset relative to rank 0's start, median over 43 layers (us)")
for t in sorted(model):
    print(f"  rank {t}: start {statistics.median(model[t][i][0] - model[0][i][0] for i in range(layers)):7.1f}"
          f" duration {statistics.median(model[t][i][1] for i in range(layers)):7.1f}"
          f" end {statistics.median(model[t][i][0] + model[t][i][1] - model[0][i][0] for i in range(layers)):7.1f}")
bench = {}
for rank in range(8):
    events = json.load(open(f"{sys.argv[1]}/rank{rank}.json"))
    events = events["traceEvents"] if isinstance(events, dict) else events
    bench[rank] = sorted([(e["ts"], e["dur"]) for e in events if e.get("cat") == "kernel" and e["name"].startswith(R)])
calls = len(bench[0])
print(f"benchmark: {calls} back-to-back calls; per call k: start and end of each rank relative to rank 0's start (us)")
for k in range(calls):
    print(f"  call {k}: start", [round(bench[r][k][0] - bench[0][k][0], 1) for r in range(8)],
          "end", [round(bench[r][k][0] + bench[r][k][1] - bench[0][k][0], 1) for r in range(8)])
period = [bench[0][k + 1][0] - bench[0][k][0] for k in range(calls - 1)]
print("rank 0 start-to-start period us, median", round(statistics.median(period), 1), period and [round(p, 1) for p in period])

"""Split the uniform-prefill forwards' AllGather and ReduceScatter kernels into the 43 EP calls and the TP calls.

The EP calls are the 43 longest kernels of each name; the printed gap between the
43rd and 44th longest shows whether the split is clean.
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from v4_runtime_census import forward_kernel_events

LAYERS = 43
path, first, last = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
per_forward = defaultdict(list)
for fw in forward_kernel_events("prefill", path):
    index = fw.record["index"]
    if not first <= index <= last:
        continue
    durations = defaultdict(list)
    for kernel in fw.kernels_by_segment["dispatch"]:
        name = kernel["name"].split("(")[0]
        if name in ("ncclDevKernel_AllGather_RING_LL", "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL"):
            durations[name].append(kernel["dur"])
    for name, values in sorted(durations.items()):
        values.sort(reverse=True)
        ep, rest = values[:LAYERS], values[LAYERS:]
        per_forward[name].append(statistics.median(ep))
        print(index, name, "EP median", round(statistics.median(ep), 1), "EP min", round(min(ep), 1),
              "next longest", round(rest[0], 1), "TP median", round(statistics.median(rest), 1))
for name, values in per_forward.items():
    print("ALL", name, "median of forward medians", round(statistics.median(values), 1), "forwards", len(values))

"""Print, for one uniform-prefill forward, the 40th to 48th longest AllGather and ReduceScatter kernels."""
import sys
from collections import defaultdict
from pathlib import Path

from v4_runtime_census import forward_kernel_events

path, index = Path(sys.argv[1]), int(sys.argv[2])
for fw in forward_kernel_events("prefill", path):
    if fw.record["index"] != index:
        continue
    durations = defaultdict(list)
    for kernel in fw.kernels_by_segment["dispatch"]:
        name = kernel["name"].split("(")[0]
        if name in ("ncclDevKernel_AllGather_RING_LL", "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL"):
            durations[name].append(kernel["dur"])
    for name, values in sorted(durations.items()):
        values.sort(reverse=True)
        print(index, name, "count", len(values), "ranks 40-48", [round(v, 1) for v in values[39:48]],
              "count above 700", sum(v > 700 for v in values))

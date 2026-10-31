"""Per-forward EP dispatch and combine kernel names and median per-layer durations on one rank."""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from v4_runtime_census import forward_kernel_events

path, first, last = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
for fw in forward_kernel_events("prefill", path):
    index = fw.record["index"]
    if not first <= index <= last:
        continue
    durations = defaultdict(list)
    for kernel in fw.kernels_by_segment["dispatch"]:
        name = kernel["name"].split("(")[0]
        if name.startswith(("ncclDevKernel_AllGather", "ncclDevKernel_Reduce")):
            durations[name].append(kernel["dur"])
    print(index, fw.record["class"], {name: (len(values), round(statistics.median(values), 1))
                                      for name, values in sorted(durations.items())})

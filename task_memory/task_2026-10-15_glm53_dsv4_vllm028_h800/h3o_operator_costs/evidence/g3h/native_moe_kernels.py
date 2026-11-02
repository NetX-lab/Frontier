"""List the native routed-MoE and gate kernels per forward class.

Usage: native_moe_kernels.py <torch_traces dir> <output json>
Kernels count per forward, keyed by (name, grid), over the dispatch segment of
each lane's TP0 trace (the census segmentation).
"""

import collections
import json
import sys
from pathlib import Path

from v4_runtime_census import forward_kernel_events, lane_traces

PATTERNS = ("marlin_moe_wna16", "moe_align_block_size", "count_and_sort_expert_tokens",
            "_swiglu_limit_pad_aware_kernel", "moe_sum_vec", "moe_sum_scalar",
            "cute_dsl", "nvjet_sm90_tss_64x32_64x16_2x4_h_bz_TNT",
            "FillFunctor<int>", "CUDAFunctorOnSelf_add<float>")

trace_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
classes = collections.defaultdict(lambda: {"forwards": 0, "kernels": collections.Counter(),
                                           "tokens": collections.Counter()})
for role, dp, path in lane_traces(trace_dir):
    for forward in forward_kernel_events(role, path):
        record = forward.record
        key = f"{role} dp{dp} {record['mode']} {record['class']}"
        entry = classes[key]
        entry["forwards"] += 1
        step = record.get("step")
        if step is not None:
            entry["tokens"][step["context_tokens"] + step["generation_tokens"]] += 1
        for event in forward.kernels_by_segment["dispatch"]:
            if any(pattern in event["name"] for pattern in PATTERNS):
                grid = tuple(event["args"].get("grid", ()))
                entry["kernels"][(event["name"], grid)] += 1
    print("lane done", role, dp, flush=True)
result = {
    key: {"forwards": entry["forwards"],
          "scheduled_tokens": {str(k): v for k, v in sorted(entry["tokens"].items())},
          "kernels": [{"name": name, "grid": list(grid), "per_forward": count / entry["forwards"]}
                      for (name, grid), count in sorted(entry["kernels"].items())]}
    for key, entry in sorted(classes.items())
}
output.write_text(json.dumps(result, indent=1))

"""Break down A4 preview errors: actual signed gaps per scope versus class means (diagnostic)."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from frontier.execution_time_predictor.dsv4_composition import segment_class
from frontier.profiling.kernel_gap.dsv4_replay_gaps import activity_gaps, build_layer_calls, read_replay_records, reduce_dsv4_gaps

root = Path(sys.argv[1])
records = read_replay_records([root / m / "raw" / "dsv4_raw_e0.jsonl" for m in ("NONE", "PIECEWISE", "FULL_AND_PIECEWISE")])
values = reduce_dsv4_gaps(records).values_us
print("class means us", values)
gaps = activity_gaps(records)
by_layer = defaultdict(list)
for g in gaps:
    if g.attention_owned and g.interval["activity"] == "kernel":
        by_layer[(g.source, g.interval["case_id"], g.interval["step_index"], g.interval["global_layer_id"])].append(g)
calls = build_layer_calls(records)
summary = defaultdict(lambda: defaultdict(list))
for call in calls:
    key = (call.source, call.step["case_id"], call.step["step_index"], call.global_layer_id)
    owned = by_layer[key]
    if not owned or any(not g.host_ahead for g in owned):
        continue
    mode = call.step["cudagraph_dispatch"]["cg_mode"]
    emode = call.engine["cudagraph_capture"]["cudagraph_mode"]
    first = min(owned, key=lambda g: g.interval["start_us"])
    target = max(g.interval["end_us"] for g in owned) - first.interval["start_us"] + first.gap_us
    group = (emode, mode, call.step["case_id"], call.variant)
    s = summary[group]
    s["target_us"].append(target)
    s["kernels"].append(len(owned))
    s["cost_us"].append(sum(g.interval["end_us"] - g.interval["start_us"] for g in owned))
    for g in owned:
        seg = segment_class(mode, g.interval["scope"])
        s[f"gap:{seg}"].append(g.gap_us)
    per_scope = defaultdict(list)
    for g in owned:
        per_scope[g.interval["scope"]].append(g.gap_us)
    for scope, vals in per_scope.items():
        s[f"scope:{scope}"].extend(vals)
for group in sorted(summary):
    s = summary[group]
    print(f"\n{group}: calls {len(s['target_us'])} target {np.median(s['target_us']):.1f} us, kernels {np.median(s['kernels']):.0f}, kernel time {np.median(s['cost_us']):.1f} us")
    for k in sorted(s):
        if k.startswith("gap:"):
            seg = k[4:]
            v = np.asarray(s[k])
            print(f"   {seg}: n/call {len(v)/len(s['target_us']):.0f} actual mean {v.mean():.3f} us vs class {values.get(seg, float('nan')):.3f}; excess per call {(values.get(seg, np.nan) - v.mean()) * len(v)/len(s['target_us']):.2f} us")
    if "--scopes" in sys.argv:
        for k in sorted(s):
            if k.startswith("scope:"):
                v = np.asarray(s[k])
                print(f"      {k[6:]:45s} n/call {len(v)/len(s['target_us']):5.1f} mean {v.mean():7.3f} median {np.median(v):7.3f} min {v.min():7.3f}")

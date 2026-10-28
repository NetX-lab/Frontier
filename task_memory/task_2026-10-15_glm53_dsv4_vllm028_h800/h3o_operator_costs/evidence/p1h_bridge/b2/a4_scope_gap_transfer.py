"""Price each call with per-scope gap totals measured at a neighbouring decode shape (diagnostic)."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from frontier.execution_time_predictor import dsv4_composition as comp
from frontier.profiling.kernel_gap.dsv4_replay_gaps import activity_gaps, build_layer_calls, read_replay_records

NEIGHBOUR = {"e0-decode-b1-p256": "e0-decode-b4-p2048", "e0-decode-b4-p2048": "e0-decode-b8-p2052",
             "e0-decode-b8-p2052": "e0-decode-b4-p2048"}
root = Path(sys.argv[1])
records = read_replay_records([root / m / "raw" / "dsv4_raw_e0.jsonl" for m in ("NONE", "PIECEWISE", "FULL_AND_PIECEWISE")])
by_layer = defaultdict(list)
for g in activity_gaps(records):
    if g.attention_owned:
        by_layer[(g.source, g.interval["case_id"], g.interval["step_index"], g.interval["global_layer_id"])].append(g)
calls, totals = [], defaultdict(list)
for call in build_layer_calls(records):
    owned = by_layer[(call.source, call.step["case_id"], call.step["step_index"], call.global_layer_id)]
    kernels = [g for g in owned if g.interval["activity"] == "kernel"]
    if not kernels or any(g.dependency_end_us is None or not g.host_ahead for g in owned):
        continue
    mode = call.step["cudagraph_dispatch"]["cg_mode"]
    own = defaultdict(float)
    for g in kernels:
        own[g.interval["scope"]] += g.gap_us * 1e-3
    first = min(kernels, key=lambda g: g.interval["start_us"])
    target = (max(g.interval["end_us"] for g in kernels) - first.interval["start_us"] + first.gap_us) * 1e-3
    key = (call.engine["cudagraph_capture"]["cudagraph_mode"], mode, call.variant, call.global_layer_id)
    for scope in {row["scope"] for row in call.calls}:
        totals[(*key, call.step["case_id"], scope)].append(own[scope])
    calls.append((key, call, target))
errors = defaultdict(list)
for key, call, target in calls:
    case = call.step["case_id"]
    if case not in NEIGHBOUR:
        continue
    mode = call.step["cudagraph_dispatch"]["cg_mode"]
    weight = {}
    for row in call.calls:
        measured = totals.get((*key, NEIGHBOUR[case], row["scope"]))
        if measured is None:
            break
        weight[row["scope"]] = row["time_ms"] + float(np.mean(measured))
    else:
        options = dict(variant=call.variant, graph_runtime_mode=mode, rows=call.step["cudagraph_dispatch"]["num_tokens"],
                       multi_stream_gemm_token_threshold=call.engine["multi_stream_gemm_token_threshold"],
                       indexer_short_context=call.short_context)
        zero = dict.fromkeys({comp.segment_class(mode, n) for n in weight}, 0.0)
        value = comp.compose_attention_phase(weight, dict.fromkeys(weight, 0), zero, **options).elapsed_ms
        errors[(key[0], key[1], case, key[2])].append((value - target) / target)
for group in sorted(errors):
    e = np.asarray(errors[group])
    print(f"{group} neighbour {NEIGHBOUR[group[2]][3:]}: calls {len(e)} median {np.median(np.abs(e)):.3f} p90 {np.percentile(np.abs(e), 90):.3f} signed {np.median(e):+.3f}")

"""Print one host-ahead layer call: measured scope windows and streams against the composed schedule (diagnostic)."""
import sys
from collections import defaultdict
from pathlib import Path

from frontier.execution_time_predictor import dsv4_composition as comp
from frontier.profiling.kernel_gap.dsv4_replay_gaps import activity_gaps, build_layer_calls, read_replay_records

root, engine_mode, case, variant = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
pick = int(sys.argv[5]) if len(sys.argv) > 5 else 0
records = read_replay_records([root / m / "raw" / "dsv4_raw_e0.jsonl" for m in ("NONE", "PIECEWISE", "FULL_AND_PIECEWISE")])
by_layer = defaultdict(list)
for g in activity_gaps(records):
    if g.attention_owned:
        by_layer[(g.source, g.interval["case_id"], g.interval["step_index"], g.interval["global_layer_id"])].append(g)
chosen = []
for call in build_layer_calls(records):
    if call.engine["cudagraph_capture"]["cudagraph_mode"] != engine_mode or call.step["case_id"] != case or call.variant != variant:
        continue
    owned = by_layer[(call.source, call.step["case_id"], call.step["step_index"], call.global_layer_id)]
    if owned and all(g.host_ahead and g.dependency_end_us is not None for g in owned):
        chosen.append((call, owned))
call, owned = chosen[pick]
mode = call.step["cudagraph_dispatch"]["cg_mode"]
rows = call.step["cudagraph_dispatch"]["num_tokens"]
kernels = sorted((g for g in owned if g.interval["activity"] == "kernel"), key=lambda g: g.interval["start_us"])
origin = kernels[0].interval["start_us"] - kernels[0].gap_us
target = max(g.interval["end_us"] for g in kernels) - origin
scope_ms = {row["scope"]: row["time_ms"] for row in call.calls}
counts = {row["scope"]: row["kernel_count"] for row in call.calls}
own = defaultdict(float)
for g in kernels:
    own[g.interval["scope"]] += g.gap_us * 1e-3
options = dict(variant=call.variant, graph_runtime_mode=mode, rows=rows,
               multi_stream_gemm_token_threshold=call.engine["multi_stream_gemm_token_threshold"],
               indexer_short_context=call.short_context)
sched = comp.compose_attention_phase({n: v + own[n] for n, v in scope_ms.items()}, counts,
                                     dict.fromkeys({comp.segment_class(mode, n) for n in scope_ms}, 0.0), **options)
preds = comp.attention_scope_predecessors(scope_ms, **options)
print(f"{engine_mode} {mode} {case} layer {call.global_layer_id} rows {rows} target {target:.1f} us composed {sched.elapsed_ms*1e3:.1f} us")
by_scope = defaultdict(list)
for g in kernels:
    by_scope[g.interval["scope"]].append(g)
order = sorted(by_scope, key=lambda s: by_scope[s][0].interval["start_us"])
print(f"{'scope':44s} {'streams':10s} {'start':>7s} {'end':>7s} {'composed':>8s} {'weight':>7s} {'gaps':>6s}  predecessors")
for s in order:
    ks = by_scope[s]
    streams = sorted({k.interval["stream"] for k in ks})
    start = ks[0].interval["start_us"] - origin
    end = max(k.interval["end_us"] for k in ks) - origin
    weight = (scope_ms[s] + own[s]) * 1e3
    print(f"{s:44s} {','.join(map(str, streams)):10s} {start:7.1f} {end:7.1f} {sched.start_ms[s]*1e3:8.1f} {weight:7.1f} {own[s]*1e3:6.1f}  {sorted(preds[s])}")
    if "--kernels" in sys.argv:
        for k in ks:
            print(f"      {k.interval['name'][:60]:60s} s{k.interval['stream']} {k.interval['start_us']-origin:7.1f}-{k.interval['end_us']-origin:7.1f} gap {k.gap_us:6.2f} kind {k.interval['launch_kind']}")

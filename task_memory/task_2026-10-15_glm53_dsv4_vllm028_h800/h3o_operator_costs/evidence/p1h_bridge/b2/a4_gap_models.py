"""Compare V4 gap models on the saved replay layer calls (diagnostic, not acceptance).

Models price each owned kernel's gap:
  class            one mean per segment class (the B2 contract)
  class_position   mean per (class, scope entry or intra-scope kernel)
  class_scope      mean per (class, scope)
  fold_class_edge  intra-scope gaps folded into the scope cost; entry gap per (class, same- or cross-stream entry)
  fold_class_scope intra-scope gaps folded into the scope cost; entry gap per (class, scope)
Each model is fitted in-sample and with one case held out at a time.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from frontier.execution_time_predictor import dsv4_composition as comp
from frontier.profiling.kernel_gap.dsv4_replay_gaps import activity_gaps, build_layer_calls, read_replay_records

root = Path(sys.argv[1])
records = read_replay_records([root / m / "raw" / "dsv4_raw_e0.jsonl" for m in ("NONE", "PIECEWISE", "FULL_AND_PIECEWISE")])
by_layer = defaultdict(list)
for g in activity_gaps(records):
    if g.attention_owned:
        by_layer[(g.source, g.interval["case_id"], g.interval["step_index"], g.interval["global_layer_id"])].append(g)


def program_streams(variant, mode, multi, short):
    streams = {}
    for step, stream, operand in comp._layer_program(variant, mode, multi, short):
        if step == "run":
            for name in operand:
                streams[name] = stream
    return streams


samples = []
for call in build_layer_calls(records):
    owned = by_layer[(call.source, call.step["case_id"], call.step["step_index"], call.global_layer_id)]
    kernels = sorted((g for g in owned if g.interval["activity"] == "kernel"), key=lambda g: g.interval["start_us"])
    if not kernels or any(g.dependency_end_us is None or not g.host_ahead for g in owned):
        continue
    mode = call.step["cudagraph_dispatch"]["cg_mode"]
    rows = call.step["cudagraph_dispatch"]["num_tokens"]
    options = dict(variant=call.variant, graph_runtime_mode=mode, rows=rows,
                   multi_stream_gemm_token_threshold=call.engine["multi_stream_gemm_token_threshold"],
                   indexer_short_context=call.short_context)
    scope_ms = {row["scope"]: row["time_ms"] for row in call.calls}
    counts = {row["scope"]: row["kernel_count"] for row in call.calls}
    preds = comp.attention_scope_predecessors(scope_ms, **options)
    streams = program_streams(call.variant, mode, rows <= options["multi_stream_gemm_token_threshold"], call.short_context)
    first_seen, items = set(), []
    for g in kernels:
        scope = g.interval["scope"]
        entry = scope not in first_seen
        first_seen.add(scope)
        edge = "cross" if entry and any(streams[p] != streams[scope] for p in preds[scope]) else "same"
        items.append((scope, comp.segment_class(mode, scope), entry, edge, g.gap_us))
    origin = kernels[0].interval["start_us"] - kernels[0].gap_us
    target = (max(g.interval["end_us"] for g in kernels) - origin) * 1e-3
    group = (call.engine["cudagraph_capture"]["cudagraph_mode"], mode, call.step["case_id"], call.variant)
    samples.append(dict(group=group, case=call.step["case_id"], options=options, scope_ms=scope_ms, counts=counts,
                        items=items, target=target))

MODELS = {
    "class": lambda it: (it[1],),
    "class_position": lambda it: (it[1], it[2]),
    "class_scope": lambda it: (it[1], it[0]),
    "fold_class_edge": lambda it: (it[1], it[3]),
    "fold_class_scope": lambda it: (it[1], it[0]),
}


def fit(model, train):
    values = defaultdict(list)
    for s in train:
        for it in s["items"]:
            if model.startswith("fold") and not it[2]:
                continue
            values[MODELS[model](it)].append(it[4])
    return {k: float(np.mean(v)) for k, v in values.items()}


def predict(model, table, s):
    weight = dict(s["scope_ms"])
    for it in s["items"]:
        if model.startswith("fold") and not it[2]:
            weight[it[0]] += it[4] * 1e-3  # Measured intra-scope gap folded into the scope cost.
            continue
        key = MODELS[model](it)
        if key not in table:
            return None
        weight[it[0]] += table[key] * 1e-3
    zero = dict.fromkeys({comp.segment_class(s["options"]["graph_runtime_mode"], n) for n in weight}, 0.0)
    return comp.compose_attention_phase(weight, dict.fromkeys(weight, 0), zero, **s["options"]).elapsed_ms


def evaluate(model, held_out):
    errors = defaultdict(list)
    cases = sorted({s["case"] for s in samples})
    for case in (cases if held_out else [None]):
        train = [s for s in samples if s["case"] != case]
        table = fit(model, train)
        for s in samples:
            if held_out and s["case"] != case:
                continue
            value = predict(model, table, s)
            if value is not None:
                errors[s["group"]].append((value - s["target"]) / s["target"])
    return errors


for held_out in (False, True):
    print("\n== held-out case ==" if held_out else "== in-sample ==")
    for model in MODELS:
        errors = evaluate(model, held_out)
        stats = {g: (np.median(np.abs(e)), np.percentile(np.abs(e), 90)) for g, e in errors.items()}
        passed = sum(m <= 0.05 and p <= 0.10 for m, p in stats.values())
        worst = max(stats.items(), key=lambda kv: kv[1][0])
        print(f"{model:18s} groups {len(stats):3d} pass {passed:3d}  worst median {worst[1][0]:.3f} {worst[0]}")
        if "-v" in sys.argv:
            for g in sorted(stats):
                print(f"      {g} median {stats[g][0]:.3f} p90 {stats[g][1]:.3f} signed {np.median(errors[g]):+.3f}")

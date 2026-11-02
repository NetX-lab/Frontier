"""Lead check (B8 Acceptance 7): summarize one structural run for a before/after diff.

Real lanes: attention scope call counts per (batch, layer) from op traces.
Idle lanes emit no attention trace; for EP waves whose participants are all
request-free dummies, the time from the previous layer's wave end to this
wave's start is the idle attention phase (synthetic 0.01 ms per call).
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1])
out = Path(sys.argv[2])
stages = {}
for line in (run / "frontier_stage_batch_ledger.jsonl").open():
    row = json.loads(line)
    stages[row["batch_id"]] = row
per_layer = defaultdict(dict)
variant_of = {}
with (run / "op_traces.jsonl").open() as stream:
    for line in stream:
        if '"source_tensor_calls"' not in line:
            continue
        row = json.loads(line)
        meta = row.get("meta", {})
        if "phase" in meta or row.get("batch_id") not in stages or "source_tensor_calls" not in meta:
            continue
        for layer in meta.get("global_layer_ids", [row["layer_id"]]):
            per_layer[(row["batch_id"], layer)][row["name"]] = sum(
                call.get("global_layer_id", layer) == layer for call in meta["source_tensor_calls"])
            if "attention_variant_id" in meta:
                variant_of[(row["batch_id"], layer)] = meta["attention_variant_id"]
real = Counter()
real_scopes = defaultdict(Counter)
for key, scopes in per_layer.items():
    batch, layer = key
    stage = stages[batch]
    q = stage["request_num_tokens"][0]
    real[(variant_of.get(key), q, sum(scopes.values()))] += 1
    for name, calls in scopes.items():
        real_scopes[variant_of.get(key)][name] += calls
waves = defaultdict(list)
for line in (run / "frontier_ep_wave_lane_ledger.jsonl").open():
    row = json.loads(line)
    waves[(row["replica_id"], row["wave_start_time_s"])].append(
        (row["global_layer_ids"][0], row["attention_variant_id"], row["source_forward_kind"],
         row["wave_end_time_s"]))
ordered = sorted(waves.items(), key=lambda item: item[0][1])
idle_gaps = defaultdict(Counter)
previous_end = None
for (_, start), rows in ordered:
    layers = {r[0] for r in rows}
    layer, = layers
    variant = rows[0][1]
    all_idle = all(r[2] == "request-free dummy" for r in rows)
    if all_idle and previous_end is not None and layer > 0:
        idle_gaps[variant][round((start - previous_end) * 1000, 9)] += 1
    previous_end = max(r[3] for r in rows)
system = json.loads((run / "system_metrics.json").read_text())
with (run / "request_metrics.csv").open(newline="") as stream:
    requests = list(csv.DictReader(stream))
summary = dict(
    real_attention_calls_per_layer={f"{v}|q={q}|calls={n}": c for (v, q, n), c in sorted(real.items(), key=str)},
    real_scope_calls={v: dict(sorted(c.items())) for v, c in sorted(real_scopes.items(), key=str)},
    all_idle_wave_attention_gap_ms={v: {str(k): c for k, c in sorted(g.items())} for v, g in sorted(idle_gaps.items())},
    system_metrics={k: system[k] for k in ("ttft_statistics", "tpot_statistics", "request_e2e_time_statistics",
                                            "throughput_metrics") if k in system},
    request_metrics=requests,
    stage_total_time_ms=sorted(round(s["execution_time"]["total_time_ms"], 9) for s in stages.values()),
)
out.write_text(json.dumps(summary, indent=1, default=str) + "\n")
print("layers", len(per_layer), "idle waves", sum(sum(g.values()) for g in idle_gaps.values()))

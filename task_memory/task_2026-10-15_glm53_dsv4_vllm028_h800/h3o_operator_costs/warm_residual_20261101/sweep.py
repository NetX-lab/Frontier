"""Read-only price-neighborhood averaging and warm-native span allocation.

The price average is the scratch analysis's phase proxy, not the phase mean
of a fixed-price, longer replay. Both are retained so they cannot be confused.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

from fifo_model import mean_wait, simulate

OUT = Path(sys.argv[1])
S = json.load(open(OUT / "summary.json"))
TASK = Path(__file__).resolve().parents[2]
wave = Path("/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-wave-20261030-01/artifacts")
excluded = set(json.load(open(TASK / "h3o_operator_costs/a10/remainder/first_contact.json")))
proxy = {r["request_id"]: r for r in map(json.loads, open(wave / "run/proxy_requests.jsonl"))}
ingress = {"prefill": [], "decode": []}
loops = {"prefill": [], "decode": []}
for path in wave.glob("nodes/*/scheduler_lifecycle/*.jsonl"):
    start, enqueue = None, {}
    for r in map(json.loads, open(path)):
        if r["event"] == "start":
            start = r
        if r["event"] == "add_enqueue":
            enqueue[r["request_id"]] = r["t"]
        if r["event"] != "add" or "formal" not in r["request_id"]:
            continue
        key = r["request_id"].split("cmpl-", 1)[1].rsplit("-0-", 1)[0]
        if key in excluded:
            continue
        role = start["role"]
        endpoint = "proxy_arrival_time" if role == "prefill" else "prefill_response_time"
        ingress[role].append((enqueue[r["request_id"]] - proxy[key][endpoint]) * 1000)
        loops[role].append((r["t"] - enqueue[r["request_id"]]) * 1000)
N64 = S["native_diagnostics"]["wave64"]["parts"]
N32 = S["native_diagnostics"]["lifecycle32"]["parts"]
alloc = {}
for case in ("p2_64", "p2_32"):
    native = S[case]["native"]["warm"]
    if case == "p2_64":
        p_ingress = statistics.mean(ingress["prefill"])
        p_response = N64["p_update->prefill_response"]
        p_step = 289.6  # native non-queued sample, A10 section 5.5, n=6
        p_queue = native["p_side"] - p_ingress - p_response - p_step
        d_ingress = statistics.mean(ingress["decode"])
        kv = statistics.mean(loops["decode"]) + N64["d_add->d_recv_done"]
        d_first = N64["d_recv_done->d_schedule"] + N64["d_schedule->d_update"]
        egress = N64["d_update->decode_first_byte"] + native["client_egress"]
    else:
        # No enqueue endpoint on P2-32. Native P ingress stops at engine add.
        p_ingress = N32["proxy_arrival->p_add"]
        p_response = N32["p_update->prefill_response"]
        p_queue = N32["p_add->p_schedule"]
        p_step = native["p_side"] - p_ingress - p_response - p_queue
        d_ingress = statistics.mean(ingress["decode"])  # transferred input, not a P2-32 observation
        kv = N32["prefill_response->d_add"] - d_ingress + N32["d_add->d_recv_done"]
        d_first = N32["d_recv_done->d_schedule"] + N32["d_schedule->d_update"]
        egress = N32["d_update->decode_first_byte"] + native["client_egress"]
    n = dict(client_dispatch=native["client_dispatch"], prefill_ingress=p_ingress, prefill_queue=p_queue,
             prefill_step=p_step, prefill_response=p_response, decode_ingress=d_ingress,
             kv=kv, first_decode=d_first, egress=egress)
    n["unallocated_clean_probe"] = native["ttft"] - sum(n.values())
    n["ttft"] = native["ttft"]
    alloc[case] = n

smoothed = {}
for gen, data in S["p2_64"]["frontier"].items():
    p = data["fifo"]["p"]
    costs = [24.20 + i * 0.001 for i in range(301)]
    baseline = [mean_wait(simulate(g, p)) for g in costs]
    narrow = [v for g, v in zip(costs, baseline) if 24.33 <= g <= 24.39]
    smoothed[gen] = dict(wait_mean=statistics.mean(baseline), wait_min=min(baseline), wait_max=max(baseline),
                         narrow_wait_mean=statistics.mean(narrow), narrow_wait_min=min(narrow), narrow_wait_max=max(narrow))
    smoothed[gen]["ttft"] = data["means"]["ttft"] - data["means"]["prefill_queue"] + smoothed[gen]["wait_mean"]
    smoothed[gen]["deltas"] = {}
    coarse = [24.20 + i * 0.005 for i in range(61)]
    coarse_base = statistics.mean(mean_wait(simulate(g, p)) for g in coarse)
    for delta in (2.7, 7.5, 9.7, 10, 12):
        step_wait = statistics.mean(mean_wait(simulate(g, p + delta)) for g in coarse)
        smoothed[gen]["deltas"][str(delta)] = delta + step_wait - coarse_base

fields = list(alloc["p2_64"])
tables = []
for case, n in alloc.items():
    for field in fields:
        r = dict(case=case, span=field, native=n[field])
        for gen, d in S[case]["frontier"].items():
            f = d["means"].get(field, 0.0)
            r[gen] = f
            r[gen + "_minus_native"] = f - n[field]
        tables.append(r)
with open(OUT / "span_tables.csv", "w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
    writer.writeheader()
    writer.writerows(tables)
result = dict(native_allocation=alloc, measured_wave_ingress={k: statistics.mean(v) for k, v in ingress.items()},
              measured_wave_engine_loop={k: statistics.mean(v) for k, v in loops.items()},
              price_neighborhood_averaging=smoothed)
json.dump(result, open(OUT / "sweep_summary.json", "w"), indent=2)
print(json.dumps(result, indent=2))

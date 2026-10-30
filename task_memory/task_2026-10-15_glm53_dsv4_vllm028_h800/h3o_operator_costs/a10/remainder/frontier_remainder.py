"""Frontier PDD TTFT decomposition after the prefill side (lead scratch).

    python frontier_remainder.py <metrics run dir> <output json>

Per request, from metrics_ground_truth.jsonl and frontier_transfer_ledger.jsonl:
arrival, prefill completion, KV transfer start and end, first decode token.
"""
import json
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


done = {r["request_id"]: r for r in rows(run / "metrics_ground_truth.jsonl") if r["event_type"] == "request_completion"}
kv = {}
for r in rows(run / "frontier_transfer_ledger.jsonl"):
    if r["transfer_kind"] == "kv_cache":
        [rid] = r["request_ids"]
        kv[rid] = r

out = []
for rid, d in sorted(done.items()):
    k = kv[rid]
    t = dict(arrival=d["arrived_at"], prefill_done=d["prefill_completed_at"], kv_start=k["start_ts_s"],
             kv_end=k["end_ts_s"], first_decode=d["first_decode_token_completed_at"])
    names = list(t)
    parts = {f"{a}->{b}": (t[b] - t[a]) * 1000 for a, b in zip(names, names[1:])}
    out.append(dict(request=rid, d_replica=k["target_replica_id"], d_lane=k["target_replica_local_id"],
                    ttft_ms=(t["first_decode"] - t["arrival"]) * 1000,
                    p_side_ms=(t["prefill_done"] - t["arrival"]) * 1000,
                    remainder_ms=(t["first_decode"] - t["prefill_done"]) * 1000, kv_bytes=k["bytes"], parts=parts))

summary = {}
for key in ("ttft_ms", "p_side_ms", "remainder_ms"):
    values = [o[key] for o in out]
    summary[key] = (statistics.mean(values), min(values), max(values), statistics.median(values))
for part in out[0]["parts"]:
    values = [o["parts"][part] for o in out]
    summary[part] = (statistics.mean(values), min(values), max(values), statistics.median(values))
print("requests", len(out), "kv bytes", sorted({o["kv_bytes"] for o in out}))
for key, (mean, low, high, median) in summary.items():
    print(f"{key:30s} mean {mean:8.1f} min {low:8.1f} max {high:8.1f} median {median:8.1f}")
json.dump(dict(summary=summary, rows=out), open(sys.argv[2], "w"), indent=1)

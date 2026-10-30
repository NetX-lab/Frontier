"""Native P2-64 (prefill lane, decode lane) pairs and the decode-side ingress split (lead scratch).

For every request (warmup and formal) the decode engine adds, its prefill lane
(from the prefill engine that served its remote_request_id) and decode lane, in
add order; marks the first request of each pair. Writes the formal
requests that open a pair to <output json>. For formal requests, when the run
records add_enqueue, splits prefill response -> decode add into response ->
add_enqueue and add_enqueue -> add.

    python pairs.py <artifacts> <remainder_out.json> <nixl_detail_out.json> <output json>
"""
import json
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])
remainder = {r["request"]: r for r in json.load(open(sys.argv[2]))["rows"]}
nixl = {r["request"]: r for r in json.load(open(sys.argv[3]))}


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


p_lane, d_add, d_enqueue = {}, {}, {}
for path in sorted(run.glob("nodes/*/scheduler_lifecycle/*.jsonl")):
    records = rows(path)
    start = next(r for r in records if r["event"] == "start")
    for r in records:
        if r["event"] == "add":
            if start["role"] == "prefill":
                p_lane[r["request_id"]] = start["dp_rank"]
            else:
                d_add[r["request_id"]] = (r["t"], start["dp_rank"], r.get("remote_request_id"))
        elif r["event"] == "add_enqueue" and start["role"] == "decode":
            d_enqueue[r["request_id"]] = r["t"]

seen = set()
firsts = []
for rid, (t, lane, remote) in sorted(d_add.items(), key=lambda kv: kv[1][0]):
    pair = (p_lane.get(remote), lane)
    key = rid.split("cmpl-", 1)[1].rsplit("-0-", 1)[0]
    if pair not in seen:
        seen.add(pair)
        firsts.append((key, pair))
print("pairs used", len(seen), sorted(seen, key=str))
json.dump([key for key, _ in firsts if "formal" in key], open(sys.argv[4], "w"), indent=1)
print("first request of each pair:")
for key, pair in firsts:
    print("  ", pair, key, "wait_ms", round(nixl[key]["wait_ms"], 1) if key in nixl else "-")

if not d_enqueue:
    raise SystemExit("no add_enqueue records: ingress split not available")
proxy = {r["request_id"]: r for r in rows(run / "run/proxy_requests.jsonl")}
split = []
for rid, (t, lane, remote) in d_add.items():
    key = rid.split("cmpl-", 1)[1].rsplit("-0-", 1)[0]
    if key not in remainder:
        continue
    response = proxy[key]["prefill_response_time"]
    split.append(((d_enqueue[rid] - response) * 1000, (t - d_enqueue[rid]) * 1000))
for name, i in (("prefill_response->d_add_enqueue", 0), ("d_add_enqueue->d_add", 1)):
    values = [s[i] for s in split]
    print(f"{name:34s} n {len(values)} mean {statistics.mean(values):6.1f} median {statistics.median(values):6.1f} "
          f"min {min(values):6.1f} max {max(values):6.1f}")

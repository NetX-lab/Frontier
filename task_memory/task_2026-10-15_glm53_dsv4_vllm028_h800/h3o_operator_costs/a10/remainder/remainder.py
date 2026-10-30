"""Native P2-64 TTFT decomposition after the prefill side (lead scratch).

    python remainder.py <artifacts> <output json>

Joins, per formal request, the proxy record, the prefill engine's lifecycle
(add, schedule, first-token update) and the decode engine's lifecycle (add,
finished_recving, first-tail schedule, first-token update). Times are host wall
clocks (`t`); the client's first token is mapped through the proxy's
monotonic/wall pair of the same request.
"""
import collections
import json
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


engines = {}
for path in sorted(run.glob("nodes/*/scheduler_lifecycle/*.jsonl")):
    records = rows(path)
    start = next(r for r in records if r["event"] == "start")
    engines[(start["role"], start["dp_rank"])] = records

prefill = {}
decode = {}
for (role, rank), records in engines.items():
    pending = {}
    first_tail = {}
    for r in records:
        if r["event"] == "add" and "formal" in r["request_id"]:
            target = prefill if role == "prefill" else decode
            target[r["request_id"]] = dict(lane=rank, add=r["t"], arrival=r["arrival_time"],
                                           remote=r.get("remote_request_id"))
        elif r["event"] == "schedule":
            for rid, *_ in r.get("new", []) or []:
                if "formal" not in rid:
                    continue
                if role == "prefill":
                    prefill[rid]["schedule"] = r["t"]
                    pending[r["step"]] = rid
                else:
                    decode[rid]["schedule"] = r["t"]
                    first_tail[r["step"]] = rid
        elif r["event"] == "update":
            for rid in r.get("finished_recving", []) or []:
                if rid in decode:
                    decode[rid]["finished_recving"] = r["t"]
            if role == "prefill" and r.get("step") in pending:
                prefill[pending.pop(r["step"])]["update"] = r["t"]
            if role == "decode":
                for rid in r.get("first_token", []) or []:
                    if rid in decode:
                        decode[rid]["update"] = r["t"]
    for rid in decode:
        pass

proxy = {r["request_id"]: r for r in rows(run / "run/proxy_requests.jsonl")}
client = {r["request_id"]: r for r in rows(run / "run/client_requests.jsonl")}
by_remote = {d["remote"]: rid for rid, d in decode.items()}

out = []
for pid, p in prefill.items():
    did = by_remote.get(pid)
    if did is None or "update" not in p:
        continue
    d = decode[did]
    key = pid.split("cmpl-", 1)[1].rsplit("-0-", 1)[0]
    x = proxy.get(key)
    c = client.get(key)
    if x is None or c is None or "update" not in d:
        continue
    client_first_wall = x["proxy_arrival_time"] + (c["first_token_monotonic"] - x["proxy_arrival_monotonic"])
    t = dict(
        proxy_arrival=x["proxy_arrival_time"], p_add=p["add"], p_schedule=p["schedule"], p_update=p["update"],
        prefill_response=x["prefill_response_time"], d_add=d["add"], d_recv_done=d.get("finished_recving"),
        d_schedule=d["schedule"], d_update=d["update"], decode_first_byte=x["decode_first_byte_time"],
        client_first=client_first_wall,
    )
    names = list(t)
    parts = {f"{a}->{b}": (t[b] - t[a]) * 1000 for a, b in zip(names, names[1:]) if t[a] and t[b]}
    out.append(dict(request=key, p_lane=p["lane"], d_lane=d["lane"], ttft_ms=c["ttft_ms"],
                    p_side_ms=(t["p_update"] - t["proxy_arrival"]) * 1000,
                    remainder_ms=c["ttft_ms"] - (t["p_update"] - t["proxy_arrival"]) * 1000, parts=parts))

print("joined", len(out), "of prefill", len(prefill), "decode", len(decode))
summary = {}
for key in ("ttft_ms", "p_side_ms", "remainder_ms"):
    values = [o[key] for o in out]
    summary[key] = (statistics.mean(values), min(values), max(values))
    print(f"{key:40s} mean {statistics.mean(values):8.1f} min {min(values):8.1f} max {max(values):8.1f}")
for part in out[0]["parts"]:
    values = [o["parts"][part] for o in out if part in o["parts"]]
    summary[part] = (statistics.mean(values), min(values), max(values), statistics.median(values))
    print(f"{part:40s} mean {statistics.mean(values):8.1f} min {min(values):8.1f} max {max(values):8.1f} median {statistics.median(values):8.1f} n {len(values)}")
json.dump(dict(summary=summary, rows=out), open(sys.argv[2], "w"), indent=1)

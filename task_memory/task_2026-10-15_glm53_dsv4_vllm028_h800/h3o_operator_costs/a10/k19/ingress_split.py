"""Split the native prefill ingress: proxy arrival (proxy host) -> API-server arrival_time
(prefill host) -> add_enqueue (if recorded) -> engine add (prefill host).
Usage: split.py <artifacts>"""
import json, statistics, sys
from pathlib import Path

run = Path(sys.argv[1])
rows = lambda p: [json.loads(l) for l in open(p) if l.strip()]
proxy = {r["request_id"]: r for r in rows(run / "run/proxy_requests.jsonl")}
spans = {"proxy_arrival->api_arrival": [], "api_arrival->add_enqueue": [], "add_enqueue->add": [],
         "api_arrival->add": [], "proxy_arrival->add": []}
events = {}
for path in sorted(run.glob("nodes/*/scheduler_lifecycle/*.jsonl")):
    records = rows(path)
    start = next(r for r in records if r["event"] == "start")
    if start["role"] != "prefill":
        continue
    enqueue = {}
    for r in records:
        events[r["event"]] = events.get(r["event"], 0) + 1
        if r["event"] == "add_enqueue" and "formal" in r.get("request_id", ""):
            enqueue[r["request_id"]] = r["t"]
        if r["event"] == "add" and "formal" in r["request_id"]:
            key = r["request_id"].split("cmpl-", 1)[1].rsplit("-0-", 1)[0]
            x = proxy.get(key)
            if x is None:
                continue
            spans["proxy_arrival->api_arrival"].append((r["arrival_time"] - x["proxy_arrival_time"]) * 1e3)
            spans["api_arrival->add"].append((r["t"] - r["arrival_time"]) * 1e3)
            spans["proxy_arrival->add"].append((r["t"] - x["proxy_arrival_time"]) * 1e3)
            if r["request_id"] in enqueue:
                spans["api_arrival->add_enqueue"].append((enqueue[r["request_id"]] - r["arrival_time"]) * 1e3)
                spans["add_enqueue->add"].append((r["t"] - enqueue[r["request_id"]]) * 1e3)
print("prefill lifecycle events", events)
for k, v in spans.items():
    if v:
        print(f"{k:32s} n {len(v):4d} mean {statistics.mean(v):7.2f} median {statistics.median(v):7.2f} "
              f"min {min(v):7.2f} max {max(v):7.2f}")

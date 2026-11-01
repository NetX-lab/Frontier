"""(prefill lane, decode lane) pair coverage by warmup, per lifecycle run and repeat (read-only).

A pair's first KV pull pays the lazy NIXL handshake (vLLM nixl/pull_worker.py:65-70,
base_worker.py:870-915). Lists, per repeat, pairs first opened by a formal request and
that request's decode add -> receive-done wait.
"""
import collections, json, re, sys
from pathlib import Path

for tag in sys.argv[1:]:
    run = Path("/data/ycfeng/tmp/glm53_dsv4/h200/downloads") / tag / "artifacts"
    p_lane, d_add, d_recv = {}, {}, {}
    for path in sorted(run.glob("nodes/*/scheduler_lifecycle/*.jsonl")):
        records = [json.loads(l) for l in open(path) if l.strip()]
        start = next(r for r in records if r["event"] == "start")
        for r in records:
            if r["event"] == "add":
                if start["role"] == "prefill":
                    p_lane[r["request_id"]] = start["dp_rank"]
                else:
                    d_add[r["request_id"]] = (r["t"], start["dp_rank"], r.get("remote_request_id"))
            elif r["event"] == "finished_recving" and start["role"] == "decode":
                d_recv[r["request_id"]] = r["t"]
    seen, firsts = set(), []
    lanes_p, lanes_d = set(), set()
    for rid, (t, lane, remote) in sorted(d_add.items(), key=lambda kv: kv[1][0]):
        pair = (p_lane.get(remote), lane)
        lanes_p.add(pair[0]); lanes_d.add(lane)
        if pair not in seen:
            seen.add(pair)
            key = rid.split("cmpl-", 1)[1].rsplit("-0-", 1)[0]
            wait = (d_recv[rid] - t) * 1000 if rid in d_recv else None
            firsts.append((key, pair, wait))
    warm = collections.Counter(re.search(r"(warmup|formal).*(repeat\d+)", k).groups() for k, _, _ in firsts)
    n_warmup = collections.Counter(re.search(r"(warmup|formal).*(repeat\d+)", k.split("cmpl-", 1)[1]).groups()
                                   for k in d_add if "cmpl-" in k)
    print(f"== {tag}: P lanes {sorted(lanes_p, key=str)} D lanes {sorted(lanes_d)} pairs used {len(seen)}")
    print("   decode adds by (kind, repeat):", dict(sorted(n_warmup.items())))
    print("   pairs opened by (kind, repeat):", dict(sorted(warm.items())))
    for key, pair, wait in firsts:
        if "formal" in key:
            print(f"   formal opener {key} pair {pair} add->recv_done {wait:.1f} ms" if wait is not None else f"   formal opener {key} pair {pair}")

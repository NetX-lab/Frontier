"""Proxy-side split of the slow native P2-64 requests (lead scratch).

    python handoff_slow.py <proxy_requests.jsonl> <client_requests.jsonl>

For formal requests with client TTFT above 1,000 ms: the P side (proxy arrival
to P response) and the handoff (P response to the first D byte), against the
medians of the other formal requests of the same repeat.
"""
import collections
import json
import statistics
import sys

proxy = {r["request_id"]: r for r in (json.loads(line) for line in open(sys.argv[1]) if line.strip())}
client = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
by_repeat = collections.defaultdict(list)
for c in client:
    if "formal" not in c["request_id"]:
        continue
    p = proxy[c["request_id"]]
    p_side = (p["prefill_response_monotonic"] - p["proxy_arrival_monotonic"]) * 1000
    handoff = (p["decode_first_byte_monotonic"] - p["prefill_response_monotonic"]) * 1000
    by_repeat[c["repeat"]].append((c["request_id"], c["ttft_ms"], p_side, handoff,
                                   p["kv_transfer_params"]["remote_engine_id"].rsplit("_", 1)[1]))
for repeat, rows in sorted(by_repeat.items()):
    fast = [r for r in rows if r[1] <= 1000]
    print(f"repeat {repeat}: others n {len(fast)} p_side median {statistics.median(r[2] for r in fast):.1f} "
          f"handoff median {statistics.median(r[3] for r in fast):.1f} mean {statistics.mean(r[3] for r in fast):.1f}")
    for r in rows:
        if r[1] > 1000:
            print(f"  slow {r[0]} ttft {r[1]:.1f} p_side {r[2]:.1f} handoff {r[3]:.1f} prefill lane {r[4]}")

"""Proxy-clock split of the native reference TTFT, without first-contact requests (lead scratch).

    python reference_split.py <run dir>

Per formal request whose handoff is at most HANDOFF_LIMIT_MS
(reference_without_handshake.py): client dispatch -> proxy arrival, proxy
arrival -> P response (P side), P response -> first D byte (handoff), first D
byte -> client first token. The client and proxy run on one host
(monotonic clocks).
"""
import json
import statistics
import sys
from pathlib import Path

from reference_without_handshake import HANDOFF_LIMIT_MS

run = Path(sys.argv[1])
proxy = {r["request_id"]: r for r in (json.loads(l) for l in open(run / "proxy_requests.jsonl") if l.strip())}
parts = {"client dispatch -> proxy arrival": [], "P side": [], "handoff": [], "first D byte -> client": [], "TTFT": []}
for c in (json.loads(l) for l in open(run / "client_requests.jsonl") if l.strip()):
    if "formal" not in c["request_id"]:
        continue
    p = proxy[c["request_id"]]
    handoff = (p["decode_first_byte_monotonic"] - p["prefill_response_monotonic"]) * 1000
    if handoff > HANDOFF_LIMIT_MS:
        continue
    parts["client dispatch -> proxy arrival"].append((p["proxy_arrival_monotonic"] - c["dispatch_monotonic"]) * 1000)
    parts["P side"].append((p["prefill_response_monotonic"] - p["proxy_arrival_monotonic"]) * 1000)
    parts["handoff"].append(handoff)
    parts["first D byte -> client"].append((c["first_token_monotonic"] - p["decode_first_byte_monotonic"]) * 1000)
    parts["TTFT"].append(c["ttft_ms"])
for name, values in parts.items():
    print(f"{name:34s} n {len(values)} mean {statistics.mean(values):7.1f} median {statistics.median(values):7.1f}")

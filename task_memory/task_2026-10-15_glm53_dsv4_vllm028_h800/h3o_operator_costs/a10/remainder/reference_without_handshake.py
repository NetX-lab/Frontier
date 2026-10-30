"""Native P2-64 reference TTFT with and without first-contact requests (lead scratch).

    python reference_without_handshake.py <run dir> [<first_contact.json>]

Without lifecycle records, a first-contact request is one whose proxy handoff
(P response to the first D byte) exceeds HANDOFF_LIMIT_MS; the wave and
lifecycle runs show those handoffs only for the requests that open a (prefill
lane, decode lane) pair (first_contact.json, pairs.py). With first_contact.json,
the two sets are compared.
"""
import json
import statistics
import sys
from pathlib import Path

HANDOFF_LIMIT_MS = 1000.0


def main():
    run = Path(sys.argv[1])
    proxy = {r["request_id"]: r for r in (json.loads(l) for l in open(run / "proxy_requests.jsonl") if l.strip())}
    client = [json.loads(l) for l in open(run / "client_requests.jsonl") if l.strip()]
    formal = [c for c in client if "formal" in c["request_id"]]
    slow = {c["request_id"] for c in formal
            if (proxy[c["request_id"]]["decode_first_byte_monotonic"]
                - proxy[c["request_id"]]["prefill_response_monotonic"]) * 1000 > HANDOFF_LIMIT_MS}
    kept = [c["ttft_ms"] for c in formal if c["request_id"] not in slow]
    print(f"formal {len(formal)} mean {statistics.mean(c['ttft_ms'] for c in formal):.1f}; "
          f"handoff > {HANDOFF_LIMIT_MS:.0f} ms: {len(slow)} {sorted(slow)}; "
          f"mean without them {statistics.mean(kept):.1f} (n {len(kept)})")
    if len(sys.argv) > 2:
        first = set(json.load(open(sys.argv[2])))
        print("same set as the first-contact requests:", slow == first)


if __name__ == "__main__":
    main()

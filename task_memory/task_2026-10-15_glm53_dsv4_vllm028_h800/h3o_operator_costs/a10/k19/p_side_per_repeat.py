"""Native prefill side (proxy arrival -> prefill response, proxy clock) per repeat, without first-contact requests.

    python p_side_per_repeat.py <downloads dir> <run tag> [<run tag> ...]

Run with a10/remainder on PYTHONPATH (imports HANDOFF_LIMIT_MS from reference_without_handshake.py).
"""
import json
import re
import statistics
import sys
from pathlib import Path

from reference_without_handshake import HANDOFF_LIMIT_MS

downloads = Path(sys.argv[1])
for tag in sys.argv[2:]:
    run = downloads / tag / "artifacts" / "run"
    proxy = {r["request_id"]: r for r in (json.loads(l) for l in open(run / "proxy_requests.jsonl") if l.strip())}
    by_repeat = {}
    for c in (json.loads(l) for l in open(run / "client_requests.jsonl") if l.strip()):
        if "formal" not in c["request_id"]:
            continue
        p = proxy[c["request_id"]]
        if (p["decode_first_byte_monotonic"] - p["prefill_response_monotonic"]) * 1000 > HANDOFF_LIMIT_MS:
            continue
        repeat = re.search(r"repeat\d+", c["request_id"])
        by_repeat.setdefault(repeat.group(0) if repeat else "single", []).append(
            (p["prefill_response_monotonic"] - p["proxy_arrival_monotonic"]) * 1000)
    print(tag, " ".join(f"{k}: n {len(v)} P side {statistics.mean(v):.1f}" for k, v in sorted(by_repeat.items())))

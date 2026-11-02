"""d split by D iterations, and the first request of each prefill-to-decode DP lane pair, for one lifecycle run."""
import glob
import json
import re
import statistics
import sys
from pathlib import Path

run_root, requests_path = Path(sys.argv[1]), Path(sys.argv[2])
requests = [json.loads(line) for line in open(requests_path)]
decoding = [r for r in requests if r["lane_state"] == "decoding"]
for title, rows in (("decoding lanes, all repetitions", decoding),
                    ("decoding lanes, repetitions 2-3", [r for r in decoding if r["repeat"] > 1])):
    print(title)
    by_iterations = {}
    for r in rows:
        by_iterations.setdefault(r["d_iterations"], []).append(r)
    for k in sorted(by_iterations):
        group = by_iterations[k]
        d = statistics.mean(r["d"] for r in group)
        transfer_max = statistics.mean(r["read_transfer_max_ms"] for r in group)
        spread = statistics.mean(max(r["read_transfer_duration_ms"]) - min(r["read_transfer_duration_ms"]) for r in group)
        print(f"  d_iterations {k:3d} requests {len(group):4d} d {d:8.1f} transfer_max {transfer_max:6.1f} "
              f"d-transfer_max {d - transfer_max:7.1f} spread {spread:5.1f}")


def request_key(request_id: str) -> str:
    match = re.match(r"cmpl-(.*)-0-[0-9a-f]+$", request_id)
    return match.group(1) if match else request_id


adds = {"prefill": {}, "decode": {}}
for path in glob.glob(str(run_root / "nodes/*/scheduler_lifecycle/*.jsonl")):
    with open(path) as lines:
        header = json.loads(lines.readline())
        for line in lines:
            record = json.loads(line)
            if record["event"] == "add":
                adds[header["role"]][request_key(record["request_id"])] = (header["dp_rank"], record["t"])
d_by_request = {r["request_id"]: r for r in requests}
first = {}
for key, (d_rank, t) in sorted(adds["decode"].items(), key=lambda item: item[1][1]):
    pair = (adds["prefill"][key][0], d_rank)
    if pair not in first:
        first[pair] = key
        r = d_by_request.get(key)
        detail = f"d {r['d']:.1f} ms, d_iterations {r['d_iterations']}" if r else "warmup"
        print(f"first P{pair[0]}->D{pair[1]}: {key} ({detail})")
print(f"{len(first)} lane pairs")

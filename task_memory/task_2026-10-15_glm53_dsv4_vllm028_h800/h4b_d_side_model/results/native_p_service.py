"""Native P service (proxy arrival -> P response) per run and repetition, and P overlap between requests.

Usage: native_p_service.py <task directory>

For each persisted native run, prints the GPU hosts named in its persisted files and, per
repetition of the formal segment, the P service statistics, the smallest gap between consecutive
proxy arrivals, and the number of requests that reach the proxy while an earlier request of the
same repetition is still in P service. Times are in ms.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RUNS = (
    ("native06", "h4a_p0_diagnostic/inputs/native06"),
    ("H800l", "h4b_lifecycle_measurement/inputs/h800l"),
    ("profile02", "h4a_p0_diagnostic/inputs/profile02"),
    ("P2-32r", "h200_mainline/diagnostic/inputs/p2_32r"),
    ("P2-32l", "h200_mainline/diagnostic/inputs/p2_32l"),
    ("P2-32p", "h200_mainline/diagnostic/inputs/p2_32p"),
    ("P2-64r", "h200_mainline/diagnostic/inputs/p2_64r"),
    ("P2-64l", "h200_mainline/diagnostic/inputs/p2_64l"),
    ("P2-64p", "h200_mainline/diagnostic/inputs/p2_64p"),
)
HOST = re.compile(r"gpu-h\d+-\d{4}")
HOST_SOURCES = (".json", ".jsonl")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def hosts(run):
    names = set()
    for path in run.rglob("*"):
        if path.is_file() and (path.suffix in HOST_SOURCES or path.parent.name == "nodes"):
            names.update(HOST.findall(path.read_text(errors="ignore")))
        if path.is_dir() and path.parent.name == "nodes":
            names.update(HOST.findall(path.name))
    return sorted(names)


def main() -> None:
    root = Path(sys.argv[1])
    for name, relative in RUNS:
        run = root / relative
        client = {r["request_id"]: r for r in read_jsonl(run / "run/client_requests.jsonl") if r["segment"] == "formal"}
        proxy = {r["request_id"]: r for r in read_jsonl(run / "run/proxy_requests.jsonl")}
        repeats = defaultdict(list)
        for request_id, c in client.items():
            p = proxy[request_id]
            repeats[c["repeat"]].append((p["proxy_arrival_monotonic"], p["prefill_response_monotonic"]))
        print(f"== {name}: hosts {hosts(run)}")
        for repeat, rows in sorted(repeats.items()):
            rows.sort()
            service = [(end - start) * 1e3 for start, end in rows]
            gap = min(b[0] - a[0] for a, b in zip(rows, rows[1:])) * 1e3
            overlaps, latest_end = 0, rows[0][1]
            for start, end in rows[1:]:
                overlaps += start < latest_end
                latest_end = max(latest_end, end)
            print(f"  repetition {repeat}: n {len(rows)}  P service mean {statistics.fmean(service):.2f}  "
                  f"median {statistics.median(service):.2f}  min {min(service):.2f}  max {max(service):.2f}  "
                  f"arrival gap min {gap:.1f}  arrivals during an earlier P service {overlaps}")


if __name__ == "__main__":
    main()

"""READ wait of every D request of a lifecycle run, warmup included: READ schedule to the finished_recving update.

The first D request of each prefill-to-decode DP lane pair is a first contact. A later request on the same pair
whose READ is scheduled while the first contact is still pending is listed with it.
"""
import glob
import json
import re
import statistics
import sys


def request_key(request_id):
    match = re.match(r"cmpl-(.*)-0-[0-9a-f]+$", request_id)
    return match.group(1) if match else request_id


for root in sys.argv[1:]:
    adds = {"prefill": {}, "decode": {}}
    read_t, recv_t = {}, {}
    for path in glob.glob(root + "/nodes/*/scheduler_lifecycle/*.jsonl"):
        with open(path) as lines:
            header = json.loads(lines.readline())
            for line in lines:
                record = json.loads(line)
                if record["event"] == "add":
                    adds[header["role"]][request_key(record["request_id"])] = (header["dp_rank"], record["t"])
                elif header["role"] == "decode" and record["event"] == "schedule":
                    for request_id in record.get("recv", []):
                        read_t.setdefault(request_key(request_id), record["t"])
                elif header["role"] == "decode" and record["event"] == "update":
                    for request_id in record.get("finished_recving", []):
                        recv_t.setdefault(request_key(request_id), record["t"])
    pair = {key: (adds["prefill"][key][0], d_rank) for key, (d_rank, _) in adds["decode"].items()}
    first = {}
    for key, _ in sorted(adds["decode"].items(), key=lambda item: item[1][1]):
        first.setdefault(pair[key], key)
    wait = {key: (recv_t[key] - read_t[key]) * 1e3 for key in adds["decode"]}
    behind = {key for key in wait if key not in first.values()
              and read_t[first[pair[key]]] <= read_t[key] < recv_t[first[pair[key]]]}
    print(f"{root}: {len(wait)} D requests, {len(first)} lane pairs")
    for p in sorted(first):
        key = first[p]
        followers = sorted(k for k in behind if pair[k] == p)
        same_update = all(recv_t[k] == recv_t[key] for k in followers)
        print(f"  first P{p[0]}->D{p[1]} {key} wait {wait[key]:.1f} ms; "
              f"behind it {[(k, round(wait[k], 1)) for k in followers]} same update {same_update}")
    rest = [w for key, w in wait.items() if key not in first.values() and key not in behind]
    print(f"  other D requests {len(rest)}: mean {statistics.mean(rest):.1f} p50 {statistics.median(rest):.1f} "
          f"max {max(rest):.1f} ms")

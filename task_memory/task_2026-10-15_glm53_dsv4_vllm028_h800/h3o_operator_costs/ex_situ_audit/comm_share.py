"""Share of the collective scopes in each forward class of an A10 run.

Usage: comm_share.py <run root>...; reads <run root>/forwards.jsonl.gz.
"""
import gzip
import json
import sys
from collections import defaultdict

COMM = ("dsv4_ep_", "dsv4_sp_", "allreduce", "all_gather", "allgather", "reduce_scatter")


def forward_class(record):
    if record["role"] == "prefill":
        if not record["real"]:
            return "prefill_dummy"
        return "prefill_6144" if record["local_tokens"] == 6144 else "prefill_other"
    return "decode_real" if record["real"] else "decode_dummy"


for root in sys.argv[1:]:
    totals = defaultdict(lambda: [0, 0.0, 0.0, defaultdict(float)])
    with gzip.open(f"{root}/forwards.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("phase") != "serving":
                continue
            entry = totals[forward_class(record)]
            comm = {k: v for k, v in record.get("scope_ms", {}).items() if any(t in k for t in COMM)}
            entry[0] += 1
            entry[1] += record["composed_ms"]
            entry[2] += sum(comm.values())
            for key, value in comm.items():
                entry[3][key] += value
    print("==", root)
    for name, (count, composed, comm, scopes) in sorted(totals.items()):
        print(f"{name}: n={count} mean_composed_ms={composed / count:.3f} "
              f"mean_comm_scope_ms={comm / count:.3f} share={comm / composed:.3%}")
        for key, value in sorted(scopes.items(), key=lambda item: -item[1])[:6]:
            print(f"    {key}: {value / count:.3f} ms/forward")

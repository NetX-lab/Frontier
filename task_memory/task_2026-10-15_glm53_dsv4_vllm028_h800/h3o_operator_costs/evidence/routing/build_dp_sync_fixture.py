#!/usr/bin/env python3
"""Write the distinct native DP synchronization tuples of the retained counters.

  python build_dp_sync_fixture.py <output.json> <run artifacts>...

For every serving and dummy counter row of every PREFILL and DECODE node, the
tuple is the DP group's post-all-reduce vectors (local token counts, desired
modes, uniform token counts with 0 for none, max query lengths), the recording
rank's DP rank and own request count, its dispatched descriptor and
`num_tokens_across_dp`. Each node's rank 0 records the forwards of its DP lane.
Every node must share one capture-size list and one largest FULL request count.
"""

import glob
import json
import sys

capture_sizes = max_full_reqs = None
tuples = {}
for base in sys.argv[2:]:
    run = base.rstrip("/").split("/")[-2 if base.rstrip("/").endswith("artifacts") else -1]
    for path in sorted(glob.glob(base + "/nodes/*/serving/rank0.json")):
        data = json.load(open(path))
        graph = data["inventory"]["effective_graph"]
        dp_rank = data["inventory"]["groups"]["dp"]["rank_in_group"]
        full_reqs = max(d["num_reqs"] for d in graph["captured_descriptors"] if d["cg_mode"] == "FULL")
        if capture_sizes is None:
            capture_sizes, max_full_reqs = graph["capture_sizes"], full_reqs
        if (graph["capture_sizes"], full_reqs) != (capture_sizes, max_full_reqs):
            raise SystemExit(f"{path}: capture sizes or FULL request bound differ")
        for row in data["counters"]["rows"]:
            if row.get("kind") not in ("serving", "dummy"):
                continue
            evidence = row["dispatch_evidence"]
            dispatch = evidence["dispatch"]
            descriptor = dispatch["descriptor"]
            local, desired, uniform, max_query = evidence["dp_post_all_reduce"]
            key = json.dumps([local, desired, uniform, max_query, dp_rank, dispatch["inputs"]["num_reqs"],
                              descriptor["cg_mode"], descriptor["num_tokens"], descriptor["num_reqs"],
                              descriptor["uniform_token_count"], dispatch["num_tokens_across_dp"]])
            entry = tuples.setdefault(key, {"forwards": 0, "sources": set()})
            entry["forwards"] += row.get("successful_forwards") or 0
            entry["sources"].add(f"{run}/{data['role']}/{row['kind']}")

fields = ["local_num_tokens", "desired_modes", "uniform_token_counts", "max_query_lens",
          "recording_dp_rank", "recording_num_reqs", "descriptor_mode", "descriptor_num_tokens", "descriptor_num_reqs",
          "descriptor_uniform_token_count", "num_tokens_across_dp", "forwards", "sources"]
source_names = sorted({source for entry in tuples.values() for source in entry["sources"]})
rows = [json.loads(key) + [entry["forwards"], sorted(source_names.index(s) for s in entry["sources"])]
        for key, entry in sorted(tuples.items())]
with open(sys.argv[1], "w") as handle:
    json.dump({"capture_sizes": capture_sizes, "max_full_num_reqs": max_full_reqs,
               "mode_order": ["NONE", "PIECEWISE", "FULL"], "source_names": source_names,
               "fields": fields, "rows": rows}, handle, separators=(",", ":"))
    handle.write("\n")
print(f"{len(rows)} distinct tuples, {sum(row[-2] for row in rows)} forwards")

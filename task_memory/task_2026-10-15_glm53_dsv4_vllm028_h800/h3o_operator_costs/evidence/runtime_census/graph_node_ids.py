#!/usr/bin/env python3
"""Graph node identity of replayed CUDA-graph kernels in one native torch-profiler trace.

  python graph_node_ids.py <trace.pt.trace.json.gz>

For the three graphs with the most replayed kernels: the number of distinct
`graph node id` values, their local index range (the id minus `graph id << 32`),
and, over the first replay, the start-order inversions of the local index and
the streams the driver placed the kernels on.
"""

import collections
import gzip
import json
import sys

with gzip.open(sys.argv[1], "rt") as handle:
    events = json.load(handle)["traceEvents"]
by_graph = collections.defaultdict(list)
for event in events:
    args = event.get("args") or {}
    if event.get("ph") == "X" and args.get("graph id"):
        by_graph[args["graph id"]].append(event)
for graph_id, kernels in sorted(by_graph.items(), key=lambda item: -len(item[1]))[:3]:
    local = sorted({kernel["args"]["graph node id"] - (graph_id << 32) for kernel in kernels})
    print(f"graph {graph_id}: {len(kernels)} replayed kernels, {len(local)} distinct nodes, "
          f"local index {local[0]}-{local[-1]}")
    first = sorted(kernels, key=lambda kernel: kernel["ts"])[: len(local)]
    order = [kernel["args"]["graph node id"] - (graph_id << 32) for kernel in first]
    print(f"  first replay: {sum(b < a for a, b in zip(order, order[1:]))} start-order inversions; "
          f"categories {dict(collections.Counter(kernel['cat'] for kernel in first))}; "
          f"streams {collections.Counter(kernel['tid'] for kernel in first).most_common(4)}")

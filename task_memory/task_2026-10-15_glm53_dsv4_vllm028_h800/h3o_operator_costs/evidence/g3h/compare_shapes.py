"""Compare G3h routed-MoE and gate kernel shapes with the native forwards.

Usage: compare_shapes.py <native json> <g3h profiler_traces dir>
A shape is (kernel name, grid). Native shapes come from the dispatch segment
of every forward class; G3h shapes from the gating and grouped-GEMM traces.
"""

import collections
import json
import sys
from pathlib import Path

native = json.load(open(sys.argv[1]))
native_shapes = collections.defaultdict(set)
for key, entry in native.items():
    for kernel in entry["kernels"]:
        native_shapes[(kernel["name"], tuple(kernel["grid"]))].add(key)

g3h_shapes = collections.defaultdict(set)
for path in sorted(Path(sys.argv[2]).glob("*.json")):
    events = json.load(open(path))["traceEvents"]
    scopes = {e["name"] for e in events if e.get("cat") == "user_annotation"}
    if "vidur_moe_shuffling" in scopes:
        continue
    for event in events:
        if event.get("cat") == "kernel":
            g3h_shapes[(event["name"], tuple(event["args"]["grid"]))].add(path.name)


def label(name):
    return name[:80]


print("native shapes not in G3h:")
for shape in sorted(set(native_shapes) - set(g3h_shapes)):
    print(f"  grid={list(shape[1])} {label(shape[0])}  ({len(native_shapes[shape])} classes)")
print("G3h shapes not in native:")
for shape in sorted(set(g3h_shapes) - set(native_shapes)):
    print(f"  grid={list(shape[1])} {label(shape[0])}  ({len(g3h_shapes[shape])} traces)")
print("shared shapes:", len(set(native_shapes) & set(g3h_shapes)))

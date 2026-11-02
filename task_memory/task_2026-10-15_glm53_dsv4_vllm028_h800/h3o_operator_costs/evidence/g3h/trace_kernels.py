"""Group G3h RECORD_FUNCTION traces by annotation and kernel multiset.

Usage: trace_kernels.py <profiler_traces dir> ...
"""

import collections
import json
import sys
from pathlib import Path


def short(name):
    return name if len(name) <= 90 else name[:90] + "..."


for directory in sys.argv[1:]:
    groups = collections.defaultdict(list)
    for path in sorted(Path(directory).glob("*.json")):
        events = json.load(open(path))["traceEvents"]
        annotations = collections.Counter(
            e["name"] for e in events if e.get("cat") == "user_annotation"
        )
        kernels = collections.Counter(
            (short(e["name"]), tuple(e.get("args", {}).get("grid", ())))
            for e in events if e.get("cat") == "kernel"
        )
        key = (tuple(sorted(annotations.items())), tuple(sorted(kernels.items())))
        groups[key].append(path.name)
    print("==", directory, "distinct trace shapes:", len(groups))
    for (annotations, kernels), names in sorted(groups.items(), key=lambda item: item[0][0]):
        print(f"-- {len(names)} traces, annotations {dict(annotations)}")
        for (name, grid), count in kernels:
            print(f"   {count:4d} grid={list(grid)} {name}")

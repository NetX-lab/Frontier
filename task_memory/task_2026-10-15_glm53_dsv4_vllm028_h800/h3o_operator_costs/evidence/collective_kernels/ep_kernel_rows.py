"""Print the lane-0 NCCL kernel duration rows of comparator summaries.

Usage: ep_kernel_rows.py <summary.txt> ...
"""

import ast
import re
import sys

PATTERN = re.compile(r"DURATION (\w+) lane=(\d+) (\S+).* grid=(\[[^\]]*\]).* native_us=(\{[^}]*\}) benchmark_us=(\{[^}]*\})")
for path in sys.argv[1:]:
    print("==", path)
    group = None
    for line in open(path):
        if line.startswith("group "):
            group = line.split()[1:4]
            continue
        match = PATTERN.search(line)
        if not match or group is None or match.group(2) != "0":
            continue
        verdict, _, name, grid, native, bench = match.groups()
        if not name.startswith("ncclDevKernel"):
            continue
        native, bench = ast.literal_eval(native), ast.literal_eval(bench)
        print(f"group {group[0]} {group[2]:9s} {name[13:40]:28s} grid={grid:9s} {verdict:6s} "
              f"native min/med {native['min']:7.1f}/{native['median']:7.1f}  bench min/med {bench['min']:7.1f}/{bench['median']:7.1f}  "
              f"d_med {bench['median'] - native['median']:6.1f}")

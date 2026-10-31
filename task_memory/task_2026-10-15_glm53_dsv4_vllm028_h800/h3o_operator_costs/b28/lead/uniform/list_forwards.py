"""List each native prefill forward's class and scheduled tokens on rank 0 of each DP lane."""
import sys
from pathlib import Path

from v4_runtime_census import dispatch_kernel_multisets

lanes = []
for d in sys.argv[1:]:
    path = sorted(Path(d).glob("*rank0*.pt.trace.json.gz")) or sorted(Path(d).glob("*.pt.trace.json.gz"))[:1]
    lanes.append({f.index: (f.step_class, f.scheduled_tokens, f.mode) for f in dispatch_kernel_multisets("prefill", path[0])})
indices = sorted(set().union(*lanes))
for i in indices:
    print(i, [lane.get(i) for lane in lanes])

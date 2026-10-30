"""Kernels in the post segment before, and the segments of, the native real prefill forward."""
import collections
import sys
from pathlib import Path

from v4_runtime_census import forward_kernel_events, union_us

path = Path(sys.argv[1])
previous = None
for fw in forward_kernel_events("prefill", path):
    r = fw.record
    if r["real"] and r["mode"] == "NONE":
        print("previous forward", previous.record["index"], previous.record["mode"], previous.record["class"])
        for seg in ("dispatch", "post"):
            ks = previous.kernels_by_segment[seg]
            if ks:
                print(f"  previous {seg}: last kernel end {max(k['ts'] + k['dur'] for k in ks):.0f}")
        post = previous.kernels_by_segment["post"]
        by = collections.Counter()
        for k in post:
            by[k["name"][:110]] += k["dur"]
        print("  previous post kernels", len(post), "sum us", sum(by.values()))
        for n, us in by.most_common(10):
            print(f"    {us:9.1f} {n}")
        for seg in ("prepare", "dispatch", "post"):
            ks = fw.kernels_by_segment[seg]
            if ks:
                print(f"  {seg}: first {min(k['ts'] for k in ks):.0f} last end {max(k['ts'] + k['dur'] for k in ks):.0f} n {len(ks)}")
        print("  host bounds", {k: (round(a), round(b)) for k, (a, b) in fw.segment_bounds.items()})
        print("  host bounds previous", {k: (round(a), round(b)) for k, (a, b) in previous.segment_bounds.items()})
    previous = fw

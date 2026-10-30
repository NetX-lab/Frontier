"""Group the device kernels of native NONE prefill forwards by kernel name.

    python native_kernels.py <trace.json.gz> [role]
"""
import collections
import sys
from pathlib import Path

from v4_runtime_census import forward_kernel_events, union_us

path = Path(sys.argv[1])
role = sys.argv[2] if len(sys.argv) > 2 else "prefill"
for fw in forward_kernel_events(role, path):
    r = fw.record
    if r["mode"] != "NONE":
        continue
    print("forward", r["index"], r["mode"], r["class"], r.get("step"), "real", r["real"])
    for seg in ("prepare", "dispatch", "post"):
        ks = fw.kernels_by_segment[seg]
        if not ks:
            continue
        spans = [(k["ts"], k["ts"] + k["dur"]) for k in ks]
        print(f"  {seg}: kernels {len(ks)} sum {sum(k['dur'] for k in ks)/1e3:.3f} ms union {union_us(spans)/1e3:.3f} ms "
              f"span {(max(s for _, s in spans) - min(s for s, _ in spans))/1e3:.3f} ms streams {collections.Counter(k['args'].get('stream') for k in ks)}")
    by = collections.defaultdict(lambda: [0, 0.0])
    for k in fw.kernels_by_segment["dispatch"]:
        by[k["name"]][0] += 1
        by[k["name"]][1] += k["dur"]
    for name, (n, us) in sorted(by.items(), key=lambda x: -x[1][1])[:60]:
        print(f"    {n:6d} {us/1e3:9.3f}  {name[:150]}")

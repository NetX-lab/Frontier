"""Price the P2-32 and P2-64 prefill EP calls with the current table and with the planned rule.

The planned rule (G5h report, revised decision of 2026-10-30) is mocked here only to get the
expected effect: EP rows of eager scenarios with a 6,144-token lane come from the isolated runs,
and an uneven dispatch reads its own measured AllGatherV row (median over passes).
"""
import csv
import sys
from statistics import median

from frontier.cc_backend import MeasuredCCBackend, MeasuredCCBackendConfig
from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths
from frontier.operators.spec import CollectiveImplementation
from frontier.types import ClusterType

ROOT = sys.argv[1]
ISO = {16: "/data/ycfeng/tmp/glm53_dsv4/h200/g5h16_iso_01/download/summary.csv",
       32: "/data/ycfeng/tmp/glm53_dsv4/h200/g5h32_iso_01/download/summary.csv"}
DIM0 = CollectiveImplementation.GROUP_COORDINATOR_DIM0
LAYERS = 43
backend = MeasuredCCBackend(MeasuredCCBackendConfig(summary_csv_path=f"{ROOT}/data/profiling/network/h200_dgx/summary.csv"),
                            ClusterType.MONOLITHIC, "h200", "h200_dgx", 32)


def iso_median(path, scenario, call):
    with open(path, newline="") as stream:
        values = [float(r["median_us"]) for r in csv.DictReader(stream) if r["scenario"] == scenario and r["call"] == call]
    assert len(values) == 3, (scenario, call, values)
    return median(values)


for name, lanes in (("P2-32", (6144, 1)), ("P2-64", (6144, 1, 1, 1))):
    ep = 8 * len(lanes)
    widths = sequence_parallel_moe_token_widths(lanes, 8)
    dispatch_bytes = max(widths.local_tokens) * 8264
    combine_bytes = widths.gathered_tokens * 8192
    old_d = backend.predict_operator_collective("allgather", DIM0, dispatch_bytes, ep, "MOE_EP") * 1e3
    old_c = backend.predict_operator_collective("reduce_scatter", DIM0, combine_bytes, ep, "MOE_EP", widths.local_tokens) * 1e3
    scenario = "eager_t" + "_".join(map(str, lanes))
    new_d = iso_median(ISO[ep], scenario, "moe_dispatch_all_gatherv")
    new_c = iso_median(ISO[ep], scenario, "moe_combine_reduce_scatterv")
    dd, dc = (new_d - old_d) * LAYERS / 1e3, (new_c - old_c) * LAYERS / 1e3
    print(f"{name} EP{ep} dispatch bytes {dispatch_bytes}: current {old_d:.1f} us (uniform curve), planned {new_d:.1f} us ({scenario} isolated)")
    print(f"{name} EP{ep} combine bytes {combine_bytes}: current {old_c:.1f} us (exact back-to-back row), planned {new_c:.1f} us ({scenario} isolated)")
    print(f"{name} per prefill forward of {LAYERS} layers: dispatch {dd:+.2f} ms, combine {dc:+.2f} ms, total {dd + dc:+.2f} ms")

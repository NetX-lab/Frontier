"""Held-out EP collective kernel counts per native forward (D-WARM-TWO-FIX).

The accounting finding used DP0 TP0 prefill forwards only. This reads TP0 of
DP1 prefill and decode traces, which that finding did not use, and counts per
forward every collective kernel family the SP and EP paths can launch.

vLLM all_gatherv (cuda_communicator.py:604-658) drops `sizes` when every rank
has the same row count and then issues plain all_gather calls inside one NCCL
group; otherwise it issues one AllGatherV per tensor. So a forward whose DP
lanes are unequal shows AllGatherV kernels and EP Reduce kernels, and an
equal-lane forward shows only AllGather and ReduceScatter kernels.

Usage: heldout_counts.py <output json>
"""

import collections
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
sys.path.insert(0, str(REPO / "tests/analysis"))
from v4_runtime_census import forward_kernel_events  # noqa: E402

DOWNLOADS = Path("/data/ycfeng/tmp/glm53_dsv4/h200/downloads")
RUNS = {"p2_64": "dsv4-h200-p2-64-profile-20261025-01", "p2_32": "dsv4-h200-p2-profile-20261025-01"}
MOE_LAYERS = 43
FAMILIES = {
    "allgatherv": "ncclDevKernel_AllGatherV_",
    "reduce": "ncclDevKernel_Reduce_Sum_",
    "allgather": "ncclDevKernel_AllGather_",
    "reduce_scatter": "ncclDevKernel_ReduceScatter_",
    "lamport_all_gather": "void vllm::mnnvl_lamport_all_gather",
    "lamport_reduce_scatter": "void vllm::mnnvl_lamport_reduce_scatter",
}


def family_counts(names):
    return {family: sum(name.startswith(prefix) for name in names) for family, prefix in FAMILIES.items()}


def ep_kernels(record, counts):
    """Split the collective kernels into SP, logits and EP calls.

    Per forward, SP gathers 43 attention inputs plus the final MHC state and
    reduce-scatters 43 attention outputs; each call is a lamport kernel or an
    NCCL kernel depending on size. A real forward also gathers the logits.
    The remaining AllGather/AllGatherV kernels are EP dispatch and the
    remaining ReduceScatter/Reduce kernels are EP combine.
    """
    sp_nccl_gathers = MOE_LAYERS + 1 - counts["lamport_all_gather"]
    sp_nccl_scatters = MOE_LAYERS - counts["lamport_reduce_scatter"]
    logits = 1 if record.get("real") else 0
    return dict(
        dispatch_allgather=counts["allgather"] - sp_nccl_gathers - logits,
        dispatch_allgatherv=counts["allgatherv"],
        combine=counts["reduce_scatter"] - sp_nccl_scatters + counts["reduce"],
    )


def ep_dispatch_kernels_per_layer(ep):
    """Dispatch kernels per MoE layer, or None when the counts do not fit one path."""
    if ep["combine"] != MOE_LAYERS:
        return None
    if ep["dispatch_allgatherv"] == 3 * MOE_LAYERS and ep["dispatch_allgather"] == 0:
        return 3
    if ep["dispatch_allgatherv"] == 0 and ep["dispatch_allgather"] == MOE_LAYERS:
        return 1
    return None


def count_forwards(role, trace):
    classes = collections.defaultdict(collections.Counter)
    for forward in forward_kernel_events(role, trace):
        record = forward.record
        names = [event["name"] for events in forward.kernels_by_segment.values() for event in events]
        counts = family_counts(names)
        lanes = "unequal" if counts["allgatherv"] else "equal"
        key = f"{record['mode']}|{record.get('class')}|real={record.get('real')}|lanes={lanes}"
        classes[key]["forwards"] += 1
        ep = ep_kernels(record, counts)
        classes[key][json.dumps(dict(kernels=counts, ep=ep), sort_keys=True)] += 1
        classes[key][f"dispatch_per_layer={ep_dispatch_kernels_per_layer(ep)}"] += 1
    return {key: dict(value) for key, value in sorted(classes.items())}


def main(output):
    result = {}
    for case, run in RUNS.items():
        for role in ("prefill", "decode"):
            trace = next((DOWNLOADS / run / "artifacts/torch_traces" / role / "dp1").glob("dp1_*_tp0_*.json.gz"))
            classes = count_forwards(role, trace)
            result[f"{case}/{role}/dp1/tp0"] = dict(trace=str(trace), classes=classes)
            print(case, role, trace.name)
            for key, value in classes.items():
                print("  ", key, value)
    checked = [(key, value) for item in result.values() for key, value in item["classes"].items()]
    explained = all(value.get("dispatch_per_layer=None", 0) == 0 for _, value in checked)
    rule_holds = all(
        value.get(f"dispatch_per_layer={3 if 'lanes=unequal' in key else 1}", 0) == value["forwards"]
        for key, value in checked
    )
    status = "PASS" if checked and explained and rule_holds else "FAIL"
    document = dict(
        status=status, moe_layers=MOE_LAYERS,
        rule="EP dispatch launches 3 kernels per layer when DP lanes are unequal (AllGatherV) and 1 when they are equal",
        runs=result,
    )
    Path(output).write_text(json.dumps(document, indent=1) + "\n")
    print(status)


if __name__ == "__main__":
    main(sys.argv[1])

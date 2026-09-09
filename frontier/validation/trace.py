"""Associate synchronized batch markers with kernels in a Kineto trace."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from frontier.validation.batch_record import read_batches


def interval_union_us(events: list[dict]) -> float:
    """GPU busy time: overlaps count once, unlike the sum of kernel durations."""
    end = float("-inf")
    total = 0.0
    for event in sorted(events, key=lambda e: float(e["ts"])):
        start = float(event["ts"])
        stop = start + float(event["dur"])
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def extract_batch_kernels(trace: dict) -> dict[str, list[dict]]:
    """Use synchronized CPU marker intervals; tolerate uncorrelated HIP graphs.

    A trace must contain exactly one GPU. Separate rank files preserve rank
    identity; summing kernels from every GPU is never a latency estimate.
    """
    kernels, markers = [], []
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        is_marker = event.get("cat") == "user_annotation" and name.startswith("frontier.batch:")
        if event.get("cat") == "kernel" or is_marker:
            ts, dur = float(event["ts"]), float(event["dur"])
            if not math.isfinite(ts) or not math.isfinite(dur) or dur < 0:
                raise ValueError("Trace event has invalid timestamp/duration")
            (kernels if event.get("cat") == "kernel" else markers).append(event)
    devices = {e.get("args", {}).get("device", e.get("pid")) for e in kernels}
    if len(devices) != 1:
        raise ValueError("Batch trace must contain kernels from exactly one GPU")
    if not markers:
        raise ValueError("No frontier.batch markers found")
    result = {}
    previous_end = float("-inf")
    for marker in sorted(markers, key=lambda e: float(e["ts"])):
        name = marker["name"].removeprefix("frontier.batch:")
        start, stop = float(marker["ts"]), float(marker["ts"]) + float(marker["dur"])
        if start < previous_end or name in result:
            raise ValueError("Batch markers overlap or duplicate an ID")
        previous_end = stop
        selected = [e for e in kernels if start <= float(e["ts"]) and
                    float(e["ts"]) + float(e["dur"]) <= stop]
        if not selected:
            raise ValueError(f"Batch {name} has no kernels; profiler boundary may be incomplete")
        result[name] = sorted(selected, key=lambda e: float(e["ts"]))
    return result


def summarize_batch_kernels(events: list[dict], *, phase: str, gdn_layers: int) -> dict:
    kernel_sum = sum(float(e["dur"]) for e in events) / 1000
    busy = interval_union_us(events) / 1000
    span = (max(float(e["ts"]) + float(e["dur"]) for e in events)
            - min(float(e["ts"]) for e in events)) / 1000
    allreduce = [e for e in events if any(name in str(e["name"]) for name in
                 ("quickreduce::allreduce", "cross_device_reduce", "ncclDevKernel_AllReduce",
                  "ncclKernel_AllReduce"))]
    summary = {
        "kernel_count": len(events), "step_kernel_sum_ms": kernel_sum,
        "step_gpu_busy_ms": busy, "step_gpu_span_ms": span,
        "step_gpu_gap_ms": max(0, span - busy),
        "step_kernel_overlap_ms": max(0, kernel_sum - busy),
        "allreduce_kernel_sum_ms": sum(float(e["dur"]) for e in allreduce) / 1000,
        "allreduce_count": len(allreduce),
    }
    if gdn_layers:
        from frontier.profiling.gdn.sglang_trace import extract_sglang_gdn_event_samples
        samples = extract_sglang_gdn_event_samples(events, phase=phase)
        count = len(samples["gdn_layer_e2e"])
        if count != gdn_layers:
            raise ValueError(f"Batch contains {count} GDN layers; expected exactly {gdn_layers}")
        summary["gdn_samples_ms"] = samples
        summary["gdn_kernel_sum_ms"] = sum(samples["gdn_layer_e2e"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from frontier.profiling.common.model_config import ModelConfig
    config = ModelConfig.from_model_name(args.model)
    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt", encoding="utf-8") as stream:
        batches = extract_batch_kernels(json.load(stream))
    records = {r.batch_id: r for r in read_batches(args.ledger) if r.rank == args.rank}
    rows = []
    for batch_id, kernels in batches.items():
        record = records[batch_id]
        if not record.profiled:
            raise ValueError("Trace markers must refer to profiled ledger rows")
        rows.append({"batch_id": batch_id, "rank": args.rank, "shape": record.to_dict(),
                     **summarize_batch_kernels(kernels, phase=record.phase,
                                              gdn_layers=config.get_num_gdn_layers())})
    with Path(args.output).open("x") as stream:
        json.dump(rows, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

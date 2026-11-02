#!/usr/bin/env python3
"""Audit retained V4 trace ownership without GPU, torch import, or pricing.

Uses the sealed dsa_kernel_trace parser/index and aggregate, preserves every
input byte, and writes a census plus explicit unowned kernel evidence. Host
envelopes are observations, not reconstructed forward boundaries/dependencies.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(index, event):
    return {"trace_event_index": index, **{k: event.get(k) for k in
            ("cat", "name", "pid", "tid", "ts", "dur", "args")}}


def intervals(events):
    spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    end, union = float("-inf"), 0.0
    for start, stop in spans:
        union += max(0, stop - max(start, end))
        end = max(end, stop)
    return {"kernel_count": len(events), "sum_ms": sum(e["dur"] for e in events) / 1000,
            "union_ms": union / 1000,
            "span_ms": (end - spans[0][0]) / 1000 if spans else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acceptance", "package", "native", "trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    root = Path.cwd()
    accepted = json.loads(args.acceptance.read_text())
    manifest_path = args.package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((args.package / "config.json").read_text())
    snapshot = args.package / "source" / accepted["source_revision"]
    indexed = {args.acceptance: digest(args.acceptance), manifest_path: accepted["manifest_sha256"],
               args.trace: accepted["trace_sha256"]}
    indexed.update({root / p: v["sha256"] for p, v in accepted["retained_files"].items()})
    result_path = args.acceptance.with_name("results_receipt.json")
    indexed[result_path] = accepted["evidence_sha256"]["results_receipt.json"]
    assert digest(result_path) == indexed[result_path]
    results = json.loads(result_path.read_text())
    indexed.update({Path(results["original_artifact_root"]) / p: v["sha256"]
                    for p, v in results["artifacts"].items()})
    indexed.update({args.package / p: v for p, v in manifest["files"].items()})
    native_files = ("vllm/models/deepseek_v4/attention.py",
                    "vllm/models/deepseek_v4/compressor.py",
                    "vllm/models/deepseek_v4/nvidia/flashmla.py",
                    "vllm/models/deepseek_v4/nvidia/ops/o_proj.py")
    indexed.update({args.native / p: config["expected_source_hashes"][p] for p in native_files})
    indexed[args.native / "vllm/models/deepseek_v4/nvidia/model.py"] = manifest[
        "copied_runtime_files"]["official_model.py"]["base_sha256"]
    extra_sources = ("vllm/utils/multi_stream_utils.py", "vllm/utils/deep_gemm.py",
                     "vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py",
                     "vllm/models/common/ops/fused_qk_rmsnorm.py", "vllm/platforms/cuda.py",
                     "vllm/model_executor/kernels/mhc/tilelang.py",
                     "vllm/model_executor/kernels/mhc/tilelang_kernels.py",
                     "vllm/models/deepseek_v4/sparse_mla.py",
                     "vllm/v1/attention/backends/mla/compressor_utils.py",
                     "vllm/v1/attention/backends/mla/sparse_swa.py")
    indexed.update({args.native / p: digest(args.native / p) for p in extra_sources})
    before = {str(p): {"sha256": digest(p), "expected_sha256": h,
                       "bytes": p.stat().st_size} for p, h in indexed.items()}
    assert all(v["sha256"] == v["expected_sha256"] for v in before.values())
    module_path = snapshot / "frontier/profiling/attention/dsa_kernel_trace.py"
    spec = importlib.util.spec_from_file_location("retained_dsa_kernel_trace", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    events = module.load_trace_events(args.trace)
    ranges, launches, cpu_ids, by_layer = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    annotations = []
    for i, e in enumerate(events):
        if e.get("cat") == "user_annotation":
            key = module.parse_annotation_name(e.get("name"))
            if key is not None:
                annotations.append((i, e, key))
                ranges[module._thread(e)].append((e["ts"], e["ts"] + e["dur"], key))
                by_layer[key.step_index, key.global_layer_id].append(e)
        elif e.get("cat") in ("cuda_runtime", "cuda_driver"):
            launches[e.get("args", {}).get("correlation")].append((i, e))
        elif e.get("cat") == "cpu_op":
            cpu_ids[e.get("args", {}).get("External id")].append((i, e))
    for values in ranges.values():
        values.sort(key=lambda v: v[0])
    starts = {thread: [v[0] for v in values] for thread, values in ranges.items()}

    def owner(launch):
        thread = module._thread(launch)
        pos = bisect.bisect_right(starts.get(thread, []), launch["ts"]) - 1
        return ranges[thread][pos][2] if pos >= 0 and launch["ts"] <= ranges[thread][pos][1] else None

    kernels, scoped, unowned, ambiguous, missing_parent = [], defaultdict(list), [], [], []
    per_stream = defaultdict(list)
    first = min(e["ts"] for _, e, _ in annotations)
    last = max(e["ts"] + e["dur"] for _, e, _ in annotations)
    region_counts = Counter()
    for i, e in enumerate(events):
        if e.get("cat") != "kernel":
            continue
        kernels.append(e)
        matches = launches.get(e.get("args", {}).get("correlation"), [])
        parents = cpu_ids.get(e.get("args", {}).get("External id"), [])
        if not parents:
            missing_parent.append(i)
        keys = [owner(launch) for _, launch in matches]
        key = keys[0] if len(keys) == 1 else None
        if len(matches) != 1:
            ambiguous.append(compact(i, e))
        if key is not None:
            scoped[key].append(e)
        else:
            stamp = matches[0][1]["ts"] if len(matches) == 1 else None
            region = ("no_unique_launch" if stamp is None else "before_first_measured_scope"
                      if stamp < first else "after_last_measured_scope" if stamp > last
                      else "between_first_and_last_measured_scope")
            region_counts[region] += 1
            unowned.append({"kernel": compact(i, e), "region": region,
                            "launches": [compact(j, z) for j, z in matches],
                            "external_id_cpu_parents": [compact(j, z) for j, z in parents],
                            "attention_family": "UNKNOWN_NOT_ASSIGNED_BY_KERNEL_NAME"})
        per_stream[e["args"]["stream"]].append((i, e, key, matches))
    aggregate = module.aggregate_kernel_trace(events)
    raw_path = next(root / p for p in accepted["retained_files"] if p.endswith("dsv4_raw_e0.jsonl"))
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    raw_calls = {(r["step_index"], r["global_layer_id"], r["scope"]): r for r in raw
                 if r["kind"] == "call" and r["measurement"] == "KERNEL_ONLY"}
    assert len(raw_calls) == len(aggregate)
    for key, time in aggregate.items():
        call = raw_calls[key.step_index, key.global_layer_id, key.scope]
        assert call["kernel_count"] == time.kernel_count
        assert abs(call["time_ms"] - time.kernel_time_ms) < 1e-12
        assert len(scoped[key]) == time.kernel_count
    comparisons = []
    engine = next(r for r in raw if r["kind"] == "engine")
    variants = {r["global_layer_id"]: r["attention_variant"] for r in engine["layers"]}
    for (step, layer), annotations_in_layer in sorted(by_layer.items()):
        subset = [e for key, values in scoped.items() if
                  (key.step_index, key.global_layer_id) == (step, layer) for e in values]
        start = min(e["ts"] for e in annotations_in_layer)
        stop = max(e["ts"] + e["dur"] for e in annotations_in_layer)
        outside = [u["kernel"]["trace_event_index"] for u in unowned if any(
            start <= launch["ts"] <= stop for launch in u["launches"])]
        comparisons.append({"step_index": step, "layer": layer, "variant": variants[layer],
                            **intervals(subset), "unowned_inside_component_envelope": outside,
                            "host_component_envelope_us": [start, stop],
                            "streams": {str(s): intervals([e for e in subset if e["args"]["stream"] == s])
                                        for s in sorted({e["args"]["stream"] for e in subset})}})
    order, overlaps = {}, []
    for stream, values in per_stream.items():
        values.sort(key=lambda v: v[1]["ts"])
        reversals = 0
        for left, right in zip(values, values[1:]):
            i, e, key, lm = left
            j, z, other, rm = right
            reversals += bool(lm and rm and lm[0][1]["ts"] > rm[0][1]["ts"])
            if e["ts"] + e["dur"] > z["ts"]:
                overlaps.append({"stream": stream, "left_index": i, "right_index": j,
                                 "overlap_us": e["ts"] + e["dur"] - z["ts"],
                                 "left_scope": asdict(key) if key else None,
                                 "right_scope": asdict(other) if other else None})
        order[str(stream)] = {"kernels": len(values), "launch_order_reversals": reversals}
    sync = [compact(i, e) for i, e in enumerate(events) if e.get("name") in
            ("cudaEventRecordWithFlags", "cudaStreamWaitEvent", "cudaEventSynchronize", "cudaDeviceSynchronize")]
    after = {str(p): digest(p) for p in indexed}
    assert all(after[p] == v["sha256"] for p, v in before.items())
    event_samples = []
    for step in sorted({r["step_index"] for r in raw if r["kind"] == "step" and r["measurement"] == "CUDA_EVENT"}):
        for layer in sorted(variants):
            calls = [r for r in raw if r["kind"] == "call" and r["measurement"] == "CUDA_EVENT"
                     and (r["step_index"], r["global_layer_id"]) == (step, layer)]
            event_samples.append({"step_index": step, "layer": layer,
                                  "component_sum_ms": sum(r["time_ms"] for r in calls if r["scope"] != "dsv4_attention_layer"),
                                  "composite_ms": next(r["time_ms"] for r in calls if r["scope"] == "dsv4_attention_layer")})
    receipt = {"task_id": "W128-V4-KERNEL-COVERAGE", "status": "AUDIT_COMPLETE_TOTALS_BLOCKED",
               "Modification History": [{"date": "2026-10-23", "change": "Retained-data ownership/dependency census and original-byte preservation audit."}],
               "script_sha256": digest(Path(__file__)),
               "formal_op_gap_entry": False, "serving_admission": False,
               "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
               "command": sys.orig_argv, "cwd": str(root), "python": platform.python_version(),
               "native_commit_declared": manifest["native_git"], "torch_imported": "torch" in sys.modules,
               "input_hashes_before": before, "input_hashes_after": after, "originals_unchanged": True,
               "trace_fields": {"kernel_args": sorted(set().union(*(e["args"] for e in kernels))),
                                "sync_args": sorted(set().union(*(e["args"] for e in sync)))},
               "census": {"trace_events": len(events), "kernels": len(kernels), "measured_annotations": len(aggregate),
                          "scoped_unique_kernels": sum(map(len, scoped.values())), "unowned_kernels": len(unowned),
                          "ambiguous_or_missing_launch": len(ambiguous), "no_external_cpu_parent": len(missing_parent),
                          "unowned_regions": dict(region_counts), "sync_calls": dict(Counter(e["name"] for e in sync))},
               "scopes": [{**asdict(key), **intervals(scoped[key])} for key in aggregate],
               "per_layer_repeat": comparisons, "cuda_event_independent_samples": event_samples,
               "within_stream_order": order,
               "same_stream_interval_overlaps": overlaps, "synchronization_evidence": sync,
               "no_external_cpu_parent_indices": missing_parent, "ambiguous_launch_evidence": ambiguous,
               "unowned_kernel_evidence": unowned,
               "limits": ["No enclosing attention/forward boundary or warmup/request IDs.",
                          "CUDA event/stream handles are absent from event record/wait fields.",
                          "Source synchronization semantics do not prove actual event-handle edges.",
                          "PDL permits intra-stream overlap; enqueue order is not completion order.",
                          "Unowned regions/fingerprints are not attention or warmup denominators.",
                          "CUDA_EVENT was collected independently; differences are not CPU cost."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), "Refuse to overwrite an audit receipt"
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "census": receipt["census"], "originals_unchanged": True}))


if __name__ == "__main__":
    main()

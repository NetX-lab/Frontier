"""Validate and summarize disjoint HIP graph scopes without native CSV relabeling."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from frontier.validation.batch_record import read_batches, validate_rank_cohorts
from frontier.validation.operator_trace import model_layer_kinds
from frontier.validation.report import summarize_profile_perturbation


_PARENTS = {
    "decoder_layer": None,
    "input_layernorm": "decoder_layer",
    "gdn": "decoder_layer", "full_attention": "decoder_layer",
    "attention_reduce_and_mlp_norm": "decoder_layer",
    "post_attention_layernorm": "attention_reduce_and_mlp_norm",
    "mlp_including_tp_reduce": "decoder_layer",
    **{f"gdn_{name}": "gdn" for name in ("input_projections", "attn", "norm", "out_proj")},
    **{f"attention_{name}": "full_attention" for name in ("qkv_proj", "rotary_emb", "attn", "o_proj")},
    **{f"moe_{name}": "mlp_including_tp_reduce" for name in ("shared_expert", "gate", "topk", "experts")},
}


def expected_scopes(layer_kinds, layer_ids, components):
    if (not layer_ids or len(set(layer_ids)) != len(layer_ids)
            or any(type(i) is not int or i < 0 or i >= len(layer_kinds) for i in layer_ids)):
        raise ValueError("Selected graph layers must exist in the model schedule")
    selected = set(components)
    if not selected or len(selected) != len(components) or selected - set(_PARENTS):
        raise ValueError("Unknown, duplicate or empty graph components")
    for component in selected:
        parent = _PARENTS[component]
        while parent is not None:
            if parent in selected:
                raise ValueError("Nested graph scopes are not additive; select disjoint components")
            parent = _PARENTS[parent]
    result = set()
    for layer in layer_ids:
        if layer_kinds[layer] not in {"gdn", "attention"}:
            raise ValueError("Unknown model layer kind")
        excluded = ({"full_attention", "attention_qkv_proj", "attention_rotary_emb",
                     "attention_attn", "attention_o_proj"} if layer_kinds[layer] == "gdn"
                    else {"gdn", "gdn_input_projections", "gdn_attn", "gdn_norm", "gdn_out_proj"})
        result.update((layer, component) for component in selected - excluded)
    if not result:
        raise ValueError("No selected timing scopes exist in the model")
    return result


def summarize_graph_events(records, samples, *, tensor_parallel_size, expected, quality):
    cohorts = validate_rank_cohorts(records, tensor_parallel_size=tensor_parallel_size)
    ledger = {(r.batch_id, r.rank): r for r in records if r.phase == "decode"}
    if not ledger or any(not r.profiled or r.graph_mode != "FULL" or r.forward_gpu_ms is None
                         for r in ledger.values()):
        raise ValueError("Graph timing requires separate profiled full-decode cohorts")
    quality_by_batch = {r["batch_id"]: r for r in quality["batches"]}
    if any(batch_id not in quality_by_batch for batch_id, _ in ledger):
        raise ValueError("Missing baseline perturbation check")
    seen, captures, capture_sizes, scope_rows = defaultdict(set), {}, {}, defaultdict(list)
    scope_totals = defaultdict(float)
    for sample in samples:
        if any(type(sample[k]) is not int or sample[k] < 0 for k in ("rank", "layer_id", "physical_size")):
            raise ValueError("Graph rank/layer/physical size must be nonnegative integers")
        key = (sample["batch_id"], sample["rank"])
        if key not in ledger:
            raise ValueError("Graph event has no matching decode batch/rank")
        record = ledger[key]
        identity = (sample["layer_id"], sample["component"])
        if identity not in expected or identity in seen[key]:
            raise ValueError("Unexpected or duplicate graph event scope")
        seen[key].add(identity)
        if (sample["measurement_type"] != "HIP_GRAPH_EVENT" or sample["parent_id"] is not None
                or sample["physical_size"] != record.capture_size):
            raise ValueError("Graph measurement family, disjoint scope or physical size mismatch")
        capture_key = (record.rank, record.capture_size)
        capture_id = sample["capture_id"]
        if type(capture_id) is not int or capture_id < 1:
            raise ValueError("Invalid graph capture ID")
        if captures.setdefault(capture_key, capture_id) != capture_id:
            raise ValueError("Multiple captured graph variants for one rank/physical size")
        if capture_sizes.setdefault((record.rank, capture_id), record.capture_size) != record.capture_size:
            raise ValueError("One captured graph cannot have multiple physical sizes")
        duration = sample["inclusive_gpu_ms"]
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0 or duration > record.forward_gpu_ms):
            raise ValueError("Invalid graph scope GPU duration")
        scope_totals[key] += duration
        input_len = max(record.context_lens) - record.step + 1
        group = (len(record.request_ids), input_len, record.capture_size, record.rank, *identity)
        scope_rows[group].append((record, duration, quality_by_batch[record.batch_id]["passes_latency_check"]))
    if set(seen) != set(ledger) or any(identities != expected for identities in seen.values()):
        raise ValueError("Incomplete graph scope coverage for a decode batch/rank")
    if any(total > ledger[key].forward_gpu_ms + 1e-5 for key, total in scope_totals.items()):
        raise ValueError("Disjoint scope durations exceed their same-rank forward timing")
    summaries = []
    for (bs, length, physical, rank, layer, component), rows in sorted(scope_rows.items()):
        durations = [value for _, value, _ in rows]
        repetitions = defaultdict(list)
        for record, value, _ in rows:
            repetitions[record.repetition].append(value)
        medians = [statistics.median(values) for values in repetitions.values()]
        summaries.append({"batch_size": bs, "input_len": length, "physical_size": physical,
            "rank": rank, "layer_id": layer, "component": component,
            "samples": len(rows), "repetitions": len(repetitions),
            "gpu_median_of_repeat_medians_ms": statistics.median(medians),
            "gpu_min_ms": min(durations), "gpu_max_ms": max(durations),
            "repeat_median_cv_pct": 100 * statistics.pstdev(medians) / statistics.mean(medians),
            "samples_failing_forward_latency_check": sum(not accepted for _, _, accepted in rows)})
    decode_quality = [quality_by_batch[batch_id] for batch_id in cohorts if (batch_id, 0) in ledger]
    return {"measurement_type": "HIP_GRAPH_EVENT", "decode_batches": len(decode_quality),
        "scope_samples": sum(len(rows) for rows in scope_rows.values()),
        "selected_scopes_per_rank": len(expected),
        "decode_batches_failing_latency_check": sum(not r["passes_latency_check"] for r in decode_quality),
        "all_decode_inputs_fixed": all(r.decode_input_sha256 is not None for r in ledger.values()),
        "all_decode_batches_pass_latency_check": all(r["passes_latency_check"] for r in decode_quality),
        "summaries": summaries,
        "native_profile_export_admitted": False,
        "limitations": [
            "Disjoint event spans, not kernel sums. Endpoint costs can bias small scopes even when forward latency passes.",
            "All samples, including rejected ones, remain in diagnostic summaries; no trimming or correction factor.",
            "Rank/layer identities remain separate; summing per-scope rank maxima is not a measured critical path.",
            "Fused mixer/MLP scopes and reduction boundaries do not map one-to-one to native compute operators.",
            "Timing scopes alone contain no routing vectors; consult a separate routing audit when available. "
            "Selected timing layers do not establish full decoder/forward coverage or EP/multi-node scaling.",
            "Fixed token inputs do not guarantee identical activations, expert routes or numerical correctness.",
        ]}


def summarize_decoder_graph_events(
        records, samples, *, tensor_parallel_size, num_layers, quality):
    """Summarize the single span matching the runtime-cost decoder boundary."""
    cohorts = validate_rank_cohorts(records, tensor_parallel_size=tensor_parallel_size)
    ledger = {(record.batch_id, record.rank): record
              for record in records if record.phase == "decode"}
    if (not ledger or type(num_layers) is not int or num_layers < 1
            or any(not record.profiled or record.graph_mode != "FULL"
                   or record.forward_gpu_ms is None for record in ledger.values())):
        raise ValueError("Decoder graph timing requires separate profiled full-decode cohorts")
    quality_by_batch = {row["batch_id"]: row for row in quality["batches"]}
    if any(batch_id not in quality_by_batch for batch_id, _ in ledger):
        raise ValueError("Missing baseline perturbation check")

    seen, captures, capture_sizes = set(), {}, {}
    grouped = defaultdict(list)
    for sample in samples:
        key = (sample.get("batch_id"), sample.get("rank"))
        if key not in ledger or key in seen:
            raise ValueError("Decoder graph event has no matching batch/rank or is duplicated")
        seen.add(key)
        record = ledger[key]
        if (sample.get("component") != "decoder"
                or sample.get("measurement_type") != "HIP_GRAPH_EVENT"
                or sample.get("first_layer_id") != 0
                or sample.get("last_layer_id") != num_layers - 1
                or type(sample.get("physical_size")) is not int
                or sample["physical_size"] != record.capture_size):
            raise ValueError("Decoder graph boundary, measurement family or size mismatch")
        capture_id = sample.get("capture_id")
        if type(capture_id) is not int or capture_id < 1:
            raise ValueError("Invalid decoder graph capture ID")
        capture_key = (record.rank, record.capture_size)
        if captures.setdefault(capture_key, capture_id) != capture_id:
            raise ValueError("Multiple captured decoder graph variants for one rank/size")
        if capture_sizes.setdefault((record.rank, capture_id), record.capture_size) != record.capture_size:
            raise ValueError("One decoder graph capture cannot have multiple physical sizes")
        duration = sample.get("inclusive_gpu_ms")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0
                or duration > record.forward_gpu_ms):
            raise ValueError("Invalid decoder graph event duration")
        input_len = max(record.context_lens) - record.step + 1
        group = (len(record.request_ids), input_len, record.capture_size, record.rank)
        grouped[group].append((record, duration,
            quality_by_batch[record.batch_id]["passes_latency_check"]))
    if seen != set(ledger):
        raise ValueError("Incomplete decoder graph event coverage")

    summaries = []
    for (batch_size, input_len, physical_size, rank), rows in sorted(grouped.items()):
        durations = [duration for _, duration, _ in rows]
        gaps = [record.forward_gpu_ms - duration for record, duration, _ in rows]
        repetitions = defaultdict(list)
        for record, duration, _ in rows:
            repetitions[record.repetition].append(duration)
        medians = [statistics.median(values) for values in repetitions.values()]
        summaries.append({
            "batch_size": batch_size,
            "input_len": input_len,
            "physical_size": physical_size,
            "rank": rank,
            "samples": len(rows),
            "repetitions": len(repetitions),
            "decoder_gpu_median_of_repeat_medians_ms": statistics.median(medians),
            "decoder_gpu_min_ms": min(durations),
            "decoder_gpu_max_ms": max(durations),
            "forward_minus_decoder_median_ms": statistics.median(gaps),
            "repeat_median_cv_pct": (
                100 * statistics.pstdev(medians) / statistics.mean(medians)),
            "samples_failing_forward_latency_check": sum(
                not accepted for _, _, accepted in rows),
        })
    decode_quality = [quality_by_batch[batch_id]
                      for batch_id in cohorts if (batch_id, 0) in ledger]
    return {
        "measurement_type": "HIP_GRAPH_EVENT",
        "boundary": "first decoder layer entry through last decoder layer exit",
        "decode_batches": len(decode_quality),
        "decoder_scope_samples": len(samples),
        "all_decode_inputs_fixed": all(
            record.decode_input_sha256 is not None for record in ledger.values()),
        "all_decode_batches_pass_latency_check": all(
            row["passes_latency_check"] for row in decode_quality),
        "decode_batches_failing_latency_check": sum(
            not row["passes_latency_check"] for row in decode_quality),
        "summaries": summaries,
        "native_profile_export_admitted": False,
        "limitations": [
            "Diagnostic event span, not an isolated primitive profile or native CSV row.",
            "The two event nodes are inside the recaptured graph and can perturb its execution.",
            "Rejected forward-latency samples remain visible; no trimming or correction is applied.",
            "The span excludes embedding, final residual/norm, logits, sampling and host work.",
            "One-node TP-only validation does not establish EP or multi-node scaling.",
        ],
    }


def import_graph_capture(path, *, model_config):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("engine") != "sglang":
        raise ValueError("Requires a complete SGLang capture")
    if Path(manifest["model_path"]).name != Path(model_config.name).name:
        raise ValueError("Model configuration does not match capture checkpoint")
    if manifest["topology"] != {"tp": 8, "ep": 1, "pp": 1, "nodes": 1}:
        raise ValueError("Initial graph scope contract requires one-node TP=8/EP=1/PP=1")
    expected = expected_scopes(model_layer_kinds(model_config),
        manifest["options"]["graph_event_layers"], manifest["options"]["graph_event_components"])
    records = read_batches(path / "graph-batches.jsonl")
    baseline = [r for r in read_batches(path / "batches.jsonl") if not r.profiled]
    quality = summarize_profile_perturbation(baseline + records, tensor_parallel_size=8)
    samples, sources = [], []
    for rank in range(8):
        source = path / f"graph-events-rank{rank}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if any(row["rank"] != rank for row in rows):
            raise ValueError("Graph sample rank disagrees with source file")
        samples.extend(rows)
        sources.append({"file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    return {"schema_version": 1, "manifest": manifest, "sources": sources,
        "profile_perturbation": quality, **summarize_graph_events(records, samples,
            tensor_parallel_size=8, expected=expected, quality=quality)}


def import_decoder_graph_capture(path, *, model_config):
    manifest = json.loads((path / "manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("engine") != "sglang"
            or not manifest.get("options", {}).get("graph_decoder_event")):
        raise ValueError("Requires a complete SGLang decoder graph-event capture")
    if (Path(manifest["model_path"]).name != Path(model_config.name).name
            or manifest["topology"] != {"tp": 8, "ep": 1, "pp": 1, "nodes": 1}):
        raise ValueError("Decoder graph capture model/topology mismatch")
    records = read_batches(path / "graph-batches.jsonl")
    baseline = [record for record in read_batches(path / "batches.jsonl")
                if not record.profiled]
    quality = summarize_profile_perturbation(
        baseline + records, tensor_parallel_size=8)
    samples, sources = [], []
    for rank in range(8):
        source = path / f"decoder-events-rank{rank}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()
                if line.strip()]
        if any(row.get("rank") != rank for row in rows):
            raise ValueError("Decoder graph sample rank disagrees with source file")
        samples.extend(rows)
        sources.append({"file": source.name,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    return {
        "schema_version": 1,
        "manifest": manifest,
        "sources": sources,
        "profile_perturbation": quality,
        **summarize_decoder_graph_events(
            records, samples, tensor_parallel_size=8,
            num_layers=model_config.num_layers, quality=quality),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--decoder", action="store_true",
                        help="Import the whole-decoder event span instead of layer scopes")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from frontier.profiling.common.model_config import ModelConfig
    importer = import_decoder_graph_capture if args.decoder else import_graph_capture
    result = importer(args.capture_dir, model_config=ModelConfig.from_model_name(args.model))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items()
                     if k not in {"summaries", "manifest", "sources", "profile_perturbation"}}, indent=2))


if __name__ == "__main__":
    main()

"""Audit per-layer expert loads, TP agreement and fixed-input routing stability."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from frontier.validation.batch_record import read_batches, validate_rank_cohorts
from frontier.validation.report import summarize_profile_perturbation
from frontier.validation.routing import read_routing


def graph_routing_enabled(options):
    """Graph routing exists with either layer scopes or the decoder-wide scope."""
    return bool(options.get("graph_event_layers") or options.get("graph_decoder_event"))


def audit_routing(records, routes, *, layer_ids, tensor_parallel_size, model_config, feature_writer=None):
    cohorts = validate_rank_cohorts(records, tensor_parallel_size=tensor_parallel_size)
    ledger = {(r.batch_id, r.rank): r for r in records if r.phase == "decode"}
    if not ledger or any(not r.profiled or r.graph_mode != "FULL" or r.decode_input_sha256 is None
                         for r in ledger.values()):
        raise ValueError("Routing audit requires separate fixed-input full-decode cohorts")
    expected = set(layer_ids)
    if not expected or len(expected) != len(layer_ids) or any(type(i) is not int or i not in range(model_config.num_layers) for i in expected):
        raise ValueError("Invalid selected routing layers")
    seen, captures, reference, groups = defaultdict(set), {}, {}, defaultdict(list)
    tp_logical_mismatches, tp_padding_mismatches = [], []
    rank0 = {}
    count = 0
    for row in routes:  # importer supplies rank 0 first, then other ranks
        key = row.batch_id, row.rank
        if key not in ledger or row.layer_id not in expected or row.layer_id in seen[key]:
            raise ValueError("Unexpected/duplicate routing batch, rank or layer")
        seen[key].add(row.layer_id)
        record = ledger[key]
        if (row.logical_size != len(record.request_ids) or row.physical_size != record.capture_size
                or row.num_experts != model_config.num_experts or row.top_k != model_config.num_experts_per_tok):
            raise ValueError("Routing shape/model does not match batch ledger")
        if captures.setdefault((row.rank, row.physical_size), row.capture_id) != row.capture_id:
            raise ValueError("Multiple routing graph variants for one physical size")
        reference_key = row.batch_id, row.layer_id
        logical = row.logical_assignment_sha256, row.logical_weights_sha256, row.logical_expert_counts
        padding = row.padding_assignment_sha256, row.padding_weights_sha256, row.padding_expert_counts
        if row.rank == 0:
            reference[reference_key] = logical, padding
            paired_key = record.shape_key(), record.step, record.repetition, record.decode_input_sha256, row.layer_id
            if paired_key in rank0:
                raise ValueError("Duplicate routing workload/step/repetition")
            rank0[paired_key] = row
        else:
            if reference_key not in reference:
                raise ValueError("Rank-0 routing reference must precede peer ranks")
            if logical != reference[reference_key][0]:
                tp_logical_mismatches.append((row.batch_id, row.rank, row.layer_id))
            if padding != reference[reference_key][1]:
                tp_padding_mismatches.append((row.batch_id, row.rank, row.layer_id))
        counts = [a + b for a, b in zip(row.logical_expert_counts, row.padding_expert_counts)]
        active = sum(v > 0 for v in counts)
        max_load_ratio = max(counts) * len(counts) / sum(counts)
        input_len = max(record.context_lens) - record.step + 1
        group = (row.logical_size, input_len, row.physical_size, row.rank, row.layer_id)
        groups[group].append((active, max_load_ratio, sum(row.padding_expert_counts),
            row.padding_positive_weight_slots, row.logical_assignment_sha256,
            record.step, record.repetition, row.logical_expert_set_sha256))
        if feature_writer is not None and row.rank == 0:
            feature_writer({"batch_id": row.batch_id, "rank": 0, "layer_id": row.layer_id,
                "logical_size": row.logical_size, "physical_size": row.physical_size,
                "logical_features": row.native_load_features(model_config=model_config, include_padding=False),
                "physical_features": row.native_load_features(model_config=model_config, include_padding=True),
                "note": "Observed EP=1 top-k selection features, not a measured GEMM profile or latency prediction."})
        count += 1
    if set(seen) != set(ledger) or any(layers != expected for layers in seen.values()):
        raise ValueError("Incomplete routing layer coverage for a decode batch/rank")
    summaries = []
    for (logical, input_len, physical, rank, layer), values in sorted(groups.items()):
        by_step = defaultdict(list)
        for value in values:
            by_step[value[5]].append(value)
        varying_steps = sum(len({value[4] for value in rows}) > 1 for rows in by_step.values())
        varying_sets = sum(len({value[7] for value in rows}) > 1 for rows in by_step.values())
        summaries.append({"batch_size": logical, "input_len": input_len, "physical_size": physical, "rank": rank, "layer_id": layer,
            "samples": len(values), "active_experts_median": statistics.median(v[0] for v in values),
            "max_load_ratio_median": statistics.median(v[1] for v in values),
            "padding_selected_slots_median": statistics.median(v[2] for v in values),
            "padding_positive_weight_slots_median": statistics.median(v[3] for v in values),
            "steps_with_changed_logical_assignments": varying_steps,
            "steps_with_changed_logical_expert_sets": varying_sets, "steps": len(by_step)})
    return {"decode_batches": sum((batch_id, 0) in ledger for batch_id in cohorts),
        "routing_records": count, "selected_layers": sorted(expected),
        "all_model_layers_covered": expected == set(range(model_config.num_layers)),
        "tp_logical_mismatches": tp_logical_mismatches,
        "tp_padding_mismatches": tp_padding_mismatches, "summaries": summaries}, rank0


def compare_routing_passes(reference, observed):
    rows = []
    for key, row in observed.items():
        if key not in reference:
            raise ValueError("No matching fixed-input routing workload/step/repetition/layer")
        base = reference[key]
        l1 = sum(abs(a - b) for a, b in zip(base.logical_expert_counts, row.logical_expert_counts))
        padding_l1 = sum(abs(a - b) for a, b in zip(base.padding_expert_counts, row.padding_expert_counts))
        padding_total = sum(base.padding_expert_counts) + sum(row.padding_expert_counts)
        rows.append({"batch_id": row.batch_id, "reference_batch_id": base.batch_id,
            "batch_size": row.logical_size, "physical_size": row.physical_size, "layer_id": row.layer_id,
            "identical_logical_assignments": base.logical_assignment_sha256 == row.logical_assignment_sha256,
            "identical_logical_expert_sets": base.logical_expert_set_sha256 == row.logical_expert_set_sha256,
            "identical_logical_weights": base.logical_weights_sha256 == row.logical_weights_sha256,
            "logical_histogram_l1_fraction": l1 / (2 * row.logical_size * row.top_k),
            "padding_histogram_l1_fraction": padding_l1 / padding_total if padding_total else 0.0})
    if not rows:
        raise ValueError("Empty routing comparison")
    return {"rank": 0, "paired_layer_samples": len(rows),
        "identical_logical_assignments": sum(r["identical_logical_assignments"] for r in rows),
        "identical_logical_expert_sets": sum(r["identical_logical_expert_sets"] for r in rows),
        "identical_logical_weights": sum(r["identical_logical_weights"] for r in rows),
        "logical_histogram_l1_fraction_median": statistics.median(r["logical_histogram_l1_fraction"] for r in rows),
        "logical_histogram_l1_fraction_max": max(r["logical_histogram_l1_fraction"] for r in rows),
        "rows": rows}


def import_routing_capture(path, *, model_config, feature_writer=None):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("engine") != "sglang":
        raise ValueError("Requires a complete SGLang capture")
    if (manifest["topology"] != {"tp": 8, "ep": 1, "pp": 1, "nodes": 1}
            or Path(manifest["model_path"]).name != Path(model_config.name).name):
        raise ValueError("Routing contract requires the matching checkpoint and one-node TP=8/EP=1/PP=1")
    options = manifest["options"]
    layers = list(range(model_config.num_layers)) if options["routing_all_layers"] else options["routing_layers"]
    baseline = [r for r in read_batches(path / "batches.jsonl") if not r.profiled]
    reports, references, sources = {}, {}, []
    for prefix in ("routing", "graph"):
        if prefix == "graph" and not graph_routing_enabled(options):
            continue
        records = read_batches(path / f"{prefix}-batches.jsonl")
        def routes():
            for rank in range(8):
                source = path / f"{prefix}-routing-rank{rank}.jsonl.gz"
                sources.append({"file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
                for row in read_routing(source):
                    if row.rank != rank:
                        raise ValueError("Routing rank disagrees with source file")
                    yield row
        writer = (lambda row, prefix=prefix: feature_writer({"pass": prefix, **row})) if feature_writer else None
        report, reference = audit_routing(records, routes(), layer_ids=layers, tensor_parallel_size=8,
                                          model_config=model_config, feature_writer=writer)
        reports[prefix] = {**report, "profile_perturbation": summarize_profile_perturbation(
            baseline + records, tensor_parallel_size=8)}
        references[prefix] = reference
    comparison = compare_routing_passes(references["routing"], references["graph"]) if "graph" in references else None
    return {"schema_version": 1, "manifest": manifest, "sources": sources, "passes": reports,
        "timing_vs_routing_only": comparison,
        "limitations": [
            "Top-k buffer references are sampled after forward; retaining tensors may change graph pool allocation.",
            "Positive/zero routing weights do not establish whether the backend executes or prunes a GEMM slot.",
            "Features use native EPLaneWorkload/MoELoadImbalanceInput for observed EP=1 only; no synthetic EP extrapolation.",
            "No ground-truth latency or profile is invented from expert counts. Unprofiled timing checks still apply.",
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-features-output", type=Path)
    args = parser.parse_args()
    from frontier.profiling.common.model_config import ModelConfig
    features = gzip.open(args.native_features_output, "xt") if args.native_features_output else None
    try:
        result = import_routing_capture(args.capture_dir,
            model_config=ModelConfig.from_model_name(args.model),
            feature_writer=(lambda row: features.write(json.dumps(row) + "\n")) if features else None)
    finally:
        if features:
            features.close()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passes": {name: {k: v for k, v in report.items()
        if k not in {"summaries", "profile_perturbation", "tp_logical_mismatches", "tp_padding_mismatches"}}
        for name, report in result["passes"].items()}, "comparison": {k: v for k, v in
        (result["timing_vs_routing_only"] or {}).items() if k != "rows"}}, indent=2))


if __name__ == "__main__":
    main()

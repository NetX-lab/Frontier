"""Correlate an exact runtime-cost prediction with a decoder HIP graph span.

This is a post-prediction diagnostic. Observed timings never enter query
construction, profile admission, cost composition, or a correction factor.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from frontier.runtime_cost.sglang import DecodeWorkload, RuntimeIdentity
from frontier.validation.batch_record import read_batches, validate_rank_cohorts
from frontier.validation.graph_event_report import import_decoder_graph_capture
from frontier.validation.routing import read_routing
from frontier.validation.runtime_cost import (
    identity_for_capture, plan_captured_batch,
)


def correlate_exact_decoder(
        prediction_payload, records, samples, *, identity, expected_queries,
        routing_audit, quality, all_capture_batches_pass):
    """Validate matching prediction/observation identities and report error."""
    records, samples, expected_queries = tuple(records), tuple(samples), tuple(expected_queries)
    try:
        serialized_identity = RuntimeIdentity(**prediction_payload.get("identity", {}))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Runtime prediction schema/identity mismatch") from error
    if (not isinstance(prediction_payload, dict)
            or type(prediction_payload.get("schema_version")) is not int
            or prediction_payload["schema_version"] != 1
            or not isinstance(identity, RuntimeIdentity)
            or serialized_identity != identity):
        raise ValueError("Runtime prediction schema/identity mismatch")
    batch_id = prediction_payload.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("Runtime prediction requires a captured batch ID")
    coverage = prediction_payload.get("profile_coverage", {})
    prediction = prediction_payload.get("prediction")
    if (not isinstance(coverage, dict)
            or coverage.get("complete") is not True
            or coverage.get("missing") != []
            or not isinstance(prediction, dict)
            or prediction.get("routing_conditioned") is not True
            or prediction.get("full_forward_parity_admitted") is not False
            or prediction.get("diagnostic") is not False):
        raise ValueError("Correlation requires a complete isolated-profile prediction")
    predicted_ms = prediction.get("decoder_ms")
    if (isinstance(predicted_ms, bool) or not isinstance(predicted_ms, (int, float))
            or not math.isfinite(predicted_ms) or predicted_ms <= 0):
        raise ValueError("Prediction requires a finite positive decoder duration")

    serialized = prediction_payload.get("queries")
    expected_keys = [query.key for query in expected_queries]
    expected_serialized = [
        {"query": asdict(query), "query_key": query.key}
        for query in expected_queries]
    if (not isinstance(serialized, list) or not expected_keys
            or any(not isinstance(item, dict) for item in serialized)
            or len(expected_keys) != len(set(expected_keys))
            or json.dumps(serialized, sort_keys=True, separators=(",", ":"))
            != json.dumps(expected_serialized, sort_keys=True, separators=(",", ":"))
            or type(coverage.get("queries")) is not int
            or coverage["queries"] != len(expected_keys)):
        raise ValueError("Prediction queries do not match the exact captured routing plan")
    if (not isinstance(routing_audit, dict)
            or routing_audit.get("all_model_layers_covered") is not True
            or isinstance(routing_audit.get("routing_records"), bool)
            or not isinstance(routing_audit.get("routing_records"), int)
            or routing_audit["routing_records"] < 1
            or routing_audit.get("tp_logical_mismatches")
            or routing_audit.get("tp_padding_mismatches")):
        raise ValueError("Exact decoder correlation requires all-rank routing agreement")

    layer_ids = {query.layer_id for query in expected_queries}
    if layer_ids != set(range(len(layer_ids))):
        raise ValueError("Exact runtime plan must cover a contiguous decoder layer schedule")

    cohorts = validate_rank_cohorts(records, tensor_parallel_size=identity.tensor_parallel_size)
    if (set(cohorts) != {batch_id} or len(records) != identity.tensor_parallel_size
            or any(record.phase != "decode" or not record.profiled
                   or record.graph_mode != "FULL" or record.forward_gpu_ms is None
                   or record.decode_input_sha256 is None
                   for record in records)):
        raise ValueError("Correlation requires one fixed-input profiled decode TP cohort")
    by_rank = {record.rank: record for record in records}
    sample_by_rank = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Malformed decoder event span")
        rank = sample.get("rank")
        if (type(rank) is not int or rank not in by_rank or rank in sample_by_rank
                or sample.get("batch_id") != batch_id
                or sample.get("component") != "decoder"
                or sample.get("measurement_type") != "HIP_GRAPH_EVENT"
                or sample.get("first_layer_id") != 0
                or sample.get("last_layer_id") != len(layer_ids) - 1
                or sample.get("physical_size") != by_rank[rank].capture_size):
            raise ValueError("Missing, duplicate or mismatched decoder event span")
        duration = sample.get("inclusive_gpu_ms")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0
                or duration > by_rank[rank].forward_gpu_ms):
            raise ValueError("Invalid decoder event duration")
        sample_by_rank[rank] = sample
    if set(sample_by_rank) != set(by_rank):
        raise ValueError("Decoder event span does not cover every TP rank")

    quality_rows = quality.get("batches", ()) if isinstance(quality, dict) else ()
    if (not isinstance(quality_rows, (list, tuple))
            or any(not isinstance(row, dict) for row in quality_rows)):
        raise ValueError("Malformed decoder-event perturbation report")
    matching_quality = [row for row in quality_rows if row.get("batch_id") == batch_id]
    if (len(matching_quality) != 1
            or matching_quality[0].get("passes_latency_check") is not True):
        raise ValueError("Exact decoder event failed or lacks the perturbation gate")
    quality_row = matching_quality[0]
    signed_change_pct = quality_row.get("signed_change_pct")
    if (isinstance(signed_change_pct, bool)
            or not isinstance(signed_change_pct, (int, float))
            or not math.isfinite(signed_change_pct)):
        raise ValueError("Invalid decoder-event perturbation measurement")
    if type(all_capture_batches_pass) is not bool or not all_capture_batches_pass:
        raise ValueError("Every captured decoder batch must pass the perturbation gate")
    observed_ms = max(row["inclusive_gpu_ms"] for row in sample_by_rank.values())
    forward_ms = max(record.forward_gpu_ms for record in records)
    signed_error = predicted_ms - observed_ms
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "identity": prediction_payload["identity"],
        "boundary": "first decoder layer entry through last decoder layer exit",
        "queries": len(expected_keys),
        "routing_records": routing_audit["routing_records"],
        "predicted_decoder_ms": predicted_ms,
        "observed_decoder_max_rank_ms": observed_ms,
        "signed_error_ms": signed_error,
        "absolute_relative_error": abs(signed_error) / observed_ms,
        "profiled_forward_max_rank_ms": forward_ms,
        "forward_minus_decoder_ms": forward_ms - observed_ms,
        "observer_signed_change_pct": signed_change_pct,
        "observer_latency_check_passed": True,
        "all_capture_decode_batches_passed": all_capture_batches_pass,
        "full_forward_parity_admitted": False,
        "correction_applied": False,
        "note": "Post-prediction exact-route decoder diagnostic; observation is not a profile, fit target, or full-forward parity claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from frontier.profiling.common.model_config import ModelConfig

    payload = json.loads(args.prediction.read_text())
    model = ModelConfig.from_model_name(args.model)
    manifest = json.loads((args.capture_dir / "manifest.json").read_text())
    serialized_identity = RuntimeIdentity(**payload["identity"])
    identity = identity_for_capture(manifest, model, serialized_identity.device)
    if identity != serialized_identity:
        raise ValueError("Prediction and decoder capture have different runtime identities")
    batch_id = payload["batch_id"]
    records = [record for record in read_batches(args.capture_dir / "graph-batches.jsonl")
               if record.batch_id == batch_id]
    routes = []
    for rank in range(identity.tensor_parallel_size):
        routes.extend(route for route in read_routing(
            args.capture_dir / f"graph-routing-rank{rank}.jsonl.gz")
            if route.batch_id == batch_id)
    first_query = payload["queries"][0]["query"]
    workload = DecodeWorkload(**first_query["workload"])
    padding_context_lens = workload.physical_context_lens[workload.logical_size:]
    _, planned_workload, _, queries, routing_audit = plan_captured_batch(
        records, routes, model_config=model, identity=identity,
        padding_context_lens=padding_context_lens)
    if planned_workload != workload:
        raise ValueError("Prediction workload differs from the observed decoder batch")
    decoder_report = import_decoder_graph_capture(args.capture_dir, model_config=model)
    samples = []
    for rank in range(identity.tensor_parallel_size):
        source = args.capture_dir / f"decoder-events-rank{rank}.jsonl"
        samples.extend(row for row in (
            json.loads(line) for line in source.read_text().splitlines() if line.strip())
            if row.get("batch_id") == batch_id)
    result = correlate_exact_decoder(
        payload, records, samples, identity=identity, expected_queries=queries,
        routing_audit=routing_audit, quality=decoder_report["profile_perturbation"],
        all_capture_batches_pass=decoder_report["all_decode_batches_pass_latency_check"])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

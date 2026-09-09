from copy import deepcopy
from dataclasses import asdict

import pytest

from frontier.runtime_cost.sglang import CostQuery, DecodeWorkload, RuntimeIdentity
from frontier.validation.batch_record import BatchRecord
from frontier.validation.runtime_cost_correlation import correlate_exact_decoder


IDENTITY = RuntimeIdentity("a" * 64, "synthetic-unit-device", 2, "synthetic-unit-stack")
WORKLOAD = DecodeWorkload(2, (1025, 2049, 1, 1))
QUERY = CostQuery(IDENTITY, 0, "gdn", "input_layernorm", WORKLOAD)


def record(rank):
    return BatchRecord(
        batch_id="exact-batch", rank=rank, phase="decode",
        request_ids=("a", "b"), query_lens=(1, 1), context_lens=(1024, 2048),
        prefill_mask=(False, False), graph_mode="FULL", capture_size=4,
        forward_gpu_ms=10.0 + rank, step_wall_ms=12.0, profiled=True,
        decode_input_sha256="c" * 64,
    )


def fixtures():
    prediction = {
        "schema_version": 1,
        "batch_id": "exact-batch",
        "identity": asdict(IDENTITY),
        "profile_coverage": {"complete": True, "missing": [], "queries": 1},
        "queries": [{"query": asdict(QUERY), "query_key": QUERY.key}],
        "prediction": {
            "decoder_ms": 8.5,
            "routing_conditioned": True,
            "full_forward_parity_admitted": False,
            "diagnostic": False,
        },
    }
    records = [record(0), record(1)]
    samples = [{
        "batch_id": "exact-batch", "rank": rank, "component": "decoder",
        "measurement_type": "HIP_GRAPH_EVENT", "first_layer_id": 0,
        "last_layer_id": 0, "physical_size": 4, "inclusive_gpu_ms": 8.0 + rank,
    } for rank in range(2)]
    routing = {
        "routing_records": 2, "all_model_layers_covered": True,
        "tp_logical_mismatches": [], "tp_padding_mismatches": [],
    }
    quality = {"batches": [{
        "batch_id": "exact-batch", "passes_latency_check": True,
        "signed_change_pct": 0.5,
    }]}
    return prediction, records, samples, routing, quality


def correlate(*, mutate=None, queries=(QUERY,), capture_pass=True):
    prediction, records, samples, routing, quality = fixtures()
    values = {
        "prediction": prediction, "records": records, "samples": samples,
        "routing": routing, "quality": quality,
    }
    if mutate:
        mutate(values)
    return correlate_exact_decoder(
        values["prediction"], values["records"], values["samples"],
        identity=IDENTITY, expected_queries=queries, routing_audit=values["routing"],
        quality=values["quality"], all_capture_batches_pass=capture_pass,
    )


def test_exact_decoder_correlation_uses_slowest_rank_and_applies_no_correction():
    result = correlate()
    assert result["predicted_decoder_ms"] == 8.5
    assert result["observed_decoder_max_rank_ms"] == 9.0
    assert result["signed_error_ms"] == -0.5
    assert result["absolute_relative_error"] == pytest.approx(0.5 / 9.0)
    assert result["profiled_forward_max_rank_ms"] == 11.0
    assert result["forward_minus_decoder_ms"] == 2.0
    assert result["observer_latency_check_passed"]
    assert result["all_capture_decode_batches_passed"]
    assert not result["correction_applied"]
    assert not result["full_forward_parity_admitted"]


@pytest.mark.parametrize("mutation", [
    lambda value: value["prediction"]["profile_coverage"].update(complete=False),
    lambda value: value["prediction"]["queries"][0].update(query_key="b" * 64),
    lambda value: value["prediction"]["prediction"].update(diagnostic=True),
    lambda value: value["routing"]["tp_logical_mismatches"].append({"layer": 0}),
    lambda value: value["records"].pop(),
    lambda value: value["samples"].pop(),
    lambda value: value["samples"][0].update(inclusive_gpu_ms=20.0),
    lambda value: value["quality"]["batches"][0].update(passes_latency_check=False),
])
def test_correlation_rejects_nonexact_or_untrusted_inputs(mutation):
    with pytest.raises(ValueError):
        correlate(mutate=mutation)


def test_correlation_rejects_noncontiguous_layer_plan_and_nonboolean_capture_gate():
    changed = CostQuery(IDENTITY, 2, "gdn", "input_layernorm", WORKLOAD,
                        residual_from_layer=1)

    def use_changed_query(value):
        value["prediction"]["queries"] = [
            {"query": asdict(changed), "query_key": changed.key}]

    with pytest.raises(ValueError, match="contiguous"):
        correlate(queries=(changed,), mutate=use_changed_query)
    with pytest.raises(ValueError, match="perturbation gate"):
        correlate(capture_pass=1)
    with pytest.raises(ValueError, match="perturbation gate"):
        correlate(capture_pass=False)


def test_correlation_rejects_duplicate_quality_rows():
    def duplicate(value):
        value["quality"]["batches"].append(deepcopy(value["quality"]["batches"][0]))

    with pytest.raises(ValueError, match="perturbation gate"):
        correlate(mutate=duplicate)

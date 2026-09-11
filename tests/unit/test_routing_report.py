from dataclasses import replace
from types import SimpleNamespace

import pytest

from frontier.validation.routing_report import (
    audit_routing, compare_routing_passes, graph_routing_enabled,
)
from tests.unit.test_batch_validation import record
from tests.unit.test_routing import sample


MODEL = SimpleNamespace(num_layers=1, num_experts=4, num_experts_per_tok=2, embedding_dim=8, mlp_hidden_dim=16)


def test_graph_routing_accepts_layer_or_decoder_event_capture():
    assert graph_routing_enabled({"graph_event_layers": [0]})
    assert graph_routing_enabled({"graph_event_layers": [], "graph_decoder_event": True})
    assert not graph_routing_enabled({"graph_event_layers": []})


def fixtures():
    rows = [record(rank=rank, batch_id="b", request_ids=("a", "b"), query_lens=(1, 1),
        context_lens=(1024, 1024), prefill_mask=(False, False), step=1, capture_size=3,
        profiled=True, decode_input_sha256="a" * 64) for rank in (0, 1)]
    return rows, [sample(rank=rank) for rank in (0, 1)]


def audit(records, routes, writer=None):
    return audit_routing(records, routes, layer_ids=[0], tensor_parallel_size=2,
                         model_config=MODEL, feature_writer=writer)


def test_full_coverage_rank_agreement_and_native_features():
    records, routes = fixtures()
    features = []
    report, references = audit(records, routes, features.append)
    assert report["all_model_layers_covered"] and report["routing_records"] == 2
    assert not report["tp_logical_mismatches"] and not report["tp_padding_mismatches"]
    assert len(features) == 1  # rank-0 feature export, not TP-summed counts
    assert features[0]["physical_features"]["total_routed_tokens"] == 6
    assert len(references) == 1
    comparison = compare_routing_passes(references, references)
    assert comparison["identical_logical_assignments"] == 1
    assert comparison["logical_histogram_l1_fraction_max"] == 0


def test_tp_route_divergence_is_reported_even_with_identical_histograms():
    records, routes = fixtures()
    routes[1] = sample(ids=[[1, 0], [2, 1], [0, 0]], rank=1)
    report, _ = audit(records, routes)
    assert report["tp_logical_mismatches"] == [("b", 1, 0)]
    assert not report["tp_padding_mismatches"]


def test_paired_pass_comparison_retains_per_token_set_and_histogram_distinctions():
    records, routes = fixtures()
    _, base = audit(records, routes)
    changed = [sample(ids=[[1, 0], [2, 1], [0, 0]], rank=rank) for rank in (0, 1)]
    _, observed = audit(records, changed)
    result = compare_routing_passes(base, observed)
    assert result["identical_logical_assignments"] == 0
    assert result["identical_logical_expert_sets"] == 1
    assert result["logical_histogram_l1_fraction_median"] == 0
    with pytest.raises(ValueError, match="No matching"):
        compare_routing_passes({}, observed)


def test_incomplete_duplicate_wrong_shape_and_greedy_inputs_fail():
    records, routes = fixtures()
    for bad in (routes[:1], routes + routes[:1], [replace(routes[0], batch_id="other"), routes[1]]):
        with pytest.raises(ValueError):
            audit(records, bad)
    with pytest.raises(ValueError, match="fixed-input"):
        audit([replace(r, decode_input_sha256=None) for r in records], routes)
    with pytest.raises(ValueError, match="shape/model"):
        audit([replace(r, capture_size=4) for r in records], routes)

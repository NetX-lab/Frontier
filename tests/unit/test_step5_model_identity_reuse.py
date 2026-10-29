"""Reuse fixed provenance from attached estimators through real public queries."""

from copy import deepcopy
import json

import pytest

from frontier.execution_time_predictor.prediction_model_registry import PredictionModelRegistry
from frontier.types import ClusterType, MeasurementType
from tests.unit.test_step5_measured_ffn_replay import fitted, _batch, _predictor


def _stage(model, predictor, tokens=16):
    return predictor.predict_stage_execution_time(
        _batch(model, tokens), 0, ClusterType.MONOLITHIC,
        layer_id=3, include_attention=False,
    ).layer_execution_times[0]._step5_ffn_operator_costs


def _marker_reads(monkeypatch, markers):
    original = json.loads
    reads = []

    def loads(value, *args, **kwargs):
        if isinstance(value, str) and value in markers:
            reads.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", loads)
    return reads


def _change_marker(estimator, **changes):
    identity = json.loads(estimator._frontier_measurement_identity)
    identity.update(changes)
    marker = PredictionModelRegistry._canonical_measurement_identity(identity)
    estimator._frontier_measurement_identity = marker
    estimator._frontier_kernel_count_model._frontier_measurement_identity = marker
    return marker


def test_attachment_to_public_stage_never_reparses_fixed_identity(fitted, monkeypatch):
    model, manager = fitted
    estimators = list(manager._contract_model_registry("kernel_only").values())
    markers = {estimator._frontier_measurement_identity for estimator in estimators}
    for estimator in estimators:
        manager._set_measurement_identity(estimator, json.loads(estimator._frontier_measurement_identity))
    predictor = _predictor(model, manager)
    reads = _marker_reads(monkeypatch, markers)
    first = _stage(model, predictor)
    second = _stage(model, predictor, 32)
    assert len(first) == len(second) == 9
    assert all(cost["evidence_kind"] == "cpu_fixture" for cost in second.values())
    assert all(cost["kernel_count"] == 7 and cost["exact_lookup"] for cost in second.values())
    assert second["share_expert_up_proj"]["duration_ms"] == .32
    assert reads == [], len(reads)


def test_selected_estimator_marker_mutation_reparses_once_and_replaces_proof(fitted, monkeypatch):
    model, manager = fitted
    predictor = _predictor(model, manager)
    _stage(model, predictor)
    estimator = manager.get_model("share_expert_up_proj", precision="BF16")
    with monkeypatch.context() as change:
        marker = _change_marker(deepcopy(estimator), evidence_kind="cpu_fixture_changed")
        change.setattr(estimator, "_frontier_measurement_identity", marker)
        change.setattr(estimator._frontier_kernel_count_model, "_frontier_measurement_identity", marker)
        reads = _marker_reads(change, {marker})
        first, second = _stage(model, predictor), _stage(model, predictor, 32)
        assert first["share_expert_up_proj"]["evidence_kind"] == "cpu_fixture_changed"
        assert second["share_expert_up_proj"]["evidence_kind"] == "cpu_fixture_changed"
        assert len(reads) == 1, len(reads)
    assert _stage(model, predictor)["share_expert_up_proj"]["evidence_kind"] == "cpu_fixture"


def test_registry_replacement_uses_its_marker_not_old_hash_or_model(fitted, monkeypatch):
    model, manager = fitted
    predictor = _predictor(model, manager)
    _stage(model, predictor)
    estimator = manager.get_model("share_expert_up_proj", precision="BF16")
    replacement = deepcopy(estimator)
    marker = _change_marker(replacement, evidence_kind="cpu_fixture_replacement")
    # Deliberately retain model hashes; a hash is not marker authority.
    bucket = manager._contract_precision_registry("kernel_only")["BF16"]
    key = next(key for key, value in bucket.items() if value is estimator)
    monkeypatch.setitem(bucket, key, replacement)
    reads = _marker_reads(monkeypatch, {marker})
    result = _stage(model, predictor)
    assert result["share_expert_up_proj"]["evidence_kind"] == "cpu_fixture_replacement"
    assert result["share_expert_up_proj"]["duration_model_hash"] == estimator._frontier_model_hash
    assert _stage(model, predictor, 32)["share_expert_up_proj"]["duration_ms"] == .32
    assert len(reads) == 1, len(reads)
    replacement._frontier_kernel_count_model._frontier_exact_lookup[(32.,)] = 0.
    with pytest.raises(ValueError, match="invalid measured kernel-count coverage"):
        _stage(model, predictor, 32)


@pytest.mark.parametrize("marker", ["{invalid", "[]", "null", "{}", None,
                                    '{"evidence_kind":null}', '{"evidence_kind":NaN}',
                                    '{"evidence_kind":""}'])
def test_warm_public_stage_rejects_malformed_changed_identity(fitted, monkeypatch, marker):
    model, manager = fitted
    predictor = _predictor(model, manager)
    _stage(model, predictor)
    estimator = manager.get_model("share_expert_up_proj", precision="BF16")
    monkeypatch.setattr(estimator, "_frontier_measurement_identity", marker)
    monkeypatch.setattr(estimator._frontier_kernel_count_model, "_frontier_measurement_identity", marker)
    with pytest.raises(ValueError):
        _stage(model, predictor)


def test_warm_public_stage_rejects_pair_conflict_and_retains_dynamic_validation(fitted, monkeypatch):
    model, manager = fitted
    predictor = _predictor(model, manager)
    _stage(model, predictor)
    with pytest.raises(ValueError, match="above declared maximum"):
        _stage(model, predictor, 64)
    estimator = manager.get_model("share_expert_up_proj", precision="BF16")
    with pytest.raises(ValueError, match="conflicts"):
        manager._set_measurement_identity(estimator, {"evidence_kind": "conflicting"})
    monkeypatch.setattr(estimator._frontier_kernel_count_model, "_frontier_measurement_identity", "{}")
    with pytest.raises(ValueError, match="measurement identities differ"):
        _stage(model, predictor)


def test_bundle_load_to_public_attention_reuses_identity_and_checks_mutation(fitted, tmp_path, monkeypatch):
    from frontier.execution_time_predictor.step5_cost import predict_step5_attention_time, step5_predictions
    from tests.unit.test_step5_measured_replay import _bundle_identity
    from tests.unit.test_step5_measured_training import _manager, _row, _train, _write

    model, ffn_manager = fitted
    manager = _manager(tmp_path / "cache")
    source = [_row(model, 0, context=8176)]
    models = _train(manager, model, _write(tmp_path, source))
    receipt = manager.publish_model_bundle(_bundle_identity(model, models), models)
    loaded = manager.load_model_bundle(receipt["path"], receipt["identity"])
    predictor = _predictor(model, ffn_manager)
    predictor._block_size = 16
    predictor._step5_max_model_len = 8192
    predictor._models.update(loaded)
    predictor._predictions.update(step5_predictions(loaded, MeasurementType.KERNEL_ONLY))
    batch = _batch(model, routed=False)
    batch.num_context_tokens = [8176]
    root = loaded["step5_layer_e2e"]
    count = loaded["step5_layer_e2e_kernel_count"]
    marker = root._frontier_measurement_identity
    reads = _marker_reads(monkeypatch, {marker})
    for _ in range(2):
        attention = predictor.predict_attention_layer_time(batch, 0, ClusterType.MONOLITHIC)
        assert attention.operator_costs["step5_layer_e2e"].evidence_kind == "cpu_fixture"
    assert reads == [], len(reads)
    changed = json.loads(marker)
    changed["device"] = "wrong_device"
    changed_marker = json.dumps(changed)
    monkeypatch.setattr(root, "_frontier_measurement_identity", changed_marker)
    monkeypatch.setattr(count, "_frontier_measurement_identity", changed_marker)
    with pytest.raises(ValueError, match="measured physical identity"):
        predict_step5_attention_time(predictor, batch, 0)
    monkeypatch.setattr(count, "_frontier_measurement_identity", marker)
    with pytest.raises(ValueError, match="measurement identities differ"):
        predict_step5_attention_time(predictor, batch, 0)

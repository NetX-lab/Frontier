"""Fixed Step bindings are reused without retaining dynamic measured outputs."""

import cProfile
from collections import Counter
from copy import deepcopy

import pytest

from frontier.execution_time_predictor.step5_cost import step5_batch_features
from frontier.moe_load_imbalance import MOE_LOAD_IMBALANCE_FEATURES, MoELoadImbalanceInput
from frontier.types import ClusterType, MeasurementType
import tests.unit.test_step5_measured_ffn as ffn_fixture
from tests.unit.test_step5_measured_training import _manager
from tests.unit.test_step5_measured_ffn_replay import fitted, _batch, _predictor


def _calls(profile):
    return Counter({name: sum(entry.callcount for entry in profile.getstats()
                             if hasattr(entry.code, "co_name") and entry.code.co_name == name)
                    for name in ("describe_step5_layer", "resolve_module_precision",
                                 "resolve_layer_contract", "_get_family_model")})


def _features(model, tokens):
    batch = _batch(model, tokens)
    features = MoELoadImbalanceInput(
        num_tokens=tokens, num_experts_per_device=batch.lane_workload.local_expert_width,
        hidden_dim=model.embedding_dim, expert_hidden_dim=model.mlp_hidden_dim,
        router_topk=model.num_experts_per_tok,
        expert_token_counts=list(batch.lane_workload.local_token_counts),
    ).to_features_dict()
    return {name: features[name] for name in MOE_LOAD_IMBALANCE_FEATURES}


def test_repeated_real_queries_construct_fixed_bindings_once(fitted):
    model, manager = fitted
    predictor = _predictor(model, manager)
    predictor._block_size = 16
    predictor._step5_max_model_len = 8192
    batch = _batch(model, routed=False)
    batch.num_context_tokens = [16]
    profile = cProfile.Profile()
    profile.enable()
    for iteration in range(3):
        if iteration == 1:
            profile.disable()
            assert _calls(profile)["describe_step5_layer"] == 2
            profile = cProfile.Profile()
            profile.enable()
        for layer, precision, duration in ((3, "FP8", .32), (88, "BF16", .48)):
            with predictor._step5_measured_layer(layer):
                value = predictor._predict_step5_ffn_operator("moe_grouped_gemm", _features(model, 16))
                cost = predictor._step5_ffn_costs["moe_grouped_gemm"]
                assert value == pytest.approx(duration)
                assert cost["precision"] == precision and cost["kernel_count"] == 7
                assert cost["global_layer_id"] == layer and cost["exact_lookup"]
                features, workload = step5_batch_features(predictor, batch, layer)
                assert workload.valid_tokens == 16
                assert features["global_layer_id"] == layer
    profile.disable()
    counts = _calls(profile)
    assert counts["describe_step5_layer"] == 0, counts
    assert counts["resolve_module_precision"] == 0, counts
    assert counts["resolve_layer_contract"] == 0, counts
    assert counts["_get_family_model"] == 4, counts


def test_warm_binding_still_validates_dynamic_features_and_count_model(fitted, monkeypatch):
    model, manager = fitted
    predictor = _predictor(model, manager)
    with predictor._step5_measured_layer(3):
        assert predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 16}) == .16
        assert predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 32}) == .32
        for features, message in (({"num_tokens": 64}, "above declared maximum"),
                                  ({"num_tokens": float("nan")}, "finite"),
                                  ({"num_tokens": 16, "extra": 1}, "unexpected features")):
            with pytest.raises(ValueError, match=message):
                predictor._predict_step5_ffn_operator("share_expert_up_proj", features)
        estimator = manager.get_model("share_expert_up_proj", precision="BF16")
        with monkeypatch.context() as patch:
            patch.delattr(estimator, "_frontier_kernel_count_model")
            with pytest.raises(ValueError, match="actual per-operator kernel-count coverage"):
                predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 16})


def test_warm_binding_selects_current_registry_and_rejects_invalid_count(fitted, monkeypatch):
    model, manager = fitted
    predictor = _predictor(model, manager)
    with predictor._step5_measured_layer(3):
        assert predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 16}) == .16
        estimator = manager.get_model("share_expert_up_proj", precision="BF16")
        replacement = deepcopy(estimator)
        replacement._frontier_kernel_count_model._frontier_exact_lookup[(32.,)] = 0.
        bucket = next(bucket for bucket in manager._contract_precision_registry("kernel_only").values()
                      if estimator in bucket.values())
        key = next(key for key, value in bucket.items() if value is estimator)
        monkeypatch.setitem(bucket, key, replacement)
        predictions = predictor._predictions
        with pytest.raises(ValueError, match="invalid measured kernel-count coverage"):
            predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 32})
        assert predictor._predictions is predictions
        monkeypatch.delitem(bucket, key)
        with pytest.raises(ValueError, match="coverage is unavailable"):
            predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 16})


def test_nested_layers_restore_active_metadata_and_costs_after_failure(fitted):
    model, manager = fitted
    predictor = _predictor(model, manager)
    with predictor._step5_measured_layer(3) as outer:
        predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 16})
        outer_costs = predictor._step5_ffn_costs
        with pytest.raises(ValueError, match="above declared maximum"):
            with predictor._step5_measured_layer(88) as inner:
                assert inner is not outer
                assert predictor._step5_ffn_costs == {}
                predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 64})
        assert predictor._step5_layer_metadata is outer
        assert predictor._step5_ffn_costs is outer_costs
        with predictor._step5_measured_layer(3) as same:
            assert same is outer and predictor._step5_ffn_costs == {}
        assert predictor._step5_ffn_costs is outer_costs
    assert predictor._step5_layer_metadata is None and predictor._step5_ffn_costs is None


def test_predictor_instances_bind_layers_independently(fitted):
    model, manager = fitted
    first, second = _predictor(model, manager), _predictor(model, manager)
    first._cluster_type = ClusterType.PREFILL
    first._replica_config.attn_dp = 1
    second._cluster_type = ClusterType.DECODE
    second._replica_config.attn_tensor_parallel_size = 4
    second._replica_config.attn_dp = 2
    with first._step5_measured_layer(3) as first_layer:
        first._predict_step5_ffn_operator("moe_grouped_gemm", _features(model, 16))
        profile = cProfile.Profile()
        profile.enable()
        with second._step5_measured_layer(3) as second_layer:
            second._predict_step5_ffn_operator("moe_grouped_gemm", _features(model, 16))
        profile.disable()
        assert first_layer is not second_layer
        assert first_layer.geometry.local_q_heads == 8
        assert second_layer.geometry.local_q_heads == 16
        assert _calls(profile)["describe_step5_layer"] == 1
    with second._step5_measured_layer(88):
        second._predict_step5_ffn_operator("moe_grouped_gemm", _features(model, 16))
        assert second._step5_ffn_costs["moe_grouped_gemm"]["precision"] == "BF16"
    with pytest.raises(ValueError, match="global_layer_id"):
        with first._step5_measured_layer(True):
            pass


def test_binding_retains_measurement_family_selection_and_missing_family_error(fitted):
    model, manager = fitted
    predictor = _predictor(model, manager)
    with predictor._step5_measured_layer(3):
        predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 16})
        with predictor._temporary_measurement_type(MeasurementType.CUDA_EVENT):
            with pytest.raises(ValueError, match="coverage is unavailable"):
                predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 16})
        assert predictor._active_measurement_type is MeasurementType.KERNEL_ONLY
        assert predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 32}) == .32


def test_two_measured_families_share_fixed_binding_not_model_or_proof(fitted, tmp_path, monkeypatch):
    model, manager = fitted
    source_metadata = ffn_fixture._metadata
    with monkeypatch.context() as patch:
        patch.setattr(ffn_fixture, "_metadata", lambda config: {
            **source_metadata(config), "measurement_type": MeasurementType.CUDA_EVENT.value,
        })
        paths = ffn_fixture._tables(tmp_path, model)
    event_manager = _manager(tmp_path / "cache", MeasurementType.CUDA_EVENT)
    ffn_fixture._ffn(event_manager, model, paths)
    for name in ("_trained_models_eager", "_trained_models_eager_by_contract",
                 "_models_by_precision_eager", "_models_by_precision_eager_by_contract"):
        monkeypatch.setattr(manager, name, getattr(event_manager, name))
    predictor = _predictor(model, manager)
    costs = []
    profile = cProfile.Profile()
    with predictor._step5_measured_layer(3):
        predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 16})
        costs.append(dict(predictor._step5_ffn_costs["share_expert_up_proj"]))
        profile.enable()
        with predictor._temporary_measurement_type(MeasurementType.CUDA_EVENT):
            assert predictor._predict_step5_ffn_operator("share_expert_up_proj", {"num_tokens": 32}) == .32
            costs.append(dict(predictor._step5_ffn_costs["share_expert_up_proj"]))
        profile.disable()
    assert [cost["measurement_type"] for cost in costs] == ["KERNEL_ONLY", "CUDA_EVENT"]
    assert [cost["duration_ms"] for cost in costs] == [.16, .32]
    assert costs[0]["duration_model_hash"] != costs[1]["duration_model_hash"]
    assert costs[0]["kernel_count_model_hash"] != costs[1]["kernel_count_model_hash"]
    assert _calls(profile)["resolve_layer_contract"] == 0
    assert _calls(profile)["_get_family_model"] == 1

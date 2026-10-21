"""Actual Step FFN cache lookup and costs on explicitly unadmitted CPU fixtures."""

from collections import defaultdict
from types import SimpleNamespace

import pytest

from frontier.attention.step5 import describe_step5_layer
from frontier.config.model_config import BaseModelConfig
from frontier.entities import ExecutionTime, StageExecutionTime
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import SklearnMoEExecutionTimePredictor
from frontier.moe_ep_workload import EPLaneWorkload
from frontier.moe_load_imbalance import MoELoadImbalanceInput
from frontier.types import ClusterType, MeasurementType
from tests.unit.test_step5_measured_ffn import _config, _ffn, _replica, _tables
from tests.unit.test_step5_measured_training import _manager


class _MeasuredMoE(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        raise AssertionError("estimator fitting belongs to the real shared manager")

    def _get_estimator(self):
        raise AssertionError("estimator fitting belongs to the real shared manager")


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    path = tmp_path_factory.mktemp("step5_ffn_replay")
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    paths = _tables(path, model)
    cold = _manager(path / "cache")
    _ffn(cold, model, paths)
    warm = _manager(path / "cache")
    _ffn(warm, model, paths)
    warm._train_step5_input_layernorm_for_cluster(
        ClusterType.MONOLITHIC, _replica(model), _config(), str(paths[0]), set(),
    )
    return model, warm


def _lane(model, tokens=16):
    load = MoELoadImbalanceInput(
        num_tokens=tokens, num_experts_per_device=model.num_experts // 8,
        hidden_dim=model.embedding_dim, expert_hidden_dim=model.mlp_hidden_dim,
        router_topk=model.num_experts_per_tok,
    )
    local = tuple(load.expert_token_counts)
    return EPLaneWorkload(
        ep_id=0, moe_expert_parallel_size=8, total_expert_num=model.num_experts,
        owned_expert_ids=tuple(range(len(local))), local_token_counts=local,
        global_token_counts=local + (0,) * (model.num_experts - len(local)),
        routed_token_count=sum(local), router_topk=model.num_experts_per_tok,
    )


def _batch(model, tokens=16, *, routed=True):
    return SimpleNamespace(
        id="cpu_fixture", size=1, requests=[], is_idle=False, num_tokens=[tokens],
        total_num_tokens=tokens * model.num_experts_per_tok if routed else tokens,
        num_prefill_tokens=tokens, num_decode_tokens=0,
        lane_workload=_lane(model, tokens) if routed else None,
        get_effective_total_tokens_rounded=lambda _: tokens,
        get_effective_total_tokens_for_compute=lambda _: tokens,
        get_decode_cuda_graph_runtime_mode=lambda: "NONE",
    )


def _predictor(model, manager):
    predictor = _MeasuredMoE.__new__(_MeasuredMoE)
    predictor._model_config = model
    predictor._replica_config = _replica(model)
    predictor._replica_config.num_pipeline_stages = 1
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._model_manager = manager
    predictor._config = _config()
    predictor._enable_dummy_mode = False
    predictor._router_topk = model.num_experts_per_tok
    predictor._moe_tp_size = 1
    predictor._moe_ep_size = 8
    predictor._kernel_gap_ms = {"eager": .001, "cuda_graph": .002}
    predictor._active_measurement_type = MeasurementType.KERNEL_ONLY
    predictor._step_measurement_type = MeasurementType.CUDA_EVENT
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._models = {
        **manager._trained_models_kernel_only,
        **{name: estimator for (name, _), estimator in manager._contract_model_registry("kernel_only").items()},
    }
    predictor._predictions = {
        name: {"_on_demand_prediction": True, "_model": estimator,
               "_feature_names": tuple(estimator._frontier_feature_names)}
        for name, estimator in predictor._models.items()
        if not name.endswith("_kernel_count")
    }
    predictor._two_stream_eager_pricing = True
    predictor._attention_query_cache_hits = predictor._attention_query_cache_misses = 0
    for suffix in ("eager", "device_event", "kernel_only"):
        setattr(predictor, f"_models_{suffix}", predictor._models if suffix == "kernel_only" else {})
        setattr(predictor, f"_predictions_{suffix}", predictor._predictions if suffix == "kernel_only" else {})
        for kind in ("compute", "attention", "moe", "cpu_overhead"):
            setattr(predictor, f"_{kind}_input_file_{suffix}", "")
    # Explicit zero fixture isolates measured compute from external CC/PD and
    # stage-owned CPU work. No compute predictor or estimator is replaced.
    predictor._predict_comm_operator = lambda *args, **kwargs: 0.
    predictor.predict_dp_moe_allreduce_times = lambda *args, **kwargs: (0., 0.)
    predictor._get_operator_calibration_scale = lambda _: 1.
    predictor._get_optional_operator_phase_calibration_scale = lambda *args: None
    for name in ("_get_schedule_time", "_get_sampler_e2e_time", "_get_prepare_inputs_e2e_time",
                 "_get_process_model_outputs_time", "_get_ray_comm_time", "_get_forward_launch_time",
                 "_get_forward_drain_time", "_get_pp_producer_send_path_runtime_time",
                 "_get_pp_receiver_head_runtime_time", "_get_pp_prefill_consumer_active_runtime_time",
                 "_get_pp_stage_boundary_handoff_time"):
        setattr(predictor, name, lambda *args, **kwargs: 0.)
    return predictor


@pytest.mark.parametrize("layer", [0, 91])
def test_real_dense_layer_replay_uses_bf16_mlp_not_shared(fitted, layer):
    model, manager = fitted
    predictor = _predictor(model, manager)
    batch = _batch(model, routed=False)
    result = predictor._predict_layer_with_kernel_gap(
        predictor._get_execution_time_internal, batch, 0,
        include_moe=False, include_attention=False, layer_id=layer, include_stage_owned=False,
    )
    assert isinstance(result, ExecutionTime)
    proof = result._step5_ffn_operator_costs
    assert set(proof) == {"mlp_up_proj", "mlp_down_proj", "mlp_act", "post_attention_layernorm"}
    assert {cost["precision"] for cost in proof.values()} == {"BF16"}
    assert {cost["kernel_count"] for cost in proof.values()} == {7.}
    assert result.moe_or_mlp_time_component.total_time() == pytest.approx(.16 * 4)
    assert result.kernel_gap_time == pytest.approx(.001 * 7 * 4)
    assert result.get_single_layer_post_attention_time() == pytest.approx(.16 * 4 + .028)
    assert "step5_layer_e2e" not in result.op_times


@pytest.mark.parametrize("layer,precision,duration", [(3, "FP8", .32), (88, "BF16", .48)])
def test_real_routed_layer_replay_prices_source_precision_and_merge_once(fitted, layer, precision, duration):
    model, manager = fitted
    predictor = _predictor(model, manager)
    batch = _batch(model)
    result = predictor._predict_layer_with_kernel_gap(
        predictor._get_execution_time_internal, batch, 0,
        moe_tokens_input=batch.lane_workload, include_attention=False,
        layer_id=layer, include_stage_owned=False,
    )
    proof = result._step5_ffn_operator_costs
    assert proof["moe_grouped_gemm"]["precision"] == precision
    assert proof["moe_grouped_gemm"]["duration_ms"] == pytest.approx(duration)
    assert proof["moe_gating_linear"]["precision"] == "FP32"
    assert proof["share_expert_up_proj"]["precision"] == "BF16"
    assert proof["step5_ffn_merge"]["duration_ms"] == .16
    assert {cost["kernel_count"] for cost in proof.values()} == {7.}
    assert {cost["evidence_kind"] for cost in proof.values()} == {"cpu_fixture"}
    assert len(proof) == 9
    assert result.moe_operator_times.op_times["step5_ffn_merge"] == .16
    expected = sum(cost["duration_ms"] for cost in proof.values())
    assert result.moe_or_mlp_time_component.total_time() == pytest.approx(expected)
    assert result.kernel_gap_time == pytest.approx(.001 * 7 * 9)
    assert result.get_single_layer_post_attention_time() == pytest.approx(expected + .063)
    stage = predictor._assemble_stage([result], first_layer_id=layer)
    assert isinstance(stage, StageExecutionTime)
    assert stage.layer_execution_times[0].global_layer_id == layer
    assert stage.layer_execution_times[0]._step5_ffn_operator_costs == proof


def test_public_moe_component_matches_the_execution_component(fitted):
    model, manager = fitted
    predictor = _predictor(model, manager)
    batch = _batch(model)
    component = predictor.predict_moe_layer_time(batch, 88, ClusterType.MONOLITHIC)
    assert component.operator_times.op_times["step5_ffn_merge"] == .16
    assert component._step5_ffn_operator_costs["moe_grouped_gemm"]["precision"] == "BF16"


def test_public_stage_replays_typed_cache_costs_without_attention_or_cpu_costs(fitted):
    model, manager = fitted
    predictor = _predictor(model, manager)
    stage = predictor.predict_stage_execution_time(
        _batch(model), 0, ClusterType.MONOLITHIC, layer_id=3, include_attention=False,
    )
    layer = stage.layer_execution_times[0]
    assert layer.kernel_gap_time == pytest.approx(.001 * 7 * 9)
    assert layer._step5_ffn_operator_costs["moe_grouped_gemm"]["precision"] == "FP8"
    assert layer.moe_operator_times.op_times["step5_ffn_merge"] == .16
    expected = sum(cost["duration_ms"] for cost in layer._step5_ffn_operator_costs.values())
    assert layer.get_single_layer_post_attention_time() == pytest.approx(expected + .063)


def test_measured_ffn_refuses_outside_token_bounds_and_missing_gap(fitted):
    model, manager = fitted
    predictor = _predictor(model, manager)
    with pytest.raises(ValueError, match="above declared maximum"):
        predictor._get_execution_time_internal(
            _batch(model, 64, routed=False), 0, include_moe=False,
            include_attention=False, layer_id=0, include_stage_owned=False,
        )
    predictor._kernel_gap_ms = None
    with pytest.raises(ValueError, match="kernel_gap.csv coverage"):
        predictor._get_execution_time_internal(
            _batch(model, routed=False), 0, include_moe=False,
            include_attention=False, layer_id=0, include_stage_owned=False,
        )


def test_gpu_active_runtime_rejects_a_missing_count_model(fitted, monkeypatch):
    model, manager = fitted
    predictor = _predictor(model, manager)
    estimator = manager.get_model("mlp_up_proj", precision="BF16")
    monkeypatch.delattr(estimator, "_frontier_kernel_count_model")
    with pytest.raises(ValueError, match="actual per-operator kernel-count coverage"):
        predictor._get_execution_time_internal(
            _batch(model, routed=False), 0, include_moe=False,
            include_attention=False, layer_id=0, include_stage_owned=False,
        )


def test_attention_cache_preserves_proof_and_does_not_invent_layer_coverage(fitted, tmp_path):
    from tests.unit.test_step5_measured_training import _row, _train, _write
    from frontier.execution_time_predictor.step5_cost import step5_predictions

    model, manager = fitted
    models = _train(manager, model, _write(tmp_path, [_row(model, 0, context=8176)]))
    predictor = _predictor(model, manager)
    predictor._models.update(models)
    predictor._predictions.update(step5_predictions(models, MeasurementType.KERNEL_ONLY))
    predictor._block_size = 16
    predictor._step5_max_model_len = 8192
    batch = _batch(model, routed=False)
    batch.num_context_tokens = [8176]
    cache = {}
    first = predictor._predict_attention_layer_time_with_query_cache(
        batch=batch, layer_id=0, cluster_type=ClusterType.MONOLITHIC, cache=cache,
    )
    cached = predictor._predict_attention_layer_time_with_query_cache(
        batch=batch, layer_id=0, cluster_type=ClusterType.MONOLITHIC, cache=cache,
    )
    assert first.operator_costs == cached.operator_costs
    assert cached.operator_costs["step5_layer_e2e"].evidence_kind == "cpu_fixture"
    with pytest.raises(ValueError, match="coverage is unavailable"):
        predictor._predict_attention_layer_time_with_query_cache(
            batch=batch, layer_id=1, cluster_type=ClusterType.MONOLITHIC, cache=cache,
        )


@pytest.mark.parametrize("layer", [0, 91])
def test_full_stage_never_omits_unmeasured_native_boundary_steps(fitted, layer):
    model, manager = fitted
    predictor = _predictor(model, manager)
    with pytest.raises(ValueError, match="unavailable source boundary coverage"):
        predictor.predict_stage_execution_time(_batch(model), 0, ClusterType.MONOLITHIC, layer_id=layer)


def test_all92_source_classification_retains_precisions_and_physical_geometry(fitted):
    model, _ = fitted
    for layer in range(model.num_layers):
        metadata = describe_step5_layer(model, layer, attention_tp_size=8,
                                       moe_tp_size=1, expert_parallel_size=8)
        projections = {p.operator_name: p.weight_precision.name for p in metadata.projections}
        if model.is_moe_layer(layer):
            assert projections["moe_grouped_gemm"] == ("FP8" if layer < 88 else "BF16")
            assert projections["moe_gating_linear"] == "FP32"
            assert projections["share_expert_up_proj"] == "BF16"
        else:
            assert projections["mlp_up_proj"] == "BF16"
        assert metadata.geometry.head_dim == 192
        assert metadata.dense_swiglu_limit == 7.

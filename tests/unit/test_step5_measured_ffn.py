"""Real shared-cache fitting of explicitly labelled CPU Step compute fixtures."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.attention.step5 import describe_step5_layer
from frontier.config.config import RandomForrestExecutionTimePredictorConfig
from frontier.config.model_config import BaseModelConfig
from frontier.config.precision_type import PrecisionType
from frontier.operators.families import MOE_FAMILY
from frontier.moe_gating_runtime import DEFAULT_MOE_GATING_RUNTIME_CONTEXT
from frontier.moe_load_imbalance import MoELoadImbalanceInput
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.profiling.utils.replicated_ops import split_replicated_result
from frontier.profiling.moe.routed_layer_groups import resolve_routed_layer_groups
from frontier.types import ClusterType, MeasurementType
from tests.unit.test_step5_measured_training import _manager


def test_position_runtime_requires_actual_empty_physical_exact_pair(model, monkeypatch):
    from tests.unit.test_moe_grouped_gemm_padded_features import _ConcreteMoEPredictor
    from frontier.execution_time_predictor import sklearn_moe_execution_time_predictor as runtime
    predictor = object.__new__(_ConcreteMoEPredictor)
    predictor._step5_layer_metadata = SimpleNamespace(attention_spec=SimpleNamespace(global_layer_id=3))
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._predictions, predictor._step5_ffn_costs = {}, {}
    native = dict(model_architecture_profile="step5", routing_runtime_path="position_cyclic",
                  quant_dtype=None, defer_input_quant=False)
    count = SimpleNamespace(_frontier_feature_names=["num_tokens", "live_routing_token_count"],
        _frontier_exact_lookup={(8., 0.): 3., (8., 4.): 3.}, _frontier_measurement_identity=json.dumps(dict(evidence_kind="cpu_fixture", native_context=json.dumps(native))),
        _frontier_model_hash="count", _frontier_target_col="time_stats.moe_gating_linear.kernel_count")
    fitted = SimpleNamespace(_frontier_feature_names=count._frontier_feature_names,
        _frontier_exact_lookup={(8., 0.): 1., (8., 4.): 2.}, _frontier_kernel_count_model=count,
        _frontier_measurement_identity=count._frontier_measurement_identity, _frontier_model_hash="duration",
        _frontier_target_col="time_stats.moe_gating_linear.median")
    predictor._model_manager = SimpleNamespace(_get_family_model=lambda *args, **kwargs: fitted)
    monkeypatch.setattr(runtime, "step5_operator_binding", lambda *args, **kwargs: (PrecisionType.FP32, None))
    assert predictor._get_on_demand_prediction("moe_gating_linear", dict(num_tokens=8, live_routing_token_count=0), context=native) == 1.
    assert predictor._step5_ffn_costs["moe_gating_linear"]["kernel_count"] == 3.
    count._frontier_exact_lookup.pop((8., 0.))
    with pytest.raises(ValueError, match="exact feature keys differ"):
        predictor._predict_step5_ffn_operator("moe_gating_linear", dict(num_tokens=8, live_routing_token_count=0), context=native)
    fitted._frontier_exact_lookup.pop((8., 0.))
    with pytest.raises(ValueError, match="requires an exact measured"):
        predictor._predict_step5_ffn_operator("moe_gating_linear", dict(num_tokens=8, live_routing_token_count=0), context=native)
    with pytest.raises(ValueError, match="requires an exact measured"):
        predictor._get_prediction_for_features("moe_gating_linear", dict(num_tokens=8, live_routing_token_count=2), context=native)
    with pytest.raises(ValueError, match="not an applicable"):
        predictor._get_on_demand_prediction("moe_shuffling", dict(num_tokens=8), context=native)


def test_position_real_estimator_reaches_execution_time_without_shuffle(model,tmp_path,monkeypatch):
    from frontier.config import global_vars
    from tests.integration import step5_measured_compute_fixture as fixture
    from tests.unit.test_step5_measured_ffn_replay import _predictor
    from tests.unit.test_moe_position_routing import source,materialize
    from frontier.execution_time_predictor.moe_dataset_training import position_moe_operator_facts
    from frontier.cc_backend.collective_plan import CollectivePlanCost
    from frontier.config.replica_config import ReplicaConfig
    monkeypatch.setattr(global_vars,"CUDAGRAPH_CAPTURE_SIZES",list(fixture.CAPTURE_SIZES))
    monkeypatch.setattr(global_vars,"DECODE_CUDA_GRAPH_MODE","full_and_piecewise")
    monkeypatch.setattr(fixture,"range",lambda *a:(6144,) if a==(6143,7167) else range(*a),raising=False)
    paths,_=fixture.position_compute_tables(tmp_path,model,device="h800",measurement=MeasurementType.KERNEL_ONLY)
    replica=ReplicaConfig(model_name=model._model_name,device="h800",attn_tensor_parallel_size=8,attn_dp=2,moe_expert_parallel_size=16,moe_routing_distribution_type="position_cyclic")
    manager=_manager(tmp_path/"cache");facts=position_moe_operator_facts(pd.read_csv(paths[1]),replica)[1]
    manager._train_ffn_models_for_cluster(ClusterType.MONOLITHIC,replica,_config(),str(paths[0]),str(paths[1]),True,set(),operator_facts=facts)
    p=_predictor(model,manager);p._replica_config=replica;p._moe_ep_size=16;p._moe_gating_routing_runtime_path="position_cyclic"
    p._moe_input_file,p._moe_routing_distribution_type=str(paths[1]),"position_cyclic"
    p._register_additional_profiling_metadata_from_files()
    lane=materialize(source(padding=16)).lane(0)
    path=fixture.position_collective_costs(tmp_path,paths[1],device="h800");artifact=json.loads(path.read_text())
    p._cc_backend=SimpleNamespace(collective_profile_bytes=lambda raw=path.read_bytes():raw,
        predict_collective_plan=lambda plan:CollectivePlanCost(**artifact["counts"][plan.cache_key()]["cost"]))
    with p._step5_measured_layer(88):
        result=p._predict_step5_position_execution(SimpleNamespace(),lane,layer_id=88,include_attention=False,include_stage_owned=False,pipeline_stage=0)
        assert result.moe_or_mlp_time_component.operator_times.physical_frame["live_local_assignments"]==0
        assert result.op_times["moe_grouped_gemm"]>0 and "moe_shuffling" not in result.op_times and result.get_single_layer_post_attention_time()>.12
        p._step5_ffn_model("post_attention_layernorm")[0]._frontier_kernel_count_model._frontier_exact_lookup.pop((16.,))
        with pytest.raises(ValueError,match="exact feature keys differ"):
            p._predict_step5_position_execution(SimpleNamespace(),lane,layer_id=88,include_attention=False,include_stage_owned=False,pipeline_stage=0)


def test_native_feature_schema_keeps_exact_order_backend_and_legacy_ranges():
    from frontier.moe_load_imbalance import moe_grouped_gemm_schema, MOE_GROUPED_GEMM_PADDED_FEATURES
    legacy = moe_grouped_gemm_schema(MOE_GROUPED_GEMM_PADDED_FEATURES)
    assert legacy.uses_block_size_m_ranges and legacy.experts_class is None
    from frontier.execution_time_predictor import step5_cost
    if not hasattr(step5_cost, "STEP5_NATIVE_EXPERT_SCHEMAS"):
        pytest.skip("PRICING06bae09 selected schema registry is not yet import-authorized")
    for experts, features in (("TritonExperts", ("num_tokens", "total_routed_tokens", "num_tokens_post_padded")),
            ("DeepGemmExperts", ("num_tokens", "total_routed_tokens", "native_capacity_rows", "native_alignment"))):
        context = dict(experts_class=experts, physical_scope="native_experts_inclusive")
        selected = moe_grouped_gemm_schema(features, native_context=context)
        assert selected.features == features and not selected.uses_block_size_m_ranges
        for invalid in (features[::-1], features + ("other",), MOE_GROUPED_GEMM_PADDED_FEATURES):
            with pytest.raises(ValueError, match="actual selected native expert"):
                moe_grouped_gemm_schema(invalid, native_context=context)
        with pytest.raises(ValueError, match="actual selected native expert"):
            moe_grouped_gemm_schema(features, native_context={**context, "experts_class": "unselected"})
    assert moe_grouped_gemm_schema(("num_tokens",)) is None


def test_step_selected_model_uses_real_precision_registry_not_generic_prediction(tmp_path, monkeypatch):
    from tests.unit.test_moe_grouped_gemm_padded_features import _ConcreteMoEPredictor
    from frontier.execution_time_predictor import sklearn_moe_execution_time_predictor as runtime
    from frontier.execution_time_predictor.prediction_model_identity import _serialize_selected_layer_cache_identity
    from tests.unit.test_step5_measured_training import _manager
    config = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    manager = _manager(tmp_path / "cache", MeasurementType.CUDA_EVENT)
    contract = config.get_model_architecture_profile().resolve_layer_contract(config,
        layer_id=3, operator_name="moe_grouped_gemm", moe_tp_size=1, expert_parallel_size=16)
    identity = _serialize_selected_layer_cache_identity(contract)
    fitted = {precision: SimpleNamespace(_frontier_layer_cache_identity=identity) for precision in (PrecisionType.FP8, PrecisionType.BF16)}
    for precision, model in fitted.items():
        manager._store_model_precision("moe_grouped_gemm", precision.name, model, layer_contract=contract)
    predictor = object.__new__(_ConcreteMoEPredictor)
    predictor._model_manager, predictor._active_measurement_type = manager, MeasurementType.CUDA_EVENT
    predictor._predictions = {"moe_grouped_gemm": pytest.fail}
    predictor._step5_layer_metadata = SimpleNamespace(attention_spec=SimpleNamespace(global_layer_id=3))
    monkeypatch.setattr(runtime, "step5_operator_binding", lambda *a, context, **k: (context["precision"], identity))
    for precision in fitted:
        assert predictor._step5_ffn_model("moe_grouped_gemm", context={"precision": precision}) == (fitted[precision], precision)


@pytest.fixture(scope="module")
def model():
    return BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")


def _replica(model):
    return SimpleNamespace(
        model_config=model, device="h800", model_name="stepfun-ai/Step-5-Preview",
        attn_tensor_parallel_size=8, moe_tensor_parallel_size=1, moe_expert_parallel_size=8,
        moe_routing_distribution_type="balanced", speculative_decoding_config=None,
    )


def _stats(tokens, *, scale=1.):
    value = scale * tokens / 100
    return dict(min=value, max=value, mean=value, median=value, std=0., count=10, kernel_count=7)


def _metadata(model):
    return dict(
        model_architecture_profile="step5", model_arch=model.model_arch,
        quant_signature=model.get_quant_signature(),
        profiling_precision="BF16", measurement_type="KERNEL_ONLY",
        runtime_stack_signature="cpu_fixture_unadmitted", evidence_kind="cpu_fixture",
    )


def _tables(tmp_path, model, token_counts=(16, 32), *, expert_parallel_size=8,
            include_stage_boundaries=False):
    linear = []
    for is_moe in (False, True):
        plan = build_profiling_plan(model, tp_size=8, attn_tp=[8], ffn_tp=[8], moe_tp=[1], is_moe=is_moe)
        for tokens in token_counts:
            row = dict(
                num_tokens=tokens, num_tensor_parallel_workers=8, n_embd=model.embedding_dim,
                n_expanded_embd=plan["padded_n_expanded_embd"],
                padded_n_embd=plan["padded_n_embd"],
                padded_n_expanded_embd=plan["padded_n_expanded_embd"],
                time_stats={name: _stats(tokens) for name in plan["enabled_ops"]},
                typed_operator_contracts=plan["typed_operator_contracts"], **_metadata(model),
            )
            for physical in split_replicated_result(row, set(plan["replicated_ops"]),
                                                    unpadded_n_embd=model.embedding_dim):
                # The Step plan measures both stage boundaries in its TP1 rows.
                if not include_stage_boundaries:
                    for name in ("input_cast", "final_residual"):
                        physical["time_stats"].pop(name, None)
                        physical["typed_operator_contracts"].pop(name, None)
                physical["typed_operator_contracts"] = json.dumps(physical["typed_operator_contracts"])
                linear.append(physical)
    linear_path = tmp_path / "linear_op.csv"
    pd.json_normalize(linear).drop_duplicates().to_csv(linear_path, index=False)
    routed = []
    replica = _replica(model)
    replica.moe_expert_parallel_size = expert_parallel_size
    manager = _manager(tmp_path / "contracts")
    for group in resolve_routed_layer_groups(model):
        for tokens in token_counts:
            contracts = {}
            for operator in MOE_FAMILY.profiling_ops():
                contract = manager._resolve_typed_layer_contract(
                    operator.name, ClusterType.MONOLITHIC, replica, is_moe_model=True,
                    layer_id=group.layer_ids[0],
                )
                contracts[operator.name] = contract.typed_metadata_identity()
            routed.append(dict(
                num_tokens=tokens, num_tensor_parallel_workers=1, expert_parallel_size=expert_parallel_size,
                num_experts=model.num_experts, router_topk=model.num_experts_per_tok,
                hidden_dim=model.embedding_dim,
                expert_hidden_dim=contracts["moe_grouped_gemm"]["effective_ffn_width"],
                typed_operator_contracts=json.dumps(contracts),
                time_stats={op.name: _stats(tokens, scale=2. if group.precision is PrecisionType.FP8 else 3.)
                            for op in MOE_FAMILY.profiling_ops()},
                routing_runtime_path="standard_fused_topk", gating_runtime_context=DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
                moe_quantization_mode=group.precision.value,
                **group.row_metadata(), **_metadata(model),
            ))
            routed[-1].update(MoELoadImbalanceInput(
                num_tokens=tokens, num_experts_per_device=model.num_experts // expert_parallel_size,
                hidden_dim=model.embedding_dim,
                expert_hidden_dim=contracts["moe_grouped_gemm"]["effective_ffn_width"],
                router_topk=model.num_experts_per_tok,
            ).to_features_dict())
    moe_path = tmp_path / "moe.csv"
    pd.json_normalize(routed).to_csv(moe_path, index=False)
    return linear_path, moe_path


def _config():
    return RandomForrestExecutionTimePredictorConfig(
        num_estimators=[2], max_depth=[3], min_samples_split=[2],
        k_fold_cv_splits=2, num_training_job_threads=1,
    )


def _ffn(manager, model, paths):
    manager._cluster_configs = {ClusterType.MONOLITHIC: SimpleNamespace(replica_config=_replica(model))}
    return manager._train_ffn_models_for_cluster(
        ClusterType.MONOLITHIC, _replica(model), _config(), str(paths[0]), str(paths[1]), True, set(),
    )


def test_step_existing_ffn_loop_fits_dense_shared_and_both_routed_precisions(model, tmp_path):
    manager = _manager(tmp_path / "cache")
    paths = _tables(tmp_path, model)
    models = _ffn(manager, model, paths)
    assert {"mlp_up_proj", "mlp_down_proj", "mlp_act", "share_expert_up_proj",
            "share_expert_down_proj", "share_expert_act", "moe_grouped_gemm",
            "post_attention_layernorm"} <= set(models)
    contract = manager._resolve_typed_layer_contract(
        "moe_grouped_gemm", ClusterType.MONOLITHIC, _replica(model), is_moe_model=True, layer_id=3,
    )
    for precision, expected in ((PrecisionType.FP8, .32), (PrecisionType.BF16, .48)):
        estimator = manager.get_model("moe_grouped_gemm", precision=precision.name, layer_contract=contract)
        assert estimator is not None
        row = pd.read_csv(paths[1], float_precision="round_trip").query("num_tokens == 16").iloc[0]
        key = tuple(float(row[name]) for name in estimator._frontier_feature_names)
        assert estimator._frontier_exact_lookup[key] == pytest.approx(expected)
        assert estimator._frontier_kernel_count_model._frontier_exact_lookup[key] == 7
        identity = json.loads(estimator._frontier_measurement_identity)
        assert identity["operator_precision"] == precision.name
        assert identity["source_profiling_precision"] == "BF16"
        assert identity["moe_layer_ids"] == ["3-87" if precision is PrecisionType.FP8 else "88-90"]
        assert identity["evidence_kind"] == "cpu_fixture"
    for name in ("mlp_up_proj", "share_expert_up_proj"):
        assert manager.get_model(name, precision="BF16") is not None
        with pytest.raises(ValueError, match="not available for precision"):
            manager.get_model(name, precision="FP8")
    assert manager.get_model("moe_gating_linear", precision="FP32") is not None
    assert "step5_ffn_merge" in models
    assert manager._models_view_for_family("kernel_only", ClusterType.MONOLITHIC)["step5_ffn_merge"] is models["step5_ffn_merge"]


def test_step_outside_root_norm_is_explicit_shared_cache_helper(model, tmp_path):
    manager = _manager(tmp_path / "cache")
    linear, _ = _tables(tmp_path, model)
    manager._cluster_configs = {ClusterType.MONOLITHIC: SimpleNamespace(replica_config=_replica(model))}
    models = manager._train_step5_input_layernorm_for_cluster(
        ClusterType.MONOLITHIC, _replica(model), _config(), str(linear), set(),
    )
    assert set(models) == {"input_layernorm"}
    assert models["input_layernorm"]._frontier_exact_lookup[(16.,)] == .16
    assert models["input_layernorm"]._frontier_kernel_count_model._frontier_exact_lookup[(16.,)] == 7


def test_step_ffn_warm_cache_registers_both_precisions_without_refitting(model, tmp_path, monkeypatch):
    paths = _tables(tmp_path, model)
    cold = _manager(tmp_path / "cache")
    models = _ffn(cold, model, paths)
    warm = _manager(tmp_path / "cache")
    monkeypatch.setattr(warm, "_create_estimator_and_params", lambda *_: pytest.fail("warm cache refitted"))
    loaded = _ffn(warm, model, paths)
    for name in models:
        assert loaded[name]._frontier_model_hash == models[name]._frontier_model_hash
    for precision in ("BF16", "FP8"):
        estimator = warm.get_model("moe_grouped_gemm", precision=precision)
        assert estimator._frontier_kernel_count_model._frontier_exact_lookup


@pytest.mark.parametrize("mutation, message", [
    ({"time_stats.moe_grouped_gemm.kernel_count": None}, "actual positive integral kernel coverage"),
    ({"time_stats.moe_grouped_gemm.kernel_count": 0}, "actual positive integral kernel coverage"),
    ({"time_stats.moe_grouped_gemm.kernel_count": True}, "actual positive integral kernel coverage"),
    ({"moe_routed_precision": "bf16"}, "exclusion metadata conflicts"),
    ({"moe_activation": "silu"}, "exclusion metadata conflicts"),
    ({"measurement_type": "CUDA_EVENT"}, "do not relabel"),
])
def test_step_ffn_rejects_unmeasured_counts_and_false_source_contracts(model, tmp_path, mutation, message):
    paths = _tables(tmp_path, model)
    frame = pd.read_csv(paths[1])
    for name, value in mutation.items():
        frame[name] = value
    frame.to_csv(paths[1], index=False)
    with pytest.raises(ValueError, match=message):
        _ffn(_manager(tmp_path / "cache"), model, paths)


def test_step_input_norm_refuses_sample_count_as_launch_count(model, tmp_path):
    linear, _ = _tables(tmp_path, model)
    frame = pd.read_csv(linear).drop(columns=["time_stats.input_layernorm.kernel_count"])
    frame.to_csv(linear, index=False)
    manager = _manager(tmp_path / "cache")
    with pytest.raises(ValueError, match="actual positive integral kernel coverage"):
        manager._train_step5_input_layernorm_for_cluster(
            ClusterType.MONOLITHIC, _replica(model), _config(), str(linear), set(),
        )
    assert not list((tmp_path / "cache").glob("*.pkl"))


def test_step_all_layer_source_precision_and_native_activation_exclusions(model):
    for layer in range(92):
        metadata = describe_step5_layer(model, layer, attention_tp_size=8,
                                       moe_tp_size=1, expert_parallel_size=8)
        assert metadata.model_precision is PrecisionType.BF16
        for projection in metadata.projections:
            if projection.operator_name == "moe_grouped_gemm":
                assert projection.weight_precision is (PrecisionType.FP8 if layer < 88 else PrecisionType.BF16)
            elif projection.operator_name == "moe_gating_linear":
                assert projection.weight_precision is PrecisionType.FP32
            else:
                assert projection.weight_precision is PrecisionType.BF16
        assert metadata.dense_swiglu_limit == 7.
        if layer in model.get_moe_layer_ids():
            assert metadata.routed_swiglu_limit == 7.
        assert metadata.geometry.head_dim == 192

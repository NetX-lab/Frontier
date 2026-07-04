from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.operators.binding import build_operator_manifest
from frontier.operators.families import FFN_FAMILY, MEMORY_FAMILY
from frontier.operators.spec import OperatorPhase, OperatorRole, ResourceClass, TraceKind
from frontier.types import ClusterType
from frontier.types import ActivationType, NormType


class _ConcreteSklearnExecutionTimePredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise NotImplementedError


def _base_model_config(*, is_moe: bool = False) -> BaseModelConfig:
    return BaseModelConfig(
        num_layers=2,
        num_q_heads=32,
        num_kv_heads=8,
        embedding_dim=4096,
        mlp_hidden_dim=11008,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        is_moe=is_moe,
        num_experts=8 if is_moe else 0,
        num_experts_per_tok=2 if is_moe else 0,
        model_type="unit_moe_model" if is_moe else "unit_dense_model",
    )


def test_ffn_family_declares_dense_mlp_ops_in_execution_order() -> None:
    assert FFN_FAMILY.family_id == "ffn"
    assert FFN_FAMILY.resource_class is ResourceClass.COMP
    assert [operator.name for operator in FFN_FAMILY.operators] == [
        "mlp_up_proj",
        "mlp_act",
        "mlp_down_proj",
    ]
    assert [operator.execution_time_attr for operator in FFN_FAMILY.operators] == [
        "mlp_layer_up_proj_execution_time",
        "mlp_layer_act_execution_time",
        "mlp_layer_down_proj_execution_time",
    ]
    assert [operator.calibration_key for operator in FFN_FAMILY.operators] == [
        "mlp_up_proj",
        None,
        "mlp_down_proj",
    ]
    assert all(operator.trace_kind is TraceKind.COMPUTE for operator in FFN_FAMILY.operators)
    assert [operator.role for operator in FFN_FAMILY.operators] == [
        OperatorRole.PROJECTION,
        OperatorRole.ACTIVATION,
        OperatorRole.PROJECTION,
    ]
    assert all(
        operator.phases
        == (OperatorPhase.PREFILL, OperatorPhase.DECODE, OperatorPhase.MIXED)
        for operator in FFN_FAMILY.operators
    )


def test_memory_family_declares_compute_folded_memory_ops() -> None:
    assert MEMORY_FAMILY.family_id == "memory"
    assert MEMORY_FAMILY.resource_class is ResourceClass.MEMORY
    assert [operator.name for operator in MEMORY_FAMILY.operators] == [
        "input_layernorm",
        "post_attention_layernorm",
        "add_attn_residual",
        "add_ffn_residual",
        "emb",
    ]
    assert [operator.execution_time_attr for operator in MEMORY_FAMILY.operators] == [
        "attn_norm_time",
        "mlp_norm_time",
        "add_attn_residual_time",
        "add_ffn_residual_time",
        None,
    ]
    assert [operator.precision_name() for operator in MEMORY_FAMILY.operators] == [
        "input_layernorm",
        "post_attention_layernorm",
        "add",
        "add",
        "emb",
    ]
    assert [operator.e2e_trace_target for operator in MEMORY_FAMILY.operators] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert all(operator.trace_kind is TraceKind.COMPUTE for operator in MEMORY_FAMILY.operators)
    assert [operator.role for operator in MEMORY_FAMILY.operators] == [
        OperatorRole.NORMALIZATION,
        OperatorRole.NORMALIZATION,
        OperatorRole.RESIDUAL,
        OperatorRole.RESIDUAL,
        OperatorRole.EMBEDDING,
    ]


def test_dense_manifest_appends_memory_and_ffn_after_attention() -> None:
    manifest = build_operator_manifest(_base_model_config(is_moe=False))

    assert [binding.family_id for binding in manifest.family_bindings] == [
        "dense_attention",
        "memory",
        "ffn",
    ]
    assert [operator.name for operator in manifest.operators()] == [
        "attn_kv_cache_save",
        "attn_prefill",
        "attn_decode",
        "input_layernorm",
        "post_attention_layernorm",
        "add_attn_residual",
        "add_ffn_residual",
        "emb",
        "mlp_up_proj",
        "mlp_act",
        "mlp_down_proj",
    ]


def test_moe_manifest_appends_memory_and_moe_without_dense_ffn() -> None:
    manifest = build_operator_manifest(_base_model_config(is_moe=True))

    assert [binding.family_id for binding in manifest.family_bindings] == [
        "dense_attention",
        "memory",
        "moe",
    ]
    assert "ffn" not in {binding.family_id for binding in manifest.family_bindings}


def test_linear_profiling_plan_uses_family_derived_ffn_memory_lists() -> None:
    from frontier.profiling.linear_op import profiling_plan

    assert not hasattr(profiling_plan, "FFN_OPS")
    assert not hasattr(profiling_plan, "FFN_REPLICATED_OPS")
    assert not hasattr(profiling_plan, "COMMON_REPLICATED_OPS")

    plan = profiling_plan.build_profiling_plan(
        _base_model_config(is_moe=False),
        tp_size=1,
        attn_tp=[1],
        ffn_tp=[1],
        disable_replicated=False,
        is_moe=False,
    )

    assert [op.name for op in MEMORY_FAMILY.profiling_ops()] == [
        "input_layernorm",
        "post_attention_layernorm",
        "add_attn_residual",
        "add_ffn_residual",
        "emb",
    ]
    assert [op.profiling_name() for op in MEMORY_FAMILY.profiling_ops()] == [
        "input_layernorm",
        "post_attention_layernorm",
        "add",
        "add",
        "emb",
    ]
    assert [op.name for op in FFN_FAMILY.profiling_ops()] == [
        "mlp_up_proj",
        "mlp_down_proj",
        "mlp_act",
    ]
    assert plan["enabled_ops"] == [
        "input_layernorm",
        "post_attention_layernorm",
        "add",
        "emb",
        "attn_pre_proj",
        "attn_rope",
        "attn_post_proj",
        "mlp_up_proj",
        "mlp_down_proj",
        "mlp_act",
    ]
    assert plan["replicated_ops"] == [
        "input_layernorm",
        "post_attention_layernorm",
        "add",
        "emb",
    ]


def test_linear_profiling_enabled_ops_resolves_memory_physical_ops_to_profiling_keys() -> None:
    from frontier.profiling.linear_op import profiling_plan

    enabled_ops = {
        "input_layernorm",
        "post_attention_layernorm",
        "add",
        "emb",
    }

    assert profiling_plan.memory_operator_enabled(enabled_ops, "input_layernorm")
    assert profiling_plan.memory_operator_enabled(enabled_ops, "post_attention_layernorm")
    assert profiling_plan.memory_operator_enabled(enabled_ops, "add_attn_residual")
    assert profiling_plan.memory_operator_enabled(enabled_ops, "add_ffn_residual")
    assert profiling_plan.memory_operator_enabled(enabled_ops, "emb")
    assert not profiling_plan.memory_operator_enabled({"input_layernorm"}, "add_ffn_residual")
    with pytest.raises(ValueError, match="Unknown MEMORY profiling operator"):
        profiling_plan.memory_operator_enabled(enabled_ops, "not_memory")


def test_predictor_linear_tp_mapping_uses_declared_ffn_family_membership() -> None:
    predictor = cast(Any, object.__new__(_ConcreteSklearnExecutionTimePredictor))
    predictor._model_config = _base_model_config(is_moe=False)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=4,
    )

    assert predictor._get_linear_op_tp_key("mlp_up_proj") == 2
    assert predictor._get_linear_op_tp_key("mlp_down_proj") == 2
    assert predictor._get_linear_op_tp_key("mlp_act") == 2
    with pytest.raises(ValueError, match="Unsupported linear op"):
        predictor._get_linear_op_tp_key("mlp_not_declared")


def test_predictor_operator_calibration_scales_use_declared_ffn_calibration_keys() -> None:
    predictor = cast(Any, object.__new__(_ConcreteSklearnExecutionTimePredictor))
    predictor._config = SimpleNamespace(
        mlp_up_proj_calibration_scale=2.0,
        prefill_phase_mlp_up_proj_calibration_scale=3.0,
    )
    up_proj, act, _down_proj = FFN_FAMILY.operators

    assert predictor._get_operator_calibration_scale(up_proj) == pytest.approx(2.0)
    assert predictor._get_optional_operator_phase_calibration_scale(
        up_proj,
        "prefill_phase",
    ) == pytest.approx(3.0)
    assert predictor._get_operator_calibration_scale(act) == pytest.approx(1.0)
    assert (
        predictor._get_optional_operator_phase_calibration_scale(act, "prefill_phase")
        is None
    )


def test_shared_prediction_manager_linear_tp_mapping_handles_ffn_family_ops() -> None:
    manager = object.__new__(ExecutionTimePredictionModelManager)
    replica_config = SimpleNamespace(
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=4,
        model_config=_base_model_config(is_moe=False),
        speculative_decoding_config=None,
    )

    assert (
        manager._get_linear_op_tp_key(
            "mlp_up_proj",
            ClusterType.MONOLITHIC,
            replica_config,
            is_moe_model=True,
        )
        == 4
    )
    with pytest.raises(ValueError, match="Unsupported linear op"):
        manager._get_linear_op_tp_key(
            "mlp_not_declared",
            ClusterType.MONOLITHIC,
            replica_config,
            is_moe_model=True,
        )


def test_trace_precision_mapping_uses_declared_memory_precision_ops(monkeypatch) -> None:
    import frontier.metrics.op_trace_utils as op_trace_utils

    assert op_trace_utils.map_trace_op_to_precision_op("add_attn_residual") == "add"
    assert op_trace_utils.map_trace_op_to_precision_op("add_ffn_residual") == "add"

    unit_family = MEMORY_FAMILY.__class__(
        family_id="unit_memory",
        display_name="Unit Memory",
        supported_variants=("unit",),
        operators=(
            MEMORY_FAMILY.operators[0].__class__(
                name="unit_residual",
                role=OperatorRole.RESIDUAL,
                phases=(OperatorPhase.PREFILL,),
                execution_time_attr="unit_residual_time",
                resource_class=ResourceClass.MEMORY,
                precision_op="add",
            ),
        ),
        resource_class=ResourceClass.MEMORY,
    )
    monkeypatch.setattr(op_trace_utils, "MEMORY_FAMILY", unit_family)

    assert op_trace_utils.map_trace_op_to_precision_op("unit_residual") == "add"


def _op_trace_context() -> Any:
    from frontier.metrics.op_trace_utils import OpTraceContext

    return OpTraceContext(
        cluster_type=ClusterType.MONOLITHIC,
        model_config=_base_model_config(is_moe=False),
        replica_config=SimpleNamespace(
            attn_tensor_parallel_size=1,
            attn_data_parallel_size=1,
            moe_tensor_parallel_size=1,
            moe_expert_parallel_size=1,
            num_pipeline_stages=1,
            router_topk=0,
        ),
        total_tokens=8,
        effective_tokens_compute=8,
        effective_tokens_transfer=8,
        effective_tokens_rounded=8,
        tokens_are_post_routing=False,
    )


def test_trace_meta_uses_declared_memory_and_ffn_family_ops(monkeypatch) -> None:
    import frontier.metrics.op_trace_utils as op_trace_utils

    memory_family = MEMORY_FAMILY.__class__(
        family_id="unit_memory",
        display_name="Unit Memory",
        supported_variants=("unit",),
        operators=(
            MEMORY_FAMILY.operators[0].__class__(
                name="unit_norm",
                role=OperatorRole.NORMALIZATION,
                phases=(OperatorPhase.PREFILL,),
                execution_time_attr="mlp_norm_time",
                resource_class=ResourceClass.MEMORY,
                precision_op="post_attention_layernorm",
            ),
            MEMORY_FAMILY.operators[0].__class__(
                name="unit_residual",
                role=OperatorRole.RESIDUAL,
                phases=(OperatorPhase.PREFILL,),
                execution_time_attr="add_ffn_residual_time",
                resource_class=ResourceClass.MEMORY,
                precision_op="add",
            ),
        ),
        resource_class=ResourceClass.MEMORY,
    )
    ffn_family = FFN_FAMILY.__class__(
        family_id="unit_ffn",
        display_name="Unit FFN",
        supported_variants=("unit",),
        operators=(
            FFN_FAMILY.operators[0].__class__(
                name="unit_up",
                role=OperatorRole.PROJECTION,
                phases=(OperatorPhase.PREFILL,),
                execution_time_attr="mlp_layer_up_proj_execution_time",
                resource_class=ResourceClass.COMP,
                projection_ownership=FFN_FAMILY.operators[0].projection_ownership,
                precision_op="mlp_up_proj",
            ),
            FFN_FAMILY.operators[1].__class__(
                name="unit_act",
                role=OperatorRole.ACTIVATION,
                phases=(OperatorPhase.PREFILL,),
                execution_time_attr="mlp_layer_act_execution_time",
                resource_class=ResourceClass.COMP,
                precision_op="mlp_act",
            ),
            FFN_FAMILY.operators[2].__class__(
                name="unit_down",
                role=OperatorRole.PROJECTION,
                phases=(OperatorPhase.PREFILL,),
                execution_time_attr="mlp_layer_down_proj_execution_time",
                resource_class=ResourceClass.COMP,
                projection_ownership=FFN_FAMILY.operators[2].projection_ownership,
                precision_op="mlp_down_proj",
            ),
        ),
        resource_class=ResourceClass.COMP,
    )
    monkeypatch.setattr(op_trace_utils, "MEMORY_FAMILY", memory_family)
    monkeypatch.setattr(op_trace_utils, "FFN_FAMILY", ffn_family)
    ctx = _op_trace_context()

    assert op_trace_utils.compute_op_trace_meta("unit_norm", "COMPUTE", ctx)[
        "tensor_shape"
    ] == {"input": [8, 4096], "output": [8, 4096]}
    assert op_trace_utils.compute_op_trace_meta("unit_residual", "COMPUTE", ctx)[
        "tensor_shape"
    ] == {
        "input_a": [8, 4096],
        "input_b": [8, 4096],
        "output": [8, 4096],
    }
    assert op_trace_utils.compute_op_trace_meta("unit_up", "COMPUTE", ctx)[
        "tensor_shape"
    ] == {"input": [8, 4096], "output": [8, 11008]}
    assert op_trace_utils.compute_op_trace_meta("unit_act", "COMPUTE", ctx)[
        "tensor_shape"
    ] == {"input": [8, 11008], "output": [8, 11008]}
    assert op_trace_utils.compute_op_trace_meta("unit_down", "COMPUTE", ctx)[
        "tensor_shape"
    ] == {"input": [8, 11008], "output": [8, 4096]}


def test_metrics_store_family_execution_time_iterator_uses_declared_attrs() -> None:
    from frontier.metrics.metrics_store import _iter_family_execution_times

    execution_time = SimpleNamespace(
        mlp_layer_up_proj_execution_time=1.25,
        mlp_layer_act_execution_time=0.50,
        mlp_layer_down_proj_execution_time=0.75,
    )

    assert list(_iter_family_execution_times(FFN_FAMILY, cast(Any, execution_time))) == [
        ("mlp_up_proj", 1.25),
        ("mlp_act", 0.50),
        ("mlp_down_proj", 0.75),
    ]


def test_metrics_store_memory_execution_time_iterator_uses_declared_attrs() -> None:
    from frontier.metrics.metrics_store import _iter_memory_execution_times

    execution_time = SimpleNamespace(
        attn_norm_time=1.0,
        mlp_norm_time=2.0,
        add_attn_residual_time=3.0,
        add_ffn_residual_time=4.0,
    )

    assert list(_iter_memory_execution_times(cast(Any, execution_time))) == [
        ("input_layernorm", 1.0),
        ("post_attention_layernorm", 2.0),
        ("add_attn_residual", 3.0),
        ("add_ffn_residual", 4.0),
    ]
    assert list(
        _iter_memory_execution_times(
            cast(Any, execution_time),
            per_layer_count=2,
            include_input_layernorm=False,
            include_add_attn_residual=False,
        )
    ) == [
        ("post_attention_layernorm", 1.0),
        ("add_ffn_residual", 2.0),
    ]


def _minimal_execution_time_kwargs() -> dict[str, Any]:
    return {
        "num_layers_per_pipeline_stage": 1,
        "attention_rope_execution_time": 0.0,
        "attention_kv_cache_save_execution_time": 0.0,
        "attention_decode_execution_time": 0.0,
        "attention_prefill_execution_time": 0.0,
        "attention_layer_pre_proj_execution_time": 0.0,
        "attention_layer_post_proj_execution_time": 0.0,
        "attn_norm_time": 0.0,
        "mlp_norm_time": 0.0,
        "add_time": 0.0,
        "tensor_parallel_communication_time": 0.0,
        "pipeline_parallel_communication_time": 0.0,
        "expert_parallel_communication_time": 0.0,
        "moe_gating_time": 0.0,
        "moe_shuffling_time": 0.0,
        "schedule_time": 0.0,
        "sampler_e2e_time": 0.0,
        "prepare_inputs_e2e_time": 0.0,
        "process_model_outputs_time": 0.0,
        "ray_comm_time": 0.0,
    }


def test_mlp_time_legacy_total_remains_unchanged_without_structured_operator_times() -> None:
    from frontier.entities import time_components

    mlp_time = time_components.MLPTime(
        mlp_layer_up_proj_execution_time=1.25,
        mlp_layer_act_execution_time=0.50,
        mlp_layer_down_proj_execution_time=0.75,
        mlp_norm_time=0.25,
    )

    assert mlp_time.total_time() == pytest.approx(2.75)


def test_mlp_time_structured_operator_times_replace_covered_legacy_fields() -> None:
    from frontier.entities import time_components

    operator_times = time_components.MLPOperatorTimes(
        op_times={
            "post_attention_layernorm": 4.0,
            "mlp_up_proj": 10.0,
            "mlp_act": 20.0,
            "mlp_down_proj": 30.0,
        }
    )
    mlp_time = time_components.MLPTime(
        mlp_layer_up_proj_execution_time=1.0,
        mlp_layer_act_execution_time=2.0,
        mlp_layer_down_proj_execution_time=3.0,
        mlp_norm_time=0.5,
        operator_times=operator_times,
    )

    assert mlp_time.total_time() == pytest.approx(64.0)


def test_mlp_operator_times_fail_fast_for_invalid_or_missing_timings() -> None:
    from frontier.entities import time_components

    with pytest.raises(ValueError, match="Negative MLP operator timing"):
        time_components.MLPOperatorTimes(op_times={"mlp_up_proj": -0.1})

    operator_times = time_components.MLPOperatorTimes(op_times={"mlp_up_proj": 1.0})
    with pytest.raises(ValueError, match="missing structured MLP operator timing"):
        operator_times.get_required_time("mlp_down_proj")


def test_execution_time_accepts_dense_mlp_structured_operator_times() -> None:
    from frontier.entities.execution_time import ExecutionTime
    from frontier.entities.time_components import MLPOperatorTimes

    operator_times = MLPOperatorTimes(
        op_times={
            "post_attention_layernorm": 4.0,
            "mlp_up_proj": 10.0,
            "mlp_act": 20.0,
            "mlp_down_proj": 30.0,
        }
    )
    execution_time_kwargs = _minimal_execution_time_kwargs()
    execution_time_kwargs.update(
        {
            "is_moe": False,
            "mlp_norm_time": 0.5,
            "mlp_layer_up_proj_execution_time": 1.0,
            "mlp_layer_act_execution_time": 2.0,
            "mlp_layer_down_proj_execution_time": 3.0,
            "mlp_operator_times": operator_times,
        }
    )
    execution_time = ExecutionTime(
        **execution_time_kwargs,
    )

    actual_operator_times = execution_time.mlp_operator_times
    assert actual_operator_times is not operator_times
    assert actual_operator_times is not None
    assert actual_operator_times.op_times == operator_times.op_times
    assert execution_time.get_single_layer_block_time() == pytest.approx(64.0)


def test_execution_time_rejects_mlp_operator_times_for_moe_components() -> None:
    from frontier.entities.execution_time import ExecutionTime
    from frontier.entities.time_components import MLPOperatorTimes

    with pytest.raises(ValueError, match="mlp_operator_times are only valid for dense MLP"):
        ExecutionTime(
            **_minimal_execution_time_kwargs(),
            is_moe=True,
            mlp_operator_times=MLPOperatorTimes(op_times={"mlp_up_proj": 1.0}),
        )


def test_dense_mlp_predictor_writes_structured_mlp_operator_times() -> None:
    predictor = cast(Any, object.__new__(_ConcreteSklearnExecutionTimePredictor))
    predictor._enable_dummy_mode = False
    predictor._supports_operation = lambda op_name: op_name in {
        "mlp_up_proj",
        "mlp_act",
        "mlp_down_proj",
    }
    predictor._get_mlp_layer_up_proj_execution_time = lambda batch: 1.25
    predictor._get_mlp_layer_act_execution_time = lambda batch: 0.50
    predictor._get_mlp_layer_down_proj_execution_time = lambda batch: 0.75
    predictor._get_mlp_norm_layer_act_execution_time = lambda batch: 0.25
    batch = SimpleNamespace(
        id=7,
        total_num_tokens=16,
        requests=[SimpleNamespace(id=101, num_prefill_tokens=16)],
    )

    mlp_time = predictor.predict_mlp_layer_time(
        batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert mlp_time.operator_times is not None
    assert mlp_time.operator_times.op_times == {
        "post_attention_layernorm": 0.25,
        "mlp_up_proj": 1.25,
        "mlp_act": 0.50,
        "mlp_down_proj": 0.75,
    }
    assert mlp_time.total_time() == pytest.approx(2.75)

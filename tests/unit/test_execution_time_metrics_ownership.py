"""Metrics adapters preserve timing scope, numerical payloads, and identities."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.entities import ExecutionTime, StageExecutionTime
from frontier.scheduler.utils.dense_metrics import (
    build_prefill_metrics_execution_time,
    predict_dense_reference,
)
from frontier.scheduler.utils.execution_time_metrics import (
    build_metrics_execution_time,
    build_single_layer_metrics_execution_time,
)
from frontier.types import ClusterType


def _layer(layer_id=7, **overrides):
    values = dict.fromkeys(
        (
            "attention_rope_execution_time",
            "attention_kv_cache_save_execution_time",
            "attention_decode_execution_time",
            "attention_prefill_execution_time",
            "attention_layer_pre_proj_execution_time",
            "attention_layer_post_proj_execution_time",
            "attn_norm_time", "mlp_norm_time", "add_time",
            "tensor_parallel_communication_time",
            "pipeline_parallel_communication_time",
            "expert_parallel_communication_time", "moe_gating_time",
            "moe_shuffling_time", "schedule_time", "sampler_e2e_time",
            "prepare_inputs_e2e_time", "process_model_outputs_time", "ray_comm_time",
        ),
        0.0,
    )
    values.update(
        num_layers_per_pipeline_stage=1,
        is_moe=False,
        global_layer_id=layer_id,
        attention_family_id="mla",
        attention_variant_id="mla",
        op_times={
            "attn_mla_decode": 2.0,
            "attn_mla_v_up_proj": 3.0,
            "mlp_up_proj": 5.0,
            "mlp_act": 7.0,
            "mlp_down_proj": 11.0,
            "attn_tensor_parallel_allreduce": 0.0,
            "moe_tensor_parallel_allreduce": 0.0,
        },
    )
    values.update(overrides)
    return ExecutionTime(**values)


@pytest.mark.parametrize("wrap_stage", [False, True])
def test_single_layer_copy_preserves_complete_payload_and_entity_id(wrap_stage):
    original = _layer(
        tensor_parallel_communication_time=99.0,
        pipeline_parallel_communication_time=13.0,
        pp_stage_boundary_residual_runtime_time=17.0,
        pp_stage_boundary_handoff_time=19.0,
        decode_draft_proposer_time=23.0,
        mtp_terminal_overshoot_time=29.0,
    )
    source = StageExecutionTime((original,)) if wrap_stage else original
    previous_id = ExecutionTime.generate_id()

    copied = build_single_layer_metrics_execution_time(source)

    assert ExecutionTime.generate_id() == previous_id + 1
    assert copied is not original
    assert copied.id == original.id
    assert copied.global_layer_id == 7
    assert copied.attention_family_id == "mla"
    assert copied.attention_variant_id == "mla"
    assert dict(copied.op_times) == dict(original.op_times)
    assert copied.attn_mla_decode_time == pytest.approx(2.0)
    assert copied.attn_mla_v_up_proj_time == pytest.approx(3.0)
    assert copied.attention_all_reduce_time == pytest.approx(0.0)
    assert copied.mlp_all_reduce_time == pytest.approx(0.0)
    assert copied.model_time_ms == pytest.approx(original.model_time_ms)
    assert copied.total_time == pytest.approx(original.total_time)
    assert copied.diagnostic_total_time_ms == pytest.approx(original.diagnostic_total_time_ms)
    copied._trace_dense_layer_id = 7
    assert not hasattr(original, "_trace_dense_layer_id")


def test_single_layer_copy_rejects_multilayer_stage():
    original = StageExecutionTime((_layer(7), _layer(8)))

    with pytest.raises(ValueError, match="exactly one"):
        build_single_layer_metrics_execution_time(original)


def test_scope_preserving_copy_keeps_stage_layers_and_explicit_owner():
    first, second = _layer(7), _layer(8, op_times={"attn_mla_decode": 31.0})
    owner = _layer(20, pipeline_parallel_communication_time=37.0, schedule_time=41.0)
    original = StageExecutionTime((first, second), stage_execution_time=owner)
    previous_id = ExecutionTime.generate_id()

    copied = build_metrics_execution_time(original)

    assert ExecutionTime.generate_id() == previous_id + 1
    assert isinstance(copied, StageExecutionTime)
    assert copied is not original
    assert copied.global_layer_ids == (7, 8)
    assert copied.pipeline_parallel_communication_time == pytest.approx(37.0)
    assert copied.schedule_time == pytest.approx(41.0)
    assert copied.model_time_ms == pytest.approx(original.model_time_ms)
    assert dict(copied.op_times) == dict(original.op_times)
    copied._trace_execution_time_override = original
    assert not hasattr(original, "_trace_execution_time_override")
    first._replace_operator_time_values({"attn_mla_decode": 1000.0})
    assert copied.op_times["attn_mla_decode"] == pytest.approx(33.0)


def test_layer_metrics_snapshot_isolated_from_later_source_edits():
    original = _layer()
    copied = build_metrics_execution_time(original)

    original._replace_operator_time_values({"attn_mla_decode": 1000.0})

    assert copied.op_times["attn_mla_decode"] == pytest.approx(2.0)
    assert copied.attn_mla_decode_time == pytest.approx(2.0)


def _model():
    return SimpleNamespace(is_moe=True, num_layers=2, get_moe_layer_ids=lambda: [1])


def test_dense_reference_explicitly_unwraps_singleton_stage():
    source = _layer(0)
    predictor = SimpleNamespace(
        predict_stage_execution_time=Mock(return_value=StageExecutionTime((source,)))
    )
    batch = object()

    result = predict_dense_reference(
        predictor=predictor, batch=batch, stage_id=0,
        cluster_type=ClusterType.PREFILL, model_config=_model(),
    )

    assert isinstance(result, ExecutionTime)
    assert result.global_layer_id == 0
    assert result.id == source.id
    assert dict(result.op_times) == dict(source.op_times)
    predictor.predict_stage_execution_time.assert_called_once_with(
        batch, 0, cluster_type=ClusterType.PREFILL, num_layers=1, layer_id=0,
    )


def test_dense_reference_rejects_unexpected_multilayer_prediction():
    predictor = SimpleNamespace(
        predict_stage_execution_time=Mock(
            return_value=StageExecutionTime((_layer(0), _layer(1)))
        )
    )

    with pytest.raises(ValueError, match="exactly one"):
        predict_dense_reference(
            predictor=predictor, batch=object(), stage_id=0,
            cluster_type=ClusterType.PREFILL, model_config=_model(),
        )


def test_prefill_dense_layers_preserve_scope_and_do_not_mutate_sources():
    source = StageExecutionTime((_layer(7), _layer(8)))
    reference = StageExecutionTime((_layer(7),))
    predictor = SimpleNamespace(predict_stage_execution_time=Mock(return_value=reference))
    model = SimpleNamespace(
        is_moe=True, num_layers=9, get_moe_layer_ids=lambda: [8],
        is_moe_layer=lambda layer_id: layer_id == 8,
    )
    batch = object()
    result = build_prefill_metrics_execution_time(
        original_execution_time=source, sample_batch=batch, predictor=predictor,
        stage_id=0, cluster_type=ClusterType.PREFILL, model_config=model,
    )

    assert isinstance(result, StageExecutionTime)
    assert result.global_layer_ids == (7, 8)
    assert result.model_time_ms == pytest.approx(source.model_time_ms)
    assert result.layer_execution_times[0].mlp_layer_up_proj_execution_time == pytest.approx(5.0)
    assert result.layer_execution_times[0].mlp_layer_act_execution_time == pytest.approx(7.0)
    assert result.layer_execution_times[0].mlp_layer_down_proj_execution_time == pytest.approx(11.0)
    predictor.predict_stage_execution_time.assert_called_once_with(
        batch, 0, cluster_type=ClusterType.PREFILL, num_layers=1, layer_id=7,
    )
    assert not hasattr(source, "_trace_dense_layer_id")
    assert not hasattr(reference, "_trace_dense_layer_id")

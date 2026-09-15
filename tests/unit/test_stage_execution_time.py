from __future__ import annotations

from typing import Any

import pytest

from frontier.entities import ExecutionTime, StageExecutionTime
from frontier.scheduler.utils.execution_time_metrics import (
    build_single_layer_metrics_execution_time,
)


def _execution_kwargs(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "num_layers_per_pipeline_stage": 1,
        "attention_rope_execution_time": 1.0,
        "attention_kv_cache_save_execution_time": 2.0,
        "attention_decode_execution_time": 3.0,
        "attention_prefill_execution_time": 4.0,
        "attention_layer_pre_proj_execution_time": 5.0,
        "attention_layer_post_proj_execution_time": 6.0,
        "attn_norm_time": 7.0,
        "mlp_norm_time": 8.0,
        "add_time": 9.0,
        "tensor_parallel_communication_time": 10.0,
        "pipeline_parallel_communication_time": 11.0,
        "expert_parallel_communication_time": 0.0,
        "moe_gating_time": 0.0,
        "moe_shuffling_time": 0.0,
        "schedule_time": 12.0,
        "sampler_e2e_time": 13.0,
        "prepare_inputs_e2e_time": 14.0,
        "process_model_outputs_time": 15.0,
        "ray_comm_time": 16.0,
        "is_moe": False,
        "mlp_layer_up_proj_execution_time": 17.0,
        "mlp_layer_down_proj_execution_time": 18.0,
        "mlp_layer_act_execution_time": 19.0,
    }
    values.update(overrides)
    return values


def _layer(*, layer_id: int, **overrides: Any) -> ExecutionTime:
    values = _execution_kwargs(
        global_layer_id=layer_id,
        attention_family_id="dense_attention",
        attention_variant_id="standard",
    )
    values.update(overrides)
    return ExecutionTime(**values)


def test_identity_bearing_execution_time_is_one_real_layer() -> None:
    execution_time = ExecutionTime(
        **_execution_kwargs(
            num_layers_per_pipeline_stage=7,
            global_layer_id=3,
            attention_family_id="gated_delta_net",
            attention_variant_id="qwen3_5",
        )
    )

    assert execution_time.num_layers == 1
    assert execution_time.global_layer_id == 3
    assert execution_time.attention_family_id == "gated_delta_net"
    assert execution_time.attention_variant_id == "qwen3_5"
    assert execution_time.model_time_ms == pytest.approx(
        execution_time.get_single_layer_block_time()
        + execution_time.pipeline_parallel_communication_time
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"global_layer_id": 1},
        {"attention_family_id": "dense_attention"},
        {"attention_variant_id": "standard"},
    ],
)
def test_partial_layer_identity_fails_fast(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="global_layer_id, attention_family_id"):
        ExecutionTime(**_execution_kwargs(**overrides))


def test_as_single_layer_does_not_consume_entity_id() -> None:
    source = ExecutionTime(**_execution_kwargs())
    next_id = ExecutionTime.generate_id()
    layer = source.as_single_layer(global_layer_id=9)

    assert layer.id == source.id
    assert ExecutionTime.generate_id() == next_id + 1
    assert layer.num_layers == 1
    assert layer.global_layer_id == 9


def test_stage_sums_ordered_layers_and_charges_stage_work_once() -> None:
    first = _layer(layer_id=4)
    second = _layer(layer_id=5, attention_prefill_execution_time=40.0)
    stage = StageExecutionTime(
        (first, second),
        stage_execution_time=first,
    )

    expected_blocks = first.get_single_layer_block_time() + second.get_single_layer_block_time()
    assert stage.global_layer_ids == (4, 5)
    assert stage.model_time_ms == pytest.approx(
        expected_blocks + first.pipeline_parallel_communication_time
    )
    assert stage.total_time == pytest.approx(
        stage.model_time
        + (
            first.schedule_time
            + first.sampler_e2e_time
            + first.prepare_inputs_e2e_time
            + first.process_model_outputs_time
            + first.ray_comm_time
        )
        * 1e-3
    )
    # Layer probes remain one-layer probes for event paths that advance a
    # stage one transformer layer at a time.
    assert stage.get_single_layer_block_time() == pytest.approx(
        first.get_single_layer_block_time()
    )


def test_stage_model_time_reuses_immutable_layer_aggregate(monkeypatch) -> None:
    first = _layer(layer_id=0)
    second = _layer(layer_id=1)
    calls = 0
    original = first.get_single_layer_block_time

    def counted_block_time() -> float:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(first, "get_single_layer_block_time", counted_block_time)
    stage = StageExecutionTime((first, second), stage_execution_time=first)

    assert stage.model_time_ms == pytest.approx(stage.model_time_ms)
    assert calls == 1


def test_stage_model_time_refreshes_after_layer_mutation() -> None:
    first = _layer(layer_id=0)
    second = _layer(layer_id=1)
    stage = StageExecutionTime((first, second), stage_execution_time=first)

    before = stage.model_time_ms
    first._replace_operator_time_values({"attn_prefill": 7.0})

    assert stage.model_time_ms == pytest.approx(
        before + (7.0 - 4.0)
    )


def test_stage_keeps_distinct_moe_layer_records() -> None:
    first = _layer(
        layer_id=1,
        is_moe=True,
        mlp_layer_up_proj_execution_time=0.0,
        mlp_layer_down_proj_execution_time=0.0,
        mlp_layer_act_execution_time=0.0,
        moe_grouped_gemm_time=12.0,
        moe_gating_time=4.0,
        moe_operator_times=None,
    )
    second = _layer(
        layer_id=2,
        is_moe=True,
        mlp_layer_up_proj_execution_time=0.0,
        mlp_layer_down_proj_execution_time=0.0,
        mlp_layer_act_execution_time=0.0,
        moe_grouped_gemm_time=27.0,
        moe_gating_time=9.0,
        moe_operator_times=None,
    )
    stage = StageExecutionTime((first, second), stage_execution_time=first)

    assert [layer.moe_grouped_gemm_time for layer in stage.layer_execution_times] == [
        pytest.approx(12.0),
        pytest.approx(27.0),
    ]
    assert stage.moe_grouped_gemm_time == pytest.approx(39.0)


def test_pipeline_operator_is_not_multiplied_by_layer_count() -> None:
    first = _layer(
        layer_id=0,
        op_times={
            "attn_prefill": 4.0,
            "pipeline_parallel_send_recv": 11.0,
        },
    )
    stage = StageExecutionTime.from_execution_time(
        first,
        num_layers=3,
        first_layer_id=0,
    )

    assert stage.communication_operator_times is not None
    assert stage.communication_operator_times.op_times["pipeline_parallel_send_recv"] == pytest.approx(11.0)
    assert stage.op_times["attn_prefill"] == pytest.approx(12.0)


def test_stage_expansion_isolates_mutable_component_and_operator_maps() -> None:
    source = _layer(
        layer_id=0,
        op_times={"attn_prefill": 4.0, "mlp_up_proj": 2.0},
    )
    stage = StageExecutionTime.from_execution_time(
        source,
        num_layers=2,
        first_layer_id=0,
    )
    first, second = stage.layer_execution_times

    assert first.attention_time_component is not second.attention_time_component
    assert first.attention_operator_times is not second.attention_operator_times
    assert first.attention_operator_times.op_times is not second.attention_operator_times.op_times

    first.attention_operator_times.op_times["attn_prefill"] = 99.0
    assert second.attention_operator_times.op_times["attn_prefill"] == pytest.approx(4.0)
    first._replace_operator_time_values({"attn_prefill": 7.0})
    assert second.attention_operator_times.op_times["attn_prefill"] == pytest.approx(4.0)


def test_fast_stage_expansion_detaches_before_mutation() -> None:
    source = _layer(
        layer_id=0,
        op_times={"attn_prefill": 4.0, "mlp_up_proj": 2.0},
    )
    stage = StageExecutionTime.from_execution_time(
        source,
        num_layers=2,
        first_layer_id=0,
        copy_components=False,
    )
    first, second = stage.layer_execution_times

    # The fast path shares the immutable prediction payload while retaining
    # distinct layer identities. A mutating compatibility operation must
    # detach the first layer before changing its operator values.
    assert first._attention_time is second._attention_time
    assert first.global_layer_id == 0
    assert second.global_layer_id == 1
    first._replace_operator_time_values({"attn_prefill": 7.0})
    assert first.attention_operator_times.op_times["attn_prefill"] == pytest.approx(7.0)
    assert second.attention_operator_times.op_times["attn_prefill"] == pytest.approx(4.0)
    assert first._attention_time is not second._attention_time


def test_single_layer_fast_path_adopts_source_identity() -> None:
    source = ExecutionTime(**_execution_kwargs())

    stage = StageExecutionTime.from_execution_time(
        source,
        num_layers=1,
        first_layer_id=7,
        copy_components=False,
    )

    assert stage.layer_execution_times == (source,)
    assert source.global_layer_id == 7
    assert source.num_layers == 1


def test_stage_owner_terminal_work_and_diagnostic_overhead_are_once_only() -> None:
    owner = _layer(
        layer_id=0,
        decode_draft_proposer_time=5.0,
        mtp_terminal_overshoot_time=7.0,
        pp_stage_boundary_handoff_time=3.0,
    )
    second = _layer(layer_id=1, decode_draft_proposer_time=99.0)
    stage = StageExecutionTime((owner, second), stage_execution_time=owner)

    expected_model_ms = (
        owner.get_single_layer_block_time()
        + second.get_single_layer_block_time()
        + owner.pipeline_parallel_communication_time
        + 5.0
        + 7.0
    )
    assert stage.model_time_ms == pytest.approx(expected_model_ms)
    assert stage.total_time * 1e3 == pytest.approx(expected_model_ms + owner.total_time * 1e3 - owner.model_time_ms)
    assert stage.diagnostic_total_time_ms == pytest.approx(
        stage.total_time * 1e3 + 3.0
    )


def test_stage_rejects_duplicate_global_layer_ids() -> None:
    first = _layer(layer_id=2)
    duplicate = _layer(layer_id=2)

    with pytest.raises(ValueError, match="global_layer_ids must be unique"):
        StageExecutionTime((first, duplicate), stage_execution_time=first)


def test_stage_preserves_mixed_attention_family_order() -> None:
    dense = _layer(layer_id=3)
    gdn = _layer(
        layer_id=4,
        attention_family_id="gated_delta_net",
        attention_variant_id="qwen3_5",
    )

    stage = StageExecutionTime((dense, gdn), stage_execution_time=dense)

    assert stage.global_layer_ids == (3, 4)
    assert stage.attention_family_ids == ("dense_attention", "gated_delta_net")
    assert stage.attention_variant_ids == ("standard", "qwen3_5")


def test_stage_private_fields_remain_single_layer_for_metrics_adapter() -> None:
    owner = _layer(
        layer_id=0,
        attn_tensor_parallel_allreduce_time=3.0,
        pipeline_parallel_communication_time=11.0,
    )
    second = _layer(
        layer_id=1,
        attn_tensor_parallel_allreduce_time=7.0,
        pipeline_parallel_communication_time=99.0,
    )
    stage = StageExecutionTime((owner, second), stage_execution_time=owner)

    metrics_execution_time = build_single_layer_metrics_execution_time(stage)

    assert metrics_execution_time.num_layers == 1
    assert metrics_execution_time.attention_all_reduce_time == pytest.approx(3.0)
    assert metrics_execution_time.pipeline_parallel_communication_time == pytest.approx(
        11.0
    )

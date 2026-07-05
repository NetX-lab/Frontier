from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]

from frontier.entities.execution_time import ExecutionTime
from frontier.entities.time_components import (
    AttentionTime,
    AttentionOperatorTimes,
    CommunicationTime,
    CommunicationOperatorTimes,
    MLPTime,
    MLPOperatorTimes,
    MoEOperatorTimes,
    MoETime,
)


def _base_execution_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        num_layers_per_pipeline_stage=2,
        attention_rope_execution_time=100.0,
        attention_kv_cache_save_execution_time=100.0,
        attention_decode_execution_time=100.0,
        attention_prefill_execution_time=100.0,
        attention_layer_pre_proj_execution_time=100.0,
        attention_layer_post_proj_execution_time=100.0,
        attn_norm_time=100.0,
        mlp_norm_time=100.0,
        add_time=0.0,
        tensor_parallel_communication_time=100.0,
        pipeline_parallel_communication_time=100.0,
        expert_parallel_communication_time=100.0,
        moe_gating_time=0.0,
        moe_shuffling_time=0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=False,
        mlp_layer_up_proj_execution_time=100.0,
        mlp_layer_down_proj_execution_time=100.0,
        mlp_layer_act_execution_time=100.0,
        add_attn_residual_time=100.0,
        add_ffn_residual_time=100.0,
        attn_tensor_parallel_allreduce_time=100.0,
        moe_tensor_parallel_allreduce_time=100.0,
    )
    kwargs.update(overrides)
    return kwargs


def _assert_conflicting_setter_is_atomic(
    execution_time: ExecutionTime,
    getter: Any,
    setter_name: str,
    new_operator_times: Any,
    conflict_op_name: str,
) -> None:
    old_operator_times = getter(execution_time)
    assert old_operator_times is not None
    old_operator_time_values = dict(old_operator_times.op_times)
    old_op_times = dict(execution_time.op_times)
    old_model_time_ms = execution_time.model_time_ms

    with pytest.raises(
        ValueError,
        match=f"Conflicting operator timing for {conflict_op_name}",
    ):
        setattr(execution_time, setter_name, new_operator_times)

    current_operator_times = getter(execution_time)
    assert current_operator_times is not None
    assert dict(current_operator_times.op_times) == old_operator_time_values
    assert dict(execution_time.op_times) == old_op_times
    assert execution_time.model_time_ms == pytest.approx(old_model_time_ms)


def _assert_operator_times_input_mutation_is_isolated(
    execution_time: ExecutionTime,
    source_operator_times: Any,
    getter: Any,
    op_name: str,
    scalar_getter: Any,
) -> None:
    cast(Any, source_operator_times.op_times)[op_name] = 99.0

    current_operator_times = getter(execution_time)
    assert current_operator_times is not None
    assert current_operator_times.op_times[op_name] == pytest.approx(4.0)
    assert execution_time.op_times[op_name] == pytest.approx(4.0)
    assert scalar_getter(execution_time) == pytest.approx(4.0)


def _moe_override_state_snapshot(execution_time: ExecutionTime) -> dict[str, Any]:
    moe_component = execution_time.moe_or_mlp_time_component
    assert isinstance(moe_component, MoETime)
    moe_operator_times = execution_time.moe_operator_times
    communication_operator_times = execution_time.communication_operator_times

    return {
        "op_times": dict(execution_time.op_times),
        "moe_operator_times": (
            None if moe_operator_times is None else dict(moe_operator_times.op_times)
        ),
        "communication_operator_times": (
            None
            if communication_operator_times is None
            else dict(communication_operator_times.op_times)
        ),
        "moe_grouped_gemm_time": execution_time.moe_grouped_gemm_time,
        "moe_gating_linear_time": execution_time.moe_gating_linear_time,
        "moe_gating_routing_topk_time": execution_time.moe_gating_routing_topk_time,
        "moe_gating_time": execution_time.moe_gating_time,
        "moe_shuffling_time": execution_time.moe_shuffling_time,
        "expert_parallel_communication_time": (
            execution_time.expert_parallel_communication_time
        ),
        "component_moe_grouped_gemm_time": moe_component.moe_grouped_gemm_time,
        "component_moe_gating_linear_time": moe_component.moe_gating_linear_time,
        "component_moe_gating_routing_topk_time": (
            moe_component.moe_gating_routing_topk_time
        ),
        "component_moe_shuffling_time": moe_component.moe_shuffling_time,
        "model_time_ms": execution_time.model_time_ms,
    }


def _build_moe_override_atomicity_case(*, with_op_times: bool) -> ExecutionTime:
    kwargs = _base_execution_kwargs(
        is_moe=True,
        num_layers_per_pipeline_stage=1,
        attention_rope_execution_time=0.0,
        attention_kv_cache_save_execution_time=0.0,
        attention_decode_execution_time=0.0,
        attention_prefill_execution_time=0.0,
        attention_layer_pre_proj_execution_time=0.0,
        attention_layer_post_proj_execution_time=0.0,
        attn_norm_time=0.0,
        mlp_norm_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=8.0,
        moe_grouped_gemm_time=4.0,
        moe_gating_linear_time=5.0,
        moe_gating_routing_topk_time=6.0,
        moe_shuffling_time=7.0,
        mlp_layer_up_proj_execution_time=0.0,
        mlp_layer_down_proj_execution_time=0.0,
        mlp_layer_act_execution_time=0.0,
        add_attn_residual_time=0.0,
        add_ffn_residual_time=0.0,
        attn_tensor_parallel_allreduce_time=0.0,
        moe_tensor_parallel_allreduce_time=0.0,
    )
    if with_op_times:
        kwargs["op_times"] = {
            "moe_grouped_gemm": 4.0,
            "moe_gating_linear": 5.0,
            "moe_gating_routing_topk": 6.0,
            "moe_shuffling": 7.0,
            "expert_parallel_allreduce": 8.0,
        }
    return ExecutionTime(**kwargs)


def test_execution_time_op_times_drive_dense_views_and_legacy_properties():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            op_times={
                "pipeline_parallel_send_recv": 13.0,
                "mlp_down_proj": 8.0,
                "attn_prefill": 2.0,
                "add_ffn_residual": 10.0,
                "attn_kv_cache_save": 1.0,
                "mlp_tensor_parallel_allreduce": 12.0,
                "post_attention_layernorm": 5.0,
                "input_layernorm": 4.0,
                "mlp_act": 7.0,
                "attn_decode": 3.0,
                "attn_tensor_parallel_allreduce": 11.0,
                "mlp_up_proj": 6.0,
                "add_attn_residual": 9.0,
            }
        )
    )

    assert tuple(execution_time.op_times) == (
        "attn_kv_cache_save",
        "attn_prefill",
        "attn_decode",
        "input_layernorm",
        "post_attention_layernorm",
        "add_attn_residual",
        "add_ffn_residual",
        "mlp_up_proj",
        "mlp_act",
        "mlp_down_proj",
        "attn_tensor_parallel_allreduce",
        "mlp_tensor_parallel_allreduce",
        "pipeline_parallel_send_recv",
    )
    assert execution_time.attention_kv_cache_save_execution_time == pytest.approx(2.0)
    assert execution_time.attention_prefill_execution_time == pytest.approx(4.0)
    assert execution_time.attention_decode_execution_time == pytest.approx(6.0)
    assert execution_time.attn_norm_time == pytest.approx(8.0)
    assert execution_time.mlp_norm_time == pytest.approx(10.0)
    assert execution_time.mlp_layer_up_proj_execution_time == pytest.approx(12.0)
    assert execution_time.mlp_layer_act_execution_time == pytest.approx(14.0)
    assert execution_time.mlp_layer_down_proj_execution_time == pytest.approx(16.0)
    assert execution_time.add_attn_residual_time == pytest.approx(18.0)
    assert execution_time.add_ffn_residual_time == pytest.approx(20.0)
    assert execution_time.get_single_layer_add_time() == pytest.approx(19.0)
    assert execution_time.attention_all_reduce_time == pytest.approx(22.0)
    assert execution_time.mlp_all_reduce_time == pytest.approx(24.0)
    assert execution_time.pipeline_parallel_communication_time == pytest.approx(13.0)

    with pytest.raises(TypeError):
        cast(Any, execution_time.op_times)["attn_prefill"] = 99.0


def test_execution_time_component_setters_keep_top_level_op_times_in_sync():
    execution_time = ExecutionTime(**_base_execution_kwargs())

    execution_time.attention_operator_times = AttentionOperatorTimes(
        {"attn_prefill": 1.25}
    )
    assert execution_time.op_times["attn_prefill"] == pytest.approx(1.25)
    assert execution_time.attention_prefill_execution_time == pytest.approx(2.5)

    execution_time.attention_operator_times = None
    assert "attn_prefill" not in execution_time.op_times
    assert execution_time.attention_prefill_execution_time == pytest.approx(200.0)


def test_execution_time_moe_setter_keeps_legacy_views_in_sync_with_op_times():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(is_moe=True, expert_parallel_communication_time=0.0)
    )

    execution_time.moe_operator_times = MoEOperatorTimes(
        {
            "moe_gating_linear": 1.0,
            "moe_gating_routing_topk": 2.0,
            "moe_shuffling": 4.0,
            "moe_grouped_gemm": 3.0,
            "share_expert_up_proj": 4.0,
            "share_expert_act": 5.0,
            "share_expert_down_proj": 6.0,
        }
    )

    assert execution_time.moe_gating_linear_time == pytest.approx(2.0)
    assert execution_time.moe_gating_routing_topk_time == pytest.approx(4.0)
    assert execution_time.moe_gating_time == pytest.approx(6.0)
    assert execution_time.moe_grouped_gemm_time == pytest.approx(6.0)
    assert execution_time.share_expert_up_proj_time == pytest.approx(8.0)
    assert execution_time.share_expert_act_time == pytest.approx(10.0)
    assert execution_time.share_expert_down_proj_time == pytest.approx(12.0)
    assert execution_time.share_expert_time == pytest.approx(30.0)
    assert execution_time.get_single_layer_moe_comp_time() == pytest.approx(6.0)
    assert execution_time.moe_comp_time == pytest.approx(12.0)
    assert execution_time.get_single_layer_moe_comm_time() == pytest.approx(4.0)
    assert execution_time.moe_comm_time == pytest.approx(8.0)


def test_execution_time_comm_setter_keeps_legacy_views_in_sync_with_op_times():
    execution_time = ExecutionTime(**_base_execution_kwargs())

    execution_time.communication_operator_times = CommunicationOperatorTimes(
        {
            "attn_tensor_parallel_allreduce": 1.0,
            "mlp_tensor_parallel_allreduce": 2.0,
            "pipeline_parallel_send_recv": 3.0,
        }
    )

    assert execution_time.attention_all_reduce_time == pytest.approx(2.0)
    assert execution_time.mlp_all_reduce_time == pytest.approx(4.0)
    assert execution_time.pipeline_parallel_communication_time == pytest.approx(3.0)


def test_execution_time_comm_setter_updates_expert_parallel_view():
    execution_time = ExecutionTime(**_base_execution_kwargs(is_moe=True))

    execution_time.communication_operator_times = CommunicationOperatorTimes(
        {"expert_parallel_allreduce": 9.0}
    )

    assert execution_time.op_times["expert_parallel_allreduce"] == pytest.approx(9.0)
    assert execution_time.expert_parallel_communication_time == pytest.approx(18.0)
    assert execution_time.get_single_layer_moe_comm_time() == pytest.approx(9.0)
    assert execution_time.moe_comm_time == pytest.approx(18.0)


def test_execution_time_rejects_conflicting_top_level_and_component_op_times():
    with pytest.raises(ValueError, match="Conflicting operator timing for mlp_up_proj"):
        ExecutionTime(
            **_base_execution_kwargs(
                op_times={"mlp_up_proj": 1.0},
                mlp_operator_times=MLPOperatorTimes({"mlp_up_proj": 2.0}),
            )
        )


def test_execution_time_rejects_invalid_top_level_op_times():
    with pytest.raises(
        ValueError,
        match="Unsupported ExecutionTime operator timing: unknown_operator",
    ):
        ExecutionTime(**_base_execution_kwargs(op_times={"unknown_operator": 1.0}))

    with pytest.raises(
        ValueError,
        match="Negative ExecutionTime operator timing is invalid: attn_prefill=-1.0",
    ):
        ExecutionTime(**_base_execution_kwargs(op_times={"attn_prefill": -1.0}))


def test_execution_time_allows_identical_top_level_and_component_op_times():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            op_times={"mlp_up_proj": 1.0},
            mlp_operator_times=MLPOperatorTimes({"mlp_up_proj": 1.0}),
        )
    )

    assert execution_time.op_times["mlp_up_proj"] == pytest.approx(1.0)
    assert execution_time.mlp_layer_up_proj_execution_time == pytest.approx(2.0)


def test_execution_time_op_times_drive_moe_and_expert_comm_views():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=True,
            op_times={
                "post_attention_layernorm": 1.0,
                "moe_gating_linear": 2.0,
                "moe_gating_routing_topk": 3.0,
                "moe_shuffling": 4.0,
                "moe_grouped_gemm": 5.0,
                "share_expert_up_proj": 6.0,
                "share_expert_act": 7.0,
                "share_expert_down_proj": 8.0,
                "expert_parallel_allreduce": 9.0,
            },
        )
    )

    assert execution_time.mlp_norm_time == pytest.approx(2.0)
    assert execution_time.moe_gating_linear_time == pytest.approx(4.0)
    assert execution_time.moe_gating_routing_topk_time == pytest.approx(6.0)
    assert execution_time.moe_gating_time == pytest.approx(10.0)
    assert execution_time.moe_shuffling_time == pytest.approx(8.0)
    assert execution_time.moe_grouped_gemm_time == pytest.approx(10.0)
    assert execution_time.share_expert_up_proj_time == pytest.approx(12.0)
    assert execution_time.share_expert_act_time == pytest.approx(14.0)
    assert execution_time.share_expert_down_proj_time == pytest.approx(16.0)
    assert execution_time.expert_parallel_communication_time == pytest.approx(18.0)
    assert execution_time.get_single_layer_moe_comm_time() == pytest.approx(13.0)
    assert execution_time.communication_operator_times is not None
    assert execution_time.communication_operator_times.op_times[
        "expert_parallel_allreduce"
    ] == pytest.approx(9.0)


def test_execution_time_moe_grouped_gemm_override_updates_canonical_op_times():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=True,
            op_times={"moe_grouped_gemm": 1.0},
        )
    )

    execution_time.override_moe_grouped_gemm_time(7.0)

    assert execution_time.op_times["moe_grouped_gemm"] == pytest.approx(7.0)
    assert execution_time.moe_operator_times is not None
    assert execution_time.moe_operator_times.op_times["moe_grouped_gemm"] == (
        pytest.approx(7.0)
    )
    assert execution_time.moe_grouped_gemm_time == pytest.approx(14.0)


def test_execution_time_override_moe_times_updates_canonical_op_times():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=True,
            op_times={
                "moe_grouped_gemm": 1.0,
                "moe_gating_linear": 2.0,
                "moe_gating_routing_topk": 3.0,
                "moe_shuffling": 4.0,
                "expert_parallel_allreduce": 5.0,
            },
        )
    )

    execution_time.override_moe_times(
        grouped_gemm_time=7.0,
        expert_parallel_comm_time=8.0,
        gating_time=0.0,
        shuffling_time=10.0,
        gating_linear_time=11.0,
        gating_routing_topk_time=12.0,
    )

    assert execution_time.op_times["moe_grouped_gemm"] == pytest.approx(7.0)
    assert execution_time.op_times["moe_gating_linear"] == pytest.approx(11.0)
    assert execution_time.op_times["moe_gating_routing_topk"] == pytest.approx(12.0)
    assert execution_time.op_times["moe_shuffling"] == pytest.approx(10.0)
    assert execution_time.op_times["expert_parallel_allreduce"] == pytest.approx(8.0)
    assert execution_time.moe_operator_times is not None
    assert execution_time.communication_operator_times is not None
    assert execution_time.moe_operator_times.op_times["moe_grouped_gemm"] == (
        pytest.approx(7.0)
    )
    assert execution_time.communication_operator_times.op_times[
        "expert_parallel_allreduce"
    ] == pytest.approx(8.0)
    assert execution_time.moe_gating_time == pytest.approx(46.0)
    assert execution_time.expert_parallel_communication_time == pytest.approx(16.0)


@pytest.mark.parametrize("with_op_times", [False, True])
def test_execution_time_moe_override_methods_are_atomic_on_invalid_timing(
    with_op_times: bool,
):
    execution_time = _build_moe_override_atomicity_case(with_op_times=with_op_times)
    old_state = _moe_override_state_snapshot(execution_time)

    with pytest.raises(
        ValueError,
        match="Negative ExecutionTime operator timing is invalid: "
        "moe_grouped_gemm=-1.0",
    ):
        execution_time.override_moe_grouped_gemm_time(-1.0)

    assert _moe_override_state_snapshot(execution_time) == old_state

    old_state = _moe_override_state_snapshot(execution_time)
    with pytest.raises(
        ValueError,
        match="Negative ExecutionTime operator timing is invalid: "
        "expert_parallel_allreduce=-2.0",
    ):
        execution_time.override_moe_times(
            grouped_gemm_time=9.0,
            expert_parallel_comm_time=-2.0,
            gating_time=0.0,
            shuffling_time=10.0,
            gating_linear_time=11.0,
            gating_routing_topk_time=12.0,
        )

    assert _moe_override_state_snapshot(execution_time) == old_state


def test_execution_time_comm_setter_updates_pipeline_time_helper():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=True,
            pipeline_parallel_communication_time=100.0,
        )
    )

    execution_time.communication_operator_times = CommunicationOperatorTimes(
        {"pipeline_parallel_send_recv": 3.0}
    )

    assert execution_time.op_times["pipeline_parallel_send_recv"] == pytest.approx(3.0)
    assert execution_time.pipeline_parallel_communication_time == pytest.approx(3.0)
    assert execution_time.pipeline_time == pytest.approx(3.0)


def test_execution_time_comm_setter_updates_moe_block_and_model_time_helpers():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=True,
            num_layers_per_pipeline_stage=1,
            attention_rope_execution_time=0.0,
            attention_kv_cache_save_execution_time=0.0,
            attention_decode_execution_time=0.0,
            attention_prefill_execution_time=0.0,
            attention_layer_pre_proj_execution_time=0.0,
            attention_layer_post_proj_execution_time=0.0,
            attn_norm_time=0.0,
            mlp_norm_time=0.0,
            add_attn_residual_time=0.0,
            add_ffn_residual_time=0.0,
            tensor_parallel_communication_time=0.0,
            pipeline_parallel_communication_time=0.0,
            expert_parallel_communication_time=0.0,
            moe_gating_time=0.0,
            moe_shuffling_time=0.0,
            moe_grouped_gemm_time=0.0,
            moe_gating_linear_time=0.0,
            moe_gating_routing_topk_time=0.0,
            share_expert_up_proj_time=0.0,
            share_expert_down_proj_time=0.0,
            share_expert_act_time=0.0,
            mlp_layer_up_proj_execution_time=0.0,
            mlp_layer_down_proj_execution_time=0.0,
            mlp_layer_act_execution_time=0.0,
            attn_tensor_parallel_allreduce_time=0.0,
            moe_tensor_parallel_allreduce_time=0.0,
        )
    )

    execution_time.communication_operator_times = CommunicationOperatorTimes(
        {
            "moe_tensor_parallel_allgather": 7.0,
            "share_expert_tensor_parallel_allreduce": 11.0,
        }
    )

    assert execution_time.moe_tensor_parallel_allgather_time == pytest.approx(7.0)
    assert execution_time.share_expert_tensor_parallel_allreduce_time == pytest.approx(
        11.0
    )
    assert execution_time.get_single_layer_block_time() == pytest.approx(18.0)
    assert execution_time.model_time_ms == pytest.approx(18.0)
    assert execution_time.model_time == pytest.approx(0.018)


def test_execution_time_explicit_zero_tp_allreduce_op_times_override_legacy_fallback():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=False,
            num_layers_per_pipeline_stage=2,
            tensor_parallel_communication_time=100.0,
            attn_tensor_parallel_allreduce_time=0.0,
            moe_tensor_parallel_allreduce_time=0.0,
            op_times={
                "attn_tensor_parallel_allreduce": 0.0,
                "mlp_tensor_parallel_allreduce": 0.0,
            },
        )
    )

    assert execution_time.attention_all_reduce_time == pytest.approx(0.0)
    assert execution_time.mlp_all_reduce_time == pytest.approx(0.0)


def test_execution_time_explicit_zero_tp_allreduce_op_times_override_legacy_component_total_on_construction():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=False,
            num_layers_per_pipeline_stage=2,
            tensor_parallel_communication_time=100.0,
            pipeline_parallel_communication_time=0.0,
            expert_parallel_communication_time=0.0,
            attn_tensor_parallel_allreduce_time=0.0,
            moe_tensor_parallel_allreduce_time=0.0,
            op_times={
                "attn_tensor_parallel_allreduce": 0.0,
                "mlp_tensor_parallel_allreduce": 0.0,
            },
        )
    )

    assert execution_time.communication_time_component.tensor_parallel_allreduce_time == (
        pytest.approx(0.0)
    )
    assert execution_time.communication_time_component.total_time() == pytest.approx(0.0)


def test_execution_time_explicit_zero_tp_allreduce_comm_setter_overrides_legacy_component_total():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=False,
            num_layers_per_pipeline_stage=2,
            tensor_parallel_communication_time=100.0,
            pipeline_parallel_communication_time=0.0,
            expert_parallel_communication_time=0.0,
            attn_tensor_parallel_allreduce_time=0.0,
            moe_tensor_parallel_allreduce_time=0.0,
        )
    )

    execution_time.communication_operator_times = CommunicationOperatorTimes(
        {
            "attn_tensor_parallel_allreduce": 0.0,
            "mlp_tensor_parallel_allreduce": 0.0,
        }
    )

    assert execution_time.attention_all_reduce_time == pytest.approx(0.0)
    assert execution_time.mlp_all_reduce_time == pytest.approx(0.0)
    assert execution_time.communication_time_component.total_time() == pytest.approx(0.0)


def test_execution_time_explicit_zero_legacy_split_tp_allreduce_fields_override_legacy_fallback_without_op_times():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=False,
            num_layers_per_pipeline_stage=2,
            tensor_parallel_communication_time=100.0,
            pipeline_parallel_communication_time=0.0,
            expert_parallel_communication_time=0.0,
            attn_tensor_parallel_allreduce_time=0.0,
            moe_tensor_parallel_allreduce_time=0.0,
        )
    )

    assert execution_time.attention_all_reduce_time == pytest.approx(0.0)
    assert execution_time.mlp_all_reduce_time == pytest.approx(0.0)
    assert execution_time.communication_time_component.total_time() == pytest.approx(0.0)


def test_execution_time_omitted_split_tp_allreduce_fields_preserve_legacy_fallback():
    kwargs = _base_execution_kwargs(
        is_moe=False,
        num_layers_per_pipeline_stage=2,
        tensor_parallel_communication_time=100.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
    )
    kwargs.pop("attn_tensor_parallel_allreduce_time")
    kwargs.pop("moe_tensor_parallel_allreduce_time")

    execution_time = ExecutionTime(**kwargs)

    assert execution_time.attention_all_reduce_time == pytest.approx(200.0)
    assert execution_time.mlp_all_reduce_time == pytest.approx(200.0)
    assert execution_time.communication_time_component.total_time() == pytest.approx(100.0)


def test_corrected_execution_time_copy_preserves_omitted_split_tp_legacy_fallback():
    kwargs = _base_execution_kwargs(
        is_moe=False,
        num_layers_per_pipeline_stage=2,
        tensor_parallel_communication_time=100.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
    )
    kwargs.pop("attn_tensor_parallel_allreduce_time")
    kwargs.pop("moe_tensor_parallel_allreduce_time")
    original_execution_time = ExecutionTime(**kwargs)

    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        BaseClusterScheduler,
    )

    corrected_execution_time = BaseClusterScheduler._create_corrected_execution_time_for_metrics(
        cast(Any, None),
        original_execution_time,
        actual_execution_time_ms=0.0,
        original_start_time=0.0,
    )

    assert corrected_execution_time._has_attn_tensor_parallel_allreduce_time is False
    assert corrected_execution_time._has_moe_tensor_parallel_allreduce_time is False
    assert corrected_execution_time.attention_all_reduce_time == pytest.approx(100.0)
    assert corrected_execution_time.mlp_all_reduce_time == pytest.approx(100.0)
    assert corrected_execution_time.communication_time_component.total_time() == pytest.approx(
        100.0
    )


def test_corrected_execution_time_copy_preserves_explicit_zero_split_tp_override():
    original_execution_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=False,
            num_layers_per_pipeline_stage=2,
            tensor_parallel_communication_time=100.0,
            pipeline_parallel_communication_time=0.0,
            expert_parallel_communication_time=0.0,
            attn_tensor_parallel_allreduce_time=0.0,
            moe_tensor_parallel_allreduce_time=0.0,
        )
    )

    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        BaseClusterScheduler,
    )

    corrected_execution_time = BaseClusterScheduler._create_corrected_execution_time_for_metrics(
        cast(Any, None),
        original_execution_time,
        actual_execution_time_ms=0.0,
        original_start_time=0.0,
    )

    assert corrected_execution_time._has_attn_tensor_parallel_allreduce_time is True
    assert corrected_execution_time._has_moe_tensor_parallel_allreduce_time is True
    assert corrected_execution_time.attention_all_reduce_time == pytest.approx(0.0)
    assert corrected_execution_time.mlp_all_reduce_time == pytest.approx(0.0)
    assert corrected_execution_time.communication_time_component.total_time() == pytest.approx(
        0.0
    )


def test_execution_time_component_operator_time_setters_are_atomic_on_conflict():
    attention_time = ExecutionTime(
        **_base_execution_kwargs(
            op_times={"attn_prefill": 1.0},
            attention_operator_times=AttentionOperatorTimes({"attn_decode": 4.0}),
        )
    )
    _assert_conflicting_setter_is_atomic(
        attention_time,
        lambda execution_time: execution_time.attention_operator_times,
        "attention_operator_times",
        AttentionOperatorTimes({"attn_prefill": 2.0}),
        "attn_prefill",
    )

    mlp_time = ExecutionTime(
        **_base_execution_kwargs(
            op_times={"mlp_up_proj": 1.0},
            mlp_operator_times=MLPOperatorTimes({"mlp_down_proj": 4.0}),
        )
    )
    _assert_conflicting_setter_is_atomic(
        mlp_time,
        lambda execution_time: execution_time.mlp_operator_times,
        "mlp_operator_times",
        MLPOperatorTimes({"mlp_up_proj": 2.0}),
        "mlp_up_proj",
    )

    moe_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=True,
            op_times={"moe_grouped_gemm": 1.0},
            moe_operator_times=MoEOperatorTimes({"moe_shuffling": 4.0}),
        )
    )
    _assert_conflicting_setter_is_atomic(
        moe_time,
        lambda execution_time: execution_time.moe_operator_times,
        "moe_operator_times",
        MoEOperatorTimes({"moe_grouped_gemm": 2.0}),
        "moe_grouped_gemm",
    )

    communication_time = ExecutionTime(
        **_base_execution_kwargs(
            op_times={"pipeline_parallel_send_recv": 1.0},
            communication_operator_times=CommunicationOperatorTimes(
                {"expert_parallel_allreduce": 4.0}
            ),
        )
    )
    _assert_conflicting_setter_is_atomic(
        communication_time,
        lambda execution_time: execution_time.communication_operator_times,
        "communication_operator_times",
        CommunicationOperatorTimes({"pipeline_parallel_send_recv": 2.0}),
        "pipeline_parallel_send_recv",
    )


def test_execution_time_component_accessors_return_defensive_snapshots():
    execution_time = ExecutionTime(
        **_base_execution_kwargs(
            op_times={
                "attn_prefill": 1.0,
                "mlp_up_proj": 2.0,
                "pipeline_parallel_send_recv": 3.0,
            }
        )
    )

    attention_component = execution_time.attention_time_component
    assert isinstance(attention_component, AttentionTime)
    attention_component.attention_prefill_execution_time = 99.0
    assert (
        execution_time.attention_time_component.attention_prefill_execution_time
        == pytest.approx(1.0)
    )

    attention_operator_times = execution_time.attention_operator_times
    assert attention_operator_times is not None
    cast(Any, attention_operator_times.op_times)["attn_prefill"] = 99.0
    assert execution_time.attention_operator_times is not None
    assert execution_time.attention_operator_times.op_times["attn_prefill"] == (
        pytest.approx(1.0)
    )

    mlp_component = execution_time.moe_or_mlp_time_component
    assert isinstance(mlp_component, MLPTime)
    mlp_component.mlp_layer_up_proj_execution_time = 99.0
    fresh_mlp_component = execution_time.moe_or_mlp_time_component
    assert isinstance(fresh_mlp_component, MLPTime)
    assert fresh_mlp_component.mlp_layer_up_proj_execution_time == pytest.approx(2.0)

    mlp_operator_times = execution_time.mlp_operator_times
    assert mlp_operator_times is not None
    cast(Any, mlp_operator_times.op_times)["mlp_up_proj"] = 99.0
    assert execution_time.mlp_operator_times is not None
    assert execution_time.mlp_operator_times.op_times["mlp_up_proj"] == pytest.approx(
        2.0
    )

    communication_component = execution_time.communication_time_component
    assert isinstance(communication_component, CommunicationTime)
    communication_component.pipeline_parallel_send_recv_time = 99.0
    assert (
        execution_time.communication_time_component.pipeline_parallel_send_recv_time
        == pytest.approx(3.0)
    )

    communication_operator_times = execution_time.communication_operator_times
    assert communication_operator_times is not None
    cast(Any, communication_operator_times.op_times)[
        "pipeline_parallel_send_recv"
    ] = 99.0
    assert execution_time.communication_operator_times is not None
    assert execution_time.communication_operator_times.op_times[
        "pipeline_parallel_send_recv"
    ] == pytest.approx(3.0)


def test_execution_time_constructor_operator_times_inputs_are_defensively_copied():
    attention_operator_times = AttentionOperatorTimes({"attn_decode": 4.0})
    attention_time = ExecutionTime(
        **_base_execution_kwargs(
            num_layers_per_pipeline_stage=1,
            attention_operator_times=attention_operator_times,
        )
    )
    _assert_operator_times_input_mutation_is_isolated(
        attention_time,
        attention_operator_times,
        lambda execution_time: execution_time.attention_operator_times,
        "attn_decode",
        lambda execution_time: execution_time.attention_decode_execution_time,
    )

    mlp_operator_times = MLPOperatorTimes({"mlp_up_proj": 4.0})
    mlp_time = ExecutionTime(
        **_base_execution_kwargs(
            num_layers_per_pipeline_stage=1,
            mlp_operator_times=mlp_operator_times,
        )
    )
    _assert_operator_times_input_mutation_is_isolated(
        mlp_time,
        mlp_operator_times,
        lambda execution_time: execution_time.mlp_operator_times,
        "mlp_up_proj",
        lambda execution_time: execution_time.mlp_layer_up_proj_execution_time,
    )

    moe_operator_times = MoEOperatorTimes({"moe_grouped_gemm": 4.0})
    moe_time = ExecutionTime(
        **_base_execution_kwargs(
            is_moe=True,
            num_layers_per_pipeline_stage=1,
            moe_operator_times=moe_operator_times,
        )
    )
    _assert_operator_times_input_mutation_is_isolated(
        moe_time,
        moe_operator_times,
        lambda execution_time: execution_time.moe_operator_times,
        "moe_grouped_gemm",
        lambda execution_time: execution_time.moe_grouped_gemm_time,
    )

    communication_operator_times = CommunicationOperatorTimes(
        {"pipeline_parallel_send_recv": 4.0}
    )
    communication_time = ExecutionTime(
        **_base_execution_kwargs(
            num_layers_per_pipeline_stage=1,
            communication_operator_times=communication_operator_times,
        )
    )
    _assert_operator_times_input_mutation_is_isolated(
        communication_time,
        communication_operator_times,
        lambda execution_time: execution_time.communication_operator_times,
        "pipeline_parallel_send_recv",
        lambda execution_time: execution_time.pipeline_parallel_communication_time,
    )


def test_execution_time_setter_operator_times_inputs_are_defensively_copied():
    attention_time = ExecutionTime(
        **_base_execution_kwargs(num_layers_per_pipeline_stage=1)
    )
    attention_operator_times = AttentionOperatorTimes({"attn_decode": 4.0})
    attention_time.attention_operator_times = attention_operator_times
    _assert_operator_times_input_mutation_is_isolated(
        attention_time,
        attention_operator_times,
        lambda execution_time: execution_time.attention_operator_times,
        "attn_decode",
        lambda execution_time: execution_time.attention_decode_execution_time,
    )

    mlp_time = ExecutionTime(**_base_execution_kwargs(num_layers_per_pipeline_stage=1))
    mlp_operator_times = MLPOperatorTimes({"mlp_up_proj": 4.0})
    mlp_time.mlp_operator_times = mlp_operator_times
    _assert_operator_times_input_mutation_is_isolated(
        mlp_time,
        mlp_operator_times,
        lambda execution_time: execution_time.mlp_operator_times,
        "mlp_up_proj",
        lambda execution_time: execution_time.mlp_layer_up_proj_execution_time,
    )

    moe_time = ExecutionTime(
        **_base_execution_kwargs(is_moe=True, num_layers_per_pipeline_stage=1)
    )
    moe_operator_times = MoEOperatorTimes({"moe_grouped_gemm": 4.0})
    moe_time.moe_operator_times = moe_operator_times
    _assert_operator_times_input_mutation_is_isolated(
        moe_time,
        moe_operator_times,
        lambda execution_time: execution_time.moe_operator_times,
        "moe_grouped_gemm",
        lambda execution_time: execution_time.moe_grouped_gemm_time,
    )

    communication_time = ExecutionTime(
        **_base_execution_kwargs(num_layers_per_pipeline_stage=1)
    )
    communication_operator_times = CommunicationOperatorTimes(
        {"pipeline_parallel_send_recv": 4.0}
    )
    communication_time.communication_operator_times = communication_operator_times
    _assert_operator_times_input_mutation_is_isolated(
        communication_time,
        communication_operator_times,
        lambda execution_time: execution_time.communication_operator_times,
        "pipeline_parallel_send_recv",
        lambda execution_time: execution_time.pipeline_parallel_communication_time,
    )

"""Numerical ownership at the layer-to-stage publication boundary."""

from __future__ import annotations

from typing import Any

import pytest

from frontier.entities import ExecutionTime, StageExecutionTime


def _layer(layer_id: int | None = 7, **overrides: Any) -> ExecutionTime:
    values: dict[str, Any] = dict(
        num_layers_per_pipeline_stage=1,
        attention_rope_execution_time=0.0,
        attention_kv_cache_save_execution_time=0.0,
        attention_decode_execution_time=0.0,
        attention_prefill_execution_time=0.0,
        attention_layer_pre_proj_execution_time=0.0,
        attention_layer_post_proj_execution_time=0.0,
        attn_norm_time=0.0,
        mlp_norm_time=0.0,
        add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0,
        moe_shuffling_time=0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=False,
    )
    if layer_id is not None:
        values.update(
            global_layer_id=layer_id,
            attention_family_id="dense_attention",
            attention_variant_id="standard",
        )
    values.update(overrides)
    return ExecutionTime(**values)


def _mixed_layers() -> tuple[ExecutionTime, ExecutionTime]:
    dense = _layer(
        7,
        op_times={
            "attn_prefill": 2.0,
            "mlp_up_proj": 3.0,
            "attn_tensor_parallel_allreduce": 5.0,
            "pipeline_parallel_send_recv": 11.0,
        },
        schedule_time=13.0,
        decode_draft_proposer_time=17.0,
        mtp_terminal_overshoot_time=19.0,
        pp_stage_boundary_handoff_time=23.0,
    )
    moe = _layer(
        9,
        is_moe=True,
        attention_family_id="mla",
        attention_variant_id="mla",
        op_times={
            "attn_mla_prefill": 29.0,
            "moe_grouped_gemm": 31.0,
            "moe_gating_routing_topk": 37.0,
            "expert_parallel_alltoall_dispatch": 41.0,
            "expert_parallel_alltoall_combine": 43.0,
            "attn_tensor_parallel_allreduce": 47.0,
            "pipeline_parallel_send_recv": 999.0,
        },
        schedule_time=999.0,
        decode_draft_proposer_time=999.0,
        mtp_terminal_overshoot_time=999.0,
        pp_stage_boundary_handoff_time=999.0,
    )
    return dense, moe


@pytest.mark.parametrize("layer_id", [None, 7])
def test_layer_numerics_do_not_depend_on_identity_or_legacy_count(layer_id):
    layer = _layer(
        layer_id,
        num_layers_per_pipeline_stage=5,
        op_times={"attn_prefill": 2.0, "mlp_up_proj": 3.0},
    )

    assert layer.num_layers == 1
    assert layer.attention_prefill_execution_time == pytest.approx(2.0)
    assert layer.mlp_layer_up_proj_execution_time == pytest.approx(3.0)
    assert layer.get_single_layer_block_time() == pytest.approx(5.0)
    assert layer.model_time_ms == pytest.approx(5.0)


def test_stage_preserves_real_order_and_charges_owner_work_once():
    dense, moe = _mixed_layers()
    stage = StageExecutionTime((dense, moe), stage_execution_time=dense)

    assert stage.global_layer_ids == (7, 9)
    assert stage.attention_family_ids == ("dense_attention", "mla")
    assert [layer.get_single_layer_block_time() for layer in stage.layer_execution_times] == pytest.approx([10.0, 228.0])
    # Blocks: 10 + 228; owner PP/draft/terminal: 11 + 17 + 19.
    assert stage.model_time_ms == pytest.approx(285.0)
    assert stage.model_time == pytest.approx(0.285)
    assert stage.total_time == pytest.approx(0.298)
    assert stage.diagnostic_total_time_ms == pytest.approx(321.0)


def test_generic_component_and_operator_views_have_stage_scope():
    dense, moe = _mixed_layers()
    stage = StageExecutionTime((dense, moe), stage_execution_time=dense)

    assert dict(stage.attention_operator_times.op_times) == {
        "attn_prefill": 2.0,
        "attn_mla_prefill": 29.0,
    }
    assert dict(stage.mlp_operator_times.op_times) == {"mlp_up_proj": 3.0}
    assert dict(stage.moe_operator_times.op_times) == {
        "moe_grouped_gemm": 31.0,
        "moe_gating_routing_topk": 37.0,
    }
    assert stage.attention_time_component.total_time() == pytest.approx(31.0)
    assert stage.communication_time_component.total_time() == pytest.approx(147.0)
    communication = stage.communication_operator_times.op_times
    assert communication["attn_tensor_parallel_allreduce"] == pytest.approx(52.0)
    assert communication["pipeline_parallel_send_recv"] == pytest.approx(11.0)
    assert stage.op_times["moe_grouped_gemm"] == pytest.approx(31.0)


def test_each_moe_layer_keeps_its_own_routing_and_compute_values():
    first = _layer(
        11, is_moe=True,
        op_times={"moe_grouped_gemm": 3.0, "moe_gating_routing_topk": 5.0},
    )
    second = _layer(
        12, is_moe=True,
        op_times={"moe_grouped_gemm": 7.0, "moe_gating_routing_topk": 11.0},
    )
    stage = StageExecutionTime((first, second), stage_execution_time=first)

    assert [layer.moe_gating_routing_topk_time for layer in stage.layer_execution_times] == pytest.approx([5.0, 11.0])
    assert [layer.moe_grouped_gemm_time for layer in stage.layer_execution_times] == pytest.approx([3.0, 7.0])
    assert stage.moe_operator_times.op_times["moe_gating_routing_topk"] == pytest.approx(16.0)
    assert stage.model_time_ms == pytest.approx(26.0)


def test_source_mutation_before_first_read_cannot_change_published_stage():
    dense, moe = _mixed_layers()
    stage = StageExecutionTime((dense, moe), stage_execution_time=dense)
    sibling = StageExecutionTime((dense, moe), stage_execution_time=dense)

    moe.override_moe_grouped_gemm_time(1000.0)
    dense._replace_operator_time_values({"pipeline_parallel_send_recv": 2000.0})

    for published in (stage, sibling):
        assert published.model_time_ms == pytest.approx(285.0)
        assert published.op_times["moe_grouped_gemm"] == pytest.approx(31.0)
        assert published.pipeline_parallel_communication_time == pytest.approx(11.0)
        assert published.layer_execution_times[1].moe_grouped_gemm_time == pytest.approx(31.0)


def test_published_layer_mutation_is_rejected_without_affecting_siblings():
    dense, moe = _mixed_layers()
    stage = StageExecutionTime((dense, moe), stage_execution_time=dense)
    sibling = StageExecutionTime(stage.layer_execution_times, stage_execution_time=dense)

    with pytest.raises(ValueError, match="finalized"):
        stage.layer_execution_times[1].override_moe_grouped_gemm_time(1000.0)

    assert stage.model_time_ms == pytest.approx(285.0)
    assert sibling.model_time_ms == pytest.approx(285.0)
    assert moe.moe_grouped_gemm_time == pytest.approx(31.0)


def test_returned_component_mutation_cannot_change_stage_or_source():
    dense, moe = _mixed_layers()
    stage = StageExecutionTime((dense, moe), stage_execution_time=dense)
    component = stage.communication_time_component
    try:
        component.pipeline_parallel_send_recv_time = 1000.0
    except (AttributeError, TypeError):
        pass

    assert stage.communication_time_component.pipeline_parallel_send_recv_time == pytest.approx(11.0)
    assert dense.pipeline_parallel_communication_time == pytest.approx(11.0)
    assert stage.model_time_ms == pytest.approx(285.0)


def test_returned_operator_map_is_read_only_or_detached():
    dense, moe = _mixed_layers()
    stage = StageExecutionTime((dense, moe), stage_execution_time=dense)
    returned = stage.attention_operator_times.op_times
    try:
        returned["attn_prefill"] = 12345.0
    except TypeError:
        pass

    assert stage.attention_operator_times.op_times["attn_prefill"] == pytest.approx(2.0)
    assert stage.op_times["attn_prefill"] == pytest.approx(2.0)
    assert dense.op_times["attn_prefill"] == pytest.approx(2.0)
    assert stage.model_time_ms == pytest.approx(285.0)


def test_repeated_communication_reads_preserve_sources_and_owner_scope():
    dense, moe = _mixed_layers()
    owner = _layer(20, op_times={"pipeline_parallel_send_recv": 53.0})
    stage = StageExecutionTime((dense, moe), stage_execution_time=owner)
    before = [dict(layer.op_times) for layer in (dense, moe, owner)]

    for _ in range(3):
        component = stage.communication_time_component
        assert component.pipeline_parallel_send_recv_time == pytest.approx(53.0)
        assert component.operator_times.op_times["attn_tensor_parallel_allreduce"] == pytest.approx(52.0)

    assert [dict(layer.op_times) for layer in (dense, moe, owner)] == before
    assert dense.communication_time_component.pipeline_parallel_send_recv_time == pytest.approx(11.0)
    assert moe.communication_time_component.pipeline_parallel_send_recv_time == pytest.approx(999.0)


@pytest.mark.parametrize("layer_ids", [(None,), (7, None)])
def test_published_stage_requires_complete_layer_identities(layer_ids):
    layers = tuple(_layer(layer_id) for layer_id in layer_ids)
    with pytest.raises(ValueError):
        StageExecutionTime(layers)


def test_single_layer_probe_requires_an_unambiguous_layer():
    dense, moe = _mixed_layers()
    single = StageExecutionTime((dense,), stage_execution_time=dense)
    multiple = StageExecutionTime((dense, moe), stage_execution_time=dense)

    assert single.get_single_layer_attention_scope_time() == pytest.approx(7.0)
    with pytest.raises(ValueError):
        multiple.get_single_layer_attention_scope_time()


def test_stage_does_not_delegate_private_layer_implementation():
    dense, moe = _mixed_layers()
    stage = StageExecutionTime((dense, moe), stage_execution_time=dense)

    with pytest.raises(AttributeError):
        getattr(stage, "_has_attn_tensor_parallel_allreduce_time")


def test_stage_publication_does_not_consume_simulator_entity_ids():
    dense, moe = _mixed_layers()
    last_id = ExecutionTime.generate_id()

    StageExecutionTime((dense, moe), stage_execution_time=dense)

    assert ExecutionTime.generate_id() == last_id + 1

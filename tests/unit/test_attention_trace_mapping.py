from __future__ import annotations

import pytest

import frontier.entities.time_components as time_components
from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
)
from frontier.attention.ops import (
    AttentionFamilySpec,
    AttentionMemoryLayout,
    AttentionOperatorRole,
    AttentionOperatorSpec,
    AttentionPhase,
    ProjectionOwnership,
)

from frontier.operators.spec import ResourceClass
from frontier.attention.trace_mapping import get_attention_trace_op_times
from frontier.entities.execution_time import ExecutionTime
from frontier.entities.time_components import AttentionOperatorTimes


def _build_execution_time(
    *,
    num_layers: int = 2,
    dense_kernel_times: tuple[float, float, float] = (0.2, 0.4, 0.3),
    mla_times: tuple[float, float, float, float, float, float] = (
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
    ),
) -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=num_layers,
        attention_rope_execution_time=0.5,
        attention_kv_cache_save_execution_time=dense_kernel_times[0],
        attention_decode_execution_time=dense_kernel_times[2],
        attention_prefill_execution_time=dense_kernel_times[1],
        attention_layer_pre_proj_execution_time=0.7,
        attention_layer_post_proj_execution_time=0.8,
        attn_norm_time=0.4,
        mlp_norm_time=0.9,
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
        attn_mla_kv_cache_save_time=mla_times[0],
        attn_mla_prefill_kv_up_proj_time=mla_times[1],
        attn_mla_prefill_time=mla_times[2],
        attn_mla_decode_q_latent_proj_time=mla_times[3],
        attn_mla_decode_time=mla_times[4],
        attn_mla_v_up_proj_time=mla_times[5],
    )


def test_dense_trace_mapper_preserves_family_order_and_execution_time_values() -> None:
    execution_time = _build_execution_time(num_layers=2)

    op_times = get_attention_trace_op_times(
        execution_time,
        DENSE_ATTENTION_FAMILY,
    )

    assert [(op.name, time_ms) for op, time_ms in op_times] == [
        ("attn_kv_cache_save", pytest.approx(0.4)),
        ("attn_prefill", pytest.approx(0.8)),
        ("attn_decode", pytest.approx(0.6)),
    ]


def test_trace_mapper_uses_family_execution_time_attrs_not_operator_names() -> None:
    execution_time = _build_execution_time(num_layers=2)
    family = AttentionFamilySpec(
        family_id="renamed_dense_trace_attrs",
        display_name="Renamed Dense Trace Attrs",
        supported_variants=("gqa",),
        operators=(
            AttentionOperatorSpec(
                name="role_prefill",
                role=AttentionOperatorRole.PREFILL_KERNEL,
                resource_class=ResourceClass.COMP,
                phases=(AttentionPhase.PREFILL,),
                execution_time_attr="attention_prefill_execution_time",
            ),
            AttentionOperatorSpec(
                name="role_decode",
                role=AttentionOperatorRole.DECODE_KERNEL,
                resource_class=ResourceClass.COMP,
                phases=(AttentionPhase.DECODE,),
                execution_time_attr="attention_decode_execution_time",
            ),
            AttentionOperatorSpec(
                name="role_cache",
                role=AttentionOperatorRole.CACHE_WRITE,
                resource_class=ResourceClass.MEMORY,
                phases=(AttentionPhase.PREFILL, AttentionPhase.DECODE),
                execution_time_attr="attention_kv_cache_save_execution_time",
            ),
        ),
        memory_layout=AttentionMemoryLayout.DENSE_KV,
        dense_compatible=True,
        requires_runtime_kv_helpers=False,
    )

    op_times = get_attention_trace_op_times(execution_time, family)

    assert [(op.name, time_ms) for op, time_ms in op_times] == [
        ("role_prefill", pytest.approx(0.8)),
        ("role_decode", pytest.approx(0.6)),
        ("role_cache", pytest.approx(0.4)),
    ]


def test_trace_mapper_prefers_structured_attention_operator_times() -> None:
    execution_time = _build_execution_time(
        num_layers=3,
        dense_kernel_times=(0.2, 0.4, 0.3),
    )
    execution_time.attention_operator_times = AttentionOperatorTimes(
        {
            "attn_kv_cache_save": 0.01,
            "attn_prefill": 0.02,
            "attn_decode": 0.03,
        }
    )

    op_times = get_attention_trace_op_times(
        execution_time,
        DENSE_ATTENTION_FAMILY,
    )

    assert [(op.name, time_ms) for op, time_ms in op_times] == [
        ("attn_kv_cache_save", pytest.approx(0.03)),
        ("attn_prefill", pytest.approx(0.06)),
        ("attn_decode", pytest.approx(0.09)),
    ]
    assert execution_time.attention_kv_cache_save_execution_time == pytest.approx(0.03)


def test_trace_mapper_rejects_missing_structured_attention_operator_time() -> None:
    execution_time = _build_execution_time(num_layers=2)
    execution_time.attention_operator_times = AttentionOperatorTimes(
        {
            "attn_kv_cache_save": 0.01,
            "attn_prefill": 0.02,
        }
    )

    with pytest.raises(ValueError, match="missing structured attention operator"):
        get_attention_trace_op_times(
            execution_time,
            DENSE_ATTENTION_FAMILY,
        )


def test_structured_attention_operator_times_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="Negative attention operator timing"):
        AttentionOperatorTimes(
            {
                "attn_kv_cache_save": 0.01,
                "attn_prefill": -0.02,
                "attn_decode": 0.03,
            }
        )


def test_attention_total_time_includes_structured_operator_times_once() -> None:
    execution_time = _build_execution_time(
        num_layers=4,
        dense_kernel_times=(0.0, 0.0, 0.0),
        mla_times=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    execution_time.attention_operator_times = AttentionOperatorTimes(
        {
            "attn_kv_cache_save": 0.11,
            "attn_prefill": 0.12,
        }
    )

    assert execution_time.get_single_layer_attention_time() == pytest.approx(
        0.5 + 0.7 + 0.8 + 0.4 + 0.11 + 0.12
    )
    assert execution_time.attention_time == pytest.approx(
        (0.5 + 0.7 + 0.8 + 0.4 + 0.11 + 0.12) * 4
    )


def test_attention_total_time_uses_structured_values_without_legacy_double_count() -> None:
    execution_time = _build_execution_time(
        num_layers=2,
        dense_kernel_times=(0.2, 0.4, 0.3),
        mla_times=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    execution_time.attention_operator_times = AttentionOperatorTimes(
        {
            "attn_kv_cache_save": 0.01,
            "attn_prefill": 0.02,
            "attn_decode": 0.03,
        }
    )

    assert execution_time.get_single_layer_attention_time() == pytest.approx(
        0.5 + 0.7 + 0.8 + 0.4 + 0.01 + 0.02 + 0.03
    )
    assert execution_time.attention_kv_cache_save_execution_time == pytest.approx(0.02)


def test_structured_attention_legacy_coverage_uses_family_execution_time_attrs(
    monkeypatch,
) -> None:
    custom_family = AttentionFamilySpec(
        family_id="custom_structured_total_coverage",
        display_name="Custom Structured Total Coverage",
        supported_variants=("gqa",),
        operators=(
            AttentionOperatorSpec(
                name="custom_attn_core",
                role=AttentionOperatorRole.PREFILL_KERNEL,
                resource_class=ResourceClass.COMP,
                phases=(AttentionPhase.PREFILL,),
                execution_time_attr="attn_norm_time",
            ),
        ),
        memory_layout=AttentionMemoryLayout.DENSE_KV,
        dense_compatible=True,
        requires_runtime_kv_helpers=False,
    )
    monkeypatch.setattr(
        time_components,
        "iter_execution_enabled_families",
        lambda: (custom_family,),
        raising=False,
    )
    attention_time = time_components.AttentionTime(
        attn_norm_time=1.7,
        operator_times=time_components.AttentionOperatorTimes(
            {"custom_attn_core": 0.2}
        ),
    )

    assert attention_time.operator_times is not None
    assert attention_time.operator_times.legacy_covered_time(attention_time) == (
        pytest.approx(1.7)
    )
    assert attention_time.total_time() == pytest.approx(0.2)


def test_partial_structured_legacy_coverage_keeps_total_time_but_fails_trace() -> None:
    execution_time = _build_execution_time(
        num_layers=2,
        dense_kernel_times=(0.2, 0.4, 0.3),
        mla_times=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    execution_time.attention_operator_times = AttentionOperatorTimes(
        {
            "attn_kv_cache_save": 0.01,
            "attn_prefill": 0.02,
        }
    )

    assert execution_time.get_single_layer_attention_time() == pytest.approx(
        0.5 + 0.7 + 0.8 + 0.4 + 0.3 + 0.01 + 0.02
    )
    with pytest.raises(ValueError, match="missing structured attention operator"):
        get_attention_trace_op_times(execution_time, DENSE_ATTENTION_FAMILY)


def test_mla_trace_mapper_preserves_vllm_order_and_execution_time_values() -> None:
    execution_time = _build_execution_time(num_layers=2)

    op_times = get_attention_trace_op_times(
        execution_time,
        LATENT_MLA_ATTENTION_FAMILY,
    )

    assert [(op.name, time_ms) for op, time_ms in op_times] == [
        ("attn_mla_kv_cache_save", pytest.approx(0.22)),
        ("attn_mla_prefill_kv_up_proj", pytest.approx(0.24)),
        ("attn_mla_prefill", pytest.approx(0.26)),
        ("attn_mla_decode_q_latent_proj", pytest.approx(0.28)),
        ("attn_mla_decode", pytest.approx(0.30)),
        ("attn_mla_v_up_proj", pytest.approx(0.32)),
    ]


def test_mla_trace_mapper_allows_documented_disjoint_model_projection_scope() -> None:
    execution_time = _build_execution_time(num_layers=2)

    op_times = get_attention_trace_op_times(
        execution_time,
        LATENT_MLA_ATTENTION_FAMILY,
    )

    assert execution_time.attention_pre_proj_time == pytest.approx(1.4)
    assert [op.name for op, _ in op_times] == [
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ]


def test_trace_mapper_rejects_mla_projection_double_count_without_disjoint_rule() -> None:
    execution_time = _build_execution_time(num_layers=1)
    family = AttentionFamilySpec(
        family_id="custom_mla_without_projection_rule",
        display_name="Custom MLA Without Projection Rule",
        supported_variants=("mla",),
        operators=(
            AttentionOperatorSpec(
                name="custom_mla_prefill_kv_up_proj",
                role=AttentionOperatorRole.PROJECTION,
                resource_class=ResourceClass.COMP,
                phases=(AttentionPhase.PREFILL,),
                execution_time_attr="attn_mla_prefill_kv_up_proj_time",
                projection_ownership=(
                    ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE
                ),
            ),
        ),
        memory_layout=AttentionMemoryLayout.LATENT_MLA,
        dense_compatible=False,
        requires_runtime_kv_helpers=True,
    )

    with pytest.raises(
        ValueError,
        match=(
            "projection ownership.*attention_layer_pre_proj_execution_time"
            ".*custom_mla_prefill_kv_up_proj"
        ),
    ):
        get_attention_trace_op_times(
            execution_time,
            family,
            skip_zero=False,
        )


def test_trace_mapper_rejects_unknown_disjoint_projection_attr() -> None:
    execution_time = _build_execution_time(num_layers=1)
    family = AttentionFamilySpec(
        family_id="custom_mla_with_typo_projection_rule",
        display_name="Custom MLA With Typo Projection Rule",
        supported_variants=("mla",),
        operators=(
            AttentionOperatorSpec(
                name="custom_mla_prefill_kv_up_proj",
                role=AttentionOperatorRole.PROJECTION,
                resource_class=ResourceClass.COMP,
                phases=(AttentionPhase.PREFILL,),
                execution_time_attr="attn_mla_prefill_kv_up_proj_time",
                projection_ownership=(
                    ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE
                ),
            ),
        ),
        memory_layout=AttentionMemoryLayout.LATENT_MLA,
        dense_compatible=False,
        requires_runtime_kv_helpers=True,
        disjoint_model_projection_attrs=("missing_projection_attr",),
    )

    with pytest.raises(ValueError, match="Unknown disjoint model projection attr"):
        get_attention_trace_op_times(
            execution_time,
            family,
            skip_zero=False,
        )


def test_projection_ownership_guard_checks_structured_operator_times() -> None:
    execution_time = _build_execution_time(num_layers=1, mla_times=(0.0,) * 6)
    execution_time.attention_operator_times = AttentionOperatorTimes(
        {
            "attn_mla_prefill_kv_up_proj": 0.12,
        }
    )
    family = AttentionFamilySpec(
        family_id="custom_structured_mla_without_projection_rule",
        display_name="Custom Structured MLA Without Projection Rule",
        supported_variants=("mla",),
        operators=(
            AttentionOperatorSpec(
                name="attn_mla_prefill_kv_up_proj",
                role=AttentionOperatorRole.PROJECTION,
                resource_class=ResourceClass.COMP,
                phases=(AttentionPhase.PREFILL,),
                execution_time_attr="attn_mla_prefill_kv_up_proj_time",
                projection_ownership=(
                    ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE
                ),
            ),
        ),
        memory_layout=AttentionMemoryLayout.LATENT_MLA,
        dense_compatible=False,
        requires_runtime_kv_helpers=True,
    )

    with pytest.raises(ValueError, match="projection ownership"):
        get_attention_trace_op_times(
            execution_time,
            family,
            skip_zero=False,
        )


def test_trace_mapper_supports_per_layer_division_without_changing_total_source() -> None:
    execution_time = _build_execution_time(num_layers=4)

    op_times = get_attention_trace_op_times(
        execution_time,
        LATENT_MLA_ATTENTION_FAMILY,
        per_layer_count=4,
    )

    assert [(op.name, time_ms) for op, time_ms in op_times] == [
        ("attn_mla_kv_cache_save", pytest.approx(0.11)),
        ("attn_mla_prefill_kv_up_proj", pytest.approx(0.12)),
        ("attn_mla_prefill", pytest.approx(0.13)),
        ("attn_mla_decode_q_latent_proj", pytest.approx(0.14)),
        ("attn_mla_decode", pytest.approx(0.15)),
        ("attn_mla_v_up_proj", pytest.approx(0.16)),
    ]


def test_trace_mapper_skips_zero_mla_timings_by_default() -> None:
    execution_time = _build_execution_time(mla_times=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    op_times = get_attention_trace_op_times(
        execution_time,
        LATENT_MLA_ATTENTION_FAMILY,
    )

    assert op_times == ()


def test_trace_mapper_can_emit_zero_timings_for_diagnostics() -> None:
    execution_time = _build_execution_time(mla_times=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    op_times = get_attention_trace_op_times(
        execution_time,
        LATENT_MLA_ATTENTION_FAMILY,
        skip_zero=False,
    )

    assert [op.name for op, _ in op_times] == [
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ]
    assert all(time_ms == pytest.approx(0.0) for _, time_ms in op_times)


def test_trace_mapper_rejects_negative_timings() -> None:
    execution_time = _build_execution_time(
        mla_times=(0.11, 0.12, -0.13, 0.14, 0.15, 0.16)
    )

    with pytest.raises(ValueError, match="Negative attention trace timing"):
        get_attention_trace_op_times(
            execution_time,
            LATENT_MLA_ATTENTION_FAMILY,
            skip_zero=False,
        )


def test_trace_mapper_rejects_frozen_dsa_execution() -> None:
    execution_time = _build_execution_time()

    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        get_attention_trace_op_times(execution_time, DSA_ATTENTION_FAMILY)

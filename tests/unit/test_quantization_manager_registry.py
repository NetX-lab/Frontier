import pytest

import frontier.attention as attention_package
import frontier.config.quantization_manager as quantization_manager_module
from frontier.attention.ops import (
    AttentionFamilySpec,
    AttentionMemoryLayout,
    AttentionOperatorRole,
    AttentionOperatorSpec,
    AttentionPhase,
)

from frontier.operators.spec import ResourceClass
from frontier.config.quantization_manager import QuantizationManager


def _dense_operator(
    name: str,
    role: AttentionOperatorRole,
    execution_time_attr: str,
) -> AttentionOperatorSpec:
    return AttentionOperatorSpec(
        name=name,
        role=role,
        resource_class=(
            ResourceClass.MEMORY
            if role is AttentionOperatorRole.CACHE_WRITE
            else ResourceClass.COMP
        ),
        phases=(AttentionPhase.PREFILL, AttentionPhase.DECODE),
        execution_time_attr=execution_time_attr,
    )


def test_default_quantization_registry_derives_dense_attention_ops_from_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renamed_dense_family = AttentionFamilySpec(
        family_id="dense_attention",
        display_name="Renamed Dense Attention",
        supported_variants=("gqa",),
        operators=(
            _dense_operator(
                "quant_cache",
                AttentionOperatorRole.CACHE_WRITE,
                "attention_kv_cache_save_execution_time",
            ),
            _dense_operator(
                "quant_prefill",
                AttentionOperatorRole.PREFILL_KERNEL,
                "attention_prefill_execution_time",
            ),
            _dense_operator(
                "quant_decode",
                AttentionOperatorRole.DECODE_KERNEL,
                "attention_decode_execution_time",
            ),
        ),
        memory_layout=AttentionMemoryLayout.DENSE_KV,
        dense_compatible=True,
        requires_runtime_kv_helpers=False,
    )
    monkeypatch.setattr(
        quantization_manager_module,
        "DENSE_ATTENTION_FAMILY",
        renamed_dense_family,
    )
    QuantizationManager.reset()
    try:
        manager = QuantizationManager()
        compute_ops = set(manager.get_supported_operations()["compute_operations"])
        assert {"quant_cache", "quant_prefill", "quant_decode"}.issubset(
            compute_ops
        )
        assert not {
            "attn_kv_cache_save",
            "attn_prefill",
            "attn_decode",
        } & compute_ops
    finally:
        QuantizationManager.reset()


def test_default_quantization_registry_derives_all_enabled_attention_family_ops() -> None:
    from frontier.attention.families import iter_attention_families

    QuantizationManager.reset()
    try:
        manager = QuantizationManager()
        compute_ops = set(manager.get_supported_operations()["compute_operations"])
        expected_attention_ops = {
            operator.name
            for family in iter_attention_families()
            if not family.dsa_frozen
            for operator in family.profiling_ops()
        }

        assert expected_attention_ops.issubset(compute_ops)
        for op_name in expected_attention_ops:
            assert manager.is_compute_operation(op_name)
            assert manager.get_precision(op_name).name == "FP16"
    finally:
        QuantizationManager.reset()


def test_attention_package_keeps_trace_mapper_lazy_export_after_config_import() -> None:
    from frontier.attention import get_attention_trace_op_times

    assert attention_package.get_attention_trace_op_times.__name__ == (
        "get_attention_trace_op_times"
    )
    assert get_attention_trace_op_times.__name__ == "get_attention_trace_op_times"

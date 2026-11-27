import pytest
from types import SimpleNamespace

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


def test_precision_metadata_excludes_inactive_gdn_operations_for_dense_model() -> None:
    """Dense simulations must not gain hybrid-only metadata rows."""

    QuantizationManager.reset()
    try:
        manager = QuantizationManager()
        dense_model = SimpleNamespace(
            get_default_precision=lambda: quantization_manager_module.PrecisionType.FP16,
            get_name=lambda: "dense-fixture",
            torch_dtype="float16",
            quantization_config=None,
            get_quant_signature=lambda: "none",
        )
        manager.configure_from_model_config(dense_model)
        names = {row["operation"] for row in manager.get_operation_precision_metadata()}
        assert not {"gdn_core_prefill", "gdn_core_decode", "gdn_input_projections", "gdn_output_projection"} & names
    finally:
        QuantizationManager.reset()


def test_precision_metadata_includes_gdn_operations_for_hybrid_model() -> None:
    QuantizationManager.reset()
    try:
        manager = QuantizationManager()
        hybrid_model = SimpleNamespace(
            get_default_precision=lambda: quantization_manager_module.PrecisionType.FP16,
            get_name=lambda: "hybrid-fixture",
            torch_dtype="float16",
            quantization_config=None,
            get_quant_signature=lambda: "none",
            get_num_gdn_layers=lambda: 1,
        )
        manager.configure_from_model_config(hybrid_model)
        names = {row["operation"] for row in manager.get_operation_precision_metadata()}
        assert {"gdn_core_prefill", "gdn_core_decode", "gdn_input_projections", "gdn_output_projection"} <= names
    finally:
        QuantizationManager.reset()


def test_optional_quantized_operations_none_preserves_config_hash_identity() -> None:
    QuantizationManager.reset()
    try:
        manager = QuantizationManager()
        model = SimpleNamespace(
            get_default_precision=lambda: quantization_manager_module.PrecisionType.BF16,
            get_name=lambda: "moe-fixture",
            torch_dtype="bfloat16",
            quantization_config=SimpleNamespace(
                quant_method=None,
                to_dict=lambda: {
                    "quant_method": None,
                    "activation_scheme": None,
                    "is_checkpoint_fp8_serialized": False,
                    "weight_block_size": None,
                    "ignored_layers": [],
                    "quantized_operations": None,
                },
            ),
            get_quant_signature=lambda: "none",
        )
        manager.configure_from_model_config(model)
        assert "quantized_operations" not in manager._config["quantization_config"]
    finally:
        QuantizationManager.reset()


def test_attention_package_keeps_trace_mapper_lazy_export_after_config_import() -> None:
    from frontier.attention import get_attention_trace_op_times

    assert attention_package.get_attention_trace_op_times.__name__ == (
        "get_attention_trace_op_times"
    )
    assert get_attention_trace_op_times.__name__ == "get_attention_trace_op_times"


def test_register_profiling_metadata_uses_per_operator_precision() -> None:
    """A mixed FP8/BF16 table must not approximate its BF16 operators."""

    QuantizationManager.reset()
    try:
        manager = QuantizationManager()
        signature = "method=fp8|act=none|serialized=False|block=none"
        model = SimpleNamespace(
            get_default_precision=lambda: quantization_manager_module.PrecisionType.BF16,
            get_name=lambda: "dense-fixture",
            torch_dtype="bfloat16",
            quantization_config=SimpleNamespace(
                quant_method="fp8",
                quantized_operations=None,
                to_dict=lambda: {
                    "quant_method": "fp8",
                    "activation_scheme": None,
                    "is_checkpoint_fp8_serialized": False,
                    "weight_block_size": None,
                    "ignored_layers": [],
                    "quantized_operations": None,
                },
            ),
            get_quant_signature=lambda: signature,
        )
        manager.configure_from_model_config(model)
        manager.register_profiling_metadata(
            operation_names=[
                "mlp_up_proj",
                "mlp_down_proj",
                "mlp_act",
                "post_attention_layernorm",
            ],
            profiling_precision=quantization_manager_module.PrecisionType.FP8,
            profiling_precision_by_op={
                "mlp_up_proj": quantization_manager_module.PrecisionType.FP8,
                "mlp_down_proj": quantization_manager_module.PrecisionType.FP8,
                "mlp_act": quantization_manager_module.PrecisionType.BF16,
                "post_attention_layernorm": quantization_manager_module.PrecisionType.BF16,
            },
            profiling_quant_signature=signature,
            expected_quant_signature=signature,
            file_path="fixture.csv",
        )

        metadata = {
            row["operation"]: row
            for row in manager.get_operation_precision_metadata()
        }
        assert metadata["mlp_up_proj"]["precision"] == "FP8"
        assert metadata["mlp_up_proj"]["data_source"] == "profiling"
        assert metadata["mlp_act"]["precision"] == "BF16"
        assert metadata["mlp_act"]["data_source"] == "profiling"
        assert metadata["mlp_act"]["approximation_factor"] is None
        assert metadata["post_attention_layernorm"]["data_source"] == "profiling"
    finally:
        QuantizationManager.reset()

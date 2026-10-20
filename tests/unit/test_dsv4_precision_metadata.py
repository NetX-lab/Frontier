"""Model/source precision inventory must not impersonate measured V4 support."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.config.precision_type import PrecisionType
from frontier.config.quantization_manager import QuantizationManager


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> BaseModelConfig:
    monkeypatch.chdir(REPO_ROOT)
    return BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")


@pytest.fixture
def manager() -> QuantizationManager:
    QuantizationManager.reset()
    instance = QuantizationManager()
    yield instance
    QuantizationManager.reset()


def _rows(manager: QuantizationManager) -> dict[str, dict]:
    return {
        row["operation"]: row
        for row in manager.get_operation_precision_metadata()
    }


def test_routed_and_shared_experts_do_not_share_one_fp8_precision(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    manager.configure_from_model_config(model)
    rows = _rows(manager)
    routed = rows["moe_grouped_gemm"]
    assert routed["precision"] == "FP4"
    assert routed["quantization_format"] == "mxfp4_e2m1"
    assert routed["weight_dtype"] == routed["scale_storage_dtype"] == "uint8"
    assert routed["quantized_value_dtype"] == "float4_e2m1"
    assert routed["precision_basis"] == "decoded quantized weight format"
    assert routed["weight_block_size"] == [1, 32]
    assert routed["packed_values_per_byte"] == 2
    assert routed["scale_dtype"] == "ue8m0"
    for operation in ("share_expert_up_proj", "share_expert_down_proj"):
        shared = rows[operation]
        assert shared["precision"] == "FP8"
        assert shared["quantization_format"] == "fp8_e4m3"
        assert shared["weight_block_size"] == [128, 128]
        assert shared["scale_dtype"] == "ue8m0"
    assert manager.get_precision("moe_grouped_gemm") is PrecisionType.FP4
    assert manager.get_precision("share_expert_up_proj") is PrecisionType.FP8


def test_inventory_uses_canonical_v4_scopes_without_generic_duplicate_owners(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
    from frontier.operators.families import (
        DEEPSEEK_V4_MHC_FAMILY, MOE_FAMILY, SHARE_EXPERT_FAMILY,
    )

    manager.configure_from_model_config(model)
    rows = _rows(manager)
    expected = {
        operator.name
        for family in (
            DEEPSEEK_V4_ATTENTION_FAMILY, DEEPSEEK_V4_MHC_FAMILY,
            MOE_FAMILY, SHARE_EXPERT_FAMILY,
        )
        for operator in family.operators
    }
    assert set(rows) == expected | {"moe_gating"}
    assert not {"add", "emb", "input_layernorm", "attn_pre_proj"} & rows.keys()
    for row in rows.values():
        assert row["source_inventory_only"] is True
        assert row["data_source"] == "model_config_and_vllm_source"
        assert row["precision_source"] == "model_config_and_vllm_source"
        assert row["approximation_factor"] is None
        assert row["source_scope"].startswith("vllm/")
    assert DEEPSEEK_V4_ATTENTION_FAMILY.execution_enabled is False
    assert not DEEPSEEK_V4_ATTENTION_FAMILY.profiling_ops()
    assert DEEPSEEK_V4_MHC_FAMILY.execution_enabled is False


def test_compressor_projection_and_index_weight_projection_have_distinct_outputs(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    manager.configure_from_model_config(model)
    rows = _rows(manager)
    for name in ("dsv4_compressor_projection", "dsv4_index_compressor_projection"):
        assert rows[name]["weight_dtype"] == "bfloat16"
        assert rows[name]["activation_dtype"] == "bfloat16"
        assert rows[name]["output_dtype"] == "float32"
        assert rows[name]["accumulation_dtype"] == "float32"
    assert rows["dsv4_index_weights_projection"]["output_dtype"] == "bfloat16"
    for name in (
        "dsv4_compressor_save_partial_states",
        "dsv4_index_compressor_save_partial_states",
    ):
        assert rows[name]["precision"] == "FP32"
        assert rows[name]["state_dtype"] == "float32"


def test_fused_query_cache_and_index_preparation_preserve_mixed_tensor_contracts(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    manager.configure_from_model_config(model)
    rows = _rows(manager)
    fused = rows["dsv4_fused_q_kv_prepare"]
    assert fused["precision"] == "BF16"
    assert fused["precision_basis"] == "input activation dtype"
    assert fused["output_tensor_dtypes"] == {
        "query": "bfloat16", "kv_cache": "uint8",
    }
    index = rows["dsv4_index_fused_query_prepare"]
    assert index["output_tensor_dtypes"] == {
        "query": "float8_e4m3fn", "weights": "float32",
    }
    assert index["query_scale_tensor"] is False
    for name in (
        "dsv4_compressor_fused_compress_store",
        "dsv4_index_compressor_fused_compress_store",
    ):
        assert rows[name]["norm_weight_dtype"] == "bfloat16"
        assert rows[name]["ape_dtype"] == "float32"
        assert rows[name]["output_dtype"] == "uint8"


def test_mhc_does_not_turn_bf16_activations_into_uniform_fp32_gemms(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    manager.configure_from_model_config(model)
    rows = _rows(manager)
    for name in (
        "dsv4_mhc_first_pre", "dsv4_mhc_ffn_post_attn_pre",
        "dsv4_mhc_attn_post_ffn_pre", "dsv4_mhc_head",
    ):
        row = rows[name]
        assert row["precision"] == "BF16"
        assert row["activation_dtype"] == row["output_dtype"] == "bfloat16"
        assert row["function_weight_dtype"] == row["state_dtype"] == "float32"
        assert "accumulation_dtype" not in row
    assert "function_weight_dtype" not in rows["dsv4_mhc_final_post"]


def test_grouped_output_projection_distinguishes_stored_and_runtime_scales(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    manager.configure_from_model_config(model)
    row = _rows(manager)["dsv4_grouped_output_projection"]
    assert row["scale_dtype"] == "ue8m0"
    assert row["runtime_scale_dtype"] == "float32"
    assert row["runtime_scale_condition"] == "SM90 grouped einsum source branch"
    assert "selected_backend" not in row
    assert row["source_inventory_only"] is True


def test_integer_fill_is_described_without_a_floating_precision_lookup(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    manager.configure_from_model_config(model)
    rows = _rows(manager)
    fill = rows["dsv4_index_short_context_fill"]
    assert fill["precision"] == "INT32"
    assert fill["output_dtype"] == "int32"
    with pytest.raises(ValueError, match="integer-only source tensor"):
        manager.get_precision("dsv4_index_short_context_fill")
    topk = rows["dsv4_index_topk"]
    assert topk["precision"] == "FP32"
    assert topk["precision_basis"] == "input score dtype"
    assert topk["output_dtype"] == "int32"


def test_lm_head_default_and_unrepresented_runtime_override_are_distinguished(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    manager.configure_from_model_config(model)
    row = _rows(manager)["dsv4_lm_head"]
    assert row["weight_dtype"] == row["default_output_dtype"] == "bfloat16"
    assert row["output_dtype"] == "source_scope_dependent"
    assert "head_dtype override" in row["output_dtype_condition"]


def test_source_projection_shapes_use_unpartitioned_constructor_dimensions(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    manager.configure_from_model_config(model)
    rows = _rows(manager)
    expected = {
        "dsv4_fused_qkv_projection": {"fused_wqa_wkv.weight": [1536, 4096]},
        "dsv4_fused_qkv_rmsnorm": {
            "q_norm.weight": [1024], "kv_norm.weight": [512],
        },
        "dsv4_q_projection": {"wq_b.weight": [32768, 1024]},
        "dsv4_grouped_output_projection": {"wo_a.weight": [8192, 4096]},
        "dsv4_output_projection": {"wo_b.weight": [4096, 8192]},
        "dsv4_index_query_projection": {"wq_b.weight": [8192, 1024]},
        "dsv4_index_weights_projection": {"weights_proj.weight": [64, 4096]},
    }
    for operation, shapes in expected.items():
        assert rows[operation]["source_parameter_shapes"] == shapes
        assert rows[operation]["shape_scope"] == (
            "unpartitioned source dimensions before TP or backend layout conversion; "
            "not runtime workload shapes or memory accounting"
        )
    assert rows["dsv4_grouped_output_projection"][
        "source_grouped_weight_view_shape"
    ] == [8, 1024, 4096]


def test_source_compressor_shapes_preserve_csa_hca_and_indexer_contracts(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    manager.configure_from_model_config(model)
    rows = _rows(manager)
    assert rows["dsv4_compressor_projection"][
        "source_parameter_shapes_by_attention_variant"
    ] == {
        "csa": {"fused_wkv_wgate.weight": [2048, 4096]},
        "hca": {"fused_wkv_wgate.weight": [1024, 4096]},
    }
    save = rows["dsv4_compressor_save_partial_states"]
    assert save["source_parameter_shapes_by_attention_variant"] == {
        "csa": {"ape": [4, 1024]}, "hca": {"ape": [128, 512]},
    }
    assert save["source_state_row_width_by_attention_variant"] == {
        "csa": 2048, "hca": 1024,
    }
    assert rows["dsv4_compressor_fused_compress_store"][
        "source_parameter_shapes_by_attention_variant"
    ] == {
        "csa": {"norm.weight": [512]}, "hca": {"norm.weight": [512]},
    }
    index_expected = {
        "dsv4_index_compressor_projection": {
            "fused_wkv_wgate.weight": [512, 4096],
        },
        "dsv4_index_compressor_save_partial_states": {"ape": [4, 256]},
        "dsv4_index_compressor_fused_compress_store": {"norm.weight": [128]},
    }
    for operation, shapes in index_expected.items():
        assert rows[operation]["source_parameter_shapes_by_attention_variant"] == {
            "csa": shapes,
        }
    assert rows["dsv4_index_compressor_save_partial_states"][
        "source_state_row_width_by_attention_variant"
    ] == {"csa": 512}


def test_source_shapes_are_config_derived_without_implying_runtime_support(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    altered = replace(
        model, embedding_dim=6144, num_q_heads=96, o_groups=12,
        q_lora_rank=1536, o_lora_rank=512, index_n_heads=32,
    )
    manager.configure_from_model_config(altered)
    rows = _rows(manager)
    assert rows["dsv4_fused_qkv_projection"]["source_parameter_shapes"] == {
        "fused_wqa_wkv.weight": [2048, 6144],
    }
    assert rows["dsv4_q_projection"]["source_parameter_shapes"] == {
        "wq_b.weight": [49152, 1536],
    }
    grouped = rows["dsv4_grouped_output_projection"]
    assert grouped["source_parameter_shapes"] == {"wo_a.weight": [6144, 4096]}
    assert grouped["source_grouped_weight_view_shape"] == [12, 512, 4096]
    assert rows["dsv4_output_projection"]["source_parameter_shapes"] == {
        "wo_b.weight": [6144, 6144],
    }
    assert rows["dsv4_index_query_projection"]["source_parameter_shapes"] == {
        "wq_b.weight": [4096, 1536],
    }
    assert rows["dsv4_index_weights_projection"]["source_parameter_shapes"] == {
        "weights_proj.weight": [32, 6144],
    }
    assert rows["dsv4_compressor_projection"][
        "source_parameter_shapes_by_attention_variant"
    ]["csa"] == {"fused_wkv_wgate.weight": [2048, 6144]}
    assert all(row["source_inventory_only"] for row in rows.values())


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_dim", 0), ("head_dim", None), ("num_q_heads", True),
        ("q_lora_rank", -1), ("o_groups", 3), ("o_lora_rank", 1.5),
        ("index_n_heads", "64"), ("index_head_dim", 0),
    ],
)
def test_malformed_source_dimensions_fail_without_replacing_configured_inventory(
    model: BaseModelConfig, manager: QuantizationManager, field: str, value: object,
) -> None:
    manager.configure_from_model_config(model)
    before = json.dumps(manager.get_operation_precision_metadata(), sort_keys=True)
    before_hash = manager.get_config_hash()
    altered = replace(model)
    setattr(altered, field, value)
    with pytest.raises(ValueError, match=f"DeepSeek-V4 source dimensions.*{field}"):
        manager.configure_from_model_config(altered)
    assert json.dumps(manager.get_operation_precision_metadata(), sort_keys=True) == before
    assert manager.get_config_hash() == before_hash


def test_source_shapes_do_not_assign_dynamic_workloads_or_duplicate_parameter_totals(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    manager.configure_from_model_config(model)
    rows = _rows(manager)
    for operation in (
        "dsv4_sparse_attention", "dsv4_fused_q_kv_prepare",
        "dsv4_output_inv_rope_quant", "dsv4_index_scores", "dsv4_index_topk",
        "dsv4_index_short_context_fill", "moe_grouped_gemm", "dsv4_lm_head",
    ):
        assert "shape_scope" not in rows[operation]
    for row in rows.values():
        assert not {
            "num_tokens", "padded_heads", "tensor_parallel_size", "selected_backend",
            "compressed_token_count", "parameter_bytes", "parameter_count",
        } & row.keys()


def _legacy_model() -> SimpleNamespace:
    return SimpleNamespace(
        get_default_precision=lambda: PrecisionType.FP16,
        get_name=lambda: "legacy-dense",
        torch_dtype="float16",
        quantization_config=None,
        get_quant_signature=lambda: "none",
    )


def test_reconfiguration_and_reset_preserve_legacy_rows_and_config_identity(
    model: BaseModelConfig, manager: QuantizationManager,
) -> None:
    legacy = _legacy_model()
    manager.configure_from_model_config(legacy)
    baseline = json.dumps(manager.get_operation_precision_metadata(), sort_keys=True)
    baseline_hash = manager.get_config_hash()
    for row in manager.get_operation_precision_metadata():
        assert set(row) == {
            "operation", "precision", "data_source", "approximation_factor",
        }
        assert not row["operation"].startswith("dsv4_")
    manager.configure_from_model_config(model)
    assert manager._config["expert_dtype"] == "fp4"
    manager.configure_from_model_config(legacy)
    assert json.dumps(manager.get_operation_precision_metadata(), sort_keys=True) == baseline
    assert manager.get_config_hash() == baseline_hash
    assert "expert_dtype" not in manager._config
    manager.configure_from_model_config(model)
    manager.load_config()
    assert not any(
        row["operation"].startswith("dsv4_")
        for row in manager.get_operation_precision_metadata()
    )
    assert manager.get_precision("moe_grouped_gemm") is PrecisionType.FP16


@pytest.mark.parametrize(
    "field,value",
    [
        ("expert_dtype", "bf16"), ("torch_dtype", "float16"),
        ("quant_method", "fp4"), ("activation_scheme", "static"),
        ("weight_block_size", (64, 128)), ("scale_fmt", "float32"),
        ("ignored_layers", ["lm_head"]),
        ("quantized_operations", ["moe_grouped_gemm"]),
        ("is_checkpoint_fp8_serialized", False),
    ],
)
def test_unsupported_v4_precision_does_not_silently_reuse_original_inventory(
    model: BaseModelConfig, manager: QuantizationManager, field: str, value: object,
) -> None:
    manager.configure_from_model_config(model)
    before = json.dumps(manager.get_operation_precision_metadata(), sort_keys=True)
    if field in {"expert_dtype", "torch_dtype"}:
        altered = replace(model, **{field: value})
    else:
        # Mutate a copied quantization config so its constructor's independent
        # FP8 validation does not mask the inventory's own unsupported guard.
        quant = replace(model.quantization_config)
        setattr(quant, field, value)
        altered = replace(model, quantization_config=quant)
    with pytest.raises(ValueError, match="DeepSeek-V4 source precision inventory requires"):
        manager.configure_from_model_config(altered)
    assert json.dumps(manager.get_operation_precision_metadata(), sort_keys=True) == before


def test_existing_metrics_csv_writer_retains_source_precision_fields(
    model: BaseModelConfig, manager: QuantizationManager, tmp_path: Path,
) -> None:
    from frontier.metrics.metrics_store import MetricsStore

    manager.configure_from_model_config(model)
    store = SimpleNamespace(_config=SimpleNamespace(output_dir=str(tmp_path)))
    MetricsStore._write_op_precision_metadata(store)
    frame = pd.read_csv(tmp_path / "op_precision_metadata.csv").set_index("operation")
    assert set(frame.index) == set(_rows(manager))
    assert frame.loc["moe_grouped_gemm", "precision"] == "FP4"
    assert frame.loc["moe_grouped_gemm", "weight_block_size"] == "[1, 32]"
    assert frame.loc["dsv4_compressor_projection", "output_dtype"] == "float32"
    assert frame.loc["dsv4_index_short_context_fill", "precision"] == "INT32"
    assert frame["source_inventory_only"].all()
    assert set(frame["data_source"]) == {"model_config_and_vllm_source"}
    for operation, row in _rows(manager).items():
        for field in (
            "source_parameter_shapes", "source_grouped_weight_view_shape",
            "source_parameter_shapes_by_attention_variant",
            "source_state_row_width_by_attention_variant",
        ):
            if field in row:
                assert ast.literal_eval(frame.loc[operation, field]) == row[field]
        if "shape_scope" in row:
            assert frame.loc[operation, "shape_scope"] == row["shape_scope"]

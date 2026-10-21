"""Native exact exclusions and packed-module precision without vLLM imports."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from frontier.attention.step5 import STEP5_MODULE_PREFIX_RENAMES, STEP5_PACKED_PROJECTIONS
from frontier.config.model_config import QuantizationConfig
from frontier.config.precision_type import PrecisionType


MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/config/models/stepfun-ai__Step-5-Preview.json"
)
# Pinned native model.py: STEP5_PACKED_MODULES_MAPPING. quant_utils.py uses
# exact fused-name precedence, uniform shard exclusions and expert weight names.
NATIVE_PACKED_MODULES = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "qkvg_proj": ("q_proj", "k_proj", "v_proj", "g_proj"),
    "qkv_indexer_proj": (
        "q_proj", "k_proj", "v_proj", "sparse_indexer_q",
        "sparse_indexer_k", "sparse_indexer_z", "g_proj",
    ),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


@pytest.fixture
def native_quantization() -> QuantizationConfig:
    raw = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return QuantizationConfig.from_dict(raw["quantization_config"])


def test_step_packed_projection_mapping_retains_every_native_shard() -> None:
    assert STEP5_PACKED_PROJECTIONS == NATIVE_PACKED_MODULES


@pytest.mark.parametrize("model_precision", [
    PrecisionType.BF16, PrecisionType.FP16, PrecisionType.FP32,
])
def test_exact_module_exclusion_returns_the_supplied_model_precision(
    model_precision: PrecisionType,
) -> None:
    quantization = QuantizationConfig(
        quant_method="fp8", ignored_layers=["model.layers.8.self_attn.o_proj"],
    )
    assert quantization.resolve_module_precision(
        "model.layers.8.self_attn.o_proj", model_precision,
    ) is model_precision
    assert quantization.resolve_module_precision(
        "model.layers.88.self_attn.o_proj", model_precision,
    ) is PrecisionType.FP8


@pytest.mark.parametrize("module_name", list(NATIVE_PACKED_MODULES))
def test_exact_fused_name_precedes_incomplete_shard_exclusions(module_name: str) -> None:
    shards = NATIVE_PACKED_MODULES[module_name]
    prefix = f"model.layers.3.self_attn.{module_name}"
    quantization = QuantizationConfig(
        quant_method="fp8",
        ignored_layers=[prefix, f"model.layers.3.self_attn.{shards[0]}"],
    )
    assert quantization.resolve_module_precision(
        prefix, PrecisionType.BF16, packed_modules=shards,
    ) is PrecisionType.BF16


@pytest.mark.parametrize("module_name", list(NATIVE_PACKED_MODULES))
@pytest.mark.parametrize("exclude_shards", [False, True])
def test_packed_modules_require_one_precision_for_all_checkpoint_shards(
    module_name: str, exclude_shards: bool,
) -> None:
    shards = NATIVE_PACKED_MODULES[module_name]
    parent = "model.layers.3.self_attn"
    ignored = [f"{parent}.{name}" for name in shards] if exclude_shards else []
    quantization = QuantizationConfig(quant_method="fp8", ignored_layers=ignored)
    expected = PrecisionType.BF16 if exclude_shards else PrecisionType.FP8
    assert quantization.resolve_module_precision(
        f"{parent}.{module_name}", PrecisionType.BF16, packed_modules=shards,
    ) is expected


@pytest.mark.parametrize(("module_name", "excluded_index"), [
    (module_name, index)
    for module_name, shards in NATIVE_PACKED_MODULES.items()
    for index in range(len(shards))
])
def test_each_partial_packed_projection_exclusion_fails_closed(
    module_name: str, excluded_index: int,
) -> None:
    shards = NATIVE_PACKED_MODULES[module_name]
    parent = "model.layers.3.self_attn"
    quantization = QuantizationConfig(
        quant_method="fp8", ignored_layers=[f"{parent}.{shards[excluded_index]}"],
    )
    with pytest.raises(ValueError, match="mixed quantization exclusions"):
        quantization.resolve_module_precision(
            f"{parent}.{module_name}", PrecisionType.BF16, packed_modules=shards,
        )


@pytest.mark.parametrize(("excluded_layer", "queried_layer"), [(8, 88), (88, 8)])
def test_expert_weight_exclusions_distinguish_canonical_layer_8_and_88_prefixes(
    excluded_layer: int, queried_layer: int,
) -> None:
    quantization = QuantizationConfig(
        quant_method="fp8",
        ignored_layers=[f"model.layers.{excluded_layer}.moe.experts.w13_weight"],
    )
    assert quantization.resolve_module_precision(
        f"model.layers.{excluded_layer}.moe.experts", PrecisionType.BF16,
        routed_experts=True,
    ) is PrecisionType.BF16
    assert quantization.resolve_module_precision(
        f"model.layers.{queried_layer}.moe.experts", PrecisionType.BF16,
        routed_experts=True,
    ) is PrecisionType.FP8


@pytest.mark.parametrize("excluded_weight", [
    "checkpoint.model.layers.88.moe.experts.w13_weight",
    "wrapped_model.layers.88.moe.experts.w2_weight",
])
def test_routed_expert_lookup_preserves_native_substring_matches_in_weight_paths(
    excluded_weight: str,
) -> None:
    # quant_utils.is_layer_skipped uses substring containment for experts.
    # The second path contains the canonical prefix inside a component.
    prefix = "model.layers.88.moe.experts"
    assert prefix in excluded_weight
    quantization = QuantizationConfig(
        quant_method="fp8", ignored_layers=[excluded_weight],
    )
    assert quantization.resolve_module_precision(
        prefix, PrecisionType.BF16, routed_experts=True,
    ) is PrecisionType.BF16
    assert quantization.resolve_module_precision(
        prefix, PrecisionType.BF16,
    ) is PrecisionType.FP8


def test_native_expert_weight_exclusions_keep_85_fp8_and_three_bf16_layers(
    native_quantization: QuantizationConfig,
) -> None:
    expected_weight_exclusions = [
        f"model.layers.{layer_id}.moe.experts.{weight}"
        for layer_id in (88, 89, 90)
        for weight in ("w13_weight", "w2_weight")
    ]
    assert [
        name for name in native_quantization.ignored_layers if ".moe.experts." in name
    ] == expected_weight_exclusions
    actual = [
        native_quantization.resolve_module_precision(
            f"model.layers.{layer_id}.moe.experts", PrecisionType.BF16,
            routed_experts=True,
        )
        for layer_id in range(3, 91)
    ]
    assert actual == [PrecisionType.FP8] * 85 + [PrecisionType.BF16] * 3
    assert Counter(actual) == {PrecisionType.FP8: 85, PrecisionType.BF16: 3}
    for layer_id, expected in zip(range(3, 91), actual):
        for weight in ("w13_weight", "w2_weight"):
            assert native_quantization.resolve_module_precision(
                f"model.layers.{layer_id}.moe.experts.{weight}", PrecisionType.BF16,
            ) is expected


def test_native_attention_fused_projection_exclusions_cover_all_92_layers(
    native_quantization: QuantizationConfig,
) -> None:
    for layer_id in range(92):
        module = "qkv_indexer_proj" if layer_id % 4 == 3 else "qkv_proj"
        prefix = f"model.layers.{layer_id}.self_attn.{module}"
        assert prefix in native_quantization.ignored_layers
        assert native_quantization.resolve_module_precision(
            prefix, PrecisionType.BF16, packed_modules=NATIVE_PACKED_MODULES[module],
        ) is PrecisionType.BF16
        assert native_quantization.resolve_module_precision(
            f"model.layers.{layer_id}.self_attn.o_proj", PrecisionType.BF16,
        ) is PrecisionType.BF16


def test_checkpoint_dense_shared_and_router_exclusion_prefixes_preserve_bf16(
    native_quantization: QuantizationConfig,
) -> None:
    for layer_id in (0, 1, 2, 91):
        prefix = f"model.layers.{layer_id}.mlp"
        assert native_quantization.resolve_module_precision(
            f"{prefix}.gate_up_proj", PrecisionType.BF16,
            packed_modules=NATIVE_PACKED_MODULES["gate_up_proj"],
        ) is PrecisionType.BF16
        assert native_quantization.resolve_module_precision(
            f"{prefix}.down_proj", PrecisionType.BF16,
        ) is PrecisionType.BF16
    for layer_id in range(3, 91):
        prefix = f"model.layers.{layer_id}.share_expert"
        assert native_quantization.resolve_module_precision(
            f"{prefix}.gate_up_proj", PrecisionType.BF16,
            packed_modules=NATIVE_PACKED_MODULES["gate_up_proj"],
        ) is PrecisionType.BF16
        assert native_quantization.resolve_module_precision(
            f"{prefix}.down_proj", PrecisionType.BF16,
        ) is PrecisionType.BF16
        assert native_quantization.resolve_module_precision(
            f"model.layers.{layer_id}.moe.gate", PrecisionType.BF16,
        ) is PrecisionType.BF16


def test_native_shared_expert_renames_preserve_bf16_without_mutating_checkpoint_exclusions(
    native_quantization: QuantizationConfig,
) -> None:
    original = list(native_quantization.ignored_layers)
    signature = native_quantization.get_quant_signature()
    for layer_id in range(3, 91):
        checkpoint_prefix = f"model.layers.{layer_id}.share_expert"
        native_prefix = f"model.layers.{layer_id}.moe.share_expert"
        assert f"{checkpoint_prefix}.gate_up_proj" in original
        assert f"{native_prefix}.gate_up_proj" not in original
        for module, shards in (("gate_up_proj", NATIVE_PACKED_MODULES["gate_up_proj"]), ("down_proj", ())):
            assert native_quantization.resolve_module_precision(
                f"{native_prefix}.{module}", PrecisionType.BF16,
                packed_modules=shards, module_prefix_renames=STEP5_MODULE_PREFIX_RENAMES,
            ) is PrecisionType.BF16
    assert native_quantization.ignored_layers == original
    assert native_quantization.get_quant_signature() == signature


def test_native_rename_applies_to_shards_before_packed_precision_resolution() -> None:
    quantization = QuantizationConfig(
        quant_method="fp8", ignored_layers=[
            "model.layers.3.share_expert.gate_proj",
            "model.layers.3.share_expert.up_proj",
        ],
    )
    assert quantization.resolve_module_precision(
        "model.layers.3.moe.share_expert.gate_up_proj", PrecisionType.BF16,
        packed_modules=NATIVE_PACKED_MODULES["gate_up_proj"],
        module_prefix_renames=STEP5_MODULE_PREFIX_RENAMES,
    ) is PrecisionType.BF16


@pytest.mark.parametrize("model_precision", [
    PrecisionType.BF16, PrecisionType.FP16, PrecisionType.FP32,
])
def test_no_quantization_preserves_model_precision_for_each_module_kind(
    model_precision: PrecisionType,
) -> None:
    quantization = QuantizationConfig()
    assert quantization.resolve_module_precision(
        "model.layers.3.self_attn.o_proj", model_precision,
    ) is model_precision
    assert quantization.resolve_module_precision(
        "model.layers.3.self_attn.qkv_proj", model_precision,
        packed_modules=NATIVE_PACKED_MODULES["qkv_proj"],
    ) is model_precision
    assert quantization.resolve_module_precision(
        "model.layers.3.moe.experts", model_precision, routed_experts=True,
    ) is model_precision


@pytest.mark.parametrize("prefix", ["", None, 3])
def test_module_precision_rejects_missing_or_non_string_prefixes(prefix: object) -> None:
    quantization = QuantizationConfig(quant_method="fp8")
    with pytest.raises(ValueError, match="module_prefix"):
        quantization.resolve_module_precision(prefix, PrecisionType.BF16)


@pytest.mark.parametrize("model_precision", ["bf16", None, 2])
def test_module_precision_requires_the_declared_precision_enum(model_precision: object) -> None:
    quantization = QuantizationConfig(quant_method="fp8")
    with pytest.raises(ValueError, match="model_precision"):
        quantization.resolve_module_precision(
            "model.layers.3.self_attn.qkv_proj", model_precision,
        )


@pytest.mark.parametrize("ignored", [[""], [None], [3]])
def test_invalid_exclusion_entries_cannot_silently_quantize_a_module(ignored: list) -> None:
    with pytest.raises(ValueError, match="exclusions"):
        QuantizationConfig(quant_method="fp8", ignored_layers=ignored).resolve_module_precision(
            "model.layers.3.self_attn.qkv_proj", PrecisionType.BF16,
        )


@pytest.mark.parametrize(("prefix", "shards"), [
    ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    ("model.layers.3.self_attn.qkv_proj", ("self_attn.q_proj",)),
    ("model.layers.3.self_attn.qkv_proj", ("",)),
])
def test_packed_projection_arguments_require_qualified_module_and_local_shard_names(
    prefix: str, shards: tuple[str, ...],
) -> None:
    quantization = QuantizationConfig(quant_method="fp8")
    with pytest.raises(ValueError, match="packed_modules"):
        quantization.resolve_module_precision(
            prefix, PrecisionType.BF16, packed_modules=shards,
        )


def test_precision_resolution_preserves_the_native_exclusion_inventory_and_signature(
    native_quantization: QuantizationConfig,
) -> None:
    exclusions = list(native_quantization.ignored_layers)
    signature = native_quantization.get_quant_signature()
    native_quantization.resolve_module_precision(
        "model.layers.3.self_attn.qkv_proj", PrecisionType.BF16,
        packed_modules=NATIVE_PACKED_MODULES["qkv_proj"],
    )
    native_quantization.resolve_module_precision(
        "model.layers.88.moe.experts", PrecisionType.BF16, routed_experts=True,
    )
    assert len(exclusions) == 1406
    assert native_quantization.ignored_layers == exclusions
    assert native_quantization.get_quant_signature() == signature

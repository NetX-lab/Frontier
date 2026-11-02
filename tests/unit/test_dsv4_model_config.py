"""Original DeepSeek-V4-Flash metadata without claiming executable support."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention.families import get_attention_family
from frontier.attention.gdn.config import LayerAttentionSpec
from frontier.attention.model_binding import bind_attention_family, bind_layer_attention
from frontier.config.model_config import BaseModelConfig
from frontier.model_architectures import LayerKind, get_model_architecture_profile
from frontier.operators.binding import build_operator_manifest
from frontier.operators.spec import TensorParallelMode
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig


MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "data/config/models/deepseek-ai__DeepSeek-V4-Flash.json"


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> BaseModelConfig:
    monkeypatch.chdir(REPO_ROOT)
    return BaseModelConfig.create_from_name(MODEL_NAME)


def test_original_dsv4_has_43_routed_and_shared_ffn_layers(
    model: BaseModelConfig,
) -> None:
    assert model.num_layers == 43
    assert model.is_moe is True
    assert model.num_experts == 256
    assert model.num_experts_per_tok == 6
    assert model.get_moe_layer_ids() == list(range(43))
    assert model.get_num_moe_layers() == 43
    assert model.routed_mlp_hidden_dim == model.mlp_hidden_dim == 2048
    assert model.share_expert_dim == 2048

    profile = model.get_model_architecture_profile()
    assert profile.profile_id == "deepseek_v4"
    assert model.supports_share_expert() is True
    for layer_id in range(43):
        routed = profile.resolve_layer_contract(model, layer_id=layer_id)
        shared = profile.resolve_layer_contract(
            model, layer_id=layer_id, layer_kind=LayerKind.SHARED
        )
        assert (routed.layer_kind, routed.width) == (LayerKind.ROUTED, 2048)
        assert (shared.layer_kind, shared.width) == (LayerKind.SHARED, 2048)


def test_original_dsv4_attention_schedule_excludes_the_next_token_entry(
    model: BaseModelConfig,
) -> None:
    native = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    assert model.compress_ratios == tuple(native["compress_ratios"])
    assert len(model.compress_ratios) == 44
    assert model.compress_ratios[43:] == (0,)

    specs = model.get_layer_attention_specs()
    assert len(specs) == 43
    assert all(isinstance(spec, LayerAttentionSpec) for spec in specs)
    assert [spec.global_layer_id for spec in specs] == list(range(43))
    assert {spec.family_id for spec in specs} == {"deepseek_v4_attention"}
    assert [spec.variant_id for spec in specs[:4]] == [
        "uncompressed", "uncompressed", "csa", "hca"
    ]
    assert [spec.global_layer_id for spec in specs if spec.variant_id == "csa"] == (
        list(range(2, 43, 2))
    )
    assert [spec.global_layer_id for spec in specs if spec.variant_id == "hca"] == (
        list(range(3, 43, 2))
    )
    assert all(spec.index_producer_layer_id is None for spec in specs)
    with pytest.raises(ValueError, match="out of range"):
        bind_layer_attention(model, 43)


def test_original_dsv4_preserves_dimensions_and_model_specific_metadata(
    model: BaseModelConfig,
) -> None:
    assert model.use_mla is False
    assert model.num_q_heads == 64
    assert model.num_kv_heads == 1
    assert model.get_head_dim() == 512
    assert model.q_lora_rank == 1024
    assert model.qk_rope_head_dim == 64
    assert model.kv_lora_rank is None
    assert model.index_topk == 512
    assert model.index_n_heads == 64
    assert model.index_head_dim == 128
    assert model.sliding_window == 128
    assert model.num_hash_layers == 3
    assert model.hc_mult == 4
    assert model.hc_sinkhorn_iters == 20
    assert model.hc_eps == 1e-6
    assert model.o_groups == 8
    assert model.o_lora_rank == 1024
    assert model.compress_rope_theta == 160000
    assert model.swiglu_limit == 10.0
    assert model.scoring_func == "sqrtsoftplus"
    assert model.norm_topk_prob is True
    assert model.routed_scaling_factor == 1.5


def test_original_dsv4_keeps_fp8_base_and_fp4_expert_precision_distinct(
    model: BaseModelConfig,
) -> None:
    assert model.torch_dtype == "bfloat16"
    assert model.expert_dtype == "fp4"
    quant = model.quantization_config
    assert quant is not None
    assert quant.quant_method == "fp8"
    assert quant.fmt == "e4m3"
    assert quant.scale_fmt == "ue8m0"
    assert quant.activation_scheme == "dynamic"
    assert quant.is_checkpoint_fp8_serialized is True
    assert quant.weight_block_size == (128, 128)


def test_dsv4_profiling_and_serialization_preserve_the_runtime_contract(
    model: BaseModelConfig,
) -> None:
    profiling = ProfilingModelConfig.from_model_name(MODEL_NAME)
    restored = ProfilingModelConfig(**profiling.to_dict())
    fields = (
        "compress_ratios", "sliding_window", "num_hash_layers", "hc_mult",
        "hc_sinkhorn_iters", "hc_eps", "o_groups", "o_lora_rank",
        "compress_rope_theta", "swiglu_limit", "expert_dtype", "index_topk",
        "index_n_heads", "index_head_dim", "q_lora_rank", "qk_rope_head_dim",
        "scoring_func", "norm_topk_prob", "routed_scaling_factor",
    )
    for config in (profiling, restored):
        assert config.get_model_architecture_profile().profile_id == "deepseek_v4"
        assert config.use_mla is False
        assert config.get_moe_layer_ids() == model.get_moe_layer_ids()
        assert config.get_layer_attention_specs() == model.get_layer_attention_specs()
        assert config.get_quant_signature() == model.get_quant_signature()
        assert config.quantization_config is not None
        assert config.quantization_config.to_dict() == (
            model.quantization_config.to_dict()
        )
        for field_name in fields:
            assert getattr(config, field_name) == getattr(model, field_name), field_name


@pytest.mark.parametrize(
    ("model_type", "expected_profile"),
    [("deepseek_v4", "deepseek_v4"), ("unit_model", "generic")],
)
def test_dsv4_profile_uses_model_type_instead_of_the_name(
    model_type: str, expected_profile: str,
) -> None:
    config = SimpleNamespace(model_type=model_type, name=MODEL_NAME)
    assert get_model_architecture_profile(config).profile_id == expected_profile


def test_dsv4_rejects_compression_schedule_shorter_than_the_backbone(
    model: BaseModelConfig,
) -> None:
    with pytest.raises(ValueError, match="compress_ratios"):
        replace(model, compress_ratios=model.compress_ratios[:42]).get_layer_attention_specs()


@pytest.mark.parametrize("invalid_ratio", [8, -1, True])
def test_dsv4_rejects_unsupported_backbone_compression_ratios(
    model: BaseModelConfig, invalid_ratio: object,
) -> None:
    ratios = list(model.compress_ratios)
    ratios[2] = invalid_ratio
    with pytest.raises(ValueError, match="compress_ratios"):
        replace(model, compress_ratios=tuple(ratios)).get_layer_attention_specs()


def test_dsv4_representation_keeps_execution_explicitly_disabled(
    model: BaseModelConfig,
) -> None:
    family = get_attention_family(bind_layer_attention(model, 42).family_id)
    assert family.family_id == "deepseek_v4_attention"
    assert family.execution_enabled is False
    with pytest.raises(NotImplementedError, match="DeepSeek-V4"):
        family.require_enabled_for_execution()


def test_dsv4_whole_model_manifest_cannot_hide_compressed_layers(
    model: BaseModelConfig,
) -> None:
    for config in (model, ProfilingModelConfig.from_model_name(MODEL_NAME)):
        with pytest.raises(ValueError, match="explicit global layer id"):
            bind_attention_family(config)
        with pytest.raises(ValueError, match="explicit global layer id"):
            build_operator_manifest(config)


@pytest.mark.parametrize(
    ("tp", "dp", "pp", "ep", "expected_mode", "expected_tp"),
    [
        (8, 2, 1, 16, TensorParallelMode.REPLICATED, 1),
        (8, 4, 1, 32, TensorParallelMode.REPLICATED, 1),
        (8, 1, 1, 8, TensorParallelMode.ATTENTION_TP, 8),
        (8, 2, 2, 16, TensorParallelMode.ATTENTION_TP, 8),
        (8, 2, 1, 1, TensorParallelMode.ATTENTION_TP, 8),
        (1, 2, 1, 2, TensorParallelMode.ATTENTION_TP, 1),
    ],
)
def test_dsv4_shared_weight_contract_follows_source_sp_topology(
    model, tp, dp, pp, ep, expected_mode, expected_tp,
) -> None:
    contract = model.get_model_architecture_profile().resolve_layer_contract(
        model, layer_id=0, layer_kind=LayerKind.SHARED,
        attention_tp_size=tp, attention_dp_size=dp,
        num_pipeline_stages=pp, expert_parallel_size=ep,
    )
    assert contract.tensor_parallel_mode is expected_mode
    assert contract.tensor_parallel_size == expected_tp
    assert contract.effective_ffn_width == 2048


def test_new_routing_metadata_is_absent_for_legacy_model() -> None:
    from frontier.config.utils import dataclass_to_dict

    model = BaseModelConfig.create_from_name("step-moe-noquant")
    serialized = dataclass_to_dict(model)
    for name in ("scoring_func", "norm_topk_prob", "routed_scaling_factor"):
        assert getattr(model, name) is None
        assert name not in serialized

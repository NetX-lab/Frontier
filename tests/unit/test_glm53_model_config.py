"""GLM-5.3 model representation without claiming executable DSA support."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)

from frontier.attention.families import get_attention_family
from frontier.attention.gdn.config import LayerAttentionSpec
from frontier.attention.model_binding import bind_layer_attention
from frontier.config.model_config import BaseModelConfig
from frontier.config.model_config import QuantizationConfig
from frontier.config.utils import dataclass_to_dict
from frontier.model_architectures import LayerKind, get_model_architecture_profile
from frontier.operators.spec import TensorParallelMode
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig


MODEL_NAME = "zai-org/GLM-5.3"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(("model_name", "dummy_mode"), [
    ("deepseek-ai/DeepSeek-V4-Flash", False),
])
def test_predictor_initialization_rejects_frozen_model_before_mode_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_name: str, dummy_mode: bool,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    model = BaseModelConfig.create_from_name(model_name)
    with pytest.raises(NotImplementedError, match="(DSA|DeepSeek-V4)"):
        RandomForrestExecutionTimePredictor(
            # No V4 step-cost table sits next to this attention input path.
            predictor_config=SimpleNamespace(
                enable_dummy_mode=dummy_mode, atten_input_file=str(tmp_path / "attention.csv"),
            ),
            replica_config=SimpleNamespace(
                model_config=model, router_topk=model.num_experts_per_tok,
                moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
                device="h800", network_device="h800_dgx",
            ),
            replica_scheduler_config=SimpleNamespace(),
            metrics_config=SimpleNamespace(),
        )


def test_v4_dummy_constructor_requires_resolved_cache_selection(tmp_path) -> None:
    from frontier.config import (
        MetricsConfig, RandomForrestExecutionTimePredictorConfig, ReplicaConfig,
        VllmV1SchedulerConfig,
    )

    replica = ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h800",
        network_device="h800_dgx", attn_tensor_parallel_size=8, attn_dp=2,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
        attention_backend="FLASHMLA_SPARSE_DSV4", kv_cache_dtype="fp8",
        indexer_kv_cache_dtype="fp8",
    )
    predictor = RandomForrestExecutionTimePredictor(
        RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True), replica,
        VllmV1SchedulerConfig(block_size=256),
        MetricsConfig(cache_dir=str(tmp_path / "cache")),
    )
    assert predictor._enable_dummy_mode is True
    family = get_attention_family("deepseek_v4_attention")
    assert family.structural_execution_enabled is True
    assert family.execution_enabled is False
    family.require_enabled_for_execution(structural=True)
    with pytest.raises(NotImplementedError, match="DeepSeek-V4"):
        family.require_enabled_for_execution()
    replica.attention_backend = "FLASHMLA_SPARSE"
    with pytest.raises(NotImplementedError, match="DeepSeek-V4"):
        RandomForrestExecutionTimePredictor(
            RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True), replica,
            VllmV1SchedulerConfig(block_size=256),
            MetricsConfig(cache_dir=str(tmp_path / "unsupported-cache")),
        )


MODEL_PATH = REPO_ROOT / "data/config/models/zai-org__GLM-5.3.json"


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> BaseModelConfig:
    monkeypatch.chdir(REPO_ROOT)
    return BaseModelConfig.create_from_name(MODEL_NAME)


def test_native_glm53_ffn_schedule_and_widths(model: BaseModelConfig) -> None:
    assert model.num_layers == 78
    assert model.is_moe
    assert model.num_experts == 256
    assert model.num_experts_per_tok == 8
    assert model.get_moe_layer_ids() == list(range(3, 78))
    assert model.dense_mlp_hidden_dim == 12288
    assert model.routed_mlp_hidden_dim == model.mlp_hidden_dim == 2048
    assert model.share_expert_dim == 2048

    profile = model.get_model_architecture_profile()
    assert profile.profile_id == "glm_moe_dsa"
    dense = profile.resolve_layer_contract(model, layer_id=0)
    routed = profile.resolve_layer_contract(model, layer_id=3)
    shared = profile.resolve_layer_contract(
        model, layer_id=3, layer_kind=LayerKind.SHARED
    )
    assert (dense.layer_kind, dense.width) == (LayerKind.DENSE, 12288)
    assert (routed.layer_kind, routed.width) == (LayerKind.ROUTED, 2048)
    assert (shared.layer_kind, shared.width) == (LayerKind.SHARED, 2048)
    assert dense.tp_mode is shared.tp_mode is TensorParallelMode.ATTENTION_TP
    assert routed.tp_mode is TensorParallelMode.MOE_TP
    resolved_shared = profile.resolve_layer_contract(
        model, layer_id=3, layer_kind=LayerKind.SHARED,
        attention_tp_size=8, moe_tp_size=1, expert_parallel_size=16,
    )
    assert resolved_shared.tensor_parallel_size == 8


def test_native_glm53_attention_precision_and_rope(model: BaseModelConfig) -> None:
    native = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    assert model.num_q_heads == model.num_kv_heads == 64
    assert model.q_lora_rank == 2048
    assert model.kv_lora_rank == 512
    assert model.qk_nope_head_dim == 192
    assert model.qk_rope_head_dim == 64
    assert model.qk_head_dim == model.v_head_dim == 256
    assert model.get_qk_head_dim() == 256
    assert model.torch_dtype == "bfloat16"
    assert model.rope_theta == 8000000
    assert model.is_neox_style is False
    assert model.indexer_rope_interleave is True

    quant = model.quantization_config
    assert quant is not None
    assert quant.quant_method == "fp8"
    assert quant.activation_scheme == "dynamic"
    assert quant.is_checkpoint_fp8_serialized is True
    assert quant.weight_block_size == (128, 128)
    assert quant.ignored_layers == native["quantization_config"]["modules_to_not_convert"]
    assert quant.ignored_layers


def test_native_glm53_index_schedule_retains_producer_and_backbone_extent(
    model: BaseModelConfig,
) -> None:
    assert model.index_topk == 2048
    assert model.index_n_heads == 32
    assert model.index_head_dim == 128
    assert model.index_topk_freq == 4
    assert model.index_topk_pattern is None
    assert model.index_skip_topk_offset == 3

    specs = model.get_layer_attention_specs()
    assert len(specs) == 78
    assert all(isinstance(spec, LayerAttentionSpec) for spec in specs)
    assert [spec.global_layer_id for spec in specs] == list(range(78))
    full_ids = [spec.global_layer_id for spec in specs if spec.variant_id == "full_index"]
    assert full_ids == [0, 1, 2, *range(6, 75, 4)]
    assert sum(spec.variant_id == "shared_index" for spec in specs) == 57
    assert {spec.family_id for spec in specs} == {"dsa_attention"}
    assert specs[3].index_producer_layer_id == 2
    assert specs[77].index_producer_layer_id == 74
    assert all(specs[index].index_producer_layer_id == index for index in full_ids)


def test_glm53_profiling_preserves_runtime_layer_and_precision_contract(
    model: BaseModelConfig,
) -> None:
    profiling = ProfilingModelConfig.from_model_name(MODEL_NAME)
    assert profiling.get_model_architecture_profile().profile_id == "glm_moe_dsa"
    assert profiling.get_moe_layer_ids() == model.get_moe_layer_ids()
    assert profiling.get_layer_attention_specs() == model.get_layer_attention_specs()
    assert profiling.get_quant_signature() == model.get_quant_signature()
    assert profiling.rope_theta == model.rope_theta
    assert profiling.is_neox_style == model.is_neox_style
    assert profiling.indexer_rope_interleave is True
    assert bind_layer_attention(profiling, 77) == bind_layer_attention(model, 77)
    restored = ProfilingModelConfig(**profiling.to_dict())
    assert restored.get_layer_attention_specs() == model.get_layer_attention_specs()


@pytest.mark.parametrize(
    ("model_type", "expected_profile"),
    [("glm_moe_dsa", "glm_moe_dsa"), ("unit_model", "generic")],
)
def test_glm53_profile_uses_model_type_instead_of_model_name(
    model_type: str, expected_profile: str,
) -> None:
    config = SimpleNamespace(model_type=model_type, name=MODEL_NAME)
    assert get_model_architecture_profile(config).profile_id == expected_profile


@pytest.mark.parametrize("frequency", [0, -1, 1.5, True])
def test_glm53_rejects_invalid_index_frequency(
    model: BaseModelConfig, frequency: object,
) -> None:
    with pytest.raises(ValueError, match="index_topk_freq"):
        replace(model, index_topk_freq=frequency).get_layer_attention_specs()


def test_glm53_rejects_negative_index_offset(model: BaseModelConfig) -> None:
    with pytest.raises(ValueError, match="index_skip_topk_offset"):
        replace(model, index_skip_topk_offset=-1).get_layer_attention_specs()


def test_glm53_explicit_pattern_leaves_suffix_full(model: BaseModelConfig) -> None:
    specs = replace(model, index_topk_pattern="FSS").get_layer_attention_specs()
    assert [spec.variant_id for spec in specs[:4]] == [
        "full_index", "shared_index", "shared_index", "full_index"
    ]
    assert [spec.index_producer_layer_id for spec in specs[:4]] == [0, 0, 0, 3]
    assert all(spec.variant_id == "full_index" for spec in specs[3:])


def test_glm53_rejects_shared_index_without_prior_producer(model: BaseModelConfig) -> None:
    with pytest.raises(ValueError, match="[Pp]roducer"):
        replace(model, index_topk_pattern="S").get_layer_attention_specs()


def test_glm53_representation_enables_measured_execution(model: BaseModelConfig) -> None:
    spec = bind_layer_attention(model, 77)
    family = get_attention_family(spec.family_id)
    assert family.execution_enabled is True
    family.require_enabled_for_execution()
    assert model.get_runtime_head_size() == model.kv_lora_rank + model.qk_rope_head_dim


@pytest.mark.parametrize("model_name", ["llama3.3-70b", "Phi-tiny-MoE-instruct"])
def test_absent_index_metadata_preserves_existing_model_serialization(
    model_name: str,
) -> None:
    model = BaseModelConfig.create_from_name(model_name)
    runtime = dataclass_to_dict(model)
    profiling = ProfilingModelConfig.from_model_name(model_name).to_dict()
    assert "index_topk" not in runtime
    assert "index_topk" not in profiling
    assert "sliding_window" not in runtime
    assert "sliding_window" not in profiling
    assert runtime["share_q_dim"] is None
    assert profiling["share_q_dim"] is None


@pytest.mark.parametrize("model_name", [
    "meta-llama/Llama-2-7b-hf", "microsoft/phi-2", "Qwen/Qwen3-4B",
])
def test_unknown_embedding_tying_preserves_legacy_profiling_default(model_name: str) -> None:
    model = BaseModelConfig.create_from_name(model_name)
    assert model.tie_word_embeddings is None
    assert "tie_word_embeddings" not in dataclass_to_dict(model)
    assert ProfilingModelConfig.from_model_name(model_name).tie_word_embeddings is True


def test_explicit_untied_glm_metadata_survives_profiling_roundtrip(model: BaseModelConfig) -> None:
    assert model.tie_word_embeddings is False
    assert model.topk_method == "noaux_tc"
    profiling = ProfilingModelConfig.from_model_name(MODEL_NAME)
    assert profiling.tie_word_embeddings is False
    assert profiling.topk_method == "noaux_tc"
    assert profiling.to_dict()["topk_method"] == "noaux_tc"


def test_explicit_online_fp8_does_not_waive_block_serialization_guard() -> None:
    with pytest.raises(ValueError, match="is_checkpoint_fp8_serialized=True"):
        QuantizationConfig.from_dict({
            "quant_method": "fp8", "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "is_checkpoint_fp8_serialized": False,
        })

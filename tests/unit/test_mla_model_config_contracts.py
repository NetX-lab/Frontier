#!/usr/bin/env python3
"""Unit tests for DeepSeek-V2 MLA topology and runtime head semantics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.config.config import ReplicaConfig
from frontier.entities.replica import Replica
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig
from frontier.request_generator.synthetic_request_generator import (
    SyntheticRequestGeneratorConfig,
)
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ActivationType, NormType
from frontier.types import ClusterType


def _deepseek_v2_hf_config() -> dict[str, object]:
    return {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "num_hidden_layers": 60,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "hidden_size": 5120,
        "intermediate_size": 12288,
        "max_position_embeddings": 163840,
        "vocab_size": 102400,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "torch_dtype": "bfloat16",
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "rope_theta": 10000,
    }


def test_base_model_config_parses_first_class_mla_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_name = "deepseek-ai/DeepSeek-V2-MLA-Unit"
    config_dir = tmp_path / "data" / "config" / "models"
    config_dir.mkdir(parents=True)
    (config_dir / "deepseek-ai__DeepSeek-V2-MLA-Unit.json").write_text(
        json.dumps(_deepseek_v2_hf_config()),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = BaseModelConfig.create_from_name(model_name)

    assert config.model_type == "deepseek_v2"
    assert config.use_mla is True
    assert config.q_lora_rank == 1536
    assert config.kv_lora_rank == 512
    assert config.qk_nope_head_dim == 128
    assert config.qk_rope_head_dim == 64
    assert config.qk_head_dim == 192
    assert config.v_head_dim == 128
    assert config.get_runtime_num_kv_heads() == 1
    assert config.get_runtime_head_size() == 576
    assert config.get_qk_head_dim() == 192


def test_base_model_config_delegates_runtime_cache_helpers_to_family_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "deepseek-ai/DeepSeek-V2-MLA-Family-Resolver-Unit"
    config_dir = tmp_path / "data" / "config" / "models"
    config_dir.mkdir(parents=True)
    (config_dir / "deepseek-ai__DeepSeek-V2-MLA-Family-Resolver-Unit.json").write_text(
        json.dumps(_deepseek_v2_hf_config()),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = BaseModelConfig.create_from_name(model_name)
    family = config.get_attention_family()

    assert family.family_id == "latent_mla_attention"
    assert config.get_runtime_num_kv_heads() == family.resolve_runtime_num_kv_heads(
        config
    )
    assert config.get_runtime_head_size() == family.resolve_runtime_head_size(config)


def test_base_model_config_rebinds_attention_family_after_mutation() -> None:
    config = BaseModelConfig(
        num_layers=1,
        num_q_heads=128,
        num_kv_heads=128,
        embedding_dim=5120,
        mlp_hidden_dim=12288,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=102400,
        model_type="generic",
        use_mla=False,
    )

    assert config.get_attention_family().family_id == "dense_attention"

    config.use_mla = True
    config.kv_lora_rank = 512
    config.qk_nope_head_dim = 128
    config.qk_rope_head_dim = 64
    config.qk_head_dim = 192
    config.v_head_dim = 128

    assert config.get_attention_family().family_id == "latent_mla_attention"
    assert config.uses_mla() is True
    assert config.get_runtime_num_kv_heads() == 1
    assert config.get_runtime_head_size() == 576


def test_base_model_config_rejects_incomplete_mla_topology() -> None:
    with pytest.raises(ValueError, match="kv_lora_rank"):
        BaseModelConfig(
            num_layers=1,
            num_q_heads=128,
            num_kv_heads=128,
            embedding_dim=5120,
            mlp_hidden_dim=12288,
            max_position_embeddings=4096,
            use_gated_mlp=True,
            use_bias=False,
            use_qkv_bias=False,
            activation=ActivationType.SILU,
            norm=NormType.RMS_NORM,
            post_attn_norm=True,
            vocab_size=102400,
            model_type="deepseek_v2",
            use_mla=True,
            q_lora_rank=1536,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )


def test_profiling_model_config_uses_vllm_mla_runtime_cache_semantics() -> None:
    model_config = ModelConfig(
        name="deepseek-ai/DeepSeek-V2-MLA-Unit",
        num_layers=60,
        num_q_heads=128,
        num_kv_heads=128,
        embedding_dim=5120,
        mlp_hidden_dim=12288,
        max_position_embeddings=163840,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=102400,
        dtype="bfloat16",
        model_type="deepseek_v2",
        use_mla=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
    )
    parallel_config = ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=8)

    assert model_config.use_mla is True
    assert model_config.get_num_kv_heads(parallel_config) == 1
    assert model_config.get_head_size() == 576
    assert model_config.get_runtime_head_size() == 576
    assert model_config.get_qk_head_dim() == 192


def test_profiling_model_config_delegates_runtime_cache_helpers_to_family_spec() -> None:
    model_config = ModelConfig(
        name="deepseek-ai/DeepSeek-V2-MLA-Unit",
        num_layers=60,
        num_q_heads=128,
        num_kv_heads=128,
        embedding_dim=5120,
        mlp_hidden_dim=12288,
        max_position_embeddings=163840,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=102400,
        dtype="bfloat16",
        model_type="deepseek_v2",
        use_mla=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
    )
    parallel_config = ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=8)
    family = model_config.get_attention_family()

    assert family.family_id == "latent_mla_attention"
    assert model_config.get_runtime_num_kv_heads() == (
        family.resolve_runtime_num_kv_heads(model_config)
    )
    assert model_config.get_runtime_head_size() == (
        family.resolve_runtime_head_size(model_config)
    )
    assert model_config.get_num_kv_heads(parallel_config) == 1


def test_profiling_model_config_get_head_size_binds_family_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from frontier.attention.ops import AttentionMemoryLayout

    model_config = ModelConfig(
        name="deepseek-ai/DeepSeek-V2-MLA-Unit",
        num_layers=60,
        num_q_heads=128,
        num_kv_heads=128,
        embedding_dim=5120,
        mlp_hidden_dim=12288,
        max_position_embeddings=163840,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=102400,
        dtype="bfloat16",
        model_type="deepseek_v2",
        use_mla=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
    )
    bind_calls = 0

    class FakeFamily:
        memory_layout = AttentionMemoryLayout.LATENT_MLA

        @staticmethod
        def resolve_runtime_head_size(config: ModelConfig) -> int:
            return 576

    def fake_bind_attention_family(config: ModelConfig) -> SimpleNamespace:
        nonlocal bind_calls
        bind_calls += 1
        assert config is model_config
        return SimpleNamespace(family=FakeFamily())

    monkeypatch.setattr(
        "frontier.profiling.common.model_config.bind_attention_family",
        fake_bind_attention_family,
    )

    assert model_config.get_head_size() == 576
    assert bind_calls == 1


def test_profiling_model_config_rebinds_attention_family_after_mutation() -> None:
    model_config = ModelConfig(
        name="dense-unit",
        num_layers=1,
        num_q_heads=128,
        num_kv_heads=128,
        embedding_dim=5120,
        mlp_hidden_dim=12288,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=102400,
        dtype="bfloat16",
        model_type="generic",
        use_mla=False,
    )

    assert model_config.get_attention_family().family_id == "dense_attention"

    model_config.use_mla = True
    model_config.kv_lora_rank = 512
    model_config.qk_nope_head_dim = 128
    model_config.qk_rope_head_dim = 64
    model_config.qk_head_dim = 192
    model_config.v_head_dim = 128

    assert model_config.get_attention_family().family_id == "latent_mla_attention"
    assert model_config.get_runtime_num_kv_heads() == 1
    assert model_config.get_runtime_head_size() == 576
    assert model_config.get_qk_head_dim() == 192


def test_runtime_memory_planner_uses_mla_latent_kv_cache_page_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "deepseek-ai/DeepSeek-V2-MLA-Runtime-Unit"
    config_dir = tmp_path / "data" / "config" / "models"
    config_dir.mkdir(parents=True)
    (config_dir / "deepseek-ai__DeepSeek-V2-MLA-Runtime-Unit.json").write_text(
        json.dumps(_deepseek_v2_hf_config()),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    replica_config = ReplicaConfig(
        model_name=model_name,
        num_pipeline_stages=1,
        attn_tensor_parallel_size=8,
        attn_data_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )
    replica = Replica(
        replica_config=replica_config,
        generator_config=SyntheticRequestGeneratorConfig(),
        cluster_type=ClusterType.MONOLITHIC,
    )
    planner = MemoryPlanner(
        replica_config=replica_config,
        replica=replica,
        cluster_type=ClusterType.MONOLITHIC,
    )

    actual_page_bytes = planner._get_kv_cache_memory_per_layer_per_block(
        block_size=64
    )
    dense_kv_page_bytes = 2 * 2 * 64 * 16 * 40

    assert replica.attention_head_dim == 576
    assert replica.kv_heads_per_tensor_parallel_worker == 1
    assert actual_page_bytes == 64 * 1 * 576 * 2
    assert actual_page_bytes == 73728
    assert actual_page_bytes != dense_kv_page_bytes


def test_runtime_memory_planner_delegates_kv_layout_to_attention_memory_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "deepseek-ai/DeepSeek-V2-MLA-Adapter-Unit"
    config_dir = tmp_path / "data" / "config" / "models"
    config_dir.mkdir(parents=True)
    (config_dir / "deepseek-ai__DeepSeek-V2-MLA-Adapter-Unit.json").write_text(
        json.dumps(_deepseek_v2_hf_config()),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    replica_config = ReplicaConfig(
        model_name=model_name,
        num_pipeline_stages=1,
        attn_tensor_parallel_size=8,
        attn_data_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )
    replica = Replica(
        replica_config=replica_config,
        generator_config=SyntheticRequestGeneratorConfig(),
        cluster_type=ClusterType.MONOLITHIC,
    )
    planner = MemoryPlanner(
        replica_config=replica_config,
        replica=replica,
        cluster_type=ClusterType.MONOLITHIC,
    )
    calls: list[tuple[str, int, int]] = []

    class _FakeLayout:
        elements_per_token_per_worker = 777

    def _fake_layout(family, *, runtime_num_kv_heads_per_worker, runtime_head_size):
        calls.append(
            (
                family.family_id,
                runtime_num_kv_heads_per_worker,
                runtime_head_size,
            )
        )
        return _FakeLayout()

    monkeypatch.setattr(
        "frontier.scheduler.utils.memory_planner.get_attention_runtime_kv_layout",
        _fake_layout,
    )

    assert planner._get_kv_cache_elements_per_token_per_worker() == 777
    assert calls == [("latent_mla_attention", 1, 576)]

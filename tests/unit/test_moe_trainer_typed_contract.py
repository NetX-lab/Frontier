"""Regression coverage for profile-owned routed MoE trainer dimensions."""

from __future__ import annotations

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.model_architectures import ModelArchitectureProfile
import frontier.training.moe_trainer as moe_trainer_module


def _build_typed_trainer(
    config: BaseModelConfig,
    *,
    tensor_parallel_size: int,
    expert_parallel_size: int,
) -> moe_trainer_module.MoETrainer:
    """Build a routed trainer with the profile-owned contract transport."""

    profile = config.get_model_architecture_profile()
    contract = profile.resolve_layer_contract(
        config,
        operator_name="moe_grouped_gemm",
        moe_tp_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
    )
    return moe_trainer_module.MoETrainer(
        dataset_path="synthetic-moe.csv",
        output_dir="unused-moe-trainer-cache",
        num_experts=config.num_experts,
        router_topk=config.num_experts_per_tok,
        hidden_dim=config.embedding_dim,
        expert_hidden_dim=contract.effective_ffn_width,
        moe_tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
        model_name=config.get_name(),
        device="h200",
        model_config=config,
        layer_contract=contract,
    )


def test_factory_uses_profile_owned_routed_width(monkeypatch, tmp_path) -> None:
    """The convenience factory must ignore a stale model-wide MLP width."""

    model_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    model_config.mlp_hidden_dim = 9999

    monkeypatch.setattr(
        BaseModelConfig,
        "create_from_name",
        classmethod(lambda _cls, _name: model_config),
    )

    captured: dict[str, object] = {}

    class StubMoETrainer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(moe_trainer_module, "MoETrainer", StubMoETrainer)

    result = moe_trainer_module.create_moe_trainer_from_model_config(
        dataset_path=str(tmp_path / "moe.csv"),
        output_dir=str(tmp_path / "models"),
        model_name="step3-moe-noquant",
        moe_tensor_parallel_size=1,
        expert_parallel_size=1,
    )

    assert isinstance(result, StubMoETrainer)
    assert captured["expert_hidden_dim"] == 5120
    assert captured["expert_hidden_dim"] == model_config.routed_mlp_hidden_dim


def test_factory_passes_profile_owned_contract_to_standalone_trainer(
    monkeypatch, tmp_path
) -> None:
    """The factory must carry the resolved config and routed contract into caching."""

    model_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        BaseModelConfig,
        "create_from_name",
        classmethod(lambda _cls, _name: model_config),
    )

    class StubMoETrainer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(moe_trainer_module, "MoETrainer", StubMoETrainer)

    moe_trainer_module.create_moe_trainer_from_model_config(
        dataset_path=str(tmp_path / "moe.csv"),
        output_dir=str(tmp_path / "models"),
        model_name="step3-moe-noquant",
        moe_tensor_parallel_size=8,
        expert_parallel_size=8,
    )

    assert captured["model_config"] is model_config
    contract = captured["layer_contract"]
    assert contract.layer_kind.value == "routed"
    assert contract.effective_ffn_width == 5120
    assert contract.tensor_parallel_size == 8
    assert contract.expert_parallel_size == 8


def test_standalone_moe_cache_identity_includes_routed_contract() -> None:
    """Routed width, TP/EP, and the MoE layer map isolate standalone caches."""

    first_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    second_config = BaseModelConfig.create_from_name("step3-moe-noquant")
    second_config.routed_mlp_hidden_dim = 18432
    second_config.moe_layers_enum = ",".join(str(layer_id) for layer_id in range(56))

    first = _build_typed_trainer(
        first_config,
        tensor_parallel_size=1,
        expert_parallel_size=1,
    )
    second = _build_typed_trainer(
        second_config,
        tensor_parallel_size=8,
        expert_parallel_size=8,
    )

    frame = __import__("pandas").DataFrame({"num_tokens": [1], "value": [2.0]})
    first_identity = first._get_model_hash_identity("moe_grouped_gemm")
    second_identity = second._get_model_hash_identity("moe_grouped_gemm")

    assert first_identity is not None
    assert second_identity is not None
    assert first_identity != second_identity
    assert first._get_model_hash("moe_grouped_gemm", frame) != second._get_model_hash(
        "moe_grouped_gemm", frame
    )

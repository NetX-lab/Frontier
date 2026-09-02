from __future__ import annotations

from types import SimpleNamespace

from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
    _serialize_layer_contract_identity,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.types import ClusterType, MeasurementType


def _step3_config() -> SimpleNamespace:
    return SimpleNamespace(
        is_moe=True,
        num_layers=61,
        num_kv_heads=8,
        num_experts=48,
        moe_layers_enum=",".join(str(layer_id) for layer_id in range(4, 60)),
        mlp_hidden_dim=5120,
        dense_mlp_hidden_dim=18432,
        routed_mlp_hidden_dim=5120,
        share_expert_dim=5120,
        supports_share_expert=lambda: True,
        get_model_architecture_profile=ModelArchitectureProfile.step3_text,
    )


def _manager_with_decode_attn_models() -> ExecutionTimePredictionModelManager:
    config = _step3_config()
    profile = ModelArchitectureProfile.step3_text()
    dense_tp8 = profile.resolve_layer_contract(
        config,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    dense_tp4 = profile.resolve_layer_contract(
        config,
        operator_name="mlp_up_proj",
        attention_tp_size=4,
        moe_tp_size=1,
        expert_parallel_size=8,
    )

    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._all_dummy_mode = False
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._trained_models_eager = {"attn_pre_proj": object()}
    manager._trained_models_kernel_only = {}
    manager._trained_models_eager_by_contract = {
        ("mlp_up_proj", _serialize_layer_contract_identity(dense_tp8)): object(),
        ("mlp_up_proj", _serialize_layer_contract_identity(dense_tp4)): object(),
    }
    manager._trained_models_kernel_only_by_contract = {}
    manager._models_by_precision_eager = {}
    manager._models_by_precision_kernel_only = {}
    manager._models_by_precision_eager_by_contract = {}
    manager._models_by_precision_kernel_only_by_contract = {}
    manager._cluster_configs = {
        ClusterType.DECODE_ATTN: SimpleNamespace(
            replica_config=SimpleNamespace(
                model_config=config,
                attn_tensor_parallel_size=8,
            )
        )
    }
    manager._is_kernel_only_measurement_enabled_for_cluster = lambda _cluster: False
    return manager


def test_decode_attn_view_excludes_ambiguous_ffn_typed_variants(monkeypatch) -> None:
    """Attention-only views must not enumerate FFN typed model variants."""

    import frontier.execution_time_predictor.shared_prediction_model_manager as module

    monkeypatch.setattr(module.global_vars, "get_sys_arch", lambda: "co-location")
    manager = _manager_with_decode_attn_models()

    models = manager.get_models_for_cluster(ClusterType.DECODE_ATTN)

    assert set(models["eager"]) == {"attn_pre_proj"}
    assert models["kernel_only"] == {}


def test_decode_attn_zero_domain_skips_ffn_signature_resolution() -> None:
    """Attention-only roles must not resolve or hash an FFN domain."""

    config = _step3_config()
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=0,
        moe_tensor_parallel_size=0,
        moe_expert_parallel_size=0,
    )
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )

    assert manager._resolve_ffn_layer_contracts(
        ClusterType.DECODE_ATTN,
        replica_config,
        is_moe_model=True,
    ) == ()
    assert manager._get_ffn_contract_signature(
        ClusterType.DECODE_ATTN,
        replica_config,
        is_moe_model=True,
    ) == "none"

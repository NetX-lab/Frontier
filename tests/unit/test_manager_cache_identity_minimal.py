"""Direct regressions for the selected typed-domain model-cache identity."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
    _serialize_layer_contract_identity,
    _serialize_selected_layer_cache_identity,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.types import MeasurementType


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
    )


def _hash_config() -> SimpleNamespace:
    return SimpleNamespace(
        linear_op_input_file="linear.csv",
        atten_input_file="attention.csv",
        all_reduce_input_file="all_reduce.csv",
        send_recv_input_file="send_recv.csv",
        moe_input_file="moe.csv",
        linear_op_kernel_only_input_file="linear_kernel.csv",
        atten_kernel_only_input_file="attention_kernel.csv",
        moe_kernel_only_input_file="moe_kernel.csv",
        cpu_overhead_input_file="cpu.csv",
        cpu_overhead_kernel_only_input_file="cpu_kernel.csv",
        kv_cache_prediction_granularity=16,
        prediction_max_prefill_chunk_size=1024,
        prediction_max_batch_size=64,
        prediction_max_tokens_per_request=4096,
        attention_decode_batching_overhead_fraction=0.0,
        attention_prefill_batching_overhead_fraction=0.0,
        attn_pre_proj_calibration_scale=1.0,
        prefill_phase_attn_pre_proj_calibration_scale=1.0,
        attn_post_proj_calibration_scale=1.0,
        prefill_phase_attn_post_proj_calibration_scale=1.0,
        attn_decode_calibration_scale=1.0,
        attn_decode_in_mixed_calibration_scale=1.0,
        late_decode_attn_decode_calibration_scale=1.0,
        attn_kv_cache_save_calibration_scale=1.0,
        prefill_phase_attn_kv_cache_save_calibration_scale=1.0,
        mlp_up_proj_calibration_scale=1.0,
        prefill_phase_mlp_up_proj_calibration_scale=1.0,
        mlp_down_proj_calibration_scale=1.0,
        decode_phase_mlp_down_proj_calibration_scale=1.0,
        nccl_cpu_launch_overhead_ms=0.0,
        nccl_cpu_skew_overhead_per_device_ms=0.0,
    )


def _dense_contract(*, layer_id: int | None = None):
    config = _step3_config()
    return ModelArchitectureProfile.step3_text().resolve_layer_contract(
        config,
        layer_id=layer_id,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )


def _cache_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_tokens": [1],
            "target": [1.0],
            "profiling_precision": ["FP16"],
            "measurement_type": [MeasurementType.CUDA_EVENT.value],
        }
    )


def _initialized_manager(tmp_path) -> ExecutionTimePredictionModelManager:
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._cache_dir = str(tmp_path)
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._all_dummy_mode = False
    manager._trained_models_eager = {}
    manager._trained_models_kernel_only = {}
    manager._models_by_precision_eager = {}
    manager._models_by_precision_kernel_only = {}
    manager._trained_models_eager_by_contract = {}
    manager._trained_models_kernel_only_by_contract = {}
    manager._models_by_precision_eager_by_contract = {}
    manager._models_by_precision_kernel_only_by_contract = {}
    manager._cluster_configs = {}
    return manager


def test_selected_cache_hash_reuses_equivalent_layer_ids() -> None:
    """Physical layer identity must not split one semantic typed-domain cache."""

    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    config = _hash_config()
    frame = pd.DataFrame({"num_tokens": [1], "value": [2.0]})
    layer_zero = _dense_contract(layer_id=0)
    layer_one = _dense_contract(layer_id=1)

    hash_zero = manager._get_model_hash(
        "mlp_up_proj", frame, config, "FP16", MeasurementType.CUDA_EVENT, layer_zero
    )
    hash_one = manager._get_model_hash(
        "mlp_up_proj", frame, config, "FP16", MeasurementType.CUDA_EVENT, layer_one
    )

    assert hash_zero == hash_one
    assert _serialize_selected_layer_cache_identity(layer_zero) == (
        _serialize_selected_layer_cache_identity(layer_one)
    )


def test_new_typed_model_keeps_only_selected_cache_identity(tmp_path) -> None:
    """New model metadata does not record a physical layer occurrence."""

    manager = _initialized_manager(tmp_path)
    contract = _dense_contract(layer_id=0)
    model = SimpleNamespace()

    selected = manager._model_contract_identity(model, contract)

    assert selected == _serialize_selected_layer_cache_identity(contract)
    assert model._frontier_layer_cache_identity == selected
    assert not hasattr(model, "_frontier_layer_contract_identity")


def test_legacy_read_does_not_mutate_typed_registry(tmp_path) -> None:
    """Legacy compatibility lookup remains read-only for canonical storage."""

    manager = _initialized_manager(tmp_path)
    legacy_model = SimpleNamespace()
    manager._trained_models_eager["legacy_op"] = legacy_model

    assert manager.get_model("legacy_op") is legacy_model
    assert manager._trained_models_eager_by_contract == {}


@pytest.mark.parametrize(
    "variant",
    [
        lambda contract: replace(contract, effective_ffn_width=18440),
        lambda contract: replace(contract, operator_family_id="share_expert"),
        lambda contract: replace(contract, tensor_parallel_size=4),
        lambda contract: replace(contract, expert_parallel_size=2),
        lambda contract: replace(
            contract,
            selected_padded_ffn_width=contract.selected_padded_ffn_width + 8,
        ),
    ],
)
def test_selected_cache_hash_distinguishes_typed_domain_fields(variant) -> None:
    """Width, family, TP, EP, and padded width each affect cache identity."""

    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    config = _hash_config()
    frame = pd.DataFrame({"num_tokens": [1], "value": [2.0]})
    base = _dense_contract()
    changed = variant(base)

    base_hash = manager._get_model_hash(
        "mlp_up_proj", frame, config, "FP16", MeasurementType.CUDA_EVENT, base
    )
    changed_hash = manager._get_model_hash(
        "mlp_up_proj", frame, config, "FP16", MeasurementType.CUDA_EVENT, changed
    )

    assert changed_hash != base_hash


def test_selected_cache_identity_ignores_producer_envelopes() -> None:
    """A producer's wider family/TP envelope must not split one selected domain."""

    base = _dense_contract()
    envelope_variant = replace(
        base,
        operator_family_ids=(base.operator_family_id, "unused_sibling"),
        tensor_parallel_sizes=(1, 2, 4, 8),
    )

    assert _serialize_selected_layer_cache_identity(base) == (
        _serialize_selected_layer_cache_identity(envelope_variant)
    )


def test_cache_pickle_reuses_selected_domain_across_layer_ids(tmp_path) -> None:
    """A current typed pickle must load for an equivalent physical layer."""

    manager = _initialized_manager(tmp_path)
    config = _hash_config()
    frame = _cache_dataframe()
    layer_zero = _dense_contract(layer_id=0)
    layer_one = _dense_contract(layer_id=1)
    model_hash = manager._get_model_hash(
        "mlp_up_proj", frame, config, "FP16", MeasurementType.CUDA_EVENT, layer_zero
    )
    cached = SimpleNamespace(
        _frontier_layer_contract_identity=_serialize_layer_contract_identity(layer_zero),
        _frontier_layer_cache_identity=_serialize_selected_layer_cache_identity(
            layer_zero
        ),
    )
    manager._store_model_in_cache("mlp_up_proj", model_hash, cached)

    loaded = manager._train_single_model(
        model_name="mlp_up_proj",
        df=frame,
        feature_cols=["num_tokens"],
        target_col="target",
        execution_time_predictor_config=config,
        layer_contract=layer_one,
    )

    assert loaded._frontier_layer_cache_identity == (
        _serialize_selected_layer_cache_identity(layer_one)
    )
    # The occurrence-specific identity is provenance from the producing layer;
    # cache reuse must not rewrite it to the requesting layer.
    assert loaded._frontier_layer_contract_identity == (
        _serialize_layer_contract_identity(layer_zero)
    )


def test_legacy_typed_pickle_without_selected_identity_stays_exact(tmp_path) -> None:
    """An old typed pickle cannot be guessed as another physical layer."""

    manager = _initialized_manager(tmp_path)
    config = _hash_config()
    frame = _cache_dataframe()
    layer_zero = _dense_contract(layer_id=0)
    layer_one = _dense_contract(layer_id=1)
    model_hash = manager._get_model_hash(
        "mlp_up_proj", frame, config, "FP16", MeasurementType.CUDA_EVENT, layer_zero
    )
    legacy_cached = SimpleNamespace(
        _frontier_layer_contract_identity=_serialize_layer_contract_identity(layer_zero)
    )
    manager._store_model_in_cache("mlp_up_proj", model_hash, legacy_cached)

    with pytest.raises(ValueError, match="layer contract identity mismatch"):
        manager._train_single_model(
            model_name="mlp_up_proj",
            df=frame,
            feature_cols=["num_tokens"],
            target_col="target",
            execution_time_predictor_config=config,
            layer_contract=layer_one,
        )


def test_explicitly_empty_selected_identity_is_rejected(tmp_path) -> None:
    """A present but empty selected identity is malformed, not legacy metadata."""

    manager = _initialized_manager(tmp_path)
    config = _hash_config()
    frame = _cache_dataframe()
    layer_zero = _dense_contract(layer_id=0)
    model_hash = manager._get_model_hash(
        "mlp_up_proj", frame, config, "FP16", MeasurementType.CUDA_EVENT, layer_zero
    )
    malformed_cached = SimpleNamespace(
        _frontier_layer_contract_identity=_serialize_layer_contract_identity(layer_zero),
        _frontier_layer_cache_identity=None,
    )
    manager._store_model_in_cache("mlp_up_proj", model_hash, malformed_cached)

    with pytest.raises(ValueError, match="selected layer cache identity"):
        manager._train_single_model(
            model_name="mlp_up_proj",
            df=frame,
            feature_cols=["num_tokens"],
            target_col="target",
            execution_time_predictor_config=config,
            layer_contract=layer_zero,
        )

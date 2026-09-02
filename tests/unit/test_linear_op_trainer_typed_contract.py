"""Regression coverage for profile-owned typed linear-op training contracts."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.model_architectures import LayerKind
from frontier.operators.spec import TensorParallelMode
from frontier.operators.typed_contracts import serialize_typed_operator_contracts
from frontier.training.linear_op_trainer import LinearOpTrainer


def _trainer(
    model_name: str,
    *,
    tensor_parallel_size: int = 8,
    output_dir: str = "unused-linear-trainer-cache",
) -> LinearOpTrainer:
    """Build a trainer shell without invoking the training estimator."""

    trainer = object.__new__(LinearOpTrainer)
    trainer.dataset_path = "synthetic-linear-op.csv"
    trainer.model_name = model_name
    trainer.device = "h200"
    trainer.tensor_parallel_size = tensor_parallel_size
    trainer.model_config = BaseModelConfig.create_from_name(model_name)
    trainer.is_moe = bool(trainer.model_config.is_moe)
    trainer._has_target_embedded_mtp_ops = False
    return trainer


def test_standalone_trainer_cache_identity_includes_typed_contract(tmp_path) -> None:
    """Different mixed-layer widths must not reuse a standalone cache entry."""

    first = LinearOpTrainer(
        dataset_path="synthetic-linear-op.csv",
        output_dir=str(tmp_path / "first"),
        model_name="step3-moe-noquant",
        device="h200",
        tensor_parallel_size=8,
    )
    second = LinearOpTrainer(
        dataset_path="synthetic-linear-op.csv",
        output_dir=str(tmp_path / "second"),
        model_name="step3-moe-noquant",
        device="h200",
        tensor_parallel_size=8,
    )
    second.model_config.dense_mlp_hidden_dim = 12288
    frame = pd.DataFrame({"num_tokens": [1], "value": [2.0]})

    assert first._get_model_hash("mlp_up_proj", frame) != second._get_model_hash(
        "mlp_up_proj", frame
    )


def test_standalone_trainer_cache_identity_includes_moe_layer_map(tmp_path) -> None:
    """Changing only MoE layer placement must invalidate training caches."""

    first = LinearOpTrainer(
        dataset_path="synthetic-linear-op.csv",
        output_dir=str(tmp_path / "first"),
        model_name="step3-moe-noquant",
        device="h200",
        tensor_parallel_size=8,
    )
    second = LinearOpTrainer(
        dataset_path="synthetic-linear-op.csv",
        output_dir=str(tmp_path / "second"),
        model_name="step3-moe-noquant",
        device="h200",
        tensor_parallel_size=8,
    )
    second.model_config.moe_layers_enum = ",".join(
        str(layer_id) for layer_id in range(0, 56)
    )
    frame = pd.DataFrame({"num_tokens": [1], "value": [2.0]})

    assert first._get_model_hash("mlp_up_proj", frame) != second._get_model_hash(
        "mlp_up_proj", frame
    )


def _contract_metadata(
    config: BaseModelConfig,
    operator_name: str,
    *,
    tensor_parallel_size: int = 8,
    width: int | None = None,
) -> dict[str, object]:
    profile = config.get_model_architecture_profile()
    contract = profile.resolve_layer_contract(
        config,
        operator_name=operator_name,
        attention_tp_size=tensor_parallel_size,
        ffn_tp_size=tensor_parallel_size,
        moe_tp_size=1,
        expert_parallel_size=8 if config.is_moe else None,
    )
    return {
        "profile_id": contract.profile_id,
        "operator_family_id": contract.operator_family_id,
        "operator_family_ids": [contract.operator_family_id],
        "layer_kind": contract.layer_kind.value,
        "dimension_source": contract.dimension_source.value,
        "effective_ffn_width": (
            contract.effective_ffn_width if width is None else width
        ),
        "tensor_parallel_mode": contract.tensor_parallel_mode.value,
        "expert_parallel_mode": contract.expert_parallel_mode.value,
        "selected_expert_parallel_size": contract.expert_parallel_size,
        "tensor_parallel_sizes": [contract.tensor_parallel_size],
        "selected_tensor_parallel_size": contract.tensor_parallel_size,
        "selected_padded_ffn_width": (
            contract.effective_ffn_width if width is None else width
        ),
    }


def _typed_row(
    config: BaseModelConfig,
    operator_name: str,
    *,
    target_value: float,
    width: int | None = None,
    tensor_parallel_size: int = 8,
) -> dict[str, object]:
    metadata = _contract_metadata(
        config,
        operator_name,
        tensor_parallel_size=tensor_parallel_size,
        width=width,
    )
    return {
        "num_tokens": 1,
        "num_tensor_parallel_workers": tensor_parallel_size,
        # Keep the legacy scalar deliberately identical across typed rows.
        "n_expanded_embd": 5120,
        "time_stats.%s.median" % operator_name: target_value,
        "typed_operator_contracts": serialize_typed_operator_contracts(
            {operator_name: metadata}
        ),
    }


def test_step3_mixed_trainer_discovers_dense_and_shared_linear_models() -> None:
    trainer = _trainer("step3-moe-noquant")

    names = trainer._get_model_names()

    assert {"mlp_up_proj", "mlp_down_proj", "mlp_act"}.issubset(names)
    assert {
        "share_expert_up_proj",
        "share_expert_down_proj",
        "share_expert_act",
    }.issubset(names)
    assert not {"moe_grouped_gemm", "moe_gating_linear"}.intersection(names)


def test_pure_moe_trainer_keeps_routed_rows_in_moe_trainer() -> None:
    trainer = _trainer("mixtral_8x7b_moe")

    names = trainer._get_model_names()

    assert not {"mlp_up_proj", "mlp_down_proj", "mlp_act"}.intersection(names)
    assert not {
        "share_expert_up_proj",
        "share_expert_down_proj",
        "share_expert_act",
    }.intersection(names)


def test_step3_linear_contracts_use_profile_owned_attention_tp() -> None:
    trainer = _trainer("step3-moe-noquant")

    dense = trainer._get_training_layer_contract("mlp_up_proj")
    shared = trainer._get_training_layer_contract("share_expert_up_proj")

    assert dense.layer_kind is LayerKind.DENSE
    assert dense.effective_ffn_width == 18432
    assert dense.tensor_parallel_mode is TensorParallelMode.ATTENTION_TP
    assert dense.tensor_parallel_size == 8
    assert shared.layer_kind is LayerKind.SHARED
    assert shared.effective_ffn_width == 5120
    assert shared.tensor_parallel_mode is TensorParallelMode.ATTENTION_TP
    assert shared.tensor_parallel_size == 8


def test_linear_trainer_surfaces_registry_binding_failure(monkeypatch) -> None:
    """A registry failure must not be downgraded to the generic TP resolver."""

    import frontier.training.linear_op_trainer as trainer_module

    trainer = _trainer("step3-moe-noquant")

    def _raise_registry_failure(*_args, **_kwargs):
        raise ValueError("registry collision")

    monkeypatch.setattr(
        trainer_module,
        "bind_operator_query",
        _raise_registry_failure,
    )

    with pytest.raises(ValueError, match="registry collision"):
        trainer._get_training_layer_contract("mlp_up_proj")


def test_mixed_typed_rows_are_isolated_by_family_and_effective_width() -> None:
    trainer = _trainer("step3-moe-noquant")
    config = trainer.model_config
    frame = pd.DataFrame(
        [
            _typed_row(
                config,
                "mlp_up_proj",
                target_value=1.0,
                width=18432,
            ),
            _typed_row(
                config,
                "mlp_up_proj",
                target_value=2.0,
                width=5120,
            ),
            _typed_row(
                config,
                "share_expert_up_proj",
                target_value=3.0,
                width=5120,
            ),
            _typed_row(
                config,
                "share_expert_up_proj",
                target_value=4.0,
                width=18432,
            ),
        ]
    )

    dense = trainer._get_training_df_for_model(
        df=frame,
        model_name="mlp_up_proj",
        feature_cols=["num_tokens"],
        target_col="time_stats.mlp_up_proj.median",
    )
    shared = trainer._get_training_df_for_model(
        df=frame,
        model_name="share_expert_up_proj",
        feature_cols=["num_tokens"],
        target_col="time_stats.share_expert_up_proj.median",
    )

    assert dense["time_stats.mlp_up_proj.median"].tolist() == [1.0]
    assert shared["time_stats.share_expert_up_proj.median"].tolist() == [3.0]


def test_mixed_typed_row_rejects_missing_schema_field() -> None:
    """Training admission rejects a parsed row that lacks a required field."""

    trainer = _trainer("step3-moe-noquant")
    contract = trainer.model_config.get_model_architecture_profile().resolve_layer_contract(
        trainer.model_config,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        ffn_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    metadata = {
        **contract.typed_metadata_identity(),
        "operator_family_ids": [contract.operator_family_id],
        "tensor_parallel_sizes": [8],
        "selected_padded_ffn_width": contract.effective_ffn_width,
    }
    metadata.pop("tensor_parallel_sizes")
    frame = pd.DataFrame(
        {
            "num_tokens": [1],
            "num_tensor_parallel_workers": [8],
            "n_expanded_embd": [5120],
            "time_stats.mlp_up_proj.median": [1.0],
            "typed_operator_contracts": [
                serialize_typed_operator_contracts({"mlp_up_proj": metadata})
            ],
        }
    )

    with pytest.raises(ValueError, match="invalid typed operator metadata"):
        trainer._get_training_df_for_model(
            df=frame,
            model_name="mlp_up_proj",
            feature_cols=["num_tokens"],
            target_col="time_stats.mlp_up_proj.median",
        )


def test_mixed_typed_contract_is_required_when_metadata_column_is_absent() -> None:
    trainer = _trainer("step3-moe-noquant")
    frame = pd.DataFrame(
        {
            "num_tokens": [1],
            "num_tensor_parallel_workers": [8],
            "n_expanded_embd": [5120],
            "time_stats.mlp_up_proj.median": [1.0],
        }
    )

    with pytest.raises(ValueError, match="typed_operator_contracts"):
        trainer._get_training_df_for_model(
            df=frame,
            model_name="mlp_up_proj",
            feature_cols=["num_tokens"],
            target_col="time_stats.mlp_up_proj.median",
        )


def test_typed_width_mismatch_reports_available_widths() -> None:
    """A typed coverage miss must raise a contract error, not a NameError."""

    trainer = _trainer("step3-moe-noquant")
    frame = pd.DataFrame(
        [
            _typed_row(
                trainer.model_config,
                "mlp_up_proj",
                target_value=1.0,
                width=5120,
            )
        ]
    )

    with pytest.raises(ValueError, match=r"available_widths=\[5120\]"):
        trainer._get_training_df_for_model(
            df=frame,
            model_name="mlp_up_proj",
            feature_cols=["num_tokens"],
            target_col="time_stats.mlp_up_proj.median",
        )


def test_mixed_dataset_verification_requires_active_shared_columns() -> None:
    trainer = _trainer("step3-moe-noquant")
    config = trainer.model_config
    dense_metadata = _contract_metadata(config, "mlp_up_proj")
    frame = pd.DataFrame(
        {
            "num_tokens": [1],
            "time_stats.emb.median": [1.0],
            "time_stats.input_layernorm.median": [1.0],
            "time_stats.post_attention_layernorm.median": [1.0],
            "time_stats.mlp_up_proj.median": [1.0],
            "time_stats.mlp_down_proj.median": [1.0],
            "time_stats.mlp_act.median": [1.0],
            "typed_operator_contracts": [
                json.dumps({"mlp_up_proj": dense_metadata})
            ],
        }
    )

    with pytest.raises(ValueError, match="share_expert_up_proj"):
        trainer._verify_dataset_columns(frame)


def test_pure_dense_legacy_rows_keep_scalar_width_fallback() -> None:
    trainer = _trainer("llama3.1-8b")
    trainer.tensor_parallel_size = 1
    frame = pd.DataFrame(
        {
            "num_tokens": [1],
            "num_tensor_parallel_workers": [1],
            "n_expanded_embd": [14336],
            "time_stats.mlp_up_proj.median": [1.0],
        }
    )

    filtered = trainer._get_training_df_for_model(
        df=frame,
        model_name="mlp_up_proj",
        feature_cols=["num_tokens"],
        target_col="time_stats.mlp_up_proj.median",
    )

    assert len(filtered) == 1
    assert filtered["n_expanded_embd"].tolist() == [14336]

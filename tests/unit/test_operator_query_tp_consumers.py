from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from frontier.execution_time_predictor.attention_tp_policy import (
    resolve_effective_attention_tp_size,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.operators import binding as operator_binding
from frontier.operators.spec import TensorParallelMode
import frontier.spec_decode.mtp_registry as mtp_registry
from frontier.training.attention_trainer import AttentionTrainer
from frontier.training.linear_op_trainer import LinearOpTrainer
from frontier.types import ClusterType


class _ConcreteSklearnExecutionTimePredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise NotImplementedError


def _model_config(profile: ModelArchitectureProfile | None = None) -> Any:
    selected_profile = profile or ModelArchitectureProfile.generic()
    return SimpleNamespace(
        num_kv_heads=8,
        is_moe=False,
        get_model_architecture_profile=lambda: selected_profile,
    )


def _predictor(*, profile: ModelArchitectureProfile | None = None):
    predictor = cast(Any, object.__new__(_ConcreteSklearnExecutionTimePredictor))
    predictor._model_config = _model_config(profile)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=4,
        speculative_decoding_config=None,
    )
    return predictor


def test_operator_query_tp_mode_uses_exact_registry_and_profile_membership() -> None:
    step2_profile = ModelArchitectureProfile.step2_mini()
    resolver = getattr(operator_binding, "resolve_operator_query_tp_mode", None)
    assert callable(resolver)

    assert (
        resolver("mlp_up_proj")
        is TensorParallelMode.FFN_TP
    )
    assert (
        resolver("input_layernorm")
        is TensorParallelMode.REPLICATED
    )
    assert (
        resolver(
            "attn_inter_norm",
            architecture_profile=step2_profile,
        )
        is TensorParallelMode.ATTENTION_TP
    )
    assert (
        resolver(
            "attn_pre_proj_qkv",
            architecture_profile=ModelArchitectureProfile.step3_text(),
        )
        is TensorParallelMode.REPLICATED
    )

    with pytest.raises(ValueError, match="Unsupported operator query"):
        resolver(
            "attn_inter_norm",
            architecture_profile=ModelArchitectureProfile.generic(),
        )
    with pytest.raises(ValueError, match="Unsupported operator query"):
        resolver("attn_not_declared")


def test_operator_query_tp_mode_scopes_many_to_one_memory_alias() -> None:
    resolver = getattr(operator_binding, "resolve_operator_query_tp_mode", None)
    assert callable(resolver)
    assert resolver("add") is TensorParallelMode.REPLICATED
    assert resolver("add", family_id="memory") is TensorParallelMode.REPLICATED


def test_predictor_rejects_undeclared_attention_name_before_tp_policy() -> None:
    predictor = _predictor()

    with pytest.raises(ValueError, match="Unsupported linear op"):
        predictor._get_linear_op_tp_key("attn_not_declared")


def test_predictor_routes_declared_architecture_attention_name() -> None:
    predictor = _predictor(profile=ModelArchitectureProfile.step2_mini())

    assert predictor._get_linear_op_tp_key("attn_inter_norm") == 2


def test_shared_manager_rejects_undeclared_attention_name() -> None:
    manager = object.__new__(ExecutionTimePredictionModelManager)
    replica_config = SimpleNamespace(
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=4,
        model_config=_model_config(),
        speculative_decoding_config=None,
    )

    with pytest.raises(ValueError, match="Unsupported linear op"):
        manager._get_linear_op_tp_key(
            "attn_not_declared",
            ClusterType.MONOLITHIC,
            replica_config,
            is_moe_model=False,
        )


def test_linear_trainer_rejects_undeclared_mlp_name() -> None:
    trainer = object.__new__(LinearOpTrainer)
    trainer.model_config = _model_config()
    trainer.tensor_parallel_size = 2
    trainer._has_target_embedded_mtp_ops = False

    with pytest.raises(ValueError, match="Unsupported linear op"):
        trainer._get_training_tp_key("mlp_not_declared")


def test_attention_trainer_rejects_undeclared_attention_name() -> None:
    trainer = object.__new__(AttentionTrainer)
    trainer.model_config = _model_config()
    trainer.tensor_parallel_size = 2

    with pytest.raises(ValueError, match="Unsupported compute model"):
        trainer._get_compute_tp_key("attn_not_declared")


def test_target_embedded_mtp_registry_extension_updates_all_tp_consumers(
    monkeypatch,
) -> None:
    probe_name = "mtp_registry_extension_probe"
    monkeypatch.setattr(
        mtp_registry,
        "_TARGET_EMBEDDED_MTP_LINEAR_OPS",
        mtp_registry.get_target_embedded_mtp_linear_ops() + (probe_name,),
    )

    predictor = _predictor()
    manager = object.__new__(ExecutionTimePredictionModelManager)
    replica_config = SimpleNamespace(
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=4,
        model_config=_model_config(),
        speculative_decoding_config=None,
    )
    trainer = object.__new__(LinearOpTrainer)
    trainer.model_config = _model_config()
    trainer.tensor_parallel_size = 2
    trainer._has_target_embedded_mtp_ops = True

    assert predictor._get_linear_op_tp_key(probe_name) == 2
    assert (
        manager._get_linear_op_tp_key(
            probe_name,
            ClusterType.MONOLITHIC,
            replica_config,
            is_moe_model=False,
        )
        == 2
    )
    assert trainer._get_training_tp_key(probe_name) == 2

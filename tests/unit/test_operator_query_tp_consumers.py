from __future__ import annotations

from dataclasses import replace
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
import frontier.model_architectures as model_architectures
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


def test_mixed_moe_dense_ffn_uses_attention_tp_domain() -> None:
    """Mixed-layer MoE dense MLP queries use attention TP outside DECODE_FFN."""
    predictor = cast(Any, object.__new__(_ConcreteSklearnExecutionTimePredictor))
    predictor._model_config = SimpleNamespace(
        is_moe=True,
        num_layers=61,
        num_kv_heads=8,
        num_experts=48,
        moe_layers_enum=",".join(str(layer_id) for layer_id in range(4, 60)),
        mlp_hidden_dim=5120,
        dense_mlp_hidden_dim=18432,
        routed_mlp_hidden_dim=5120,
        share_expert_dim=5120,
        get_num_moe_layers=lambda: 56,
        supports_share_expert=lambda: True,
        get_model_architecture_profile=ModelArchitectureProfile.step3_text,
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        speculative_decoding_config=None,
    )

    assert predictor._get_linear_op_tp_key("mlp_up_proj") == 8


def test_mixed_moe_routed_and_shared_ops_use_typed_domains() -> None:
    predictor = cast(Any, object.__new__(_ConcreteSklearnExecutionTimePredictor))
    predictor._model_config = SimpleNamespace(
        is_moe=True,
        num_layers=61,
        num_kv_heads=8,
        num_experts=48,
        moe_layers_enum=",".join(str(layer_id) for layer_id in range(4, 60)),
        mlp_hidden_dim=5120,
        dense_mlp_hidden_dim=18432,
        routed_mlp_hidden_dim=5120,
        share_expert_dim=5120,
        get_num_moe_layers=lambda: 56,
        supports_share_expert=lambda: True,
        get_model_architecture_profile=ModelArchitectureProfile.step3_text,
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        speculative_decoding_config=None,
    )

    assert predictor._get_linear_op_tp_key("moe_grouped_gemm") == 1
    assert predictor._get_linear_op_tp_key("share_expert_up_proj") == 8


def test_typed_dense_domain_preserves_decode_ffn_role_tp() -> None:
    predictor = cast(Any, object.__new__(_ConcreteSklearnExecutionTimePredictor))
    predictor._model_config = SimpleNamespace(
        is_moe=True,
        num_layers=61,
        num_kv_heads=1,
        num_experts=48,
        moe_layers_enum=",".join(str(layer_id) for layer_id in range(4, 60)),
        mlp_hidden_dim=5120,
        dense_mlp_hidden_dim=18432,
        routed_mlp_hidden_dim=5120,
        share_expert_dim=5120,
        get_num_moe_layers=lambda: 56,
        supports_share_expert=lambda: True,
        get_model_architecture_profile=ModelArchitectureProfile.step3_text,
    )
    predictor._cluster_type = ClusterType.DECODE_FFN
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=8,
        moe_expert_parallel_size=8,
        speculative_decoding_config=None,
    )

    assert predictor._get_linear_op_tp_key("mlp_up_proj") == 8
    assert predictor._get_linear_op_tp_key("share_expert_down_proj") == 8


def _step3_typed_contract_config() -> Any:
    return SimpleNamespace(
        model_type="step3_text",
        model_architecture_profile="step3_text",
        is_moe=True,
        num_layers=61,
        num_experts=48,
        moe_layers_enum=",".join(str(layer_id) for layer_id in range(4, 60)),
        mlp_hidden_dim=5120,
        dense_mlp_hidden_dim=18432,
        routed_mlp_hidden_dim=5120,
        share_expert_dim=5120,
        supports_share_expert=lambda: True,
    )


def test_profile_owned_typed_contract_resolves_step3_layer_domains() -> None:
    profile = ModelArchitectureProfile.step3_text()
    config = _step3_typed_contract_config()
    layer_kind = getattr(model_architectures, "LayerKind", None)
    dimension_source = getattr(model_architectures, "LayerDimensionSource", None)
    expert_parallel_mode = getattr(model_architectures, "ExpertParallelMode", None)
    assert layer_kind is not None
    assert dimension_source is not None
    assert expert_parallel_mode is not None

    dense = profile.resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    routed = profile.resolve_layer_contract(
        config,
        layer_id=4,
        operator_name="moe_grouped_gemm",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    shared = profile.resolve_layer_contract(
        config,
        layer_id=4,
        operator_name="share_expert_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )

    assert dense.layer_kind is layer_kind.DENSE
    assert dense.dimension_source is dimension_source.DENSE
    assert dense.effective_ffn_width == 18432
    assert dense.tensor_parallel_size == 8
    assert dense.expert_parallel_mode is expert_parallel_mode.OFF
    assert routed.layer_kind is layer_kind.ROUTED
    assert routed.dimension_source is dimension_source.ROUTED
    assert routed.effective_ffn_width == 5120
    assert routed.tensor_parallel_size == 1
    assert routed.expert_parallel_mode is expert_parallel_mode.ON
    assert routed.expert_parallel_size == 8
    assert shared.layer_kind is layer_kind.SHARED
    assert shared.dimension_source is dimension_source.SHARED
    assert shared.effective_ffn_width == 5120
    assert shared.tensor_parallel_size == 8
    assert shared.expert_parallel_mode is expert_parallel_mode.OFF
    assert {dense.profile_id, routed.profile_id, shared.profile_id} == {"step3_text"}


def test_profile_contract_family_ownership_is_declarative() -> None:
    """A profile can bind an operator family to a typed domain without resolver mapping code."""
    layer_kind = model_architectures.LayerKind
    dimension_source = model_architectures.LayerDimensionSource
    contract = model_architectures.LayerContractSpec(
        layer_kind.ROUTED,
        dimension_source.ROUTED,
        TensorParallelMode.MOE_TP,
        operator_family_ids=("ffn",),
    )
    profile = replace(
        ModelArchitectureProfile.generic(),
        profile_id="unit_declarative_family_profile",
        layer_contracts=(contract,),
    )
    config = SimpleNamespace(
        is_moe=True,
        num_layers=2,
        num_experts=8,
        moe_layers_enum="0,1",
        mlp_hidden_dim=512,
        routed_mlp_hidden_dim=512,
    )

    resolved = profile.resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        moe_tp_size=2,
        expert_parallel_size=4,
    )

    assert resolved.layer_kind is layer_kind.ROUTED
    assert resolved.operator_family_id == "ffn"


def test_shared_expert_contract_rejects_dense_base_layer() -> None:
    with pytest.raises(ValueError, match="shared.*routed"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            _step3_typed_contract_config(),
            layer_id=0,
            operator_name="share_expert_up_proj",
            attention_tp_size=8,
        )


def test_profile_contract_rejects_conflicting_generic_and_domain_tp_aliases() -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size.*attention_tp_size"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            _step3_typed_contract_config(),
            layer_id=0,
            operator_name="mlp_up_proj",
            attention_tp_size=8,
            tensor_parallel_size=4,
        )


def test_profile_contract_rejects_conflicting_attention_tp_aliases() -> None:
    with pytest.raises(ValueError, match="attention_tp_size.*attn_tp_size"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            _step3_typed_contract_config(),
            layer_id=0,
            operator_name="mlp_up_proj",
            attention_tp_size=8,
            attn_tp_size=4,
        )


def test_profile_contract_rejects_conflicting_expert_tp_aliases() -> None:
    with pytest.raises(ValueError, match="expert_parallel_size.*ep_size"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            _step3_typed_contract_config(),
            layer_id=4,
            operator_name="moe_grouped_gemm",
            moe_tp_size=1,
            expert_parallel_size=8,
            ep_size=4,
        )


@pytest.mark.parametrize("bad_attention_tp", [8.0, "8", True, 0, -1])
def test_profile_contract_rejects_invalid_attention_tp_alias(bad_attention_tp) -> None:
    with pytest.raises(ValueError, match="attention_tp_size"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            _step3_typed_contract_config(),
            layer_id=0,
            operator_name="mlp_up_proj",
            attention_tp_size=bad_attention_tp,
        )


@pytest.mark.parametrize("bad_num_experts", [48.0, "48", None, True])
def test_profile_contract_rejects_non_integer_num_experts(bad_num_experts) -> None:
    config = _step3_typed_contract_config()
    config.num_experts = bad_num_experts
    with pytest.raises(ValueError, match="num_experts"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            config,
            layer_id=4,
            operator_name="moe_grouped_gemm",
            moe_tp_size=1,
            expert_parallel_size=8,
        )


@pytest.mark.parametrize("bad_expert_parallel_size", [8.0, "8", True, 0, -1])
def test_profile_contract_rejects_non_positive_integer_expert_parallel_size(
    bad_expert_parallel_size,
) -> None:
    with pytest.raises(ValueError, match="expert_parallel_size"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            _step3_typed_contract_config(),
            layer_id=4,
            operator_name="moe_grouped_gemm",
            moe_tp_size=1,
            expert_parallel_size=bad_expert_parallel_size,
        )


@pytest.mark.parametrize("malformed_layers", ["4,4,5", "-1,4", "4,61", "bad,4"])
def test_profile_contract_rejects_malformed_moe_layer_map(malformed_layers) -> None:
    config = _step3_typed_contract_config()
    config.moe_layers_enum = malformed_layers
    with pytest.raises(ValueError, match="moe_layers_enum"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            config,
            layer_id=4,
            operator_name="moe_grouped_gemm",
            moe_tp_size=1,
            expert_parallel_size=8,
        )


def test_profile_owned_typed_contract_preserves_pure_dense_and_moe_defaults() -> None:
    dense_config = SimpleNamespace(
        is_moe=False,
        num_layers=2,
        mlp_hidden_dim=256,
        dense_mlp_hidden_dim=256,
        routed_mlp_hidden_dim=None,
        share_expert_dim=None,
    )
    dense = ModelArchitectureProfile.generic().resolve_layer_contract(
        dense_config,
        layer_id=1,
        operator_name="mlp_down_proj",
        attention_tp_size=2,
    )
    layer_kind = getattr(model_architectures, "LayerKind", None)
    assert layer_kind is not None
    assert dense.layer_kind is layer_kind.DENSE
    assert dense.effective_ffn_width == 256
    assert dense.tensor_parallel_size == 2

    moe_config = SimpleNamespace(
        is_moe=True,
        num_layers=2,
        num_experts=8,
        mlp_hidden_dim=512,
        routed_mlp_hidden_dim=512,
        dense_mlp_hidden_dim=None,
        share_expert_dim=None,
    )
    routed = ModelArchitectureProfile.generic().resolve_layer_contract(
        moe_config,
        layer_id=1,
        operator_name="moe_grouped_gemm",
        moe_tp_size=2,
        expert_parallel_size=4,
    )
    assert routed.layer_kind is layer_kind.ROUTED
    assert routed.effective_ffn_width == 512
    assert routed.tensor_parallel_size == 2
    assert routed.expert_parallel_size == 4


@pytest.mark.parametrize("layer_id", [-1, 61, True, 1.0, "1"])
def test_profile_owned_typed_contract_rejects_invalid_step3_layer_id(layer_id) -> None:
    with pytest.raises(ValueError, match="layer_id"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            _step3_typed_contract_config(),
            layer_id=layer_id,
            operator_name="mlp_up_proj",
            attention_tp_size=8,
        )


def test_profile_owned_typed_contract_rejects_invalid_width_and_parallel_divisibility() -> None:
    config = _step3_typed_contract_config()
    config.dense_mlp_hidden_dim = 0
    with pytest.raises(ValueError, match="dense.*width"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            config,
            layer_id=0,
            operator_name="mlp_up_proj",
            attention_tp_size=8,
        )

    config.dense_mlp_hidden_dim = 18432
    with pytest.raises(ValueError, match="divisible"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            config,
            layer_id=0,
            operator_name="mlp_up_proj",
            attention_tp_size=7,
        )

    with pytest.raises(ValueError, match="num_experts.*expert_parallel_size"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            config,
            layer_id=4,
            operator_name="moe_grouped_gemm",
            moe_tp_size=1,
            expert_parallel_size=5,
        )


def test_profile_owned_typed_contract_requires_identity_for_mixed_untyped_aggregate() -> None:
    with pytest.raises(ValueError, match="layer_id"):
        ModelArchitectureProfile.step3_text().resolve_layer_contract(
            _step3_typed_contract_config(),
            operator_name=None,
            attention_tp_size=8,
            moe_tp_size=1,
        )


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

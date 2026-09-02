"""Focused regressions for typed-layer filtering in the shared model manager."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
    _serialize_selected_layer_cache_identity,
    _serialize_layer_contract_identity,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.operators.typed_contracts import serialize_typed_operator_contracts
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


def test_linear_loader_filters_rows_by_typed_ffn_width(tmp_path) -> None:
    """A typed dense contract must select its width in addition to TP."""

    input_file = tmp_path / "linear_op.csv"
    pd.DataFrame(
        {
            "num_tokens": [1, 1],
            "num_tensor_parallel_workers": [8, 8],
            "n_expanded_embd": [18432, 5120],
            "time_stats.mlp_up_proj.median": [1.0, 2.0],
        }
    ).to_csv(input_file, index=False)

    config = _step3_config()
    dense_contract = ModelArchitectureProfile.step3_text().resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    filtered = manager._load_linear_op_df(
        str(input_file),
        8,
        layer_contract=dense_contract,
    )

    assert filtered["n_expanded_embd"].tolist() == [18432]
    assert filtered["time_stats.mlp_up_proj.median"].tolist() == [1.0]


def test_linear_loader_scopes_typed_metadata_to_target_operator(tmp_path) -> None:
    """A sibling operator's metadata must not satisfy the requested target."""

    input_file = tmp_path / "linear_op.csv"
    config = _step3_config()
    dense_contract = ModelArchitectureProfile.step3_text().resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    dense_metadata = dense_contract.typed_metadata_identity()
    wrong_metadata = dict(dense_metadata)
    wrong_metadata["effective_ffn_width"] = 9999
    pd.DataFrame(
        {
            "num_tokens": [1],
            "num_tensor_parallel_workers": [8],
            "n_expanded_embd": [18432],
            "typed_operator_contracts": [
                    serialize_typed_operator_contracts({
                        "mlp_up_proj": wrong_metadata,
                        "mlp_down_proj": dense_metadata,
                    })
                ],
            "time_stats.mlp_up_proj.median": [1.0],
        }
    ).to_csv(input_file, index=False)
    manager = object.__new__(ExecutionTimePredictionModelManager)

    with pytest.raises(ValueError, match="effective_ffn_width"):
        manager._load_linear_op_df(
            str(input_file),
            8,
            layer_contract=dense_contract,
            operator_name="mlp_up_proj",
        )


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("operator_family_ids", ["moe"]),
        ("tensor_parallel_sizes", [1]),
        ("selected_padded_ffn_width", 9999),
    ],
)
def test_linear_loader_rejects_inconsistent_complete_typed_row(
    tmp_path, field_name: str, wrong_value: object
) -> None:
    """The production loader must not admit a row rejected by typed validation."""

    config = _step3_config()
    dense_contract = ModelArchitectureProfile.step3_text().resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    metadata = dense_contract.typed_metadata_identity()
    metadata.update(
        {
            "operator_family_ids": [dense_contract.operator_family_id],
            "tensor_parallel_sizes": [dense_contract.tensor_parallel_size],
            "selected_padded_ffn_width": dense_contract.effective_ffn_width,
        }
    )
    metadata[field_name] = wrong_value
    input_file = tmp_path / "inconsistent_typed_linear_op.csv"
    pd.DataFrame(
        {
            "num_tokens": [1],
            "num_tensor_parallel_workers": [8],
            "n_expanded_embd": [18432],
            "typed_operator_contracts": [
                serialize_typed_operator_contracts({"mlp_up_proj": metadata})
            ],
            "time_stats.mlp_up_proj.median": [1.0],
        }
    ).to_csv(input_file, index=False)
    manager = object.__new__(ExecutionTimePredictionModelManager)

    with pytest.raises(ValueError, match=field_name):
        manager._load_linear_op_df(
            str(input_file),
            8,
            layer_contract=dense_contract,
            operator_name="mlp_up_proj",
        )


@pytest.mark.parametrize(
    "contracts",
    [
        {"mlp_down_proj": "dense"},
        {"mlp_up_proj": "dense", "mlp_down_proj": "dense"},
    ],
)
def test_linear_loader_requires_operator_scope_for_typed_rows(
    tmp_path, contracts
) -> None:
    """Typed rows cannot be admitted without an exact operator scope."""

    config = _step3_config()
    dense_contract = ModelArchitectureProfile.step3_text().resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    metadata = dense_contract.typed_metadata_identity()
    typed_contracts = {
        operator_name: metadata
        for operator_name in contracts
    }
    input_file = tmp_path / "typed_linear_op.csv"
    pd.DataFrame(
        {
            "num_tokens": [1],
            "num_tensor_parallel_workers": [8],
            "n_expanded_embd": [18432],
            "typed_operator_contracts": [
                serialize_typed_operator_contracts(typed_contracts)
            ],
            "time_stats.mlp_up_proj.median": [1.0],
        }
    ).to_csv(input_file, index=False)
    manager = object.__new__(ExecutionTimePredictionModelManager)

    with pytest.raises(ValueError, match="requires operator_name"):
        manager._load_linear_op_df(
            str(input_file),
            8,
            layer_contract=dense_contract,
        )


def test_linear_loader_rejects_sibling_only_typed_row_for_target_operator(
    tmp_path,
) -> None:
    """A sibling-only row must not satisfy a different typed operator query."""

    config = _step3_config()
    dense_contract = ModelArchitectureProfile.step3_text().resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    input_file = tmp_path / "sibling_only_linear_op.csv"
    pd.DataFrame(
        {
            "num_tokens": [1],
            "num_tensor_parallel_workers": [8],
            "n_expanded_embd": [18432],
            "typed_operator_contracts": [
                serialize_typed_operator_contracts(
                    {"mlp_down_proj": dense_contract.typed_metadata_identity()}
                )
            ],
            "time_stats.mlp_up_proj.median": [1.0],
        }
    ).to_csv(input_file, index=False)
    manager = object.__new__(ExecutionTimePredictionModelManager)

    with pytest.raises(ValueError, match="No linear-op rows match"):
        manager._load_linear_op_df(
            str(input_file),
            8,
            layer_contract=dense_contract,
            operator_name="mlp_up_proj",
        )


def test_moe_loader_filters_rows_by_typed_routed_width(tmp_path) -> None:
    """A routed contract must select expert width instead of legacy model width."""

    input_file = tmp_path / "moe.csv"
    pd.DataFrame(
        {
            "num_tokens": [1, 1],
            "num_experts": [48, 48],
            "router_topk": [3, 3],
            "hidden_dim": [7168, 7168],
            "expert_hidden_dim": [5120, 9999],
            "num_tensor_parallel_workers": [1, 1],
            "expert_parallel_size": [8, 8],
            "time_stats.moe_grouped_gemm.median": [1.0, 2.0],
        }
    ).to_csv(input_file, index=False)

    config = _step3_config()
    config.embedding_dim = 7168
    config.num_experts_per_tok = 3
    config.mlp_hidden_dim = 9999
    routed_contract = ModelArchitectureProfile.step3_text().resolve_layer_contract(
        config,
        layer_id=4,
        operator_name="moe_grouped_gemm",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    filtered = manager._load_moe_df(
        str(input_file),
        SimpleNamespace(
            model_config=config,
            moe_tensor_parallel_size=1,
            moe_expert_parallel_size=8,
        ),
        load_imbalance=False,
        tensor_parallel_size=1,
        expert_parallel_size=8,
        layer_contract=routed_contract,
    )

    assert filtered["expert_hidden_dim"].tolist() == [5120]
    assert filtered["time_stats.moe_grouped_gemm.median"].tolist() == [1.0]


def test_manager_resolves_each_step3_ffn_family_from_profile_contract() -> None:
    """The manager must bind each FFN family to its declared typed domain."""

    config = _step3_config()
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    dense = manager._resolve_typed_layer_contract(
        "mlp_up_proj",
        ClusterType.MONOLITHIC,
        replica_config,
        is_moe_model=False,
        layer_id=0,
    )
    routed = manager._resolve_typed_layer_contract(
        "moe_grouped_gemm",
        ClusterType.MONOLITHIC,
        replica_config,
        is_moe_model=True,
        layer_id=4,
    )
    shared = manager._resolve_typed_layer_contract(
        "share_expert_up_proj",
        ClusterType.MONOLITHIC,
        replica_config,
        is_moe_model=True,
        layer_id=4,
    )

    assert (dense.layer_kind.value, dense.width, dense.tensor_parallel_size) == (
        "dense",
        18432,
        8,
    )
    assert (
        routed.layer_kind.value,
        routed.width,
        routed.tensor_parallel_size,
        routed.expert_parallel_size,
    ) == ("routed", 5120, 1, 8)
    assert (shared.layer_kind.value, shared.width, shared.tensor_parallel_size) == (
        "shared",
        5120,
        8,
    )


def test_decode_attn_without_ffn_domain_skips_typed_ffn_resolution() -> None:
    """PD-AF attention-only clusters do not own a typed FFN selector."""

    config = _step3_config()
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=0,
        moe_tensor_parallel_size=0,
        moe_expert_parallel_size=0,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    assert manager._resolve_typed_layer_contract(
        "mlp_up_proj",
        ClusterType.DECODE_ATTN,
        replica_config,
        is_moe_model=False,
        layer_id=0,
    ) is None


def test_decode_attn_nonzero_ffn_domain_fails_before_profile_resolution() -> None:
    """A malformed attention-only role must not silently discard its FFN domain."""

    config = _step3_config()
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    with pytest.raises(ValueError, match="exact zero MoE/FFN parallel sizes"):
        manager._resolve_typed_layer_contract(
            "mlp_up_proj",
            ClusterType.DECODE_ATTN,
            replica_config,
            is_moe_model=False,
            layer_id=0,
        )


def test_manager_enumerates_profile_owned_contracts_without_fixed_family_arity(
    monkeypatch,
) -> None:
    """A profile's declared contract set, rather than a manager tuple, owns enumeration."""

    import frontier.execution_time_predictor.shared_prediction_model_manager as module
    from frontier.model_architectures import LayerContractSpec, LayerKind, LayerDimensionSource
    from frontier.operators.spec import TensorParallelMode

    profile = replace(
        ModelArchitectureProfile.generic(),
        profile_id="unit_dense_only_profile",
        layer_contracts=(
            LayerContractSpec(
                LayerKind.DENSE,
                LayerDimensionSource.DENSE,
                TensorParallelMode.FFN_TP,
                operator_family_ids=("ffn",),
            ),
        ),
    )
    config = SimpleNamespace(
        is_moe=False,
        num_layers=2,
        mlp_hidden_dim=256,
        dense_mlp_hidden_dim=256,
        get_model_architecture_profile=lambda: profile,
    )
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    # This is the old implementation's fixed-arity dependency. A profile
    # contract must remain resolvable when that legacy alias is absent.
    monkeypatch.setattr(module, "_TYPED_FFN_FAMILIES", (), raising=False)

    entries = manager._resolve_ffn_layer_contracts(
        ClusterType.MONOLITHIC,
        replica_config,
        is_moe_model=False,
    )

    assert [family_id for family_id, _ in entries] == ["ffn"]
    assert entries[0][1].width == 256


def test_registered_binding_error_is_not_downgraded_to_legacy_path(monkeypatch) -> None:
    """An ambiguous registered alias must remain an observable binding error."""

    import frontier.execution_time_predictor.shared_prediction_model_manager as module

    config = _step3_config()
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    def _ambiguous_binding(_op_name, **_kwargs):
        raise ValueError("Operator query 'mlp_up_proj' is an ambiguous profiling alias")

    monkeypatch.setattr(module, "bind_operator_query", _ambiguous_binding)

    with pytest.raises(ValueError, match="ambiguous profiling alias"):
        manager._resolve_typed_layer_contract(
            "mlp_up_proj",
            ClusterType.MONOLITHIC,
            replica_config,
            is_moe_model=False,
        )


@pytest.mark.parametrize(
    "cluster_type",
    [ClusterType.MONOLITHIC, ClusterType.PREFILL, ClusterType.DECODE_FFN],
)
def test_memory_add_alias_keeps_legacy_replicated_tp_mapping(cluster_type) -> None:
    """The shared ``add`` profiling alias must retain MEMORY TP semantics."""

    config = _step3_config()
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        speculative_decoding_config=None,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    assert (
        manager._get_linear_op_tp_key(
            "add", cluster_type, replica_config, is_moe_model=True
        )
        == 1
    )


def test_linear_loader_rejects_tp_mismatch_with_typed_contract(tmp_path) -> None:
    """A typed linear load must reject a TP selector that differs from its contract."""

    input_file = tmp_path / "linear_op.csv"
    pd.DataFrame(
        {
            "num_tokens": [1],
            "num_tensor_parallel_workers": [4],
            "n_expanded_embd": [18432],
            "time_stats.mlp_up_proj.median": [1.0],
        }
    ).to_csv(input_file, index=False)
    contract = ModelArchitectureProfile.step3_text().resolve_layer_contract(
        _step3_config(),
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    with pytest.raises(ValueError, match="contract TP.*tensor_parallel_size"):
        manager._load_linear_op_df(
            str(input_file),
            4,
            layer_contract=contract,
        )


def test_moe_loader_rejects_ep_mismatch_but_allows_ep_agnostic_selection(tmp_path) -> None:
    """Routed loads enforce explicit EP while preserving wildcard EP-agnostic loads."""

    input_file = tmp_path / "moe.csv"
    pd.DataFrame(
        {
            "num_tokens": [1],
            "num_experts": [48],
            "router_topk": [3],
            "hidden_dim": [7168],
            "expert_hidden_dim": [5120],
            "num_tensor_parallel_workers": [1],
            "expert_parallel_size": [8],
            "time_stats.moe_gating_linear.median": [1.0],
        }
    ).to_csv(input_file, index=False)
    config = _step3_config()
    config.embedding_dim = 7168
    config.num_experts_per_tok = 3
    contract = ModelArchitectureProfile.step3_text().resolve_layer_contract(
        config,
        operator_name="moe_grouped_gemm",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)
    replica_config = SimpleNamespace(
        model_config=config,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )

    with pytest.raises(ValueError, match="contract EP.*expert_parallel_size"):
        manager._load_moe_df(
            str(input_file),
            replica_config,
            load_imbalance=False,
            tensor_parallel_size=1,
            expert_parallel_size=4,
            layer_contract=contract,
        )

    filtered = manager._load_moe_df(
        str(input_file),
        replica_config,
        load_imbalance=False,
        tensor_parallel_size=1,
        expert_parallel_size=None,
        layer_contract=contract,
    )
    assert len(filtered) == 1


def test_manager_linear_tp_key_uses_profile_domain_for_shared_expert() -> None:
    """Shared-expert linear rows use the profile-owned attention TP domain."""

    config = _step3_config()
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        speculative_decoding_config=None,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    assert manager._get_linear_op_tp_key(
        "share_expert_up_proj",
        ClusterType.MONOLITHIC,
        replica_config,
        is_moe_model=True,
    ) == 8


def test_dense_training_passes_typed_contract_to_loader_and_context(tmp_path) -> None:
    """Dense training must preserve the resolved contract across its boundary."""

    linear_file = tmp_path / "linear_op.csv"
    linear_file.write_text("placeholder\n", encoding="utf-8")
    config = _step3_config()
    config.supports_share_expert = lambda: False
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)
    captured: dict[str, object] = {}

    def _load_linear(*args, **kwargs):
        captured["loader_contract"] = kwargs.get("layer_contract")
        captured.setdefault("loader_operator_names", []).append(
            kwargs.get("operator_name")
        )
        return pd.DataFrame(
            {
                "num_tokens": [1, 2],
                "time_stats.mlp_up_proj.median": [1.0, 2.0],
                "time_stats.mlp_down_proj.median": [1.0, 2.0],
                "time_stats.mlp_act.median": [1.0, 2.0],
            }
        )

    def _train_single(**kwargs):
        captured.setdefault("contexts", []).append(kwargs["training_context"])
        return kwargs["model_name"]

    manager._load_linear_op_df = _load_linear
    manager._train_single_model = _train_single

    manager._train_dense_mlp_models_for_cluster(
        cluster_type=ClusterType.MONOLITHIC,
        replica_config=replica_config,
        execution_time_predictor_config=SimpleNamespace(),
        linear_ops_file=str(linear_file),
        ffn_signature="dense",
        ffn_tp_key=8,
        training_context={"model_name": "step3-moe-noquant"},
        trained_model_signatures=set(),
        models={},
    )

    loader_contract = captured["loader_contract"]
    assert loader_contract.width == 18432
    assert loader_contract.tensor_parallel_size == 8
    assert captured["loader_operator_names"] == ["mlp_up_proj"]
    contexts = captured["contexts"]
    assert contexts
    assert all(context["layer_contract"].width == 18432 for context in contexts)


def test_moe_dataset_validation_uses_routed_typed_width(tmp_path) -> None:
    """MoE coverage validation must not reuse the legacy model-wide width."""

    input_file = tmp_path / "moe.csv"
    pd.DataFrame(
        {
            "num_experts": [48],
            "router_topk": [3],
            "hidden_dim": [7168],
            "expert_hidden_dim": [5120],
            "num_tensor_parallel_workers": [1],
            "expert_parallel_size": [8],
            "time_stats.moe_grouped_gemm.median": [1.0],
        }
    ).to_csv(input_file, index=False)
    config = _step3_config()
    config.embedding_dim = 7168
    config.num_experts_per_tok = 3
    config.mlp_hidden_dim = 18432
    replica_config = SimpleNamespace(
        model_config=config,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        moe_routing_distribution_type="balanced",
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    manager._validate_moe_dataset_contract(
        str(input_file),
        replica_config,
        ["moe_grouped_gemm"],
        ClusterType.MONOLITHIC,
    )


def test_moe_training_passes_routed_contract_to_loader_and_context(tmp_path) -> None:
    """Routed MoE training must carry its profile-owned contract end to end."""

    linear_file = tmp_path / "linear_op.csv"
    moe_file = tmp_path / "moe.csv"
    linear_file.write_text("placeholder\n", encoding="utf-8")
    moe_file.write_text("placeholder\n", encoding="utf-8")
    config = _step3_config()
    config.supports_share_expert = lambda: False
    replica_config = SimpleNamespace(
        device="h200",
        model_name="step3-moe-noquant",
        model_config=config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        speculative_decoding_config=None,
    )
    config.get_model_arch = lambda: "step3_text"
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._measurement_family_name = lambda _measurement_type: "cuda_event"
    manager._validate_moe_dataset_contract = lambda *args, **kwargs: None
    manager._get_moe_op_tp_key = lambda *_args, **_kwargs: 1
    manager._is_moe_op_ep_agnostic = lambda *_args, **_kwargs: False
    manager._load_moe_df = lambda *args, **kwargs: pd.DataFrame(
        {"num_tokens": [1], "time_stats.moe_grouped_gemm.median": [1.0]}
    )
    manager._load_linear_op_df = lambda *args, **kwargs: pd.DataFrame(
        {
            "num_tokens": [1],
            "n_expanded_embd": [18432],
            "time_stats.post_attention_layernorm.median": [1.0],
            "time_stats.mlp_up_proj.median": [1.0],
            "time_stats.mlp_down_proj.median": [1.0],
            "time_stats.mlp_act.median": [1.0],
        }
    )
    captured = []
    manager._train_single_model = lambda **kwargs: captured.append(kwargs) or kwargs[
        "model_name"
    ]

    import frontier.execution_time_predictor.shared_prediction_model_manager as module

    original_names = module._get_moe_family_model_names
    module._get_moe_family_model_names = lambda: ["moe_grouped_gemm"]
    try:
        manager._train_ffn_models_for_cluster(
            ClusterType.MONOLITHIC,
            replica_config,
            SimpleNamespace(),
            str(linear_file),
            str(moe_file),
            True,
            set(),
        )
    finally:
        module._get_moe_family_model_names = original_names

    routed = next(row for row in captured if row["model_name"] == "moe_grouped_gemm")
    assert routed["layer_contract"].width == 5120
    assert routed["training_context"]["layer_contract"].width == 5120

    layernorm = next(
        row for row in captured if row["model_name"] == "post_attention_layernorm"
    )
    assert "layer_contract" not in layernorm["training_context"]


def test_shared_training_passes_shared_contract_to_loader(tmp_path) -> None:
    """Shared-expert training must use the shared width and TP domain."""

    linear_file = tmp_path / "linear_op.csv"
    moe_file = tmp_path / "moe.csv"
    linear_file.write_text("placeholder\n", encoding="utf-8")
    moe_file.write_text("placeholder\n", encoding="utf-8")
    config = _step3_config()
    replica_config = SimpleNamespace(
        device="h200",
        model_name="step3-moe-noquant",
        model_config=config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        speculative_decoding_config=None,
    )
    config.get_model_arch = lambda: "step3_text"
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._measurement_family_name = lambda _measurement_type: "cuda_event"
    manager._validate_moe_dataset_contract = lambda *args, **kwargs: None
    manager._get_moe_op_tp_key = lambda *_args, **_kwargs: 1
    manager._is_moe_op_ep_agnostic = lambda *_args, **_kwargs: False
    manager._load_moe_df = lambda *args, **kwargs: pd.DataFrame(
        {"num_tokens": [1], "time_stats.moe_grouped_gemm.median": [1.0]}
    )
    loader_contracts = []
    loader_operator_names = []

    def _load_linear(*args, **kwargs):
        loader_contracts.append(kwargs.get("layer_contract"))
        loader_operator_names.append(kwargs.get("operator_name"))
        return pd.DataFrame(
            {
                "num_tokens": [1],
                "n_expanded_embd": [5120],
                "time_stats.share_expert_up_proj.median": [1.0],
                "time_stats.share_expert_down_proj.median": [1.0],
                "time_stats.share_expert_act.median": [1.0],
                "time_stats.post_attention_layernorm.median": [1.0],
            }
        )

    manager._load_linear_op_df = _load_linear
    manager._train_single_model = lambda **kwargs: kwargs["model_name"]

    import frontier.execution_time_predictor.shared_prediction_model_manager as module

    original_moe_names = module._get_moe_family_model_names
    original_family_names = module.get_family_profiling_names
    module._get_moe_family_model_names = lambda: ["moe_grouped_gemm"]
    module.get_family_profiling_names = lambda family: (
        ["share_expert_up_proj", "share_expert_down_proj", "share_expert_act"]
        if family is module.SHARE_EXPERT_FAMILY
        else ["moe_grouped_gemm"]
    )
    try:
        manager._train_ffn_models_for_cluster(
            ClusterType.MONOLITHIC,
            replica_config,
            SimpleNamespace(),
            str(linear_file),
            str(moe_file),
            True,
            set(),
        )
    finally:
        module._get_moe_family_model_names = original_moe_names
        module.get_family_profiling_names = original_family_names

    assert any(contract is not None and contract.width == 5120 for contract in loader_contracts)
    assert "share_expert_up_proj" in loader_operator_names


def test_model_hash_includes_typed_layer_contract_identity() -> None:
    """Different typed widths/domains must never share a model cache key."""

    manager = object.__new__(ExecutionTimePredictionModelManager)
    config = SimpleNamespace(
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
    dataframe = pd.DataFrame({"num_tokens": [1], "value": [2.0]})
    step3 = _step3_config()
    dense = ModelArchitectureProfile.step3_text().resolve_layer_contract(
        step3,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    routed = ModelArchitectureProfile.step3_text().resolve_layer_contract(
        step3,
        operator_name="moe_grouped_gemm",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    dense_hash = manager._get_model_hash(
        "ffn", dataframe, config, "FP16", MeasurementType.CUDA_EVENT, dense
    )
    routed_hash = manager._get_model_hash(
        "ffn", dataframe, config, "FP16", MeasurementType.CUDA_EVENT, routed
    )
    assert dense_hash != routed_hash

    # Cache identity is scoped to the selected typed domain, not to one
    # concrete layer occurrence or the producer's wider TP envelope.
    dense_layer_zero = replace(dense, layer_id=0)
    dense_layer_one = replace(dense, layer_id=1)
    assert manager._get_model_hash(
        "ffn", dataframe, config, "FP16", MeasurementType.CUDA_EVENT, dense_layer_zero
    ) == manager._get_model_hash(
        "ffn", dataframe, config, "FP16", MeasurementType.CUDA_EVENT, dense_layer_one
    )

    padded_variant = replace(
        dense,
        selected_padded_ffn_width=dense.selected_padded_ffn_width + 8,
    )
    assert manager._get_model_hash(
        "ffn", dataframe, config, "FP16", MeasurementType.CUDA_EVENT, padded_variant
    ) != dense_hash

    tp_domain_variant = replace(
        dense,
        tensor_parallel_size=4,
        tensor_parallel_sizes=(4,),
        selected_padded_ffn_width=18432,
    )
    assert manager._get_model_hash(
        "ffn", dataframe, config, "FP16", MeasurementType.CUDA_EVENT, tp_domain_variant
    ) != dense_hash

    assert manager._get_model_hash(
        "ffn", dataframe, config, "BF16", MeasurementType.CUDA_EVENT, dense
    ) != dense_hash
    assert manager._get_model_hash(
        "ffn", dataframe, config, "FP16", MeasurementType.KERNEL_ONLY, dense
    ) != dense_hash


def test_ffn_signature_includes_profile_owned_moe_layer_map() -> None:
    """Changing only MoE layer placement must invalidate the manager signature."""

    config_official = _step3_config()
    config_alternate = _step3_config()
    config_alternate.moe_layers_enum = ",".join(
        str(layer_id) for layer_id in range(0, 56)
    )
    replica_official = SimpleNamespace(
        model_config=config_official,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    replica_alternate = SimpleNamespace(
        model_config=config_alternate,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    official_signature = manager._get_ffn_contract_signature(
        ClusterType.MONOLITHIC,
        replica_official,
        is_moe_model=True,
    )
    alternate_signature = manager._get_ffn_contract_signature(
        ClusterType.MONOLITHIC,
        replica_alternate,
        is_moe_model=True,
    )

    assert official_signature != alternate_signature


def test_model_hash_ignores_profile_wide_layer_map_for_selected_domain() -> None:
    """A selected typed domain must not inherit profile-wide layer-map state."""

    manager = object.__new__(ExecutionTimePredictionModelManager)
    execution_config = SimpleNamespace(
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
    dataframe = pd.DataFrame({"num_tokens": [1], "value": [2.0]})
    official_config = _step3_config()
    alternate_config = _step3_config()
    alternate_config.moe_layers_enum = ",".join(
        str(layer_id) for layer_id in range(0, 56)
    )
    alternate_config.num_layers = 62
    profile = ModelArchitectureProfile.step3_text()
    dense_official = profile.resolve_layer_contract(
        official_config,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    dense_alternate = profile.resolve_layer_contract(
        alternate_config,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    official_identity = profile.serialize_layer_contract_identity(
        official_config,
        layer_contract=dense_official,
    )
    alternate_identity = profile.serialize_layer_contract_identity(
        alternate_config,
        layer_contract=dense_alternate,
    )

    official_hash = manager._get_model_hash(
        "ffn",
        dataframe,
        execution_config,
        "FP16",
        MeasurementType.CUDA_EVENT,
        dense_official,
        profile_layer_contract_identity=official_identity,
    )
    alternate_hash = manager._get_model_hash(
        "ffn",
        dataframe,
        execution_config,
        "FP16",
        MeasurementType.CUDA_EVENT,
        dense_alternate,
        profile_layer_contract_identity=alternate_identity,
    )

    assert official_hash == alternate_hash


def _bare_contract_view_manager(
    contracts: dict[tuple[str, str], object],
    cluster_configs=None,
):
    """Build the smallest manager shell needed to exercise model views."""

    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._all_dummy_mode = False
    manager._trained_models_eager = {}
    manager._trained_models_kernel_only = {}
    manager._trained_models_eager_by_contract = contracts
    manager._trained_models_kernel_only_by_contract = {}
    manager._models_by_precision_eager = {}
    manager._models_by_precision_kernel_only = {}
    manager._models_by_precision_eager_by_contract = {}
    manager._models_by_precision_kernel_only_by_contract = {}
    manager._cluster_configs = cluster_configs or {}
    return manager


def test_get_models_rejects_ambiguous_typed_contract_without_cluster_context() -> None:
    """A bare model-name view must not guess between typed contract variants."""

    config = _step3_config()
    profile = ModelArchitectureProfile.step3_text()
    tp8 = profile.resolve_layer_contract(
        config,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    tp4 = profile.resolve_layer_contract(
        config,
        operator_name="mlp_up_proj",
        attention_tp_size=4,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    manager = _bare_contract_view_manager(
        {
            ("mlp_up_proj", _serialize_selected_layer_cache_identity(tp8)): object(),
            ("mlp_up_proj", _serialize_selected_layer_cache_identity(tp4)): object(),
        }
    )

    with pytest.raises(ValueError, match="multiple layer contracts"):
        manager.get_models()


def test_cluster_model_view_selects_profile_resolved_typed_contract() -> None:
    """A cluster view must select the identity declared by its replica config."""

    config = _step3_config()
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    profile = ModelArchitectureProfile.step3_text()
    requested = profile.resolve_layer_contract(
        config,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    requested_identity = _serialize_selected_layer_cache_identity(requested)
    selected = object()
    other = object()
    alternate_identity = _serialize_selected_layer_cache_identity(
        profile.resolve_layer_contract(
            config,
            operator_name="mlp_up_proj",
            attention_tp_size=4,
            moe_tp_size=1,
            expert_parallel_size=8,
        )
    )
    manager = _bare_contract_view_manager(
        {
            ("mlp_up_proj", requested_identity): selected,
            ("mlp_up_proj", alternate_identity): other,
        },
        cluster_configs={
            ClusterType.MONOLITHIC: SimpleNamespace(replica_config=replica_config)
        },
    )
    manager._is_kernel_only_measurement_enabled_for_cluster = lambda _cluster: False

    models = manager.get_models_for_cluster(ClusterType.MONOLITHIC)

    assert models["eager"]["mlp_up_proj"] is selected


def _typed_contract_pair():
    config = _step3_config()
    profile = ModelArchitectureProfile.step3_text()
    dense = profile.resolve_layer_contract(
        config,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    routed = profile.resolve_layer_contract(
        config,
        operator_name="moe_grouped_gemm",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    return dense, routed


def _cache_manager(tmp_path):
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._cache_dir = str(tmp_path)
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._get_model_hash = lambda *_args, **_kwargs: "typed-cache"
    manager._store_model_precision = lambda *_args, **_kwargs: None
    return manager


def _cache_training_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_tokens": [1],
            "target": [1.0],
            "profiling_precision": ["FP16"],
            "measurement_type": [MeasurementType.CUDA_EVENT.value],
        }
    )


def test_explicit_and_context_layer_contract_conflict_fails_fast() -> None:
    """An explicit contract must never silently overwrite context provenance."""

    dense, routed = _typed_contract_pair()
    manager = object.__new__(ExecutionTimePredictionModelManager)

    with pytest.raises(ValueError, match="conflicting layer_contract"):
        manager._train_single_model(
            model_name="mlp_up_proj",
            df=pd.DataFrame(),
            feature_cols=["num_tokens"],
            target_col="target",
            execution_time_predictor_config=SimpleNamespace(),
            training_context={"layer_contract": routed},
            layer_contract=dense,
        )


def test_context_layer_contract_identity_conflict_fails_fast() -> None:
    """Serialized context identity must describe its attached contract exactly."""

    dense, routed = _typed_contract_pair()
    manager = object.__new__(ExecutionTimePredictionModelManager)

    with pytest.raises(ValueError, match="layer_contract_identity"):
        manager._train_single_model(
            model_name="mlp_up_proj",
            df=pd.DataFrame(),
            feature_cols=["num_tokens"],
            target_col="target",
            execution_time_predictor_config=SimpleNamespace(),
            training_context={
                "layer_contract": dense,
                "layer_contract_identity": _serialize_layer_contract_identity(routed),
            },
        )


def test_typed_cache_missing_contract_identity_is_rejected(tmp_path) -> None:
    """A legacy pickle without typed identity cannot serve a typed request."""

    dense, _ = _typed_contract_pair()
    manager = _cache_manager(tmp_path)
    manager._store_model_in_cache("mlp_up_proj", "typed-cache", SimpleNamespace())

    with pytest.raises(ValueError, match="missing layer contract identity"):
        manager._train_single_model(
            model_name="mlp_up_proj",
            df=_cache_training_dataframe(),
            feature_cols=["num_tokens"],
            target_col="target",
            execution_time_predictor_config=SimpleNamespace(),
            layer_contract=dense,
        )


def test_typed_cache_identity_mismatch_is_rejected(tmp_path) -> None:
    """A pickle for a different typed domain cannot be reused."""

    dense, routed = _typed_contract_pair()
    manager = _cache_manager(tmp_path)
    cached = SimpleNamespace(
        _frontier_layer_contract_identity=_serialize_layer_contract_identity(routed)
    )
    manager._store_model_in_cache("mlp_up_proj", "typed-cache", cached)

    with pytest.raises(ValueError, match="layer contract identity mismatch"):
        manager._train_single_model(
            model_name="mlp_up_proj",
            df=_cache_training_dataframe(),
            feature_cols=["num_tokens"],
            target_col="target",
            execution_time_predictor_config=SimpleNamespace(),
            layer_contract=dense,
        )


def test_typed_cache_selected_identity_mismatch_is_rejected(tmp_path) -> None:
    """A cache key collision cannot bypass selected-domain metadata checks."""

    dense, routed = _typed_contract_pair()
    manager = _cache_manager(tmp_path)
    cached = SimpleNamespace(
        _frontier_layer_contract_identity=_serialize_layer_contract_identity(dense),
        _frontier_layer_cache_identity=_serialize_selected_layer_cache_identity(
            routed
        ),
    )
    manager._store_model_in_cache("mlp_up_proj", "typed-cache", cached)

    with pytest.raises(ValueError, match="selected layer cache identity mismatch"):
        manager._train_single_model(
            model_name="mlp_up_proj",
            df=_cache_training_dataframe(),
            feature_cols=["num_tokens"],
            target_col="target",
            execution_time_predictor_config=SimpleNamespace(),
            layer_contract=dense,
        )


def test_typed_cache_identity_survives_pickle_round_trip(tmp_path) -> None:
    """The profile-owned identity remains attached after cache serialization."""

    dense, _ = _typed_contract_pair()
    expected_identity = _serialize_layer_contract_identity(dense)
    manager = _cache_manager(tmp_path)
    cached = SimpleNamespace(
        _frontier_layer_contract_identity=expected_identity,
    )
    manager._store_model_in_cache("mlp_up_proj", "typed-cache", cached)

    loaded = manager._load_model_from_cache("mlp_up_proj", "typed-cache")

    assert loaded._frontier_layer_contract_identity == expected_identity


def test_typed_cache_reuses_same_selected_domain_across_layer_ids(tmp_path) -> None:
    """A selected-domain cache entry is reusable across equivalent layers."""

    config = _step3_config()
    profile = ModelArchitectureProfile.step3_text()
    layer_zero = profile.resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    layer_one = profile.resolve_layer_contract(
        config,
        layer_id=1,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    manager = _cache_manager(tmp_path)
    cached = SimpleNamespace(
        _frontier_layer_contract_identity=_serialize_layer_contract_identity(layer_zero),
        _frontier_layer_cache_identity=_serialize_selected_layer_cache_identity(
            layer_zero
        ),
    )
    manager._store_model_in_cache("mlp_up_proj", "typed-cache", cached)

    loaded = manager._train_single_model(
        model_name="mlp_up_proj",
        df=_cache_training_dataframe(),
        feature_cols=["num_tokens"],
        target_col="target",
        execution_time_predictor_config=SimpleNamespace(),
        layer_contract=layer_one,
    )

    assert loaded._frontier_layer_cache_identity == cached._frontier_layer_cache_identity
    assert loaded._frontier_layer_contract_identity == cached._frontier_layer_contract_identity


def _initialized_storage_manager() -> ExecutionTimePredictionModelManager:
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._all_dummy_mode = False
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._trained_models_eager = {}
    manager._trained_models_kernel_only = {}
    manager._models_by_precision_eager = {}
    manager._models_by_precision_kernel_only = {}
    manager._cluster_configs = {}
    return manager


def test_typed_model_storage_keeps_contract_variants_isolated() -> None:
    """Models sharing an operator name must remain isolated by typed identity."""

    dense, routed = _typed_contract_pair()
    manager = _initialized_storage_manager()
    dense_model = SimpleNamespace()
    routed_model = SimpleNamespace()

    manager._store_model_precision(
        "ffn_operator", "FP16", dense_model, layer_contract=dense
    )
    manager._store_model_precision(
        "ffn_operator", "FP16", routed_model, layer_contract=routed
    )

    assert (
        manager.get_model("ffn_operator", layer_contract=dense) is dense_model
    )
    assert (
        manager.get_model("ffn_operator", layer_contract=routed) is routed_model
    )
    with pytest.raises(ValueError, match="multiple layer contracts"):
        manager.get_model("ffn_operator")


def test_typed_model_storage_reuses_selected_domain_across_layer_ids() -> None:
    """Equivalent physical layers share one selected-domain registry entry."""

    config = _step3_config()
    profile = ModelArchitectureProfile.step3_text()
    layer_zero = profile.resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    layer_one = profile.resolve_layer_contract(
        config,
        layer_id=1,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    assert layer_zero.layer_id != layer_one.layer_id
    assert _serialize_selected_layer_cache_identity(layer_zero) == (
        _serialize_selected_layer_cache_identity(layer_one)
    )

    manager = _initialized_storage_manager()
    first = SimpleNamespace(label="layer-zero")
    second = SimpleNamespace(label="layer-one")
    manager._store_model_precision(
        "mlp_up_proj", "FP16", first, layer_contract=layer_zero
    )
    manager._store_model_precision(
        "mlp_up_proj", "FP16", second, layer_contract=layer_one
    )

    selected_identity = _serialize_selected_layer_cache_identity(layer_zero)
    assert set(manager._trained_models_eager_by_contract) == {
        ("mlp_up_proj", selected_identity)
    }
    assert manager.get_model("mlp_up_proj", layer_contract=layer_zero) is second
    assert manager.get_model("mlp_up_proj", layer_contract=layer_one) is second


def test_cluster_model_view_selects_the_contract_for_that_replica() -> None:
    """Cluster-specific retrieval must select the configured typed TP domain."""

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
    manager = _initialized_storage_manager()
    manager._cluster_configs = {
        ClusterType.PREFILL: SimpleNamespace(
            replica_config=SimpleNamespace(
                model_name="step3-moe-noquant",
                device="h200",
                model_config=config,
                attn_tensor_parallel_size=8,
                moe_tensor_parallel_size=1,
                moe_expert_parallel_size=8,
            )
        )
    }
    manager._is_kernel_only_measurement_enabled_for_cluster = lambda _cluster: False
    model_tp8 = SimpleNamespace()
    model_tp4 = SimpleNamespace()
    manager._store_model_precision(
        "mlp_up_proj", "FP16", model_tp8, layer_contract=dense_tp8
    )
    manager._store_model_precision(
        "mlp_up_proj", "FP16", model_tp4, layer_contract=dense_tp4
    )

    selected = manager.get_models_for_cluster(ClusterType.PREFILL)
    assert selected["eager"]["mlp_up_proj"] is model_tp8

    manager._cluster_configs[ClusterType.PREFILL].replica_config.attn_tensor_parallel_size = 4
    selected = manager.get_models_for_cluster(ClusterType.PREFILL)
    assert selected["eager"]["mlp_up_proj"] is model_tp4


def test_cluster_model_view_preserves_untyped_memory_add_alias() -> None:
    """Cluster views must keep the legacy MEMORY ``add`` model reachable."""

    config = _step3_config()
    manager = _initialized_storage_manager()
    manager._cluster_configs = {
        ClusterType.PREFILL: SimpleNamespace(
            replica_config=SimpleNamespace(
                model_name="step3-moe-noquant",
                device="h200",
                model_config=config,
                attn_tensor_parallel_size=8,
                moe_tensor_parallel_size=1,
                moe_expert_parallel_size=8,
            )
        )
    }
    manager._is_kernel_only_measurement_enabled_for_cluster = lambda _cluster: False
    model = SimpleNamespace()
    manager._store_model_precision("add", "FP16", model)

    selected = manager.get_models_for_cluster(ClusterType.PREFILL)

    assert selected["eager"]["add"] is model


def test_legacy_untyped_model_storage_remains_compatible() -> None:
    """Legacy callers without a typed contract retain name-based retrieval."""

    manager = _initialized_storage_manager()
    model = SimpleNamespace()
    manager._store_model_precision("legacy_op", "FP16", model)

    assert manager.get_model("legacy_op") is model
    assert manager.get_model("legacy_op", precision="fp16") is model


def test_same_typed_contract_across_precisions_keeps_one_legacy_identity() -> None:
    """Changing precision must not manufacture a second, untyped contract."""

    dense, _ = _typed_contract_pair()
    manager = _initialized_storage_manager()
    fp16_model = SimpleNamespace()
    bf16_model = SimpleNamespace()

    manager._store_model_precision(
        "mlp_up_proj", "FP16", fp16_model, layer_contract=dense
    )
    manager._store_model_precision(
        "mlp_up_proj", "BF16", bf16_model, layer_contract=dense
    )

    identity = _serialize_selected_layer_cache_identity(dense)
    assert set(manager._trained_models_eager_by_contract) == {
        ("mlp_up_proj", identity)
    }
    assert "mlp_up_proj" not in {
        name
        for name, contract_identity in manager._trained_models_eager_by_contract
        if contract_identity is None
    }
    # The bare map is a legacy read source; the public view is projected from
    # the typed registry and still exposes the latest model for this domain.
    assert manager.get_models()["eager"]["mlp_up_proj"] is bf16_model
    assert manager.get_model("mlp_up_proj", precision="FP16", layer_contract=dense) is fp16_model
    assert manager.get_model("mlp_up_proj", precision="BF16", layer_contract=dense) is bf16_model
    assert manager.get_model("mlp_up_proj") is bf16_model


def test_different_typed_contracts_across_precisions_remain_ambiguous() -> None:
    """Different typed domains must stay explicit even when precisions differ."""

    dense, routed = _typed_contract_pair()
    manager = _initialized_storage_manager()
    dense_model = SimpleNamespace()
    routed_model = SimpleNamespace()

    manager._store_model_precision(
        "ffn_operator", "FP16", dense_model, layer_contract=dense
    )
    manager._store_model_precision(
        "ffn_operator", "BF16", routed_model, layer_contract=routed
    )

    identities = {
        identity
        for name, identity in manager._trained_models_eager_by_contract
        if name == "ffn_operator"
    }
    assert identities == {
        _serialize_selected_layer_cache_identity(dense),
        _serialize_selected_layer_cache_identity(routed),
    }
    assert ("ffn_operator", None) not in manager._trained_models_eager_by_contract
    with pytest.raises(ValueError, match="multiple layer contracts"):
        manager.get_model("ffn_operator")
    assert manager.get_model("ffn_operator", precision="FP16", layer_contract=dense) is dense_model
    assert manager.get_model("ffn_operator", precision="BF16", layer_contract=routed) is routed_model


def test_existing_untyped_model_remains_a_real_legacy_contract() -> None:
    """A pre-migration bare model remains readable without registry migration."""

    dense, _ = _typed_contract_pair()
    manager = _initialized_storage_manager()
    legacy_model = SimpleNamespace()
    typed_model = SimpleNamespace()
    manager._trained_models_eager["mlp_up_proj"] = legacy_model
    manager._models_by_precision_eager.setdefault("FP16", {})[
        "mlp_up_proj"
    ] = legacy_model

    manager._store_model_precision(
        "mlp_up_proj", "BF16", typed_model, layer_contract=dense
    )

    assert ("mlp_up_proj", None) not in manager._trained_models_eager_by_contract
    assert manager.get_model("mlp_up_proj", precision="FP16") is legacy_model
    assert manager.get_model("mlp_up_proj", layer_contract=dense) is typed_model
    with pytest.raises(ValueError, match="multiple layer contracts"):
        manager.get_model("mlp_up_proj")


def test_typed_precision_views_keep_distinct_contracts_across_precisions() -> None:
    """Precision-specific public lookup selects its unique typed identity."""

    dense, routed = _typed_contract_pair()
    manager = _initialized_storage_manager()
    dense_model = SimpleNamespace()
    routed_model = SimpleNamespace()
    manager._store_model_precision(
        "ffn_operator", "FP16", dense_model, layer_contract=dense
    )
    manager._store_model_precision(
        "ffn_operator", "BF16", routed_model, layer_contract=routed
    )

    assert manager.get_model("ffn_operator", precision="FP16") is dense_model
    assert manager.get_model("ffn_operator", precision="BF16") is routed_model


def test_typed_precision_views_keep_same_contract_in_every_precision_bucket() -> None:
    """A typed identity remains addressable after another precision is stored."""

    dense, _ = _typed_contract_pair()
    manager = _initialized_storage_manager()
    fp16_model = SimpleNamespace()
    bf16_model = SimpleNamespace()
    manager._store_model_precision(
        "mlp_up_proj", "FP16", fp16_model, layer_contract=dense
    )
    manager._store_model_precision(
        "mlp_up_proj", "BF16", bf16_model, layer_contract=dense
    )

    assert manager.get_model(
        "mlp_up_proj", precision="FP16", layer_contract=dense
    ) is fp16_model
    assert manager.get_model(
        "mlp_up_proj", precision="BF16", layer_contract=dense
    ) is bf16_model
    assert manager.get_models()["eager"]["mlp_up_proj"] is bf16_model


def test_cross_contract_write_keeps_precision_specific_typed_lookup() -> None:
    """An unrelated contract write leaves each explicit typed lookup intact."""

    dense, routed = _typed_contract_pair()
    manager = _initialized_storage_manager()
    dense_fp16 = SimpleNamespace(label="dense-fp16")
    dense_bf16 = SimpleNamespace(label="dense-bf16")
    routed_fp16 = SimpleNamespace(label="routed-fp16")
    manager._store_model_precision(
        "ffn_operator", "FP16", dense_fp16, layer_contract=dense
    )
    manager._store_model_precision(
        "ffn_operator", "BF16", dense_bf16, layer_contract=dense
    )
    manager._store_model_precision(
        "ffn_operator", "FP16", routed_fp16, layer_contract=routed
    )

    assert manager.get_model(
        "ffn_operator", precision="FP16", layer_contract=dense
    ) is dense_fp16
    assert manager.get_model(
        "ffn_operator", precision="FP16", layer_contract=routed
    ) is routed_fp16
    assert manager.get_model(
        "ffn_operator", precision="BF16", layer_contract=dense
    ) is dense_bf16
    with pytest.raises(ValueError, match="multiple layer contracts"):
        manager.get_model("ffn_operator", precision="FP16")


def test_legacy_write_after_typed_write_remains_explicitly_available() -> None:
    """A legacy caller must not lose its model when a typed variant exists."""

    dense, _ = _typed_contract_pair()
    manager = _initialized_storage_manager()
    typed_model = SimpleNamespace()
    legacy_model = SimpleNamespace()

    manager._store_model_precision(
        "mlp_up_proj", "FP16", typed_model, layer_contract=dense
    )
    manager._store_model_precision("mlp_up_proj", "BF16", legacy_model)

    assert manager._trained_models_eager_by_contract[
        ("mlp_up_proj", _serialize_selected_layer_cache_identity(dense))
    ] is typed_model
    assert manager._trained_models_eager_by_contract[("mlp_up_proj", None)] is legacy_model
    assert manager.get_model("mlp_up_proj", precision="BF16") is legacy_model
    with pytest.raises(ValueError, match="multiple layer contracts"):
        manager.get_model("mlp_up_proj")


def test_typed_storage_keeps_only_authoritative_and_legacy_projections() -> None:
    """Typed writes must not create private alias/provenance registries."""

    dense, _ = _typed_contract_pair()
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._all_dummy_mode = False
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._trained_models_eager = {}
    manager._trained_models_kernel_only = {}
    manager._models_by_precision_eager = {}
    manager._models_by_precision_kernel_only = {}
    manager._cluster_configs = {}

    model = SimpleNamespace()
    manager._store_model_precision(
        "mlp_up_proj", "FP16", model, layer_contract=dense
    )

    assert manager.get_model(
        "mlp_up_proj", precision="FP16", layer_contract=dense
    ) is model
    assert manager.get_models()["eager"]["mlp_up_proj"] is model
    assert not hasattr(manager, "_models_by_precision")
    assert not hasattr(manager, "_model_profiling_precision")
    assert not hasattr(manager, "_contract_last_write_precision_eager")
    assert not hasattr(manager, "_contract_last_write_precision_kernel_only")


def test_legacy_lowercase_precision_bucket_migrates_before_projection() -> None:
    """Legacy precision keys are normalized without losing the model."""

    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._all_dummy_mode = False
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._trained_models_eager = {}
    manager._trained_models_kernel_only = {}
    legacy_model = SimpleNamespace()
    manager._models_by_precision_eager = {"fp16": {"legacy_op": legacy_model}}
    manager._models_by_precision_kernel_only = {}

    manager._store_model_precision("legacy_op", "BF16", SimpleNamespace())

    assert manager.get_model("legacy_op", precision="FP16") is legacy_model


def test_legacy_lowercase_precision_bucket_is_read_without_a_new_write() -> None:
    """A historical lowercase bucket remains readable before any migration write."""

    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._all_dummy_mode = False
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._trained_models_eager = {}
    manager._trained_models_kernel_only = {}
    legacy_model = SimpleNamespace()
    manager._models_by_precision_eager = {"fp16": {"legacy_op": legacy_model}}
    manager._models_by_precision_kernel_only = {}

    assert manager.get_model("legacy_op", precision="FP16") is legacy_model

"""Minimal RED regressions for the profile-owned typed profiling contract."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.types import ClusterType, MeasurementType


class _LoaderProbePredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        return None


class _StandaloneMoELoaderProbePredictor(
    importlib.import_module(
        "frontier.execution_time_predictor.sklearn_moe_execution_time_predictor"
    ).SklearnMoEExecutionTimePredictor
):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        return None


def _step3_config() -> BaseModelConfig:
    return BaseModelConfig.create_from_name("step-moe-noquant")


def _require_symbol(module_name: str, symbol_name: str):
    """Turn a missing proposed boundary into an assertion failure, not collection error."""

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"{module_name}.{symbol_name} is required by the typed profiling contract"
        )
        raise AssertionError("unreachable") from exc
    symbol = getattr(module, symbol_name, None)
    assert callable(symbol), f"{module_name}.{symbol_name} must be callable"
    return symbol


def test_step3_resolver_separates_dense_routed_and_shared_domains() -> None:
    """A mixed Step3 model resolves width and TP/EP from the layer family."""

    from frontier.model_architectures import LayerKind

    config = _step3_config()
    profile = config.get_model_architecture_profile()
    resolver = getattr(profile, "resolve_layer_contract", None)
    assert callable(resolver), "ModelArchitectureProfile must expose resolve_layer_contract"

    dense = resolver(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    routed = resolver(
        config,
        layer_id=4,
        operator_name="moe_grouped_gemm",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    shared = resolver(
        config,
        layer_id=4,
        layer_kind=LayerKind.SHARED,
        operator_name="share_expert_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )

    assert (dense.layer_kind.value, dense.effective_ffn_width) == ("dense", 18432)
    assert dense.tensor_parallel_mode.value == "attention_tp"
    assert dense.expert_parallel_mode.value == "off"
    assert (routed.layer_kind.value, routed.effective_ffn_width) == ("routed", 5120)
    assert routed.tensor_parallel_mode.value == "moe_tp"
    assert routed.expert_parallel_mode.value == "on"
    assert (shared.layer_kind.value, shared.effective_ffn_width) == ("shared", 5120)
    assert shared.tensor_parallel_mode.value == "attention_tp"
    assert shared.expert_parallel_mode.value == "off"


def test_routed_resolver_uses_registry_owned_ep_semantics() -> None:
    """Routing work is EP-agnostic while grouped GEMM keeps exact EP."""

    config = _step3_config()
    profile = config.get_model_architecture_profile()

    gating = profile.resolve_layer_contract(
        config,
        layer_id=4,
        operator_name="moe_gating_linear",
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    grouped_gemm = profile.resolve_layer_contract(
        config,
        layer_id=4,
        operator_name="moe_grouped_gemm",
        moe_tp_size=1,
        expert_parallel_size=8,
    )

    assert gating.expert_parallel_size is None
    assert gating.typed_metadata_identity()["selected_expert_parallel_size"] is None
    assert grouped_gemm.expert_parallel_size == 8
    assert grouped_gemm.typed_metadata_identity()["selected_expert_parallel_size"] == 8


def test_moe_plan_uses_routed_width_for_is_moe_placeholder() -> None:
    """MoE linear profiling does not allocate a mixed model's dense-width MLP."""

    config = _step3_config()
    moe_plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[8],
        is_moe=True,
    )
    dense_plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[8],
        is_moe=False,
    )

    assert moe_plan["padded_n_expanded_embd"] == 5120
    assert dense_plan["padded_n_expanded_embd"] == 18432


def test_profiling_plan_emits_typed_contracts_with_independent_moe_tp_domain() -> None:
    """The producer carries dense and routed domains in one canonical plan."""

    config = _step3_config()
    signature = inspect.signature(build_profiling_plan)
    assert "moe_tp" in signature.parameters, "plan must accept an explicit routed TP domain"

    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=True,
    )
    contracts = {item["layer_kind"]: item for item in plan["typed_layer_contracts"]}

    assert contracts["dense"]["effective_ffn_width"] == 18432
    assert contracts["dense"]["tensor_parallel_mode"] == "attention_tp"
    assert contracts["dense"]["selected_tensor_parallel_size"] == 8
    assert contracts["routed"]["effective_ffn_width"] == 5120
    assert contracts["routed"]["tensor_parallel_mode"] == "moe_tp"
    assert contracts["routed"]["tensor_parallel_sizes"] == [1]
    assert contracts["shared"]["effective_ffn_width"] == 5120
    assert contracts["shared"]["selected_tensor_parallel_size"] == 8


def test_linear_plan_declares_routed_domain_without_emitting_moe_contracts() -> None:
    """Linear profiling leaves routed operator metadata to the MoE producer."""

    from frontier.operators.families import MOE_FAMILY

    config = _step3_config()
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=True,
    )

    layer_contracts = {
        item["layer_kind"]: item for item in plan["typed_layer_contracts"]
    }
    assert layer_contracts["routed"]["tensor_parallel_sizes"] == [1]

    routed_names = {
        operator.profiling_name() for operator in MOE_FAMILY.profiling_ops()
    }
    assert routed_names.isdisjoint(plan["typed_operator_contracts"])
    assert set(plan["typed_operator_contracts"]).issubset(plan["enabled_ops"])


def test_typed_metadata_rejects_wrong_family_tp_and_padded_width() -> None:
    """Admission rejects each semantic field mismatch against one resolved contract."""

    from frontier.model_architectures import LayerKind

    config = _step3_config()
    profile = config.get_model_architecture_profile()
    resolver = getattr(profile, "resolve_layer_contract", None)
    assert callable(resolver), "typed metadata test requires the profile resolver"
    contract = resolver(
        config,
        layer_id=0,
        layer_kind=LayerKind.DENSE,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    metadata = contract.typed_metadata_identity()
    validator = _require_symbol(
        "frontier.operators.typed_contracts", "validate_typed_operator_metadata"
    )
    expected = dict(metadata)

    for field_name, wrong_value in (
        ("operator_family_id", "moe"),
        ("tensor_parallel_sizes", [1]),
        ("selected_padded_ffn_width", 9999),
    ):
        candidate = dict(metadata)
        candidate[field_name] = wrong_value
        with pytest.raises(ValueError, match=field_name):
            validator(
                candidate,
                operator_name="mlp_up_proj",
                expected_metadata=expected,
            )


def test_legacy_csv_without_typed_column_uses_explicit_scalar_path(tmp_path) -> None:
    """A legacy row without the typed column remains loadable through scalar fields."""

    config = _step3_config()
    predictor = object.__new__(_LoaderProbePredictor)
    predictor._model_config = config
    predictor._replica_config = SimpleNamespace(attn_tensor_parallel_size=8)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._get_compute_model_names = lambda: ["mlp_up_proj"]
    predictor._get_profiling_metadata = lambda _df, _path: SimpleNamespace(
        profiling_precision=config.get_default_precision(),
        quant_signature=config.get_quant_signature(),
        model_arch=config.model_arch,
        model_architecture_profile=config.get_model_architecture_profile().profile_id,
        measurement_type=MeasurementType.CUDA_EVENT,
    )
    predictor._validate_active_measurement_type = lambda *_args: None
    predictor._register_profiling_metadata_for_ops = lambda *_args: None

    row = {
        "n_head": config.num_q_heads,
        "n_kv_head": config.num_kv_heads,
        "n_embd": config.embedding_dim,
        "n_expanded_embd": config.mlp_hidden_dim,
        "use_gated_mlp": config.use_gated_mlp,
        "vocab_size": config.vocab_size,
        "num_tensor_parallel_workers": 8,
        "profiling_precision": "BF16",
        "model_arch": config.model_arch,
        "model_architecture_profile": config.get_model_architecture_profile().profile_id,
        "quant_signature": config.get_quant_signature(),
        "measurement_type": MeasurementType.CUDA_EVENT.value,
        "time_stats.mlp_up_proj.median": 1.0,
    }
    path = tmp_path / "legacy-linear.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    signature = inspect.signature(predictor._load_compute_df)
    assert "layer_contract" in signature.parameters
    loaded = predictor._load_compute_df(
        str(path),
        tensor_parallel_size=8,
        operator_name="mlp_up_proj",
    )

    assert loaded["n_expanded_embd"].tolist() == [config.mlp_hidden_dim]
    assert "typed_operator_contracts" not in loaded.columns


def test_sklearn_typed_loader_admits_dense_width_before_legacy_scalar_filter(
    tmp_path,
) -> None:
    """Typed mixed-model rows reach contract admission before legacy width filtering."""

    from frontier.model_architectures import LayerKind
    from frontier.operators.typed_contracts import serialize_typed_operator_contracts

    config = _step3_config()
    dense_contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=0,
        layer_kind=LayerKind.DENSE,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    dense_metadata = dense_contract.typed_metadata_identity()
    common = {
        "n_head": config.num_q_heads,
        "n_kv_head": config.num_kv_heads,
        "n_embd": config.embedding_dim,
        "use_gated_mlp": config.use_gated_mlp,
        "vocab_size": config.vocab_size,
        "num_tensor_parallel_workers": 8,
        "profiling_precision": "BF16",
        "model_arch": config.model_arch,
        "model_architecture_profile": config.get_model_architecture_profile().profile_id,
        "quant_signature": config.get_quant_signature(),
        "measurement_type": MeasurementType.CUDA_EVENT.value,
    }
    path = tmp_path / "typed-mixed-linear.csv"
    pd.DataFrame(
        [
            {
                **common,
                "n_expanded_embd": 18432,
                "typed_operator_contracts": serialize_typed_operator_contracts(
                    {"mlp_up_proj": dense_metadata}
                ),
                "time_stats.mlp_up_proj.median": 1.0,
            },
            {
                **common,
                "n_expanded_embd": 5120,
                "typed_operator_contracts": serialize_typed_operator_contracts(
                    {
                        "mlp_up_proj": {
                            **dense_metadata,
                            "effective_ffn_width": 5120,
                            "selected_padded_ffn_width": 5120,
                        }
                    }
                ),
                "time_stats.mlp_up_proj.median": 2.0,
            },
        ]
    ).to_csv(path, index=False)

    predictor = object.__new__(_LoaderProbePredictor)
    predictor._model_config = config
    predictor._replica_config = SimpleNamespace(attn_tensor_parallel_size=8)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._get_compute_model_names = lambda: ["mlp_up_proj"]
    predictor._get_profiling_metadata = lambda _df, _path: SimpleNamespace(
        profiling_precision=config.get_default_precision(),
        quant_signature=config.get_quant_signature(),
        model_arch=config.model_arch,
        model_architecture_profile=config.get_model_architecture_profile().profile_id,
        measurement_type=MeasurementType.CUDA_EVENT,
    )
    predictor._validate_active_measurement_type = lambda *_args: None
    predictor._register_profiling_metadata_for_ops = lambda *_args: None

    loaded = predictor._load_compute_df(
        str(path),
        tensor_parallel_size=8,
        operator_name="mlp_up_proj",
        layer_contract=dense_contract,
    )

    assert loaded["n_expanded_embd"].tolist() == [18432]
    assert loaded["time_stats.mlp_up_proj.median"].tolist() == [1.0]


def test_sklearn_typed_loader_rejects_malformed_attention_metadata(tmp_path) -> None:
    """Typed rows are schema-validated even when attention has no layer contract."""

    from frontier.operators.typed_contracts import serialize_typed_operator_contracts

    config = _step3_config()
    input_file = tmp_path / "malformed-sklearn-attention.csv"
    pd.DataFrame(
        [
            {
                "n_head": config.num_q_heads,
                "n_kv_head": config.num_kv_heads,
                "n_embd": config.embedding_dim,
                "n_expanded_embd": config.mlp_hidden_dim,
                "use_gated_mlp": config.use_gated_mlp,
                "vocab_size": config.vocab_size,
                "num_tensor_parallel_workers": 8,
                "profiling_precision": "BF16",
                "model_arch": config.model_arch,
                "model_architecture_profile": config.get_model_architecture_profile().profile_id,
                "quant_signature": config.get_quant_signature(),
                "measurement_type": MeasurementType.CUDA_EVENT.value,
                "typed_operator_contracts": serialize_typed_operator_contracts(
                    {"attn_pre_proj": {}}
                ),
                "time_stats.attn_pre_proj.median": 1.0,
            }
        ]
    ).to_csv(input_file, index=False)

    predictor = object.__new__(_LoaderProbePredictor)
    predictor._model_config = config
    predictor._replica_config = SimpleNamespace(attn_tensor_parallel_size=8)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._get_compute_model_names = lambda: ["attn_pre_proj"]
    predictor._get_profiling_metadata = lambda _df, _path: SimpleNamespace(
        profiling_precision=config.get_default_precision(),
        quant_signature=config.get_quant_signature(),
        model_arch=config.model_arch,
        model_architecture_profile=config.get_model_architecture_profile().profile_id,
        measurement_type=MeasurementType.CUDA_EVENT,
    )
    predictor._validate_active_measurement_type = lambda *_args: None
    predictor._register_profiling_metadata_for_ops = lambda *_args: None

    with pytest.raises(ValueError, match="missing required fields"):
        predictor._load_compute_df(
            str(input_file),
            tensor_parallel_size=8,
            operator_name="attn_pre_proj",
        )


def test_decode_attn_zero_domain_skips_ffn_contract_resolution() -> None:
    """An attention-only role returns no FFN contracts when its role fields are zero."""

    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    resolver = getattr(manager, "_resolve_ffn_layer_contracts", None)
    signature = getattr(manager, "_get_ffn_contract_signature", None)
    assert callable(resolver), "manager must expose the FFN contract boundary"
    assert callable(signature), "manager must expose the FFN contract signature boundary"

    config = _step3_config()
    replica_config = SimpleNamespace(
        model_config=config,
        attn_tensor_parallel_size=0,
        moe_tensor_parallel_size=0,
        moe_expert_parallel_size=0,
    )
    assert resolver(
        ClusterType.DECODE_ATTN,
        replica_config,
        is_moe_model=True,
    ) == ()
    assert signature(
        ClusterType.DECODE_ATTN,
        replica_config,
        is_moe_model=True,
    ) == "none"


def test_mixed_profile_requires_explicit_routed_width() -> None:
    """Mixed MoE profiles do not collapse routed width into the legacy scalar."""

    config = _step3_config()
    config.routed_mlp_hidden_dim = None
    profile = config.get_model_architecture_profile()

    with pytest.raises(ValueError, match="routed_mlp_hidden_dim"):
        profile.resolve_layer_contract(
            config,
            layer_id=4,
            operator_name="moe_grouped_gemm",
            attention_tp_size=8,
            moe_tp_size=1,
            expert_parallel_size=8,
        )


def test_metadata_reader_propagates_unexpected_value_error(monkeypatch) -> None:
    """Malformed metadata read errors remain visible to the caller."""

    predictor = object.__new__(_LoaderProbePredictor)
    predictor._register_missing_profiling_metadata = MagicMock()

    def _raise_value_error(_path):
        raise ValueError("malformed profiling CSV")

    monkeypatch.setattr(pd, "read_csv", _raise_value_error)

    with pytest.raises(ValueError, match="malformed profiling CSV"):
        predictor._register_profiling_metadata_from_file(
            "malformed.csv",
            ["mlp_up_proj"],
        )
    predictor._register_missing_profiling_metadata.assert_not_called()


def test_selected_cache_identity_ignores_physical_layer_occurrence() -> None:
    """Equivalent semantic contracts share a cache identity across layer IDs."""

    from frontier.model_architectures import LayerKind

    config = _step3_config()
    profile = config.get_model_architecture_profile()
    resolver = getattr(profile, "resolve_layer_contract", None)
    assert callable(resolver), "cache identity test requires the profile resolver"
    layer_zero = resolver(
        config,
        layer_id=0,
        layer_kind=LayerKind.DENSE,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    layer_one = resolver(
        config,
        layer_id=1,
        layer_kind=LayerKind.DENSE,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    serializer = _require_symbol(
        "frontier.execution_time_predictor.shared_prediction_model_manager",
        "_serialize_selected_layer_cache_identity",
    )
    assert serializer(layer_zero) == serializer(layer_one)


def test_selected_cache_identity_respects_operator_ep_semantics() -> None:
    """EP-agnostic routing shares identity while grouped GEMM remains EP-specific."""

    from frontier.model_architectures import LayerKind

    config = _step3_config()
    profile = config.get_model_architecture_profile()
    serializer = _require_symbol(
        "frontier.execution_time_predictor.shared_prediction_model_manager",
        "_serialize_selected_layer_cache_identity",
    )

    def resolve(operator_name: str, expert_parallel_size: int):
        return profile.resolve_layer_contract(
            config,
            layer_id=4,
            layer_kind=LayerKind.ROUTED,
            operator_name=operator_name,
            moe_tp_size=1,
            expert_parallel_size=expert_parallel_size,
        )

    for operator_name in (
        "moe_gating_linear",
        "moe_gating_routing_topk",
        "moe_shuffling",
    ):
        assert serializer(resolve(operator_name, 1)) == serializer(
            resolve(operator_name, 8)
        )

    assert serializer(resolve("moe_grouped_gemm", 1)) != serializer(
        resolve("moe_grouped_gemm", 8)
    )


def test_linear_wrapper_result_carries_plan_typed_operator_contracts() -> None:
    """The real producer row keeps profile-owned metadata beside timings."""

    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper

    config = ModelConfig.from_model_name("step-moe-noquant")
    typed_contracts = {
        "mlp_up_proj": {
            "profile_id": "step3_text",
            "operator_family_id": "ffn",
            "operator_family_ids": ["ffn"],
            "layer_kind": "dense",
            "dimension_source": "dense_mlp_hidden_dim",
            "effective_ffn_width": 18432,
            "tensor_parallel_mode": "attention_tp",
            "expert_parallel_mode": "off",
            "selected_expert_parallel_size": None,
            "tensor_parallel_sizes": [8],
            "selected_tensor_parallel_size": 8,
            "selected_padded_ffn_width": 18432,
        }
    }
    wrapper = object.__new__(LinearOpWrapper)
    wrapper.model_config = config
    wrapper.num_tensor_parallel_workers = 8
    wrapper.profiling_plan = {
        "padded_n_embd": config.embedding_dim,
        "padded_n_expanded_embd": 18432,
        "typed_operator_contracts": typed_contracts,
    }

    result = wrapper._build_profile_result(  # pylint: disable=protected-access
        {"mlp_up_proj": {"mean": 1.0}},
        num_tokens=2,
    )

    assert result["typed_operator_contracts"] == typed_contracts
    assert result["typed_operator_contracts"] is not typed_contracts


def test_linear_wrapper_result_omits_unmeasured_typed_operator_contracts() -> None:
    """A result row declares typed contracts only for measured operators."""

    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper

    config = ModelConfig.from_model_name("step-moe-noquant")
    wrapper = object.__new__(LinearOpWrapper)
    wrapper.model_config = config
    wrapper.num_tensor_parallel_workers = 1
    wrapper.profiling_plan = {
        "padded_n_embd": config.embedding_dim,
        "padded_n_expanded_embd": config.mlp_hidden_dim,
        "typed_operator_contracts": {
            "mlp_up_proj": {"operator_family_id": "ffn"},
            "moe_grouped_gemm": {"operator_family_id": "moe"},
        },
    }

    result = wrapper._build_profile_result(  # pylint: disable=protected-access
        {"mlp_up_proj": {"mean": 1.0}},
        num_tokens=2,
    )

    assert set(result["typed_operator_contracts"]) == {"mlp_up_proj"}


def test_replicated_split_partitions_typed_operator_contracts() -> None:
    """Each split row carries metadata only for operators it times."""

    from frontier.profiling.utils.replicated_ops import split_replicated_result

    result = {
        "num_tokens": 2,
        "num_tensor_parallel_workers": 8,
        "time_stats": {
            "input_layernorm": {"mean": 1.0},
            "mlp_up_proj": {"mean": 2.0},
        },
        "typed_operator_contracts": {
            "input_layernorm": {"operator_family_id": "memory"},
            "mlp_up_proj": {"operator_family_id": "ffn"},
        },
    }

    sharded, replicated = split_replicated_result(
        result,
        {"input_layernorm"},
        unpadded_n_embd=7168,
        unpadded_n_expanded_embd=18432,
    )

    assert set(sharded["typed_operator_contracts"]) == {"mlp_up_proj"}
    assert set(replicated["typed_operator_contracts"]) == {"input_layernorm"}
    assert replicated["num_tensor_parallel_workers"] == 1


def test_linear_output_boundary_serializes_typed_contracts_as_canonical_json() -> None:
    """CSV output uses the shared deterministic typed-contract serializer."""

    from frontier.profiling.linear_op import main as linear_op_main
    from frontier.operators.typed_contracts import parse_typed_operator_contracts

    metadata = {
        "profile_id": "step3_text",
        "operator_family_id": "memory",
        "operator_family_ids": ["memory"],
        "layer_kind": None,
        "dimension_source": None,
        "effective_ffn_width": None,
        "tensor_parallel_mode": "replicated",
        "expert_parallel_mode": "off",
        "selected_expert_parallel_size": None,
        "tensor_parallel_sizes": [1],
        "selected_tensor_parallel_size": 1,
        "selected_padded_ffn_width": None,
    }
    frame = pd.DataFrame({"typed_operator_contracts": [{"input_layernorm": metadata}]})

    output = linear_op_main._serialize_linear_op_output(frame)  # pylint: disable=protected-access

    assert isinstance(output.loc[0, "typed_operator_contracts"], str)
    assert parse_typed_operator_contracts(output.loc[0, "typed_operator_contracts"])[
        "input_layernorm"
    ] == metadata


def test_linear_tp_ranges_keep_moe_domain_independent() -> None:
    """The producer CLI preserves an explicitly supplied routed TP domain."""

    from frontier.profiling.linear_op import main as linear_op_main

    args = SimpleNamespace(
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        num_tensor_parallel_workers=[8],
    )

    assert linear_op_main._resolve_tp_ranges(args) == ([8], [8], [1], [1, 8])  # pylint: disable=protected-access


def test_real_linear_producer_csv_round_trip_reaches_manager_loader(tmp_path) -> None:
    """A plan-built producer row survives CSV serialization and typed admission."""

    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.linear_op import main as linear_op_main
    from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper

    config = ModelConfig.from_model_name("step-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=False,
    )
    wrapper = object.__new__(LinearOpWrapper)
    wrapper.model_config = config
    wrapper.num_tensor_parallel_workers = 8
    wrapper.profiling_plan = plan
    result = wrapper._build_profile_result(  # pylint: disable=protected-access
        {
            "mlp_up_proj": {"median": 1.0},
            "input_layernorm": {"median": 2.0},
        },
        num_tokens=2,
    )
    source = pd.DataFrame([result])
    frame = (
        pd.json_normalize(source["time_stats"])
        .add_prefix("time_stats.")
        .join(source.drop(columns=["time_stats"]))
    )
    path = tmp_path / "linear-op.csv"
    linear_op_main._serialize_linear_op_output(frame).to_csv(path, index=False)  # pylint: disable=protected-access

    contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)
    loaded = manager._load_linear_op_df(
        str(path),
        8,
        layer_contract=contract,
        operator_name="mlp_up_proj",
    )

    assert loaded["time_stats.mlp_up_proj.median"].tolist() == [1.0]


def test_manager_linear_loader_uses_selected_dense_typed_width(tmp_path) -> None:
    """Manager linear admission selects the dense width in a mixed model."""

    from frontier.model_architectures import LayerKind
    from frontier.operators.typed_contracts import serialize_typed_operator_contracts

    config = _step3_config()
    contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=0,
        layer_kind=LayerKind.DENSE,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    rows = [
        {
            "num_tensor_parallel_workers": 8,
            "n_expanded_embd": 18432,
            "typed_operator_contracts": serialize_typed_operator_contracts(
                {"mlp_up_proj": contract.typed_metadata_identity()}
            ),
            "time_stats.mlp_up_proj.median": 1.0,
        },
        {
            "num_tensor_parallel_workers": 8,
            "n_expanded_embd": 5120,
            "typed_operator_contracts": serialize_typed_operator_contracts(
                {
                    "mlp_up_proj": {
                        **contract.typed_metadata_identity(),
                        "effective_ffn_width": 5120,
                        "selected_padded_ffn_width": 5120,
                    }
                }
            ),
            "time_stats.mlp_up_proj.median": 2.0,
        },
    ]
    input_file = tmp_path / "typed-linear.csv"
    pd.DataFrame(rows).to_csv(input_file, index=False)

    manager = object.__new__(ExecutionTimePredictionModelManager)
    filtered = manager._load_linear_op_df(
        str(input_file),
        8,
        layer_contract=contract,
        operator_name="mlp_up_proj",
    )

    assert filtered["n_expanded_embd"].tolist() == [18432]
    assert filtered["time_stats.mlp_up_proj.median"].tolist() == [1.0]


def test_manager_moe_loader_uses_selected_routed_width_and_ep(tmp_path) -> None:
    """Manager MoE admission selects routed width and expert-parallel domain."""

    from frontier.model_architectures import LayerKind
    from frontier.operators.typed_contracts import serialize_typed_operator_contracts

    config = _step3_config()
    contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=4,
        layer_kind=LayerKind.ROUTED,
        operator_name="moe_grouped_gemm",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    metadata = contract.typed_metadata_identity()
    rows = [
        {
            "num_experts": config.num_experts,
            "router_topk": config.num_experts_per_tok,
            "hidden_dim": config.embedding_dim,
            "expert_hidden_dim": 5120,
            "num_tensor_parallel_workers": 1,
            "expert_parallel_size": 8,
            "typed_operator_contracts": serialize_typed_operator_contracts(
                {"moe_grouped_gemm": metadata}
            ),
            "num_tokens": 1,
            "time_stats.moe_grouped_gemm.median": 1.0,
        },
        {
            "num_experts": config.num_experts,
            "router_topk": config.num_experts_per_tok,
            "hidden_dim": config.embedding_dim,
            "expert_hidden_dim": 5120,
            "num_tensor_parallel_workers": 1,
            "expert_parallel_size": 4,
            "typed_operator_contracts": serialize_typed_operator_contracts(
                {
                    "moe_grouped_gemm": {
                        **metadata,
                        "selected_expert_parallel_size": 4,
                    }
                }
            ),
            "num_tokens": 1,
            "time_stats.moe_grouped_gemm.median": 2.0,
        },
    ]
    input_file = tmp_path / "typed-moe.csv"
    pd.DataFrame(rows).to_csv(input_file, index=False)

    manager = object.__new__(ExecutionTimePredictionModelManager)
    replica_config = SimpleNamespace(
        model_config=config,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    filtered = manager._load_moe_df(
        str(input_file),
        replica_config,
        load_imbalance=False,
        tensor_parallel_size=1,
        expert_parallel_size=8,
        layer_contract=contract,
        operator_name="moe_grouped_gemm",
    )

    assert filtered["expert_parallel_size"].tolist() == [8]
    assert filtered["time_stats.moe_grouped_gemm.median"].tolist() == [1.0]


@pytest.mark.parametrize(
    "operator_name",
    [
        "moe_gating_linear",
        "moe_gating_routing_topk",
        "moe_shuffling",
    ],
)
def test_manager_moe_loader_keeps_ep_agnostic_typed_rows(
    tmp_path, operator_name: str
) -> None:
    """Typed EP-agnostic operators admit rows profiled at multiple EP sizes."""

    from frontier.operators.typed_contracts import serialize_typed_operator_contracts
    from frontier.model_architectures import LayerKind

    config = _step3_config()
    profile = config.get_model_architecture_profile()
    rows = []
    for ep_size, target in ((1, 1.0), (8, 8.0)):
        contract = profile.resolve_layer_contract(
            config,
            layer_id=4,
            layer_kind=LayerKind.ROUTED,
            operator_name=operator_name,
            moe_tp_size=1,
            expert_parallel_size=ep_size,
        )
        rows.append(
            {
                "num_experts": config.num_experts,
                "router_topk": config.num_experts_per_tok,
                "hidden_dim": config.embedding_dim,
                "expert_hidden_dim": 5120,
                "num_tensor_parallel_workers": 1,
                "expert_parallel_size": ep_size,
                "num_tokens": 1,
                f"time_stats.{operator_name}.median": target,
                "typed_operator_contracts": serialize_typed_operator_contracts(
                    {operator_name: contract.typed_metadata_identity()}
                ),
            }
        )

    input_file = tmp_path / f"typed-{operator_name}.csv"
    pd.DataFrame(rows).to_csv(input_file, index=False)
    runtime_contract = profile.resolve_layer_contract(
        config,
        layer_id=4,
        layer_kind=LayerKind.ROUTED,
        operator_name=operator_name,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)
    replica_config = SimpleNamespace(
        model_config=config,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )

    filtered = manager._load_moe_df(
        str(input_file),
        replica_config,
        load_imbalance=False,
        tensor_parallel_size=1,
        expert_parallel_size=None,
        layer_contract=runtime_contract,
        operator_name=operator_name,
    )

    assert filtered["expert_parallel_size"].tolist() == [1, 8]
    assert filtered[f"time_stats.{operator_name}.median"].tolist() == [1.0, 8.0]


def test_moe_wrapper_typed_result_round_trips_through_manager(tmp_path) -> None:
    """The existing MoE producer emits a routed contract that the manager admits."""

    from frontier.operators.typed_contracts import (
        serialize_typed_operator_contract_column,
    )
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.moe.moe_wrapper import MoEWrapper

    config = ModelConfig.from_model_name("step-moe-noquant")
    wrapper = object.__new__(MoEWrapper)
    wrapper.model_config = config
    wrapper.num_tensor_parallel_workers = 1
    wrapper.expert_parallel_size = 8
    wrapper.num_experts = config.num_experts
    wrapper.num_experts_per_device = config.num_experts // wrapper.expert_parallel_size
    wrapper.router_topk = config.num_experts_per_tok
    wrapper.hidden_dim = config.embedding_dim
    wrapper.expert_hidden_dim = config.routed_mlp_hidden_dim
    wrapper.use_gated = config.use_gated_mlp
    wrapper.routing_runtime_metadata = {}
    wrapper.gating_runtime_context_metadata = {}

    result = wrapper._build_profile_result(  # pylint: disable=protected-access
        {
            "moe_gating_linear": {"mean": 0.1},
            "moe_gating_routing_topk": {"mean": 0.2},
            "moe_shuffling": {"mean": 0.3},
            "moe_grouped_gemm": {"mean": 0.4},
        },
        num_tokens=1,
    )
    frame = pd.DataFrame([result])
    frame = (
        pd.json_normalize(frame["time_stats"])
        .add_prefix("time_stats.")
        .join(frame.drop(columns=["time_stats"]))
    )
    frame = serialize_typed_operator_contract_column(frame)
    input_file = tmp_path / "moe-round-trip.csv"
    frame.to_csv(input_file, index=False)

    config_contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=4,
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

    loaded = manager._load_moe_df(
        str(input_file),
        replica_config,
        load_imbalance=False,
        tensor_parallel_size=1,
        expert_parallel_size=8,
        layer_contract=config_contract,
        operator_name="moe_grouped_gemm",
    )

    assert loaded["expert_parallel_size"].tolist() == [8]
    from frontier.operators.typed_contracts import parse_typed_operator_contracts

    metadata = parse_typed_operator_contracts(
        loaded.iloc[0]["typed_operator_contracts"]
    )
    assert metadata["moe_grouped_gemm"]["selected_expert_parallel_size"] == 8


def test_manager_typed_loader_requires_operator_scope(tmp_path) -> None:
    """Typed manager admission fails when the target operator is unspecified."""

    from frontier.model_architectures import LayerKind
    from frontier.operators.typed_contracts import serialize_typed_operator_contracts

    config = _step3_config()
    contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=0,
        layer_kind=LayerKind.DENSE,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    input_file = tmp_path / "typed-scope.csv"
    pd.DataFrame(
        [
            {
                "num_tensor_parallel_workers": 8,
                "n_expanded_embd": 18432,
                "typed_operator_contracts": serialize_typed_operator_contracts(
                    {"mlp_up_proj": contract.typed_metadata_identity()}
                ),
                "time_stats.mlp_up_proj.median": 1.0,
            }
        ]
    ).to_csv(input_file, index=False)

    manager = object.__new__(ExecutionTimePredictionModelManager)
    with pytest.raises(ValueError, match="operator_name"):
        manager._load_linear_op_df(
            str(input_file),
            8,
            layer_contract=contract,
        )


def test_manager_linear_loader_allows_unscoped_typed_rows_for_attention_paths(tmp_path) -> None:
    """Generic attention loading validates typed JSON without selecting an FFN domain."""

    from frontier.operators.typed_contracts import serialize_typed_operator_contracts

    config = _step3_config()
    input_file = tmp_path / "typed-attention.csv"
    pd.DataFrame(
        [
            {
                "num_tensor_parallel_workers": 8,
                "n_head": config.num_q_heads,
                "n_kv_head": config.num_kv_heads,
                "n_embd": config.embedding_dim,
                "vocab_size": config.vocab_size,
                "use_gated_mlp": config.use_gated_mlp,
                "typed_operator_contracts": serialize_typed_operator_contracts(
                    {
                        "attn_pre_proj": {
                            "profile_id": "step3_text",
                            "operator_family_id": "dense_attention",
                            "operator_family_ids": ["dense_attention"],
                            "layer_kind": None,
                            "dimension_source": None,
                            "effective_ffn_width": None,
                            "tensor_parallel_mode": "attention_tp",
                            "expert_parallel_mode": "off",
                            "selected_expert_parallel_size": None,
                            "tensor_parallel_sizes": [8],
                            "selected_tensor_parallel_size": 8,
                            "selected_padded_ffn_width": None,
                        }
                    }
                ),
                "time_stats.attn_pre_proj.median": 1.0,
            }
        ]
    ).to_csv(input_file, index=False)

    manager = object.__new__(ExecutionTimePredictionModelManager)
    loaded = manager._load_linear_op_df(str(input_file), 8)

    assert loaded["time_stats.attn_pre_proj.median"].tolist() == [1.0]


def test_manager_unscoped_typed_loader_rejects_malformed_metadata(tmp_path) -> None:
    """Unscoped typed rows still receive complete canonical schema validation."""

    from frontier.operators.typed_contracts import serialize_typed_operator_contracts

    config = _step3_config()
    input_file = tmp_path / "malformed-typed-attention.csv"
    pd.DataFrame(
        [
            {
                "num_tensor_parallel_workers": 8,
                "n_head": config.num_q_heads,
                "n_kv_head": config.num_kv_heads,
                "n_embd": config.embedding_dim,
                "vocab_size": config.vocab_size,
                "use_gated_mlp": config.use_gated_mlp,
                "typed_operator_contracts": serialize_typed_operator_contracts(
                    {"post_attention_layernorm": {}}
                ),
                "time_stats.post_attention_layernorm.median": 1.0,
            }
        ]
    ).to_csv(input_file, index=False)

    manager = object.__new__(ExecutionTimePredictionModelManager)
    with pytest.raises(ValueError, match="missing required fields"):
        manager._load_linear_op_df(str(input_file), 8)


def test_standalone_moe_predictor_rejects_malformed_typed_metadata(
    tmp_path, monkeypatch
) -> None:
    """Independent MoE predictor training validates typed rows before filtering."""

    import frontier.execution_time_predictor.sklearn_moe_execution_time_predictor as moe_module
    from frontier.operators.typed_contracts import serialize_typed_operator_contracts

    input_file = tmp_path / "malformed-standalone-moe.csv"
    pd.DataFrame(
        [
            {
                "num_tensor_parallel_workers": 1,
                "expert_parallel_size": 1,
                "typed_operator_contracts": serialize_typed_operator_contracts(
                    {"moe_gating_linear": {}}
                ),
            }
        ]
    ).to_csv(input_file, index=False)

    predictor = object.__new__(_StandaloneMoELoaderProbePredictor)
    predictor._moe_input_file = str(input_file)
    predictor._replica_config = SimpleNamespace(
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    )
    predictor._model_config = SimpleNamespace(is_moe=True)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._get_profiling_metadata = lambda *_args: SimpleNamespace(
        measurement_type=MeasurementType.CUDA_EVENT,
    )
    predictor._validate_active_measurement_type = lambda *_args: None
    predictor._validate_moe_dataset_contract = lambda dataframe, *_args: dataframe
    predictor._register_profiling_metadata_for_ops = lambda *_args: None
    predictor._get_requested_moe_gating_routing_runtime_path = lambda: (
        "standard_fused_topk"
    )
    monkeypatch.setattr(moe_module, "_get_moe_family_model_names", lambda: [])
    monkeypatch.setattr(
        moe_module,
        "should_enable_prefill_hot_moe_gating_contract",
        lambda **_kwargs: False,
    )

    with pytest.raises(ValueError, match="missing required fields"):
        predictor._train_moe_models()


def test_moe_trainer_rejects_malformed_typed_metadata_before_scalar_filtering(
    tmp_path,
) -> None:
    """Standalone MoE trainer validates typed rows at dataset admission."""

    from frontier.training.moe_trainer import MoETrainer, _get_moe_family_model_names
    from frontier.operators.typed_contracts import serialize_typed_operator_contracts

    input_file = tmp_path / "malformed-trainer-moe.csv"
    row = {
        "num_experts": 8,
        "router_topk": 2,
        "hidden_dim": 4096,
        "expert_hidden_dim": 5120,
        "num_tensor_parallel_workers": 1,
        "expert_parallel_size": 1,
        "num_tokens": 1,
        "profiling_precision": "BF16",
        "measurement_type": MeasurementType.CUDA_EVENT.value,
        "typed_operator_contracts": serialize_typed_operator_contracts(
            {"moe_gating_linear": {}}
        ),
    }
    row.update(
        {
            f"time_stats.{model_name}.median": 1.0
            for model_name in _get_moe_family_model_names()
        }
    )
    row.update(
        {
            feature_name: 1.0
            for feature_name in MoETrainer.LOAD_IMBALANCE_FEATURES
            if feature_name not in row
        }
    )
    pd.DataFrame([row]).to_csv(input_file, index=False)

    trainer = object.__new__(MoETrainer)
    trainer.dataset_path = str(input_file)
    trainer.num_experts = 8
    trainer.router_topk = 2
    trainer.hidden_dim = 4096
    trainer.expert_hidden_dim = 5120
    trainer.moe_tensor_parallel_size = 1
    trainer.expert_parallel_size = 1
    trainer._set_dataset_metadata = lambda *_args, **_kwargs: None

    with pytest.raises(ValueError, match="missing required fields"):
        trainer._load_dataset()


def test_linear_result_assembly_skips_empty_split_rows() -> None:
    """A replicated split with no sharded measurements emits only its non-empty side."""

    from frontier.model_architectures import LayerKind
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.linear_op import main as linear_op_main

    config = ModelConfig.from_model_name("step-moe-noquant-small")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[1],
        ffn_tp=[1],
        moe_tp=[8],
        is_moe=True,
    )
    rows = linear_op_main._materialize_profile_result_rows(  # pylint: disable=protected-access
        {
            "num_tokens": 1,
            "num_tensor_parallel_workers": 8,
            "time_stats": {
                "input_layernorm": {"mean": 1.0},
            },
            "typed_operator_contracts": plan["typed_operator_contracts"],
        },
        measurement_type=MeasurementType.CUDA_EVENT.value,
        should_split=True,
        replicated_op_names=set(plan["replicated_ops"]),
        model_config=config,
    )

    assert len(rows) == 1
    assert rows[0]["time_stats"]
    assert set(rows[0]["time_stats"]) == {"input_layernorm"}


def test_dense_training_carries_selected_contract_to_loader_and_context(tmp_path) -> None:
    """Dense MLP training preserves its profile-owned contract at both boundaries."""

    from frontier.model_architectures import LayerKind

    linear_file = tmp_path / "linear.csv"
    linear_file.write_text("placeholder\n", encoding="utf-8")
    config = _step3_config()
    config.supports_share_expert = lambda: False
    replica_config = SimpleNamespace(
        model_config=config,
        device="h200",
        model_name="step-moe-noquant",
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)
    captured: dict[str, object] = {"contexts": [], "loader": []}

    def _load_linear(*args, **kwargs):
        captured["loader"].append(kwargs)
        return pd.DataFrame(
            {
                "num_tokens": [1, 2],
                "time_stats.mlp_up_proj.median": [1.0, 2.0],
                "time_stats.mlp_down_proj.median": [1.0, 2.0],
                "time_stats.mlp_act.median": [1.0, 2.0],
            }
        )

    def _train_single(**kwargs):
        captured["contexts"].append(kwargs["training_context"])
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
        training_context={"model_name": "step-moe-noquant"},
        trained_model_signatures=set(),
        models={},
    )

    expected = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_kind=LayerKind.DENSE,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    assert len(captured["loader"]) == 1
    assert captured["loader"][0]["layer_contract"].semantic_identity() == expected.semantic_identity()
    assert captured["loader"][0]["operator_name"] == "mlp_up_proj"
    assert captured["contexts"]
    assert all(
        context["layer_contract"].semantic_identity() == expected.semantic_identity()
        for context in captured["contexts"]
    )


def test_routed_and_shared_training_carry_distinct_contracts(tmp_path) -> None:
    """Routed and shared-expert training use their own typed domains."""

    import frontier.execution_time_predictor.shared_prediction_model_manager as module

    linear_file = tmp_path / "linear.csv"
    moe_file = tmp_path / "moe.csv"
    linear_file.write_text("placeholder\n", encoding="utf-8")
    moe_file.write_text("placeholder\n", encoding="utf-8")
    config = _step3_config()
    replica_config = SimpleNamespace(
        model_config=config,
        device="h200",
        model_name="step-moe-noquant",
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
        speculative_decoding_config=None,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._measurement_family_name = lambda _measurement_type: "cuda_event"
    manager._validate_moe_dataset_contract = lambda *args, **kwargs: None
    manager._get_moe_op_tp_key = lambda *_args, **_kwargs: 1
    manager._is_moe_op_ep_agnostic = lambda *_args, **_kwargs: False
    manager._load_moe_df = lambda *args, **kwargs: pd.DataFrame(
        {"num_tokens": [1], "time_stats.moe_grouped_gemm.median": [1.0]}
    )
    loader_calls = []
    contexts = []

    def _load_linear(*args, **kwargs):
        loader_calls.append(kwargs)
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
    manager._train_single_model = lambda **kwargs: contexts.append(kwargs) or kwargs["model_name"]
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

    routed = next(row for row in contexts if row["model_name"] == "moe_grouped_gemm")
    assert routed["layer_contract"].layer_kind.value == "routed"
    assert routed["layer_contract"].effective_ffn_width == 5120
    shared_calls = [
        call for call in loader_calls if call.get("operator_name") == "share_expert_up_proj"
    ]
    assert shared_calls
    assert shared_calls[0]["layer_contract"].layer_kind.value == "shared"
    assert shared_calls[0]["layer_contract"].effective_ffn_width == 5120
    layernorm = next(
        row for row in contexts if row["model_name"] == "post_attention_layernorm"
    )
    assert "layer_contract" not in layernorm["training_context"]


def _hash_config() -> SimpleNamespace:
    """Return the smallest predictor config accepted by the manager hash path."""

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


def _dense_contract(*, layer_id: int | None = None, tp_size: int = 8):
    config = _step3_config()
    return config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=layer_id,
        operator_name="mlp_up_proj",
        attention_tp_size=tp_size,
        moe_tp_size=1,
        expert_parallel_size=8,
    )


def _cache_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_tokens": [1, 2],
            "target": [1.0, 2.0],
            "profiling_precision": ["BF16", "BF16"],
            "measurement_type": [
                MeasurementType.CUDA_EVENT.value,
                MeasurementType.CUDA_EVENT.value,
            ],
        }
    )


def _bare_manager(tmp_path=None) -> ExecutionTimePredictionModelManager:
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._all_dummy_mode = False
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._cache_dir = str(tmp_path) if tmp_path is not None else "."
    manager._cluster_configs = {}
    manager._trained_models_eager = {}
    manager._trained_models_kernel_only = {}
    manager._models_by_precision_eager = {}
    manager._models_by_precision_kernel_only = {}
    manager._model_profiling_precision_eager = {}
    manager._model_profiling_precision_kernel_only = {}
    manager._models_by_precision = {}
    manager._model_profiling_precision = {}
    return manager


def test_train_single_model_uses_typed_cache_hit_without_training(tmp_path) -> None:
    """A typed cache hit returns the persisted estimator and skips fitting."""

    manager = _bare_manager(tmp_path)
    contract = _dense_contract(layer_id=0)
    frame = _cache_training_frame()
    predictor_config = _hash_config()
    model_hash = manager._get_model_hash(
        "mlp_up_proj",
        frame,
        predictor_config,
        "BF16",
        MeasurementType.CUDA_EVENT,
        layer_contract=contract,
    )
    cached_model = SimpleNamespace()
    manager._model_contract_identity(cached_model, contract)
    manager._store_model_in_cache("mlp_up_proj", model_hash, cached_model)
    manager._create_estimator_and_params = lambda *_args: pytest.fail(
        "cache hit must not create a training estimator"
    )

    loaded = manager._train_single_model(
        model_name="mlp_up_proj",
        df=frame,
        feature_cols=["num_tokens"],
        target_col="target",
        execution_time_predictor_config=predictor_config,
        persist_exact_lookup=False,
        layer_contract=contract,
    )

    assert loaded._frontier_layer_cache_identity == manager._model_contract_identity(
        cached_model, contract
    )


def test_train_single_model_rejects_typed_cache_without_identity_marker(tmp_path) -> None:
    """A typed cache pickle without its selected identity fails immediately."""

    manager = _bare_manager(tmp_path)
    contract = _dense_contract(layer_id=0)
    frame = _cache_training_frame()
    predictor_config = _hash_config()
    model_hash = manager._get_model_hash(
        "mlp_up_proj",
        frame,
        predictor_config,
        "BF16",
        MeasurementType.CUDA_EVENT,
        layer_contract=contract,
    )
    manager._store_model_in_cache("mlp_up_proj", model_hash, SimpleNamespace())

    with pytest.raises(ValueError, match="missing selected layer cache identity"):
        manager._train_single_model(
            model_name="mlp_up_proj",
            df=frame,
            feature_cols=["num_tokens"],
            target_col="target",
            execution_time_predictor_config=predictor_config,
            persist_exact_lookup=False,
            layer_contract=contract,
        )


def test_equivalent_physical_layers_reuse_one_typed_cache_file(tmp_path) -> None:
    """Equivalent physical layer occurrences resolve to one cache key/file."""

    manager = _bare_manager(tmp_path)
    frame = _cache_training_frame()
    predictor_config = _hash_config()
    layer_zero = _dense_contract(layer_id=0)
    layer_one = _dense_contract(layer_id=1)
    assert manager._get_model_hash(
        "mlp_up_proj",
        frame,
        predictor_config,
        "BF16",
        MeasurementType.CUDA_EVENT,
        layer_contract=layer_zero,
    ) == manager._get_model_hash(
        "mlp_up_proj",
        frame,
        predictor_config,
        "BF16",
        MeasurementType.CUDA_EVENT,
        layer_contract=layer_one,
    )
    model_hash = manager._get_model_hash(
        "mlp_up_proj",
        frame,
        predictor_config,
        "BF16",
        MeasurementType.CUDA_EVENT,
        layer_contract=layer_zero,
    )
    cached_model = SimpleNamespace()
    manager._model_contract_identity(cached_model, layer_zero)
    manager._store_model_in_cache("mlp_up_proj", model_hash, cached_model)

    first = manager._train_single_model(
        model_name="mlp_up_proj",
        df=frame,
        feature_cols=["num_tokens"],
        target_col="target",
        execution_time_predictor_config=predictor_config,
        persist_exact_lookup=False,
        layer_contract=layer_zero,
    )
    second = manager._train_single_model(
        model_name="mlp_up_proj",
        df=frame,
        feature_cols=["num_tokens"],
        target_col="target",
        execution_time_predictor_config=predictor_config,
        persist_exact_lookup=False,
        layer_contract=layer_one,
    )

    assert first._frontier_layer_cache_identity == second._frontier_layer_cache_identity
    assert len(list(tmp_path.glob("mlp_up_proj_*_model.pkl"))) == 0
    assert len(list(tmp_path.glob("mlp_up_proj_*.pkl"))) == 1


def test_train_single_model_keeps_legacy_untyped_cache_compatible(tmp_path) -> None:
    """A legacy untyped cache remains loadable without a typed contract."""

    manager = _bare_manager(tmp_path)
    frame = _cache_training_frame()
    predictor_config = _hash_config()
    model_hash = manager._get_model_hash(
        "mlp_up_proj",
        frame,
        predictor_config,
        "BF16",
        MeasurementType.CUDA_EVENT,
    )
    legacy_model = SimpleNamespace()
    manager._store_model_in_cache("mlp_up_proj", model_hash, legacy_model)

    loaded = manager._train_single_model(
        model_name="mlp_up_proj",
        df=frame,
        feature_cols=["num_tokens"],
        target_col="target",
        execution_time_predictor_config=predictor_config,
        persist_exact_lookup=False,
    )

    assert not hasattr(loaded, "_frontier_layer_cache_identity")


def test_manager_keeps_precision_and_measurement_buckets_isolated() -> None:
    """Precision and measurement families never overwrite one another."""

    manager = _bare_manager()
    contract = _dense_contract(tp_size=8)
    eager_bf16 = SimpleNamespace()
    eager_fp8 = SimpleNamespace()
    kernel_bf16 = SimpleNamespace()
    manager._store_model_precision(
        "mlp_up_proj", "BF16", eager_bf16, layer_contract=contract
    )
    manager._store_model_precision(
        "mlp_up_proj", "FP8", eager_fp8, layer_contract=contract
    )
    manager._active_measurement_type = MeasurementType.KERNEL_ONLY
    manager._store_model_precision(
        "mlp_up_proj", "BF16", kernel_bf16, layer_contract=contract
    )

    assert manager._get_family_model(
        "eager",
        "mlp_up_proj",
        precision_key="BF16",
        requested_identity=manager._model_contract_identity(eager_bf16, contract),
    ) is eager_bf16
    assert manager._get_family_model(
        "eager",
        "mlp_up_proj",
        precision_key="FP8",
        requested_identity=manager._model_contract_identity(eager_fp8, contract),
    ) is eager_fp8
    assert manager._get_family_model(
        "kernel_only",
        "mlp_up_proj",
        precision_key="BF16",
        requested_identity=manager._model_contract_identity(kernel_bf16, contract),
    ) is kernel_bf16


def test_manager_cache_hash_uses_selected_semantic_contract() -> None:
    """Selected typed fields affect cache keys while physical layer occurrence does not."""

    manager = _bare_manager()
    frame = pd.DataFrame({"num_tokens": [1], "target": [2.0]})
    config = _hash_config()
    layer_zero = _dense_contract(layer_id=0)
    layer_one = _dense_contract(layer_id=1)
    changed_width = replace(layer_zero, effective_ffn_width=18440)

    hash_zero = manager._get_model_hash(
        "mlp_up_proj",
        frame,
        config,
        "BF16",
        MeasurementType.CUDA_EVENT,
        layer_contract=layer_zero,
    )
    hash_one = manager._get_model_hash(
        "mlp_up_proj",
        frame,
        config,
        "BF16",
        MeasurementType.CUDA_EVENT,
        layer_contract=layer_one,
    )
    changed_hash = manager._get_model_hash(
        "mlp_up_proj",
        frame,
        config,
        "BF16",
        MeasurementType.CUDA_EVENT,
        layer_contract=changed_width,
    )

    assert hash_zero == hash_one
    assert changed_hash != hash_zero


def test_manager_cache_marker_is_written_and_validated(tmp_path) -> None:
    """Typed cache objects carry one selected identity and reject conflicts."""

    manager = _bare_manager(tmp_path)
    contract = _dense_contract(layer_id=0)
    model = SimpleNamespace()

    identity = manager._model_contract_identity(model, contract)
    assert identity
    assert model._frontier_layer_cache_identity == identity

    manager._store_model_in_cache("mlp_up_proj", "cache-key", model)
    loaded = manager._load_model_from_cache("mlp_up_proj", "cache-key")
    assert loaded._frontier_layer_cache_identity == identity
    manager._validate_cached_layer_cache_identity(
        model_name="mlp_up_proj",
        model=loaded,
        requested_identity=identity,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        manager._validate_cached_layer_cache_identity(
            model_name="mlp_up_proj",
            model=loaded,
            requested_identity=identity + "-other",
        )


def test_manager_typed_registry_is_canonical_and_requires_context_for_ambiguity() -> None:
    """Multiple typed variants stay isolated and bare lookup refuses guessing."""

    manager = _bare_manager()
    tp8 = _dense_contract(tp_size=8)
    tp4 = _dense_contract(tp_size=4)
    model8 = SimpleNamespace()
    model4 = SimpleNamespace()

    manager._store_model_precision(
        "mlp_up_proj", "BF16", model8, layer_contract=tp8
    )
    manager._store_model_precision(
        "mlp_up_proj", "BF16", model4, layer_contract=tp4
    )

    assert manager.get_model("mlp_up_proj", "BF16", layer_contract=tp8) is model8
    assert manager.get_model("mlp_up_proj", "BF16", layer_contract=tp4) is model4
    with pytest.raises(ValueError, match="multiple layer contracts"):
        manager.get_model("mlp_up_proj", "BF16")


def test_manager_projection_selects_cluster_contract_and_rejects_unresolved_variants() -> None:
    """A cluster view selects its replica contract; an absent view rejects ambiguity."""

    config = _step3_config()
    manager = _bare_manager()
    tp8 = _dense_contract(tp_size=8)
    tp4 = _dense_contract(tp_size=4)
    model8 = SimpleNamespace()
    model4 = SimpleNamespace()
    manager._store_model_precision(
        "mlp_up_proj", "BF16", model8, layer_contract=tp8
    )
    manager._store_model_precision(
        "mlp_up_proj", "BF16", model4, layer_contract=tp4
    )
    manager._cluster_configs = {
        ClusterType.PREFILL: SimpleNamespace(
            replica_config=SimpleNamespace(
                model_config=config,
                model_name="step-moe-noquant",
                device="h200",
                attn_tensor_parallel_size=8,
                moe_tensor_parallel_size=1,
                moe_expert_parallel_size=8,
            )
        )
    }

    projected = manager.get_models_for_cluster(ClusterType.PREFILL)
    assert projected["eager"]["mlp_up_proj"] is model8
    with pytest.raises(ValueError, match="multiple layer contracts"):
        manager.get_models()


def test_decode_attn_projection_excludes_typed_ffn_variants() -> None:
    """The attention-only role exposes attention models and no FFN variants."""

    config = _step3_config()
    manager = _bare_manager()
    manager._trained_models_eager["attn_pre_proj"] = object()
    dense = _dense_contract(tp_size=8)
    typed_model = SimpleNamespace()
    manager._store_model_precision(
        "mlp_up_proj", "BF16", typed_model, layer_contract=dense
    )
    manager._cluster_configs = {
        ClusterType.DECODE_ATTN: SimpleNamespace(
            replica_config=SimpleNamespace(
                model_config=config,
                model_name="step-moe-noquant",
                device="h200",
                attn_tensor_parallel_size=0,
                moe_tensor_parallel_size=0,
                moe_expert_parallel_size=0,
            )
        )
    }
    manager._is_kernel_only_measurement_enabled_for_cluster = lambda _cluster: False

    projected = manager.get_models_for_cluster(ClusterType.DECODE_ATTN)
    assert set(projected["eager"]) == {"attn_pre_proj"}


def test_manager_keeps_eager_and_kernel_only_typed_registries_separate() -> None:
    """The same typed operator can have independent eager and kernel-only models."""

    manager = _bare_manager()
    contract = _dense_contract(tp_size=8)
    eager = SimpleNamespace()
    kernel = SimpleNamespace()
    manager._store_model_precision("mlp_up_proj", "BF16", eager, layer_contract=contract)
    manager._active_measurement_type = MeasurementType.KERNEL_ONLY
    manager._store_model_precision("mlp_up_proj", "BF16", kernel, layer_contract=contract)

    assert manager.get_models()["eager"]["mlp_up_proj"] is eager
    assert manager.get_models()["kernel_only"]["mlp_up_proj"] is kernel


def test_moe_wrapper_initializes_runtime_with_routed_width(monkeypatch, tmp_path) -> None:
    """The existing MoE producer uses the profile-owned routed width end to end."""

    moe_module = importlib.import_module("frontier.profiling.moe.moe_wrapper")
    captured = {}

    class _FakeModule:
        def __init__(self, **kwargs):
            captured.setdefault(type(self).__name__, []).append(kwargs)

        def to(self, **_kwargs):
            return self

        def cuda(self):
            return self

        def eval(self):
            return self

        def parameters(self):
            return ()

    class _FakeGating(_FakeModule):
        pass

    class _FakeShuffler(_FakeModule):
        pass

    class _FakeGroupedGemm(_FakeModule):
        pass

    monkeypatch.setattr(moe_module, "MoEGatingNetwork", _FakeGating)
    monkeypatch.setattr(moe_module, "MoETokenShuffler", _FakeShuffler)
    monkeypatch.setattr(moe_module, "MoEGroupedGEMM", _FakeGroupedGemm)
    monkeypatch.setattr(moe_module.MoEWrapper, "_initialize_weights", lambda self: None)
    monkeypatch.setattr(
        moe_module.MoEWrapper,
        "_init_gating_runtime_context_state",
        lambda self: None,
    )

    config = importlib.import_module("frontier.profiling.common.model_config").ModelConfig.from_model_name(
        "step-moe-noquant"
    )
    config.mlp_hidden_dim = 18432
    moe_module.MoEWrapper(
        model_config=config,
        num_tensor_parallel_workers=1,
        expert_parallel_size=8,
        profile_method="cuda_event",
        rank=0,
        output_dir=str(tmp_path),
        use_vllm_kernel=False,
    )

    assert captured["_FakeShuffler"][0]["expert_hidden_dim"] == 5120
    assert captured["_FakeGroupedGemm"][0]["expert_hidden_dim"] == 5120


@pytest.mark.parametrize("config_factory", [_step3_config])
def test_builtin_moe_layer_getters_delegate_to_strict_parser(monkeypatch, config_factory) -> None:
    """Built-in config getters share the canonical strict layer parser."""

    model_architectures = importlib.import_module("frontier.model_architectures")
    calls = []

    def _parse(raw_layers, num_layers):
        calls.append((raw_layers, num_layers))
        return (1,)

    monkeypatch.setattr(model_architectures, "parse_moe_layer_ids", _parse)
    config = config_factory()
    config.moe_layers_enum = "1,2"
    config._moe_layer_ids_cache = None
    assert config.get_moe_layer_ids() == [1]
    assert calls == [("1,2", config.num_layers)]


def test_custom_moe_getter_and_predicate_conflict_fails_fast() -> None:
    """A custom config cannot expose contradictory MoE layer semantics."""

    profile = _step3_config().get_model_architecture_profile()
    config = SimpleNamespace(
        is_moe=True,
        num_layers=4,
        moe_layers_enum=None,
        get_moe_layer_ids=lambda: [1],
        is_moe_layer=lambda layer_id: layer_id == 2,
        routed_mlp_hidden_dim=5120,
        dense_mlp_hidden_dim=18432,
        mlp_hidden_dim=18432,
        embedding_dim=7168,
        num_experts=8,
    )
    with pytest.raises(ValueError, match="disagree|conflict"):
        profile.resolve_layer_contract(config, layer_id=1)


def test_typed_metadata_requires_registry_owned_operator_and_family() -> None:
    """Typed rows admit only names and family IDs owned by the registry."""

    config = _step3_config()
    contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    validator = _require_symbol(
        "frontier.operators.typed_contracts", "validate_typed_operator_metadata"
    )
    metadata = contract.typed_metadata_identity()
    with pytest.raises(ValueError, match="Unknown operator|operator family"):
        validator(metadata, operator_name="unknown_linear", expected_metadata={})
    wrong_family = dict(metadata)
    wrong_family["operator_family_id"] = "moe"
    with pytest.raises(ValueError, match="family"):
        validator(wrong_family, operator_name="mlp_up_proj", expected_metadata={})


def test_typed_architecture_operator_requires_profile_context() -> None:
    """Architecture-owned attention names require selected profile context."""

    validator = _require_symbol(
        "frontier.operators.typed_contracts", "validate_typed_operator_metadata"
    )
    metadata = {
        "profile_id": "step2_mini",
        "operator_family_id": "dense_attention",
        "operator_family_ids": ["dense_attention"],
        "layer_kind": None,
        "dimension_source": None,
        "effective_ffn_width": None,
        "tensor_parallel_mode": "attention_tp",
        "expert_parallel_mode": "off",
        "selected_expert_parallel_size": None,
        "tensor_parallel_sizes": [8],
        "selected_tensor_parallel_size": 8,
        "selected_padded_ffn_width": None,
    }
    with pytest.raises(ValueError, match="architecture_profile"):
        validator(metadata, operator_name="attn_inter_norm", expected_metadata={})


def test_profile_file_identity_check_is_shared_by_predictor_and_manager(tmp_path) -> None:
    """Both CSV consumers reject outer profile IDs that differ from runtime profile."""

    from frontier.execution_time_predictor.profiling_metadata import (
        validate_model_architecture_profile,
    )

    frame = pd.DataFrame({"model_architecture_profile": ["generic"]})
    with pytest.raises(ValueError, match="model_architecture_profile mismatch"):
        validate_model_architecture_profile(
            frame,
            file_path=str(tmp_path / "wrong-profile.csv"),
            expected_profile="step3_text",
        )

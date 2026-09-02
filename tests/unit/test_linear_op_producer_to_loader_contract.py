"""Direct producer-to-loader coverage for mixed typed FFN rows."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.model_architectures import serialize_layer_contract_identity
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op import main as linear_op_main
from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.types import ClusterType, MeasurementType


class _LoaderProbePredictor(SklearnExecutionTimePredictor):
    """Concrete shell for exercising the predictor's CSV loader in isolation."""

    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        return None


def _step3_linear_frame() -> tuple[ModelConfig, pd.DataFrame]:
    """Build a CSV-shaped row through the real linear producer boundary."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
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
            "mlp_up_proj": {"mean": 1.0},
            "share_expert_up_proj": {"mean": 2.0},
        },
        num_tokens=2,
    )

    source = pd.DataFrame([result])
    frame = (
        pd.json_normalize(source["time_stats"])
        .add_prefix("time_stats.")
        .join(source.drop(columns=["time_stats"]))
    )
    return config, linear_op_main._serialize_linear_op_output(  # pylint: disable=protected-access
        frame
    )


def test_mixed_producer_row_is_selected_by_typed_contract_not_legacy_scalar(
    tmp_path,
) -> None:
    """Dense and shared consumers must ignore one mixed-model scalar width."""

    config, source_frame = _step3_linear_frame()
    row_a = source_frame.iloc[0].copy()
    row_b = row_a.copy()
    row_b["n_expanded_embd"] = 18432
    row_b["time_stats.mlp_up_proj.mean"] = 9.0
    row_b["time_stats.share_expert_up_proj.mean"] = 10.0
    contracts = json.loads(row_b["typed_operator_contracts"])
    for operator_name in (
        "mlp_up_proj",
        "mlp_down_proj",
        "mlp_act",
        "share_expert_up_proj",
        "share_expert_down_proj",
        "share_expert_act",
    ):
        contracts.pop(operator_name, None)
    row_b["typed_operator_contracts"] = json.dumps(
        contracts,
        sort_keys=True,
        separators=(",", ":"),
    )

    input_file = tmp_path / "linear_op.csv"
    pd.DataFrame([row_a, row_b]).to_csv(input_file, index=False)

    profile = config.get_model_architecture_profile()
    dense_contract = profile.resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    shared_contract = profile.resolve_layer_contract(
        config,
        layer_id=4,
        operator_name="share_expert_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)

    dense = manager._load_linear_op_df(  # pylint: disable=protected-access
        str(input_file),
        8,
        layer_contract=dense_contract,
        operator_name="mlp_up_proj",
    )
    shared = manager._load_linear_op_df(  # pylint: disable=protected-access
        str(input_file),
        8,
        layer_contract=shared_contract,
        operator_name="share_expert_up_proj",
    )

    assert dense["time_stats.mlp_up_proj.mean"].tolist() == [1.0]
    assert shared["time_stats.share_expert_up_proj.mean"].tolist() == [2.0]


def test_routed_loader_uses_profile_owned_width_when_legacy_scalar_differs(
    tmp_path,
) -> None:
    """The routed MoE loader keeps its typed width independent of mlp_hidden_dim."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    config.mlp_hidden_dim = 9999
    config.embedding_dim = 7168
    config.num_experts_per_tok = 3
    routed_contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        operator_name="moe_grouped_gemm",
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    input_file = tmp_path / "moe.csv"
    pd.DataFrame(
        {
            "num_tokens": [2],
            "num_experts": [48],
            "router_topk": [3],
            "hidden_dim": [7168],
            "expert_hidden_dim": [5120],
            "num_tensor_parallel_workers": [1],
            "expert_parallel_size": [8],
            "time_stats.moe_grouped_gemm.mean": [3.0],
        }
    ).to_csv(input_file, index=False)

    manager = object.__new__(ExecutionTimePredictionModelManager)
    replica_config = type(
        "ReplicaConfig",
        (),
        {
            "model_config": config,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": 8,
        },
    )()
    loaded = manager._load_moe_df(  # pylint: disable=protected-access
        str(input_file),
        replica_config,
        load_imbalance=False,
        tensor_parallel_size=1,
        expert_parallel_size=8,
        layer_contract=routed_contract,
    )

    assert loaded["expert_hidden_dim"].tolist() == [5120]
    assert loaded["expert_parallel_size"].tolist() == [8]

    with pytest.raises(ValueError, match="No data matches|expert_parallel_size"):
        manager._load_moe_df(  # pylint: disable=protected-access
            str(input_file),
            replica_config,
            load_imbalance=False,
            tensor_parallel_size=1,
            expert_parallel_size=2,
            layer_contract=routed_contract,
        )


def test_sklearn_loader_accepts_real_mixed_producer_row_for_dense_contract(
    tmp_path,
) -> None:
    """The independent predictor must honor typed width over the legacy scalar."""

    config, source_frame = _step3_linear_frame()
    input_file = tmp_path / "linear_op.csv"
    source_frame.to_csv(input_file, index=False)

    dense_contract = config.get_model_architecture_profile().resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    predictor = object.__new__(_LoaderProbePredictor)
    predictor._model_config = config
    predictor._replica_config = type(
        "ReplicaConfig",
        (),
        {
            "attn_tensor_parallel_size": 8,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": 8,
        },
    )()
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._resolve_typed_linear_contract = lambda _operator_name: dense_contract
    predictor._get_compute_model_names = lambda: ["mlp_up_proj"]
    predictor._get_profiling_metadata = lambda _df, _path: None
    predictor._validate_active_measurement_type = lambda _metadata, _path=None: None
    predictor._register_profiling_metadata_for_ops = lambda *_args: None

    loaded = predictor._load_compute_df(  # pylint: disable=protected-access
        str(input_file),
        tensor_parallel_size=8,
        operator_name="mlp_up_proj",
        layer_contract=dense_contract,
    )

    assert loaded["n_expanded_embd"].tolist() == [5120]
    assert loaded["typed_operator_contracts"].notna().all()


def test_sklearn_loader_accepts_operator_only_contract_for_explicit_layer(
    tmp_path,
) -> None:
    """Loader matching ignores layer identity granularity, not contract semantics."""

    config, source_frame = _step3_linear_frame()
    input_file = tmp_path / "linear_op.csv"
    source_frame.to_csv(input_file, index=False)

    profile = config.get_model_architecture_profile()
    explicit_contract = profile.resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    operator_only_contract = replace(explicit_contract, layer_id=None)

    predictor = object.__new__(_LoaderProbePredictor)
    predictor._model_config = config
    predictor._replica_config = type(
        "ReplicaConfig",
        (),
        {
            "attn_tensor_parallel_size": 8,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": 8,
        },
    )()
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._resolve_typed_linear_contract = lambda _operator_name: operator_only_contract
    predictor._get_compute_model_names = lambda: ["mlp_up_proj"]
    predictor._get_profiling_metadata = lambda _df, _path: None
    predictor._validate_active_measurement_type = lambda _metadata, _path=None: None
    predictor._register_profiling_metadata_for_ops = lambda *_args: None

    loaded = predictor._load_compute_df(  # pylint: disable=protected-access
        str(input_file),
        tensor_parallel_size=8,
        operator_name="mlp_up_proj",
        layer_contract=explicit_contract,
    )

    assert loaded["time_stats.mlp_up_proj.mean"].tolist() == [1.0]


def test_layer_contract_semantic_equivalence_preserves_cache_identity() -> None:
    """Layer granularity is semantic metadata, while cache identity stays exact."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    profile = config.get_model_architecture_profile()
    explicit_contract = profile.resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    operator_only_contract = replace(explicit_contract, layer_id=None)

    assert explicit_contract.is_semantically_equivalent(operator_only_contract)
    assert serialize_layer_contract_identity(explicit_contract) != (
        serialize_layer_contract_identity(operator_only_contract)
    )

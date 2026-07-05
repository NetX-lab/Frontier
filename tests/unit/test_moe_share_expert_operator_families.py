from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.operators.binding import build_operator_manifest
from frontier.operators.spec import (
    OperatorPhase,
    OperatorRole,
    ResourceClass,
    TensorParallelMode,
    TraceKind,
)
from frontier.types import ActivationType, NormType
from frontier.types import ClusterType


class _ConcreteSklearnMoEExecutionTimePredictor(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise NotImplementedError


def _moe_model_config(*, share_expert_dim: int | None = None) -> BaseModelConfig:
    return BaseModelConfig(
        num_layers=2,
        num_q_heads=32,
        num_kv_heads=8,
        embedding_dim=4096,
        mlp_hidden_dim=11008,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        share_expert_dim=share_expert_dim,
        model_type="unit_moe_model",
    )


def test_moe_family_declares_router_dispatch_and_grouped_gemm_ops() -> None:
    from frontier.operators.families import MOE_FAMILY

    assert MOE_FAMILY.family_id == "moe"
    assert MOE_FAMILY.resource_class is ResourceClass.COMP
    assert [operator.name for operator in MOE_FAMILY.operators] == [
        "moe_gating_linear",
        "moe_gating_routing_topk",
        "moe_shuffling",
        "moe_grouped_gemm",
    ]
    assert [operator.execution_time_attr for operator in MOE_FAMILY.operators] == [
        "moe_gating_linear_time",
        "moe_gating_routing_topk_time",
        "moe_shuffling_time",
        "moe_grouped_gemm_time",
    ]
    assert [operator.resource_class for operator in MOE_FAMILY.operators] == [
        ResourceClass.COMP,
        ResourceClass.MEMORY,
        ResourceClass.MEMORY,
        ResourceClass.COMP,
    ]
    assert [operator.role for operator in MOE_FAMILY.operators] == [
        OperatorRole.PROJECTION,
        OperatorRole.RESHAPE,
        OperatorRole.RESHAPE,
        OperatorRole.PROJECTION,
    ]
    assert all(operator.trace_kind is TraceKind.COMPUTE for operator in MOE_FAMILY.operators)
    assert all(
        operator.phases
        == (OperatorPhase.PREFILL, OperatorPhase.DECODE, OperatorPhase.MIXED)
        for operator in MOE_FAMILY.operators
    )


def test_moe_family_declares_predictor_tp_and_ep_metadata() -> None:
    from frontier.operators.families import MOE_FAMILY

    assert [operator.tp_mode for operator in MOE_FAMILY.operators] == [
        TensorParallelMode.MOE_TP,
        TensorParallelMode.MOE_TP,
        TensorParallelMode.MOE_TP,
        TensorParallelMode.MOE_TP,
    ]
    assert [operator.ep_agnostic for operator in MOE_FAMILY.operators] == [
        True,
        True,
        True,
        False,
    ]
    assert [operator.precision_name() for operator in MOE_FAMILY.operators] == [
        "moe_gating",
        "moe_gating",
        "moe_shuffling",
        "moe_grouped_gemm",
    ]


def test_sklearn_moe_predictor_resolves_tp_and_ep_from_moe_family_metadata() -> None:
    predictor_cls = SklearnMoEExecutionTimePredictor

    assert predictor_cls._get_moe_op_tp_key("moe_gating_linear", moe_tp_size=4) == 4
    assert predictor_cls._get_moe_op_tp_key("moe_gating_routing_topk", moe_tp_size=4) == 4
    assert predictor_cls._get_moe_op_tp_key("moe_shuffling", moe_tp_size=4) == 4
    assert predictor_cls._get_moe_op_tp_key("moe_grouped_gemm", moe_tp_size=4) == 4

    assert predictor_cls._is_moe_op_ep_agnostic("moe_gating_linear") is True
    assert predictor_cls._is_moe_op_ep_agnostic("moe_gating_routing_topk") is True
    assert predictor_cls._is_moe_op_ep_agnostic("moe_shuffling") is True
    assert predictor_cls._is_moe_op_ep_agnostic("moe_grouped_gemm") is False

    with pytest.raises(ValueError, match="Unsupported MoE op for TP mapping"):
        predictor_cls._get_moe_op_tp_key("unknown_moe_op", moe_tp_size=4)
    with pytest.raises(ValueError, match="Unsupported MoE op for EP mapping"):
        predictor_cls._is_moe_op_ep_agnostic("unknown_moe_op")


def test_moe_auxiliary_tp_key_preserves_deferred_pdd_legacy_scope() -> None:
    predictor_cls = SklearnMoEExecutionTimePredictor
    replica_config = SimpleNamespace(moe_tensor_parallel_size=4)

    for op_name in (
        "moe_gating_linear",
        "moe_gating_routing_topk",
        "moe_shuffling",
    ):
        assert (
            predictor_cls._get_moe_op_tp_key(
                op_name,
                moe_tp_size=4,
                cluster_type=ClusterType.MONOLITHIC,
            )
            == 4
        )
        assert (
            ExecutionTimePredictionModelManager._get_moe_op_tp_key(
                op_name,
                replica_config,
                ClusterType.MONOLITHIC,
            )
            == 4
        )

        for cluster_type in (
            ClusterType.PREFILL,
            ClusterType.DECODE,
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
        ):
            assert (
                predictor_cls._get_moe_op_tp_key(
                    op_name,
                    moe_tp_size=4,
                    cluster_type=cluster_type,
                )
                == 1
            )
            assert (
                ExecutionTimePredictionModelManager._get_moe_op_tp_key(
                    op_name,
                    replica_config,
                    cluster_type,
                )
                == 1
            )

    assert (
        predictor_cls._get_moe_op_tp_key(
            "moe_grouped_gemm",
            moe_tp_size=4,
            cluster_type=ClusterType.PREFILL,
        )
        == 4
    )
    assert (
        ExecutionTimePredictionModelManager._get_moe_op_tp_key(
            "moe_grouped_gemm",
            replica_config,
            ClusterType.PREFILL,
        )
        == 4
    )


def test_moe_column_validation_uses_moe_family_profiling_names(monkeypatch) -> None:
    import pandas as pd
    import frontier.execution_time_predictor.sklearn_moe_execution_time_predictor as moe_module

    def _operator(name: str):
        return SimpleNamespace(name=name, profiling_name=lambda: name)

    monkeypatch.setattr(
        moe_module,
        "MOE_FAMILY",
        SimpleNamespace(
            profiling_ops=lambda: (
                _operator("moe_family_first"),
                _operator("moe_family_second"),
            )
        ),
    )

    with pytest.raises(ValueError, match="time_stats.moe_family_second.median"):
        moe_module._validate_moe_columns(
            pd.DataFrame({"time_stats.moe_family_first.median": [1.0]})
        )


def test_sklearn_moe_training_uses_moe_family_profiling_names(
    monkeypatch,
    tmp_path,
) -> None:
    import pandas as pd
    import frontier.execution_time_predictor.sklearn_moe_execution_time_predictor as moe_module

    def _operator(name: str):
        return SimpleNamespace(
            name=name,
            profiling_name=lambda: name,
            precision_name=lambda: name,
            tp_mode=TensorParallelMode.REPLICATED,
            ep_agnostic=True,
        )

    monkeypatch.setattr(
        moe_module,
        "MOE_FAMILY",
        SimpleNamespace(
            profiling_ops=lambda: (
                _operator("moe_family_first"),
                _operator("moe_family_second"),
            )
        ),
    )

    csv_path = tmp_path / "moe.csv"
    pd.DataFrame(
        {
            "num_experts": [8],
            "router_topk": [2],
            "hidden_dim": [4096],
            "expert_hidden_dim": [11008],
            "num_tensor_parallel_workers": [1],
            "expert_parallel_size": [1],
            "num_tokens": [8],
            "time_stats.moe_family_first.median": [1.0],
            "time_stats.moe_family_second.median": [2.0],
        }
    ).to_csv(csv_path, index=False)

    predictor = object.__new__(_ConcreteSklearnMoEExecutionTimePredictor)
    predictor._moe_input_file = str(csv_path)
    predictor._model_config = SimpleNamespace(
        num_experts=8,
        num_experts_per_tok=2,
        embedding_dim=4096,
        mlp_hidden_dim=11008,
    )
    predictor._replica_config = SimpleNamespace(
        moe_tensor_parallel_size=4,
        moe_expert_parallel_size=1,
        model_name="unit_moe",
    )
    predictor._get_profiling_metadata = lambda df, input_file: {}
    predictor._validate_active_measurement_type = lambda metadata, input_file: None
    registered_model_names: list[tuple[str, ...]] = []
    predictor._register_profiling_metadata_for_ops = (
        lambda model_names, metadata, input_file: registered_model_names.append(
            tuple(model_names)
        )
    )
    predictor._train_model = lambda **kwargs: SimpleNamespace(
        model_name=kwargs["model_name"],
        feature_cols=tuple(kwargs["feature_cols"]),
        target_col=kwargs["target_col"],
    )

    models = predictor._train_moe_models()

    assert tuple(models) == ("moe_family_first", "moe_family_second")
    assert registered_model_names == [("moe_family_first", "moe_family_second")]
    assert models["moe_family_first"].target_col == "time_stats.moe_family_first.median"
    assert models["moe_family_second"].target_col == "time_stats.moe_family_second.median"


def test_sklearn_moe_dataset_contract_filters_gating_context_from_operator_precision(
    monkeypatch,
) -> None:
    import pandas as pd
    import frontier.execution_time_predictor.sklearn_moe_execution_time_predictor as moe_module

    def _operator(name: str):
        return SimpleNamespace(
            name=name,
            profiling_name=lambda: name,
            precision_name=lambda: "moe_gating",
            tp_mode=TensorParallelMode.REPLICATED,
            ep_agnostic=True,
        )

    monkeypatch.setattr(
        moe_module,
        "MOE_FAMILY",
        SimpleNamespace(profiling_ops=lambda: (_operator("moe_family_gate"),)),
    )

    predictor = object.__new__(_ConcreteSklearnMoEExecutionTimePredictor)
    predictor._model_config = SimpleNamespace(
        num_experts=8,
        num_experts_per_tok=2,
        embedding_dim=4096,
        mlp_hidden_dim=11008,
    )
    predictor._get_requested_moe_gating_routing_runtime_path = lambda: "simulation"

    df = pd.DataFrame(
        {
            "num_experts": [8],
            "router_topk": [2],
            "hidden_dim": [4096],
            "expert_hidden_dim": [11008],
            "num_tensor_parallel_workers": [1],
            "expert_parallel_size": [1],
            "gating_runtime_context": ["prefill_hot"],
            "time_stats.moe_family_gate.median": [1.0],
        }
    )

    with pytest.raises(ValueError, match="standalone_legacy"):
        predictor._validate_moe_dataset_contract(
            df,
            "/tmp/moe.csv",
            ["moe_family_gate"],
            moe_tp_size=4,
            moe_ep_size=1,
        )


def test_shared_manager_validates_moe_training_names_from_moe_family(
    monkeypatch,
    tmp_path,
) -> None:
    import frontier.execution_time_predictor.shared_prediction_model_manager as manager_module

    class _StopAfterValidation(Exception):
        pass

    def _operator(name: str):
        return SimpleNamespace(
            name=name,
            profiling_name=lambda: name,
            precision_name=lambda: name,
            tp_mode=TensorParallelMode.REPLICATED,
            ep_agnostic=True,
        )

    monkeypatch.setattr(
        manager_module,
        "MOE_FAMILY",
        SimpleNamespace(
            profiling_ops=lambda: (
                _operator("moe_family_first"),
                _operator("moe_family_second"),
            )
        ),
    )

    moe_csv = tmp_path / "moe.csv"
    moe_csv.write_text("moe_family_validation_marker\n", encoding="utf-8")

    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._measurement_family_name = lambda measurement_type: "cuda_event"
    captured_model_names: list[tuple[str, ...]] = []

    captured_cluster_types: list[ClusterType] = []

    def _capture_validation(file_path, replica_config, model_names, cluster_type):
        captured_model_names.append(tuple(model_names))
        captured_cluster_types.append(cluster_type)
        raise _StopAfterValidation

    manager._validate_moe_dataset_contract = _capture_validation

    model_config = SimpleNamespace(
        get_model_arch=lambda: "unit_moe",
        supports_share_expert=lambda: False,
        use_qk_norm=False,
    )
    replica_config = SimpleNamespace(
        device="h800",
        model_name="unit_moe",
        model_config=model_config,
        moe_tensor_parallel_size=4,
        moe_expert_parallel_size=1,
    )

    with pytest.raises(_StopAfterValidation):
        manager._train_ffn_models_for_cluster(
            ClusterType.MONOLITHIC,
            replica_config,
            execution_time_predictor_config=SimpleNamespace(),
            linear_ops_file=str(tmp_path / "linear_op.csv"),
            moe_file=str(moe_csv),
            is_moe_model=True,
            trained_model_signatures=set(),
        )

    assert captured_model_names == [("moe_family_first", "moe_family_second")]
    assert captured_cluster_types == [ClusterType.MONOLITHIC]


def test_shared_manager_moe_dataset_contract_uses_pdd_legacy_auxiliary_tp_key(
    tmp_path,
) -> None:
    import pandas as pd

    csv_path = tmp_path / "moe.csv"
    pd.DataFrame(
        [
            {
                "num_experts": 8,
                "router_topk": 2,
                "hidden_dim": 4096,
                "expert_hidden_dim": 11008,
                "num_tensor_parallel_workers": 1,
                "expert_parallel_size": 1,
                "routing_runtime_path": "standard_fused_topk",
            },
            {
                "num_experts": 8,
                "router_topk": 2,
                "hidden_dim": 4096,
                "expert_hidden_dim": 11008,
                "num_tensor_parallel_workers": 4,
                "expert_parallel_size": 1,
                "routing_runtime_path": "standard_fused_topk",
            },
        ]
    ).to_csv(csv_path, index=False)

    manager = object.__new__(ExecutionTimePredictionModelManager)
    replica_config = SimpleNamespace(
        model_config=SimpleNamespace(
            num_experts=8,
            num_experts_per_tok=2,
            embedding_dim=4096,
            mlp_hidden_dim=11008,
        ),
        moe_tensor_parallel_size=4,
        moe_expert_parallel_size=1,
        moe_routing_mode="simulation",
    )

    manager._validate_moe_dataset_contract(
        str(csv_path),
        replica_config,
        [
            "moe_gating_linear",
            "moe_gating_routing_topk",
            "moe_shuffling",
            "moe_grouped_gemm",
        ],
        ClusterType.PREFILL,
    )


def test_share_expert_family_declares_shared_dense_expert_ops() -> None:
    from frontier.operators.families import SHARE_EXPERT_FAMILY

    assert SHARE_EXPERT_FAMILY.family_id == "share_expert"
    assert SHARE_EXPERT_FAMILY.resource_class is ResourceClass.COMP
    assert [operator.name for operator in SHARE_EXPERT_FAMILY.operators] == [
        "share_expert_up_proj",
        "share_expert_act",
        "share_expert_down_proj",
    ]
    assert [operator.execution_time_attr for operator in SHARE_EXPERT_FAMILY.operators] == [
        "share_expert_up_proj_time",
        "share_expert_act_time",
        "share_expert_down_proj_time",
    ]
    assert [operator.calibration_key for operator in SHARE_EXPERT_FAMILY.operators] == [
        "share_expert_up_proj",
        None,
        "share_expert_down_proj",
    ]
    assert [operator.role for operator in SHARE_EXPERT_FAMILY.operators] == [
        OperatorRole.PROJECTION,
        OperatorRole.ACTIVATION,
        OperatorRole.PROJECTION,
    ]
    assert [operator.tp_mode for operator in SHARE_EXPERT_FAMILY.operators] == [
        TensorParallelMode.FFN_TP,
        TensorParallelMode.FFN_TP,
        TensorParallelMode.FFN_TP,
    ]


def test_linear_profiling_plan_uses_share_expert_family_names() -> None:
    from frontier.profiling.linear_op import profiling_plan

    assert not hasattr(profiling_plan, "SHARE_EXPERT_OPS")

    plan = profiling_plan.build_profiling_plan(
        _moe_model_config(share_expert_dim=4096),
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        disable_replicated=True,
        is_moe=True,
    )

    assert "ffn" not in plan["enabled_ops"]
    assert plan["enabled_ops"] == [
        "attn_pre_proj",
        "attn_rope",
        "attn_post_proj",
        "share_expert_up_proj",
        "share_expert_down_proj",
        "share_expert_act",
    ]


def test_trace_precision_mapping_uses_moe_and_share_expert_family_specs() -> None:
    from frontier.metrics.op_trace_utils import map_trace_op_to_precision_op
    from frontier.operators.families import MOE_FAMILY, SHARE_EXPERT_FAMILY

    for operator in (*MOE_FAMILY.e2e_trace_ops(), *SHARE_EXPERT_FAMILY.e2e_trace_ops()):
        assert map_trace_op_to_precision_op(operator.name) == operator.precision_name()



def test_moe_manifest_appends_moe_family_without_dense_ffn() -> None:
    manifest = build_operator_manifest(_moe_model_config())

    assert [binding.family_id for binding in manifest.family_bindings] == [
        "dense_attention",
        "memory",
        "moe",
    ]
    assert "ffn" not in {binding.family_id for binding in manifest.family_bindings}


def test_share_expert_manifest_appends_share_expert_after_moe_family() -> None:
    manifest = build_operator_manifest(_moe_model_config(share_expert_dim=4096))

    assert [binding.family_id for binding in manifest.family_bindings] == [
        "dense_attention",
        "memory",
        "moe",
        "share_expert",
    ]


def test_moe_time_structured_operator_times_replace_covered_legacy_fields() -> None:
    from frontier.entities.time_components import MoEOperatorTimes, MoETime

    operator_times = MoEOperatorTimes(
        op_times={
            "post_attention_layernorm": 4.0,
            "moe_gating_linear": 10.0,
            "moe_gating_routing_topk": 20.0,
            "moe_shuffling": 30.0,
            "moe_grouped_gemm": 40.0,
            "share_expert_up_proj": 1.0,
            "share_expert_act": 2.0,
            "share_expert_down_proj": 3.0,
        }
    )
    moe_time = MoETime(
        moe_gating_linear_time=0.1,
        moe_gating_routing_topk_time=0.2,
        moe_shuffling_time=0.3,
        moe_grouped_gemm_time=0.4,
        mlp_norm_time=0.5,
        share_expert_up_proj_time=0.6,
        share_expert_act_time=0.7,
        share_expert_down_proj_time=0.8,
        operator_times=operator_times,
    )

    assert moe_time.total_time() == 110.0


def test_moe_predictor_dummy_path_writes_structured_moe_operator_times() -> None:
    predictor = object.__new__(_ConcreteSklearnMoEExecutionTimePredictor)
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 10.0
    predictor._model_config = SimpleNamespace(supports_share_expert=lambda: True)

    moe_time = predictor.predict_moe_layer_time(
        batch_or_group=object(),
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert moe_time.operator_times is not None
    assert moe_time.operator_times.op_times == {
        "post_attention_layernorm": 10.0,
        "moe_gating_linear": 5.0,
        "moe_gating_routing_topk": 5.0,
        "moe_shuffling": 10.0,
        "moe_grouped_gemm": 10.0,
        "share_expert_up_proj": 10.0,
        "share_expert_act": 10.0,
        "share_expert_down_proj": 10.0,
    }
    assert moe_time.total_time() == 70.0

def test_quantization_default_registry_uses_moe_and_share_expert_family_names(monkeypatch) -> None:
    import frontier.config.quantization_manager as quant_module
    from frontier.config.quantization_manager import QuantizationManager

    def _operator(name: str, precision_name: str | None = None):
        return SimpleNamespace(
            profiling_name=lambda: name,
            precision_name=lambda: precision_name or name,
        )

    monkeypatch.setattr(
        quant_module,
        "MOE_FAMILY",
        SimpleNamespace(
            profiling_ops=lambda: (
                _operator("moe_family_gate", "moe_precision_alias"),
                _operator("moe_family_expert"),
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        quant_module,
        "SHARE_EXPERT_FAMILY",
        SimpleNamespace(
            profiling_ops=lambda: (
                _operator("share_family_up"),
                _operator("share_family_down"),
            )
        ),
        raising=False,
    )

    registry = QuantizationManager._get_default_registry(
        object.__new__(QuantizationManager)
    )
    compute_ops = registry["compute_operations"]

    assert "moe_precision_alias" in compute_ops
    assert "moe_family_gate" in compute_ops
    assert "moe_family_expert" in compute_ops
    assert "share_family_up" in compute_ops
    assert "share_family_down" in compute_ops
    assert "moe_gating_linear" not in compute_ops
    assert "share_expert_up_proj" not in compute_ops


def test_moe_trainer_model_names_use_moe_family_and_gating_precision(
    monkeypatch,
) -> None:
    import pandas as pd
    import frontier.training.moe_trainer as trainer_module
    from frontier.training.moe_trainer import MoETrainer
    from frontier.moe_gating_runtime import (
        PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
        PREFILL_HOT_MOE_GATING_RUNTIME_IMPL,
    )

    def _operator(
        name: str,
        *,
        precision_name: str | None = None,
        tp_mode: TensorParallelMode = TensorParallelMode.REPLICATED,
    ):
        return SimpleNamespace(
            name=name,
            profiling_name=lambda: name,
            precision_name=lambda: precision_name or name,
            tp_mode=tp_mode,
        )

    monkeypatch.setattr(
        trainer_module,
        "MOE_FAMILY",
        SimpleNamespace(
            profiling_ops=lambda: (
                _operator("moe_family_gate", precision_name="moe_gating"),
                _operator(
                    "moe_family_expert",
                    tp_mode=TensorParallelMode.MOE_TP,
                ),
            )
        ),
        raising=False,
    )

    trainer = object.__new__(MoETrainer)
    trainer.model_name = "qwen3-a3b-30b-moe"
    trainer.df = pd.DataFrame(
        {
            "gating_runtime_context": [PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT],
            "gating_runtime_context_impl": [PREFILL_HOT_MOE_GATING_RUNTIME_IMPL],
        }
    )

    assert trainer._get_model_names() == [
        "moe_family_gate",
        "moe_family_expert",
        "moe_family_gate__prefill_hot",
    ]


def test_moe_trainer_required_columns_and_tp_keys_use_moe_family_metadata(
    monkeypatch,
) -> None:
    import pandas as pd
    import frontier.training.moe_trainer as trainer_module
    from frontier.training.moe_trainer import MoETrainer

    def _operator(
        name: str,
        *,
        tp_mode: TensorParallelMode,
        ep_agnostic: bool,
    ):
        return SimpleNamespace(
            name=name,
            profiling_name=lambda: name,
            precision_name=lambda: name,
            tp_mode=tp_mode,
            ep_agnostic=ep_agnostic,
        )

    monkeypatch.setattr(
        trainer_module,
        "MOE_FAMILY",
        SimpleNamespace(
            profiling_ops=lambda: (
                _operator(
                    "moe_family_replicated",
                    tp_mode=TensorParallelMode.REPLICATED,
                    ep_agnostic=True,
                ),
                _operator(
                    "moe_family_expert",
                    tp_mode=TensorParallelMode.MOE_TP,
                    ep_agnostic=False,
                ),
            )
        ),
        raising=False,
    )

    trainer = object.__new__(MoETrainer)
    trainer.moe_tensor_parallel_size = 4
    trainer.dataset_path = "/tmp/moe.csv"

    trainer._verify_dataset_columns(
        pd.DataFrame(
            {
                "num_tokens": [8],
                "time_stats.moe_family_replicated.median": [1.0],
                "time_stats.moe_family_expert.median": [2.0],
            }
        )
    )
    assert trainer._get_training_tp_key("moe_family_replicated") == 1
    assert trainer._get_training_tp_key("moe_family_expert") == 4
    with pytest.raises(ValueError, match="Unsupported MoE op for TP mapping"):
        trainer._get_training_tp_key("unknown_moe_op")


def test_moe_trainer_gating_context_filter_uses_operator_precision(
    monkeypatch,
) -> None:
    import pandas as pd
    import frontier.training.moe_trainer as trainer_module
    from frontier.training.moe_trainer import MoETrainer

    def _operator(name: str):
        return SimpleNamespace(
            name=name,
            profiling_name=lambda: name,
            precision_name=lambda: "moe_gating",
            tp_mode=TensorParallelMode.REPLICATED,
            ep_agnostic=True,
        )

    monkeypatch.setattr(
        trainer_module,
        "MOE_FAMILY",
        SimpleNamespace(profiling_ops=lambda: (_operator("moe_family_gate"),)),
        raising=False,
    )

    trainer = object.__new__(MoETrainer)
    trainer.moe_tensor_parallel_size = 4
    trainer.expert_parallel_size = 1
    trainer.dataset_path = "/tmp/moe.csv"
    trainer.gating_runtime_context = "standalone_legacy"
    trainer.routing_runtime_path = "standard_fused_topk"

    df = pd.DataFrame(
        {
            "num_tensor_parallel_workers": [1, 1],
            "expert_parallel_size": [1, 1],
            "num_tokens": [8, 8],
            "gating_runtime_context": ["standalone_legacy", "prefill_hot"],
            "gating_runtime_context_impl": ["none", "ffn_like_prefix_20x"],
            "time_stats.moe_family_gate.median": [1.0, 2.0],
        }
    )

    training_df = trainer._get_training_df_for_model(
        df=df,
        model_name="moe_family_gate",
        feature_cols=["num_tokens"],
        target_col="time_stats.moe_family_gate.median",
    )

    assert training_df["gating_runtime_context"].tolist() == ["standalone_legacy"]


def test_linear_op_wrapper_expected_keys_use_share_expert_family(monkeypatch) -> None:
    from pathlib import Path

    source = Path(
        "frontier/profiling/linear_op/linear_op_wrapper.py"
    ).read_text(encoding="utf-8")

    assert "_share_expert_profiling_names()" in source
    assert (
        '["share_expert_up_proj", "share_expert_down_proj", "share_expert_act"]'
        not in source
    )


def test_moe_profiling_result_validation_uses_moe_family_target_columns(
    monkeypatch,
) -> None:
    import pandas as pd
    import frontier.profiling.moe.main as moe_main

    def _operator(name: str):
        return SimpleNamespace(profiling_name=lambda: name)

    monkeypatch.setattr(
        moe_main,
        "MOE_FAMILY",
        SimpleNamespace(
            profiling_ops=lambda: (
                _operator("moe_profile_first"),
                _operator("moe_profile_second"),
            )
        ),
        raising=False,
    )

    moe_main._validate_canonical_moe_result_df(
        pd.DataFrame(
            {
                "num_tensor_parallel_workers": [1],
                "expert_parallel_size": [1],
                "num_tokens": [8],
                "time_stats.moe_profile_first.median": [1.0],
                "time_stats.moe_profile_second.median": [2.0],
            }
        ),
        model="unit_moe",
    )

    with pytest.raises(ValueError, match="time_stats.moe_profile_second.median"):
        moe_main._validate_canonical_moe_result_df(
            pd.DataFrame(
                {
                    "num_tensor_parallel_workers": [1],
                    "expert_parallel_size": [1],
                    "num_tokens": [8],
                    "time_stats.moe_profile_first.median": [1.0],
                }
            ),
            model="unit_moe",
        )

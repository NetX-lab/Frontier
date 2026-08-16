"""Regression tests for profiling TP and context-domain contracts."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.profiling.attention import main as attention_main
from frontier.profiling.attention.attention_input import AttentionInput
from frontier.profiling.attention import attention_input as attention_input_module
from frontier.profiling.attention.mixed_attention_input import MixedAttentionInput
from frontier.profiling.attention.true_mixed_batch_input import TrueMixedBatchInput
from frontier.profiling.attention.memory_budget import (
    get_attention_backend_workspace_reservation_bytes,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import SUPPORTED_PROFILE_TP_SIZES
from frontier.profiling.common.parallel_config import validate_profile_tp_sizes
from frontier.profiling.common.parallel_config import (
    validate_profile_backed_runtime_tp_sizes,
)
from frontier.profiling.utils.confirmation import build_attention_config_sections
from frontier.profiling.utils import calculate_max_num_blocks_from_memory
from frontier.profiling.attention.memory_budget import (
    resolve_requested_max_num_blocks,
)
from frontier.profiling.linear_op import main as linear_main
from frontier.profiling.moe import main as moe_main
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.prediction_cache_contract import (
    PREDICTION_CACHE_CONTRACT_VERSION,
)
from frontier.types import ActivationType, MeasurementType, NormType


@pytest.mark.parametrize("structure_and_flag", [
    ("linear_op", "--tensor_parallel_size"),
    ("attention", "--tensor_parallel_size"),
    ("moe", "--moe_tensor_parallel_size"),
])
def test_standalone_training_cli_rejects_unsupported_profile_tp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    structure_and_flag: tuple[str, str],
    ) -> None:
    structure, tp_flag = structure_and_flag
    argv = [
        "frontier-training",
        structure,
        "--measurement_type",
        "CUDA_EVENT",
        "--dataset_path" if structure != "attention" else "--layer_dataset_path",
        str(tmp_path / "missing.csv"),
        "--model_name",
        "test-model",
        tp_flag,
        "16",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # The CLI must reject the release-incompatible TP before any dataset/model
    # loading is attempted.
    from frontier.training import cli as training_cli

    with pytest.raises(ValueError, match="supported TP sizes"):
        training_cli.parse_args()


@pytest.mark.parametrize(
    ("trainer_module", "trainer_name", "kwargs"),
    [
        (
            "frontier.training.linear_op_trainer",
            "LinearOpTrainer",
            {"model_name": "test-model", "device": "a100", "tensor_parallel_size": 16},
        ),
        (
            "frontier.training.attention_trainer",
            "AttentionTrainer",
            {"model_name": "test-model", "device": "a100", "tensor_parallel_size": 16},
        ),
        (
            "frontier.training.moe_trainer",
            "MoETrainer",
            {
                "num_experts": 8,
                "router_topk": 2,
                "hidden_dim": 16,
                "expert_hidden_dim": 32,
                "moe_tensor_parallel_size": 16,
            },
        ),
    ],
)
def test_standalone_trainer_programmatic_entry_rejects_unsupported_tp(
    trainer_module: str,
    trainer_name: str,
    kwargs: dict,
) -> None:
    module = __import__(trainer_module, fromlist=[trainer_name])
    trainer_class = getattr(module, trainer_name)
    common = {
        "dataset_path": "/does/not/exist.csv",
        "output_dir": "/tmp/frontier-invalid-tp-test",
    }
    if trainer_name == "AttentionTrainer":
        common.update(
            {
                "layer_dataset_path": common.pop("dataset_path"),
            }
        )
    common.update(kwargs)
    with pytest.raises(ValueError, match="supported TP sizes"):
        trainer_class(**common)


@pytest.mark.parametrize(
    ("factory_module", "factory_name", "tp_kw"),
    [
        (
            "frontier.training.linear_op_trainer",
            "create_linear_op_trainer_from_model_config",
            "tensor_parallel_size",
        ),
        (
            "frontier.training.attention_trainer",
            "create_attention_trainer_from_model_config",
            "tensor_parallel_size",
        ),
        (
            "frontier.training.moe_trainer",
            "create_moe_trainer_from_model_config",
            "moe_tensor_parallel_size",
        ),
    ],
)
def test_standalone_trainer_factory_rejects_tp_before_model_config_load(
    monkeypatch: pytest.MonkeyPatch,
    factory_module: str,
    factory_name: str,
    tp_kw: str,
) -> None:
    module = __import__(factory_module, fromlist=[factory_name])
    factory = getattr(module, factory_name)
    from frontier.config.model_config import BaseModelConfig

    monkeypatch.setattr(
        BaseModelConfig,
        "create_from_name",
        staticmethod(
            lambda _name: pytest.fail(
                "model config loaded before standalone TP validation"
            )
        ),
    )
    kwargs = {
        "dataset_path": "/does/not/exist.csv",
        "output_dir": "/tmp/frontier-invalid-tp-test",
        "model_name": "test-model",
        tp_kw: 16,
    }
    if factory_name == "create_attention_trainer_from_model_config":
        kwargs["layer_dataset_path"] = kwargs.pop("dataset_path")
    with pytest.raises(ValueError, match="supported TP sizes"):
        factory(**kwargs)


def test_profile_backed_runtime_tp_validation_is_context_aware() -> None:
    replica_config = SimpleNamespace(
        attn_tensor_parallel_size=16,
        moe_tensor_parallel_size=3,
    )

    with pytest.raises(ValueError, match="attn_tensor_parallel_size.*supported TP sizes"):
        validate_profile_backed_runtime_tp_sizes(
            replica_config,
            cluster_type="MONOLITHIC",
            model_is_moe=False,
            enable_dummy_mode=False,
        )

    # A communication-only/FFN-unrelated configuration must not be rejected for
    # an unused TP field.  This is the semantic distinction that a global
    # ReplicaConfig.__post_init__ check would break.
    validate_profile_backed_runtime_tp_sizes(
        replica_config,
        cluster_type="TRANS",
        model_is_moe=False,
        enable_dummy_mode=False,
    )

    # Dummy compute intentionally bypasses the profile-backed TP boundary.
    validate_profile_backed_runtime_tp_sizes(
        replica_config,
        cluster_type="MONOLITHIC",
        model_is_moe=True,
        enable_dummy_mode=True,
    )


def test_profile_backed_runtime_tp_validation_ignores_unused_pdaf_role_fields() -> None:
    validate_profile_backed_runtime_tp_sizes(
        SimpleNamespace(attn_tensor_parallel_size=2, moe_tensor_parallel_size=0),
        cluster_type="DECODE_ATTN",
        model_is_moe=True,
        enable_dummy_mode=False,
    )
    validate_profile_backed_runtime_tp_sizes(
        SimpleNamespace(attn_tensor_parallel_size=0, moe_tensor_parallel_size=2),
        cluster_type="DECODE_FFN",
        model_is_moe=True,
        enable_dummy_mode=False,
    )


def test_dense_pdaf_decode_ffn_validates_attention_tp_for_linear_profiles() -> None:
    with pytest.raises(ValueError, match="decode_ffn attn_tensor_parallel_size.*supported TP sizes"):
        validate_profile_backed_runtime_tp_sizes(
            SimpleNamespace(attn_tensor_parallel_size=16, moe_tensor_parallel_size=0),
            cluster_type="DECODE_FFN",
            model_is_moe=False,
            enable_dummy_mode=False,
        )

    validate_profile_backed_runtime_tp_sizes(
        SimpleNamespace(attn_tensor_parallel_size=2, moe_tensor_parallel_size=0),
        cluster_type="DECODE_FFN",
        model_is_moe=False,
        enable_dummy_mode=False,
    )


def test_cluster_config_validation_rejects_profile_unsupported_compute_tp() -> None:
    from frontier.config.config import ClusterConfig

    config = ClusterConfig.__new__(ClusterConfig)
    config.execution_time_predictor_config = SimpleNamespace(enable_dummy_mode=False)
    config_replica = SimpleNamespace(
        attn_tensor_parallel_size=16,
        moe_tensor_parallel_size=1,
        num_pipeline_stages=1,
        model_config=SimpleNamespace(is_moe=False),
    )
    config_replica.model_config.num_layers = 1
    with pytest.raises(ValueError, match="supported TP sizes"):
        config._validate_replica_config(config_replica, "monolithic")


def test_direct_runtime_predictor_rejects_tp_before_base_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from frontier.execution_time_predictor.base_execution_time_predictor import (
        BaseExecutionTimePredictor,
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _Predictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    monkeypatch.setattr(
        BaseExecutionTimePredictor,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail(
            "base predictor initialized before TP validation"
        ),
    )
    replica_config = SimpleNamespace(
        attn_tensor_parallel_size=16,
        moe_tensor_parallel_size=1,
        model_config=SimpleNamespace(is_moe=False),
    )
    predictor_config = SimpleNamespace(enable_dummy_mode=False)

    with pytest.raises(ValueError, match="supported TP sizes"):
        _Predictor.__init__(
            _Predictor.__new__(_Predictor),
            predictor_config,
            replica_config,
            SimpleNamespace(),
            SimpleNamespace(),
        )


def test_on_demand_requires_explicit_domain_policy_before_predict() -> None:
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _Predictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    class _Model:
        n_features_in_ = 1

        def predict(self, _features):
            raise AssertionError("domain validation must run before predict")

    predictor = _Predictor.__new__(_Predictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._runtime_cache = {"eager": {"mixed": {}}}
    model = _Model()
    model._frontier_feature_domain = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "feature_names": ["x"],
        "domain_kind": "integer_interval_interpolation",
        "bounds": {"x": {"min": 1.0, "max": 4.0}},
    }
    predictor._predictions = {
        "mixed": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": ["x"],
        }
    }

    with pytest.raises(ValueError, match="explicit.*domain policy|domain policy"):
        predictor._get_on_demand_prediction("mixed", {"x": 8.0})


def test_on_demand_bounded_policy_rejects_domain_outside_without_cache_or_predict() -> None:
    from frontier.execution_time_predictor.prediction_cache_contract import (
        ON_DEMAND_DOMAIN_POLICY_BOUNDED,
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _Predictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    class _Model:
        n_features_in_ = 1
        calls = 0

        def predict(self, _features):
            self.calls += 1
            return [1.0]

    predictor = _Predictor.__new__(_Predictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._runtime_cache = {"eager": {"mixed": {}}}
    model = _Model()
    model._frontier_feature_domain = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "feature_names": ["x"],
        "domain_kind": "integer_interval_interpolation",
        "on_demand_policy": ON_DEMAND_DOMAIN_POLICY_BOUNDED,
        "bounds": {"x": {"min": 1.0, "max": 4.0}},
    }
    predictor._predictions = {
        "mixed": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": ["x"],
        }
    }

    with pytest.raises(ValueError, match="exceeds profile domain"):
        predictor._get_on_demand_prediction("mixed", {"x": 8.0})
    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["mixed"] == {}


def test_on_demand_unbounded_policy_must_be_explicit() -> None:
    from frontier.execution_time_predictor.prediction_cache_contract import (
        ON_DEMAND_DOMAIN_POLICY_UNBOUNDED,
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _Predictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    class _Model:
        n_features_in_ = 1
        calls = 0

        def predict(self, _features):
            self.calls += 1
            return [2.0]

    predictor = _Predictor.__new__(_Predictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._runtime_cache = {"eager": {"mixed": {}}}
    model = _Model()
    model._frontier_feature_domain = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "feature_names": ["x"],
        "domain_kind": "integer_interval_interpolation",
        "on_demand_policy": ON_DEMAND_DOMAIN_POLICY_UNBOUNDED,
        "bounds": {"x": {"min": 1.0, "max": 4.0}},
    }
    predictor._predictions = {
        "mixed": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": ["x"],
        }
    }

    assert predictor._get_on_demand_prediction("mixed", {"x": 8.0}) == 2.0
    assert model.calls == 1


def test_bounded_on_demand_exact_lookup_still_validates_domain_without_runtime_write() -> None:
    from frontier.execution_time_predictor.prediction_cache_contract import (
        ON_DEMAND_DOMAIN_POLICY_BOUNDED,
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _Predictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    class _Model:
        n_features_in_ = 1
        calls = 0

        def predict(self, _features):
            self.calls += 1
            return [3.0]

    predictor = _Predictor.__new__(_Predictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._runtime_cache = {"eager": {"mixed": {}}}
    model = _Model()
    model._frontier_feature_domain = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "feature_names": ["x"],
        "domain_kind": "integer_interval_interpolation",
        "on_demand_policy": ON_DEMAND_DOMAIN_POLICY_BOUNDED,
        "bounds": {"x": {"min": 1.0, "max": 4.0}},
    }
    predictor._predictions = {
        "mixed": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": ["x"],
            "_exact_lookup": {(2,): 1.5},
        }
    }

    assert predictor._get_on_demand_prediction("mixed", {"x": 2.0}) == 1.5
    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["mixed"] == {}


def test_bounded_on_demand_sparse_domain_rejects_unmeasured_tuple() -> None:
    """Document the conservative exact-row interpretation for sparse domains."""
    from frontier.execution_time_predictor.prediction_cache_contract import (
        ON_DEMAND_DOMAIN_POLICY_BOUNDED,
        attach_feature_domain,
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _Predictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    class _Model:
        n_features_in_ = 2
        calls = 0

        def predict(self, _features):
            self.calls += 1
            return [1.0]

    predictor = _Predictor.__new__(_Predictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._runtime_cache = {"eager": {"mixed": {}}}
    model = _Model()
    attach_feature_domain(
        model,
        pd.DataFrame({"x": [1, 2], "y": [10, 20]}),
        ["x", "y"],
        on_demand_policy=ON_DEMAND_DOMAIN_POLICY_BOUNDED,
    )
    predictor._predictions = {
        "mixed": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": ["x", "y"],
        }
    }

    with pytest.raises(ValueError, match="exceeds profile domain|unmeasured"):
        predictor._get_on_demand_prediction("mixed", {"x": 1.0, "y": 20.0})
    assert model.calls == 0


def test_on_demand_can_validate_record_level_domain_metadata() -> None:
    from frontier.execution_time_predictor.prediction_cache_contract import (
        ON_DEMAND_DOMAIN_POLICY_BOUNDED,
        build_feature_domain_descriptor,
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _Predictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    class _Model:
        n_features_in_ = 1

        def predict(self, _features):
            return [2.0]

    predictor = _Predictor.__new__(_Predictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._runtime_cache = {"eager": {"mixed": {}}}
    model = _Model()
    domain = build_feature_domain_descriptor(
        pd.DataFrame({"x": [1, 2, 4]}),
        ["x"],
        on_demand_policy=ON_DEMAND_DOMAIN_POLICY_BOUNDED,
    )
    predictor._predictions = {
        "mixed": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": ["x"],
            "_feature_domain": domain,
        }
    }

    assert predictor._get_on_demand_prediction("mixed", {"x": 2.0}) == 2.0


@pytest.mark.parametrize("invalid_tp", [0, -1, 16, 32])
def test_linear_tp_ranges_reject_unsupported_values(invalid_tp: int) -> None:
    args = SimpleNamespace(
        num_tensor_parallel_workers=[1, invalid_tp],
        attn_tp=None,
        ffn_tp=None,
    )

    with pytest.raises(ValueError, match="supported TP sizes"):
        linear_main._resolve_tp_ranges(args)  # pylint: disable=protected-access


def test_linear_parse_rejects_unsupported_tp_before_gpu_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "linear-profile",
            "--models",
            "test-model",
            "--num_tensor_parallel_workers",
            "16",
            "--output_dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="supported TP sizes"):
        linear_main.parse_args()


def test_linear_programmatic_profile_rejects_unsupported_tp_before_gpu_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "frontier.profiling.linear_op.linear_op_wrapper",
        SimpleNamespace(LinearOpWrapper=object),
    )
    monkeypatch.setattr(
        linear_main.ModelConfig,
        "from_model_name",
        staticmethod(lambda _model: SimpleNamespace()),
    )
    monkeypatch.setattr(
        linear_main,
        "_resolve_precision_for_model",
        lambda *_args: ("unused", "unused"),
    )
    monkeypatch.setattr(linear_main, "_resolve_fp8_settings", lambda *_args: None)
    monkeypatch.setattr(
        linear_main,
        "_get_available_gpus",
        lambda _num_gpus: pytest.fail("GPU discovery ran before TP validation"),
    )
    args = SimpleNamespace(
        num_tensor_parallel_workers=[16],
        attn_tp=None,
        ffn_tp=None,
        precision="auto",
        use_fp8=None,
        block_shape=None,
        profile_method="cuda_event",
        num_gpus=1,
    )

    with pytest.raises(ValueError, match="supported TP sizes"):
        linear_main.profile_model(args, "test-model", [1], SimpleNamespace())


@pytest.mark.parametrize("field_name", ["attn_tp", "ffn_tp"])
def test_linear_explicit_operator_tp_ranges_are_strict(
    field_name: str,
) -> None:
    args = SimpleNamespace(
        num_tensor_parallel_workers=[1],
        attn_tp=[1],
        ffn_tp=[1],
    )
    setattr(args, field_name, [16])

    with pytest.raises(ValueError, match="supported TP sizes"):
        linear_main._resolve_tp_ranges(args)  # pylint: disable=protected-access


def test_linear_tp_ranges_preserve_release_supported_values() -> None:
    args = SimpleNamespace(
        num_tensor_parallel_workers=[1, 2, 4, 8],
        attn_tp=[2, 8],
        ffn_tp=[1, 4],
    )

    attn_tp, ffn_tp, all_tps = linear_main._resolve_tp_ranges(args)  # pylint: disable=protected-access

    assert attn_tp == [2, 8]
    assert ffn_tp == [1, 4]
    assert all_tps == [1, 2, 4, 8]


def test_moe_parse_rejects_unsupported_tp_before_gpu_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "moe-profile",
            "--device",
            "h800",
            "--models",
            "test-model",
            "--num_tensor_parallel_workers",
            "16",
            "--output_dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="supported TP sizes"):
        moe_main.parse_args()


def test_moe_parse_preserves_release_supported_tp_sizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "moe-profile",
            "--device",
            "h800",
            "--models",
            "test-model",
            "--num_tensor_parallel_workers",
            "1",
            "2",
            "4",
            "8",
            "--output_dir",
            str(tmp_path),
        ],
    )

    args, _enable_load_imbalance = moe_main.parse_args()

    assert args.num_tensor_parallel_workers == [1, 2, 4, 8]


@pytest.mark.parametrize("invalid_tp", [0, -1, 16, 32])
def test_attention_cli_rejects_unsupported_tp_values(invalid_tp: int) -> None:
    args = SimpleNamespace(
        profile_only_prefill=False,
        profile_only_decode=False,
        decode_kv_cache_size_list=None,
        enable_true_mixed=False,
        attention_backend="FLASHINFER",
        vllm_mla_cuda_op_log=None,
        models=["test-model"],
        num_tensor_parallel_workers=[invalid_tp],
        enable_mixed_prefill=False,
    )

    with pytest.raises(ValueError, match="supported TP sizes"):
        attention_main._validate_cli_conflicts(args)  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "entrypoint_name",
    ["profile_model", "profile_mixed_prefill", "profile_true_mixed_batches"],
)
def test_attention_programmatic_profiles_reject_unsupported_tp_before_gpu_setup(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint_name: str,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "frontier.profiling.attention.attention_wrapper",
        SimpleNamespace(AttentionWrapper=object),
    )
    monkeypatch.setattr(
        attention_main.ModelConfig,
        "from_model_name",
        staticmethod(lambda _model: SimpleNamespace()),
    )
    monkeypatch.setattr(
        attention_main,
        "_resolve_precision_for_model",
        lambda *_args: ("unused", "unused"),
    )
    monkeypatch.setattr(
        attention_main,
        "_get_available_gpus",
        lambda _num_gpus: pytest.fail("GPU discovery ran before TP validation"),
    )
    args = SimpleNamespace(precision="auto", num_gpus=1)
    entrypoint = getattr(attention_main, entrypoint_name)

    with pytest.raises(ValueError, match="supported TP sizes"):
        entrypoint(
            args,
            "test-model",
            16,
            [],
            128,
            None,
            SimpleNamespace(),
        )


def test_attention_gpu_local_indices_follow_explicit_visible_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")

    assert attention_main._get_gpu_local_index_map([3, 5]) == {3: 0, 5: 1}


def test_attention_gpu_local_indices_match_physical_ids_without_visible_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert attention_main._get_gpu_local_index_map([3, 5]) == {3: 3, 5: 5}


def test_profile_tp_contract_is_explicit() -> None:
    assert SUPPORTED_PROFILE_TP_SIZES == (1, 2, 4, 8)


def test_profile_tp_contract_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_profile_tp_sizes([1, 2, 2], argument_name="profile TP")


def test_parallel_config_rejects_unsupported_profile_tp() -> None:
    from frontier.profiling.common.parallel_config import ParallelConfig

    with pytest.raises(ValueError, match="supported TP sizes"):
        ParallelConfig(tensor_parallel_size=16)


@pytest.mark.parametrize("invalid_tp", [0, -1, 16, 32])
def test_runtime_profile_backed_tp_contract_rejects_unsupported_values(
    invalid_tp: int,
) -> None:
    with pytest.raises(ValueError, match="supported TP sizes"):
        validate_profile_tp_sizes([invalid_tp], argument_name="runtime TP")


def test_shared_profile_backed_runtime_rejects_tp_before_training() -> None:
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._all_dummy_mode = False
    manager._cluster_configs = {
        "probe": SimpleNamespace(
            replica_config=SimpleNamespace(
                attn_tensor_parallel_size=16,
                moe_tensor_parallel_size=1,
            )
        )
    }

    with pytest.raises(ValueError, match="supported TP sizes"):
        manager._validate_profile_backed_runtime_tp_sizes()


def test_attention_parse_defaults_profile_limit_to_input_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attention-profile",
            "--models",
            "test-model",
            "--num_tensor_parallel_workers",
            "1",
            "--max_model_len",
            "4096",
            "--max_seq_len",
            "8192",
            "--output_dir",
            str(tmp_path),
        ],
    )

    args = attention_main.parse_args()

    assert args.max_model_len == 4096
    assert args.max_seq_len == 8192
    assert args.profile_max_seq_len == 8192


def test_attention_parse_accepts_explicit_block_allocation_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attention-profile",
            "--models",
            "test-model",
            "--num_tensor_parallel_workers",
            "1",
            "--max_seq_len",
            "129",
            "--profile_max_seq_len",
            "129",
            "--max_num_blocks",
            "18",
            "--output_dir",
            str(tmp_path),
        ],
    )

    args = attention_main.parse_args()

    assert args.max_num_blocks == 18


def test_attention_parse_rejects_non_positive_block_allocation_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attention-profile",
            "--models",
            "test-model",
            "--num_tensor_parallel_workers",
            "1",
            "--max_num_blocks",
            "0",
            "--output_dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="max_num_blocks.*positive"):
        attention_main.parse_args()


def test_attention_parse_rejects_profile_limit_smaller_than_input_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attention-profile",
            "--models",
            "test-model",
            "--num_tensor_parallel_workers",
            "1",
            "--max_seq_len",
            "8192",
            "--profile_max_seq_len",
            "4096",
            "--output_dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="cannot exceed profile_max_seq_len"):
        attention_main.parse_args()


def test_attention_parse_rejects_non_positive_serving_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attention-profile",
            "--models",
            "test-model",
            "--num_tensor_parallel_workers",
            "1",
            "--max_model_len",
            "0",
            "--output_dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="max_model_len must be positive"):
        attention_main.parse_args()


def test_attention_parse_accepts_artifact_scoped_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attention-profile",
            "--models",
            "test-model",
            "--num_tensor_parallel_workers",
            "1",
            "--run_id",
            "h800-probe-a",
            "--output_dir",
            str(tmp_path),
        ],
    )
    args = attention_main.parse_args()
    assert args.run_id == "h800-probe-a"


def test_get_max_num_blocks_rejects_uneven_pipeline_layers_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.common.parallel_config import ParallelConfig
    from frontier.profiling.utils import get_max_num_blocks

    model_config = ModelConfig(
        name="capacity-contract-test",
        num_layers=5,
        num_q_heads=4,
        num_kv_heads=4,
        embedding_dim=256,
        mlp_hidden_dim=512,
        max_position_embeddings=8192,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=False,
        vocab_size=1024,
        head_dim=64,
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (10**12, 10**12))
    with pytest.raises(ValueError, match="evenly divisible"):
        get_max_num_blocks(
            model_config,
            ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1),
            block_size=16,
            dtype=torch.float16,
            max_pipeline_parallel_size=2,
        )


def test_attention_confirmation_reports_serving_and_profile_limits() -> None:
    model_config = ModelConfig(
        name="context-confirmation-test",
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=4,
        embedding_dim=256,
        mlp_hidden_dim=512,
        max_position_embeddings=8192,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=False,
        vocab_size=1024,
        head_dim=64,
    )
    args = SimpleNamespace(
        disable_ray=True,
        num_gpus=1,
        output_dir="out",
        profile_method="cuda_event",
        attention_backend="FLASHINFER",
        num_tensor_parallel_workers=[1],
        use_fp8=False,
        block_shape=None,
        max_model_len=4096,
        max_seq_len=8192,
        profile_max_seq_len=8192,
        min_batch_size=1,
        max_batch_size=8,
        block_size=16,
        profile_only_prefill=False,
        profile_only_decode=False,
        enable_mixed_prefill=False,
    )

    sections = dict(
        build_attention_config_sections(
            args,
            model_config,
            input_combinations_count=1,
            mixed_combinations_count=0,
            true_mixed_combinations_count=0,
            precision_str="BF16",
            torch_dtype="torch.bfloat16",
        )
    )

    profiling_range = dict(sections["Profiling Range"])
    assert profiling_range["Max Model Length"] == "4096"
    assert profiling_range["Max Sequence Length"] == "8192"
    assert profiling_range["Profile Max Sequence Length"] == "8192"


def test_attention_wrapper_uses_profile_context_limit_for_validation_and_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile may exceed serving max_model_len when explicitly requested."""

    pytest.importorskip("torch")

    import frontier.profiling.attention.attention_wrapper as wrapper_module
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    class FakeBackend:
        def supports_attention_family(self, _family) -> bool:
            return True

        def init(self, *_args) -> None:
            return None

        def get_cache_block(self, num_blocks, **kwargs):
            return (num_blocks, kwargs)

    fake_backend = FakeBackend()
    monkeypatch.setattr(wrapper_module, "set_attention_backend", lambda _backend: None)
    monkeypatch.setattr(wrapper_module, "get_attention_wrapper", lambda: fake_backend)
    monkeypatch.setattr(
        wrapper_module,
        "configure_quantization_manager_for_model_name",
        lambda _name: None,
    )
    monkeypatch.setattr(wrapper_module.torch, "device", lambda name: name)

    from frontier.profiling.common.parallel_config import ParallelConfig

    model_config = ModelConfig(
        name="context-contract-test",
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=4,
        embedding_dim=256,
        mlp_hidden_dim=512,
        max_position_embeddings=8192,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=False,
        vocab_size=1024,
        head_dim=64,
    )

    wrapper = AttentionWrapper(
        model_config=model_config,
        parallel_config=ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1),
        max_num_blocks=128,
        max_model_len=4096,
        profile_max_seq_len=8192,
        block_size=64,
        attention_backend="FLASHINFER",
        dtype="bfloat16",
        profile_method="cuda_event",
        output_dir="unused",
    )

    assert wrapper._max_model_len == 4096
    assert wrapper._profile_max_seq_len == 8192
    assert wrapper._max_blocks_per_sequence == 128
    assert wrapper._is_valid_attention_input(
        AttentionInput(
            prefill_chunk_size=4096,
            kv_cache_size=4096,
            batch_size=1,
            is_prefill=True,
        )
    )


def test_attention_wrapper_rejects_insufficient_physical_kv_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    import frontier.profiling.attention.attention_wrapper as wrapper_module
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    class FakeBackend:
        def __init__(self) -> None:
            self.init_called = False

        def supports_attention_family(self, _family) -> bool:
            return True

        def init(self, *_args) -> None:
            self.init_called = True

        def get_cache_block(self, *_args, **_kwargs):
            pytest.fail("KV cache allocated before physical capacity validation")

    fake_backend = FakeBackend()
    monkeypatch.setattr(wrapper_module, "set_attention_backend", lambda _backend: None)
    monkeypatch.setattr(wrapper_module, "get_attention_wrapper", lambda: fake_backend)
    monkeypatch.setattr(
        wrapper_module,
        "configure_quantization_manager_for_model_name",
        lambda _name: None,
    )
    monkeypatch.setattr(wrapper_module.torch, "device", lambda name: name)

    from frontier.profiling.common.parallel_config import ParallelConfig

    model_config = ModelConfig(
        name="physical-capacity-test",
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=4,
        embedding_dim=256,
        mlp_hidden_dim=512,
        max_position_embeddings=8192,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=False,
        vocab_size=1024,
        head_dim=64,
    )

    with pytest.raises(
        ValueError,
        match=r"Physical KV-cache capacity.*max_num_blocks=127.*required_blocks_per_sequence=128",
    ):
        AttentionWrapper(
            model_config=model_config,
            parallel_config=ParallelConfig(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
            ),
            max_num_blocks=127,
            max_model_len=4096,
            profile_max_seq_len=8192,
            block_size=64,
            attention_backend="FLASHINFER",
            dtype="bfloat16",
            profile_method="cuda_event",
            output_dir="unused",
        )

    assert fake_backend.init_called is False


def test_attention_input_required_blocks_account_for_per_sequence_fragmentation_and_decode_token() -> None:
    """Each sequence consumes ceil(total_len / block_size), including decode's new token."""

    decode_batch = AttentionInput(
        prefill_chunk_size=0,
        kv_cache_size=16,
        batch_size=2,
        is_prefill=False,
    )

    assert decode_batch.required_blocks(block_size=16) == 4
    assert decode_batch.is_under_memory_limit(max_num_blocks=3, block_size=16) is False
    assert decode_batch.is_under_memory_limit(max_num_blocks=4, block_size=16) is True


def test_mixed_attention_required_blocks_sum_fragmented_sequences() -> None:
    mixed_input = MixedAttentionInput(seq_lens=[1, 16], kv_cache_size=0)

    # 1-token sequence and 16-token sequence each occupy one block.
    assert mixed_input.required_blocks(block_size=16) == 2
    assert mixed_input.is_under_memory_limit(max_num_blocks=1, block_size=16) is False


def test_true_mixed_required_blocks_use_independent_prefill_and_decode_lengths() -> None:
    true_mixed_input = TrueMixedBatchInput(
        prefill_seq_lens=[1],
        prefill_kv_cache_sizes=[0],
        decode_kv_cache_sizes=[16],
    )

    # Prefill: ceil(1 / 16) = 1; decode: ceil((16 + 1) / 16) = 2.
    assert true_mixed_input.required_blocks(block_size=16) == 3
    assert true_mixed_input.is_under_memory_limit(max_num_blocks=2, block_size=16) is False


@pytest.mark.parametrize(
    ("workload_family", "attention_inputs", "required_blocks"),
    [
        (
            "standard",
            [AttentionInput(0, 0, 2, False)],
            2,
        ),
        (
            "mixed_prefill",
            [MixedAttentionInput(seq_lens=[1, 1], kv_cache_size=0)],
            2,
        ),
        (
            "true_mixed",
            [
                TrueMixedBatchInput(
                    prefill_seq_lens=[1],
                    prefill_kv_cache_sizes=[0],
                    decode_kv_cache_sizes=[0],
                )
            ],
            2,
        ),
    ],
)
def test_requested_attention_capacity_is_never_silently_filtered(
    workload_family: str,
    attention_inputs: list[object],
    required_blocks: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            rf"model='capacity-probe'.*TP=1.*{workload_family}.*"
            rf"required_blocks={required_blocks}.*available_blocks=1"
        ),
    ):
        attention_main._require_inputs_fit_kv_capacity(
            attention_inputs,
            model="capacity-probe",
            tensor_parallel_size=1,
            workload_family=workload_family,
            max_num_blocks=1,
            block_size=16,
            profile_max_seq_len=16,
        )


def test_kv_block_budget_uses_free_memory_and_reserves_backend_workspace() -> None:
    gib = 1024**3
    mib = 1024**2

    assert calculate_max_num_blocks_from_memory(
        free_memory_bytes=80 * gib,
        block_memory_bytes=1 * mib,
        gpu_memory_utilization=0.9,
        reserved_memory_bytes=9 * gib,
    ) == 63 * 1024


def test_kv_block_budget_fails_when_reservation_consumes_usable_memory() -> None:
    gib = 1024**3

    with pytest.raises(ValueError, match="reserved_memory_bytes.*usable GPU memory"):
        calculate_max_num_blocks_from_memory(
            free_memory_bytes=8 * gib,
            block_memory_bytes=1 * 1024**2,
            gpu_memory_utilization=0.9,
            reserved_memory_bytes=9 * gib,
        )


def test_flashinfer_memory_budget_reserves_both_float_and_int_workspaces() -> None:
    assert get_attention_backend_workspace_reservation_bytes(
        "FLASHINFER",
        environ={
            "FRONTIER_FLASHINFER_WORKSPACE_GB": "4",
            "FRONTIER_FLASHINFER_INT_WORKSPACE_MB": "512",
        },
    ) == 9 * 1024**3


def test_native_attention_rows_record_requested_grid_and_physical_capacity() -> None:
    source = pd.DataFrame({"batch_size": [1], "kv_cache_size": [64]})

    output = attention_main._attach_native_attention_run_provenance(
        source,
        profile_input_grid_max_seq_len=96,
        profile_max_seq_len=128,
        max_num_blocks=10,
        block_size=16,
        backend_workspace_reservation_bytes=9 * 1024**3,
    )

    assert output.loc[0, "profile_input_grid_max_seq_len"] == 96
    assert output.loc[0, "profile_max_seq_len"] == 128
    assert output.loc[0, "allocated_max_num_blocks"] == 10
    assert output.loc[0, "allocated_kv_token_capacity"] == 160
    assert output.loc[0, "backend_workspace_reservation_bytes"] == 9 * 1024**3
    assert bool(output.loc[0, "is_native_profile_allocation"]) is True


def test_requested_attention_block_cap_is_validated_against_physical_and_profile_bounds() -> None:
    assert resolve_requested_max_num_blocks(
        physical_max_num_blocks=100,
        requested_max_num_blocks=None,
        profile_max_seq_len=129,
        block_size=16,
    ) == 100
    assert resolve_requested_max_num_blocks(
        physical_max_num_blocks=100,
        requested_max_num_blocks=18,
        profile_max_seq_len=129,
        block_size=16,
    ) == 18

    with pytest.raises(ValueError, match="physical maximum"):
        resolve_requested_max_num_blocks(
            physical_max_num_blocks=17,
            requested_max_num_blocks=18,
            profile_max_seq_len=129,
            block_size=16,
        )
    with pytest.raises(ValueError, match="profile_max_seq_len"):
        resolve_requested_max_num_blocks(
            physical_max_num_blocks=100,
            requested_max_num_blocks=8,
            profile_max_seq_len=129,
            block_size=16,
        )


def test_requested_attention_block_cap_covers_largest_requested_batch() -> None:
    with pytest.raises(ValueError, match="required_max_num_blocks=18"):
        resolve_requested_max_num_blocks(
            physical_max_num_blocks=100,
            requested_max_num_blocks=9,
            required_max_num_blocks=18,
            profile_max_seq_len=129,
            block_size=16,
        )


def test_required_attention_block_budget_uses_all_requested_shapes() -> None:
    inputs = [
        AttentionInput(0, 0, 1, False),
        AttentionInput(0, 128, 2, False),
    ]

    assert attention_main._required_max_num_blocks(inputs, block_size=16) == 18


def test_native_attention_provenance_records_numeric_block_budget() -> None:
    source = pd.DataFrame({"batch_size": [1], "kv_cache_size": [64]})

    output = attention_main._attach_native_attention_run_provenance(
        source,
        profile_input_grid_max_seq_len=96,
        profile_max_seq_len=128,
        physical_max_num_blocks=100,
        requested_max_num_blocks=None,
        selected_max_num_blocks=18,
        required_max_num_blocks=17,
        block_size=16,
        backend_workspace_reservation_bytes=9 * 1024**3,
    )

    assert output.loc[0, "physical_max_num_blocks"] == 100
    assert pd.isna(output.loc[0, "requested_max_num_blocks"])
    assert output.loc[0, "selected_max_num_blocks"] == 18
    assert output.loc[0, "required_max_num_blocks"] == 17
    assert output.loc[0, "allocated_max_num_blocks"] == 18
    assert output.loc[0, "allocated_kv_token_capacity"] == 288
    assert output["requested_max_num_blocks"].dtype == "Int64"


def test_standard_attention_generator_does_not_repeat_full_prefill_tuple() -> None:
    from frontier.profiling.utils import get_attention_input_combinations

    inputs = get_attention_input_combinations(
        max_seq_len=128,
        min_batch_size=1,
        max_batch_size=1,
        profile_only_prefill=False,
        profile_only_decode=False,
        batch_size_list=[1],
        enable_chunked_prefill_grid_search=True,
        fixed_chunked_prefill_size=64,
    )
    keys = [
        (
            item.prefill_chunk_size,
            item.kv_cache_size,
            item.batch_size,
            item.is_prefill,
        )
        for item in inputs
    ]

    assert len(keys) == len(set(keys))


def test_attention_result_validation_rejects_silent_none_row() -> None:
    inputs = [AttentionInput(0, 64, 1, False)]

    with pytest.raises(ValueError, match="standard.*None.*requested=1.*emitted=0"):
        attention_main._validate_attention_profile_results(
            workload_family="standard",
            requested_inputs=inputs,
            results=[None],
        )


def test_attention_result_validation_matches_and_sorts_structural_tuples() -> None:
    inputs = [
        AttentionInput(0, 64, 2, False),
        AttentionInput(0, 0, 1, False),
    ]
    results = [
        {
            "prefill_chunk_size": 0,
            "kv_cache_size": 64,
            "batch_size": 2,
            "is_prefill": False,
        },
        {
            "prefill_chunk_size": 0,
            "kv_cache_size": 0,
            "batch_size": 1,
            "is_prefill": False,
        },
    ]

    validated = attention_main._validate_attention_profile_results(
        workload_family="standard",
        requested_inputs=inputs,
        results=list(reversed(results)),
    )

    assert [row["kv_cache_size"] for row in validated] == [0, 64]


def test_attention_result_validation_uses_numeric_structural_sort() -> None:
    inputs = [
        AttentionInput(0, 128, 1, False),
        AttentionInput(0, 2, 1, False),
        AttentionInput(0, 64, 1, False),
    ]
    results = [
        {
            "prefill_chunk_size": item.prefill_chunk_size,
            "kv_cache_size": item.kv_cache_size,
            "batch_size": item.batch_size,
            "is_prefill": item.is_prefill,
        }
        for item in inputs
    ]

    validated = attention_main._validate_attention_profile_results(
        workload_family="standard",
        requested_inputs=inputs,
        results=results,
    )

    assert [row["kv_cache_size"] for row in validated] == [2, 64, 128]


def test_native_attention_provenance_rejects_legacy_string_block_sentinel() -> None:
    source = pd.DataFrame(
        {
            "batch_size": [1],
            "kv_cache_size": [64],
            "requested_max_num_blocks": ["physical_max"],
        }
    )

    with pytest.raises(ValueError, match="requested_max_num_blocks.*numeric"):
        attention_main._attach_native_attention_run_provenance(
            source,
            profile_input_grid_max_seq_len=96,
            profile_max_seq_len=128,
            physical_max_num_blocks=100,
            requested_max_num_blocks=None,
            selected_max_num_blocks=18,
            required_max_num_blocks=17,
            block_size=16,
            backend_workspace_reservation_bytes=9 * 1024**3,
        )


def test_native_attention_provenance_accepts_integral_numpy_values() -> None:
    import numpy as np

    output = attention_main._attach_native_attention_run_provenance(
        pd.DataFrame({"row_id": [1]}),
        profile_input_grid_max_seq_len=96,
        profile_max_seq_len=128,
        physical_max_num_blocks=np.int64(100),
        requested_max_num_blocks=np.int64(18),
        selected_max_num_blocks=np.int64(18),
        required_max_num_blocks=np.int64(17),
        block_size=16,
        backend_workspace_reservation_bytes=9 * 1024**3,
    )

    assert output.loc[0, "physical_max_num_blocks"] == 100
    assert output.loc[0, "requested_max_num_blocks"] == 18
    assert output.loc[0, "selected_max_num_blocks"] == 18
    assert output.loc[0, "required_max_num_blocks"] == 17


@pytest.mark.parametrize(
    ("physical", "requested", "selected", "required"),
    [
        (17, None, 18, 17),
        (100, None, 16, 17),
        (100, 19, 18, 17),
        (True, None, 1, 1),
        (100, None, 18.5, 17),
    ],
)
def test_native_attention_provenance_rejects_invalid_block_budget(
    physical,
    requested,
    selected,
    required,
) -> None:
    with pytest.raises(ValueError, match="block|physical|selected|required|requested"):
        attention_main._attach_native_attention_run_provenance(
            pd.DataFrame({"row_id": [1]}),
            profile_input_grid_max_seq_len=96,
            profile_max_seq_len=128,
            physical_max_num_blocks=physical,
            requested_max_num_blocks=requested,
            selected_max_num_blocks=selected,
            required_max_num_blocks=required,
            block_size=16,
            backend_workspace_reservation_bytes=9 * 1024**3,
        )


def test_attention_result_validation_rejects_missing_extra_and_duplicates() -> None:
    inputs = [
        AttentionInput(0, 0, 1, False),
        AttentionInput(0, 64, 1, False),
    ]
    duplicated_zero = {
        "prefill_chunk_size": 0,
        "kv_cache_size": 0,
        "batch_size": 1,
        "is_prefill": False,
    }

    with pytest.raises(
        ValueError,
        match="requested=2.*emitted=2.*missing=1.*extra=1",
    ):
        attention_main._validate_attention_profile_results(
            workload_family="standard",
            requested_inputs=inputs,
            results=[duplicated_zero, dict(duplicated_zero)],
        )


def test_attention_result_validation_supports_mixed_structural_keys() -> None:
    mixed = MixedAttentionInput(seq_lens=[16, 32], kv_cache_size=8, mode="even")
    true_mixed = TrueMixedBatchInput(
        prefill_seq_lens=[16],
        prefill_kv_cache_sizes=[0],
        decode_kv_cache_sizes=[32],
    )

    assert attention_main._validate_attention_profile_results(
        workload_family="mixed_prefill",
        requested_inputs=[mixed],
        results=[mixed.to_dict()],
    )
    assert attention_main._validate_attention_profile_results(
        workload_family="true_mixed",
        requested_inputs=[true_mixed],
        results=[true_mixed.to_dict()],
    )

@pytest.mark.parametrize(
    (
        "prefill_chunk_sizes",
        "decode_kv_cache_sizes",
        "prefill_kv_cache_size",
        "expected_message",
    ),
    [
        ([129], [0], 0, "prefill total_len=129"),
        ([1], [128], 0, "decode total_len=129"),
        ([65], [0], 64, "prefill total_len=129"),
    ],
)
def test_true_mixed_explicit_out_of_context_shapes_fail_fast(
    prefill_chunk_sizes: list[int],
    decode_kv_cache_sizes: list[int],
    prefill_kv_cache_size: int,
    expected_message: str,
) -> None:
    from frontier.profiling.utils import (
        get_true_mixed_attention_input_combinations,
    )

    with pytest.raises(ValueError, match=expected_message):
        get_true_mixed_attention_input_combinations(
            max_seq_len=128,
            prefill_batch_sizes=[1],
            prefill_chunk_sizes=prefill_chunk_sizes,
            decode_batch_sizes=[1],
            decode_kv_cache_sizes=decode_kv_cache_sizes,
            prefill_kv_cache_size=prefill_kv_cache_size,
        )


@pytest.mark.parametrize("max_num_blocks", [0, -1, True, 127])
def test_profile_kv_capacity_rejects_insufficient_or_invalid_allocation(
    max_num_blocks,
) -> None:
    assert hasattr(attention_input_module, "validate_profile_kv_capacity")
    with pytest.raises(ValueError, match="Physical KV-cache capacity|max_num_blocks"):
        attention_input_module.validate_profile_kv_capacity(
            max_num_blocks=max_num_blocks,
            profile_max_seq_len=8192,
            block_size=64,
        )


def test_profile_kv_capacity_accepts_exact_required_allocation() -> None:
    assert hasattr(attention_input_module, "validate_profile_kv_capacity")
    assert attention_input_module.validate_profile_kv_capacity(
        max_num_blocks=128,
        profile_max_seq_len=8192,
        block_size=64,
    ) == 128


def test_attention_input_decode_requires_room_for_the_new_token() -> None:
    assert not AttentionInput(
        prefill_chunk_size=0,
        kv_cache_size=8192,
        batch_size=1,
        is_prefill=False,
    ).is_valid(8192)


def test_explicit_decode_grid_rejects_context_full_kv_cache() -> None:
    from frontier.profiling.utils import get_attention_input_combinations

    with pytest.raises(ValueError, match="leave room for the new decode token"):
        get_attention_input_combinations(
            max_seq_len=8192,
            min_batch_size=1,
            max_batch_size=1,
            profile_only_prefill=False,
            profile_only_decode=True,
            decode_kv_cache_size_list=[8192],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"batch_size_list": [1.5]},
            "batch_size_list.*integer",
        ),
        (
            {"decode_kv_cache_size_list": [1.5]},
            "decode_kv_cache_size_list.*integer",
        ),
    ],
)
def test_attention_profile_lists_reject_fractional_programmatic_values(
    kwargs: dict[str, object], message: str
) -> None:
    from frontier.profiling.utils import get_attention_input_combinations

    with pytest.raises(ValueError, match=message):
        get_attention_input_combinations(
            max_seq_len=128,
            min_batch_size=1,
            max_batch_size=2,
            profile_only_prefill=False,
            profile_only_decode=True,
            **kwargs,
        )


def test_explicit_prefill_chunk_larger_than_context_is_not_clamped() -> None:
    from frontier.profiling.utils import get_attention_input_combinations

    with pytest.raises(ValueError, match="fixed_chunked_prefill_size.*max_seq_len"):
        get_attention_input_combinations(
            max_seq_len=128,
            min_batch_size=1,
            max_batch_size=1,
            profile_only_prefill=True,
            profile_only_decode=False,
            fixed_chunked_prefill_size=256,
        )


def test_explicit_mixed_kv_context_without_room_for_prefill_fails_fast() -> None:
    from frontier.profiling.utils import get_mixed_prefill_input_combinations

    with pytest.raises(ValueError, match="kv_cache_sizes.*leave room"):
        get_mixed_prefill_input_combinations(
            max_seq_len=128,
            min_batch_size=2,
            max_batch_size=2,
            mode="even",
            kv_cache_sizes=[128],
        )


def test_mixed_prefill_generator_emits_unique_structural_workloads() -> None:
    from frontier.profiling.utils import get_mixed_prefill_input_combinations

    inputs = get_mixed_prefill_input_combinations(
        max_seq_len=1024,
        min_batch_size=2,
        max_batch_size=4,
        mode="both",
        num_samples_per_config=3,
    )
    identities = [
        (tuple(item.seq_lens), item.kv_cache_size, item.mode)
        for item in inputs
    ]

    assert len(identities) == len(set(identities))


def test_attention_wrapper_allocates_disjoint_standard_batch_block_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable torch runtime and NVIDIA driver")

    import frontier.profiling.attention.attention_wrapper as wrapper_module
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper
    from frontier.profiling.common.parallel_config import ParallelConfig

    class FakeBackend:
        def supports_attention_family(self, _family) -> bool:
            return True

        def init(self, *_args) -> None:
            return None

        def get_cache_block(self, num_blocks, **kwargs):
            return (num_blocks, kwargs)

    fake_backend = FakeBackend()
    monkeypatch.setattr(wrapper_module, "set_attention_backend", lambda _backend: None)
    monkeypatch.setattr(wrapper_module, "get_attention_wrapper", lambda: fake_backend)
    monkeypatch.setattr(
        wrapper_module,
        "configure_quantization_manager_for_model_name",
        lambda _name: None,
    )
    monkeypatch.setattr(wrapper_module.torch, "device", lambda name: name)

    model_config = ModelConfig(
        name="batch-block-table-test",
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=4,
        embedding_dim=256,
        mlp_hidden_dim=512,
        max_position_embeddings=8192,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=False,
        vocab_size=1024,
        head_dim=64,
    )
    wrapper = AttentionWrapper(
        model_config=model_config,
        parallel_config=ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1),
        max_num_blocks=16,
        max_model_len=512,
        profile_max_seq_len=512,
        block_size=64,
        attention_backend="FLASHINFER",
        dtype="bfloat16",
        profile_method="cuda_event",
        output_dir="unused",
    )
    metadata, *_ = wrapper._get_input_tensors(
        AttentionInput(
            prefill_chunk_size=64,
            kv_cache_size=64,
            batch_size=2,
            is_prefill=True,
        )
    )
    assert metadata[0].block_table == [0, 1]
    assert metadata[1].block_table == [2, 3]

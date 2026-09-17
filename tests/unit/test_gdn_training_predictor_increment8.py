"""CPU Increment 8 contracts for GDN training and artifact-backed prediction."""

from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from frontier.attention.gdn import GatedDeltaNetConfig
from frontier.attention.gdn.features import GDNBatchFeatures
from frontier.config.config import BaseExecutionTimePredictorConfig
from frontier.execution_time_predictor.gdn_predictor import GDNPredictor
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.training.gdn_trainer import GDN_TASKS, GDNTrainer
from frontier.types import MeasurementType


FIXTURE = Path(__file__).parents[1] / "fixtures" / "pr31_hybrid" / "gdn.csv"


class _FixtureModelConfig:
    embedding_dim = 256

    def get_num_gdn_layers(self):
        return 1

    def get_name(self):
        return "fixture"

    def get_model_architecture_profile(self):
        return SimpleNamespace(profile_id="qwen3_5_moe")

    def get_quant_signature(self):
        return "none"

    def get_gdn_config(self):
        return GatedDeltaNetConfig(
            conv_kernel_size=4, key_head_dim=32, value_head_dim=32,
            num_key_heads=2, num_value_heads=4,
        )


def _train(tmp_path: Path, *, measurement_type: str = "DEVICE_EVENT") -> Path:
    output = tmp_path / "gdn_models"
    trainer = GDNTrainer(
        str(FIXTURE),
        str(output),
        model_architecture_profile="qwen3_5_moe",
        quant_signature="none",
        device="cpu",
        tensor_parallel_size=1,
        measurement_type=measurement_type,
        num_estimators=[4],
        max_depth=[4],
        min_samples_split=[2],
        k_fold_cv_splits=2,
        num_training_job_threads=1,
    )
    models = trainer.train()
    assert set(models) == {
        f"{operator}_{phase}" for operator, phase in GDN_TASKS
    }
    assert (output / "gdn_manifest.json").is_file()
    return output


def test_gdn_shared_features_reject_history_as_decode_dimension() -> None:
    batch = SimpleNamespace(
        num_tokens=[1, 1],
        total_num_tokens=2,
        num_prefill_tokens=0,
        num_decode_tokens=2,
        requests=[
            SimpleNamespace(num_processed_tokens=8, is_prefill_complete=True),
            SimpleNamespace(num_processed_tokens=1024, is_prefill_complete=True),
        ],
    )
    features = GDNBatchFeatures.from_batch(batch)
    assert features.phase == "decode"
    assert features.batch_size == 2
    assert features.max_query_len == 1
    assert features.query_len_cv == 0.0
    assert features.num_stateful_requests == 2
    with pytest.raises(ValueError, match=r"prefill\+decode"):
        GDNBatchFeatures.from_batch(
            SimpleNamespace(
                num_tokens=[1, 4],
                total_num_tokens=5,
                num_prefill_tokens=4,
                num_decode_tokens=1,
                requests=[SimpleNamespace(), SimpleNamespace()],
            )
        )


def test_gdn_trainer_writes_six_tasks_and_predictor_loads_them(tmp_path: Path) -> None:
    output = _train(tmp_path)
    predictor = GDNPredictor.from_directory(
        output,
        device="cpu",
        tensor_parallel_size=1,
        measurement_type="DEVICE_EVENT",
        model_architecture_profile="qwen3_5_moe",
        quant_signature="none",
    )
    prefill = predictor.predict_operator_times(
        GDNBatchFeatures(1, 16, 16, 0.0, 0, "prefill")
    )
    decode = predictor.predict_operator_times(
        GDNBatchFeatures(4, 4, 1, 0.0, 4, "decode")
    )
    assert prefill == {
        "gdn_input_projections": pytest.approx(0.11),
        "gdn_core_prefill": pytest.approx(0.22),
        "gdn_output_projection": pytest.approx(0.33),
    }
    assert decode == {
        "gdn_input_projections": pytest.approx(0.04),
        "gdn_core_decode": pytest.approx(0.05),
        "gdn_output_projection": pytest.approx(0.06),
    }


def test_gdn_predictor_returns_structured_attention_operator_times(tmp_path: Path) -> None:
    output = _train(tmp_path)
    predictor = GDNPredictor.from_directory(output)
    batch = SimpleNamespace(
        num_tokens=[16],
        total_num_tokens=16,
        num_prefill_tokens=16,
        num_decode_tokens=0,
        requests=[SimpleNamespace(num_processed_tokens=0, is_prefill_complete=False)],
    )
    result = predictor.predict_attention_time(batch, norm_time_ms=0.01)
    assert result.operator_times is not None
    assert result.operator_times.op_times["gdn_core_prefill"] == pytest.approx(0.22)
    assert result.total_time() == pytest.approx(0.67)


def test_gdn_identity_separates_measurement_families_and_tp(tmp_path: Path) -> None:
    output = _train(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        GDNPredictor.from_directory(output, measurement_type="CUDA_EVENT")
    with pytest.raises(ValueError, match="identity mismatch"):
        GDNPredictor.from_directory(output, tensor_parallel_size=2)


def test_gdn_extrapolation_uses_estimator_without_clipping(tmp_path: Path, caplog) -> None:
    output = _train(tmp_path)
    predictor = GDNPredictor.from_directory(output)
    with caplog.at_level("WARNING"):
        prediction = predictor.predict_operator_times(
            GDNBatchFeatures(2, 8, 8, 0.0, 0, "prefill")
        )
    assert set(prediction) == {
        "gdn_input_projections",
        "gdn_core_prefill",
        "gdn_output_projection",
    }
    assert "outside profiled feature bounds" in caplog.text


def test_gdn_input_path_is_public_but_kernel_only_path_is_absent() -> None:
    config = BaseExecutionTimePredictorConfig()
    assert config.gdn_input_file.endswith("/{DEVICE}/{MODEL}/gdn.csv")
    assert not hasattr(config, "gdn_kernel_only_input_file")


def test_model_manager_loads_gdn_artifacts_without_fitting(tmp_path: Path) -> None:
    output = _train(tmp_path)

    replica = SimpleNamespace(
        model_config=_FixtureModelConfig(),
        device="cpu",
        attn_tensor_parallel_size=1,
    )
    predictor_config = SimpleNamespace(
        gdn_input_file=str(FIXTURE),
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._cache_dir = str(output)
    manager._gdn_predictors = {}
    manager._load_gdn_predictor_for_cluster(
        "fixture",
        replica,
        predictor_config,
        MeasurementType.DEVICE_EVENT,
    )
    assert manager.get_gdn_predictor("fixture").runtime_stack_signature == "synthetic_cpu_v1"


def test_model_manager_rejects_artifacts_for_changed_gdn_csv(tmp_path: Path) -> None:
    output = _train(tmp_path)
    source = tmp_path / "gdn.csv"
    shutil.copyfile(FIXTURE, source)
    source.write_text(source.read_text() + "\n", encoding="utf-8")

    replica = SimpleNamespace(
        model_config=_FixtureModelConfig(),
        device="cpu",
        attn_tensor_parallel_size=1,
    )
    predictor_config = SimpleNamespace(gdn_input_file=str(source))
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._cache_dir = str(output)
    manager._gdn_predictors = {}
    with pytest.raises(ValueError, match="identity mismatch"):
        manager._load_gdn_predictor_for_cluster(
            "fixture", replica, predictor_config, MeasurementType.DEVICE_EVENT
        )


@pytest.mark.parametrize("has_gdn_shape", [False, True])
def test_supplied_model_identity_preserves_optional_gdn_shape(
    tmp_path: Path, monkeypatch, has_gdn_shape: bool,
) -> None:
    output = _train(tmp_path)
    model = _FixtureModelConfig()
    if not has_gdn_shape:
        monkeypatch.setattr(model, "get_gdn_config", lambda: None)
    predictor = GDNPredictor.from_directory(output, model_config=model)
    assert predictor.identity["hidden_size"] == model.embedding_dim
    model.embedding_dim += 1
    with pytest.raises(ValueError, match="hidden_size"):
        GDNPredictor.from_directory(output, model_config=model)

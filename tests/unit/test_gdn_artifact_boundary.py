"""W09 regressions at the real GDN fit, persistence, and load boundaries."""

import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from frontier.attention.gdn.features import GDNBatchFeatures
from frontier.execution_time_predictor.gdn_predictor import GDNPredictor
from frontier.training.gdn_trainer import GDNTrainer


FIXTURE = Path(__file__).parents[1] / "fixtures" / "pr31_hybrid" / "gdn.csv"


def _trainer(source: Path, output: Path, **selectors) -> GDNTrainer:
    return GDNTrainer(
        str(source),
        str(output),
        measurement_type="DEVICE_EVENT",
        num_estimators=[2],
        max_depth=[2],
        min_samples_split=[2],
        k_fold_cv_splits=2,
        num_training_job_threads=1,
        **selectors,
    )


@pytest.fixture(scope="module")
def fitted_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("gdn_artifact_boundary")
    original = pd.read_csv(FIXTURE, keep_default_na=False)
    rows = []
    for _, template in original.iterrows():
        for multiplier in (1, 2, 3, 4):
            row = template.copy()
            if row["batch_num_prefill_tokens"]:
                for column in ("batch_num_tokens", "batch_num_prefill_tokens", "max_query_len"):
                    row[column] *= multiplier
            else:
                for column in ("batch_size", "batch_num_tokens", "batch_num_decode_tokens"):
                    row[column] *= multiplier
            rows.append(row)
    source = root / "synthetic_cpu_gdn.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output = root / "models"
    _trainer(source, output).train()
    return source, output


@pytest.fixture
def campaign_copy(fitted_campaign, tmp_path):
    source, output = fitted_campaign
    copied = tmp_path / "models"
    shutil.copytree(output, copied)
    return source, copied


def _manifest(output: Path):
    return json.loads((output / "gdn_manifest.json").read_text())


def _save_manifest(output: Path, manifest):
    (output / "gdn_manifest.json").write_text(json.dumps(manifest))


def test_multirow_fit_fresh_load_exact_and_estimator(campaign_copy):
    source, output = campaign_copy
    predictor = GDNPredictor.from_directory(output, dataset_path=source)
    for features in (
        GDNBatchFeatures(1, 16, 16, 0.0, 0, "prefill"),
        GDNBatchFeatures(1, 24, 24, 0.0, 0, "prefill"),
    ):
        assert predictor.predict_operator_times(features) == pytest.approx({
            "gdn_input_projections": 0.11,
            "gdn_core_prefill": 0.22,
            "gdn_output_projection": 0.33,
        })
    assert predictor.predict_operator_times(
        GDNBatchFeatures(6, 6, 1, 0.0, 6, "decode")
    ) == pytest.approx({
        "gdn_input_projections": 0.04,
        "gdn_core_decode": 0.05,
        "gdn_output_projection": 0.06,
    })


@pytest.mark.parametrize("corruption", [
    "schema", "missing", "duplicate", "swapped", "task", "phase", "features", "target",
])
def test_manifest_corruption_rejected_before_prediction(campaign_copy, corruption):
    _, output = campaign_copy
    manifest = _manifest(output)
    tasks = manifest["tasks"]
    if corruption == "schema":
        manifest["schema_version"] = 999
    elif corruption == "missing":
        tasks.pop()
    elif corruption == "duplicate":
        tasks.append(dict(tasks[0]))
    elif corruption == "swapped":
        tasks[0]["artifact"], tasks[1]["artifact"] = tasks[1]["artifact"], tasks[0]["artifact"]
    elif corruption == "task":
        tasks[0]["task"] = tasks[1]["task"]
    elif corruption == "phase":
        tasks[0]["phase"] = "decode"
    elif corruption == "features":
        tasks[0]["feature_names"].reverse()
    else:
        tasks[0]["target_column"] = tasks[1]["target_column"]
    _save_manifest(output, manifest)
    with pytest.raises(ValueError):
        GDNPredictor.from_directory(output)


@pytest.mark.parametrize("metadata", ["task", "feature_names", "target_col"])
def test_artifact_metadata_must_match_declared_task(campaign_copy, metadata):
    _, output = campaign_copy
    task = _manifest(output)["tasks"][0]
    artifact = output / task["artifact"]
    estimator = pickle.loads(artifact.read_bytes())
    name = f"_frontier_gdn_{metadata}"
    value = getattr(estimator, name)
    setattr(estimator, name, list(reversed(value)) if isinstance(value, list) else "wrong")
    artifact.write_bytes(pickle.dumps(estimator))
    with pytest.raises(ValueError):
        GDNPredictor.from_directory(output)


@pytest.mark.parametrize("column, other", [
    ("model_architecture_profile", "another_profile"),
    ("quant_signature", "another_quantization"),
    ("device", "another_device"),
])
def test_omitted_selector_requires_unique_selected_identity(
    fitted_campaign, tmp_path, column, other
):
    source, _ = fitted_campaign
    frame = pd.read_csv(source, keep_default_na=False)
    frame.loc[0, column] = other
    changed = tmp_path / "mixed.csv"
    frame.to_csv(changed, index=False)
    output = tmp_path / "models"
    with pytest.raises(ValueError):
        _trainer(changed, output).train()
    assert not (output / "gdn_manifest.json").exists()


@pytest.mark.parametrize("column", [
    "model_architecture_profile", "quant_signature", "device", "runtime_stack_signature",
    "gdn_runtime_backend", "model_dtype", "hidden_size",
])
def test_selected_identity_rejects_missing_row_values(fitted_campaign, tmp_path, column):
    source, _ = fitted_campaign
    frame = pd.read_csv(source, keep_default_na=False)
    frame.loc[1, column] = np.nan
    changed = tmp_path / "missing.csv"
    frame.to_csv(changed, index=False)
    with pytest.raises(ValueError):
        _trainer(changed, tmp_path / "models").train()


def test_explicit_identity_selectors_select_one_complete_scope(fitted_campaign, tmp_path):
    source, _ = fitted_campaign
    frame = pd.read_csv(source, keep_default_na=False)
    other = frame.copy()
    other["device"] = "another_device"
    changed = tmp_path / "two_devices.csv"
    pd.concat([frame, other], ignore_index=True).to_csv(changed, index=False)
    output = tmp_path / "models"
    _trainer(changed, output, device="cpu").train()
    assert GDNPredictor.from_directory(output, device="cpu").identity["device"] == "cpu"


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_exact_predictions_require_finite_nonnegative_values(campaign_copy, invalid):
    _, output = campaign_copy
    manifest = _manifest(output)
    artifact = output / manifest["tasks"][0]["artifact"]
    estimator = pickle.loads(artifact.read_bytes())
    features = GDNBatchFeatures(1, 16, 16, 0.0, 0, "prefill")
    estimator._frontier_gdn_exact_lookup[features.exact_key()] = invalid
    artifact.write_bytes(pickle.dumps(estimator))
    with pytest.raises(ValueError):
        predictor = GDNPredictor.from_directory(output)
        predictor.predict_operator_times(features)


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_estimator_predictions_require_finite_nonnegative_values(campaign_copy, invalid):
    _, output = campaign_copy
    artifact = output / _manifest(output)["tasks"][0]["artifact"]
    estimator = pickle.loads(artifact.read_bytes())
    for tree in estimator.estimators_:
        tree.tree_.value[:] = invalid
    artifact.write_bytes(pickle.dumps(estimator))
    with pytest.raises(ValueError):
        predictor = GDNPredictor.from_directory(output)
        predictor.predict_operator_times(GDNBatchFeatures(1, 24, 24, 0.0, 0, "prefill"))


def test_multirow_training_reuses_existing_estimators(campaign_copy, monkeypatch):
    from sklearn.ensemble import RandomForestRegressor

    source, output = campaign_copy

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Existing estimator cache must avoid fitting")

    monkeypatch.setattr(RandomForestRegressor, "fit", unexpected_fit)
    assert len(_trainer(source, output).train()) == 6
    assert GDNPredictor.from_directory(output).predict_operator_times(
        GDNBatchFeatures(1, 16, 16, 0.0, 0, "prefill")
    )["gdn_core_prefill"] == pytest.approx(0.22)


def test_interrupted_artifact_publication_preserves_existing_artifact(
    campaign_copy, monkeypatch
):
    source, output = campaign_copy
    artifact = output / _manifest(output)["tasks"][0]["artifact"]
    before = artifact.read_bytes()
    original_dump = pickle.dump

    def interrupted_dump(value, stream, *args, **kwargs):
        if hasattr(value, "_frontier_gdn_task"):
            stream.write(b"interrupted serialization")
            raise OSError("injected serialization interruption")
        return original_dump(value, stream, *args, **kwargs)

    monkeypatch.setattr(pickle, "dump", interrupted_dump)
    with pytest.raises(OSError, match="injected serialization interruption"):
        _trainer(source, output).train()
    assert artifact.read_bytes() == before
    predictor = GDNPredictor.from_directory(output)
    assert predictor.predict_operator_times(
        GDNBatchFeatures(1, 16, 16, 0.0, 0, "prefill")
    )["gdn_core_prefill"] == pytest.approx(0.22)


def test_one_row_single_configuration_fits_and_reuses_cache(tmp_path, monkeypatch):
    from sklearn.ensemble import RandomForestRegressor

    output = tmp_path / "models"
    assert len(_trainer(FIXTURE, output).train()) == 6

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("One-row training must reuse a valid estimator cache")

    monkeypatch.setattr(RandomForestRegressor, "fit", unexpected_fit)
    assert len(_trainer(FIXTURE, output).train()) == 6
    assert GDNPredictor.from_directory(output).predict_operator_times(
        GDNBatchFeatures(1, 16, 16, 0.0, 0, "prefill")
    )["gdn_core_prefill"] == pytest.approx(0.22)


def test_one_row_rejects_multiple_parameter_configurations(tmp_path):
    output = tmp_path / "models"
    trainer = GDNTrainer(
        str(FIXTURE), str(output), num_estimators=[2, 3], max_depth=[2],
        min_samples_split=[2], num_training_job_threads=1,
    )
    with pytest.raises(ValueError, match="exactly one parameter configuration"):
        trainer.train()
    assert not (output / "gdn_manifest.json").exists()


def test_runtime_stack_selector_selects_one_complete_scope(fitted_campaign, tmp_path):
    source, _ = fitted_campaign
    frame = pd.read_csv(source, keep_default_na=False)
    other = frame.copy()
    other["runtime_stack_signature"] = "synthetic_cpu_v2"
    changed = tmp_path / "two_runtime_stacks.csv"
    pd.concat([frame, other], ignore_index=True).to_csv(changed, index=False)
    output = tmp_path / "models"
    _trainer(changed, output, runtime_stack_signature="synthetic_cpu_v1").train()
    assert GDNPredictor.from_directory(
        output, runtime_stack_signature="synthetic_cpu_v1"
    ).runtime_stack_signature == "synthetic_cpu_v1"
    with pytest.raises(ValueError, match="identity mismatch"):
        GDNPredictor.from_directory(output, runtime_stack_signature="synthetic_cpu_v2")


def test_manifest_requires_identity_even_without_selectors(campaign_copy):
    _, output = campaign_copy
    manifest = _manifest(output)
    del manifest["identity"]["device"]
    _save_manifest(output, manifest)
    with pytest.raises(ValueError, match="identity"):
        GDNPredictor.from_directory(output)


def test_interrupted_manifest_publication_preserves_complete_json(campaign_copy, monkeypatch):
    source, output = campaign_copy
    manifest = output / "gdn_manifest.json"
    before = manifest.read_bytes()

    def interrupted_json(value, stream, *args, **kwargs):
        stream.write('{"interrupted":')
        raise OSError("injected manifest interruption")

    monkeypatch.setattr(json, "dump", interrupted_json)
    with pytest.raises(OSError, match="injected manifest interruption"):
        _trainer(source, output).train()
    assert manifest.read_bytes() == before
    assert not list(output.glob(".*.tmp"))
    assert GDNPredictor.from_directory(output).predict_operator_times(
        GDNBatchFeatures(1, 16, 16, 0.0, 0, "prefill")
    )["gdn_core_prefill"] == pytest.approx(0.22)


def test_partial_new_campaign_does_not_publish_manifest(fitted_campaign, tmp_path, monkeypatch):
    source, _ = fitted_campaign
    output = tmp_path / "models"
    original_dump = pickle.dump
    published_tasks = []

    def interrupted_dump(value, stream, *args, **kwargs):
        if hasattr(value, "_frontier_gdn_task"):
            published_tasks.append(value._frontier_gdn_task)
            if len(published_tasks) == 3:
                stream.write(b"interrupted serialization")
                raise OSError("injected third artifact interruption")
        return original_dump(value, stream, *args, **kwargs)

    monkeypatch.setattr(pickle, "dump", interrupted_dump)
    with pytest.raises(OSError, match="injected third artifact interruption"):
        _trainer(source, output).train()
    assert len(published_tasks) == 3
    assert not (output / "gdn_manifest.json").exists()
    assert not list(output.glob(".*.tmp"))
    with pytest.raises(FileNotFoundError, match="manifest"):
        GDNPredictor.from_directory(output)

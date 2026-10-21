"""CPU fixture provenance and immutable snapshots of the existing model cache."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from frontier.execution_time_predictor import cache_io, prediction_model_registry
from frontier.execution_time_predictor.prediction_model_registry import PredictionModelRegistry


MODEL_NAME = "step5_swa_attention"


@pytest.fixture
def registry(tmp_path):
    registry = PredictionModelRegistry()
    registry._cache_dir = str(tmp_path)
    return registry


def identity(label="A"):
    return {
        "model_name": f"Step-CPU-fixture-{label}",
        "model_architecture_profile": "step5_cpu_fixture",
        "device": "cpu_fixture_device",
        "role": "prefill",
        "topology": {"tp": 1, "dp": 1, "ep": 1, "pp": 1},
        "graph_policy": "eager",
        "runtime_stack_signature": {"runtime": "sklearn_cpu_fixture", "image": "cpu_fixture_image"},
        "physical_scope": "fixture_attention_without_collective",
        "quant_signature": "none",
        "profiling_precision": "BF16",
        "backend": "sklearn_cpu_fixture",
        "evidence_kind": "cpu_fixture",
        "coverage": {"operators": [MODEL_NAME], "num_tokens": [1, 4]},
    }


def fit_and_store(registry, provenance, value=1.25, name=MODEL_NAME):
    frame = pd.DataFrame({"num_tokens": [1.0, 4.0], "time": [value, value + 2.0]})
    model = DecisionTreeRegressor(random_state=0).fit(frame[["num_tokens"]], frame["time"])
    model._frontier_model_hash = hashlib.sha256(
        (json.dumps(provenance, sort_keys=True) + frame.to_json()).encode()
    ).hexdigest()[:8]
    model._frontier_feature_names = ["num_tokens"]
    model._frontier_target_col = "time"
    model._frontier_layer_cache_identity = '{"profile_id":"step5_cpu_fixture"}'
    registry._set_measurement_identity(model, provenance)
    registry._ensure_exact_lookup_metadata(
        model_name=name, model_hash=model._frontier_model_hash, model=model,
        df=frame, feature_cols=["num_tokens"], target_col="time",
    )
    return model, frame


def artifact(registry, model, name=MODEL_NAME):
    return Path(registry._cache_dir) / f"{name}_{model._frontier_model_hash}.pkl"


def snapshot(path):
    return path.read_bytes(), path.stat().st_mtime_ns


def rewrite_manifest(registry, receipt, change):
    """Keep a manifest self-consistent to test its independent artifact checks."""
    manifest = json.loads(Path(receipt["path"]).read_text())
    change(manifest)
    payload = {key: value for key, value in manifest.items() if key != "bundle_id"}
    manifest["bundle_id"] = registry._model_bundle_id(payload)
    path = Path(registry._cache_dir) / f"bundle_{manifest['bundle_id']}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def test_fitted_model_cold_warm_reuse_preserves_artifact_and_manifest(registry, monkeypatch):
    provenance = identity()
    model, frame = fit_and_store(registry, provenance)
    path = artifact(registry, model)
    before = snapshot(path)
    receipt = registry.publish_model_bundle(provenance, {MODEL_NAME: model})
    manifest_path = Path(receipt["path"])
    manifest_before = snapshot(manifest_path)
    assert receipt["bundle_id"] == manifest_path.stem.removeprefix("bundle_")
    assert len(receipt["bundle_id"]) == 64
    assert receipt["identity"]["evidence_kind"] == "cpu_fixture"
    assert "native_admission" not in receipt
    entry = receipt["models"][MODEL_NAME]
    assert entry["artifact"] == path.name
    assert entry["sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert entry["feature_names"] == ["num_tokens"]
    assert entry["target_column"] == "time"
    assert entry["typed_identity"] == model._frontier_layer_cache_identity

    fresh = PredictionModelRegistry()
    fresh._cache_dir = registry._cache_dir
    cold = fresh.load_model_bundle(manifest_path, provenance)[MODEL_NAME]
    warm = fresh.load_model_bundle(manifest_path, provenance)[MODEL_NAME]
    for loaded in (cold, warm):
        assert loaded.predict(frame[["num_tokens"]]).tolist() == model.predict(frame[["num_tokens"]]).tolist()
        assert loaded._frontier_exact_lookup == {(1.0,): 1.25, (4.0,): 3.25}
    monkeypatch.setattr(fresh, "_store_model_in_cache", lambda *_args: pytest.fail("warm reuse wrote an artifact"))
    fresh._ensure_exact_lookup_metadata(
        model_name=MODEL_NAME, model_hash=warm._frontier_model_hash, model=warm,
        df=frame, feature_cols=["num_tokens"], target_col="time",
    )
    assert fresh.publish_model_bundle(provenance, {MODEL_NAME: warm}) == receipt
    assert snapshot(path) == before
    assert snapshot(manifest_path) == manifest_before
    assert list(Path(registry._cache_dir).glob("*.pkl")) == [path]


def test_a_b_a_bundles_share_cache_without_cross_fixture_contamination(registry):
    model_a, frame = fit_and_store(registry, identity("A"), value=1.25)
    receipt_a = registry.publish_model_bundle(identity("A"), {MODEL_NAME: model_a})
    before_a = snapshot(artifact(registry, model_a))
    model_b, _ = fit_and_store(registry, identity("B"), value=9.5)
    receipt_b = registry.publish_model_bundle(identity("B"), {MODEL_NAME: model_b})
    assert receipt_a["bundle_id"] != receipt_b["bundle_id"]
    assert model_a._frontier_model_hash != model_b._frontier_model_hash
    observed = []
    for label, receipt in (("A", receipt_a), ("B", receipt_b), ("A", receipt_a)):
        models = registry.load_model_bundle(receipt["path"], identity(label))
        observed.append(models[MODEL_NAME].predict(frame[["num_tokens"]]).tolist())
        registry._validate_cached_measurement_identity(
            model_name=MODEL_NAME, model=models[MODEL_NAME], requested_identity=identity(label),
        )
    assert observed == [[1.25, 3.25], [9.5, 11.5], [1.25, 3.25]]
    assert snapshot(artifact(registry, model_a)) == before_a


@pytest.mark.parametrize("field,value", [
    ("runtime_stack_signature", {"runtime": "other_runtime", "image": "cpu_fixture_image"}),
    ("runtime_stack_signature", {"runtime": "sklearn_cpu_fixture", "image": "other_image"}),
    ("device", "other_device"), ("role", "decode"), ("graph_policy", "FULL"),
    ("topology", {"tp": 8}), ("backend", "other_backend"), ("profiling_precision", "FP8"),
])
def test_requested_identity_mismatch_fails_before_cache_load(registry, monkeypatch, field, value):
    model, _ = fit_and_store(registry, identity())
    receipt = registry.publish_model_bundle(identity(), {MODEL_NAME: model})
    requested = {**identity(), field: value}
    monkeypatch.setattr(registry, "_load_model_from_cache", lambda *_a, **_kw: pytest.fail("identity was not checked first"))
    with pytest.raises(ValueError, match="requested identity mismatch"):
        registry.load_model_bundle(receipt["path"], requested)


def test_all_artifact_digests_checked_before_any_model_is_unpickled(registry, monkeypatch):
    first, _ = fit_and_store(registry, identity(), name="a_duration")
    second, _ = fit_and_store(registry, identity(), name="z_count")
    receipt = registry.publish_model_bundle(identity(), {"a_duration": first, "z_count": second})
    artifact(registry, second, "z_count").write_bytes(b"modified artifact, not a pickle")
    monkeypatch.setattr(registry, "_load_model_from_cache", lambda *_a, **_kw: pytest.fail("digest preflight was incomplete"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        registry.load_model_bundle(receipt["path"], identity())


@pytest.mark.parametrize("field,value", [
    ("measurement_identity", {"runtime_stack_signature": "wrong"}),
    ("typed_identity", "wrong"), ("feature_names", ["wrong_feature"]),
    ("target_column", "wrong_target"),
])
def test_manifest_cannot_relabel_artifact_metadata(registry, field, value):
    model, _ = fit_and_store(registry, identity())
    receipt = registry.publish_model_bundle(identity(), {MODEL_NAME: model})
    path = rewrite_manifest(registry, receipt, lambda manifest: manifest["models"][MODEL_NAME].update({field: value}))
    with pytest.raises(ValueError, match="identity mismatch|metadata mismatch"):
        registry.load_model_bundle(path, identity())


def test_manifest_self_identity_is_verified(registry):
    model, _ = fit_and_store(registry, identity())
    receipt = registry.publish_model_bundle(identity(), {MODEL_NAME: model})
    path = Path(receipt["path"])
    manifest = json.loads(path.read_text())
    manifest["models"][MODEL_NAME]["target_column"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="self identity mismatch"):
        registry.load_model_bundle(path, identity())


@pytest.mark.parametrize("outside", ["manifest", "artifact"])
def test_bundle_paths_are_confined_to_existing_cache(registry, tmp_path, outside):
    model, _ = fit_and_store(registry, identity())
    receipt = registry.publish_model_bundle(identity(), {MODEL_NAME: model})
    if outside == "manifest":
        path = tmp_path.parent / f"outside_{tmp_path.name}.json"
        with pytest.raises(ValueError, match="inside the existing cache"):
            registry.load_model_bundle(path, identity())
    else:
        path = rewrite_manifest(registry, receipt, lambda manifest: manifest["models"][MODEL_NAME].update(artifact="../external.pkl"))
        with pytest.raises(ValueError, match="artifact path mismatch"):
            registry.load_model_bundle(path, identity())


def test_partial_publication_and_interrupted_json_publish_leave_no_manifest(registry, monkeypatch):
    model, _ = fit_and_store(registry, identity(), name="present")
    missing = deepcopy(model)
    missing._frontier_model_hash = "notstored"
    with pytest.raises(FileNotFoundError):
        registry.publish_model_bundle(identity(), {"present": model, "missing": missing})
    assert not list(Path(registry._cache_dir).glob("bundle_*.json"))

    def interrupted_dump(_value, stream, **_kwargs):
        stream.write('{"partial":')
        raise OSError("injected manifest interruption")

    monkeypatch.setattr(cache_io.json, "dump", interrupted_dump)
    with pytest.raises(OSError, match="injected manifest interruption"):
        registry.publish_model_bundle(identity(), {"present": model})
    assert not list(Path(registry._cache_dir).glob("bundle_*.json"))
    assert not list(Path(registry._cache_dir).glob(".*.tmp"))


def test_existing_conflicting_manifest_is_never_overwritten(registry):
    model, _ = fit_and_store(registry, identity())
    receipt = registry.publish_model_bundle(identity(), {MODEL_NAME: model})
    path = Path(receipt["path"])
    path.write_bytes(b"conflicting content")
    before = snapshot(path)
    with pytest.raises(ValueError, match="conflicting existing content"):
        registry.publish_model_bundle(identity(), {MODEL_NAME: model})
    assert snapshot(path) == before


def test_marker_required_only_for_explicit_identity_and_bundle_requests(registry):
    model, _ = fit_and_store(registry, identity())
    del model._frontier_measurement_identity
    registry._store_model_in_cache(MODEL_NAME, model._frontier_model_hash, model)
    legacy = registry._load_model_from_cache(MODEL_NAME, model._frontier_model_hash)
    assert legacy.predict(pd.DataFrame({"num_tokens": [1.0]})).tolist() == [1.25]
    with pytest.raises(ValueError, match="missing measurement identity marker"):
        registry._validate_cached_measurement_identity(
            model_name=MODEL_NAME, model=legacy, requested_identity=identity(),
        )
    with pytest.raises(ValueError, match="missing measurement identity marker"):
        registry.publish_model_bundle(identity(), {MODEL_NAME: legacy})


def test_identity_marker_is_canonical_and_collision_mismatch_is_rejected(registry):
    model, _ = fit_and_store(registry, identity())
    marker = registry._set_measurement_identity(model, dict(reversed(list(identity().items()))))
    assert marker == json.dumps(identity(), sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="measurement identity mismatch"):
        registry._validate_cached_measurement_identity(
            model_name=MODEL_NAME, model=model, requested_identity=identity("B"),
        )
    with pytest.raises(ValueError, match="conflicts"):
        registry._set_measurement_identity(model, identity("B"))


def test_bundle_cannot_relabel_a_different_measured_identity(registry):
    model, _ = fit_and_store(registry, identity("A"))
    with pytest.raises(ValueError, match="measurement identity mismatch"):
        registry.publish_model_bundle(identity("B"), {MODEL_NAME: model})
    assert not list(Path(registry._cache_dir).glob("bundle_*.json"))
    receipt = registry.publish_model_bundle(identity("A"), {MODEL_NAME: model})
    path = rewrite_manifest(registry, receipt, lambda manifest: manifest.update(identity=identity("B")))
    with pytest.raises(ValueError, match="measurement identity mismatch"):
        registry.load_model_bundle(path, identity("B"))


def test_bundle_records_producer_identity_subset_without_inventing_other_fields(registry):
    model, _ = fit_and_store(registry, identity())
    producer_identity = {name: identity()[name] for name in ("device", "runtime_stack_signature", "physical_scope")}
    del model._frontier_measurement_identity
    registry._set_measurement_identity(model, producer_identity)
    registry._store_model_in_cache(MODEL_NAME, model._frontier_model_hash, model)
    receipt = registry.publish_model_bundle(identity(), {MODEL_NAME: model})
    assert receipt["models"][MODEL_NAME]["measurement_identity"] == producer_identity
    loaded = registry.load_model_bundle(receipt["path"], identity())[MODEL_NAME]
    assert json.loads(loaded._frontier_measurement_identity) == producer_identity


@pytest.mark.parametrize("invalid", [
    {}, {"runtime": ""}, {"runtime": None}, {"runtime": {}},
    {"runtime": float("nan")}, {"runtime": float("inf")}, {1: "runtime"},
    {"runtime": {"version": float("nan")}}, {"runtime": object()},
])
def test_measurement_identity_rejects_missing_empty_or_nonfinite_json(registry, invalid):
    with pytest.raises(ValueError, match="identity"):
        registry._set_measurement_identity(DecisionTreeRegressor(), invalid)


@pytest.mark.parametrize("field", prediction_model_registry.MODEL_BUNDLE_IDENTITY_FIELDS)
def test_bundle_requires_explicit_complete_identity(registry, field):
    model, _ = fit_and_store(registry, identity())
    incomplete = identity()
    del incomplete[field]
    with pytest.raises(ValueError, match=field):
        registry.publish_model_bundle(incomplete, {MODEL_NAME: model})
    assert not list(Path(registry._cache_dir).glob("bundle_*.json"))


def test_legacy_exact_lookup_migration_writes_once(registry, monkeypatch):
    model, frame = fit_and_store(registry, identity())
    del model._frontier_exact_lookup
    registry._store_model_in_cache(MODEL_NAME, model._frontier_model_hash, model)
    legacy = registry._load_model_from_cache(MODEL_NAME, model._frontier_model_hash)
    calls = []
    store = registry._store_model_in_cache

    def record(*args):
        calls.append(args)
        store(*args)

    monkeypatch.setattr(registry, "_store_model_in_cache", record)
    for _ in range(2):
        registry._ensure_exact_lookup_metadata(
            model_name=MODEL_NAME, model_hash=legacy._frontier_model_hash, model=legacy,
            df=frame, feature_cols=["num_tokens"], target_col="time",
        )
    assert len(calls) == 1
    assert registry._load_model_from_cache(MODEL_NAME, model._frontier_model_hash)._frontier_exact_lookup == {
        (1.0,): 1.25, (4.0,): 3.25,
    }

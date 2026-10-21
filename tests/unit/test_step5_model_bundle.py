"""CPU fixture provenance and immutable snapshots of the existing model cache."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from frontier.execution_time_predictor import cache_io, prediction_model_registry
from frontier.execution_time_predictor.prediction_model_registry import PredictionModelRegistry
from frontier.execution_time_predictor.step5 import (
    STEP5_FEATURE_COLUMNS,
    STEP5_GRAPH_MODES,
    STEP5_PHASES,
    STEP5_VARIANTS,
    set_step5_model_coverage,
)
from frontier.types import ClusterType


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


def test_kernel_gap_input_binding_changes_bundle_not_compute_artifact_and_restores_a(registry, tmp_path):
    provenance = identity()
    model, _ = fit_and_store(registry, provenance)
    before = snapshot(artifact(registry, model))
    gap = tmp_path / "kernel_gap.csv"
    original = "execution_mode,kernel_gap_us\neager,1.0\n"
    gap.write_text(original)

    def bound_identity():
        return {**provenance, "input_bindings": {"kernel_gap_input_file": {
            "path": str(gap), "sha256": cache_io.dataset_fingerprint(gap),
        }}}

    a = registry.publish_model_bundle(bound_identity(), {MODEL_NAME: model})
    manifest_a = snapshot(Path(a["path"]))
    gap.write_text("execution_mode,kernel_gap_us\neager,2.0\n")
    with pytest.raises(ValueError, match="input binding checksum mismatch"):
        registry.load_model_bundle(a["path"], a["identity"])
    b = registry.publish_model_bundle(bound_identity(), {MODEL_NAME: model})
    assert b["bundle_id"] != a["bundle_id"]
    assert registry.load_model_bundle(b["path"], b["identity"])
    gap.write_text(original)
    again = registry.publish_model_bundle(bound_identity(), {MODEL_NAME: model})
    assert again["path"] == a["path"]
    assert registry.load_model_bundle(again["path"], again["identity"])
    assert snapshot(Path(a["path"])) == manifest_a
    assert snapshot(artifact(registry, model)) == before


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


ROOT_NAME = "step5_layer_e2e"
MIXED_OWNERS = (
    (ROOT_NAME, "attention_without_inputnorm_tp", "CUDA_EVENT", "BF16"),
    ("input_layernorm", "linear_input_norm", "KERNEL_ONLY", "BF16"),
    ("mlp_up_proj", "linear_dense_shared", "KERNEL_ONLY", "BF16"),
    ("moe_grouped_gemm", "routed_experts", "KERNEL_ONLY", "FP8"),
)


def fitted_mixed_owners(registry, label="A", offset=0.0):
    """Fit labelled CPU targets for distinct owners, never native observations."""

    shared = identity(label)
    for field in (*prediction_model_registry.MODEL_BUNDLE_MEASUREMENT_FIELDS, "backend", "profiling_precision"):
        del shared[field]
    shared["graph_policy"] = {"prefill": "none", "decode": "full"}
    shared["coverage"] = {"operators": [item[0] for item in MIXED_OWNERS], "fixture_rows_per_owner": 2}
    measurements, models, frames = {}, {}, {}
    for index, (owner, scope, method, precision) in enumerate(MIXED_OWNERS):
        features = pd.DataFrame({"num_tokens": [1.0, 4.0]})
        if owner == ROOT_NAME:
            features = pd.DataFrame({name: [1.0, 1.0] for name in STEP5_FEATURE_COLUMNS})
            features["global_layer_id"] = [0.0, 3.0]
            features["attention_variant_code"] = [STEP5_VARIANTS["swa"], STEP5_VARIANTS["sparse_gqa"]]
            features["phase_code"] = STEP5_PHASES["decode"]
            features["graph_mode_code"] = STEP5_GRAPH_MODES["full"][0]
        producer = {
            **{key: value for key, value in shared.items() if key != "coverage"},
            "physical_scope": scope, "measurement_type": method,
            "runtime_stack_signature": {"runtime": f"{owner}_cpu_fixture_runtime", "image": f"{label}_{owner}_fixture_image"},
            "quant_signature": "fixture_fp8" if precision == "FP8" else "none",
            "profiling_precision": precision, "graph_mode": "full" if method == "CUDA_EVENT" else "none",
            "feature_columns": list(features.columns),
        }
        if owner == ROOT_NAME:
            producer["attention_backend"] = "cpu_fixture_attention"
        else:
            producer["linear_backends"] = {"gemm": "cpu_fixture_gemm", "norm": "cpu_fixture_norm"}
        for kind, targets in (
            ("median", [offset + index + 1.25, offset + index + 3.25]),
            ("kernel_count", [17.0 + index, 23.0 + index]),
        ):
            name = owner if kind == "median" else f"{owner}_kernel_count"
            target = f"time_stats.{owner}.{kind}"
            frame = features.assign(**{target: targets})
            model = DecisionTreeRegressor(random_state=0).fit(features, frame[target])
            model._frontier_model_hash = hashlib.sha256(
                (json.dumps(producer, sort_keys=True) + frame.to_json()).encode()
            ).hexdigest()[:8]
            model._frontier_feature_names = list(features.columns)
            model._frontier_target_col = target
            model._frontier_layer_cache_identity = json.dumps({"owner": owner, "profile_id": "step5_cpu_fixture"})
            registry._set_measurement_identity(model, producer)
            if owner == ROOT_NAME:
                set_step5_model_coverage(model, frame)
            registry._ensure_exact_lookup_metadata(
                model_name=name, model_hash=model._frontier_model_hash, model=model,
                df=frame, feature_cols=list(features.columns), target_col=target,
            )
            measurements[name], models[name], frames[name] = deepcopy(producer), model, frame
    # Like the real trainer, this convenience reference is attached after both
    # standalone artifacts exist. The bundle loads its actual count artifact.
    models[ROOT_NAME]._frontier_kernel_count_model = models[f"{ROOT_NAME}_kernel_count"]
    return {**shared, "measurements": measurements}, models, frames


def test_mixed_owners_duration_count_bundle_cold_warm_a_b_a(registry):
    identity_a, models_a, frames_a = fitted_mixed_owners(registry)
    receipt_a = registry.publish_model_bundle(identity_a, models_a)
    before = {name: snapshot(artifact(registry, model, name)) for name, model in models_a.items()}
    manifest_before = snapshot(Path(receipt_a["path"]))
    identity_b, models_b, frames_b = fitted_mixed_owners(registry, "B", offset=7.0)
    receipt_b = registry.publish_model_bundle(identity_b, models_b)
    assert receipt_a["bundle_id"] != receipt_b["bundle_id"]
    assert set(receipt_a["models"]) == set(identity_a["measurements"])

    fresh = PredictionModelRegistry()
    fresh._cache_dir = registry._cache_dir
    for requested, receipt, original, frames in (
        (identity_a, receipt_a, models_a, frames_a),
        (identity_a, receipt_a, models_a, frames_a),
        (identity_b, receipt_b, models_b, frames_b),
        (identity_a, receipt_a, models_a, frames_a),
    ):
        loaded = fresh.load_model_bundle(receipt["path"], requested)
        for name, model in loaded.items():
            frame = frames[name]
            features = frame[model._frontier_feature_names]
            assert model.predict(features).tolist() == frame[model._frontier_target_col].tolist()
            assert model._frontier_exact_lookup == original[name]._frontier_exact_lookup
            assert json.loads(model._frontier_measurement_identity) == requested["measurements"][name]
            assert receipt["models"][name]["target_column"] == model._frontier_target_col
            assert receipt["models"][name]["feature_names"] == model._frontier_feature_names
        for name in (ROOT_NAME, f"{ROOT_NAME}_kernel_count"):
            assert loaded[name]._frontier_step5_coverage == original[name]._frontier_step5_coverage
            assert loaded[name]._frontier_feature_bounds == original[name]._frontier_feature_bounds
            assert loaded[name]._frontier_step5_domains == original[name]._frontier_step5_domains
        assert loaded[ROOT_NAME]._frontier_kernel_count_model is loaded[f"{ROOT_NAME}_kernel_count"]
        assert loaded[f"{ROOT_NAME}_kernel_count"]._frontier_operator_name == ROOT_NAME
        assert fresh.publish_model_bundle(requested, loaded) == receipt
    assert all(snapshot(artifact(registry, model, name)) == before[name] for name, model in models_a.items())
    assert snapshot(Path(receipt_a["path"])) == manifest_before
    assert identity_a["measurements"][ROOT_NAME]["measurement_type"] == "CUDA_EVENT"
    assert identity_a["measurements"]["moe_grouped_gemm"]["measurement_type"] == "KERNEL_ONLY"
    assert identity_a["measurements"]["moe_grouped_gemm"]["profiling_precision"] == "FP8"
    assert receipt_a["identity"]["evidence_kind"] == "cpu_fixture"
    assert "native_admission" not in receipt_a


@pytest.mark.parametrize("field,value", [
    ("physical_scope", "wrong_scope"),
    ("runtime_stack_signature", {"runtime": "wrong_runtime", "image": "A_step5_layer_e2e_fixture_image"}),
    ("runtime_stack_signature", {"runtime": "step5_layer_e2e_cpu_fixture_runtime", "image": "wrong_image"}),
    ("profiling_precision", "FP8"), ("measurement_type", "KERNEL_ONLY"),
    ("attention_backend", "wrong_backend"), ("role", "wrong_role"),
    ("graph_mode", "none"), ("quant_signature", "wrong_quant"),
    ("topology", {"tp": 8}), ("device", "wrong_device"),
])
def test_mapped_identity_mismatch_is_rejected_before_any_unpickle(registry, monkeypatch, field, value):
    declared, models, _ = fitted_mixed_owners(registry)
    receipt = registry.publish_model_bundle(declared, models)
    wrong = deepcopy(declared)
    wrong["measurements"][ROOT_NAME][field] = value
    monkeypatch.setattr(registry, "_load_model_from_cache", lambda *_a, **_kw: pytest.fail("mismatch reached unpickle"))
    with pytest.raises(ValueError, match="requested identity mismatch"):
        registry.load_model_bundle(receipt["path"], wrong)
    with pytest.raises(ValueError, match="declared measurement identity mismatch"):
        registry.publish_model_bundle(wrong, models)
    relabelled = rewrite_manifest(registry, receipt, lambda manifest: manifest.update(identity=wrong))
    with pytest.raises(ValueError, match="declared measurement identity mismatch"):
        registry.load_model_bundle(relabelled, wrong)


@pytest.mark.parametrize("missing,extra", [(True, False), (False, True), (True, True)])
def test_per_model_declarations_cover_exact_artifact_names_before_publication(registry, monkeypatch, missing, extra):
    declared, models, _ = fitted_mixed_owners(registry)
    if missing:
        del declared["measurements"][ROOT_NAME]
    if extra:
        declared["measurements"]["unused_artifact"] = deepcopy(declared["measurements"]["input_layernorm"])
    monkeypatch.setattr(registry, "_load_model_from_cache", lambda *_a, **_kw: pytest.fail("partial mapping reached unpickle"))
    with pytest.raises(ValueError, match="must cover exactly"):
        registry.publish_model_bundle(declared, models)
    assert not list(Path(registry._cache_dir).glob("bundle_*.json"))


@pytest.mark.parametrize("missing", [True, False])
def test_manifest_cannot_add_or_remove_measurement_declarations(registry, monkeypatch, missing):
    declared, models, _ = fitted_mixed_owners(registry)
    receipt = registry.publish_model_bundle(declared, models)
    wrong = deepcopy(declared)
    if missing:
        del wrong["measurements"][ROOT_NAME]
    else:
        wrong["measurements"]["unused_artifact"] = deepcopy(wrong["measurements"][ROOT_NAME])
    path = rewrite_manifest(registry, receipt, lambda manifest: manifest.update(identity=wrong))
    monkeypatch.setattr(registry, "_load_model_from_cache", lambda *_a, **_kw: pytest.fail("partial mapping reached unpickle"))
    with pytest.raises(ValueError, match="must cover exactly"):
        registry.load_model_bundle(path, wrong)


@pytest.mark.parametrize("malformed", ["not_a_mapping", [], {ROOT_NAME: "not_an_identity"}])
def test_measurements_requires_a_typed_mapping_of_identity_mappings(registry, malformed):
    declared, models, _ = fitted_mixed_owners(registry)
    declared["measurements"] = malformed
    with pytest.raises(ValueError, match="identity|measurement"):
        registry.publish_model_bundle(declared, models)
    assert not list(Path(registry._cache_dir).glob("bundle_*.json"))


@pytest.mark.parametrize("field", prediction_model_registry.MODEL_BUNDLE_MEASUREMENT_FIELDS)
def test_per_model_declaration_cannot_omit_explicit_producer_fields(registry, field):
    declared, models, _ = fitted_mixed_owners(registry)
    del declared["measurements"][ROOT_NAME][field]
    with pytest.raises(ValueError, match=f"requires explicit {field}"):
        registry.publish_model_bundle(declared, models)


def test_mapped_identity_does_not_override_shared_role_topology_or_graph(registry, monkeypatch):
    declared, models, _ = fitted_mixed_owners(registry)
    monkeypatch.setattr(registry, "_load_model_from_cache", lambda *_a, **_kw: pytest.fail("shared mismatch reached unpickle"))
    for field, value in (("role", "decode"), ("topology", {"tp": 8}), ("graph_policy", "FULL")):
        requested = {**declared, field: value}
        with pytest.raises(ValueError, match=f"measurement identity mismatch.*{field}"):
            registry.publish_model_bundle(requested, models)


def test_per_model_declarations_match_complete_markers_not_a_selected_subset(registry):
    declared, models, _ = fitted_mixed_owners(registry)
    for action in ("missing", "extra"):
        wrong = deepcopy(declared)
        if action == "missing":
            del wrong["measurements"][ROOT_NAME]["feature_columns"]
        else:
            wrong["measurements"][ROOT_NAME]["invented_field"] = "not_in_producer"
        with pytest.raises(ValueError, match="declared measurement identity mismatch"):
            registry.publish_model_bundle(wrong, models)


def test_explicit_alias_reuses_original_legacy_artifact_without_a_rewrite(registry):
    provenance = identity()
    model, frame = fit_and_store(registry, provenance)
    path = artifact(registry, model)
    before = snapshot(path)
    legacy = registry.publish_model_bundle(provenance, {MODEL_NAME: model})
    assert "cache_model_name" not in legacy["models"][MODEL_NAME]
    # This is the supported old-cache hit: the original pickle predates the
    # explicit name, which the normal fitting/cache-hit seam now attaches.
    model._frontier_cache_model_name = MODEL_NAME
    label = f"{MODEL_NAME}:BF16:domain_A"
    declared = {**provenance, "measurements": {label: provenance}}
    receipt = registry.publish_model_bundle(declared, {label: model})
    entry = receipt["models"][label]
    assert entry["cache_model_name"] == MODEL_NAME
    assert entry["artifact"] == path.name
    loaded = registry.load_model_bundle(receipt["path"], declared)
    assert loaded[label]._frontier_cache_model_name == MODEL_NAME
    assert loaded[label].predict(frame[["num_tokens"]]).tolist() == [1.25, 3.25]
    assert registry.publish_model_bundle(declared, loaded) == receipt
    old = registry.load_model_bundle(legacy["path"], provenance)
    assert list(old) == [MODEL_NAME]
    assert registry.publish_model_bundle(provenance, old) == legacy
    assert snapshot(path) == before
    assert len(list(Path(registry._cache_dir).glob("*.pkl"))) == 1


def test_alias_cannot_guess_an_actual_cache_name_without_explicit_metadata(registry):
    model, _ = fit_and_store(registry, identity())
    with pytest.raises(FileNotFoundError):
        registry.publish_model_bundle(identity(), {"not_the_actual_cache_name": model})
    assert not list(Path(registry._cache_dir).glob("bundle_*.json"))


def test_replicated_typed_query_does_not_inherit_ffn_contract_but_gating_keeps_moe_tp(registry):
    from frontier.config.model_config import BaseModelConfig

    config = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    registry._cluster_configs = {
        ClusterType.MONOLITHIC: SimpleNamespace(replica_config=SimpleNamespace(model_config=config))
    }
    calls = []
    sentinel = object()

    def resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    registry._resolve_typed_layer_contract = resolve
    assert registry._resolve_cluster_model_contract(ClusterType.MONOLITHIC, "step5_ffn_merge") is None
    assert not calls
    assert registry._resolve_cluster_model_contract(ClusterType.MONOLITHIC, "moe_gating_linear") is sentinel
    assert calls[-1][1]["is_moe_model"] is True


def test_registered_snapshot_retains_two_contracts_of_one_canonical_cache_name(registry, monkeypatch):
    for prefix in ("_trained_models_", "_models_by_precision_"):
        setattr(registry, prefix + "kernel_only", {})
        setattr(registry, prefix + "kernel_only_by_contract", {})
    bucket = registry._models_by_precision_kernel_only_by_contract.setdefault("BF16", {})
    for index, value in enumerate((1.25, 9.25)):
        model, _ = fit_and_store(registry, identity(), value=value)
        typed = json.dumps({"profile_id": "step5_cpu_fixture", "domain": index}, sort_keys=True)
        model._frontier_layer_cache_identity = typed
        model._frontier_cache_model_name = MODEL_NAME
        registry._store_model_in_cache(MODEL_NAME, model._frontier_model_hash, model)
        bucket[(MODEL_NAME, typed)] = model
        registry._trained_models_kernel_only_by_contract[(MODEL_NAME, typed)] = model
    before = {path.name: snapshot(path) for path in Path(registry._cache_dir).glob("*.pkl")}
    receipt = registry.publish_registered_model_bundle(identity(), "kernel_only", ClusterType.MONOLITHIC)
    assert len(receipt["models"]) == 2
    records = receipt["identity"]["registrations"]
    assert {entry["typed_identity"] for entry in records.values()} == {key[1] for key in bucket}
    loaded = registry.load_model_bundle(receipt["path"], receipt["identity"])
    assert sorted(model.predict(pd.DataFrame({"num_tokens": [1.0]}))[0] for model in loaded.values()) == [1.25, 9.25]
    assert {path.name: snapshot(path) for path in Path(registry._cache_dir).glob("*.pkl")} == before
    key, model = next(iter(bucket.items()))
    bucket[(key[0], "wrong_registered_contract")] = model
    monkeypatch.setattr(registry, "_load_model_from_cache", lambda *_a, **_kw: pytest.fail("typed registration mismatch reached unpickle"))
    with pytest.raises(ValueError, match="typed identity mismatch"):
        registry.publish_registered_model_bundle(identity(), "kernel_only", ClusterType.MONOLITHIC)


def test_registered_snapshot_preserves_real_step_ffn_precisions_contracts_and_pairs(tmp_path, monkeypatch):
    from frontier.config.model_config import BaseModelConfig
    from tests.unit.test_step5_measured_ffn import _ffn, _manager, _tables

    config = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    manager = _manager(tmp_path / "cache")
    _ffn(manager, config, _tables(tmp_path, config))
    domains = {}
    for precision, bucket in manager._models_by_precision_kernel_only_by_contract.items():
        for (name, typed_identity), estimator in bucket.items():
            # Explicit provenance from actual registry keys, matching the
            # primary's fit/cache-hit hook, without modifying old cache bytes.
            estimator._frontier_cache_model_name = name
            domains[(name, precision, typed_identity)] = estimator
    for precision, bucket in manager._models_by_precision_kernel_only.items():
        for name, estimator in bucket.items():
            estimator._frontier_cache_model_name = name
            domains[(name, precision, getattr(estimator, "_frontier_layer_cache_identity", None))] = estimator
    before = {path.name: snapshot(path) for path in (tmp_path / "cache").glob("*.pkl")}
    common = {
        "model_name": "stepfun-ai/Step-5-Preview", "model_architecture_profile": "step5",
        "device": "h800", "role": "prefill",
        "topology": {"attention_tp": 8, "moe_tp": 1, "ep": 8},
        "graph_policy": "none", "evidence_kind": "cpu_fixture",
        "coverage": {"fixture": "Step_FFN_A", "num_tokens": [16, 32]},
    }
    receipt = manager.publish_registered_model_bundle(common, "kernel_only", ClusterType.MONOLITHIC)
    assert len(receipt["models"]) == len(domains)
    registrations = receipt["identity"]["registrations"]
    routed = {label: item for label, item in registrations.items() if item["cache_model_name"] == "moe_grouped_gemm"}
    assert {item["precision"] for item in routed.values()} == {"BF16", "FP8"}
    assert len(routed) == 2
    receipt_b = manager.publish_registered_model_bundle(
        {**common, "coverage": {"fixture": "Step_FFN_B", "num_tokens": [16, 32]}},
        "kernel_only", ClusterType.MONOLITHIC,
    )
    for current in (receipt, receipt, receipt_b, receipt):
        loaded = manager.load_model_bundle(current["path"], current["identity"])
        for label, registration in current["identity"]["registrations"].items():
            name = registration["cache_model_name"]
            key = (name, registration["precision"], registration.get("typed_identity"))
            estimator, original = loaded[label], domains[key]
            assert estimator._frontier_exact_lookup == original._frontier_exact_lookup
            assert estimator._frontier_model_hash == original._frontier_model_hash
            assert estimator._frontier_cache_model_name == name
            assert current["models"][label]["artifact"] == artifact(manager, original, name).name
            if "kernel_count_model" in current["models"][label]:
                count = loaded[current["models"][label]["kernel_count_model"]]
                assert estimator._frontier_kernel_count_model is count
                assert count._frontier_operator_name == name
        assert manager.publish_model_bundle(current["identity"], loaded) == current
    assert {path.name: snapshot(path) for path in (tmp_path / "cache").glob("*.pkl")} == before

    label = next(iter(routed))
    with monkeypatch.context() as guarded:
        guarded.setattr(manager, "_load_model_from_cache", lambda *_a, **_kw: pytest.fail("domain mismatch reached unpickle"))
        for field, value, message in (
            ("typed_identity", "wrong_contract", "registered typed identity mismatch"),
            ("precision", "FP32", "registered precision mismatch"),
        ):
            wrong = deepcopy(receipt["identity"])
            wrong["registrations"][label][field] = value
            path = rewrite_manifest(manager, receipt, lambda manifest: manifest.update(identity=wrong))
            with pytest.raises(ValueError, match=message):
                manager.load_model_bundle(path, wrong)
        wrong_count = next(
            current["kernel_count_model"] for name, current in receipt["models"].items()
            if name in routed and name != label
        )
        path = rewrite_manifest(manager, receipt, lambda manifest: manifest["models"][label].update(kernel_count_model=wrong_count))
        with pytest.raises(ValueError, match="kernel-count .* mismatch"):
            manager.load_model_bundle(path, receipt["identity"])

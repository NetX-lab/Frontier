"""Declared Step roles share compatible measured coverage, not first-role data."""

import json
import gc
from pathlib import Path
import weakref

import pandas as pd
import pytest

from frontier.config import (
    ClusterConfig, MetricsConfig, RandomForrestExecutionTimePredictorConfig, ReplicaConfig,
    VllmV1SchedulerConfig, global_vars,
)
from frontier.config.quantization_manager import QuantizationManager
from frontier.entities import Batch, Request
from frontier.execution_time_predictor.step5 import (
    STEP5_FEATURE_COLUMNS, load_step5_attention_input_union, step5_row_features,
)
from frontier.execution_time_predictor.step5_cost import predict_step5_attention_time
from frontier.execution_time_predictor import step5, cache_io
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.types import ClusterType, MeasurementType
from test_step5_measured_replay import _predictor
from test_step5_measured_training import _manager, _row, _write


@pytest.fixture
def declared(tmp_path, monkeypatch):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "none")
    monkeypatch.setattr(global_vars, "USE_CUDA_GRAPH", False)
    monkeypatch.setattr(global_vars, "QUANTIZATION_MANAGER", None)
    monkeypatch.setattr(QuantizationManager, "_instance", None)
    monkeypatch.setattr(QuantizationManager, "_initialized", False)
    configs, rows = {}, {}
    for role, phase, query, context, count in (
        (ClusterType.PREFILL, "prefill", 16, 8176, 23),
        (ClusterType.DECODE, "decode", 1, 8184, 31),
    ):
        replica = ReplicaConfig(
            model_name="stepfun-ai/Step-5-Preview", device="h800", network_device="h800_dgx",
            attn_tensor_parallel_size=8, attn_dp=1,
            moe_tensor_parallel_size=1, moe_expert_parallel_size=8,
            attention_backend="STEP5_DSA", kv_cache_dtype="auto",
        )
        scheduler = VllmV1SchedulerConfig(
            block_size=16, max_model_len=8192, max_tokens_in_batch=32,
        )
        directory = tmp_path / role.name
        directory.mkdir()
        rows[role] = [_row(replica.model_config, layer, query=query, context=context, phase=phase)
                      for layer in (0, 3)]
        for row in rows[role]:
            row["time_stats"]["step5_layer_e2e"]["kernel_count"] = count
        path = _write(directory, rows[role])
        config = RandomForrestExecutionTimePredictorConfig(
            atten_kernel_only_input_file=str(path),
            num_estimators=[2], max_depth=[3], min_samples_split=[2],
            k_fold_cv_splits=2, num_training_job_threads=1,
        )
        configs[role] = ClusterConfig(
            cluster_type=role, replica_config=replica, replica_scheduler_config=scheduler,
            execution_time_predictor_config=config,
        )
    return configs, rows


def _fit(manager, configs, role, signatures):
    cluster = configs[role]
    config = cluster.execution_time_predictor_config
    path = (config.atten_kernel_only_input_file if manager._active_measurement_type is MeasurementType.KERNEL_ONLY
            else config.atten_input_file)
    return manager._train_attn_models_for_cluster(
        role, cluster.replica_config, config, cluster.replica_scheduler_config,
        "deliberately_absent_linear_op.csv", path, signatures,
    )


def _runtime_batch(row):
    query = row["batch_num_tokens"]
    context = json.loads(row["context_lens"])[0]
    request = Request(arrived_at=0., num_prefill_tokens=8192 if row["phase"] == "prefill" else 8176,
                      num_decode_tokens=16, num_processed_tokens=context)
    return Batch(replica_id=0, requests=[request], num_tokens=[query], is_moe=True,
                 request_is_decoding=[row["phase"] == "decode"], num_context_tokens=[context])


def test_two_role_inputs_fit_once_and_preserve_both_runtime_queries(declared, tmp_path):
    configs, rows = declared
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    signatures = set()
    models = _fit(manager, configs, ClusterType.PREFILL, signatures)
    assert _fit(manager, configs, ClusterType.DECODE, signatures) == {}
    root = models["step5_layer_e2e"]
    expected = {tuple(step5_row_features(row)[name] for name in STEP5_FEATURE_COLUMNS)
                for source in rows.values() for row in source}
    assert root._frontier_step5_coverage == expected
    assert manager._trained_models_kernel_only["step5_layer_e2e"] is root
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 2
    predictor = _predictor(configs[ClusterType.PREFILL].replica_config.model_config,
                           models, MeasurementType.KERNEL_ONLY)
    for role, source in rows.items():
        for row in source:
            cost = predict_step5_attention_time(predictor, _runtime_batch(row), row["global_layer_id"])
            assert cost.total_time() == pytest.approx(row["time_stats"]["step5_layer_e2e"]["median"])
            assert cost.kernel_count == row["time_stats"]["step5_layer_e2e"]["kernel_count"]
            assert cost.operator_costs["step5_layer_e2e"].exact_lookup


def test_union_identity_tracks_second_input_and_warm_cache(declared, tmp_path, monkeypatch):
    configs, rows = declared
    cache = tmp_path / "cache"
    cold = _manager(cache)
    cold._cluster_configs = configs
    signatures = set()
    original = _fit(cold, configs, ClusterType.PREFILL, signatures)
    warm = _manager(cache)
    warm._cluster_configs = dict(reversed(tuple(configs.items())))
    monkeypatch.setattr(warm, "_create_estimator_and_params", lambda *_: pytest.fail("warm union refitted"))
    loaded = _fit(warm, warm._cluster_configs, ClusterType.DECODE, set())
    for name in original:
        assert loaded[name]._frontier_model_hash == original[name]._frontier_model_hash
    path = configs[ClusterType.DECODE].execution_time_predictor_config.atten_kernel_only_input_file
    frame = pd.read_csv(path)
    for stat in ("min", "max", "mean", "median"):
        frame[f"time_stats.step5_layer_e2e.{stat}"] += .5
    frame.to_csv(path, index=False)
    updated = _fit(cold, configs, ClusterType.PREFILL, signatures)
    assert len(signatures) == 2
    assert updated["step5_layer_e2e"]._frontier_model_hash != original["step5_layer_e2e"]._frontier_model_hash
    assert updated["step5_layer_e2e"]._frontier_measurement_identity != original["step5_layer_e2e"]._frontier_measurement_identity
    assert len(updated["step5_layer_e2e"]._frontier_step5_coverage) == 4
    _write(Path(path).parent, rows[ClusterType.DECODE])
    restored = _manager(cache)
    restored._cluster_configs = configs
    monkeypatch.setattr(restored, "_create_estimator_and_params", lambda *_: pytest.fail("restored input refitted"))
    assert _fit(restored, configs, ClusterType.PREFILL, set())["step5_layer_e2e"]._frontier_model_hash == original["step5_layer_e2e"]._frontier_model_hash


def test_union_shape_policy_is_part_of_signature_and_immutable_identity(declared, tmp_path):
    configs, _ = declared
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    signatures = set()
    original = _fit(manager, configs, ClusterType.PREFILL, signatures)["step5_layer_e2e"]
    configs[ClusterType.DECODE].replica_scheduler_config.max_tokens_in_batch = 64
    changed = _fit(manager, configs, ClusterType.PREFILL, signatures)["step5_layer_e2e"]
    assert len(signatures) == 2
    assert changed._frontier_model_hash != original._frontier_model_hash
    assert changed._frontier_measurement_identity != original._frontier_measurement_identity
    assert changed._frontier_step5_coverage == original._frontier_step5_coverage


def test_union_deduplicates_a_shared_declared_input(declared, tmp_path):
    configs, rows = declared
    path = Path(configs[ClusterType.PREFILL].execution_time_predictor_config.atten_kernel_only_input_file)
    _write(path.parent, rows[ClusterType.PREFILL] + rows[ClusterType.DECODE])
    configs[ClusterType.DECODE].execution_time_predictor_config.atten_kernel_only_input_file = str(path)
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    cluster = configs[ClusterType.PREFILL]
    frame, _ = load_step5_attention_input_union(
        manager, ClusterType.PREFILL, cluster.replica_config, cluster.replica_scheduler_config,
        cluster.execution_time_predictor_config, str(path), MeasurementType.KERNEL_ONLY,
    )
    assert len(frame) == 4
    assert len(frame.attrs["measurement_identity"]["attention_inputs"]) == 2


@pytest.mark.parametrize("shared_source", [False, True])
def test_same_manager_roles_reuse_validated_union_and_hash(declared, tmp_path, monkeypatch, shared_source):
    configs, rows = declared
    if shared_source:
        path = Path(configs[ClusterType.PREFILL].execution_time_predictor_config.atten_kernel_only_input_file)
        _write(path.parent, rows[ClusterType.PREFILL] + rows[ClusterType.DECODE])
        configs[ClusterType.DECODE].execution_time_predictor_config.atten_kernel_only_input_file = str(path)
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    signatures = set()
    loads, fingerprints, hashes = [], [], []
    load, fingerprint, model_hash = step5.load_step5_attention_rows, cache_io.dataset_fingerprint, manager._get_model_hash

    def counted_load(path, *args):
        loads.append(str(path))
        return load(path, *args)

    def counted_fingerprint(path):
        fingerprints.append(str(path))
        return fingerprint(path)

    def counted_hash(*args, **kwargs):
        hashes.append(args[0])
        return model_hash(*args, **kwargs)

    monkeypatch.setattr(step5, "load_step5_attention_rows", counted_load)
    monkeypatch.setattr(cache_io, "dataset_fingerprint", counted_fingerprint)
    monkeypatch.setattr(manager, "_get_model_hash", counted_hash)
    models = _fit(manager, configs, ClusterType.PREFILL, signatures)
    root = models["step5_layer_e2e"]
    identity = json.loads(root._frontier_measurement_identity)
    assert [item["role"] for item in identity["attention_inputs"]] == ["decode", "prefill"]
    assert len(identity["attention_inputs"]) == 2
    assert len(loads) == (1 if shared_source else 2)
    assert len(fingerprints) == (1 if shared_source else 2)
    loads.clear()
    fingerprints.clear()
    hashes.clear()
    manager._cluster_configs = dict(reversed(tuple(configs.items())))
    monkeypatch.setattr(manager, "_create_estimator_and_params", lambda *_: pytest.fail("unchanged roles refitted"))
    assert _fit(manager, configs, ClusterType.DECODE, signatures) == {}
    assert loads == []
    assert hashes == []
    assert len(fingerprints) == (1 if shared_source else 2)
    assert root._frontier_measurement_identity == json.dumps(identity, sort_keys=True, separators=(",", ":"))
    assert len(signatures) == 1


def test_same_manager_source_content_restore_and_malformed_changes(declared, tmp_path, monkeypatch):
    configs, _ = declared
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    original = _fit(manager, configs, ClusterType.PREFILL, set())["step5_layer_e2e"]
    path = Path(configs[ClusterType.DECODE].execution_time_predictor_config.atten_kernel_only_input_file)
    original_bytes = path.read_bytes()
    source = pd.read_csv(path)
    for stat in ("min", "max", "mean", "median"):
        source[f"time_stats.step5_layer_e2e.{stat}"] += .5
    source.to_csv(path, index=False)
    changed = _fit(manager, configs, ClusterType.PREFILL, set())["step5_layer_e2e"]
    assert changed._frontier_model_hash != original._frontier_model_hash
    assert changed._frontier_measurement_identity != original._frontier_measurement_identity
    assert changed._frontier_step5_coverage == original._frontier_step5_coverage
    path.write_bytes(original_bytes)
    monkeypatch.setattr(manager, "_create_estimator_and_params", lambda *_: pytest.fail("restored source refitted"))
    restored = _fit(manager, configs, ClusterType.DECODE, set())["step5_layer_e2e"]
    assert restored._frontier_model_hash == original._frontier_model_hash
    assert restored._frontier_measurement_identity == original._frontier_measurement_identity
    assert restored._frontier_exact_lookup == original._frontier_exact_lookup
    source = pd.read_csv(path)
    source["time_stats.step5_layer_e2e.kernel_count"] = 0
    source.to_csv(path, index=False)
    with pytest.raises(ValueError, match="kernel coverage missing"):
        _fit(manager, configs, ClusterType.PREFILL, set())


@pytest.mark.parametrize("column,value,message", [
    ("runtime_stack_signature", "changed_runtime", "runtime_stack_signature"),
    ("attention_backend", "changed_backend", "attention_backend"),
    ("measurement_type", "CUDA_EVENT", "measurement_type mismatch"),
    ("physical_scope", "changed_scope", "physical_scope"),
])
def test_same_manager_changed_source_identity_is_not_reused(declared, tmp_path, column, value, message):
    configs, _ = declared
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    _fit(manager, configs, ClusterType.PREFILL, set())
    path = configs[ClusterType.DECODE].execution_time_predictor_config.atten_kernel_only_input_file
    frame = pd.read_csv(path)
    frame[column] = value
    frame.to_csv(path, index=False)
    manager._train_single_model = lambda **_: pytest.fail("changed invalid identity reached fitting")
    with pytest.raises(ValueError, match=message):
        _fit(manager, configs, ClusterType.PREFILL, set())


def test_same_manager_fit_config_change_rehashes_current_union(declared, tmp_path, monkeypatch):
    configs, _ = declared
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    signatures = set()
    original = _fit(manager, configs, ClusterType.PREFILL, signatures)["step5_layer_e2e"]
    config = configs[ClusterType.DECODE].execution_time_predictor_config
    original_depth = config.max_depth
    config.max_depth = [4]
    changed = _fit(manager, configs, ClusterType.PREFILL, signatures)["step5_layer_e2e"]
    assert changed._frontier_model_hash != original._frontier_model_hash
    assert changed._frontier_step5_coverage == original._frontier_step5_coverage
    assert changed._frontier_measurement_identity == original._frontier_measurement_identity
    config.max_depth = original_depth
    monkeypatch.setattr(manager, "_create_estimator_and_params", lambda *_: pytest.fail("restored config refitted"))
    assert _fit(manager, configs, ClusterType.PREFILL, set())["step5_layer_e2e"]._frontier_model_hash == original._frontier_model_hash


@pytest.mark.parametrize("change,message", [
    ("attention_tp", "incompatible Step attention.*topology"),
    ("max_model_len", "max_model_len differs"),
    ("norm_dtype", "norm_dtype conflicts"),
    ("rope", "geometry disagrees with rope_theta"),
])
def test_same_manager_revalidates_changed_validation_context(declared, tmp_path, change, message):
    configs, _ = declared
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    _fit(manager, configs, ClusterType.PREFILL, set())
    decode = configs[ClusterType.DECODE]
    if change == "attention_tp":
        decode.replica_config.attn_tensor_parallel_size = 16
    elif change == "max_model_len":
        decode.replica_scheduler_config.max_model_len = 16384
    elif change == "norm_dtype":
        decode.replica_config.model_config.norm_dtype = "bfloat16"
    else:
        model = decode.replica_config.model_config
        model.layer_rope_thetas = tuple(value * 2 for value in model.layer_rope_thetas)
    with pytest.raises(ValueError, match=message):
        _fit(manager, configs, ClusterType.PREFILL, set())


@pytest.mark.parametrize("failure", [None, "training", "publication"])
def test_real_manager_initialization_releases_validated_frames(declared, tmp_path, monkeypatch, failure):
    configs, _ = declared
    cls = ExecutionTimePredictionModelManager
    manager = object.__new__(cls)
    observed = []
    monkeypatch.setattr(cls, "_uses_two_stream_eager_pricing", lambda *_: False)
    monkeypatch.setattr(cls, "_analyze_cluster_requirements", lambda *_: {})

    def training(self):
        self._active_measurement_type = MeasurementType.KERNEL_ONLY
        models = _fit(self, configs, ClusterType.PREFILL, set())
        observed.extend(weakref.ref(entry.frame) for entry in self._step5_attention_inputs.values())
        if failure == "training":
            raise ValueError("training failed")
        return models

    def publication(self):
        assert self._step5_attention_inputs
        if failure == "publication":
            raise ValueError("publication failed")

    monkeypatch.setattr(cls, "_train_all_required_models", training)
    monkeypatch.setattr(cls, "_publish_step5_model_bundles", publication)
    if failure:
        with pytest.raises(ValueError, match=f"{failure} failed"):
            manager.__init__(configs, MetricsConfig(cache_dir=str(tmp_path / "cache")))
    else:
        manager.__init__(configs, MetricsConfig(cache_dir=str(tmp_path / "cache")))
    gc.collect()
    assert observed and all(frame() is None for frame in observed)
    assert "_step5_attention_inputs" not in manager.__dict__


def test_union_merges_only_observed_kinds_from_separate_role_sources(declared, tmp_path, monkeypatch):
    from frontier.profiling.step5.inputs import MRv2DummyDescriptor, Step5ProfileInput
    from test_step5_mrv2_inputs import producer_row

    configs, rows = declared
    decode = configs[ClusterType.DECODE]
    descriptor = MRv2DummyDescriptor(1, 1, "none", None, "CPU_SOURCE_FIXTURE_NOT_P0_CENSUS")
    dummy = Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=512)
    dummy_row = producer_row(decode.replica_config.model_config, dummy, duration=.5, count=17)
    path = Path(decode.execution_time_predictor_config.atten_kernel_only_input_file)
    _write(path.parent, [dummy_row])
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    models = _fit(manager, configs, ClusterType.PREFILL, set())
    root = models["step5_layer_e2e"]
    identity = json.loads(root._frontier_measurement_identity)
    assert identity["physical_scope"] == {
        "attention_workload": "native_attention_layer_without_tp_allreduce",
        "mrv2_dp_dummy": "mrv2_dp_dummy_attention_layer_without_tp_allreduce",
    }
    assert root._frontier_step5_domains["native_workload_code"] == (0., 1.)
    assert len(root._frontier_step5_coverage) == len(rows[ClusterType.PREFILL]) + 1
    warm = _manager(tmp_path / "cache")
    warm._cluster_configs = dict(reversed(tuple(configs.items())))
    monkeypatch.setattr(warm, "_create_estimator_and_params", lambda *_: pytest.fail("warm union refitted"))
    loaded = _fit(warm, configs, ClusterType.DECODE, set())
    for name in models:
        assert loaded[name]._frontier_exact_lookup == models[name]._frontier_exact_lookup
        assert loaded[name]._frontier_measurement_identity == models[name]._frontier_measurement_identity


@pytest.mark.parametrize("column,value", [
    ("runtime_stack_signature", "different_cpu_fixture_runtime"),
    ("attention_backend", "different_cpu_fixture_backend"),
])
def test_union_rejects_incompatible_source_identity_before_fit(declared, tmp_path, column, value):
    configs, _ = declared
    path = configs[ClusterType.DECODE].execution_time_predictor_config.atten_kernel_only_input_file
    frame = pd.read_csv(path)
    frame[column] = value
    frame.to_csv(path, index=False)
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    manager._train_single_model = lambda **_: pytest.fail("incompatible union reached fitting")
    with pytest.raises(ValueError, match=column):
        _fit(manager, configs, ClusterType.PREFILL, set())


def test_union_rejects_incompatible_attention_topology_before_loading(declared, tmp_path):
    configs, _ = declared
    configs[ClusterType.DECODE].replica_config.attn_tensor_parallel_size = 16
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    manager._train_single_model = lambda **_: pytest.fail("incompatible union reached fitting")
    with pytest.raises(ValueError, match="incompatible Step attention.*topology"):
        _fit(manager, configs, ClusterType.PREFILL, set())


@pytest.mark.parametrize("mode,profile_mode", [("full_decode_only", "full"), ("piecewise", "piecewise")])
def test_union_keeps_decode_eager_coverage_and_excludes_ffn_only_roles(
    declared, tmp_path, monkeypatch, mode, profile_mode,
):
    configs, rows = declared
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", mode)
    configs[ClusterType.DECODE_FFN] = configs[ClusterType.DECODE]
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = configs
    signatures = set()
    models = _fit(manager, configs, ClusterType.PREFILL, signatures)
    expected = {tuple(step5_row_features(row)[name] for name in STEP5_FEATURE_COLUMNS)
                for source in rows.values() for row in source}
    assert models["step5_layer_e2e"]._frontier_step5_coverage == expected
    identity = json.loads(models["step5_layer_e2e"]._frontier_measurement_identity)
    assert {item["role"] for item in identity["attention_inputs"]} == {"prefill", "decode"}
    decode = configs[ClusterType.DECODE]
    event_rows = [_row(decode.replica_config.model_config, layer, query=1, context=8184,
                       phase="decode", graph_mode=profile_mode) for layer in (0, 3)]
    event_path = _write(tmp_path, event_rows, MeasurementType.CUDA_EVENT)
    decode.execution_time_predictor_config.atten_input_file = str(event_path)
    from test_step5_mrv2_inputs import producer_row
    from frontier.attention.step5_inputs import MRv2DummyDescriptor
    from frontier.profiling.step5.inputs import Step5ProfileInput

    prefill_event_rows = [producer_row(decode.replica_config.model_config,
        Step5ProfileInput.from_mrv2_dummy(layer,
            MRv2DummyDescriptor(1, 1 if profile_mode == "full" else None, profile_mode,
                                None, "CPU_P_DUMMY_EVENT_FIXTURE", 1),
            block_table_columns=512), duration=.321, count=17) for layer in (0, 3)]
    prefill_dir = tmp_path / "prefill_event"
    prefill_dir.mkdir()
    configs[ClusterType.PREFILL].execution_time_predictor_config.atten_input_file = str(
        _write(prefill_dir, prefill_event_rows, MeasurementType.CUDA_EVENT))
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    event_models = _fit(manager, configs, ClusterType.DECODE, signatures)
    assert event_models["step5_layer_e2e"]._frontier_step5_coverage == {
        tuple(step5_row_features(row)[name] for name in STEP5_FEATURE_COLUMNS)
        for row in event_rows + prefill_event_rows
    }
    assert json.loads(event_models["step5_layer_e2e"]._frontier_measurement_identity)["measurement_type"] == "CUDA_EVENT"
    assert manager._trained_models_kernel_only["step5_layer_e2e"] is models["step5_layer_e2e"]

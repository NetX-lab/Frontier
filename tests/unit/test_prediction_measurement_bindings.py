"""Real Step binding retention; CPU fixtures do not qualify native timings."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.config import ReplicaConfig, global_vars
from frontier.execution_time_predictor.moe_dataset_training import position_moe_operator_facts
from frontier.moe_load_imbalance import moe_native_binding_context
from frontier.profiling.moe.routed_layer_groups import resolve_routed_layer_groups
from frontier.types import ClusterType, MeasurementType
from tests.integration import step5_measured_compute_fixture as fixture
from tests.unit.test_step5_measured_ffn import _config
from tests.unit.test_step5_measured_ffn_replay import _predictor
from tests.unit.test_step5_measured_training import _manager


@pytest.fixture
def bound(tmp_path, monkeypatch):
    monkeypatch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", list(fixture.CAPTURE_SIZES))
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
    monkeypatch.setattr(fixture, "range", lambda *a: (6144,) if a == (6143, 7167) else range(*a), raising=False)
    replica = ReplicaConfig(model_name="stepfun-ai/Step-5-Preview", device="h800",
        attn_tensor_parallel_size=8, attn_dp=2, moe_expert_parallel_size=16,
        moe_routing_distribution_type="position_cyclic")
    paths, _ = fixture.position_compute_tables(tmp_path, replica.model_config,
        device="h800", measurement=MeasurementType.KERNEL_ONLY)
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = {ClusterType.MONOLITHIC: SimpleNamespace(replica_config=replica)}
    frame = pd.read_csv(paths[1])
    facts = position_moe_operator_facts(frame, replica)
    for selected in facts:
        manager._train_ffn_models_for_cluster(ClusterType.MONOLITHIC, replica, _config(),
            str(paths[0]), str(paths[1]), True, set(), operator_facts=selected)
    return manager, replica, frame, paths, facts


def test_actual_dual_binder_retains_fp32_gates_for_layers_3_and_88(bound):
    manager, replica, frame, _, _ = bound
    gates = [model for key, model in manager._contract_precision_registry("kernel_only")["FP32"].items()
             if key[0] == "moe_gating_linear"]
    assert len(gates) == 2, "both fitted native FP32 gate bindings must survive registration"
    assert len({model._frontier_model_hash for model in gates}) == 2
    predictor = _predictor(replica.model_config, manager)
    predictor._replica_config = replica
    for layer in (3, 88):
        group = next(group for group in resolve_routed_layer_groups(replica.model_config) if layer in group.layer_ids)
        context = json.loads(frame.loc[frame.moe_layer_ids.eq(group.row_metadata()["moe_layer_ids"]), "native_context"].iloc[0])
        with predictor._step5_measured_layer(layer):
            selected, precision = predictor._step5_ffn_model("moe_gating_linear", context=context)
        assert precision.name == "FP32"
        measured = manager._measurement_identity(selected)
        assert json.loads(measured["selected_native_metadata"])["moe_layer_ids"] == group.layer_range
        assert json.loads(measured["native_context"]) == moe_native_binding_context(context)


@pytest.mark.parametrize("name,precision,retained", [("moe_gating_linear", "FP32", 5), ("moe_grouped_gemm", "BF16", 4)])
def test_same_precision_native_source_selection_ambiguity_and_json_mapping(bound, name, precision, retained):
    manager, replica, _, _, _ = bound
    original = next(model for key, model in manager._contract_precision_registry("kernel_only")[precision].items()
                    if key[0] == name)
    measured = deepcopy(manager._measurement_identity(original))
    selector = manager._native_measurement_binding(measured)
    variant = deepcopy(original)
    alternate = {**selector, "native_source": {"source_receipt": "OTHER_CPU_FIXTURE_NOT_NATIVE"}}
    variant._frontier_measurement_identity = json.dumps({**measured, **alternate}, sort_keys=True)
    manager._store_model_precision(name, precision, variant)
    identity = original._frontier_layer_cache_identity
    def get(binding):
        return manager._get_family_model("kernel_only", name,
            precision_key=precision, requested_identity=identity, native_binding=binding)
    assert get(selector) is original and get(alternate) is variant
    assert manager.get_model(name, precision, native_binding=selector) is original
    assert get({**alternate, "native_source": json.dumps(alternate["native_source"])}) is variant
    with pytest.raises(ValueError, match="ambiguous native"):
        manager.get_model(name, precision)
    with pytest.raises(ValueError, match="native_source"):
        get({"native_context": selector["native_context"]})
    assert get({**selector, "native_source": {"source_receipt": "MISSING"}}) is None
    assert get({**selector, "selected_native_metadata": {"moe_layer_ids": "missing"}}) is None
    backend = deepcopy(original)
    changed = deepcopy(selector)
    changed["native_context"]["backend"] = "OTHER_CPU_BACKEND_NOT_NATIVE"
    changed["selected_native_metadata"]["native_binding"] = changed["native_context"]
    backend._frontier_measurement_identity = json.dumps({**measured, **changed}, sort_keys=True)
    manager._store_model_precision(name, precision, backend)
    assert get(changed) is backend
    grouped = deepcopy(original)
    other_group = deepcopy(selector)
    other_group["selected_native_metadata"]["moe_layer_ids"] = (
        "88-90" if selector["selected_native_metadata"]["moe_layer_ids"] == "3-87" else "3-87")
    grouped._frontier_measurement_identity = json.dumps({**measured, **other_group}, sort_keys=True)
    manager._store_model_precision(name, precision, grouped)
    assert get(other_group) is grouped
    assert get(selector) is original
    with pytest.raises(ValueError, match="ambiguous native"):
        get({"native_context": selector["native_context"], "native_source": selector["native_source"]})
    with pytest.raises(ValueError, match="Conflicting native model artifacts"):
        manager.get_models()
    assert len([key for key in manager._contract_precision_registry("kernel_only")[precision]
                if key[0] == name]) == retained
    assert manager._get_family_model("eager", name, native_binding=selector) is None


def test_all_native_bindings_survive_family_view_snapshot_and_warm_cache(bound, tmp_path, monkeypatch):
    manager, replica, _, paths, facts = bound
    models = manager.get_models()["kernel_only"]
    native = {f"{name}:{model._frontier_model_hash}": model
              for (name, key), model in manager._contract_model_registry("kernel_only").items()
              if isinstance(key, tuple)}
    assert native and all(models[label] is model for label, model in native.items())
    assert len([label for label in native if label.startswith("moe_gating_linear:")]) == 2
    assert len([label for label in native if label.startswith("moe_grouped_gemm:")]) == 2
    before = {path.name: path.read_bytes() for path in (tmp_path / "cache").glob("*.pkl")}
    common = dict(model_name=replica.model_name, model_architecture_profile="step5", device="h800",
        role="monolithic", topology={"attention_tp": 8, "attention_dp": 2, "ep": 16},
        graph_policy="none", evidence_kind="cpu_fixture", coverage={"layers": [3, 88]})
    receipt = manager.publish_registered_model_bundle(common, "kernel_only", ClusterType.MONOLITHIC)
    assert set(native) <= set(receipt["models"])
    loaded = manager.load_model_bundle(receipt["path"], receipt["identity"])
    for label, model in native.items():
        assert loaded[label]._frontier_model_hash == model._frontier_model_hash
        assert loaded[label]._frontier_exact_lookup == model._frontier_exact_lookup
        assert receipt["models"][label]["typed_identity"] == model._frontier_layer_cache_identity
    assert {path.name: path.read_bytes() for path in (tmp_path / "cache").glob("*.pkl")} == before
    warm = _manager(tmp_path / "cache")
    monkeypatch.setattr(warm, "_create_estimator_and_params", lambda *_: pytest.fail("warm cache refitted"))
    for selected in facts:
        warm._train_ffn_models_for_cluster(ClusterType.MONOLITHIC, replica, _config(),
            str(paths[0]), str(paths[1]), True, set(), operator_facts=selected)
    assert set(warm._models_view_for_family("kernel_only")) == set(models)
    manager._get_measurement_types_for_cluster = lambda *args: (MeasurementType.KERNEL_ONLY,)
    manager._event_measurement_type_for_replica = lambda *args: MeasurementType.CUDA_EVENT
    assert manager.get_models_for_cluster(ClusterType.MONOLITHIC)["kernel_only"] == models
    for label, model in native.items():
        cached = warm._models_view_for_family("kernel_only")[label]
        assert cached._frontier_model_hash == model._frontier_model_hash
        assert cached._frontier_exact_lookup == model._frontier_exact_lookup


def test_position_queries_enumerate_group_physical_rows_before_model_selection(bound, tmp_path):
    from tests.unit.test_moe_position_routing import materialize, source
    manager, replica, _, paths, _ = bound
    predictor = _predictor(replica.model_config, manager)
    predictor._replica_config = replica
    path = fixture.position_collective_costs(tmp_path, paths[1], device="h800")
    predictor._cc_backend = SimpleNamespace(_config=SimpleNamespace(profiling_data_dir=str(path.parents[1])))
    lane = materialize(source(padding=16)).lane(0)
    for layer in (3, 88):
        with predictor._step5_measured_layer(layer):
            context, queries, plans = predictor._step5_position_queries(lane)
            selected, _ = predictor._step5_ffn_model("moe_gating_linear", context=context)
        assert "moe_grouped_gemm" in queries and set(plans) == {
            "step5_moe_dispatch", "step5_moe_combine", "moe_tensor_parallel_allreduce",
            "share_expert_tensor_parallel_allreduce"}
        assert manager._native_measurement_binding(manager._measurement_identity(selected))["native_context"] == moe_native_binding_context(context)
        changed = deepcopy(context)
        changed["workspace_shapes"] = {"rows": [987]}
        changed["activation_layout"]["shape"][0] += 1
        assert moe_native_binding_context(changed) == moe_native_binding_context(context)


def test_unique_legacy_registry_default_and_precision_labels_unchanged(tmp_path):
    manager = _manager(tmp_path / "cache")
    first, second = SimpleNamespace(), SimpleNamespace()
    manager._store_model_precision("legacy", "bf16", first)
    manager._store_model_precision("legacy", "fp32", second)
    assert manager.get_model("legacy") is second
    assert manager.get_model("legacy", "BF16") is first
    assert manager.get_models()["kernel_only"] == {"legacy": second}
    assert manager._contract_model_registry("kernel_only") == {}


@pytest.mark.parametrize("native_first", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_mixed_native_non_native_views_keep_both_insert_orders(bound, tmp_path, native_first, typed):
    trained, replica, _, _, _ = bound
    name = "moe_gating_linear"
    native = next(model for key, model in trained._contract_precision_registry("kernel_only")["FP32"].items()
                  if key[0] == name)
    non_native = SimpleNamespace()
    if typed:
        non_native._frontier_layer_cache_identity = native._frontier_layer_cache_identity
    manager = _manager(tmp_path / "mixed")
    manager._cluster_configs = trained._cluster_configs
    manager._get_measurement_types_for_cluster = lambda *args: (MeasurementType.KERNEL_ONLY,)
    manager._event_measurement_type_for_replica = lambda *args: MeasurementType.CUDA_EVENT
    for model in (native, non_native) if native_first else (non_native, native):
        manager._store_model_precision(name, "FP32", model)
    non_native._frontier_layer_cache_identity = native._frontier_layer_cache_identity
    expected = {name: non_native, f"{name}:{native._frontier_model_hash}": native}
    assert manager.get_models()["kernel_only"] == expected
    assert manager.get_models_for_cluster(ClusterType.MONOLITHIC)["kernel_only"] == expected
    with pytest.raises(ValueError, match="ambiguous native|multiple layer contracts"):
        manager.get_model(name, "FP32")
    binding = manager._native_measurement_binding(manager._measurement_identity(native))
    assert manager.get_model(name, "FP32", native_binding=binding) is native

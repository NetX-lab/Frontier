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
    predictor._cc_backend = SimpleNamespace(collective_profile_bytes=lambda raw=path.read_bytes(): raw)
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


def test_fixed_predictor_groups_reused_by_queries_and_binding_selection(bound, tmp_path, monkeypatch):
    from frontier.profiling.moe import routed_layer_groups
    from tests.unit.test_moe_position_routing import materialize, source
    manager, replica, _, paths, _ = bound
    path = fixture.position_collective_costs(tmp_path, paths[1], device="h800")
    separate = deepcopy(replica)
    assert separate.model_config is not replica.model_config
    resolved = []
    def observe(model):
        groups = resolve_routed_layer_groups(model)
        resolved.append((model, groups))
        return groups
    monkeypatch.setattr(routed_layer_groups, "resolve_routed_layer_groups", observe)
    predictors = [_predictor(config.model_config, manager) for config in (replica, replica, separate)]
    lane = materialize(source(padding=16)).lane(0)
    snapshots = {}
    for index, (predictor, config) in enumerate(zip(predictors, (replica, replica, separate))):
        predictor._replica_config = config
        predictor._cc_backend = SimpleNamespace(collective_profile_bytes=lambda raw=path.read_bytes(): raw)
        for layer in (3, 88, 3, 88):
            with predictor._step5_measured_layer(layer):
                context, queries, plans = predictor._step5_position_queries(lane)
                selected, precision = predictor._step5_ffn_model("moe_gating_linear", context=context)
                binding = manager._native_measurement_binding(manager._measurement_identity(selected))
                assert binding["native_context"] == moe_native_binding_context(context)
                assert binding["selected_native_metadata"]["moe_layer_ids"] == ("3-87" if layer == 3 else "88-90")
                assert precision.name == "FP32"
                value = predictor._predict_step5_ffn_operator("moe_gating_linear", queries["moe_gating_linear"], context=context)
                proof = deepcopy(predictor._step5_ffn_costs["moe_gating_linear"])
                assert value > 0 and proof["kernel_count"] > 0 and proof["exact_lookup"]
                snapshot = (selected, context, queries, {name: plan.cache_key() for name, plan in plans.items()}, proof)
                if layer in snapshots:
                    assert snapshot == snapshots[layer]
                snapshots[layer] = snapshot
                foreign = deepcopy(context)
                foreign["native_facts"]["shared"] = {"source_receipt": "FOREIGN"}
                with pytest.raises(ValueError, match="coverage is unavailable"):
                    predictor._step5_ffn_model("moe_gating_linear", context=foreign)
                predictor._active_measurement_type = MeasurementType.CUDA_EVENT
                with pytest.raises(ValueError, match="coverage is unavailable"):
                    predictor._step5_ffn_model("moe_gating_linear", context=context)
                predictor._active_measurement_type = MeasurementType.KERNEL_ONLY
                with pytest.raises(ValueError, match="source/native point receipt"):
                    predictor._step5_position_queries(materialize(source(start=7000, padding=16)).lane(0))
                with monkeypatch.context() as fault:
                    fault.setattr(selected._frontier_kernel_count_model, "_frontier_exact_lookup", {})
                    with pytest.raises(ValueError, match="duration/count exact feature keys differ"):
                        predictor._predict_step5_ffn_operator("moe_gating_linear", queries["moe_gating_linear"], context=context)
        assert len(resolved) == index + 1, "both actual consumers must reuse one fixed-model resolution"
        assert resolved[-1][0] is config.model_config
    assert resolved[0][1] == resolved[1][1] == resolved[2][1]
    assert resolved[0][1] is not resolved[1][1] and resolved[1][1] is not resolved[2][1]


def test_unique_legacy_registry_default_and_precision_labels_unchanged(tmp_path):
    manager = _manager(tmp_path / "cache")
    first, second = SimpleNamespace(), SimpleNamespace()
    manager._store_model_precision("legacy", "bf16", first)
    manager._store_model_precision("legacy", "fp32", second)
    assert manager.get_model("legacy") is second
    assert manager.get_model("legacy", "BF16") is first
    assert manager.get_models()["kernel_only"] == {"legacy": second}
    assert manager._contract_model_registry("kernel_only") == {}


def test_prepared_points_follow_actual_model_rows_identity_and_fresh_queries(bound, tmp_path, monkeypatch):
    from tests.unit.test_moe_position_routing import materialize, source
    manager, replica, _, paths, _ = bound
    path = fixture.position_collective_costs(tmp_path, paths[1], device=replica.device)
    p = _predictor(replica.model_config, manager); p._replica_config = replica
    p._cc_backend = SimpleNamespace(collective_profile_bytes=lambda raw=path.read_bytes(): raw)
    lane = materialize(source(padding=16)).lane(0)
    real_loads, reads, source_reads = json.loads, [], []
    def observe(text, *args, **kwargs):
        reads.append(text)
        if '"source_frames"' in text: source_reads.append(text)
        return real_loads(text, *args, **kwargs)
    monkeypatch.setattr(json, "loads", observe)
    for layer in (3, 88):
        with p._step5_measured_layer(layer):
            first = p._step5_position_queries(lane)
            model, precision = p._step5_ffn_model("moe_grouped_gemm", context=first[0])
            before, sources_before = len(reads), len(source_reads); again = p._step5_position_queries(lane)
            assert len(reads) - before < 20, "warm selection must not rescan physical rows and plan keys"
            assert len(source_reads) == sources_before
            assert first == again and first[0] is not again[0] and first[1] is not again[1]
            identity = model._frontier_measurement_identity
            rows = model._frontier_moe_physical_rows
            for change in ("rows", "identity", "model"):
                if change == "rows": model._frontier_moe_physical_rows = tuple(list(rows))
                if change == "identity": model._frontier_measurement_identity = identity + " "
                if change == "model":
                    model = deepcopy(model); manager._store_model_precision("moe_grouped_gemm", precision.name, model)
                before = len(source_reads); result = p._step5_position_queries(lane)
                assert len(source_reads) > before and result == first
                model._frontier_moe_physical_rows = ()
                with pytest.raises(ValueError, match="actual source/native point"): p._step5_position_queries(lane)
                model._frontier_moe_physical_rows = rows
            with pytest.raises(ValueError, match="measurement identities differ"):
                p._predict_step5_ffn_operator("moe_grouped_gemm", result[1]["moe_grouped_gemm"], context=result[0])
            model._frontier_measurement_identity = identity
            value = p._predict_step5_ffn_operator("moe_grouped_gemm", result[1]["moe_grouped_gemm"], context=result[0])
            assert value > 0 and p._step5_ffn_costs["moe_grouped_gemm"]["kernel_count"] > 0
            retained = p._step5_physical_points[model]
            model._frontier_moe_physical_rows = ({**rows[0], "routing_input": "{"},)
            with pytest.raises(json.JSONDecodeError): p._step5_position_queries(lane)
            assert p._step5_physical_points[model] is retained
            model._frontier_moe_physical_rows = rows


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
    selected = manager._models_view_for_family("kernel_only", model_name=name)
    assert list(selected) == [f"{name}:{native._frontier_model_hash}", name]
    assert selected == manager.get_models()["kernel_only"]
    assert all(selected[key] is model for key, model in expected.items())


def test_named_native_view_keeps_selected_objects_and_unfiltered_order(bound):
    manager, _, _, _, _ = bound
    full = manager._models_view_for_family("kernel_only")
    expected = {key: model for key, model in full.items()
                if model._frontier_cache_model_name == "moe_grouped_gemm"}
    selected = manager._models_view_for_family("kernel_only", model_name="moe_grouped_gemm")
    assert len(selected) == 2 and list(selected) == list(expected)
    assert all(selected[key] is model for key, model in expected.items())
    again = manager._models_view_for_family("kernel_only", model_name=None)
    assert list(again) == list(full) and all(again[key] is model for key, model in full.items())
    assert manager._models_view_for_family("kernel_only", model_name="absent") == {}
    assert all(model._frontier_cache_model_name != "moe_grouped_gemm_kernel_count" for model in selected.values())
    manager._cluster_configs[ClusterType.DECODE_ATTN] = manager._cluster_configs[ClusterType.MONOLITHIC]
    assert manager._models_view_for_family("kernel_only", ClusterType.DECODE_ATTN,
        model_name="moe_grouped_gemm") == {}


@pytest.mark.parametrize("case", ["ambiguity", "conflict", "missing", "decode_attn"])
def test_named_typed_view_preserves_contract_errors_and_role_exclusion(tmp_path, case):
    from tests.unit.test_profiling_governance_minimal_red import _bare_manager, _dense_contract, _step3_config
    manager = _bare_manager(tmp_path)
    name = "mlp_up_proj"
    manager._store_model_precision(name, "BF16", SimpleNamespace(), layer_contract=_dense_contract(tp_size=4))
    role = None
    if case == "ambiguity":
        manager._store_model_precision(name, "BF16", SimpleNamespace(), layer_contract=_dense_contract())
    elif case == "conflict":
        manager._store_model_precision(name, "BF16", SimpleNamespace())
    else:
        role = ClusterType.DECODE_ATTN if case == "decode_attn" else ClusterType.PREFILL
        replica = SimpleNamespace(model_config=_step3_config(), model_name="step-moe-noquant",
            device="h200", attn_tensor_parallel_size=8, moe_tensor_parallel_size=1, moe_expert_parallel_size=8)
        manager._cluster_configs = {role: SimpleNamespace(replica_config=replica)}
    if case == "decode_attn":
        assert manager._models_view_for_family("eager", role, model_name=name) == {}
        assert manager._models_view_for_family("eager", role) == {}
    else:
        message = "No trained model" if case == "missing" else "multiple layer contracts"
        with pytest.raises(ValueError, match=message) as full:
            manager._models_view_for_family("eager", role)
        with pytest.raises(ValueError, match=message) as selected:
            manager._models_view_for_family("eager", role, model_name=name)
        assert str(selected.value) == str(full.value)

from __future__ import annotations

import pytest

from frontier.moe_routing_runtime import (
    STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH,
    UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH,
    resolve_moe_gating_routing_runtime_path,
)
from tests.unit.test_prediction_measurement_bindings import bound


def test_collective_input_binds_both_role_constructors_and_refreshes_next_run(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from frontier.config import MetricsConfig, ReplicaConfig, global_vars
    from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
    from tests.unit.test_moe_grouped_gemm_padded_features import _ConcreteMoEPredictor
    from tests.unit.test_collective_sim_cc_backend import cpu_writer_payload, CollectiveSimCCBackend, CollectivePlan, CollectiveTensor, ClusterType
    template, artifact, path = cpu_writer_payload(tmp_path, monkeypatch)
    value = json.loads(next(iter(artifact["counts"])))
    plan = CollectivePlan(value["phase"], value["sizes"], tuple(CollectiveTensor(**t) for t in value["tensors"]), value["groups"], value["graph_mode"], value["context_json"], value["buffer_json"])
    replica = ReplicaConfig(model_name="stepfun-ai/Step-5-Preview", device="h800", attn_tensor_parallel_size=8,
        attn_dp=2, moe_expert_parallel_size=16, moe_routing_distribution_type="position_cyclic")
    metrics = MetricsConfig(cache_dir=str(tmp_path / "cache"))
    manager = ExecutionTimePredictionModelManager({}, metrics)
    monkeypatch.setattr(global_vars, "SYS_ARCH", "pd-disaggregation")
    monkeypatch.setattr(SklearnExecutionTimePredictor, "__init__", lambda self, *args: setattr(self, "_cluster_type", args[5]))
    monkeypatch.setattr(_ConcreteMoEPredictor, "_init_global_routing_allocations", lambda self: {})
    read, reads = Path.read_bytes, []
    def observe(file):
        if file == path: reads.append(file)
        return read(file)
    monkeypatch.setattr(Path, "read_bytes", observe)
    backends = []
    for role in (ClusterType.PREFILL, ClusterType.DECODE):
        backend = CollectiveSimCCBackend(template.config, role, "h800", "cpu_fixture", 16)
        _ConcreteMoEPredictor(None, replica, None, metrics, manager, role, cc_backend=backend)
        backends.append(backend)
        for entry in artifact["counts"].values(): entry["cost"].update(duration_ms=.04, network_ms=.02)
        path.write_text(json.dumps(artifact))
    assert [backend.predict_collective_plan(plan).duration_ms for backend in backends] == [.03, .03]
    assert len(reads) == 1
    backend = CollectiveSimCCBackend(template.config, ClusterType.DECODE, "h800", "cpu_fixture", 16)
    _ConcreteMoEPredictor(None, replica, None, metrics, ExecutionTimePredictionModelManager({}, metrics), ClusterType.DECODE, cc_backend=backend)
    assert backend.predict_collective_plan(plan).duration_ms == .04 and len(reads) == 2
    replica.moe_routing_distribution_type = "balanced"
    path.write_text("{")
    backend = CollectiveSimCCBackend(template.config, ClusterType.PREFILL, "h800", "cpu_fixture", 16)
    _ConcreteMoEPredictor(None, replica, None, metrics, manager, ClusterType.PREFILL, cc_backend=backend)
    assert len(reads) == 2


@pytest.mark.parametrize("distribution_type", ["balanced", "skewed", "zipf"])
def test_structured_distributions_use_standard_runtime_path(
    distribution_type: str,
) -> None:
    assert (
        resolve_moe_gating_routing_runtime_path(distribution_type)
        == STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH
    )


def test_random_distribution_uses_uniform_runtime_path() -> None:
    assert (
        resolve_moe_gating_routing_runtime_path("random")
        == UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH
    )


def test_removed_routing_mode_values_fail_fast() -> None:
    with pytest.raises(ValueError, match="moe_routing_distribution_type"):
        resolve_moe_gating_routing_runtime_path("simulation")


def test_position_policy_resolves_before_parent_initialization(monkeypatch):
    from frontier.config import global_vars
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
    from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import SklearnMoEExecutionTimePredictor
    from tests.unit.predictor_cache_fixtures import predictor_fixture_config
    assert resolve_moe_gating_routing_runtime_path("position_cyclic") == "position_cyclic"
    kwargs = predictor_fixture_config()
    kwargs["replica_config"].moe_routing_distribution_type = "position_cyclic"
    monkeypatch.setattr(global_vars, "SYS_ARCH", "co-location")
    monkeypatch.setattr(SklearnExecutionTimePredictor, "__init__", lambda *a, **k: pytest.fail("Unsupported scope reached parent init"))
    with pytest.raises(ValueError, match="only Step P/D"):
        class SourcePredictor(SklearnMoEExecutionTimePredictor):
            def _get_estimator(self):
                pytest.fail("Unsupported scope must not train")
            def _get_grid_search_params(self):
                return {}
        SourcePredictor(**kwargs)


@pytest.mark.parametrize("foreign", [None, {"quant_dtype":"float8_e4m3fn"}])
def test_position_runtime_binds_retained_source_and_full_plan_policy(tmp_path, foreign, bound):
    import json
    from tests.integration.step5_measured_compute_fixture import position_collective_costs
    from tests.unit.test_step5_measured_ffn_replay import _predictor
    from tests.unit.test_moe_position_routing import source,materialize
    from frontier.execution_time_predictor.step5_cost import step5_layer_metadata
    from frontier.profiling.moe.routed_layer_groups import resolve_routed_layer_groups
    from frontier.cc_backend.backends.collective_sim_cc_backend import CollectiveSimCCBackend
    from frontier.cc_backend.cc_backend_config import CollectiveSimCCBackendConfig
    manager, replica, rows, paths, _ = bound
    lane=materialize(source(padding=16)).lane(0)
    path=position_collective_costs(tmp_path,paths[1],device=replica.device)
    config=CollectiveSimCCBackendConfig(cluster_servers=2,cluster_gpus_per_server=8,parallel_tp=8,parallel_dp=2,
        placement_order="TP,DP",profiling_data_dir=str(path.parents[1]),runner_out_dir=str(tmp_path/"runner"))
    def new_predictor():
        predictor=_predictor(replica.model_config,manager);predictor._replica_config=replica
        predictor._cc_backend=CollectiveSimCCBackend(config,predictor._cluster_type,replica.device,"cpu_fixture",16)
        predictor._step5_layer_metadata=step5_layer_metadata(predictor,88)
        return predictor
    p=new_predictor()
    group=next(g for g in resolve_routed_layer_groups(replica.model_config) if 88 in g.layer_ids)
    retained=rows.loc[rows.moe_layer_ids.eq(group.row_metadata()["moe_layer_ids"]) & rows.moe_ep_rank.eq(0) & rows.moe_input_id.eq("cpu_fixture_q16_c6128")]
    context=json.loads(retained.native_context.iloc[0])
    native,features,plans=p._step5_position_queries(lane)
    assert native==context and features["moe_gating_linear"]==dict(num_tokens=16,live_routing_token_count=16)
    counts=json.loads(path.read_text())["counts"]
    assert all(json.loads(plan.context_json)==counts[plan.cache_key()]["context"] for plan in plans.values())
    assert all(json.loads(plan.buffer_json)["compute_context"]==context for plan in plans.values())
    model,_=p._step5_ffn_model("moe_grouped_gemm",context=context)
    assert model._frontier_cache_model_name=="moe_grouped_gemm" and model._frontier_moe_physical_rows
    saved = path.read_text()
    prepared = p._step5_collective_receipts
    path.write_text("{")
    assert p._step5_position_queries(lane) == (native, features, plans)
    assert p._step5_collective_receipts[0] is prepared[0]
    assert all(p._cc_backend.predict_collective_plan(plan).duration_ms == .03 for plan in plans.values())
    with pytest.raises(json.JSONDecodeError): new_predictor()._step5_position_queries(lane)
    path.write_text(saved)
    for case in ("buffer_receipt", "context"):
        changed = json.loads(saved)
        for receipt in changed["counts"].values(): receipt[case] = {"foreign": True} if case == "buffer_receipt" else {**receipt[case], "foreign": True}
        path.write_text(json.dumps(changed))
        candidate=new_predictor()
        with pytest.raises(ValueError, match="buffer/count|context/count"): candidate._step5_position_queries(lane)
        assert not hasattr(candidate,"_step5_collective_receipts")
        assert p._step5_position_queries(lane) == (native, features, plans)
    for key,receipt in list(counts.items()):
        value=json.loads(key);value["buffer_json"]=json.dumps(dict(actual_buffer="foreign",compute_context=foreign));counts[json.dumps(value,sort_keys=True,separators=(",",":"))]=counts.pop(key)
    path.write_text(json.dumps(dict(counts=counts)))
    with pytest.raises(ValueError,match="unique measured"):new_predictor()._step5_position_queries(lane)
    model._frontier_moe_physical_rows=()
    with pytest.raises(ValueError,match="actual source/native point"):p._step5_position_queries(lane)


@pytest.mark.parametrize("name", ["moe_gating_linear", "moe_grouped_gemm", "cache_name"])
def test_named_point_keeps_selected_collision_and_cache_name_guard(bound, tmp_path, name):
    from copy import copy
    import json
    from tests.unit.test_prediction_measurement_bindings import SimpleNamespace, _predictor, fixture
    from tests.unit.test_moe_position_routing import source, materialize
    manager, replica, _, paths, _ = bound
    p = _predictor(replica.model_config, manager)
    p._replica_config = replica
    path = fixture.position_collective_costs(tmp_path, paths[1], device=replica.device)
    p._cc_backend = SimpleNamespace(collective_profile_bytes=lambda raw=path.read_bytes(): raw)
    lane = materialize(source(padding=16)).lane(0)
    with p._step5_measured_layer(3):
        normal = p._step5_position_queries(lane)
        model, precision = p._step5_ffn_model("moe_grouped_gemm" if name == "cache_name" else name, context=normal[0])
        if name == "cache_name":
            model._frontier_cache_model_name = "moe_grouped_gemm_kernel_count"
            with pytest.raises(ValueError, match="actual source/native point"): p._step5_position_queries(lane)
            return
        alternate = copy(model)
        alternate._frontier_measurement_identity = json.dumps({**manager._measurement_identity(model),
            "native_source": {"source_receipt": "OTHER_CPU_FIXTURE_NOT_NATIVE"}}, sort_keys=True)
        manager._store_model_precision(name, precision.name, alternate)
        with pytest.raises(ValueError, match="Conflicting native model artifacts"): manager.get_models()
        if name == "moe_gating_linear":
            assert p._step5_position_queries(lane) == normal
        else:
            with pytest.raises(ValueError, match="Conflicting native model artifacts"): p._step5_position_queries(lane)

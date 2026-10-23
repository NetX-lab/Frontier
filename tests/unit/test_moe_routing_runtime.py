from __future__ import annotations

import pytest

from frontier.moe_routing_runtime import (
    STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH,
    UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH,
    resolve_moe_gating_routing_runtime_path,
)
from tests.unit.test_prediction_measurement_bindings import bound


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
    from types import SimpleNamespace
    from tests.integration.step5_measured_compute_fixture import position_collective_costs
    from tests.unit.test_step5_measured_ffn_replay import _predictor
    from tests.unit.test_moe_position_routing import source,materialize
    from frontier.execution_time_predictor.step5_cost import step5_layer_metadata
    from frontier.profiling.moe.routed_layer_groups import resolve_routed_layer_groups
    manager, replica, rows, paths, _ = bound
    p=_predictor(replica.model_config,manager);p._replica_config=replica
    lane=materialize(source(padding=16)).lane(0)
    path=position_collective_costs(tmp_path,paths[1],device=replica.device)
    p._cc_backend=SimpleNamespace(_config=SimpleNamespace(profiling_data_dir=str(path.parents[1])))
    p._step5_layer_metadata=step5_layer_metadata(p,88)
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
    with pytest.raises(json.JSONDecodeError): p._step5_position_queries(lane)
    assert p._step5_collective_receipts is prepared
    path.write_text(saved)
    assert p._step5_position_queries(lane) == (native, features, plans)
    prepared = p._step5_collective_receipts
    path.write_text(saved + " ")
    assert p._step5_position_queries(lane) == (native, features, plans)
    assert p._step5_collective_receipts[2] is not prepared[2]
    path.write_text(saved)
    p._step5_position_queries(lane); prepared = p._step5_collective_receipts
    for case in ("buffer_receipt", "context"):
        changed = json.loads(saved)
        for receipt in changed["counts"].values(): receipt[case] = {"foreign": True} if case == "buffer_receipt" else {**receipt[case], "foreign": True}
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="buffer/count|context/count"): p._step5_position_queries(lane)
        assert p._step5_collective_receipts is prepared
        path.write_text(saved)
        assert p._step5_position_queries(lane) == (native, features, plans)
        prepared = p._step5_collective_receipts
    for key,receipt in list(counts.items()):
        value=json.loads(key);value["buffer_json"]=json.dumps(dict(actual_buffer="foreign",compute_context=foreign));counts[json.dumps(value,sort_keys=True,separators=(",",":"))]=counts.pop(key)
    path.write_text(json.dumps(dict(counts=counts)))
    with pytest.raises(ValueError,match="unique measured"):p._step5_position_queries(lane)
    model._frontier_moe_physical_rows=()
    with pytest.raises(ValueError,match="actual source/native point"):p._step5_position_queries(lane)

from __future__ import annotations

import pytest

from frontier.moe_routing_runtime import (
    STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH,
    UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH,
    resolve_moe_gating_routing_runtime_path,
)


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
def test_position_runtime_binds_retained_source_and_full_plan_policy(tmp_path, foreign):
    import json
    from types import SimpleNamespace
    from dataclasses import asdict
    from frontier.execution_time_predictor.moe_routing_workload import MoeRoutingWorkload
    from frontier.cc_backend.collective_plan import CollectivePlan,CollectiveTensor
    from tests.unit.test_moe_position_routing import source,materialize
    from frontier.profiling.moe.inputs import MoESourceFrame
    lane=materialize(source()).lane(0)
    tensors=[vars(CollectiveTensor("output","bfloat16",2,4,4))]
    context=dict(model_architecture_profile="step5",routing_runtime_path="position_cyclic",quant_dtype=None,defer_input_quant=False,
        native_facts=dict(shared={"image":"cpu_fixture"}),collective_tensors=dict(dispatch=tensors,output=tensors))
    frames=[asdict(MoESourceFrame(s.lane_id,tuple(i.length for i in s.intervals),tuple(i.start for i in s.intervals),s.physical_rows,s.gating_runtime_context)) for s in lane.routing_source.source_lanes]
    receipt=dict(moe_ep_rank=0,total_routed_tokens=0,routing_input=json.dumps(dict(source_frames=frames)),native_context=json.dumps(context))
    model=SimpleNamespace(_frontier_moe_physical_rows=(receipt,))
    class Predictor(MoeRoutingWorkload):
        _step5_ffn_model=lambda self,*a:(model,None)
        _build_grouped_gemm_features=lambda self,*a,**k:dict(num_tokens=17,total_routed_tokens=0)
    p=Predictor();p._replica_config=SimpleNamespace(device="h800",attn_tensor_parallel_size=8)
    p._model_config=SimpleNamespace(supports_share_expert=lambda:True)
    p._cc_backend=SimpleNamespace(_config=SimpleNamespace(profiling_data_dir=str(tmp_path)))
    measured=dict(shared=context["native_facts"]["shared"],execution_policy={"0":{"branches":["cpu_fixture"]}})
    counts={}
    for phase in ("dispatch","combine","routed_tp","shared_tp"):
        groups=tuple((i,8+i) for i in range(8)) if phase in ("dispatch","combine") else (tuple(range(8)),tuple(range(8,16)))
        plan=CollectivePlan(phase,(16,1),tuple(CollectiveTensor(**t) for t in tensors),groups,"none",json.dumps(measured),json.dumps(dict(actual_buffer="cpu_fixture",compute_context=context)))
        counts[plan.cache_key()]=dict(context=measured,buffer_receipt=json.loads(plan.buffer_json))
    path=tmp_path/"h800/collective_local.json";path.parent.mkdir();path.write_text(json.dumps(dict(counts=counts)))
    native,features,plans=p._step5_position_queries(lane)
    assert native==context and features["moe_gating_linear"]==dict(num_tokens=16,live_routing_token_count=16)
    assert all(json.loads(plan.context_json)==measured for plan in plans.values())
    for key,receipt in list(counts.items()):
        value=json.loads(key);value["buffer_json"]=json.dumps(dict(actual_buffer="foreign",compute_context=foreign));counts[json.dumps(value,sort_keys=True,separators=(",",":"))]=counts.pop(key)
    path.write_text(json.dumps(dict(counts=counts)))
    with pytest.raises(ValueError,match="unique measured"):p._step5_position_queries(lane)
    model._frontier_moe_physical_rows=()
    with pytest.raises(ValueError,match="actual source/native point"):p._step5_position_queries(lane)

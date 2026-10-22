"""Plan importer negatives and real TP formula; CPU fixtures do not qualify a run."""
from dataclasses import replace
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from frontier.cc_backend.backends.collective_sim_cc_backend import CollectiveSimCCBackend
from frontier.cc_backend.cc_backend_config import CollectiveSimCCBackendConfig
from frontier.cc_backend.collective_plan import CollectivePlan, CollectiveTensor, collective_model_context
from frontier.types import ClusterType


def setup_model(tmp_path, monkeypatch):
    config=CollectiveSimCCBackendConfig(cluster_servers=2,cluster_gpus_per_server=8,
        parallel_tp=8,parallel_dp=2,placement_order="TP,DP",profiling_data_dir=str(tmp_path),
        runner_out_dir=str(tmp_path/"runner"))
    backend=CollectiveSimCCBackend(config,ClusterType.MONOLITHIC,"h800","cpu_fixture",16)
    context={"evidence_kind":"cpu_fixture","execution_policy":{"0":{"branches":["cpu_fixture"],
        "runtime":{"pynccl_versions":{"tp":22902}},"buffers":{"input_registered":[False]}}}}
    plan=CollectivePlan("shared_tp",(16,1),(CollectiveTensor("output","bfloat16",2,4096,4096),),
        (tuple(range(8)),tuple(range(8,16))),"none",json.dumps(context),'{"ordinary":true}')
    scenario=backend._build_scenario(kind="plan",tensor_bytes=0,mapped_domain="DP",
        participant_ranks=(),parallel_dim_sizes={"TP":8,"CP":1,"DP":2,"EP":1})
    scenario["collective"].pop("participant_ranks"); scenario.pop("runner")
    scenario["collective"]["intra_server_combine_rule"]="sum"
    fit=dict(a_ms=.014,b_ms_per_byte=1.75/200e9*1000,range_bytes=[8192,8192*8192],
        design_matrix_rank=2,distinct_sizes=3,signed_residuals_ms=[-.001,0,.001],
        calibration_receipt="cpu_fixture",qualification=dict(status="PASS",independent=True,observations=["cpu_fixture"]))
    params=dict(context=collective_model_context(context),model_identity="cpu_fixture",evidence_kind="cpu_fixture",
        tp_nvlink=dict(nvlink_one_way_bw_GBps=250.,nvlink_efficiency=.8,nvlink_latency_us=.5,
                      nvlink_allreduce_launch_overhead_us=.5,fit=fit))
    artifact=dict(schema_version="step5.collective_local.v1",device_type="h800",candidate_only=False,
        evidence_kind="cpu_fixture",frozen_scenario=scenario,
        qualification=dict(status="PASS",independent=True,observations=["cpu_fixture"],parent_receipt="cpu_fixture"),
        models=[dict(context=collective_model_context(context),phase=plan.phase,graph_mode="none",uniform=False,
                     tensors=[vars(t) for t in plan.tensors],parameters=params)],
        counts={plan.cache_key():dict(plan_key=plan.cache_key(),context=context,kernel_counts=[7]*16,
            per_rank_events=[["cpu_fixture"]]*16,source_receipt="cpu_fixture",evidence_kind="cpu_fixture")})
    path=tmp_path/"h800/collective_local.json";path.parent.mkdir();path.write_text(json.dumps(artifact))
    return backend,plan,artifact,path,params


@pytest.mark.parametrize("defect", ["fixture","candidate","qualification","network","context","count","policy","missingpolicy"])
def test_importer_rejects_unqualified_mismatched_or_incomplete_input(tmp_path,monkeypatch,defect):
    backend,plan,artifact,path,_=setup_model(tmp_path,monkeypatch)
    artifact["evidence_kind"]="native_measurement"
    if defect == "fixture": artifact["evidence_kind"]="cpu_fixture"
    elif defect == "candidate": artifact["candidate_only"]=True
    elif defect == "qualification": artifact["qualification"]["independent"]=False
    elif defect == "network": artifact["frozen_scenario"]["network"]["cwnd"]+=1
    elif defect == "context": artifact["models"][0]["context"]={"changed":True}
    elif defect == "count": artifact["counts"]={}
    elif defect == "policy":
        changed={"evidence_kind":"cpu_fixture","execution_policy":{"0":{"runtime":{"pynccl_versions":{"dp":22800}}}}}
        plan=replace(plan,context_json=json.dumps(changed))
    elif defect == "missingpolicy": plan=replace(plan,context_json='{"evidence_kind":"cpu_fixture"}')
    path.write_text(json.dumps(artifact))
    with pytest.raises((ValueError,KeyError)):
        backend.predict_collective_plan(plan)


def test_real_local_tp_formula_has_no_phantom_dp_memory_service(tmp_path,monkeypatch):
    backend,plan,artifact,path,params=setup_model(tmp_path,monkeypatch)
    from collective_sim_core.schema import Scenario
    from collective_sim_core import predict_collective_time
    scenario=dict(artifact["frozen_scenario"])
    scenario["collective"]={**scenario["collective"],"plan":plan.to_dict(),
        "local_parameters":{**params,"context":json.loads(plan.context_json)}}
    result=predict_collective_time(Scenario.from_dict(scenario),repo_root=backend._repo_root)
    expected=1.75*16*8192/200e9*1000+.014
    assert result["predicted_time_ms"] == pytest.approx(expected)
    assert result["breakdown"]["network_ms"] == result["breakdown"]["remainder_ms"] == 0
    assert result["breakdown"]["local_rank_ms"][8] < expected
    params["tp_nvlink"]["nvlink_efficiency"] = .7
    with pytest.raises(ValueError,match="disagree"):
        predict_collective_time(Scenario.from_dict(scenario),repo_root=backend._repo_root)


def test_importer_groups_frozen_policy_but_keeps_complete_count_and_cache_identity(tmp_path,monkeypatch):
    """Synthetic schema labels exercise real CPU import, not native qualification."""
    backend,plan,artifact,path,params=setup_model(tmp_path,monkeypatch)
    artifact["evidence_kind"]=params["evidence_kind"]="native_measurement"
    original=artifact["counts"][plan.cache_key()]
    original.update(evidence_kind="native_measurement",profiled_calls=5,
        per_rank_events=[["cpu_schema_only"]*35 for _ in range(16)])
    path.write_text(json.dumps(artifact))
    expected=backend.predict_collective_plan(plan)
    context=json.loads(plan.context_json)
    context["execution_policy"]["0"].update(branches=["pynccl_comm.all_reduce"],
        buffers=dict(input_registered=[True]))
    context["execution_policy"]["0"]["runtime"]["enabled_communicators"]={"tp":{"fi_ar_comm":False}}
    switched=replace(plan,context_json=json.dumps(context))
    assert collective_model_context(context) == artifact["models"][0]["context"]
    assert switched.cache_key() != plan.cache_key()
    with pytest.raises(KeyError): backend.predict_collective_plan(switched)
    artifact["counts"][switched.cache_key()]={**original,"plan_key":switched.cache_key(),"context":context}
    path.write_text(json.dumps(artifact))
    result=backend.predict_collective_plan(switched)
    assert result.duration_ms == expected.duration_ms and result.plan_key == switched.cache_key()
    assert json.loads(result.count_receipt_json)["context"] == context

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
    assert next(iter(backend._prediction_cache))[-1] is backend.collective_profile_bytes()
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
    with pytest.raises(KeyError): backend.predict_collective_plan(switched)
    backend=CollectiveSimCCBackend(backend.config,backend.cluster_type,"h800","cpu_fixture",16)
    result=backend.predict_collective_plan(switched)
    assert result.duration_ms == expected.duration_ms and result.plan_key == switched.cache_key()
    assert json.loads(result.count_receipt_json)["context"] == context


def cpu_writer_source():
    """Default to this checkout; explicitly sealed references support phased integration."""
    import os,hashlib
    reference=os.environ.get("FRONTIER_TEST_CPU_WRITER_FILE")
    source=Path(reference) if reference else Path(__file__).resolve().parents[2]/"tests/integration/step5_measured_compute_fixture.py"
    if reference and hashlib.sha256(source.read_bytes()).hexdigest() != os.environ.get("FRONTIER_TEST_CPU_WRITER_SHA256"):
        raise ValueError("External CPU writer requires its exact immutable SHA256")
    return source


def cpu_writer_payload(tmp_path,monkeypatch):
    """Execute CORE's retained actual writer without importing or editing its tree."""
    import ast,pandas as pd
    backend,plan,_,path,_=setup_model(tmp_path,monkeypatch)
    context=json.loads(plan.context_json);context["shared"]={"evidence_kind":"cpu_fixture","source_receipt":"CPU_CONTRACT_FIXTURE_NOT_NATIVE"}
    native=dict(native_facts=dict(shared=context["shared"]),collective_tensors=dict(
        dispatch=[vars(CollectiveTensor("activation","bfloat16",2,4096,4096))],output=[vars(plan.tensors[0])]))
    table=tmp_path/"compute.csv"
    pd.DataFrame([dict(moe_frame_scope="dp_gathered",routing_input=json.dumps(dict(source_frames=[dict(physical_rows=16),dict(physical_rows=1)])),
        native_context=json.dumps(native),graph_mode="none")]).to_csv(table,index=False)
    source=cpu_writer_source()
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name == "position_collective_costs")
    ns=dict(pd=pd,json=json);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),"exec"),ns)
    writer_path=ns[fn.name](tmp_path,table,device="h800");artifact=json.loads(writer_path.read_text());path.write_text(json.dumps(artifact))
    return backend,artifact,path


@pytest.mark.parametrize("phase",["dispatch","combine","shared_tp","routed_tp"])
def test_real_cpu_writer_costs_use_actual_constructor(tmp_path,monkeypatch,phase):
    backend,artifact,_=cpu_writer_payload(tmp_path,monkeypatch)
    key=next(k for k in artifact["counts"] if json.loads(k)["phase"] == phase);v=json.loads(key)
    plan=CollectivePlan(v["phase"],v["sizes"],tuple(CollectiveTensor(**t) for t in v["tensors"]),v["groups"],v["graph_mode"],v["context_json"],v["buffer_json"])
    cost=backend.predict_collective_plan(plan)
    assert cost.duration_ms == .03 and cost.plan_key == key
    assert json.loads(cost.count_receipt_json)["evidence_kind"] == "cpu_fixture" and artifact["native_admitted"] is False


@pytest.mark.parametrize("fault",["artifact_kind","plan_kind","schema","device","admitted","candidate","missing_admitted","missing_counts","key","context","buffer","entry_kind","receipt_kind","cost_key","cost_sum","receipt_context","missing_candidate","missing_shared","missing_entry","missing_cost_key","missing_receipt","buffer_kind","receipt_buffer","receipt_key","receipt_counts"])
def test_cpu_writer_identity_and_unadmitted_proof_are_required(tmp_path,monkeypatch,fault):
    backend,artifact,path=cpu_writer_payload(tmp_path,monkeypatch);key=next(iter(artifact["counts"]));v=json.loads(key)
    plan=CollectivePlan(v["phase"],v["sizes"],tuple(CollectiveTensor(**t) for t in v["tensors"]),v["groups"],v["graph_mode"],v["context_json"],v["buffer_json"])
    entry=artifact["counts"][key];receipt=json.loads(entry["cost"]["count_receipt_json"])
    if fault == "artifact_kind": artifact["evidence_kind"]="native_measurement"
    elif fault == "plan_kind": c=json.loads(plan.context_json);c["shared"]["evidence_kind"]="native_measurement";plan=replace(plan,context_json=json.dumps(c))
    elif fault == "schema": artifact["schema_version"]="wrong"
    elif fault == "device": artifact["device_type"]="h200"
    elif fault == "admitted": artifact["native_admitted"]=True
    elif fault == "candidate": artifact["candidate_only"]=False
    elif fault == "missing_admitted": artifact.pop("native_admitted")
    elif fault == "missing_candidate": artifact.pop("candidate_only")
    elif fault == "missing_shared": c=json.loads(plan.context_json);c["shared"].pop("evidence_kind");plan=replace(plan,context_json=json.dumps(c))
    elif fault == "missing_counts": artifact.pop("counts")
    elif fault == "missing_entry": artifact["counts"].pop(key)
    elif fault == "missing_cost_key": entry["cost"].pop("plan_key")
    elif fault == "missing_receipt": entry["cost"]["count_receipt_json"]="{}";receipt={}
    elif fault == "buffer_kind": b=json.loads(plan.buffer_json);b["evidence_kind"]="native_measurement";plan=replace(plan,buffer_json=json.dumps(b))
    elif fault == "key": entry["plan_key"]="wrong"
    elif fault == "context": entry["context"]={"wrong":True}
    elif fault == "buffer": entry["buffer_receipt"]={"wrong":True}
    elif fault == "entry_kind": entry["evidence_kind"]="native_measurement"
    elif fault == "receipt_kind": receipt["evidence_kind"]="native_measurement"
    elif fault == "cost_key": entry["cost"]["plan_key"]="wrong"
    elif fault == "cost_sum": entry["cost"]["duration_ms"]=99.
    elif fault == "receipt_buffer": receipt["buffer_receipt"]={"wrong":True}
    elif fault == "receipt_key": receipt["plan_key"]="wrong"
    elif fault == "receipt_counts": receipt["kernel_counts"]=[7]*16
    else: receipt["context"]={"wrong":True}
    entry["cost"]["count_receipt_json"]=json.dumps(receipt);path.write_text(json.dumps(artifact))
    with pytest.raises((ValueError,KeyError,TypeError)): backend.predict_collective_plan(plan)


def test_native_import_ignores_injected_full_cost(tmp_path,monkeypatch):
    backend,plan,artifact,path,params=setup_model(tmp_path,monkeypatch)
    artifact["evidence_kind"]=params["evidence_kind"]="native_measurement"
    entry=artifact["counts"][plan.cache_key()];entry.update(evidence_kind="native_measurement",profiled_calls=5,
        per_rank_events=[["cpu_schema_only"]*35 for _ in range(16)],cost={"duration_ms":999.})
    path.write_text(json.dumps(artifact));cost=backend.predict_collective_plan(plan)
    assert cost.duration_ms == pytest.approx(1.75*16*8192/200e9*1000+.014)


@pytest.mark.parametrize("mode",["local","missing_hash","wrong_hash","missing_file","exact"])
def test_cpu_writer_reference_is_explicit_and_sealed(tmp_path,monkeypatch,mode):
    import hashlib
    monkeypatch.delenv("FRONTIER_TEST_CPU_WRITER_FILE",raising=False)
    monkeypatch.delenv("FRONTIER_TEST_CPU_WRITER_SHA256",raising=False)
    if mode == "local":
        assert cpu_writer_source() == Path(__file__).resolve().parents[2]/"tests/integration/step5_measured_compute_fixture.py"
        return
    source=tmp_path/"writer.py";source.write_text("# CPU reference identity fixture\n")
    monkeypatch.setenv("FRONTIER_TEST_CPU_WRITER_FILE",str(source if mode != "missing_file" else tmp_path/"absent.py"))
    if mode != "missing_hash": monkeypatch.setenv("FRONTIER_TEST_CPU_WRITER_SHA256",hashlib.sha256(source.read_bytes()).hexdigest() if mode != "wrong_hash" else "wrong")
    if mode == "exact": assert cpu_writer_source() == source
    else:
        with pytest.raises((ValueError,FileNotFoundError)): cpu_writer_source()

def test_collective_artifact_is_fixed_and_failed_initial_reads_are_not_published(tmp_path,monkeypatch):
    import os
    backend,artifact,path=cpu_writer_payload(tmp_path,monkeypatch)
    key=next(iter(artifact["counts"]));v=json.loads(key)
    plan=CollectivePlan(v["phase"],v["sizes"],tuple(CollectiveTensor(**t) for t in v["tensors"]),v["groups"],v["graph_mode"],v["context_json"],v["buffer_json"])
    text=path.read_text();stat=path.stat();reads=[];failure=[];original=Path.read_bytes
    def read(file):
        reads.append(file)
        if failure: raise failure[0]
        return original(file)
    monkeypatch.setattr(Path,"read_bytes",read)
    def new_backend():
        return CollectiveSimCCBackend(backend.config,backend.cluster_type,"h800","cpu_fixture",16)
    first=backend.predict_collective_plan(plan);assert backend.predict_collective_plan(plan)==first
    assert len(reads)==1
    with pytest.raises(KeyError): backend.predict_collective_plan(replace(plan,buffer_json='{"evidence_kind":"cpu_fixture"}'))
    artifact["counts"][key]["cost"].update(duration_ms=.04,network_ms=.02)
    changed=json.dumps(artifact);assert len(changed)==len(text)
    path.write_text(changed);os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns));assert path.stat().st_mtime_ns==stat.st_mtime_ns
    assert backend.predict_collective_plan(plan)==first and len(reads)==1
    assert new_backend().predict_collective_plan(plan).duration_ms==.04
    for raw,error in ((b"{",json.JSONDecodeError),(b"\xff",UnicodeDecodeError)):
        path.write_bytes(raw);candidate=new_backend()
        assert backend.predict_collective_plan(plan)==first
        for _ in range(2):
            with pytest.raises(error): candidate.predict_collective_plan(plan)
        path.write_text(changed)
        assert candidate.predict_collective_plan(plan).duration_ms==.04
    for error in (FileNotFoundError("current artifact"),PermissionError("current artifact")):
        failure[:]=[error];candidate=new_backend()
        for _ in range(2):
            with pytest.raises(type(error)) as caught: candidate.predict_collective_plan(plan)
            assert caught.value is error
        failure.clear()
        assert candidate.predict_collective_plan(plan).duration_ms==.04

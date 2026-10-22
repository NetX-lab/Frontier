"""CPU-only RDMA channel interpretation and post-run identity checks."""

import csv
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from frontier.profiling.collectives.vllm_collective_benchmark import CollectiveRunner, active_dp_rails
from frontier.profiling.collectives.vllm_forward_collectives import (
    ModelShape, ServingParallelism, call_bytes, forward_collectives, group_members, scenarios_from_grid,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tests/integration/step5_comm"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("channel,status", [
    ("Channel 00 : 0 -> 8 via NET/IB/0/GDRDMA", "PASS"),
    ("NET/IB: Using mlx5_0", "UNVERIFIED"),
    ("Channel 00 : 0 -> 8 via NET/IB/0", "UNVERIFIED"),
    ("Channel 00 : 0 -> 8 via NET/Socket/0", "UNVERIFIED"),
    ("Using network Socket", "UNVERIFIED")])
def test_actual_channels_not_available_plugins_qualify_rdma(tmp_path, channel, status):
    paths = []
    for rank in range(2):
        path = tmp_path / f"node_{rank}.log"
        path.write_text("Bootstrap: Using socket interface eth0\nNET/Socket: available plugin\n" + channel + "\n")
        paths.append(path)
    report = load("transport").check_logs(paths, 2)
    assert report["status"] == status and report["advisory_only"] is True
    assert load("transport").check_logs(paths[:1], 2)["status"] == "UNVERIFIED"
    assert load("transport").check_logs(paths[:1], 1)["status"] == "NOT_APPLICABLE"


@pytest.mark.parametrize("active", [False, True])
def test_inventory_records_verbs_devices_and_real_link_state(tmp_path, monkeypatch, active):
    module = load("transport")
    monkeypatch.setattr(module, "verbs_devices", lambda: ["mlx5_0"])
    nodes = tmp_path / "dev"
    nodes.mkdir()
    (nodes / "uverbs0").touch()
    port = tmp_path / "sys/mlx5_0/ports/1"
    port.mkdir(parents=True)
    for field, value in {"state": "4: ACTIVE" if active else "1: DOWN",
                         "link_layer": "Ethernet", "rate": "200 Gb/sec"}.items():
        (port / field).write_text(value)
    report = module.snapshot(tmp_path / "sys", nodes)
    assert report["verbs_devices"] == ["mlx5_0"]
    assert report["ports"][0]["link_layer"] == "Ethernet"
    assert report["status"] == ("PASS" if active else "UNVERIFIED")


def fixture_result(tmp_path, monkeypatch, stage, native_call_names=None, definition=None):
    monkeypatch.syspath_prepend(str(PACKAGE))
    definition = definition or json.loads((PACKAGE / f"{stage}_run.json").read_text())
    definition["grid"] = {**definition["grid"], "scenarios": definition["grid"]["scenarios"][-3:],
                          "eager":[],"cuda_graph_batch_sizes":[]}
    if native_call_names is not None:
        definition["grid"]["native_call_names"] = native_call_names
    definition["model_config"] = str(ROOT / "data/config/models/stepfun-ai__Step-5-Preview.json")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(definition))
    parallel = ServingParallelism.from_engine(definition["engine"])
    model = ModelShape.from_hf_config(json.loads(Path(definition["model_config"]).read_text()))
    runner = SimpleNamespace(parallel=parallel, instance=0, groups={})
    records, kernels, summaries = [], [], []
    for scenario in scenarios_from_grid(parallel, definition["grid"]):
        for call in forward_collectives(model, parallel, scenario,
                                        native_call_names=definition["grid"].get("native_call_names")):
            runner.groups[call.group] = SimpleNamespace(world_size=len(group_members(parallel, call.group, 0)))
            rails = active_dp_rails(parallel, call, scenario, False)
            gathered = [{"bytes": call_bytes(call, parallel, r)}
                        if call.shard_rows[r] and (call.group != "dp" or r % parallel.tensor_parallel_size in rails)
                        else None for r in range(parallel.world_size)]
            base = CollectiveRunner.call_record(runner, scenario, call, gathered)
            kernels.append({"scenario": scenario.name, "call": call.name,
                            "communicator": "gloo" if call.op == "cpu_all_reduce" else "nccl"})
            for p in range(definition["passes"]):
                record = {"pass": p, **base,
                          "per_rank_call_us": [[10.,10.,10.] if v else None for v in gathered],
                          "per_rank_launch_call_us": [[10.,10.,10.] if v else None for v in gathered],
                          "per_rank_queue_held": [([True]*3 if call.op != "cpu_all_reduce" else None)
                                                  if v else None for v in gathered]}
                records.append(record)
                summaries.append({"pass":p,"scenario":scenario.name,"call":call.name,"median_us":10.})
    collectives = tmp_path / "collectives"
    collectives.mkdir()
    for name, values in (("measurements", records), ("kernels", kernels)):
        (collectives / f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in values))
    with (collectives / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pass","scenario","call","median_us"])
        writer.writeheader(); writer.writerows(summaries)
    native = tmp_path / "native_dispatch"
    native.mkdir()
    ranks = [{"rank":r,"host":f"node_{r//8}","status":"match",
              "group_ranks":{g:group_members(parallel,g,r) for g in ("tp","dp","ep")},
              "layers":[{"layer_id":layer,"comparison":{"status":"match"},
                         "collectives":[{"completed":True}]*4} for layer in (3,88)]}
             for r in range(parallel.world_size)]
    (native / "dispatch_probe.json").write_text(json.dumps({"status":"match","ranks":ranks}))
    outputs = [tmp_path] if definition["resources"]["replica"]==1 else [tmp_path/f"node_{i}" for i in range(definition["resources"]["replica"])]
    for output in outputs:
        output.mkdir(exist_ok=True)
        for name in ("source_integrity","transport_snapshot","manifest"):
            (output/f"{name}.json").write_text('{"status":"PASS"}')
        for phase in ("collective","native_probe"):
            (output/f"{phase}_exit.json").write_text('{"returncode":0}')
    (tmp_path/"transport.json").write_text('{"status":"PASS"}')
    job = {**definition,"benchmark":str(benchmark),
           "expected":{"measurements":len(records),"kernel_rows":len(kernels)}}
    return job, records


@pytest.mark.parametrize("stage", ["s0","s1","s2"])
def test_real_call_record_metadata_passes_advisory_validation(tmp_path, monkeypatch, stage):
    job, _ = fixture_result(tmp_path, monkeypatch, stage)
    report = load("validate_result").validate(job, tmp_path)
    assert report["status"] == "PASS", report
    assert report["advisory_only"] is True and report["cost_data_qualified"] is False


@pytest.mark.parametrize("defect", ["queue","participant","communicator","source","transport"])
def test_corrupt_evidence_is_unverified_not_a_workload_exception(tmp_path, monkeypatch, defect):
    job, records = fixture_result(tmp_path, monkeypatch, "s1")
    if defect == "queue":
        records[0]["per_rank_call_us"][0] = [100.,20.,10.]
        records[0]["per_rank_queue_held"][0] = [False,True,True]
    elif defect == "participant":
        records[0]["participating_ranks"] = [0]
    elif defect == "source":
        (tmp_path/"node_0/source_integrity.json").write_text('{"status":"UNVERIFIED"}')
    elif defect == "transport":
        (tmp_path/"transport.json").write_text('{"status":"UNVERIFIED"}')
    else:
        path = tmp_path/"collectives/kernels.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        values[0]["communicator"] = "unknown"
        path.write_text("".join(json.dumps(r)+"\n" for r in values))
    (tmp_path/"collectives/measurements.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    report = load("validate_result").validate(job, tmp_path)
    assert report["status"] == "UNVERIFIED" and report["advisory_only"] is True
    if defect == "queue":
        assert report["checks"]["queue_held"] is False


def test_missing_artifacts_cli_returns_advisory_zero(tmp_path):
    config = tmp_path / "job.json"
    config.write_text('{}')
    result = subprocess.run([sys.executable,str(PACKAGE/"validate_result.py"),"--config",str(config),"--root",str(tmp_path)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)["status"] == "UNVERIFIED"


@pytest.mark.parametrize("defect", [None, "missing", "duplicate", "extra"])
def test_selected_native_plan_evidence_is_complete_and_exact(tmp_path, monkeypatch, defect):
    names=["moe_dispatch_all_gatherv_fp8", "moe_combine_reduce_scatterv",
           "shared_expert_down_proj_all_reduce", "routed_expert_all_reduce"]
    job,records=fixture_result(tmp_path,monkeypatch,"s2",names)
    assert len(records) == 3*4*job["passes"]
    assert any(r["op"] == "reduce_scatterv_into_output" for r in records)
    if defect == "missing":
        records.pop()
    elif defect == "duplicate":
        records[-1]=dict(records[0])
    elif defect == "extra":
        records[-1]={**records[-1],"call":"embedding_all_reduce"}
    (tmp_path/"collectives/measurements.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    report=load("validate_result").validate(job,tmp_path)
    assert report["status"] == ("PASS" if defect is None else "UNVERIFIED"), report


def test_selected_missing_call_cannot_pass_with_reduced_superficial_counts(tmp_path,monkeypatch):
    names=["moe_dispatch_all_gatherv_fp8","moe_combine_reduce_scatterv"]
    job,records=fixture_result(tmp_path,monkeypatch,"s2",names)
    records=[r for r in records if r["call"] != names[0]]
    job["expected"]["measurements"]=len(records)
    path=tmp_path/"collectives/measurements.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in records))
    report=load("validate_result").validate(job,tmp_path)
    assert report["checks"]["prepared_counts"] is False
    assert report["checks"]["measurement_keys"] is False


@pytest.mark.parametrize("collection",["kernels","summary"])
def test_selected_duplicate_kernel_or_summary_keys_are_unverified(tmp_path,monkeypatch,collection):
    job,_=fixture_result(tmp_path,monkeypatch,"s2",["moe_combine_reduce_scatterv"])
    if collection == "kernels":
        path=tmp_path/"collectives/kernels.jsonl";lines=path.read_text().splitlines()
        lines[-1]=lines[0];path.write_text("\n".join(lines)+"\n")
    else:
        path=tmp_path/"collectives/summary.csv"
        rows=list(csv.DictReader(path.open()));rows[-1]=dict(rows[0])
        with path.open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=load("validate_result").validate(job,tmp_path)
    key="kernel_keys" if collection == "kernels" else "summary_keys"
    assert report["status"] == "UNVERIFIED" and report["checks"][key] is False


@pytest.mark.parametrize("defect",[None,"rank","repeat","queue"])
def test_local_proxy_evidence_requires_every_rank_and_repeat(tmp_path,monkeypatch,defect):
    job,_=fixture_result(tmp_path,monkeypatch,"s2",["moe_combine_reduce_scatterv"])
    config_path=Path(job["benchmark"]);config=json.loads(config_path.read_text())
    definition=dict(name="bf16_copy",kind="copy",dtype="bfloat16",columns=4096,rows=16,
                    cuda_graph=False,split="calibration")
    config["grid"]["local_work"]=[definition];config_path.write_text(json.dumps(config))
    rows=[dict(definition,**{"pass":p},scope="independent_preallocated_operand",row_stride=4096,
        per_rank_call_us=[[1.]*config["repeats"] for _ in range(16)],
        per_rank_queue_held=[[True]*config["repeats"] for _ in range(16)]) for p in range(config["passes"])]
    if defect == "rank": rows[0]["per_rank_queue_held"].pop()
    if defect == "repeat": rows[0]["per_rank_call_us"][0].pop()
    if defect == "queue": rows[0]["per_rank_queue_held"][0][0]=False
    job["expected"]["local_work_rows"]=len(rows)
    (tmp_path/"collectives/local_work.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    report=load("validate_result").validate(job,tmp_path)
    assert report["checks"]["local_layout_queue"] is (defect is None)
    assert report["status"] == ("PASS" if defect is None else "UNVERIFIED")


@pytest.mark.parametrize("defect",[None,"source","generation","count","duplicate_probe","extra_probe",
                                  "op","group","cuda_graph","replay","runtime_policy","branch","missing_policy",
                                  "lazy_fallback","stale_fallback","buffers","frozen_environment"])
def test_bound_selected_evidence_keeps_generation_source_and_exact_probe_keys(tmp_path,monkeypatch,defect):
    from dataclasses import asdict
    from tests.unit.test_step5_fixed_probe import installed_profile_receipt
    from frontier.profiling.moe.inputs import MoESourceFrame,MoERoutingProfileInput
    manifest_path,_,manifest=installed_profile_receipt.__wrapped__(tmp_path,monkeypatch)
    lazy=defect in ("lazy_fallback","stale_fallback")
    job,records=fixture_result(tmp_path,monkeypatch,"s2",[
        "shared_expert_down_proj_all_reduce" if lazy else "moe_combine_reduce_scatterv"])
    binding=manifest["native_binding"]
    keys=("native_runtime_adapter","native_reference","profile_generation","installed_package",
          "model_config_revision","model_config_sha256","installed_source_hashes")
    shared={k:binding[k] for k in keys}
    job.update(profile_manifest=str(manifest_path),image=manifest["image"])
    inp=MoERoutingProfileInput("tail16",(MoESourceFrame(0,(16,),(6128,),16,"standalone_legacy"),
        MoESourceFrame(1,(),(),1,"standalone_legacy")),17)
    path=tmp_path/"routing.json";path.write_text(json.dumps(dict(schema_version="moe.position_cyclic.inputs.v1",
        routing_runtime_path="position_cyclic",inputs=[asdict(inp)])))
    job["native_probe"].update(moe_input_ids=["tail16"],routing_input_file=str(path))
    runtime=dict(torch="cpu_fixture",vllm="cpu_fixture",pynccl_versions={"dp":22902},
        enabled_communicators={"dp":{"pynccl_comm":True}},vllm_environment={"VLLM_USE_NCCL_SYMM_MEM":{"value":False}},
        nccl_environment={},disable_custom_all_reduce=False,disable_nccl_for_dp_synchronization=True)
    policy=dict(runtime=runtime,branches=["pynccl_comm.reduce_scatterv"],buffers={"supplied_output":True})
    startup=deepcopy(runtime)
    if lazy:
        policy=lazy_fallback_policy(monkeypatch,runtime)
        startup["enabled_communicators"]=deepcopy(policy["runtime"]["enabled_communicators"])
        startup["enabled_communicators"]["tp"]["fi_ar_comm"]=True
    for r in records: r.update(evidence_kind="native_measurement",native_context=dict(shared=shared,
        execution_policy={str(rank):deepcopy(policy) for rank in r["participating_ranks"]}))
    if defect == "runtime_policy": records[0]["native_context"]["execution_policy"]["0"]["runtime"]["pynccl_versions"]["dp"]=22800
    if defect == "buffers": records[0]["native_context"]["execution_policy"]["0"]["buffers"]["supplied_output"]=False
    if defect == "frozen_environment": records[0]["native_context"]["execution_policy"]["0"]["runtime"]["vllm_environment"]["VLLM_USE_NCCL_SYMM_MEM"]["value"]=True
    if defect == "branch": records[0]["native_context"]["execution_policy"]["0"]["branches"]=["_reduce_scatter_symm_mem"]
    if defect == "missing_policy": records[0]["native_context"].pop("execution_policy")
    if defect == "generation": records[0]["native_context"]["shared"]={**shared,"profile_generation":"different"}
    if defect in ("op","group","cuda_graph"):
        records[0][defect]={"op":"reduce_scatterv","group":"tp","cuda_graph":not records[0]["cuda_graph"]}[defect]
    (tmp_path/"collectives/measurements.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (tmp_path/"collectives/plan.json").write_text(json.dumps(dict(evidence_kind="native_measurement",
        runtime=dict(native_facts=dict(shared=shared),per_rank_startup_runtime={str(rank):startup for rank in range(16)}))))
    (tmp_path/"collectives/native_identity.json").write_text('{"status":"PASS"}')
    for i in range(job["resources"]["replica"]):
        (tmp_path/f"node_{i}/native_binding.json").write_text('{"status":"PASS"}')
    kernel_path=tmp_path/"collectives/kernels.jsonl"
    kernels=[json.loads(line) for line in kernel_path.read_text().splitlines()]
    for k in kernels: k.update(profiled_calls=5,per_rank_kernel_counts=[2]*16,per_rank_events=[["cpu_fixture"]*10 for _ in range(16)])
    if defect == "count": kernels[0]["per_rank_kernel_counts"][0]=3
    if defect == "replay": kernels[0].update(profiled_calls=1,per_rank_kernel_counts=[35]*16,
        per_rank_events=[["cpu_fixture"]*35 for _ in range(16)])
    kernel_path.write_text("".join(json.dumps(k)+"\n" for k in kernels))
    probe_path=tmp_path/"native_dispatch/dispatch_probe.json";probe=json.loads(probe_path.read_text())
    probe.update(verification_scope="eager_native_moe_module",full_model_verified=False,model_runner_verified=False)
    for r in probe["ranks"]:
        for layer in r["layers"]:
            layer.update(moe_input_id="tail16",source_frame=asdict(inp.source_frames[r["rank"]//8]),
                native_facts=dict(shared=shared,callables={"cpu_fixture":"explicit_schema_test"}),
                collectives=[dict(completed=True,op="all_reduce" if lazy else "reduce_scatterv_into_output",
                    group_ranks=r["group_ranks"]["tp" if lazy else "dp"],
                    execution_policy=deepcopy(policy),inputs=[dict(dtype="bfloat16",shape=[17,4096])])])
    if defect == "stale_fallback":
        for layer in probe["ranks"][0]["layers"]:
            layer["collectives"][0]["execution_policy"]["runtime"]["enabled_communicators"]["tp"]["fi_ar_comm"]=True
    if defect == "source": probe["ranks"][0]["layers"][0]["source_frame"]["context_lens"]=[0]
    if defect == "duplicate_probe": probe["ranks"][0]["layers"][-1]=deepcopy(probe["ranks"][0]["layers"][0])
    if defect == "extra_probe": probe["ranks"][0]["layers"][0]["moe_input_id"]="extra"
    probe_path.write_text(json.dumps(probe))
    report=load("validate_result").validate(job,tmp_path)
    assert report["status"] == ("PASS" if defect in (None,"lazy_fallback") else "UNVERIFIED"),report


def lazy_fallback_policy(monkeypatch,runtime):
    """Execute published workspace failure and successful native PyNccl fallback."""
    import ast,types
    from frontier.profiling.collectives.step5_dispatch_probe import DispatchObserver
    source=Path("/data/ycfeng/tmp/d29_published_sources01/source/vllm/distributed/device_communicators")
    if not source.exists(): pytest.skip("Published native sources unavailable")
    tensor=SimpleNamespace(shape=(16,4096),dtype="bfloat16",nbytes=131072,is_cuda=True,device="cpu",
        stride=lambda:(4096,1),element_size=lambda:2,numel=lambda:65536,is_contiguous=lambda:True,data_ptr=lambda:1)
    torch=SimpleNamespace(dtype=object,Tensor=object,bfloat16="bfloat16",float16="fp16",float32="fp32",
        tensor=lambda *a,**kw:SimpleNamespace(element_size=lambda:2),distributed=SimpleNamespace(),
        ops=SimpleNamespace(vllm=SimpleNamespace()))
    namespace=dict(torch=torch,get_fi_ar_workspace=lambda **kw:None,should_nccl_symm_mem_allreduce=lambda *a:False)
    for filename,methods in (("flashinfer_all_reduce.py",("_ensure_workspace","should_use_fi_ar")),
                             ("cuda_communicator.py",("all_reduce",))):
        tree=ast.parse((source/filename).read_text())
        nodes=[n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body
               if isinstance(n,ast.FunctionDef) and n.name in methods]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,"exec"),namespace)
    fi=SimpleNamespace(disabled=False,max_num_tokens=0,max_workspace_size=1<<20,world_size=8,rank=0,group=None)
    for name in ("_ensure_workspace","should_use_fi_ar"): setattr(fi,name,types.MethodType(namespace[name],fi))
    comm=SimpleNamespace(fi_ar_comm=fi,pynccl_comm=SimpleNamespace(world_size=8,disabled=False,all_reduce=lambda t:t),
        qr_comm=None,fi_pcie_ipc_ar_comm=None,aiter_ar_comm=None,use_aiter_allreduce=False,ca_comm=None,symm_mem_comm=None)
    allocator=types.ModuleType("vllm.distributed.device_communicators.pynccl_allocator")
    allocator.is_symmetric_memory_tensor=lambda t:False
    monkeypatch.setitem(sys.modules,allocator.__name__,allocator);monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setenv("FRONTIER_STEP5_PROFILE_MANIFEST","cpu_fixture")
    class Coordinator:
        ranks=list(range(8));device_communicator=comm
        def all_reduce(self,input_): return namespace["all_reduce"](comm,input_)
        all_gatherv=reduce_scatterv=reduce_scatterv_into_output=all_reduce
    observer=DispatchObserver(lambda event:None)
    observer.runtime_policy=lambda:{**runtime,"enabled_communicators":{"tp":{"fi_ar_comm":not fi.disabled,"pynccl_comm":True}}}
    assert observer.runtime_policy()["enabled_communicators"]["tp"]["fi_ar_comm"]
    with observer.collectives(Coordinator): result=Coordinator().all_reduce(tensor)
    policy=observer.calls[0]["execution_policy"]
    assert result is tensor and fi.disabled and policy["branches"] == ["pynccl_comm.all_reduce"]
    assert not policy["runtime"]["enabled_communicators"]["tp"]["fi_ar_comm"] and not observer.errors
    return policy

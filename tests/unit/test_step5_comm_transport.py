"""CPU-only RDMA channel interpretation and post-run identity checks."""

import csv
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

"""Offline StepMind node mapping, workload ordering and advisory isolation tests."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tests/integration/step5_comm"


def load(name):
    spec = importlib.util.spec_from_file_location("step5_multinode_" + name, PACKAGE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared_job(tmp_path, tp=8, dp=4, nodes=4):
    definition = json.loads((PACKAGE / "s0_run.json").read_text())
    definition["engine"].update(tensor_parallel_size=tp, data_parallel_size=dp)
    definition["resources"].update(replica=nodes)
    definition.update(run_id="offline", output_dir=str(tmp_path / "output"),
                      repository=str(tmp_path), benchmark=str(tmp_path / "benchmark.json"),
                      rendezvous={"master_port": 29571, "probe_port": 29572}, expected={})
    definition["native_probe"].update(tokens_per_dp_rank=[8] + [1] * (dp - 1),
                                      model_path=str(tmp_path / "model"))
    return definition


def environment(rank, nodes=4):
    return dict(NODE_RANK=str(rank), NODE_COUNT=str(nodes), PROC_PER_NODE="8",
                MASTER_ADDR="10.191.6.60", MASTER_PORT="29571",
                NCCL_IB_HCA="platform-selection", NCCL_SOCKET_IFNAME="platform-interface")


@pytest.mark.parametrize("tp,dp,nodes", [(4, 2, 1), (8, 4, 4), (8, 2, 2)])
def test_native_topologies_cover_each_rank_once(tmp_path, tp, dp, nodes):
    worker = load("worker")
    job = prepared_job(tmp_path, tp, dp, nodes)
    ranks = []
    for rank in range(nodes):
        env = environment(rank, nodes)
        before = dict(env)
        node = worker.node_identity(job, env)
        ranks.extend(node["ranks"])
        assert node["ranks"] == list(range(rank * 8, (rank + 1) * 8))
        collective, probe = worker.collective_command(job, node), worker.probe_command(job, node)
        assert "--nproc-per-node=8" in collective
        assert "--nproc-per-node=8" in probe
        if nodes == 1:
            assert "--standalone" in collective and "--standalone" in probe
        else:
            for command in (collective, probe):
                assert "--standalone" not in command
                assert f"--nnodes={nodes}" in command
                assert f"--node-rank={rank}" in command
                assert "--master-addr=10.191.6.60" in command
            assert "--master-port=29571" in collective
            assert "--master-port=29572" in probe
        for flag, expected in (("--tensor-parallel-size", tp), ("--data-parallel-size", dp),
                               ("--gpus-per-node", 8)):
            assert probe[probe.index(flag) + 1] == str(expected)
        assert probe[-(dp + 1):] == ["--tokens-per-dp-rank", "8", *["1"] * (dp - 1)]
        assert env == before
    assert ranks == list(range(tp * dp))


@pytest.mark.parametrize("path,value", [
    (("resources", "gpu"), 4), (("resources", "replica"), 2),
    (("gpus_per_node",), 4), (("engine", "tensor_parallel_size"), 4),
    (("engine", "data_parallel_size"), 2), (("engine", "pipeline_parallel_size"), 2),
    (("engine", "enable_expert_parallel"), False), (("resources", "gpu"), 0),
    (("resources", "replica"), -1), (("gpus_per_node",), True),
    (("native_probe", "tokens_per_dp_rank"), [1, 1]),
    (("native_probe", "tokens_per_dp_rank"), [8, 0, 1, 1]),
    (("native_probe", "tokens_per_dp_rank"), [8, True, 1, 1]),
    (("rendezvous", "master_port"), 0), (("rendezvous", "master_port"), 65536),
    (("rendezvous", "master_port"), "29571"), (("rendezvous", "probe_port"), 29571),
])
def test_prepared_world_and_topology_must_match(tmp_path, path, value):
    job = prepared_job(tmp_path)
    target = job
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        load("worker").validate_topology(job)


@pytest.mark.parametrize("key,value", [
    ("NODE_COUNT", "1"), ("NODE_COUNT", "2"), ("NODE_COUNT", "0"),
    ("NODE_COUNT", "5"), ("NODE_RANK", "-1"), ("NODE_RANK", "4"),
    ("NODE_RANK", "invalid"), ("PROC_PER_NODE", "4"), ("PROC_PER_NODE", "16"),
    ("MASTER_ADDR", ""), ("MASTER_ADDR", "localhost"), ("MASTER_ADDR", "LOCALHOST."),
    ("MASTER_ADDR", "127.0.0.1"), ("MASTER_ADDR", "127.3.2.1"),
    ("MASTER_ADDR", "0.0.0.0"), ("MASTER_ADDR", "::1"), ("MASTER_ADDR", "::"),
    ("MASTER_ADDR", "::ffff:127.0.0.1"), ("MASTER_ADDR", "::ffff:0.0.0.0"),
])
def test_runtime_env_rejects_wrong_ranks_and_local_rendezvous(tmp_path, key, value):
    env = environment(0)
    env[key] = value
    with pytest.raises(ValueError):
        load("worker").node_identity(prepared_job(tmp_path), env)


@pytest.mark.parametrize("key", ["NODE_COUNT", "NODE_RANK", "PROC_PER_NODE", "MASTER_ADDR"])
def test_runtime_env_does_not_guess_missing_platform_values(tmp_path, key):
    env = environment(0)
    del env[key]
    with pytest.raises(KeyError, match=key):
        load("worker").node_identity(prepared_job(tmp_path), env)


def test_commands_read_the_platform_environment_and_engine_flags(tmp_path, monkeypatch):
    worker = load("worker")
    job = prepared_job(tmp_path)
    for name, value in environment(3).items():
        monkeypatch.setenv(name, value)
    job["engine"]["disable_custom_all_reduce"] = True
    command = worker.probe_command(job)
    assert "--node-rank=3" in command
    assert command[-1] == "--disable-custom-all-reduce"
    assert worker.collective_command(job)[-2:] == ["--config", job["benchmark"]]


@pytest.mark.parametrize("nodes", [1, 2, 4])
@pytest.mark.parametrize("manifest_failure", [False, True])
def test_sibling_nodes_finish_both_phases_before_single_writer_results(tmp_path, monkeypatch, nodes, manifest_failure):
    worker = load("worker")
    job = prepared_job(tmp_path, 4 if nodes == 1 else 8, 2 if nodes == 1 else nodes, nodes)
    (tmp_path / "source.py").write_text("changed")
    job["source_hashes"] = {"source.py": hashlib.sha256(b"frozen").hexdigest()}
    coordination = tmp_path / "runtime"
    outputs = {}
    calls, common_writers = [], []
    native_output = Path(job["output_dir"]) / "native_dispatch"
    real_write = worker.write_json

    def checked_write(path, value):
        if path == Path(job["output_dir"]) / "result.json":
            assert all((coordination / f"node_{rank}_native_probe_exit.json").is_file()
                       for rank in range(nodes))
            common_writers.append(threading.current_thread().name)
        real_write(path, value)

    def process(command):
        rank = int(next((part.split("=", 1)[1] for part in command if part.startswith("--node-rank=")), "0"))
        phase = "native_probe" if "frontier.profiling.collectives.step5_dispatch_probe" in command else "collective"
        if phase == "native_probe":
            assert all((coordination / f"node_{peer}_collective_exit.json").is_file() for peer in range(nodes))
            if rank == 0:
                native_output.mkdir()
                real_write(native_output / "dispatch_probe.json", {"status": "PASS"})
        calls.append((rank, phase))
        return SimpleNamespace(returncode=0)

    def validate(prepared, root):
        assert prepared == job and root == Path(job["output_dir"])
        assert len(calls) == nodes * 2
        raise RuntimeError("Injected validator failure")

    def check_logs(paths, expected_nodes):
        assert expected_nodes == nodes and len(paths) == nodes
        return {"status": "FAIL", "reason": "Injected missing transport evidence"}

    def snapshot():
        raise RuntimeError("Injected auxiliary preflight failure")

    def manifest(job, node):
        if manifest_failure:
            raise RuntimeError("Injected runtime metadata failure")
        return {"host": "fake", "node": node}

    monkeypatch.setattr(worker, "write_json", checked_write)
    monkeypatch.setattr(worker.subprocess, "run", process)
    monkeypatch.setattr(worker, "runtime_manifest", manifest)
    monkeypatch.setattr(worker, "local_module", lambda name: SimpleNamespace(
        snapshot=snapshot, check_logs=check_logs) if name == "transport" else SimpleNamespace(validate=validate))

    def run_node(rank):
        threading.current_thread().name = f"node_{rank}"
        node = worker.node_identity(job, environment(rank, nodes))
        output = worker.prepare_output(job, node, coordination, time.monotonic() + 15)
        outputs[rank] = output
        (output / "worker.log").write_text("fake native stdout\n")
        return worker.run_job(job, node, output, coordination, time.monotonic() + 15)

    with ThreadPoolExecutor(max_workers=nodes) as pool:
        # Start the non-writer first to exercise the shared-root readiness wait.
        futures = {rank: pool.submit(run_node, rank) for rank in reversed(range(nodes))}
        results = {rank: future.result() for rank, future in futures.items()}
    assert common_writers == ["node_0"]
    assert all(results[rank] is None for rank in range(1, nodes))
    assert results[0]["execution_completed"] is True
    assert results[0]["native_dispatch_status"] == "PASS"
    assert results[0]["validation"]["status"] == "UNVERIFIED"
    assert results[0]["transport"]["status"] == "FAIL"
    assert sorted(calls) == sorted((rank, phase) for rank in range(nodes) for phase in ("collective", "native_probe"))
    assert json.loads((Path(job["output_dir"]) / "result.json").read_text()) == results[0]
    for rank, output in outputs.items():
        source = json.loads((output / "source_integrity.json").read_text())
        assert source["status"] == "UNVERIFIED" and "Source changed" in source["error"]
        manifest_record = json.loads((output / "manifest.json").read_text())
        if manifest_failure:
            assert manifest_record["status"] == "UNVERIFIED"
        else:
            assert manifest_record["node"]["node_rank"] == rank
        assert json.loads((output / "transport_snapshot.json").read_text())["status"] == "UNVERIFIED"
        for phase in ("collective", "native_probe"):
            assert json.loads((output / (phase + "_exit.json")).read_text())["returncode"] == 0


@pytest.mark.parametrize("phase,returncode", [("collective", 17), ("native_probe", 23)])
def test_native_workload_failures_still_propagate(tmp_path, monkeypatch, phase, returncode):
    worker = load("worker")
    job = prepared_job(tmp_path, 4, 2, 1)
    node = worker.node_identity(job, environment(0, 1))
    coordination = tmp_path / "runtime"
    output = worker.prepare_output(job, node, coordination, time.monotonic() + 5)
    monkeypatch.setattr(worker.subprocess, "run", lambda command: SimpleNamespace(returncode=returncode))
    with pytest.raises(SystemExit) as error:
        worker.run_phase(job, node, output, coordination, phase, ["fake"], time.monotonic() + 5)
    assert error.value.code == returncode
    assert json.loads((output / (phase + "_exit.json")).read_text())["returncode"] == returncode
    assert not (output / "result.json").exists()


def test_sibling_native_failure_propagates_before_next_phase(tmp_path, monkeypatch):
    worker = load("worker")
    job = prepared_job(tmp_path, 8, 2, 2)
    node = worker.node_identity(job, environment(0, 2))
    coordination = tmp_path / "runtime"
    output = worker.prepare_output(job, node, coordination, time.monotonic() + 5)
    worker.write_json(coordination / "node_1_collective_exit.json", {"node_rank": 1, "returncode": 31})
    monkeypatch.setattr(worker.subprocess, "run", lambda command: SimpleNamespace(returncode=0))
    with pytest.raises(SystemExit) as error:
        worker.run_phase(job, node, output, coordination, "collective", ["fake"], time.monotonic() + 5)
    assert error.value.code == 31


@pytest.mark.parametrize("value", [None, [], {"unserializable": object()}])
def test_invalid_auxiliary_results_become_unverified(tmp_path, value):
    result = load("worker").advisory("invalid", tmp_path / "result.json", lambda: value)
    assert result["status"] == "UNVERIFIED"
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_auxiliary_write_error_does_not_abort(tmp_path, monkeypatch):
    worker = load("worker")

    def failed_write(path, value):
        raise OSError("Injected auxiliary output failure")

    monkeypatch.setattr(worker, "write_json", failed_write)
    result = worker.advisory("snapshot", tmp_path / "snapshot.json", lambda: {"status": "PASS"})
    assert result["status"] == "UNVERIFIED"
    assert "Injected auxiliary output failure" in result["error"]


def test_auxiliary_failure_notifies_the_platform_without_blocking(tmp_path, monkeypatch):
    worker = load("worker")
    worker.PLATFORM_LOG_FD = 19
    notices = []
    monkeypatch.setattr(worker.os, "write", lambda fd, value: notices.append((fd, value)))
    report = worker.advisory("rdma", tmp_path / "rdma.json", lambda: {"status": "UNVERIFIED"})
    assert report["status"] == "UNVERIFIED"
    assert notices[0][0] == 19 and b"STEP5_COMM_UNVERIFIED" in notices[0][1]
    def broken_pipe(*args):
        raise OSError("operator channel unavailable")
    monkeypatch.setattr(worker.os, "write", broken_pipe)
    assert worker.advisory("rdma", tmp_path / "rdma.json", lambda: {"status": "UNVERIFIED"})["status"] == "UNVERIFIED"


def test_phase_exit_evidence_write_error_does_not_block_probe(tmp_path, monkeypatch):
    worker = load("worker")
    job = prepared_job(tmp_path, 4, 2, 1)
    node = worker.node_identity(job, environment(0, 1))
    coordination = tmp_path / "runtime"
    output = worker.prepare_output(job, node, coordination, time.monotonic() + 5)
    original = worker.write_json
    calls = []

    def write(path, value):
        if path == output / "collective_exit.json":
            raise OSError("Injected cloud receipt failure")
        original(path, value)

    monkeypatch.setattr(worker, "write_json", write)
    monkeypatch.setattr(worker.subprocess, "run", lambda command: calls.append(command) or SimpleNamespace(returncode=0))
    for phase in ("collective", "native_probe"):
        worker.run_phase(job, node, output, coordination, phase, [phase], time.monotonic() + 5)
    assert calls == [["collective"], ["native_probe"]]


@pytest.mark.parametrize("changed", ["source.py", "benchmark.json", "model/config.json"])
def test_source_benchmark_and_metadata_hashes_remain_enforced(tmp_path, changed):
    job = prepared_job(tmp_path)
    for relative in ("source.py", "benchmark.json", "model/config.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen content")
    digest = hashlib.sha256(b"frozen content").hexdigest()
    job.update(source_hashes={"source.py": digest}, benchmark_sha256=digest)
    job["native_probe"]["config_sha256"] = digest
    worker = load("worker")
    worker.verify_sources(job)
    (tmp_path / changed).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        worker.verify_sources(job)


def test_output_generation_cannot_be_reused(tmp_path):
    worker = load("worker")
    job = prepared_job(tmp_path, 4, 2, 1)
    node = worker.node_identity(job, environment(0, 1))
    coordination = tmp_path / "runtime"
    worker.prepare_output(job, node, coordination, time.monotonic() + 5)
    with pytest.raises(FileExistsError):
        worker.prepare_output(job, node, coordination, time.monotonic() + 5)


def test_missing_sibling_has_explicit_deadline(tmp_path):
    with pytest.raises(TimeoutError, match="node lifecycle"):
        load("worker").wait_records([tmp_path / "missing.json"], time.monotonic() - 1)


@pytest.mark.parametrize("nodes,dp", [(1, 2), (2, 2), (4, 4)])
def test_submission_keeps_rdma_defaults_and_explicit_port(tmp_path, nodes, dp):
    job = prepared_job(tmp_path, 4 if nodes == 1 else 8, dp, nodes)
    resources = load("submit").resource_arguments(job)
    assert resources["gpu"] == 8 and resources["replica"] == nodes
    assert "custom_resources" not in resources
    assert resources["envs"]["MASTER_PORT"] == "29571"
    assert resources["envs"]["DISTRIBUTED_JOB"] == "true"
    assert "NCCL_IB_HCA" not in resources["envs"]
    assert resources["envs"]["NCCL_DEBUG_SUBSYS"] == "INIT,NET,GRAPH"


@pytest.mark.parametrize("smi_failure", [False, True])
def test_manifest_retains_gpu_identity_and_marks_smi_failure_unverified(tmp_path, monkeypatch, smi_failure):
    worker = load("worker")
    cuda = SimpleNamespace(device_count=lambda:8, get_device_name=lambda rank:"NVIDIA H800",
                           nccl=SimpleNamespace(version=lambda:(2,29,7)))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=cuda,__version__="image-torch"))
    monkeypatch.setitem(sys.modules,"vllm",SimpleNamespace(__version__="image-vllm"))
    def smi(*args, **kwargs):
        if smi_failure:
            raise OSError("injected nvidia-smi failure")
        return "GPU 0: NVIDIA H800\n"
    monkeypatch.setattr(worker.subprocess,"check_output",smi)
    report=worker.runtime_manifest(prepared_job(tmp_path),{"gpus_per_node":8})
    assert report["status"] == ("UNVERIFIED" if smi_failure else "PASS")
    assert report["devices"] == ["NVIDIA H800"]*8


@pytest.mark.parametrize("missing", ["STEPMIND_BACKEND", "BRAINPP_ACCESS_KEY", "BRAINPP_SECRET_KEY"])
def test_submission_requires_auth_before_platform_import(tmp_path, monkeypatch, missing):
    job = prepared_job(tmp_path)
    config = tmp_path / "job.json"
    config.write_text(json.dumps(job))
    monkeypatch.setattr(sys, "argv", ["submit.py", "--config", str(config)])
    for name, value in {"STEPMIND_BACKEND": "rjob", "BRAINPP_ACCESS_KEY": "fake", "BRAINPP_SECRET_KEY": "fake"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError):
        load("submit").main()
    assert not config.with_name("submission_attempt.json").exists()


def test_submit_waits_inside_spawn_without_detaching_launcher(tmp_path, monkeypatch, capsys):
    job = prepared_job(tmp_path)
    config = tmp_path / "job.json"
    config.write_text(json.dumps(job))
    monkeypatch.setattr(sys, "argv", ["submit.py", "--config", str(config)])
    for name, value in {"STEPMIND_BACKEND": "rjob", "BRAINPP_ACCESS_KEY": "fake", "BRAINPP_SECRET_KEY": "fake"}.items():
        monkeypatch.setenv(name, value)
    events = []
    resource_module = ModuleType("steptron.exp.base_exp")
    resource_module.ResourceConfig = lambda **resources: resources
    platform_module = ModuleType("steptron.utils.stepmind")

    def spawn(resources, **kwargs):
        assert Path.cwd() == tmp_path
        assert kwargs["code_mount_point"] == str(tmp_path)
        assert kwargs["charged_group"] == "codesign"
        assert resources["replica"] == 4
        events.extend(["spawn", "workload complete"])
        return SimpleNamespace(rjob=SimpleNamespace(meta=SimpleNamespace(name="offline-job")),
                               poll=lambda: events.append("terminal poll") or "succeeded")

    platform_module.spawn_tasks = spawn
    monkeypatch.setitem(sys.modules, "steptron.exp.base_exp", resource_module)
    monkeypatch.setitem(sys.modules, "steptron.utils.stepmind", platform_module)
    monkeypatch.chdir(tmp_path)
    load("submit").main()
    assert events == ["spawn", "workload complete", "terminal poll"]
    assert "FINAL_STATUS succeeded" in capsys.readouterr().out


def test_worker_wrapper_remains_native_and_bash_valid():
    wrapper = PACKAGE / "run_on_worker.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    text = wrapper.read_text()
    assert "pip install" not in text and "NCCL_IB_HCA=" not in text
    assert "exec python3 tests/integration/step5_comm/worker.py" in text

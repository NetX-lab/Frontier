"""Execute the reviewed worker with CPU-only process and platform stand-ins."""

import contextlib
import hashlib
import json
from pathlib import Path
import tempfile
import types
from unittest.mock import patch

import pytest

from tests.comparison.calibration import native_pd_worker


def exercise(worker, temporary, case):
    directory = Path(tempfile.mkdtemp(prefix=case + "_", dir=temporary))
    root = directory / "output"
    model = directory / "model"
    scratch = directory / "scratch"
    model.mkdir()
    scratch.mkdir()
    (model / "config.json").write_text("{}")
    own_host = "host1" if case == "canonical_clean_natural_headless_exit" else "host0"
    for index in range(4):
        if f"host{index}" == own_host:
            continue
        node = root / "nodes" / f"host{index}"
        node.mkdir(parents=True)
        (node / "ready.json").write_text(json.dumps({
            "host": f"host{index}", "ip": f"10.0.0.{index + 1}"}))
        (node / "drained.json").write_text("{}")
    (root / "nodes/host2/server_ready.json").write_text("{}")
    events = []
    replay_commands = []
    server_environments = []
    injected = False

    def write_json(path, value):
        nonlocal injected
        if (case == "report_write_failure" and not injected
                and path.name == "result.json" and value.get("stage") == "await_replay"):
            injected = True
            events.append("auxiliary_report_error_after_native_health")
            raise OSError("Injected auxiliary result writer failure")
        path.write_text(json.dumps(value))

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    class Process:
        def __init__(self, kind):
            self.kind = kind
            self.returncode = None

        def poll(self):
            if (self.kind == "server" and case == "canonical_clean_natural_headless_exit"
                    and (root / "nodes/host1/drained.json").exists()):
                self.returncode = 0
            if self.kind == "replay":
                self.returncode = 4 if case == "canonical_clean_admission_failure" else 0
                if "replay_completed" not in events:
                    events.append("replay_completed")
                if case == "native_exit_at_replay_completion":
                    server.returncode = 23
                    events.append("native_exit_23_before_requested_stop")
            return self.returncode

    server = Process("server")
    replay = Process("replay")

    def popen(command, **kwargs):
        kind = "server" if command[0] == "fake-native-server" else "replay"
        if kind == "replay":
            replay_commands.append(command)
        else:
            server_environments.append(dict(kwargs['env']))
        events.append(kind + "_launch")
        return server if kind == "server" else replay

    def stop_server(process, timeout):
        events.append(process.kind + "_stop_called")
        if process.poll() is not None:
            return {"signal": None, "returncode": process.returncode}
        process.returncode = (23 if case == "native_shutdown_error"
                              and process.kind == "server" else 0)
        return {"signal": "SIGINT", "returncode": process.returncode}

    inventory = types.ModuleType("probe_runtime_inventory")
    inventory.primary_ipv4 = lambda: "10.0.0.2" if own_host == "host1" else "10.0.0.1"
    inventory.read_text = lambda path: ""
    inventory.sha256 = sha256
    inventory.weight_candidates = lambda *args: {}
    inventory.write_json = write_json
    nixl = types.ModuleType("probe_nixl_read")
    nixl.SYSFS_INFINIBAND = directory / "no_rdma_files"
    nixl.gid_table = lambda path: {}
    nixl.roce_v2_ipv4_ports = lambda value: ["mlx5_0:1"]
    pd = types.ModuleType("pd_replay")
    pd.instance_devices = lambda inherited, *, engine, role: {role: "0,1,2,3,4,5,6,7"}
    pd.instance_env = lambda *args, **kwargs: ({}, {})
    pd.kv_transfer_config = lambda *args: {}
    native = types.ModuleType("vllm_replay")
    native.server_command = lambda *args: ["fake-native-server"]
    native.stop_server = stop_server
    config = {
        "script": str(directory / "worker.py"), "harness_dir": str(directory),
        "output_dir": str(root), "cache_environment": {}, "model_dir": str(model),
        "model_hashes": {"config.json": sha256(model / "config.json")},
        "installed_package": str(directory / "vllm"), "expected_source_hashes": {},
        "replica": 4, "gpu": 8, "data_parallel_size_local": 1,
        "engine": {"data_parallel_size": 2, "tensor_parallel_size": 8, "kv_transfer": {}},
        "dp_rpc_ports": {"prefill": 30100, "decode": 30200},
        "http_ports": {"prefill": 30300, "decode": 30400},
        "proxy_port": 30500, "serve_observers": {"prefill": {}, "decode": {}},
        "peer_wait_seconds": 10, "startup_timeout_s": 10, "replay_timeout_s": 10,
        "drain_seconds": 0, "stop_timeout_s": 10, "weight_suffixes": [".bin"],
    }
    if case.startswith("canonical_clean"):
        config.update(mode="clean", idle_timeout_s=180, idle_poll_s=1)
    if case in ("canonical_clean_standard_layout", "canonical_clean_manifest_staging_failure"):
        config.update(case_dir=str(directory / "local-case"), run_tag="cpu-fixture")
    if case == "canonical_clean_natural_headless_exit":
        (root / "replay_finished.json").write_text('{"returncode":0}')
    if case.startswith('canonical_clean_serving'):
        config.update(case_dir=str(directory / 'local-case'), run_tag='cpu-fixture')
        config['engine']['worker_extension_cls'] = 'native_full_structure.FullModelInventory'
        (directory / 'native_full_structure.py').write_text('CPU fixture module, not executed\n')
        if case in ('canonical_clean_serving', 'canonical_clean_serving_invalid_generation'):
            metadata = Path(config['case_dir']) / 'runs/groundtruth_clean/cpu-fixture/run_manifest.json'
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps({'run_generation': 'cpu-serving-generation'
                if case == 'canonical_clean_serving' else None}))
    original_relative = Path.is_relative_to

    def is_relative_to(path, other):
        if path == root and other == "/mnt/codesign-exp/ycfeng":
            return True
        return original_relative(path, other)

    class Health:
        status = 200

        def __enter__(self):
            events.append("native_health_200")
            return self

        def __exit__(self, *args):
            return False

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict("os.environ"))
        stack.enter_context(patch("sys.path", list(__import__("sys").path)))
        stack.enter_context(patch.dict("sys.modules", {
            "probe_runtime_inventory": inventory, "probe_nixl_read": nixl,
            "pd_replay": pd, "vllm_replay": native}))
        stack.enter_context(patch.object(Path, "is_relative_to", is_relative_to))
        stack.enter_context(patch.object(worker.socket, "gethostname",
            return_value=own_host))
        stack.enter_context(patch.object(worker.os, "dup2"))
        stack.enter_context(patch.object(worker.tempfile, "mkdtemp", return_value=str(scratch)))
        stack.enter_context(patch.object(worker.subprocess, "Popen", side_effect=popen))
        stack.enter_context(patch.object(worker.subprocess, "check_output", return_value="fake H800"))
        stack.enter_context(patch.object(worker.importlib.metadata, "version", return_value="stub"))
        stack.enter_context(patch("urllib.request.urlopen", return_value=Health()))
        if case == "canonical_clean_standard_layout" or case.startswith('canonical_clean_serving'):
            stack.enter_context(patch.object(worker, "stage_run_manifest",
                side_effect=lambda *args: events.append("prelaunch_manifest_staged")))
        if case == "canonical_clean_natural_headless_exit":
            def publish_head_stop(_):
                events.append("head_native_stopped")
                (root / "nodes/host0/native_stopped.json").write_text('{"returncode":0,"signal":"SIGINT"}')
            stack.enter_context(patch.object(worker.time, "sleep", side_effect=publish_head_stop))
        code = worker.run(config)
    result = json.loads((root / "nodes" / own_host / "result.json").read_text())
    return {
        "case": case, "worker_returncode": code, "events": events,
        "native_server_returncode": result.get("server_stop", {}).get("returncode"),
        "native_server_stop_signal": result.get("server_stop", {}).get("signal"),
        "failed_sentinel": (root / "nodes" / own_host / "failed.json").exists(),
        "observation_errors": result["observation_errors"],
        "replay_commands": replay_commands,
        "shutdown_order": result.get("shutdown_order"),
        "server_environments": server_environments,
        "serving_instrumentation": result.get('serving_instrumentation'),
    }


@pytest.mark.parametrize("case", ["control", "report_write_failure",
                                 "native_exit_at_replay_completion", "native_shutdown_error"])
def test_worker_separates_auxiliary_report_faults_from_native_failure(tmp_path, case):
    result = exercise(native_pd_worker, tmp_path, case)
    assert "native_health_200" in result["events"]
    assert "replay_launch" in result["events"]
    assert "server_stop_called" in result["events"]
    if case.startswith("native_"):
        assert result["worker_returncode"] != 0
        assert result["native_server_returncode"] == 23
    else:
        assert result["worker_returncode"] == 0
        assert not result["failed_sentinel"]
    if case == "report_write_failure":
        assert result["observation_errors"]
        assert "Injected auxiliary result writer failure" in result["observation_errors"][0]
    if case == "native_exit_at_replay_completion":
        assert result["failed_sentinel"]


def test_worker_forwards_canonical_clean_mode_and_idle_configuration(tmp_path):
    result = exercise(native_pd_worker, tmp_path, "canonical_clean")
    assert result["worker_returncode"] == 0 and not result["failed_sentinel"]
    command = result["replay_commands"][0]
    assert command[command.index("--mode") + 1] == "clean"
    assert command[command.index("--idle-timeout-s") + 1] == "180"
    assert command[command.index("--idle-poll-s") + 1] == "1"


def test_worker_preserves_native_completion_when_client_admission_is_unverified(tmp_path):
    result = exercise(native_pd_worker, tmp_path, "canonical_clean_admission_failure")
    assert result["worker_returncode"] == 4
    assert result["native_server_returncode"] == 0
    assert not result["failed_sentinel"]
    assert result["events"].index("replay_completed") < result["events"].index("server_stop_called")


def test_worker_uses_case_run_layout_and_retains_ordered_shutdown(tmp_path):
    result = exercise(native_pd_worker, tmp_path, "canonical_clean_standard_layout")
    assert result["worker_returncode"] == 0 and not result["failed_sentinel"]
    assert result["events"].index("prelaunch_manifest_staged") < result["events"].index("server_launch")
    command = result["replay_commands"][0]
    assert Path(command[command.index("--output-dir") + 1]).name == "run"
    assert result["shutdown_order"] == {
        "drained_before_stop": True,
        "all_nodes_drained_before_head_stop": ["host0", "host1", "host2", "host3"],
        "native_returned_before_requested_stop": False,
    }


def test_manifest_copy_failure_is_soft_and_native_work_still_completes(tmp_path):
    result = exercise(native_pd_worker, tmp_path, "canonical_clean_manifest_staging_failure")
    assert result["observation_errors"]
    assert "FileNotFoundError" in result["observation_errors"][0]
    assert result["worker_returncode"] == result["native_server_returncode"] == 0
    assert not result["failed_sentinel"]
    assert result["events"].index("replay_completed") < result["events"].index("server_stop_called")


def test_headless_natural_completion_still_waits_for_head_stop_evidence(tmp_path):
    result = exercise(native_pd_worker, tmp_path, "canonical_clean_natural_headless_exit")
    assert result["worker_returncode"] == result["native_server_returncode"] == 0
    assert result["native_server_stop_signal"] is None
    assert not result["observation_errors"]
    assert result["shutdown_order"]["native_returned_before_requested_stop"] is True
    assert result["shutdown_order"]["head_stopped_before_headless_stop"] is True
    assert result["events"].index("head_native_stopped") < result["events"].index("server_stop_called")


def test_serving_extension_receives_actual_generation_role_and_node_directory(tmp_path):
    result = exercise(native_pd_worker, tmp_path, 'canonical_clean_serving')
    assert result['worker_returncode'] == 0 and not result['failed_sentinel']
    assert not result['observation_errors']
    env = result['server_environments'][0]
    assert env['FRONTIER_NATIVE_RUN_GENERATION'] == 'cpu-serving-generation'
    assert env['FRONTIER_NATIVE_ROLE'] == 'prefill'
    assert Path(env['FRONTIER_NATIVE_SERVING_EVIDENCE_DIR']).parts[-2:] == ('nodes', 'host0')
    assert result['serving_instrumentation']['native_python_callables_wrapped'] is True
    assert result['serving_instrumentation']['overhead_measured'] is False


@pytest.mark.parametrize('case', ['canonical_clean_serving_missing_provenance',
                                'canonical_clean_serving_invalid_generation'])
def test_missing_serving_provenance_is_soft_and_does_not_enable_stale_emission(tmp_path, monkeypatch, case):
    monkeypatch.setenv('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR', '/unrelated/stale')
    result = exercise(native_pd_worker, tmp_path, case)
    assert result['observation_errors']
    assert result['worker_returncode'] == result['native_server_returncode'] == 0
    assert not result['failed_sentinel']
    assert 'FRONTIER_NATIVE_SERVING_EVIDENCE_DIR' not in result['server_environments'][0]

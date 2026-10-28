"""CPU submission-lock, path-hash and runtime-gate checks in isolated files."""

import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

LAUNCH = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("g5h_runtime_check", LAUNCH / "runtime_check.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def git(workspace, *args):
    # These commands only address this test's repository; never export GIT_DIR
    # or GIT_WORK_TREE (task issues.md I-001).
    return subprocess.run(["git", "-C", str(workspace), *args], check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()


@pytest.fixture
def locked(tmp_path, monkeypatch):
    assert "GIT_DIR" not in __import__("os").environ
    assert "GIT_WORK_TREE" not in __import__("os").environ
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    git(workspace, "init")
    launch = workspace / runtime.TASK_ROOT / "h3o_operator_costs/g5h/launch"
    launch.mkdir(parents=True)
    for name in runtime.BENCHMARK_MODULES:
        path = workspace / "frontier/profiling/collectives" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# tracked benchmark {name}\n")
    (launch / "runtime_check.py").write_text((LAUNCH / "runtime_check.py").read_text())
    entry = workspace / runtime.K5_NODE_ENTRY
    entry.parent.mkdir(parents=True)
    entry.write_text("print('CPU_NODE_ENTRY_REACHED')\n")
    native, installed = tmp_path / "checkout/vllm", tmp_path / "image/vllm"
    for name in runtime.VLLM_SOURCES:
        for root in (native, installed):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# source {name}\n")
    nccl = tmp_path / "image/libnccl.so.2"
    nccl.write_bytes(b"NCCL CPU fixture")
    monkeypatch.setattr(runtime, "NCCL_LIBRARY", str(nccl))
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"installed_package": str(installed),
                                  "binary_sha256": {str(nccl): runtime.sha256(nccl)}}))
    config = {"workspace": str(workspace), "native_checkout": str(native),
              "p2_overlay": str(overlay), "output_dir": str(tmp_path / "output"),
              "runtime_check_gate": "hard", "timeout_seconds": 20,
              "cuda_compat_path": "/cpu-fixture/compat"}
    config_path = launch / "stage.json"
    config_path.write_text(json.dumps(config))
    git(workspace, "add", ".")
    git(workspace, "-c", "user.name=G5h CPU Test", "-c", "user.email=g5h-test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "CPU lock fixture")
    lock = runtime.build_lock(config, config_path)
    return config, config_path, lock


def test_lock_contains_every_expected_path_and_overlay_hash(locked):
    config, path, lock = locked
    workspace = Path(config["workspace"])
    assert lock["workspace_head"] == git(workspace, "rev-parse", "HEAD")
    expected = {str(workspace / "frontier/profiling/collectives" / name)
                for name in runtime.BENCHMARK_MODULES}
    expected.update([str(path.parent / "runtime_check.py"), str(workspace / runtime.K5_NODE_ENTRY),
                     str(path)])
    assert set(lock["benchmark_sources"]) == expected
    assert lock["benchmark_sources"] == {name: runtime.sha256(name) for name in expected}
    overlay = json.loads(Path(config["p2_overlay"]).read_text())
    assert lock["nccl"] == overlay["binary_sha256"]
    assert set(lock["vllm_sources"]) == {
        str(Path(overlay["installed_package"]) / name) for name in runtime.VLLM_SOURCES}
    assert list(lock["vllm_sources"].values()) == [
        runtime.sha256(Path(config["native_checkout"]) / name) for name in runtime.VLLM_SOURCES]
    # Unrelated workspace changes do not prevent locking the approved sources.
    (workspace / "unrelated.txt").write_text("untracked unrelated change")
    assert runtime.build_lock(config, path) == lock


@pytest.mark.parametrize("staged", [False, True])
def test_build_lock_refuses_uncommitted_benchmark_source(locked, staged):
    config, path, lock = locked
    source = Path(next(iter(lock["benchmark_sources"])))
    source.write_text(source.read_text() + "# dirty source\n")
    if staged:
        git(Path(config["workspace"]), "add", str(source))
    with pytest.raises(RuntimeError, match="Uncommitted benchmark sources") as error:
        runtime.build_lock(config, path)
    assert source.name in str(error.value)


def test_build_lock_refuses_untracked_stage_config(locked):
    config, path, _ = locked
    new_path = path.parent / "new_stage.json"
    new_path.write_text(path.read_text())
    with pytest.raises(RuntimeError, match="Uncommitted benchmark sources"):
        runtime.build_lock(config, new_path)


def _report(config):
    reports = list((Path(config["output_dir"]) / "runtime_checks").glob("*.json"))
    assert len(reports) == 1
    return json.loads(reports[0].read_text())


@pytest.mark.parametrize("gate", ["hard", "soft"])
def test_all_hashes_match_and_success_preserves_k5_node_directory(locked, gate):
    config, _, lock = locked
    config["runtime_check_gate"] = gate
    assert runtime.check_runtime(config, lock) == 0
    report = _report(config)
    assert report["workspace_head"] == lock["workspace_head"]
    assert len(report["checks"]) == 12
    assert all(row["pass"] and row["actual"] == row["expected"] for row in report["checks"])
    assert all(row["gate"] == ("soft" if row["name"] == "vllm_source" else gate)
               for row in report["checks"])
    assert not (Path(config["output_dir"]) / "nodes").exists()


@pytest.mark.parametrize("section, index", [
    ("nccl", 0), *[("benchmark_sources", i) for i in range(6)],
    *[("vllm_sources", i) for i in range(5)],
])
@pytest.mark.parametrize("gate", ["hard", "soft"])
def test_each_hash_mismatch_obeys_its_gate(locked, section, index, gate):
    config, _, lock = locked
    config["runtime_check_gate"] = gate
    path = Path(list(lock[section])[index])
    path.write_bytes(path.read_bytes() + b"mismatch")
    hard = gate == "hard" and section != "vllm_sources"
    assert runtime.check_runtime(config, lock) == int(hard)
    report = _report(config)
    failed = [row for row in report["checks"] if not row["pass"]]
    assert len(failed) == 1
    assert failed[0]["path"] == str(path)
    assert failed[0]["actual"] == runtime.sha256(path)
    assert failed[0]["expected"] == lock[section][str(path)]
    assert failed[0]["gate"] == ("soft" if section == "vllm_sources" else gate)
    marker = Path(config["output_dir"]) / "nodes" / report["host"] / "failed.json"
    assert marker.exists() == hard
    if hard:
        assert json.loads(marker.read_text()) == report


@pytest.mark.parametrize("section", ["nccl", "benchmark_sources", "vllm_sources"])
def test_missing_file_is_recorded_as_a_failed_comparison(locked, section):
    config, _, lock = locked
    path, expected = next(iter(lock[section].items()))
    missing = path + ".missing"
    lock[section] = {missing: expected}
    assert runtime.check_runtime(config, lock) == int(section != "vllm_sources")
    failed = [row for row in _report(config)["checks"] if not row["pass"]]
    assert len(failed) == 1 and failed[0]["actual"] is None
    assert failed[0]["path"] == missing and "error" in failed[0]


@pytest.mark.parametrize("section, gate, exit_code", [
    ("nccl", "hard", 1), ("benchmark_sources", "hard", 1),
    ("nccl", "soft", 0), ("benchmark_sources", "soft", 0),
    ("vllm_sources", "hard", 0),
])
def test_cli_exit_code_and_peer_failure_marker(locked, section, gate, exit_code):
    config, config_path, lock = locked
    config["runtime_check_gate"] = gate
    config_path.write_text(json.dumps(config))
    # Updating the gate config is intentional; match its hash before injecting
    # the independent mismatch this CLI case is exercising.
    lock["benchmark_sources"][str(config_path)] = runtime.sha256(config_path)
    target = next(iter(lock[section]))
    lock[section][target] = "0" * 64
    lock_path = config_path.parent / "lock.json"
    lock_path.write_text(json.dumps(lock))
    result = subprocess.run([sys.executable, str(LAUNCH / "runtime_check.py"),
                             "--config", str(config_path), "--lock", str(lock_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == exit_code, result.stderr
    assert f"exit {exit_code}" in result.stdout
    assert _report(config)["returncode"] == exit_code
    assert bool(list((Path(config["output_dir"]) / "nodes").glob("*/failed.json"))) == bool(exit_code)


@pytest.mark.parametrize("fail", [False, True])
def test_worker_command_timeout_covers_check_and_k5_entry(locked, fail):
    config, config_path, lock = locked
    lock_path = config_path.parent / "lock.json"
    if fail:
        target = next(iter(lock["nccl"]))
        lock["nccl"][target] = "0" * 64
    lock_path.write_text(json.dumps(lock))
    command = runtime.worker_command(config, config_path, lock_path)
    assert command.count("timeout ") == 1
    assert " && " in command
    assert str(Path(config["workspace"]) / runtime.K5_NODE_ENTRY) in command
    assert "UCX_NET_DEVICES" not in command
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=30)
    assert result.returncode == int(fail), result.stderr
    assert ("CPU_NODE_ENTRY_REACHED" in result.stdout) == (not fail)


def test_real_stage_worker_command_has_one_timeout_and_quoted_chain():
    path = LAUNCH / "stage_g5h16_tp8_dp2_ep16.json"
    config = json.loads(path.read_text())
    lock_path = LAUNCH / "receipts" / f"{config['job_tag']}_lock.json"
    command = runtime.worker_command(config, path, lock_path)
    exports, timeout = command.rsplit("; ", 1)
    assert exports.startswith("export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat")
    assert "export PYTHONPATH=" + config["workspace"] in exports
    parts = shlex.split(timeout)
    assert parts[:4] == ["timeout", "2700", "bash", "-c"]
    check, entry = parts[4].split(" && ")
    assert shlex.split(check) == ["python3", str(LAUNCH / "runtime_check.py"),
                                  "--config", str(path), "--lock", str(lock_path)]
    assert shlex.split(entry) == ["python3", "-u", str(Path(config["workspace"]) / runtime.K5_NODE_ENTRY),
                                  "--config", str(path)]

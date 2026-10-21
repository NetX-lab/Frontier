"""First-case H200 packet, qualified driver recipe and preserved timing evidence."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from frontier.profiling.collectives.vllm_forward_collectives import (
    DTYPE_BYTES, ModelShape, ServingParallelism, forward_collectives, group_members, scenarios_from_grid,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tests/integration/step5_comm"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definition():
    return json.loads((PACKAGE / "h200_first_prefill_run.json").read_text())


def test_first_prefill_is_full_query_not_a_decode_batch_guess():
    spec = definition()
    parallel = ServingParallelism.from_engine(spec["engine"])
    (scenario,) = scenarios_from_grid(parallel, spec["grid"])
    model = ModelShape.from_hf_config(json.loads((ROOT / load("prepare").MODEL_CONFIG).read_text()))
    calls = {call.name: call for call in forward_collectives(model, parallel, scenario)}
    assert parallel.world_size == 32
    assert scenario.tokens_per_dp_rank == (6144, 1, 1, 1)
    assert scenario.useful_tokens_per_dp_rank == (6144, 0, 0, 0)
    assert not scenario.cuda_graph
    assert len(calls) == 10
    for name, count, row_bytes in (
            ("moe_dispatch_all_gatherv_fp8", 85, 4288),
            ("moe_dispatch_all_gatherv_bf16", 3, 8256),
            ("moe_combine_reduce_scatterv", 88, 8192)):
        call = calls[name]
        assert call.calls_per_forward == count and call.group == "dp"
        assert sum(DTYPE_BYTES[d] * n for d, n in zip(call.dtypes, call.columns)) == row_bytes
        assert call.shard_rows == (6144,) * 8 + (1,) * 24
    assert {tuple(group_members(parallel, "dp", rail)) for rail in range(8)} == {
        (rail, rail + 8, rail + 16, rail + 24) for rail in range(8)}
    assert not any(call.group == "ep" or "all_to_all" in call.op for call in calls.values())
    assert spec["native_probe"]["tokens_per_dp_rank"] == [6144, 1, 1, 1]
    assert spec["coverage"]["formal_collection_ready"] is False
    assert spec["coverage"]["decode"].startswith("UNKNOWN")
    assert spec["canonical_workload"].endswith("canonical_fixed_6144_1024_req128_qps1")


def test_h200_reuses_s1_preparation_and_binds_common_bytes(tmp_path, monkeypatch):
    prepare = load("prepare")
    spec = definition()
    paths = [prepare.MODEL_CONFIG, spec["canonical_workload"] + "/manifest.json"]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    def git_output(command, **kwargs):
        if command[1] == "ls-files":
            assert spec["canonical_workload"] in command
        return {"ls-files": "\n".join(paths), "status": "", "rev-parse": "frozen\n"}[command[1]]
    monkeypatch.setattr(prepare.subprocess, "check_output", git_output)
    job = prepare.prepare(spec, tmp_path / "stage", "h200_offline", repository=tmp_path)
    assert job["expected"] == dict(scenarios=1, cases=10, measurements=30,
                                   kernel_rows=10, wall_minutes=15, gpu_hours=8.0)
    assert job["coverage"] == spec["coverage"]
    assert job["source_hashes"][paths[1]] == spec["coverage"]["shared_manifest_sha256"]
    resources = load("submit").resource_arguments(job)
    assert resources["positive_tags"] == ["H200"]
    assert resources["gpu"] * resources["replica"] == 32
    assert resources["preemptible"] is False
    assert "custom_resources" not in resources
    assert job["resources"]["charged_group"] == "step_main"
    assert job["engine"] == json.loads((PACKAGE / "s1_run.json").read_text())["engine"]
    assert job["coverage"]["queue_quality"].find("<1%") >= 0


@pytest.mark.parametrize("tags,group", [(["H200"], "codesign"), (["H800"], "step_main"),
                                       (["H100"], "step_main"), ([], "step_main")])
def test_preparation_rejects_wrong_hardware_group_pair(tmp_path, monkeypatch, tags, group):
    prepare = load("prepare")
    spec = definition()
    spec["resources"].update(positive_tags=tags, charged_group=group)
    monkeypatch.setattr(prepare.subprocess, "check_output", lambda *a, **k: "")
    with pytest.raises(ValueError, match="hardware/charged-group"):
        prepare.prepare(spec, tmp_path / "stage", "invalid", repository=tmp_path)


@pytest.mark.parametrize("driver,producer_code,expected,code", [
    ("570.124.06", 0, "/usr/local/cuda-13.0/compat", 0),
    ("580.82.07", 0, "/usr/local/nvidia/lib64", 0),
    ("590.1", 0, "/usr/local/nvidia/lib64", 0),
    ("550.1", 0, None, 1),
    (None, 9, None, 9),
    ("570.124.06", 9, None, 9),
    ("580.82.07", 9, None, 9),
])
def test_wrapper_selects_qualified_driver_without_platform_calls(tmp_path, driver, producer_code, expected, code):
    # Only fake executables run. Ignore the image-only libcuda file existence test.
    env_script = tmp_path / "bash_env"
    env_script.write_text('test() { if [ "$1" = "-e" ]; then return 0; fi; builtin test "$@"; }\n')
    producer = "#!/bin/bash\n"
    if driver is not None:
        # Close the old head reader before the remaining seven GPU rows arrive.
        producer += (f"printf '%s\\n' '{driver}'\nsleep 0.05\n"
                     f"for i in {{2..8}}; do printf '%s\\n' '{driver}'; done\n")
    if producer_code:
        producer += "echo 'nvidia-smi fixture failed' >&2\n"
    producer += f"exit {producer_code}\n"
    downstream = ('#!/bin/bash\necho "$VLLM_ENABLE_CUDA_COMPATIBILITY|$LD_LIBRARY_PATH"\n'
                  'echo "STEP5_COMM_WORKER_LOG $*"\nenv | sed -n "/^NCCL_/p"\n')
    for name, text in {"nvidia-smi": producer, "python3": downstream}.items():
        path = tmp_path / name
        path.write_text(text)
        path.chmod(0o755)
    nccl = {name: value for name, value in os.environ.items() if name.startswith("NCCL_")}
    nccl.update(NCCL_IB_HCA="=mlx5_bond100,mlx5_bond101", NCCL_SOCKET_IFNAME="bond0",
                NCCL_IB_GID_INDEX="3", NCCL_NET_PLUGIN="stepfun")
    env = {**os.environ, "PATH": str(tmp_path) + ":/usr/bin:/bin", "BASH_ENV": str(env_script),
           "LD_LIBRARY_PATH": "/existing", **nccl}
    result = subprocess.run(["bash", str(PACKAGE / "run_on_worker.sh"), "unused.json"],
                            env=env, text=True, capture_output=True, timeout=5)
    assert result.returncode != 141, result.stderr
    assert result.returncode == code, result.stderr
    if expected:
        enabled = "1" if driver.startswith("570.") else "0"
        lines = result.stdout.splitlines()
        assert lines[0] == f"{enabled}|{expected}:/existing"
        assert lines[1] == "STEP5_COMM_WORKER_LOG tests/integration/step5_comm/worker.py --config unused.json"
        assert dict(line.split("=", 1) for line in lines[2:]) == nccl
    else:
        assert "STEP5_COMM_WORKER_LOG" not in result.stdout
        message = "nvidia-smi fixture failed" if producer_code else "Unsupported observed driver branch"
        assert message in result.stderr


@pytest.mark.parametrize("actual,expected", [("NVIDIA H200", "PASS"), ("NVIDIA H800", "UNVERIFIED")])
def test_runtime_hardware_identity_is_advisory(monkeypatch, actual, expected):
    cuda = SimpleNamespace(device_count=lambda: 8, get_device_name=lambda i: actual,
                           nccl=SimpleNamespace(version=lambda: (2, 29, 7)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, __version__="fake"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="fake"))
    worker = load("worker")
    monkeypatch.setattr(worker.subprocess, "check_output", lambda *a, **k: "fake nvidia-smi")
    report = worker.runtime_manifest(definition(), {"gpus_per_node": 8})
    assert report["status"] == expected
    assert report["host_driver_versions"] == ["fake nvidia-smi"]
    assert "VLLM_ENABLE_CUDA_COMPATIBILITY" in report["cuda_driver_environment"]


def test_enqueue_evidence_does_not_claim_a_timing_cause_or_remove_samples(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE))
    record = dict(op="all_reduce", **{"pass": 0}, scenario="s", call="c", participating_ranks=[0],
                  per_rank_call_us=[[1.0, 2.0, 3.0]], per_rank_launch_call_us=[[2.0, 3.0, 4.0]],
                  per_rank_queue_held=[[False, False, True]], per_rank_queue_sleep_us=[10.0],
                  per_rank_enqueue_us=[[20.0, 2.0, 15.0]], per_rank_launch_enqueue_us=[[3.0, 4.0, 5.0]])
    result = load("queue_diagnostics").diagnose([record])
    assert result["counts"]["unheld_enqueue_over_sleep"] == 1
    assert result["counts"]["unheld_enqueue_within_sleep"] == 1
    assert result["counts"]["held_enqueue_over_sleep"] == 1
    assert len(result["unheld_rank_repeats"]) == 2
    assert result["queue_failures"][0]["critical_path_repeats_us"] == [1.0, 2.0, 3.0]
    assert result["queue_threshold"] == 0.01
    assert result["timing_changed"] is result["cost_data_qualified"] is False
    assert result["cause"].startswith("UNPROVEN")

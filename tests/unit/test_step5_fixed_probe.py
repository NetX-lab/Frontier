"""Offline reached-shape, immutable-workload and isolated native-policy checks."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.step5 import job, main
from frontier.profiling.step5.inputs import read_profile_inputs


def probe_cells():
    return read_profile_inputs(job.REPO_ROOT / job.PROBE_INPUTS, cudagraph_capture_sizes=(1, 2))


def test_reached_shapes_are_not_a_cartesian_grid_or_qps_batch_guess():
    cells = probe_cells()
    assert len(cells) == 18
    assert {cell.global_layer_id for cell in cells} == {0, 3}
    prefill = [cell for cell in cells if cell.phase == "prefill"]
    assert {(cell.query_lens, cell.context_lens) for cell in prefill} == {
        ((6144,), (0,)), ((16,), (6128,))}
    assert {cell.graph_mode for cell in prefill} == {"none"}
    decode = [cell for cell in cells if cell.phase == "decode"]
    assert {cell.batch_size for cell in decode} == {1, 2}
    assert {cell.context_lens[0] for cell in decode} == {6144, 6656, 7160, 7167}
    assert {cell.graph_mode for cell in decode} == {"full"}
    config = ModelConfig.from_model_name("stepfun-ai/Step-5-Preview")
    for cell in cells:
        workload = cell.workload(config, block_size=16, max_model_len=7168)
        assert max(workload.sequence_lengths) <= 7168
        if cell.global_layer_id == 3:
            assert all(context % config.sparse_config["region_block_size"] == 0
                       for context in cell.context_lens)
    assert max(max(cell.sequence_lens) for cell in cells) == 7168


@pytest.mark.parametrize("records", [[], [{"global_layer_id": 0, "query_lens": [1],
                                         "context_lens": [6144], "phase": "prefill"}]])
def test_explicit_inputs_reject_empty_or_false_prefill(tmp_path, records):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        read_profile_inputs(path, cudagraph_capture_sizes=(1, 2))


def test_explicit_inputs_reject_duplicate_cells(tmp_path):
    path = tmp_path / "inputs.json"
    record = json.loads((job.REPO_ROOT / job.PROBE_INPUTS).read_text())[0]
    path.write_text(json.dumps([record, record]))
    with pytest.raises(ValueError, match="distinct"):
        read_profile_inputs(path, cudagraph_capture_sizes=(1, 2))


def test_probe_manifest_uses_the_existing_producers_and_complete_native_policy(tmp_path, capsys):
    stage = tmp_path / "not_created"
    manifest = job.describe_probe(stage)
    assert not stage.exists()
    assert manifest["purpose"].startswith("SMALL_OPERATOR_PROBE_ONLY")
    assert manifest["gpu_tag"] == "H200" and manifest["charged_group"] == "step_main"
    assert (manifest["gpus"], manifest["cpus"], manifest["mem_gb"]) == (1, 8, 32)
    assert manifest["expected_attention_event_rows"] == 18
    assert manifest["expected_attention_gpu_active_rows"] == 4
    assert manifest["expected_linear_rows"] == 16
    assert manifest["linear_plan"]["num_tokens"] == [1, 2, 16, 6144]
    assert manifest["workload"]["sha256"]["manifest.json"] == job.WORKLOAD_MANIFEST_SHA256
    assert manifest["workload"]["sha256"]["SHA256SUMS"] == job.WORKLOAD_SUMS_SHA256
    assert manifest["workload"]["upstream_commit"] == "68f891b"
    assert not manifest["admitted"] and len(manifest["blocks"]) == 3
    policy = manifest["native_probe_policy"]
    assert policy["max_num_seqs"] == 2
    assert policy["max_num_batched_tokens"] >= policy["max_model_len"] == 7168
    assert policy["cudagraph_capture_sizes"] == [1, 2]
    assert not policy["enable_chunked_prefill"] and not policy["enable_prefix_caching"]
    worker = manifest["worker_command"]
    assert worker.count("-m frontier.profiling.step5.main") == 1
    assert worker.count("-m frontier.profiling.linear_op.main") == 2
    assert worker.count("--max_tokens 6144") == 2
    assert "--max-time-cv 0.05" in worker and "--profile-iterations 10" in worker
    assert "--device h200" in worker and "tee" in worker and "set -euo pipefail" in worker
    assert "--gpu-tag H200 --charged-group step_main" in manifest["submission_command"]
    assert shlex.split(manifest["submission_command"])[-1] == worker
    subprocess.run(["bash", "-n"], input=worker, text=True, check=True)
    main.main([*manifest["attention_arguments"], "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["rows"] == 18 and result["gpu_active_rows"] == 4
    assert result["native_max_num_seqs"] == 2
    assert result["native_max_num_batched_tokens"] == 14336
    assert not result["enable_chunked_prefill"]


@pytest.mark.parametrize("gpus", [0, 2, 8])
def test_probe_rejects_distributed_allocations(tmp_path, gpus):
    with pytest.raises(ValueError, match="isolated"):
        job.describe_probe(tmp_path, gpus=gpus)


def test_workload_mismatch_rejected_before_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "WORKLOAD_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="immutable D18"):
        job.describe_probe(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("driver,compat", [("570.124.06", "1"), ("580.82.07", "0"), ("590.1", "0")])
def test_worker_checks_driver_before_python_without_loading_old_compat_on_new_hosts(tmp_path, driver, compat):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, content in {
        "nvidia-smi": f'#!/bin/bash\necho "{driver}"\n',
        "python3": '#!/bin/bash\nif [ "$1" = "-" ]; then exit 0; fi\n'
                   'echo "COMPAT=$VLLM_ENABLE_CUDA_COMPATIBILITY LD=${LD_LIBRARY_PATH-}"\n',
    }.items():
        path = binaries / name
        path.write_text(content)
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
           "TMPDIR": str(tmp_path), "LD_LIBRARY_PATH": "/fixture/native/lib"}
    result = subprocess.run(["bash", str(job.REPO_ROOT / "tests/integration/step5_profiling/run_on_worker.sh"),
                             "-c", "CPU_STUB_ONLY"], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.index("WORKER_HOST_DRIVER") < result.stdout.index("COMPAT=")
    assert f"COMPAT={compat}" in result.stdout
    assert ("/usr/local/cuda-13.0/compat" in result.stdout) == (compat == "1")


def test_stage_copies_exact_workload_and_hashes_every_payload(tmp_path):
    dependencies = tmp_path / "site"
    dependencies.mkdir()
    (dependencies / "dependency_fixture.py").write_text("# CPU-only package fixture\n")
    stage = tmp_path / "stage"
    manifest = job.stage_probe(stage, dependency_site=dependencies)
    for relative_path, digest in manifest["sha256"].items():
        assert hashlib.sha256((stage / relative_path).read_bytes()).hexdigest() == digest
    for path in (job.REPO_ROOT / job.WORKLOAD).iterdir():
        assert (stage / job.WORKLOAD / path.name).read_bytes() == path.read_bytes()
    assert (stage / job.PROBE_INPUTS).is_file()
    assert (stage / "tests/integration/step5_profiling/verify_fixed_probe.py").is_file()
    assert (stage / "tests/integration/step5_profiling/judge_attention_smoke.py").is_file()
    assert not (stage / "task_memory").exists()
    assert not any("stepmind_env.sh" in name for name in manifest["sha256"])
    with pytest.raises(FileExistsError):
        job.stage_probe(stage, dependency_site=dependencies)


def test_launcher_materializes_h200_resources_without_platform_imports(tmp_path, monkeypatch, preemptible=False):
    path = job.REPO_ROOT / "tests/integration/step5_profiling/submit_profiling_job.py"
    spec = importlib.util.spec_from_file_location("step5_submit_cpu_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = {}
    resources_module = ModuleType("steptron.exp.base_exp")
    resources_module.ResourceConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    stepmind_module = ModuleType("steptron.utils.stepmind")

    def spawn(resources, **kwargs):
        calls.update(resources=vars(resources), **kwargs)
        return SimpleNamespace(rjob=SimpleNamespace(meta=SimpleNamespace(name="CPU_FIXTURE_NOT_A_JOB")),
                               poll=lambda: "CPU_FIXTURE_ONLY")

    stepmind_module.spawn_tasks = spawn
    monkeypatch.setitem(sys.modules, resources_module.__name__, resources_module)
    monkeypatch.setitem(sys.modules, stepmind_module.__name__, stepmind_module)
    monkeypatch.setenv("STEPMIND_BACKEND", "rjob")
    monkeypatch.setenv("BRAINPP_ACCESS_KEY", "CPU_FIXTURE_NOT_A_CREDENTIAL")
    monkeypatch.setenv("BRAINPP_SECRET_KEY", "CPU_FIXTURE_NOT_A_CREDENTIAL")
    manifest = job.describe_probe(tmp_path / "stage", preemptible=preemptible)
    monkeypatch.chdir(tmp_path)
    argv = shlex.split(manifest["submission_command"])
    monkeypatch.setattr(sys, "argv", argv[3:])
    assert module.main() == 0
    assert calls["charged_group"] == "step_main"
    assert calls["resources"]["positive_tags"] == ["H200"]
    assert calls["resources"]["gpu"] == 1 and calls["resources"]["cpu"] == 8
    assert calls["resources"]["mem_gb"] == 32
    assert calls["resources"]["preemptible"] is manifest["preemptible"] is preemptible
    assert calls["resources"]["image"] == manifest["image"]


def test_preemptible_launcher_materializes_h200_resources_without_platform_imports(tmp_path, monkeypatch):
    test_launcher_materializes_h200_resources_without_platform_imports(tmp_path, monkeypatch, preemptible=True)


def test_dry_run_verifier_retains_normal_cli_log_prefix(monkeypatch):
    path = job.REPO_ROOT / "tests/integration/step5_profiling/verify_fixed_probe.py"
    spec = importlib.util.spec_from_file_location("step5_verify_cpu_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = 'WARNING native runtime remains unqualified\n{\n  "rows": 18\n}\n'
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: text)
    result, logs = module.dry_run(["CPU_FIXTURE_NOT_A_COMMAND"], stage=job.REPO_ROOT, env={})
    assert result == {"rows": 18}
    assert logs == ["WARNING native runtime remains unqualified"]


@pytest.mark.parametrize("method", ["cuda_event", "record_function"])
def test_actual_fixed_probe_linear_cli_accepts_6144_prefill(method, tmp_path):
    from frontier.profiling.linear_op.step5_job import profile_arguments

    tokens = tuple(sorted({cell.num_tokens for cell in probe_cells()}))
    args = profile_arguments(tmp_path, method, tokens=tokens, device="h200")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "frontier.profiling.linear_op.main", *args, "--dry_run"],
        cwd=job.REPO_ROOT, env={**os.environ, "VLLM_TARGET_DEVICE": "cpu"},
        text=True, capture_output=True, check=True,
    )
    lines = result.stdout.splitlines()
    planned = json.loads("\n".join(lines[lines.index("{"):]))
    assert sorted(planned["num_tokens"]) == [1, 2, 16, 6144]
    plans = planned["models"]["stepfun-ai/Step-5-Preview"]["plans"]
    assert [item["padded_n_expanded_embd"] for item in plans] == [1536, 13824]

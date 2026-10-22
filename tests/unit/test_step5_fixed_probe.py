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


@pytest.fixture
def installed_profile_receipt(tmp_path, monkeypatch):
    """A generated installed-package receipt, not native runtime qualification."""
    from frontier.profiling.common.runtime_identity import file_sha256
    root = tmp_path / "installed/vllm"
    names = ("vllm", "vllm.models.step5.nvidia.model", "vllm.forward_context",
             "vllm.model_executor.layers.fused_moe.router.router_factory",
             "vllm.model_executor.layers.fused_moe.router.routing_simulator_router",
             "vllm.model_executor.layers.fused_moe.router.gate_linear")
    modules = {}
    for name in names:
        relative = Path(*name.split(".")[1:])
        path = root / ("__init__.py" if name == "vllm" else relative.with_suffix(".py"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("class GateLinear:\n    def forward(self, x):\n        return x\n")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
    sources = {"vllm/" + p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*.py")}
    pins = {key: value for key, value in sources.items() if not key.endswith("gate_linear.py")}
    for index in range(53 - len(pins)):
        path = root / f"pin_{index}.py"
        path.write_text("# CPU source inventory fixture\n")
        sources["vllm/" + path.name] = file_sha256(path)
        pins["vllm/" + path.name] = file_sha256(path)
    reference = {"image": "cpu-fixture@sha256:" + "a" * 64,
                 "base_image": "cpu-fixture@sha256:" + "b" * 64, "routing_strategy": "position_cyclic"}
    stage = tmp_path / "stage"
    registry = stage / "tests/comparison/dp_placement_pp/vllm_replay.py"
    registry.parent.mkdir(parents=True)
    registry.write_text(f"def adapter_record(version):\n    assert version == 'step5_position_cyclic'\n    return {{'native_reference': {reference!r}}}\n"
                        "def check_native_reference(reference, expected):\n    if reference != expected:\n        raise ValueError('native reference mismatch')\n")
    driver = stage / "tests/comparison/stage_admission_pp/vllm_burst_driver.py"
    driver.parent.mkdir(parents=True)
    driver.write_text("# CPU registry dependency fixture\n")
    config = stage / "data/config/models/stepfun-ai__Step-5-Preview.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"local_effective_config":true}')
    derivation = stage / "derivation.json"
    derivation.write_text(json.dumps({"base_image": reference["base_image"], "derived_source_sha256": {key: pins[key] for key in list(pins)[:4]}}))
    receipt = stage / "native_installed_sources.json"
    receipt.write_text(json.dumps({"schema_version": "native_installed_sources.v1", **{key: reference[key] for key in ("image", "base_image")},
                                   "installed_package": str(root), "python_version": sys.version.split()[0],
                                   "derivation_manifest_sha256": file_sha256(derivation), "sources": sources}))
    binding = {"native_runtime_adapter": "step5_position_cyclic", "native_reference": reference,
               "profile_generation": "cpu-source-fixture", "installed_package": str(root),
               "installed_source_hashes": pins, "serving_pin_count": 53,
               "model_config_revision": "actual-local-config-revision", "model_config_sha256": file_sha256(config),
               "native_source_receipt": receipt.name, "native_derivation_manifest": derivation.name}
    manifest = {"image": reference["image"], "native_binding": binding,
                "sha256": {p.relative_to(stage).as_posix(): file_sha256(p) for p in stage.rglob("*") if p.is_file()}}
    path = stage / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "position_cyclic")
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    return path, modules, manifest


def test_installed_source_verifier_checks_actual_bound_gate_outside_serving_subset(installed_profile_receipt):
    from frontier.profiling.common.runtime_identity import verify_step5_profile_runtime
    path, modules, manifest = installed_profile_receipt
    gate = modules["vllm.model_executor.layers.fused_moe.router.gate_linear"].GateLinear()
    facts = verify_step5_profile_runtime(path, callables=(type(gate), gate.forward))
    assert len(facts["shared"]["installed_source_hashes"]) == 53
    assert len(facts["callables"]) == 2
    assert facts["shared"]["profile_generation"] == "cpu-source-fixture"
    assert "vllm.model_executor.layers.fused_moe.router.gate_linear" in facts["imports"]


@pytest.mark.parametrize("change", ["root", "spec", "hash", "receipt", "callable"])
def test_installed_source_verifier_rejects_wrong_origin_or_missing_full_receipt(installed_profile_receipt, tmp_path, change):
    from frontier.profiling.common.runtime_identity import file_sha256, verify_step5_profile_runtime
    path, modules, manifest = installed_profile_receipt
    gate_module = modules["vllm.model_executor.layers.fused_moe.router.gate_linear"]
    gate = gate_module.GateLinear()
    foreign = tmp_path / "foreign.py"
    foreign.write_text(Path(gate_module.__file__).read_text())
    if change == "root":
        gate_module.__file__ = str(foreign)
    elif change == "spec":
        gate_module.__spec__.origin = str(foreign)
    elif change == "hash":
        Path(gate_module.__file__).write_text("# changed bytes, unchanged versions\n")
    elif change == "receipt":
        receipt_path = path.parent / "native_installed_sources.json"
        receipt = json.loads(receipt_path.read_text())
        del receipt["sources"]["vllm/model_executor/layers/fused_moe/router/gate_linear.py"]
        receipt_path.write_text(json.dumps(receipt))
        manifest["sha256"][receipt_path.name] = file_sha256(receipt_path)
        path.write_text(json.dumps(manifest))
    else:
        namespace = {"__name__": gate_module.__name__}
        exec(compile(foreign.read_text(), str(foreign), "exec"), namespace)
        gate_module.GateLinear.forward = namespace["GateLinear"].forward
    with pytest.raises((ValueError, KeyError)):
        verify_step5_profile_runtime(path, callables=(gate.forward,))


def test_legacy_identity_omits_native_facts_and_bound_fingerprint_uses_shared_only():
    from dataclasses import replace
    from frontier.profiling.common.runtime_identity import RuntimeIdentity, canonical_json
    identity = RuntimeIdentity({"vllm": "same-version"}, "source", "config", "hash", "CPU", "sm00", "schema", "quant", {"tp": 1}, "NONE")
    old = {"runtime_versions": {"vllm": "same-version"}, "source_revision": "source", "device_name": "CPU", "device_capability": "sm00", "graph_mode": "NONE"}
    assert identity.fingerprint == hashlib.sha256(canonical_json(old).encode()).hexdigest()[:16]
    assert "native_facts" not in identity.columns() and "native_facts" not in identity.to_json()
    first = replace(identity, native_facts={"shared": {"image": "derived"}, "imports": {"gate": 1}, "callables": {}})
    second = replace(first, native_facts={"shared": {"image": "derived"}, "imports": {"norm": 2}, "callables": {}})
    assert first.fingerprint == second.fingerprint != identity.fingerprint
    assert first.columns()["native_facts"] != second.columns()["native_facts"]

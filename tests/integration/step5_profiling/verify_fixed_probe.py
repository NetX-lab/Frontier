#!/usr/bin/env python3
"""Verify an immutable Step small-probe stage without invoking platform tools."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def dry_run(command, *, stage, env):
    text = subprocess.check_output(command, cwd=stage, env=env, text=True)
    lines = text.splitlines()
    json_start = lines.index("{")
    return json.loads("\n".join(lines[json_start:])), lines[:json_start]


def verify(stage):
    stage = Path(stage).resolve(strict=True)
    manifest = json.loads((stage / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((stage / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Staged payload checksum differs: {name}")
    expected_files = set(manifest["sha256"]) | {"manifest.json"}
    actual_files = {str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("Staged file inventory differs from manifest")
    p0 = manifest["purpose"] == "P0_ROLE_COMPUTE_PROFILES_NOT_COMPLETE_ADMISSION"
    assert p0 or manifest["purpose"] == "SMALL_OPERATOR_PROBE_ONLY_NOT_CANONICAL_SERVING_QUALIFICATION"
    assert (manifest["gpu_tag"], manifest["charged_group"], manifest["gpus"]) == (
        ("H800", "codesign", 1) if p0 else ("H200", "step_main", 1))
    if p0:
        assert manifest["preemptible"] is False
    assert type(manifest["preemptible"]) is bool and manifest["timeout_seconds"] == 3600
    assert ("--preemptible" in shlex.split(manifest["submission_command"])) == manifest["preemptible"]
    assert manifest["cloud_output"].startswith("/mnt/codesign-exp/ycfeng/")
    assert manifest["source_mount"] == str(stage)
    assert shlex.split(manifest["submission_command"])[-1] == manifest["worker_command"]
    binding = manifest.get("native_binding")
    if binding is not None:
        plan = json.loads((stage / "profile_artifacts/profile_plan.json").read_text())
        if plan != manifest["measurement_plan"]:
            raise ValueError("sealed profile plan differs from measurement plan")
        sys.path.insert(0, str(stage / "tests/comparison/stage_admission_pp"))
        spec = importlib.util.spec_from_file_location("profile_stage_registry", stage / "tests/comparison/dp_placement_pp/vllm_replay.py")
        registry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(registry)
        registry.check_native_reference(binding["native_reference"], registry.adapter_record(binding["native_runtime_adapter"])["native_reference"])
        receipt = json.loads((stage / binding["native_source_receipt"]).read_text())
        derivation_path = stage / binding["native_derivation_manifest"]
        derivation = json.loads(derivation_path.read_text())
        pins = binding["installed_source_hashes"]
        if (len(pins) != 53 or binding["serving_pin_count"] != 53
                or any(receipt["sources"].get(key) != value for key, value in pins.items())
                or any(pins.get(key) != value for key, value in derivation["derived_source_sha256"].items())
                or receipt["derivation_manifest_sha256"] != hashlib.sha256(derivation_path.read_bytes()).hexdigest()
                or receipt["installed_package"] != binding["installed_package"]
                or receipt["image"] != binding["native_reference"]["image"]
                or receipt["base_image"] != binding["native_reference"]["base_image"]
                or derivation["base_image"] != binding["native_reference"]["base_image"]):
            raise ValueError("profile installed-source/derivation/serving-pin identity mismatch")
        relative = "data/config/models/stepfun-ai__Step-5-Preview.json"
        committed = subprocess.check_output(["git", "show", f"{binding['model_config_revision']}:{relative}"], cwd=manifest["source"]["repository"])
        if ((stage / relative).read_bytes() != committed or hashlib.sha256(committed).hexdigest() != binding["model_config_sha256"]
                or plan["source"] != manifest["source"] or plan["generation"] != binding["profile_generation"]
                or plan["model_config_revision"] != binding["model_config_revision"]):
            raise ValueError("profile source/config revision differs from committed bytes")
        if manifest["image"] != binding["native_reference"]["image"]:
            raise ValueError("profile ResourceConfig image differs from native binding")
        submission = shlex.split(manifest["submission_command"])
        if submission[submission.index("--image") + 1] != manifest["image"]:
            raise ValueError("profile launcher image differs from manifest")
    for name in ("submit.sh", "submit_with_env.sh"):
        subprocess.run(["bash", "-n", str(stage / name)], check=True)
    subprocess.run(["bash", "-n"], input=manifest["worker_command"], text=True, check=True)
    env = {**os.environ, "PYTHONPATH": f"{stage}:{stage / 'offline_site'}",
            "PYTHONDONTWRITEBYTECODE": "1", "VLLM_TARGET_DEVICE": "cpu"}
    if binding is not None:
        env.update(FRONTIER_STEP5_PROFILE_MANIFEST=str(stage / "manifest.json"),
                   VLLM_MOE_ROUTING_SIMULATION_STRATEGY=binding["native_reference"]["routing_strategy"],
                   VLLM_MOE_SKIP_PADDING="1")
    command = [sys.executable, "-B", "-m", "frontier.profiling.step5.main",
               *manifest["attention_arguments"], "--dry-run"]
    attention, attention_logs = dry_run(command, stage=stage, env=env)
    assert (attention["rows"], attention["gpu_active_rows"]) == (
        manifest["expected_attention_event_rows"], manifest["expected_attention_gpu_active_rows"])
    policy = manifest["measurement_plan"] if p0 else manifest["native_probe_policy"]
    assert attention["native_max_num_seqs"] == policy["max_num_seqs"]
    assert attention["native_max_num_batched_tokens"] == policy["max_num_batched_tokens"]
    assert attention["cudagraph_capture_sizes"] == policy["cudagraph_capture_sizes"]
    assert not attention["enable_chunked_prefill"]
    dry_runs = {}
    logs = {"attention": attention_logs}
    for line in manifest["worker_command"].splitlines():
        if "-m frontier.profiling.linear_op.main" not in line:
            continue
        args = shlex.split(line.split(" 2>&1 | tee ", 1)[0])
        args = args[args.index("-m"):]
        result, messages = dry_run([sys.executable, "-B", *args, "--dry_run"], stage=stage, env=env)
        method = args[args.index("--profile_method") + 1]
        planned = result["models"]["stepfun-ai/Step-5-Preview"]["plans"]
        assert sorted(result["num_tokens"]) == manifest["linear_plan"]["num_tokens"]
        assert planned == json.loads(json.dumps(manifest["linear_plan"]["plans"]))
        dry_runs[method] = "PASS"
        logs[method] = messages
    assert set(dry_runs) == {"cuda_event", "record_function"}
    moe_runs = {}
    if p0:
        role = manifest["measurement_role"]
        assert attention["role"] == role
        expected = manifest["measurement_plan"]["roles"][role]
        for command in manifest["producer_commands"]:
            if command["family"] != "moe":
                continue
            result, messages = dry_run([sys.executable, "-B", *command["arguments"], "--dry_run"],
                                       stage=stage, env=env)
            assert sorted(result["num_tokens"]) == sorted(expected["moe_tokens"])
            model = result["models"]["stepfun-ai/Step-5-Preview"]
            assert model["tensor_parallel_sizes"] == [1]
            assert set(model["local_experts_by_expert_parallel_size"]) == {"16"}
            assert len(model["routed_layer_groups"]) == 2
            moe_runs[command["method"]] = {"status": "PASS", "profiled_steps": model["profiled_steps"],
                                             "routed_layer_groups": model["routed_layer_groups"]}
            logs["moe_" + command["method"]] = messages
        assert set(moe_runs) == {"cuda_event", "record_function"}
    workload = stage / manifest["workload"]["path"]
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=workload, check=True,
                   stdout=subprocess.PIPE, text=True)
    return {"status": "PASS", "staged_files": len(manifest["sha256"]),
            "stage": str(stage), "manifest_sha256": hashlib.sha256(
                (stage / "manifest.json").read_bytes()).hexdigest(),
            "attention_event_rows": attention["rows"],
            "attention_gpu_active_rows": attention["gpu_active_rows"],
            "linear_rows": 2 * manifest["linear_plan"]["expected_rows_per_method"], "linear_dry_runs": dry_runs,
            **({"moe_dry_runs": moe_runs, "measurement_role": manifest["measurement_role"]} if p0 else {}),
            "dry_run_logs": logs,
            "platform_operations": 0, "admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.stage)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()

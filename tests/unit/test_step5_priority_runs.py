"""D21 topology and scheduling-policy checks; CPU fixtures are not native evidence."""

import json
from pathlib import Path
import copy
import shutil

import pytest

from tests.integration.step5_pd import build_stage, canonical_stage, formal_release, submit_native_pd
from tests.unit.test_step5_native_pd import stage_inputs
from tests.unit.test_step5_client_metrics import step_native, synthetic_step_check
from tests.unit.test_canonical_client_metrics import native, simulator, TRACE


@pytest.mark.parametrize("case_id", canonical_stage.CASES)
def test_each_priority_case_uses_native_node_plan_and_final_resource_policy(tmp_path, case_id):
    inputs = stage_inputs(tmp_path)
    definition = canonical_stage.case_definition(json.loads(
        (canonical_stage.REPO / "tests/integration/step5_pd/canonical_run_h200.json").read_text()), case_id)
    definition["probe_helpers"] = inputs["definition"]["probe_helpers"]
    stage = tmp_path / "workspace/stage"
    config = build_stage.build(definition, stage=stage, workspace=stage.parent,
        output_dir=Path("/mnt/codesign-exp/ycfeng/d21_test"), job_tag="fixture", source={"commit": "fixture"},
        **{key: value for key, value in inputs.items() if key != "definition"})
    case = canonical_stage.CASES[case_id]
    rows = json.loads((stage / "preflight.json").read_text())
    assert len(rows) == case["replica"]
    assert [(r["role"], r["lane"]) for r in rows] == [
        (role, lane) for role in ("prefill", "decode") for lane in range(case["data_parallel_size"])]
    arguments = submit_native_pd.resource_arguments(config)
    assert arguments["preemptible"] is case["preemptible"]
    assert arguments["positive_tags"] == [case["gpu_type"]]
    assert config["charged_group"] == case["charged_group"]
    for row in rows:
        command = row["server_command"]
        assert command[command.index("--data-parallel-size") + 1] == str(case["data_parallel_size"])
        assert not any(key.startswith("NCCL_") for key in row["server_env_set"])


@pytest.fixture(params=["h800", "h200"])
def release_inputs(tmp_path, step_native, simulator, request):
    """Real receipt/row schemas with explicitly synthetic contents; never a native claim."""
    root, _, _ = step_native
    simulator = shutil.copytree(simulator, tmp_path / "frontier")
    hardware = request.param
    case32_id, case64_id = f"{hardware}_32", f"{hardware}_64"
    cell32, case32 = formal_release.declaration(case32_id)
    # The exported CPU Request fixture is not a complete simulation. Supply the
    # exact resolved configuration schema as a synthetic shape-binding oracle.
    original = {"sys_arch": "pd-disaggregation", "cluster_config": {}}
    for role in ("prefill", "decode"):
        original["cluster_config"][f"{role}_cluster_num_replicas"] = 1
        original["cluster_config"][f"{role}_replica_config"] = dict(
            model_name=case32["frontier_model_name"], device=hardware, attn_tensor_parallel_size=8,
            attn_dp=2, moe_expert_parallel_size=16, moe_tensor_parallel_size=1, num_pipeline_stages=1)
    [truth] = (simulator / "metrics").rglob("metrics_ground_truth.jsonl")
    truth.with_name("config.json").write_text(json.dumps(original))
    import os
    os.utime(truth.with_name("config.json"), (1, 1))
    receipt32 = synthetic_step_check(root, cell32, case32)
    ready = shutil.copytree(root, tmp_path / "ready/runs/groundtruth_clean/qualification")
    cell64, case64 = formal_release.declaration(case64_id)
    receipt64 = synthetic_step_check(ready, cell64, case64)
    matrix = {"cases": {case32_id: case32 | {"routing": {"status": "NOT_APPLICABLE"}}},
              "cells": [copy.deepcopy(cell32)]}
    matrix["cells"][0]["run_check"] = formal_release.pin(receipt32)
    for record in matrix["cells"][0]["trace"].values():
        record["path"] = str(TRACE / Path(record["path"]).name)
    matrix_path = tmp_path / "dependency_matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    config = json.loads((ready / "config.json").read_text())
    config["workspace"] = str(tmp_path)
    config["canonical_replay"] = {"kind": "canonical"}
    [truth] = (simulator / "metrics").rglob("metrics_ground_truth.jsonl")
    config["formal_inputs"] = {
        "readiness": {"run_directory": str(ready), "run_check": formal_release.pin(receipt64)},
        "accuracy": {"matrix": formal_release.pin(matrix_path), "artifacts_root": str(tmp_path),
                     "run_directory": str(root), "frontier_run_directory": str(simulator),
                     "frontier_artifacts": [formal_release.pin(p) for p in (truth, truth.with_name("config.json"))]}}
    return config, root, matrix_path


def test_formal_release_uses_actual_three_repeat_four_metric_schema(release_inputs):
    config, _, _ = release_inputs
    result = formal_release.validate(config)
    assert result["status"] == "PASS"
    reading = result["accuracy"]["reading"]
    assert reading["relative_error_limit"] == .10
    assert len(reading["repetitions"]) == 3
    assert all(len(r["metrics"]) == 4 for r in reading["repetitions"].values())


@pytest.mark.parametrize("fault", ["missing_readiness", "missing_accuracy", "scalar", "wrong_hardware",
    "wrong_population", "wrong_workload", "wrong_runtime", "wrong_model", "wrong_gpu_inventory", "changed_frontier",
    "wrong_frontier_hardware", "wrong_frontier_scale"])
def test_formal64_rejects_missing_or_foreign_evidence(release_inputs, fault):
    config, root, matrix_path = release_inputs
    dependency = canonical_stage.CASES[config["case_id"]]["formal_dependency"]
    other_hardware = "H200" if config["gpu_type"] == "H800" else "H800"
    if fault == "missing_readiness": config["formal_inputs"].pop("readiness")
    if fault == "missing_accuracy": config["formal_inputs"].pop("accuracy")
    if fault == "scalar": config["formal_inputs"]["accuracy"] = {"error": .01, "ready": True}
    if fault in ("wrong_hardware", "wrong_workload", "wrong_runtime"):
        matrix = json.loads(matrix_path.read_text())
        if fault == "wrong_hardware": matrix["cells"][0]["case_id"] = other_hardware.lower() + "_32"
        if fault == "wrong_workload": matrix["cells"][0]["trace"]["workload"]["sha256"] = "e" * 64
        if fault == "wrong_runtime": matrix["cases"][dependency]["native_runtime"]["vllm_version"] = "0.28.0"
        matrix_path.write_text(json.dumps(matrix))
        config["formal_inputs"]["accuracy"]["matrix"] = formal_release.pin(matrix_path)
    if fault == "wrong_model": config["model_hashes"] = {"config.json": "wrong"}
    if fault in ("wrong_population", "wrong_gpu_inventory"):
        path = root / ("replay/client_requests.jsonl" if fault == "wrong_population" else "nodes/decode1/result.json")
        if fault == "wrong_population": path.write_text("\n".join(path.read_text().splitlines()[:-1])+"\n")
        else:
            node = json.loads(path.read_text())
            node["nvidia_smi"] = node["nvidia_smi"].replace(config["gpu_type"], other_hardware)
            path.write_text(json.dumps(node))
        receipt = root.parents[2] / "analysis/run_check_status.groundtruth_clean.json"
        document = json.loads(receipt.read_text())
        check = document["runs"][root.name]
        check["artifacts"] = [formal_release.pin(path) if (root.parents[2] / p["path"]).resolve() == path.resolve() else p
                              for p in check["artifacts"]]
        receipt.write_text(json.dumps(document))
        matrix = json.loads(matrix_path.read_text())
        matrix["cells"][0]["run_check"] = formal_release.pin(receipt)
        matrix_path.write_text(json.dumps(matrix))
        config["formal_inputs"]["accuracy"]["matrix"] = formal_release.pin(matrix_path)
    if fault == "changed_frontier":
        record = config["formal_inputs"]["accuracy"]["frontier_artifacts"][0]
        record["sha256"] = "f" * 64
    if fault.startswith("wrong_frontier_"):
        accuracy = config["formal_inputs"]["accuracy"]
        path = Path(accuracy["frontier_run_directory"]) / "metrics/config.json"
        document = json.loads(path.read_text())
        field, value = ("device", other_hardware.lower()) if fault == "wrong_frontier_hardware" else ("attn_dp", 4)
        document["cluster_config"]["decode_replica_config"][field] = value
        path.write_text(json.dumps(document))
        import os
        os.utime(path, (1, 1))
        accuracy["frontier_artifacts"] = [formal_release.pin(Path(record["path"])) for record in accuracy["frontier_artifacts"]]
    with pytest.raises((ValueError, KeyError, SystemExit)):
        formal_release.validate(config)


@pytest.mark.parametrize("relative_error,passes", [(.30, True), (.3001, False)])
def test_dependency30_is_per_repeat_and_not_final10(release_inputs, relative_error, passes):
    config, root, matrix_path = release_inputs
    # Binary-exact 2.5/3.25-second endpoints make equality exactly 30%, not
    # a floating-point approximation produced by inverse endpoint arithmetic.
    path = root / "replay/client_requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        first = 2.5 if row["repeat"] == 3 else 3.25
        row["first_token_monotonic"] = row["dispatch_monotonic"] + first
        row["last_token_monotonic"] = row["dispatch_monotonic"] + first + 2.25
        row["response_monotonic"] = row["last_token_monotonic"] + .01
        row.update(ttft_ms=first * 1000, e2e_ms=(first + 2.25) * 1000, tpot_ms=2250 / 1023)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    receipt = root.parents[2] / "analysis/run_check_status.groundtruth_clean.json"
    document = json.loads(receipt.read_text())
    check = document["runs"][root.name]
    check["artifacts"] = [formal_release.pin(path) if p["path"].endswith("replay/client_requests.jsonl") else p
                          for p in check["artifacts"]]
    receipt.write_text(json.dumps(document))
    matrix = json.loads(matrix_path.read_text())
    matrix["cells"][0]["run_check"] = formal_release.pin(receipt)
    matrix_path.write_text(json.dumps(matrix))
    config["formal_inputs"]["accuracy"]["matrix"] = formal_release.pin(matrix_path)
    accuracy = config["formal_inputs"]["accuracy"]
    [truth] = (Path(accuracy["frontier_run_directory"]) / "metrics").rglob("metrics_ground_truth.jsonl")
    events = [json.loads(line) for line in truth.read_text().splitlines()]
    for event in events:
        if event["event_type"] == "request_completion":
            first = 2.5 * (1 + relative_error)
            event["first_decode_token_completed_at"] = event["arrived_at"] + first
            event["completed_at"] = event["arrived_at"] + first + 2.25
    truth.write_text("".join(json.dumps(event)+"\n" for event in events))
    accuracy["frontier_artifacts"] = [formal_release.pin(p) for p in (truth, truth.with_name("config.json"))]
    if passes:
        result = formal_release.validate(config)
        assert result["accuracy"]["reading"]["verdict"] == "FAIL"  # 10% final gate remains unchanged.
    else:
        with pytest.raises(ValueError, match="all four errors <=30%"):
            formal_release.validate(config)


def test_basic_qualification_does_not_consume_accuracy_or_readiness():
    for case_id in canonical_stage.CASES:
        assert formal_release.validate({"case_id": case_id, "canonical_replay": {"kind": "qualification"}})["status"] == "NOT_REQUIRED"


def test_formal32_requires_own_native_prerequisites_but_no_accuracy_dependency(release_inputs):
    _, root, _ = release_inputs
    config = json.loads((root / "config.json").read_text())
    config["canonical_replay"] = {"kind": "canonical"}
    with pytest.raises(ValueError, match="readiness"):
        formal_release.validate(config)
    config["formal_inputs"] = {"readiness": {"run_directory": str(root),
        "run_check": formal_release.pin(root.parents[2] / "analysis/run_check_status.groundtruth_clean.json")}}
    assert formal_release.validate(config)["status"] == "PASS"

from __future__ import annotations

import csv
import json
from pathlib import Path

from tests.e2e.operator_parity.run_golden_matrix import (
    GOLDEN_MATRIX_EXPECTED_CASES,
    OP_TRACES_CSV,
    REQUEST_METRICS_CSV,
    WORKLOAD_PROFILES,
    build_golden_cases,
    normalize_op_traces_jsonl_to_csv,
    parameter_memory_per_device_bytes,
    requested_memory_bytes_for_device,
    run_golden_matrix,
    selected_parallelism_for_model,
)
from tests.e2e.operator_parity.profile_prerequisite_audit import (
    GOLDEN_CONFIG_FILENAMES,
)
from frontier.utils.output_paths import build_metrics_run_output_dir


def _write_config(config_root: Path, filename: str, *, is_moe: bool) -> None:
    config_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"model_type": "test_moe" if is_moe else "test_dense"}
    if is_moe:
        payload["num_experts"] = 8
    (config_root / filename).write_text(json.dumps(payload), encoding="utf-8")


def _write_profile_csv(
    path: Path,
    *,
    tensor_parallel_values: tuple[int, ...] = (1,),
    expert_parallel_values: tuple[int, ...] = (1,),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name in {"moe.csv", "moe_kernel_only.csv"}:
        rows = ["num_tensor_parallel_workers,expert_parallel_size,time_stats.example.mean"]
        for tp in tensor_parallel_values:
            for ep in expert_parallel_values:
                rows.append(f"{tp},{ep},1.25")
    else:
        rows = ["num_tensor_parallel_workers,time_stats.example.mean"]
        rows.extend(f"{tp},1.25" for tp in tensor_parallel_values)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _prepare_profiles(config_root: Path, profile_root: Path) -> None:
    config_root.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(GOLDEN_CONFIG_FILENAMES):
        is_moe = index > 0
        _write_config(config_root, filename, is_moe=is_moe)
        model_name = Path(filename).stem
        required = [
            "attention.csv",
            "attention_kernel_only.csv",
            "linear_op.csv",
            "linear_op_kernel_only.csv",
        ]
        if is_moe:
            required.extend(["moe.csv", "moe_kernel_only.csv"])
        for required_file in required:
            if model_name == "step-moe-noquant-small":
                _write_profile_csv(
                    profile_root / model_name / required_file,
                    tensor_parallel_values=(1, 2, 4, 8),
                    expert_parallel_values=(1, 2, 4, 8),
                )
            else:
                _write_profile_csv(profile_root / model_name / required_file)


def _write_sim_artifacts(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "request_metrics.csv").write_text(
        "Request Id,request_e2e_time,ttft,tpot,request_num_decode_tokens\n"
        "0,10.0,2.0,1.0,8\n",
        encoding="utf-8",
    )
    (case_dir / "op_traces.jsonl").write_text(
        json.dumps({"meta": {"version": "test"}})
        + "\n"
        + json.dumps(
            {
                "type": "COMPUTE",
                "name": "attn_prefill",
                "ts_start": 0.0,
                "duration_ms": 1.5,
                "cluster": "MONOLITHIC",
                "batch_id": "b0",
                "meta": {"model_name": "demo", "parallel_context": {"tp": 1}},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_golden_cases_emits_48_cases_and_pins_measurement_type_profiles(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    profile_root = tmp_path / "profiles" / "h800"
    _prepare_profiles(config_root, profile_root)

    cases, manifest = build_golden_cases(
        config_root=config_root,
        profile_root=profile_root,
        reference_root=tmp_path / "reference",
        candidate_root=tmp_path / "candidate",
    )

    assert manifest["expected_original"] == GOLDEN_MATRIX_EXPECTED_CASES
    assert manifest["actual"] == GOLDEN_MATRIX_EXPECTED_CASES
    assert len(cases) == GOLDEN_MATRIX_EXPECTED_CASES
    assert {case["case_manifest"]["dummy_mode"] for case in cases} == {False}
    step_moe_case = next(
        case
        for case in cases
        if case["case_manifest"]["model_name"] == "step-moe-noquant-small"
        and case["case_manifest"]["sys_arch"] == "co-location"
        and case["case_manifest"]["measurement_type"] == "CUDA_EVENT"
    )
    assert step_moe_case["case_manifest"]["parallelism"] == {
        "attn_tensor_parallel_size": 4,
        "attn_data_parallel_size": 1,
        "moe_tensor_parallel_size": 4,
        "moe_expert_parallel_size": 1,
        "num_pipeline_stages": 1,
    }
    assert manifest["parallelism_by_model"]["step-moe-noquant-small"]["source"] == "pinned"
    kernel_case = next(
        case
        for case in cases
        if case["case_manifest"]["model_name"] == "Phi-tiny-MoE-instruct"
        and case["case_manifest"]["measurement_type"] == "KERNEL_ONLY"
    )
    profile_files = kernel_case["reference_input_manifest"]["profile_files"]
    assert set(profile_files) == {
        "attention.csv",
        "attention_kernel_only.csv",
        "linear_op.csv",
        "linear_op_kernel_only.csv",
        "moe.csv",
        "moe_kernel_only.csv",
    }
    assert kernel_case["reference_input_manifest"] == kernel_case["candidate_input_manifest"]
    assert kernel_case["reference_dir"] == build_metrics_run_output_dir(
        output_root=str(tmp_path / "reference"),
        model_type="Phi-tiny-MoE-instruct",
        workload_type=kernel_case["case_manifest"]["simulation_mode"],
        run_id=kernel_case["name"],
    )


def test_build_golden_cases_can_expand_multiple_workload_profiles(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    profile_root = tmp_path / "profiles" / "h800"
    config_filename = GOLDEN_CONFIG_FILENAMES[0]
    _write_config(config_root, config_filename, is_moe=False)
    for required_file in (
        "attention.csv",
        "attention_kernel_only.csv",
        "linear_op.csv",
        "linear_op_kernel_only.csv",
    ):
        _write_profile_csv(profile_root / Path(config_filename).stem / required_file)

    cases, manifest = build_golden_cases(
        config_root=config_root,
        profile_root=profile_root,
        reference_root=tmp_path / "reference",
        candidate_root=tmp_path / "candidate",
        config_filenames=(config_filename,),
        workload_profiles=("short_single", "multi_medium", "long_single"),
    )

    assert manifest["actual"] == 24
    assert manifest["workload_profiles"] == {
        "short_single": WORKLOAD_PROFILES["short_single"].to_dict(),
        "multi_medium": WORKLOAD_PROFILES["multi_medium"].to_dict(),
        "long_single": WORKLOAD_PROFILES["long_single"].to_dict(),
    }
    names = {case["name"] for case in cases}
    assert "llama2_7b_dense_example__offline_batch__co_location__cuda_event" in names
    assert (
        "llama2_7b_dense_example__offline_batch__co_location__cuda_event"
        "__workload_multi_medium"
        in names
    )
    assert (
        "llama2_7b_dense_example__offline_batch__co_location__cuda_event"
        "__workload_long_single"
        in names
    )
    assert {case["case_manifest"]["workload_profile"] for case in cases} == {
        "short_single",
        "multi_medium",
        "long_single",
    }


def test_build_golden_cases_applies_tiny_float_tolerance_only_to_request_metrics(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    profile_root = tmp_path / "profiles" / "h800"
    _prepare_profiles(config_root, profile_root)

    cases, _ = build_golden_cases(
        config_root=config_root,
        profile_root=profile_root,
        reference_root=tmp_path / "reference",
        candidate_root=tmp_path / "candidate",
    )

    case = cases[0]
    tolerances = case["simulation_tolerance_allowlist"]
    assert tolerances[REQUEST_METRICS_CSV] == {
        "*": {"absolute": 1e-12, "relative_pct": 1e-7}
    }
    assert tolerances[OP_TRACES_CSV] == {
        "ts_start": {"absolute": 1e-12, "relative_pct": 1e-7},
        "duration_ms": {"absolute": 1e-12, "relative_pct": 1e-7},
    }
    assert "*" not in tolerances[OP_TRACES_CSV]


def test_build_golden_cases_fails_when_pinned_parallelism_profile_coverage_is_missing(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    profile_root = tmp_path / "profiles" / "h800"
    _write_config(config_root, "step-moe-noquant-small.json", is_moe=True)
    for required_file in (
        "attention.csv",
        "attention_kernel_only.csv",
        "linear_op.csv",
        "linear_op_kernel_only.csv",
        "moe.csv",
        "moe_kernel_only.csv",
    ):
        _write_profile_csv(profile_root / "step-moe-noquant-small" / required_file)

    try:
        build_golden_cases(
            config_root=config_root,
            profile_root=profile_root,
            reference_root=tmp_path / "reference",
            candidate_root=tmp_path / "candidate",
            config_filenames=("step-moe-noquant-small.json",),
        )
    except ValueError as exc:
        assert "missing required profiling coverage" in str(exc)
        assert "step-moe-noquant-small" in str(exc)
        assert "num_tensor_parallel_workers=4" in str(exc)
    else:
        raise AssertionError("expected missing profile coverage to fail fast")


def test_step_moe_pinned_parallelism_is_memory_feasible_for_h800() -> None:
    parallelism = selected_parallelism_for_model("step-moe-noquant-small")

    parameter_memory = parameter_memory_per_device_bytes(
        model_name="step-moe-noquant-small",
        device="h800",
        parallelism=parallelism,
    )
    requested_memory = requested_memory_bytes_for_device(device="h800")

    assert parameter_memory < requested_memory
    assert parameter_memory == 42963156992
    assert requested_memory == 77309411328


def test_normalize_op_traces_jsonl_to_csv_writes_stable_ragged_trace_csv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "op_traces.jsonl"
    target = tmp_path / "op_traces.csv"
    source.write_text(
        json.dumps({"meta": {"timestamp": 1}})
        + "\n"
        + json.dumps(
            {
                "type": "COMPUTE",
                "name": "attn_decode",
                "ts_start": 0.5,
                "duration_ms": 2.0,
                "cluster": "MONOLITHIC",
                "layer_id": 3,
                "meta": {"b": 2, "a": 1},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "COMM",
                "name": "tensor_parallel_allreduce",
                "ts_start": 0.7,
                "duration_ms": 0.25,
                "cluster": "MONOLITHIC",
                "target_cluster": "DECODE",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = normalize_op_traces_jsonl_to_csv(source, target)

    assert report["event_count"] == 2
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["trace_index"] for row in rows] == ["0", "1"]
    assert rows[0]["name"] == "attn_decode"
    assert rows[0]["meta_json"] == json.dumps({"a": 1, "b": 2}, sort_keys=True)
    assert rows[1]["target_cluster"] == "DECODE"


def test_run_golden_matrix_normalizes_op_traces_and_reuses_attention_runner(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    profile_root = tmp_path / "profiles" / "h800"
    config_filename = GOLDEN_CONFIG_FILENAMES[0]
    _write_config(config_root, config_filename, is_moe=False)
    for required_file in (
        "attention.csv",
        "attention_kernel_only.csv",
        "linear_op.csv",
        "linear_op_kernel_only.csv",
    ):
        _write_profile_csv(profile_root / Path(config_filename).stem / required_file)

    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    cases, _ = build_golden_cases(
        config_root=config_root,
        profile_root=profile_root,
        reference_root=reference_root,
        candidate_root=candidate_root,
        config_filenames=(config_filename,),
    )
    for case in cases:
        _write_sim_artifacts(Path(case["reference_dir"]))
        _write_sim_artifacts(Path(case["candidate_dir"]))

    report = run_golden_matrix(
        config_root=config_root,
        profile_root=profile_root,
        reference_root=reference_root,
        candidate_root=candidate_root,
        output_matrix_yaml=tmp_path / "matrix_cases.yaml",
        config_filenames=(config_filename,),
    )

    assert report["matrix_report"]["status"] == "PASS"
    assert report["matrix_report"]["case_count"] == 8
    assert (Path(cases[0]["reference_dir"]) / "op_traces.csv").is_file()
    assert report["case_manifest"]["actual"] == 8


def test_run_golden_matrix_reports_candidate_numeric_trace_regression(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    profile_root = tmp_path / "profiles" / "h800"
    config_filename = GOLDEN_CONFIG_FILENAMES[0]
    _write_config(config_root, config_filename, is_moe=False)
    for required_file in (
        "attention.csv",
        "attention_kernel_only.csv",
        "linear_op.csv",
        "linear_op_kernel_only.csv",
    ):
        _write_profile_csv(profile_root / Path(config_filename).stem / required_file)

    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    cases, _ = build_golden_cases(
        config_root=config_root,
        profile_root=profile_root,
        reference_root=reference_root,
        candidate_root=candidate_root,
        config_filenames=(config_filename,),
    )
    for case in cases:
        _write_sim_artifacts(Path(case["reference_dir"]))
        _write_sim_artifacts(Path(case["candidate_dir"]))

    candidate_trace = Path(cases[0]["candidate_dir"]) / "op_traces.jsonl"
    lines = candidate_trace.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[1])
    event["duration_ms"] = 2.5
    lines[1] = json.dumps(event)
    candidate_trace.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = run_golden_matrix(
        config_root=config_root,
        profile_root=profile_root,
        reference_root=reference_root,
        candidate_root=candidate_root,
        output_matrix_yaml=tmp_path / "matrix_cases.yaml",
        config_filenames=(config_filename,),
    )

    matrix_report = report["matrix_report"]
    assert matrix_report["status"] == "CANDIDATE_MISMATCH"
    failed_results = [
        result for result in matrix_report["results"] if result["status"] != "PASS"
    ]
    assert len(failed_results) == 1
    failed_case = failed_results[0]
    assert failed_case["name"] == cases[0]["name"]
    op_trace_report = failed_case["reports"]["simulation"]["artifacts"][OP_TRACES_CSV]
    assert op_trace_report["status"] == "FAIL"
    assert op_trace_report["mismatch_count"] > 0

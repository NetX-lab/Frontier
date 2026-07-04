from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.operator_parity.run_golden_baseline import (
    build_frontier_command,
    cli_simulation_mode,
    profile_file_flags,
    run_baseline_generation,
)
import tests.e2e.operator_parity.run_golden_baseline as golden_baseline

SHORT_SINGLE_WORKLOAD_SHAPE = {
    "num_requests": 1,
    "prefill_tokens": 8,
    "decode_tokens": 2,
    "qps": 1.0,
}

MULTI_MEDIUM_WORKLOAD_SHAPE = {
    "num_requests": 2,
    "prefill_tokens": 16,
    "decode_tokens": 4,
    "qps": 2.0,
}

LONG_SINGLE_WORKLOAD_SHAPE = {
    "num_requests": 1,
    "prefill_tokens": 16,
    "decode_tokens": 4,
    "qps": 1.0,
}


def test_kernel_only_case_pins_mixed_runtime_flags_and_profiles() -> None:
    case = {
        "name": "Phi-tiny-MoE-instruct__offline_batch__co_location__kernel_only",
        "case_manifest": {
            "model_name": "Phi-tiny-MoE-instruct",
            "simulation_mode": "offline_batch",
            "sys_arch": "co-location",
            "measurement_type": "KERNEL_ONLY",
            "workload_profile": "short_single",
            "workload_shape": SHORT_SINGLE_WORKLOAD_SHAPE,
        },
        "reference_input_manifest": {
            "profile_files": {
                "attention.csv": {"path": "profiles/Phi/attention.csv"},
                "attention_kernel_only.csv": {"path": "profiles/Phi/attention_kernel_only.csv"},
                "linear_op.csv": {"path": "profiles/Phi/linear_op.csv"},
                "linear_op_kernel_only.csv": {"path": "profiles/Phi/linear_op_kernel_only.csv"},
                "moe.csv": {"path": "profiles/Phi/moe.csv"},
                "moe_kernel_only.csv": {"path": "profiles/Phi/moe_kernel_only.csv"},
            }
        },
    }

    flags = profile_file_flags(case)
    command = build_frontier_command(
        case,
        output_root=Path("baseline/reference_outputs"),
        python_bin="python",
        extra_args=("--custom_flag", "1"),
    )

    assert flags["--random_forrest_execution_time_predictor_config_atten_input_file"] == "profiles/Phi/attention.csv"
    assert flags["--random_forrest_execution_time_predictor_config_atten_kernel_only_input_file"] == "profiles/Phi/attention_kernel_only.csv"
    assert flags["--random_forrest_execution_time_predictor_config_moe_input_file"] == "profiles/Phi/moe.csv"
    assert flags["--random_forrest_execution_time_predictor_config_moe_kernel_only_input_file"] == "profiles/Phi/moe_kernel_only.csv"
    assert "--decode_cuda_graph_mode" in command
    assert command[command.index("--decode_cuda_graph_mode") + 1] == "full_decode_only"
    assert "--no-random_forrest_execution_time_predictor_config_enable_dummy_mode" in command
    assert "--metrics_config_enable_op_level_tracing" in command
    assert command[-2:] == ["--custom_flag", "1"]


def test_cuda_event_pdd_case_uses_eager_profiles_and_sequential_pdd() -> None:
    case = {
        "name": "llama2_7b_dense_example__online_serving__pd_disaggregation__cuda_event",
        "case_manifest": {
            "model_name": "llama2_7b_dense_example",
            "simulation_mode": "online_serving",
            "sys_arch": "pd-disaggregation",
            "measurement_type": "CUDA_EVENT",
            "workload_profile": "short_single",
            "workload_shape": SHORT_SINGLE_WORKLOAD_SHAPE,
        },
        "reference_input_manifest": {
            "profile_files": {
                "attention.csv": {"path": "profiles/llama/attention.csv"},
                "linear_op.csv": {"path": "profiles/llama/linear_op.csv"},
            }
        },
    }

    command = build_frontier_command(
        case,
        output_root=Path("baseline/reference_outputs"),
        python_bin="python",
    )

    assert cli_simulation_mode("offline_batch") == "offline"
    assert cli_simulation_mode("online_serving") == "online"
    assert "--no-enable_parallel_clusters" in command
    assert "--cluster_config_prefill_replica_config_device" in command
    assert "--cluster_config_decode_replica_config_device" in command
    assert command[command.index("--simulation_mode") + 1] == "online"
    assert command[command.index("--decode_cuda_graph_mode") + 1] == "none"
    assert "--random_forrest_execution_time_predictor_config_atten_kernel_only_input_file" not in command
    assert "--random_forrest_execution_time_predictor_config_moe_input_file" not in command


def test_step_moe_case_uses_pinned_parallelism_for_colocation_and_pdd() -> None:
    profile_files = {
        "attention.csv": {"path": "profiles/step/attention.csv"},
        "linear_op.csv": {"path": "profiles/step/linear_op.csv"},
        "moe.csv": {"path": "profiles/step/moe.csv"},
    }
    parallelism = {
        "attn_tensor_parallel_size": 4,
        "attn_data_parallel_size": 1,
        "moe_tensor_parallel_size": 4,
        "moe_expert_parallel_size": 1,
        "num_pipeline_stages": 1,
    }
    colocation_case = {
        "name": "step-moe-noquant-small__offline_batch__co_location__cuda_event",
        "case_manifest": {
            "model_name": "step-moe-noquant-small",
            "simulation_mode": "offline_batch",
            "sys_arch": "co-location",
            "measurement_type": "CUDA_EVENT",
            "parallelism": parallelism,
            "workload_profile": "short_single",
            "workload_shape": SHORT_SINGLE_WORKLOAD_SHAPE,
        },
        "reference_input_manifest": {"profile_files": profile_files},
    }
    pdd_case = {
        "name": "step-moe-noquant-small__offline_batch__pd_disaggregation__cuda_event",
        "case_manifest": {
            **colocation_case["case_manifest"],
            "sys_arch": "pd-disaggregation",
        },
        "reference_input_manifest": {"profile_files": profile_files},
    }

    colocation_command = build_frontier_command(
        colocation_case,
        output_root=Path("baseline/reference_outputs"),
        python_bin="python",
    )
    pdd_command = build_frontier_command(
        pdd_case,
        output_root=Path("baseline/reference_outputs"),
        python_bin="python",
    )

    assert colocation_command[colocation_command.index("--replica_config_attn_tensor_parallel_size") + 1] == "4"
    assert colocation_command[colocation_command.index("--replica_config_attn_data_parallel_size") + 1] == "1"
    assert colocation_command[colocation_command.index("--replica_config_moe_tensor_parallel_size") + 1] == "4"
    assert colocation_command[colocation_command.index("--replica_config_moe_expert_parallel_size") + 1] == "1"
    assert pdd_command[pdd_command.index("--cluster_config_prefill_replica_config_attn_tensor_parallel_size") + 1] == "4"
    assert pdd_command[pdd_command.index("--cluster_config_prefill_replica_config_moe_tensor_parallel_size") + 1] == "4"
    assert pdd_command[pdd_command.index("--cluster_config_decode_replica_config_attn_tensor_parallel_size") + 1] == "4"
    assert pdd_command[pdd_command.index("--cluster_config_decode_replica_config_moe_tensor_parallel_size") + 1] == "4"


class _DivergentProfile:
    num_requests = 999
    prefill_tokens = 888
    decode_tokens = 777
    qps = 666.0

    def to_dict(self) -> dict[str, int | float]:
        return dict(MULTI_MEDIUM_WORKLOAD_SHAPE)


def test_custom_workload_profile_updates_request_shape_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(golden_baseline.WORKLOAD_PROFILES, "multi_medium", _DivergentProfile())
    case = {
        "name": "llama2_7b_dense_example__online_serving__co_location__cuda_event__workload_multi_medium",
        "case_manifest": {
            "model_name": "llama2_7b_dense_example",
            "simulation_mode": "online_serving",
            "sys_arch": "co-location",
            "measurement_type": "CUDA_EVENT",
            "workload_profile": "multi_medium",
            "workload_shape": MULTI_MEDIUM_WORKLOAD_SHAPE,
        },
        "reference_input_manifest": {
            "profile_files": {
                "attention.csv": {"path": "profiles/llama/attention.csv"},
                "linear_op.csv": {"path": "profiles/llama/linear_op.csv"},
            }
        },
    }

    command = build_frontier_command(
        case,
        output_root=Path("baseline/reference_outputs"),
        python_bin="python",
    )

    assert command[command.index("--synthetic_request_generator_config_num_requests") + 1] == "2"
    assert command[command.index("--fixed_request_length_generator_config_prefill_tokens") + 1] == "16"
    assert command[command.index("--fixed_request_length_generator_config_decode_tokens") + 1] == "4"
    assert command[command.index("--poisson_request_interval_generator_config_qps") + 1] == "2.0"


def test_long_single_workload_profile_updates_request_shape_flags() -> None:
    case = {
        "name": "llama2_7b_dense_example__online_serving__co_location__cuda_event__workload_long_single",
        "case_manifest": {
            "model_name": "llama2_7b_dense_example",
            "simulation_mode": "online_serving",
            "sys_arch": "co-location",
            "measurement_type": "CUDA_EVENT",
            "workload_profile": "long_single",
            "workload_shape": LONG_SINGLE_WORKLOAD_SHAPE,
        },
        "reference_input_manifest": {
            "profile_files": {
                "attention.csv": {"path": "profiles/llama/attention.csv"},
                "linear_op.csv": {"path": "profiles/llama/linear_op.csv"},
            }
        },
    }

    command = build_frontier_command(
        case,
        output_root=Path("baseline/reference_outputs"),
        python_bin="python",
    )

    assert command[command.index("--synthetic_request_generator_config_num_requests") + 1] == "1"
    assert command[command.index("--fixed_request_length_generator_config_prefill_tokens") + 1] == "16"
    assert command[command.index("--fixed_request_length_generator_config_decode_tokens") + 1] == "4"
    assert command[command.index("--poisson_request_interval_generator_config_qps") + 1] == "1.0"


@pytest.mark.parametrize(
    "case_manifest,error_substring",
    (
        (
            {
                "model_name": "llama2_7b_dense_example",
                "simulation_mode": "online_serving",
                "sys_arch": "co-location",
                "measurement_type": "CUDA_EVENT",
                "workload_shape": SHORT_SINGLE_WORKLOAD_SHAPE,
            },
            "missing required golden workload_profile",
        ),
        (
            {
                "model_name": "llama2_7b_dense_example",
                "simulation_mode": "online_serving",
                "sys_arch": "co-location",
                "measurement_type": "CUDA_EVENT",
                "workload_profile": "near_model_substitute",
                "workload_shape": SHORT_SINGLE_WORKLOAD_SHAPE,
            },
            "unsupported golden workload profile",
        ),
        (
            {
                "model_name": "llama2_7b_dense_example",
                "simulation_mode": "online_serving",
                "sys_arch": "co-location",
                "measurement_type": "CUDA_EVENT",
                "workload_profile": "short_single",
            },
            "missing required golden workload_shape",
        ),
        (
            {
                "model_name": "llama2_7b_dense_example",
                "simulation_mode": "online_serving",
                "sys_arch": "co-location",
                "measurement_type": "CUDA_EVENT",
                "workload_profile": "short_single",
                "workload_shape": {
                    "num_requests": 1,
                    "prefill_tokens": 8,
                    "decode_tokens": 2,
                },
            },
            "incomplete golden workload_shape",
        ),
        (
            {
                "model_name": "llama2_7b_dense_example",
                "simulation_mode": "online_serving",
                "sys_arch": "co-location",
                "measurement_type": "CUDA_EVENT",
                "workload_profile": "short_single",
                "workload_shape": {
                    "num_requests": 2,
                    "prefill_tokens": 16,
                    "decode_tokens": 4,
                    "qps": 2.0,
                },
            },
            "golden workload_shape mismatch",
        ),
    ),
)
def test_workload_profile_must_be_explicit_and_supported(
    case_manifest: dict[str, str],
    error_substring: str,
) -> None:
    case = {
        "name": "llama2_7b_dense_example__online_serving__co_location__cuda_event",
        "case_manifest": case_manifest,
        "reference_input_manifest": {
            "profile_files": {
                "attention.csv": {"path": "profiles/llama/attention.csv"},
                "linear_op.csv": {"path": "profiles/llama/linear_op.csv"},
            }
        },
    }

    with pytest.raises(ValueError, match=error_substring):
        build_frontier_command(
            case,
            output_root=Path("baseline/reference_outputs"),
            python_bin="python",
        )


def test_baseline_generation_forwards_workload_profiles_to_case_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_golden_cases(**kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        captured["workload_profiles"] = kwargs["workload_profiles"]
        case = {
            "name": "llama2_7b_dense_example__online_serving__co_location__cuda_event__workload_multi_medium",
            "case_manifest": {
                "model_name": "llama2_7b_dense_example",
                "simulation_mode": "online_serving",
                "sys_arch": "co-location",
                "measurement_type": "CUDA_EVENT",
                "workload_profile": "multi_medium",
                "workload_shape": MULTI_MEDIUM_WORKLOAD_SHAPE,
            },
            "reference_input_manifest": {
                "profile_files": {
                    "attention.csv": {"path": "profiles/llama/attention.csv"},
                    "linear_op.csv": {"path": "profiles/llama/linear_op.csv"},
                }
            },
            "reference_dir": tmp_path / "reference",
            "candidate_dir": tmp_path / "candidate",
        }
        return [case], {"actual": 1}

    monkeypatch.setattr(golden_baseline, "build_golden_cases", fake_build_golden_cases)

    report = run_baseline_generation(
        config_root=tmp_path / "models",
        profile_root=tmp_path / "profiles",
        reference_root=tmp_path / "reference",
        candidate_root=tmp_path / "candidate",
        side="candidate",
        python_bin="python",
        repo_root=tmp_path,
        dry_run=True,
        workload_profiles=("short_single", "multi_medium"),
    )

    assert captured["workload_profiles"] == ("short_single", "multi_medium")
    assert report["status"] == "DRY_RUN"
    assert report["selected_case_count"] == 1


def test_baseline_generation_marks_selected_subset_as_partial_not_final_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = []
    for index in range(2):
        cases.append(
            {
                "name": f"case_{index}",
                "case_manifest": {
                    "model_name": "llama2_7b_dense_example",
                    "simulation_mode": "online_serving",
                    "sys_arch": "co-location",
                    "measurement_type": "CUDA_EVENT",
                    "workload_profile": "multi_medium",
                    "workload_shape": MULTI_MEDIUM_WORKLOAD_SHAPE,
                },
                "reference_input_manifest": {
                    "profile_files": {
                        "attention.csv": {"path": "profiles/llama/attention.csv"},
                        "linear_op.csv": {"path": "profiles/llama/linear_op.csv"},
                    }
                },
                "reference_dir": tmp_path / "reference" / f"case_{index}",
                "candidate_dir": tmp_path / "candidate" / f"case_{index}",
            }
        )

    monkeypatch.setattr(
        golden_baseline,
        "build_golden_cases",
        lambda **_: (cases, {"actual": 2, "expected_effective": 2}),
    )
    monkeypatch.setattr(
        golden_baseline,
        "run_simulation_case",
        lambda case, **_: {
            "case_name": case["name"],
            "returncode": 0,
            "output_dir": str(case["candidate_dir"]),
        },
    )

    report = run_baseline_generation(
        config_root=tmp_path / "models",
        profile_root=tmp_path / "profiles",
        reference_root=tmp_path / "reference",
        candidate_root=tmp_path / "candidate",
        side="candidate",
        python_bin="python",
        repo_root=tmp_path,
        max_cases=1,
        workload_profiles=("multi_medium",),
    )

    assert report["status"] == "PARTIAL_PASS"
    assert report["is_complete_gate"] is False
    assert report["total_case_count"] == 2
    assert report["selected_case_count"] == 1


def test_main_exits_nonzero_for_partial_selected_subset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = [
        {
            "name": "case_0",
            "case_manifest": {
                "model_name": "llama2_7b_dense_example",
                "simulation_mode": "online_serving",
                "sys_arch": "co-location",
                "measurement_type": "CUDA_EVENT",
                "workload_profile": "multi_medium",
                "workload_shape": MULTI_MEDIUM_WORKLOAD_SHAPE,
            },
            "reference_input_manifest": {
                "profile_files": {
                    "attention.csv": {"path": "profiles/llama/attention.csv"},
                    "linear_op.csv": {"path": "profiles/llama/linear_op.csv"},
                }
            },
            "reference_dir": tmp_path / "reference" / "case_0",
            "candidate_dir": tmp_path / "candidate" / "case_0",
        },
        {
            "name": "case_1",
            "case_manifest": {
                "model_name": "llama2_7b_dense_example",
                "simulation_mode": "online_serving",
                "sys_arch": "co-location",
                "measurement_type": "CUDA_EVENT",
                "workload_profile": "multi_medium",
                "workload_shape": MULTI_MEDIUM_WORKLOAD_SHAPE,
            },
            "reference_input_manifest": {
                "profile_files": {
                    "attention.csv": {"path": "profiles/llama/attention.csv"},
                    "linear_op.csv": {"path": "profiles/llama/linear_op.csv"},
                }
            },
            "reference_dir": tmp_path / "reference" / "case_1",
            "candidate_dir": tmp_path / "candidate" / "case_1",
        },
    ]
    output_json = tmp_path / "report.json"

    monkeypatch.setattr(
        golden_baseline,
        "build_golden_cases",
        lambda **_: (cases, {"actual": 2, "expected_effective": 2}),
    )
    monkeypatch.setattr(
        golden_baseline,
        "run_simulation_case",
        lambda case, **_: {
            "case_name": case["name"],
            "returncode": 0,
            "output_dir": str(case["candidate_dir"]),
        },
    )

    exit_code = golden_baseline.main(
        [
            "--reference-root",
            str(tmp_path / "reference"),
            "--candidate-root",
            str(tmp_path / "candidate"),
            "--side",
            "candidate",
            "--repo-root",
            str(tmp_path),
            "--output-json",
            str(output_json),
            "--workload-profiles",
            "multi_medium",
            "--max-cases",
            "1",
        ]
    )

    assert exit_code == 1

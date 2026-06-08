#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILING_DIR = REPO_ROOT / "examples" / "profiling"
EXPECTED_SCRIPTS = (
    PROFILING_DIR / "profile_linear_op.sh",
    PROFILING_DIR / "profile_attention_chunked_prefill.sh",
    PROFILING_DIR / "profile_moe.sh",
    PROFILING_DIR / "smoke_metadata.sh",
    PROFILING_DIR / "smoke_simulator_dense_csv.sh",
    PROFILING_DIR / "smoke_simulator_moe_csv.sh",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_top_level_profiling_examples_exist_and_are_shell_valid() -> None:
    for script in EXPECTED_SCRIPTS:
        assert script.exists(), f"Missing profiling example script: {script}"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"


def test_profiling_examples_cover_three_operator_classes() -> None:
    linear_text = _read(PROFILING_DIR / "profile_linear_op.sh")
    attention_text = _read(PROFILING_DIR / "profile_attention_chunked_prefill.sh")
    moe_text = _read(PROFILING_DIR / "profile_moe.sh")

    assert "frontier.profiling.linear_op.main" in linear_text
    assert "frontier.profiling.attention.main" in attention_text
    assert "frontier.profiling.moe.main" in moe_text
    assert "linear_op.csv" in linear_text
    assert "attention.csv" in attention_text
    assert "moe.csv" in moe_text


def test_attention_recipe_explicitly_profiles_chunked_prefill_state() -> None:
    text = _read(PROFILING_DIR / "profile_attention_chunked_prefill.sh")

    assert "FIXED_CHUNKED_PREFILL_SIZE" in text
    assert "--fixed_chunked_prefill_size" in text
    assert "--enable_chunked_prefill_grid_search" in text
    assert "--profile_only_prefill" in text
    assert "chunked prefill" in text.lower()


def test_profiling_scripts_route_outputs_to_compute_taxonomy() -> None:
    for script in EXPECTED_SCRIPTS:
        text = _read(script)
        assert 'DATA_DIR_BASE="${DATA_DIR_BASE:-$REPO_ROOT/data/profiling}"' in text
        assert "data/profiling/compute" in text
        assert "profiling_outputs" not in text


def test_profile_collection_scripts_reject_missing_cli_values_before_dependency_checks() -> None:
    env = {**os.environ, "PYTHON_BIN": "/definitely/missing/python"}

    for script in (
        PROFILING_DIR / "profile_linear_op.sh",
        PROFILING_DIR / "profile_attention_chunked_prefill.sh",
        PROFILING_DIR / "profile_moe.sh",
    ):
        result = subprocess.run(
            ["bash", str(script), "--model"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 2, script
        assert "--model requires a value" in result.stderr, script


def test_profile_collection_scripts_reject_invalid_bool_env_values() -> None:
    for script in (
        PROFILING_DIR / "profile_linear_op.sh",
        PROFILING_DIR / "profile_attention_chunked_prefill.sh",
        PROFILING_DIR / "profile_moe.sh",
    ):
        result = subprocess.run(
            ["bash", str(script)],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHON_BIN": "/definitely/missing/python", "DRY_RUN": "treu"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 2, script
        assert "DRY_RUN must be true or false" in result.stderr, script

    attention_result = subprocess.run(
        ["bash", str(PROFILING_DIR / "profile_attention_chunked_prefill.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHON_BIN": "/definitely/missing/python",
            "ENABLE_CHUNKED_PREFILL_GRID_SEARCH": "treu",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert attention_result.returncode == 2
    assert "ENABLE_CHUNKED_PREFILL_GRID_SEARCH must be true or false" in attention_result.stderr


def test_profile_collection_scripts_reject_non_numeric_parallel_size_lists() -> None:
    cases = (
        (PROFILING_DIR / "profile_linear_op.sh", {"TP_SIZES": "*"}, "TP_SIZES"),
        (
            PROFILING_DIR / "profile_attention_chunked_prefill.sh",
            {"TP_SIZES": "1 --bad-flag"},
            "TP_SIZES",
        ),
        (PROFILING_DIR / "profile_attention_chunked_prefill.sh", {"PP_SIZES": "*"}, "PP_SIZES"),
        (PROFILING_DIR / "profile_moe.sh", {"TP_SIZES": "*"}, "TP_SIZES"),
        (PROFILING_DIR / "profile_moe.sh", {"EP_SIZES": "1 --bad-flag"}, "EP_SIZES"),
    )

    for script, extra_env, label in cases:
        result = subprocess.run(
            ["bash", str(script), "--dry-run"],
            cwd=REPO_ROOT,
            env={**os.environ, **extra_env},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 2, script
        assert f"{label} must contain positive integer values" in result.stderr, script


def test_downstream_smoke_scripts_consume_explicit_profile_csvs_without_dummy_predictor() -> None:
    dense_text = _read(PROFILING_DIR / "smoke_simulator_dense_csv.sh")
    moe_text = _read(PROFILING_DIR / "smoke_simulator_moe_csv.sh")

    assert "--no-random_forrest_execution_time_predictor_config_enable_dummy_mode" in dense_text
    assert "--no-random_forrest_execution_time_predictor_config_enable_dummy_mode" in moe_text
    assert "--random_forrest_execution_time_predictor_config_linear_op_input_file" in dense_text
    assert "--random_forrest_execution_time_predictor_config_atten_input_file" in dense_text
    assert "--random_forrest_execution_time_predictor_config_moe_input_file" in moe_text
    assert "data/profiling/compute/rtx_pro_6000/qwen2_dense_test/linear_op.csv" in dense_text
    assert "data/profiling/compute/rtx_pro_6000/qwen2_dense_test/attention.csv" in dense_text
    assert "data/profiling/compute/rtx_pro_6000/Qwen3-30B-A3B-tiny/moe.csv" in moe_text
    assert 'CC_BACKEND_CONFIG_TYPE="${CC_BACKEND_CONFIG_TYPE:-astra_sim_analytical}"' in dense_text
    assert 'CC_BACKEND_CONFIG_TYPE="${CC_BACKEND_CONFIG_TYPE:-astra_sim_analytical}"' in moe_text
    assert '--cc_backend_config_type "$CC_BACKEND_CONFIG_TYPE"' in dense_text
    assert '--cc_backend_config_type "$CC_BACKEND_CONFIG_TYPE"' in moe_text


def test_downstream_smoke_defaults_to_release_facing_metrics_output() -> None:
    for script in (
        PROFILING_DIR / "smoke_simulator_dense_csv.sh",
        PROFILING_DIR / "smoke_simulator_moe_csv.sh",
    ):
        text = _read(script)

        assert (
            'METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$REPO_ROOT/outputs/examples/profiling-simulator}"'
            in text
        )
        assert "task_memory" not in text


def test_downstream_smokes_require_csv_paths_to_be_files_before_running_simulator() -> None:
    for script in (
        PROFILING_DIR / "smoke_simulator_dense_csv.sh",
        PROFILING_DIR / "smoke_simulator_moe_csv.sh",
    ):
        result = subprocess.run(
            [
                "bash",
                str(script),
                "--python-bin",
                "/bin/true",
                "--linear-op-csv",
                "python3",
                "--attention-csv",
                "python3",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 2, script
        assert "required profiling CSV is missing" in result.stderr, script


def test_moe_downstream_smoke_uses_routing_mode_matching_checked_in_csv() -> None:
    moe_text = _read(PROFILING_DIR / "smoke_simulator_moe_csv.sh")

    assert "--replica_config_moe_routing_mode" in moe_text
    assert "--replica_config_moe_routing_mode uniform_random" in moe_text
    assert "--replica_config_moe_routing_mode simulation" not in moe_text


def test_smoke_metadata_honors_data_path_and_fails_unknown_args() -> None:
    missing_profile_dir = REPO_ROOT / "tests" / "fixtures" / "__missing_profile_dir__"
    env = {**os.environ, "PYTHON_BIN": "python3"}

    data_path_result = subprocess.run(
        [
            "bash",
            str(PROFILING_DIR / "smoke_metadata.sh"),
            "--data_path",
            str(missing_profile_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert data_path_result.returncode == 2
    assert str(missing_profile_dir) in data_path_result.stderr

    unknown_result = subprocess.run(
        ["bash", str(PROFILING_DIR / "smoke_metadata.sh"), "--unknown-flag"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert unknown_result.returncode == 2
    assert "unknown" in unknown_result.stderr.lower()


def test_smoke_scripts_reject_unknown_cli_args_before_dependency_checks() -> None:
    env = {**os.environ, "PYTHON_BIN": "/definitely/missing/python"}

    for script in (
        PROFILING_DIR / "smoke_metadata.sh",
        PROFILING_DIR / "smoke_simulator_dense_csv.sh",
        PROFILING_DIR / "smoke_simulator_moe_csv.sh",
    ):
        result = subprocess.run(
            ["bash", str(script), "--unknown-flag"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 2, script
        assert "unknown" in result.stderr.lower(), script


def test_profiling_readme_documents_migration_scope_and_legacy_path() -> None:
    readme = _read(PROFILING_DIR / "README.md")

    assert "## Modification History" in readme
    assert "linear_op" in readme
    assert "attention" in readme
    assert "moe" in readme
    assert "chunked prefill" in readme.lower()
    assert "data/profiling/compute" in readme
    assert "frontier/profiling/example" in readme
    assert "non-destructive" in readme.lower()
    assert "PROFILE_METHOD=cuda_event" in readme
    assert "wrapper default" in readme
    assert "record_function" in readme
    assert "checked-in CSV" in readme

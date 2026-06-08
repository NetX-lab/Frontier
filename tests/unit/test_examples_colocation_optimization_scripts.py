from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COLOCATION_DIR = REPO_ROOT / "examples" / "architecture" / "co-location"

BASELINE_SCRIPTS = (
    COLOCATION_DIR / "dense_model_basic.sh",
    COLOCATION_DIR / "moe_model_basic.sh",
    COLOCATION_DIR / "thinking_mode_basic.sh",
)
ADVANCED_SCRIPTS = (
    COLOCATION_DIR / "moe_spec_dec.sh",
    COLOCATION_DIR / "moe_prefix_caching.sh",
)
ALL_SCRIPTS = (*BASELINE_SCRIPTS, *ADVANCED_SCRIPTS)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_colocation_examples_include_expected_script_set() -> None:
    for script in ALL_SCRIPTS:
        assert script.exists(), f"Missing co-location example script: {script}"


def test_colocation_examples_are_shell_syntax_valid() -> None:
    for script in ALL_SCRIPTS:
        result = subprocess.run(
            ["bash", "-n", str(script)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"


def test_baseline_colocation_scripts_enable_runtime_optimizations_by_default() -> None:
    for script in BASELINE_SCRIPTS:
        text = _read(script)
        assert 'DECODE_CUDA_GRAPH_MODE="${DECODE_CUDA_GRAPH_MODE:-full_decode_only}"' in text, script
        assert 'ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"' in text, script
        assert '--decode_cuda_graph_mode "$DECODE_CUDA_GRAPH_MODE"' in text, script
        assert '--vllm_v1_scheduler_config_enable_chunked_prefill' in text, script
        assert '--use_cuda_graph' not in text, script


def test_colocation_scripts_allow_runtime_overrides_and_emit_metrics() -> None:
    for script in ALL_SCRIPTS:
        text = _read(script)
        assert 'CC_BACKEND_CONFIG_TYPE="${CC_BACKEND_CONFIG_TYPE:-astra_sim_analytical}"' in text, script
        assert '--cc_backend_config_type "$CC_BACKEND_CONFIG_TYPE"' in text, script
        assert 'METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$REPO_ROOT/outputs/examples/co-location}"' in text, script
        assert 'RUN_ID="${RUN_ID:-' in text, script
        assert '--metrics_config_output_dir "$METRICS_OUTPUT_DIR"' in text, script
        assert '--metrics_config_run_id "$RUN_ID"' in text, script
        assert '--no-metrics_config_store_plots' in text, script
        assert 'if [ "$#" -gt 0 ]; then' in text, script
        assert 'CMD+=("$@")' in text, script

    for script in (*BASELINE_SCRIPTS, COLOCATION_DIR / "moe_spec_dec.sh"):
        assert 'NUM_REQUESTS="${NUM_REQUESTS:-' in _read(script), script


def test_colocation_scripts_fail_fast_when_python_is_missing() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHON_BIN": "/definitely/missing/python",
    }

    for script in ALL_SCRIPTS:
        result = subprocess.run(
            ["bash", str(script)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 2, script
        assert "PYTHON_BIN is not executable or not on PATH" in result.stderr, script


def test_colocation_scripts_fail_fast_on_invalid_dummy_mode_boolean() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHON_BIN": "/definitely/missing/python",
        "ENABLE_DUMMY_MODE": "treu",
    }

    for script in ALL_SCRIPTS:
        result = subprocess.run(
            ["bash", str(script)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 2, script
        assert "ENABLE_DUMMY_MODE must be true or false" in result.stderr, script


def test_colocation_scripts_report_frontier_command_failures() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHON_BIN": "/bin/false",
    }

    for script in ALL_SCRIPTS:
        result = subprocess.run(
            ["bash", str(script)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 1, script
        assert "Simulation failed (exit code: 1)" in result.stderr, script


def test_advanced_moe_recipes_fail_fast_on_invalid_chunked_prefill_boolean() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHON_BIN": "/definitely/missing/python",
        "ENABLE_CHUNKED_PREFILL": "treu",
    }

    for script in ADVANCED_SCRIPTS:
        result = subprocess.run(
            ["bash", str(script)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 2, script
        assert "ENABLE_CHUNKED_PREFILL must be true or false" in result.stderr, script


def test_moe_spec_decode_recipe_fails_fast_on_invalid_mtp_integer_values() -> None:
    script = COLOCATION_DIR / "moe_spec_dec.sh"
    invalid_cases = (
        {"MTP_N_PREDICT": "abc", "MTP_NUM_LAYERS": "1", "expected": "MTP_N_PREDICT"},
        {"MTP_N_PREDICT": "1", "MTP_NUM_LAYERS": "abc", "expected": "MTP_NUM_LAYERS"},
        {"MTP_N_PREDICT": "0", "MTP_NUM_LAYERS": "1", "expected": "requires MTP_N_PREDICT>0"},
        {"MTP_N_PREDICT": "1", "MTP_NUM_LAYERS": "0", "expected": "requires MTP_N_PREDICT>0"},
    )

    for case in invalid_cases:
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHON_BIN": ":",
            "SPEC_METHOD": "qwen3_next_mtp",
            "MTP_N_PREDICT": case["MTP_N_PREDICT"],
            "MTP_NUM_LAYERS": case["MTP_NUM_LAYERS"],
        }
        result = subprocess.run(
            ["bash", str(script)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 2, case
        assert case["expected"] in result.stderr, case
        assert "integer expression expected" not in result.stderr, case
        assert "Simulation completed successfully." not in result.stdout, case


def test_moe_spec_decode_recipe_uses_speculative_or_mtp_controls_without_cuda_graph_conflict() -> None:
    text = _read(COLOCATION_DIR / "moe_spec_dec.sh")

    assert 'DECODE_CUDA_GRAPH_MODE="${DECODE_CUDA_GRAPH_MODE:-none}"' in text
    assert "require_non_negative_integer" in text
    assert "require_positive_integer" in text
    assert '--speculative_decoding_config_enabled' in text
    assert '--speculative_decoding_config_method "$SPEC_METHOD"' in text
    assert '--speculative_decoding_config_num_speculative_tokens "$NUM_SPECULATIVE_TOKENS"' in text
    assert '--speculative_decoding_config_committed_tokens_per_iteration "$COMMITTED_TOKENS_PER_ITERATION"' in text
    assert '--speculative_decoding_config_mtp_n_predict "$MTP_N_PREDICT"' in text
    assert '--speculative_decoding_config_mtp_num_layers "$MTP_NUM_LAYERS"' in text
    assert '--allow_spec_decode_cuda_graph_diagnostic' not in text
    assert '--vllm_v1_scheduler_config_enable_prefix_caching' not in text


def test_moe_prefix_cache_recipe_uses_shared_prefix_trace_and_no_spec_decode() -> None:
    text = _read(COLOCATION_DIR / "moe_prefix_caching.sh")

    assert 'TRACE_FILE="${TRACE_FILE:-$REPO_ROOT/examples/fixtures/prefix_cache_shared_session_trace.csv}"' in text
    assert "tests/integration/fixtures/prefix_cache_shared_session_trace.csv" not in text
    assert 'EXPECTED_TRACE_REQUESTS="${EXPECTED_TRACE_REQUESTS:-2}"' in text
    assert 'NUM_REQUESTS=' not in text
    assert "Expected Trace Shape: requests=$EXPECTED_TRACE_REQUESTS" in text
    assert '--request_generator_config_type trace_replay' in text
    assert '--trace_request_generator_config_trace_file "$TRACE_FILE"' in text
    assert '--vllm_v1_scheduler_config_enable_prefix_caching' in text
    assert '--cluster_scheduler_config_type sticky_round_robin' in text
    assert '--speculative_decoding_config_enabled' not in text


def test_public_prefix_cache_trace_fixture_exists() -> None:
    fixture = REPO_ROOT / "examples" / "fixtures" / "prefix_cache_shared_session_trace.csv"

    assert fixture.exists()
    text = fixture.read_text(encoding="utf-8")
    assert "block_hash_ids" in text
    assert text.count("\n") >= 2

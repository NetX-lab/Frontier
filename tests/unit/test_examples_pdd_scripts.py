from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PDD_DIR = REPO_ROOT / "examples" / "architecture" / "pdd"

OFFLINE_SCRIPTS = (
    PDD_DIR / "offline" / "dense_model_basic.sh",
    PDD_DIR / "offline" / "moe_model_basic.sh",
    PDD_DIR / "offline" / "thinking_mode_basic.sh",
    PDD_DIR / "offline" / "moe_spec_dec.sh",
    PDD_DIR / "offline" / "moe_prefix_caching.sh",
)
ONLINE_SCRIPTS = (
    PDD_DIR / "online" / "dense_model_basic_online.sh",
    PDD_DIR / "online" / "moe_model_basic_online.sh",
    PDD_DIR / "online" / "thinking_mode_basic_online.sh",
    PDD_DIR / "online" / "moe_spec_dec_online.sh",
    PDD_DIR / "online" / "moe_prefix_caching_online.sh",
)
ALL_SCRIPTS = (*OFFLINE_SCRIPTS, *ONLINE_SCRIPTS)
COMPATIBILITY_SCRIPTS = (PDD_DIR / "dense_model_basic.sh",)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_pdd_examples_include_expected_offline_online_script_set() -> None:
    for script in ALL_SCRIPTS:
        assert script.exists(), f"Missing PDD example script: {script}"

    run_all = PDD_DIR / "run_all.sh"
    assert run_all.exists()
    run_all_text = _read(run_all)
    for script in ALL_SCRIPTS:
        relative = script.relative_to(PDD_DIR).as_posix()
        assert f'"{relative}"' in run_all_text


def test_pdd_examples_are_shell_syntax_valid() -> None:
    for script in (*ALL_SCRIPTS, *COMPATIBILITY_SCRIPTS, PDD_DIR / "run_all.sh"):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"


def test_pdd_dense_compatibility_entrypoint_forwards_to_offline_script() -> None:
    script = PDD_DIR / "dense_model_basic.sh"
    text = _read(script)

    assert script.exists()
    assert "Compatibility entrypoint" in text
    assert 'exec bash "$SCRIPT_DIR/offline/dense_model_basic.sh" "$@"' in text
    assert ("examples/architecture/pd" + "-only") not in text


def test_pdd_scripts_use_release_supported_sequential_pd_path_and_metrics() -> None:
    for script in ALL_SCRIPTS:
        text = _read(script)
        assert 'SYS_ARCH="${SYS_ARCH:-pd-disaggregation}"' in text, script
        assert '--sys_arch "$SYS_ARCH"' in text, script
        assert '--no-enable_parallel_clusters' in text, script
        assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in text, script
        assert 'METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$REPO_ROOT/outputs/examples/pdd/' in text, script
        assert 'RUN_ID="${RUN_ID:-' in text, script
        assert '--metrics_config_output_dir "$METRICS_OUTPUT_DIR"' in text, script
        assert '--metrics_config_run_id "$RUN_ID"' in text, script
        assert '--metrics_config_write_metrics' in text, script
        assert '--metrics_config_store_request_metrics' in text, script
        assert 'if [ "$#" -gt 0 ]; then' in text, script
        assert 'CMD+=("$@")' in text, script


def test_pdd_scripts_configure_prefill_decode_clusters_explicitly() -> None:
    for script in ALL_SCRIPTS:
        text = _read(script)
        assert '--cluster_config_prefill_cluster_num_replicas "$PREFILL_REPLICAS"' in text, script
        assert '--cluster_config_decode_cluster_num_replicas "$DECODE_REPLICAS"' in text, script
        assert '--cluster_config_prefill_replica_config_attn_tensor_parallel_size "$PREFILL_ATTN_TP"' in text, script
        assert '--cluster_config_decode_replica_config_attn_tensor_parallel_size "$DECODE_ATTN_TP"' in text, script
        assert '--cluster_config_prefill_replica_config_memory_margin_fraction "$PREFILL_MEMORY_MARGIN_FRACTION"' in text, script
        assert '--cluster_config_decode_replica_config_memory_margin_fraction "$DECODE_MEMORY_MARGIN_FRACTION"' in text, script
        assert '--cc_backend_config_type analytical' in text, script


def test_pdd_moe_scripts_fail_fast_on_invalid_shared_domain_parallelism() -> None:
    moe_scripts = [script for script in ALL_SCRIPTS if "moe" in script.name]
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHON_BIN": ":",
        "PREFILL_ATTN_TP": "2",
        "PREFILL_ATTN_DP": "1",
        "PREFILL_MOE_TP": "1",
        "PREFILL_MOE_EP": "1",
    }

    for script in moe_scripts:
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
        assert "PREFILL_ATTN_TP * PREFILL_ATTN_DP == PREFILL_MOE_TP * PREFILL_MOE_EP" in result.stderr, script


def test_pdd_spec_decode_scripts_use_spec_controls_without_prefix_cache_conflict() -> None:
    for script in (path for path in ALL_SCRIPTS if "spec_dec" in path.name):
        text = _read(script)
        assert 'DECODE_CUDA_GRAPH_MODE="${DECODE_CUDA_GRAPH_MODE:-none}"' in text
        assert '--speculative_decoding_config_enabled' in text
        assert '--speculative_decoding_config_method "$SPEC_METHOD"' in text
        assert '--speculative_decoding_config_mtp_n_predict "$MTP_N_PREDICT"' in text
        assert '--speculative_decoding_config_mtp_num_layers "$MTP_NUM_LAYERS"' in text
        assert '--vllm_v1_scheduler_config_enable_prefix_caching' not in text


def test_pdd_prefix_cache_scripts_use_sticky_scheduler_and_no_spec_decode() -> None:
    for script in (path for path in ALL_SCRIPTS if "prefix_caching" in path.name):
        text = _read(script)
        assert 'TRACE_FILE="${TRACE_FILE:-$REPO_ROOT/examples/fixtures/prefix_cache_shared_session_trace.csv}"' in text
        assert '--cluster_scheduler_config_type sticky_round_robin' in text
        assert '--request_generator_config_type trace_replay' in text
        assert '--trace_request_generator_config_trace_file "$TRACE_FILE"' in text
        assert '--vllm_v1_scheduler_config_enable_prefix_caching' in text
        assert '--speculative_decoding_config_enabled' not in text

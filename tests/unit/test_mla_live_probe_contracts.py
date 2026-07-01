#!/usr/bin/env python3
"""Unit tests for the DeepSeek-V2 MLA FlashInfer live-probe contract."""

import ast
import importlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.mla_h800_fixture import h800_mla_mixed_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = (
    PROJECT_ROOT
    / "tests"
    / "analysis"
    / "mla_deepseek_v2"
    / "run_vllm_flashinfer_mla_live_probe.sh"
)
VALIDATOR = (
    PROJECT_ROOT
    / "tests"
    / "analysis"
    / "mla_deepseek_v2"
    / "validate_flashinfer_mla_live_probe.py"
)
def test_mla_validator_required_scopes_are_declared_from_shared_mla_mapping() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.setdefault(node.module, set()).update(
                alias.name for alias in node.names
            )

    literal_required_scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SCOPES"
            for target in node.targets
        )
        and isinstance(node.value, ast.Tuple)
    ]
    assert literal_required_scopes == []
    assert "LATENT_MLA_ATTENTION_FAMILY" in imports.get(
        "frontier.attention.families", set()
    )
    assert "get_profiling_metric_names" in imports.get(
        "frontier.attention.profiling_mapping", set()
    )

    from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY
    from frontier.attention.profiling_mapping import get_profiling_metric_names

    validator = importlib.import_module(
        "tests.analysis.mla_deepseek_v2.validate_flashinfer_mla_live_probe"
    )
    assert validator.REQUIRED_SCOPES == get_profiling_metric_names(
        LATENT_MLA_ATTENTION_FAMILY
    )


def test_mla_flashinfer_probe_script_uses_required_backend_and_env() -> None:
    assert PROBE_SCRIPT.exists(), f"Missing MLA live probe script: {PROBE_SCRIPT}"
    source = PROBE_SCRIPT.read_text(encoding="utf-8")

    assert 'PYTHON_BIN="${PYTHON_BIN:-/local/ycfeng/anaconda3/envs/frontier/bin/python}"' in source
    assert 'FLASHINFER_PYTHON_EXPECTED_VERSION="${FLASHINFER_PYTHON_EXPECTED_VERSION:-0.3.1.post1}"' in source
    assert 'VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER_MLA}"' in source
    assert 'EXPECTED_BACKEND="${EXPECTED_BACKEND:-FLASHINFER_MLA}"' in source
    assert "VLLM_FRONTIER_INSTRUMENTATION=1" in source
    assert "VLLM_FRONTIER_RUNTIME_META_ENABLED=1" in source
    assert "VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH" in source
    assert "attn_mla_kv_cache_save,attn_mla_prefill_kv_up_proj,attn_mla_prefill,attn_mla_decode_q_latent_proj,attn_mla_decode,attn_mla_v_up_proj" in source
    assert "flashinfer_version = version(\"flashinfer-python\")" in source
    assert 'export PYTHONPATH="$PROJECT_ROOT:$VLLM_ROOT:${PYTHONPATH:-}"' in source
    assert "validate_flashinfer_mla_live_probe.py" in source
    assert "RUN_FRONTIER_IMPORT_VALIDATION" in source
    assert "FRONTIER_IMPORT_OUTPUT_DIR" in source
    assert "FRONTIER_IMPORT_SIDECAR" in source
    assert "frontier.profiling.attention.main" in source
    assert "--vllm_mla_cuda_op_log" in source
    assert "--frontier-import-sidecar" in source


def test_mla_flashinfer_probe_script_derives_frontier_import_method_from_timing_mode() -> None:
    assert PROBE_SCRIPT.exists(), f"Missing MLA live probe script: {PROBE_SCRIPT}"
    source = PROBE_SCRIPT.read_text(encoding="utf-8")

    assert (
        'FRONTIER_IMPORT_PROFILE_METHOD="${FRONTIER_IMPORT_PROFILE_METHOD:-'
        '${VLLM_FRONTIER_OP_TIMING_MODE:-record_function}}"'
    ) in source
    assert 'case "${VLLM_FRONTIER_OP_TIMING_MODE:-record_function}" in' in source
    assert 'VLLM_FRONTIER_TIMING_FAMILY="KERNEL_ONLY"' in source
    assert 'VLLM_FRONTIER_TIMING_FAMILY="CUDA_EVENT"' in source
    assert (
        "ERROR: VLLM_FRONTIER_OP_TIMING_MODE must be one of "
        "record_function or cuda_event."
    ) in source
    assert 'case "$FRONTIER_IMPORT_PROFILE_METHOD" in' in source
    assert "record_function|kernel_only)" in source
    assert 'FRONTIER_IMPORT_MEASUREMENT_FAMILY="KERNEL_ONLY"' in source
    assert "cuda|cuda_event)" in source
    assert 'FRONTIER_IMPORT_MEASUREMENT_FAMILY="CUDA_EVENT"' in source
    assert (
        "ERROR: FRONTIER_IMPORT_PROFILE_METHOD must be one of "
        "record_function, kernel_only, cuda, or cuda_event."
    ) in source
    assert (
        'if [[ "$FRONTIER_IMPORT_MEASUREMENT_FAMILY" != '
        '"$VLLM_FRONTIER_TIMING_FAMILY" ]]'
    ) in source
    assert (
        "ERROR: FRONTIER_IMPORT_PROFILE_METHOD must match "
        "VLLM_FRONTIER_OP_TIMING_MODE measurement semantics."
    ) in source
    assert '--profile_method "$FRONTIER_IMPORT_PROFILE_METHOD"' in source
    assert "--profile_method record_function" not in source


def test_mla_flashinfer_probe_script_routes_frontier_import_outputs_by_profile_method() -> None:
    assert PROBE_SCRIPT.exists(), f"Missing MLA live probe script: {PROBE_SCRIPT}"
    source = PROBE_SCRIPT.read_text(encoding="utf-8")

    assert 'FRONTIER_IMPORT_MODEL_DIR="$FRONTIER_IMPORT_OUTPUT_DIR/compute/h100/deepseek-ai/DeepSeek-V2-Lite"' in source
    assert 'FRONTIER_IMPORT_PROFILE_CSV="$FRONTIER_IMPORT_MODEL_DIR/attention.csv"' in source
    assert 'FRONTIER_IMPORT_SIDECAR="$FRONTIER_IMPORT_MODEL_DIR/attention_vllm_mla_groundtruth_comparison.csv"' in source
    assert 'FRONTIER_IMPORT_PROFILE_CSV="$FRONTIER_IMPORT_MODEL_DIR/attention_kernel_only.csv"' in source
    assert 'FRONTIER_IMPORT_SIDECAR="$FRONTIER_IMPORT_MODEL_DIR/attention_kernel_only_vllm_mla_groundtruth_comparison.csv"' in source


def test_mla_flashinfer_probe_script_runs_real_vllm_measure_ttft() -> None:
    assert PROBE_SCRIPT.exists(), f"Missing MLA live probe script: {PROBE_SCRIPT}"
    source = PROBE_SCRIPT.read_text(encoding="utf-8")

    assert 'if [[ "$VLLM_ATTENTION_BACKEND" != "FLASHINFER_MLA" ]]' in source
    assert 'examples/offline_inference/measure_ttft.py' in source
    assert '"$PYTHON_BIN" examples/offline_inference/measure_ttft.py' in source
    assert '--model "$MODEL_RUNTIME_DIR"' in source
    assert '--load-format dummy' in source
    assert '--vocab-size 102400' in source
    assert '--enforce-eager "$ENFORCE_EAGER"' in source
    assert '--results-csv "$RESULT_CSV"' in source
    assert '--results-json "$RESULT_JSON"' in source
    assert 'GPU_DEVICES="${GPU_DEVICES:-0}"' in source
    assert 'export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"' in source
    assert 'export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"' in source
    assert 'export VLLM_FRONTIER_OP_TIMING_MODE="${VLLM_FRONTIER_OP_TIMING_MODE:-record_function}"' in source
    assert 'export VLLM_FRONTIER_OP_AGG_MODE="${VLLM_FRONTIER_OP_AGG_MODE:-per_scope}"' in source
    assert 'print("use_mla=True")' not in source
    assert 'print("block_size=64")' not in source




def test_mla_probe_writes_deepseek_v2_mla_dummy_config() -> None:
    assert PROBE_SCRIPT.exists(), f"Missing MLA live probe script: {PROBE_SCRIPT}"
    source = PROBE_SCRIPT.read_text(encoding="utf-8")

    assert 'if [[ ! -f "$MODEL_RUNTIME_DIR/config.json" ]]' not in source
    assert 'cat > "$MODEL_RUNTIME_DIR/config.json"' in source
    for snippet in [
        '"architectures": ["DeepseekV2ForCausalLM"]',
        '"model_type": "deepseek_v2"',
        '"hidden_size": 5120',
        '"num_hidden_layers": 1',
        '"num_attention_heads": 128',
        '"num_key_value_heads": 128',
        '"n_routed_experts": 1',
        '"n_shared_experts": 1',
        '"num_experts_per_tok": 1',
        '"q_lora_rank": 1536',
        '"kv_lora_rank": 512',
        '"qk_nope_head_dim": 128',
        '"qk_rope_head_dim": 64',
        '"rope_scaling": {',
        '"factor": 40',
        '"type": "yarn"',
        '"original_max_position_embeddings": 4096',
        '"v_head_dim": 128',
        '"torch_dtype": "bfloat16"',
    ]:
        assert snippet in source
    assert '"n_shared_experts": null' not in source
    assert '"rope_scaling": null' not in source


def test_mla_validator_accepts_flashinfer_mla_runtime_meta(tmp_path: Path) -> None:
    assert VALIDATOR.exists(), f"Missing MLA live probe validator: {VALIDATOR}"
    cuda_log = tmp_path / "cuda_ops.jsonl"
    run_log = tmp_path / "run.log"
    scopes = [
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ]
    rows = []
    for scope in scopes:
        rows.append(
            {
                "batch_id": 0,
                "op_name": scope,
                "cuda_time_ms": 0.0064,
                "count": 1,
                "meta": {
                    "attention_backend": "FLASHINFER_MLA",
                    "use_mla": True,
                    "runtime_num_kv_heads": 1,
                    "runtime_head_size": 576,
                    "kv_lora_rank": 512,
                    "qk_nope_head_dim": 128,
                    "qk_rope_head_dim": 64,
                    "qk_head_dim": 192,
                    "v_head_dim": 128,
                    "block_size": 64,
                    "kv_cache_dtype": "auto",
                    "calculate_kv_scales": False,
                    "attn_module_sliding_window": None,
                    "alibi_slopes": None,
                    "logits_soft_cap": None,
                    "attn_type": "decoder",
                    "max_seqlen_q": 1,
                    "max_seqlen_k": 65,
                    "num_actual_tokens": 1,
                },
            }
        )
    cuda_log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    run_log.write_text(
        "Initializing a V1 LLM engine\n"
        "Using FlashInfer MLA backend on V1 engine.\n"
        "use_mla=True\n"
        "block_size=64\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--cuda-op-log",
            str(cuda_log),
            "--run-log",
            str(run_log),
            "--expected-backend",
            "FLASHINFER_MLA",
            "--expected-runtime-head-size",
            "576",
            "--expected-runtime-kv-heads",
            "1",
            "--expected-block-size",
            "64",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert "FlashInfer MLA live probe validation passed." in completed.stdout
    for scope in scopes:
        assert f"{scope}: rows=1" in completed.stdout


def test_mla_validator_rejects_dense_kv_metadata(tmp_path: Path) -> None:
    assert VALIDATOR.exists(), f"Missing MLA live probe validator: {VALIDATOR}"
    cuda_log = tmp_path / "cuda_ops.jsonl"
    run_log = tmp_path / "run.log"
    scopes = [
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ]
    rows = []
    for scope in scopes:
        rows.append(
            {
                "batch_id": 0,
                "op_name": scope,
                "cuda_time_ms": 0.0064,
                "count": 1,
                "meta": {
                    "attention_backend": "FLASHINFER_MLA",
                    "use_mla": True,
                    "runtime_num_kv_heads": 128,
                    "runtime_head_size": 128,
                    "kv_lora_rank": 512,
                    "qk_nope_head_dim": 128,
                    "qk_rope_head_dim": 64,
                    "qk_head_dim": 192,
                    "v_head_dim": 128,
                    "block_size": 64,
                    "kv_cache_dtype": "auto",
                    "calculate_kv_scales": False,
                    "attn_module_sliding_window": None,
                    "alibi_slopes": None,
                    "logits_soft_cap": None,
                    "attn_type": "DECODER",
                },
            }
        )
    cuda_log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    run_log.write_text("VLLM_ATTENTION_BACKEND=FLASHINFER_MLA\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--cuda-op-log",
            str(cuda_log),
            "--run-log",
            str(run_log),
            "--expected-backend",
            "FLASHINFER_MLA",
            "--expected-runtime-head-size",
            "576",
            "--expected-runtime-kv-heads",
            "1",
            "--expected-block-size",
            "64",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert "Unexpected runtime_num_kv_heads" in completed.stderr


def _mla_runtime_meta() -> dict[str, object]:
    return {
        "attention_backend": "FLASHINFER_MLA",
        "use_mla": True,
        "runtime_num_kv_heads": 1,
        "runtime_head_size": 576,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 192,
        "v_head_dim": 128,
        "block_size": 64,
        "kv_cache_dtype": "auto",
        "calculate_kv_scales": False,
        "attn_module_sliding_window": None,
        "alibi_slopes": None,
        "logits_soft_cap": None,
        "attn_type": "decoder",
        "max_seqlen_q": 1,
        "max_seqlen_k": 65,
        "num_actual_tokens": 1,
    }


def _write_mla_cuda_op_log(path: Path) -> list[str]:
    scopes = [
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ]
    rows = []
    for idx, scope in enumerate(scopes, 1):
        for sample_offset in (0.0, 0.002):
            rows.append(
                {
                    "batch_id": 0,
                    "batch_size": 1,
                    "batch_num_tokens": 1,
                    "batch_num_prefill_tokens": 0,
                    "batch_num_decode_tokens": 1,
                    "batch_request_num_tokens": [1],
                    "op_name": scope,
                    "cuda_time_ms": float(idx) / 100.0 + sample_offset,
                    "count": 1,
                    "meta": _mla_runtime_meta(),
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return scopes


def _write_matching_mla_import_sidecar(path: Path, scopes: list[str]) -> None:
    sidecar_rows = []
    for idx, scope in enumerate(scopes, 1):
        median_ms = float(idx) / 100.0 + 0.001
        sidecar_rows.append(
            {
                "scope": scope,
                "vllm_cuda_time_ms": median_ms,
                "frontier_profile_median_ms": median_ms,
                "absolute_error_ms": 0.0,
                "relative_error_pct": 0.0,
                "vllm_sample_count": 2,
            }
        )
    pd.DataFrame(sidecar_rows).to_csv(path, index=False)


def test_mla_validator_accepts_frontier_import_sidecar_numeric_groundtruth(
    tmp_path: Path,
) -> None:
    from tests.analysis.mla_deepseek_v2.validate_flashinfer_mla_live_probe import (
        validate_frontier_import_sidecar_against_vllm_log,
    )

    cuda_log = tmp_path / "cuda_ops.jsonl"
    sidecar = tmp_path / "attention_vllm_mla_groundtruth_comparison.csv"
    scopes = _write_mla_cuda_op_log(cuda_log)
    _write_matching_mla_import_sidecar(sidecar, scopes)

    summary = validate_frontier_import_sidecar_against_vllm_log(
        cuda_op_log=cuda_log,
        sidecar_csv=sidecar,
        max_absolute_error_ms=0.0,
        max_relative_error_pct=0.0,
    )

    assert summary.scope_count == 6
    assert summary.sample_count_sum == 12
    assert summary.max_absolute_error_ms == pytest.approx(0.0)
    assert summary.max_relative_error_pct == pytest.approx(0.0)
    assert summary.decode_vllm_ms == pytest.approx(0.051)
    assert summary.decode_frontier_ms == pytest.approx(0.051)


def test_mla_validator_rejects_sidecar_vllm_median_drift(
    tmp_path: Path,
) -> None:
    from tests.analysis.mla_deepseek_v2.validate_flashinfer_mla_live_probe import (
        validate_frontier_import_sidecar_against_vllm_log,
    )

    cuda_log = tmp_path / "cuda_ops.jsonl"
    sidecar = tmp_path / "attention_kernel_only_vllm_mla_groundtruth_comparison.csv"
    scopes = _write_mla_cuda_op_log(cuda_log)
    _write_matching_mla_import_sidecar(sidecar, scopes)

    df = pd.read_csv(sidecar)
    df.loc[df["scope"] == "attn_mla_decode", "vllm_cuda_time_ms"] = 0.050
    df.to_csv(sidecar, index=False)

    with pytest.raises(ValueError, match="vLLM median mismatch"):
        validate_frontier_import_sidecar_against_vllm_log(
            cuda_op_log=cuda_log,
            sidecar_csv=sidecar,
            max_absolute_error_ms=0.0,
            max_relative_error_pct=0.0,
        )


def test_mla_validator_rejects_sidecar_error_above_threshold(
    tmp_path: Path,
) -> None:
    from tests.analysis.mla_deepseek_v2.validate_flashinfer_mla_live_probe import (
        validate_frontier_import_sidecar_against_vllm_log,
    )

    cuda_log = tmp_path / "cuda_ops.jsonl"
    sidecar = tmp_path / "attention_vllm_mla_groundtruth_comparison.csv"
    scopes = _write_mla_cuda_op_log(cuda_log)
    _write_matching_mla_import_sidecar(sidecar, scopes)

    df = pd.read_csv(sidecar)
    decode_mask = df["scope"] == "attn_mla_decode"
    df.loc[decode_mask, "frontier_profile_median_ms"] = 0.056
    df.loc[decode_mask, "absolute_error_ms"] = 0.005
    df.loc[decode_mask, "relative_error_pct"] = 9.80392156862745
    df.to_csv(sidecar, index=False)

    with pytest.raises(ValueError, match="exceeds allowed threshold"):
        validate_frontier_import_sidecar_against_vllm_log(
            cuda_op_log=cuda_log,
            sidecar_csv=sidecar,
            max_absolute_error_ms=0.001,
            max_relative_error_pct=1.0,
        )


def test_mla_validator_rejects_sidecar_sample_count_drift(
    tmp_path: Path,
) -> None:
    from tests.analysis.mla_deepseek_v2.validate_flashinfer_mla_live_probe import (
        validate_frontier_import_sidecar_against_vllm_log,
    )

    cuda_log = tmp_path / "cuda_ops.jsonl"
    sidecar = tmp_path / "attention_vllm_mla_groundtruth_comparison.csv"
    scopes = _write_mla_cuda_op_log(cuda_log)
    _write_matching_mla_import_sidecar(sidecar, scopes)

    df = pd.read_csv(sidecar)
    df.loc[df["scope"] == "attn_mla_decode", "vllm_sample_count"] = 1
    df.to_csv(sidecar, index=False)

    with pytest.raises(ValueError, match="sample count mismatch"):
        validate_frontier_import_sidecar_against_vllm_log(
            cuda_op_log=cuda_log,
            sidecar_csv=sidecar,
            max_absolute_error_ms=0.0,
            max_relative_error_pct=0.0,
        )


def test_mla_validator_rejects_sidecar_non_finite_numeric_values(
    tmp_path: Path,
) -> None:
    from tests.analysis.mla_deepseek_v2.validate_flashinfer_mla_live_probe import (
        validate_frontier_import_sidecar_against_vllm_log,
    )

    cuda_log = tmp_path / "cuda_ops.jsonl"
    sidecar = tmp_path / "attention_vllm_mla_groundtruth_comparison.csv"
    scopes = _write_mla_cuda_op_log(cuda_log)
    _write_matching_mla_import_sidecar(sidecar, scopes)

    df = pd.read_csv(sidecar)
    df.loc[df["scope"] == "attn_mla_decode", "frontier_profile_median_ms"] = float("nan")
    df.to_csv(sidecar, index=False)

    with pytest.raises(ValueError, match="Non-finite frontier_profile_median_ms"):
        validate_frontier_import_sidecar_against_vllm_log(
            cuda_op_log=cuda_log,
            sidecar_csv=sidecar,
            max_absolute_error_ms=0.0,
            max_relative_error_pct=0.0,
        )


def test_mla_validator_accepts_row_aware_sparse_sidecar_for_h800_mixed_rows(
    tmp_path: Path,
) -> None:
    from frontier.profiling.attention.vllm_mla_profile_importer import (
        build_frontier_mla_profile_dataframe,
        build_mla_profile_groundtruth_comparison,
    )
    from tests.analysis.mla_deepseek_v2.validate_flashinfer_mla_live_probe import (
        validate_frontier_import_sidecar_against_vllm_log,
    )

    rows = h800_mla_mixed_rows()
    cuda_log = tmp_path / "cuda_ops.jsonl"
    sidecar = tmp_path / "attention_kernel_only_vllm_mla_groundtruth_comparison.csv"
    cuda_log.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    df = build_frontier_mla_profile_dataframe(
        rows,
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="KERNEL_ONLY",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )
    build_mla_profile_groundtruth_comparison(rows, df).to_csv(sidecar, index=False)

    summary = validate_frontier_import_sidecar_against_vllm_log(
        cuda_op_log=cuda_log,
        sidecar_csv=sidecar,
        max_absolute_error_ms=0.0,
        max_relative_error_pct=0.0,
    )

    assert summary.scope_count == 13
    assert summary.sample_count_sum == 13
    assert summary.max_absolute_error_ms == pytest.approx(0.0)
    assert summary.max_relative_error_pct == pytest.approx(0.0)


def test_mla_validator_rejects_partial_row_aware_sidecar_for_h800_mixed_rows(
    tmp_path: Path,
) -> None:
    from tests.analysis.mla_deepseek_v2.validate_flashinfer_mla_live_probe import (
        validate_frontier_import_sidecar_against_vllm_log,
    )

    rows = h800_mla_mixed_rows()
    cuda_log = tmp_path / "cuda_ops.jsonl"
    sidecar = tmp_path / "attention_kernel_only_vllm_mla_groundtruth_comparison.csv"
    cuda_log.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    sidecar_rows = []
    for scope in (
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ):
        scope_rows = [row for row in rows if row["op_name"] == scope]
        median_ms = statistics.median(float(row["cuda_time_ms"]) for row in scope_rows)
        sidecar_rows.append(
            {
                "scope": scope,
                "vllm_cuda_time_ms": median_ms,
                "frontier_profile_median_ms": median_ms,
                "absolute_error_ms": 0.0,
                "relative_error_pct": 0.0,
                "vllm_sample_count": len(scope_rows),
                "profile_row_index": 0,
                "batch_size": scope_rows[0]["batch_size"],
                "batch_num_tokens": scope_rows[0]["batch_num_tokens"],
                "batch_num_prefill_tokens": scope_rows[0][
                    "batch_num_prefill_tokens"
                ],
                "batch_num_decode_tokens": scope_rows[0]["batch_num_decode_tokens"],
                "max_seqlen_q": scope_rows[0]["meta"]["max_seqlen_q"],
                "max_seqlen_k": scope_rows[0]["meta"]["max_seqlen_k"],
            }
        )
    pd.DataFrame(sidecar_rows).to_csv(sidecar, index=False)

    with pytest.raises(ValueError, match="row-aware dynamic sidecar columns"):
        validate_frontier_import_sidecar_against_vllm_log(
            cuda_op_log=cuda_log,
            sidecar_csv=sidecar,
            max_absolute_error_ms=0.0,
            max_relative_error_pct=0.0,
        )

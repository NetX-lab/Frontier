from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COLOCATION_ROOT = REPO_ROOT / "examples" / "architecture" / "co-location"
RUNTIME_OUTPUT_ROOT = REPO_ROOT / "outputs" / "tests" / "colocation_examples_runtime"

CASES = [
    ("offline_dense", "offline/dense_model_basic.sh", "offline_batch", "meta_llama_llama_2_7b_hf", 2),
    ("offline_moe", "offline/moe_model_basic.sh", "offline_batch", "phi_tiny_moe_instruct", 2),
    ("offline_thinking", "offline/thinking_mode_basic.sh", "offline_batch", "meta_llama_llama_2_7b_hf", 2),
    ("offline_spec_dec", "offline/moe_spec_dec.sh", "offline_batch", "phi_tiny_moe_instruct", 2),
    ("offline_prefix", "offline/moe_prefix_caching.sh", "offline_batch", "phi_tiny_moe_instruct", 2),
    ("online_dense", "online/dense_model_basic_online.sh", "online_serving", "meta_llama_llama_2_7b_hf", 2),
    ("online_moe", "online/moe_model_basic_online.sh", "online_serving", "phi_tiny_moe_instruct", 2),
    ("online_thinking", "online/thinking_mode_basic_online.sh", "online_serving", "meta_llama_llama_2_7b_hf", 2),
    ("online_spec_dec", "online/moe_spec_dec_online.sh", "online_serving", "phi_tiny_moe_instruct", 2),
    ("online_prefix", "online/moe_prefix_caching_online.sh", "online_serving", "phi_tiny_moe_instruct", 2),
]


def _runtime_env(case_id: str, output_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1",
            "METRICS_OUTPUT_DIR": str(output_root),
            "RUN_ID": case_id,
            "CC_BACKEND": "analytical",
            "NUM_REQUESTS": "2",
            "PREFILL_TOKENS": "8",
            "DECODE_TOKENS": "2",
            "QPS": "2.0",
            "DUMMY_EXEC_TIME_MS": "1.0",
            "MAX_TOKENS_IN_BATCH": "128",
            "LONG_PREFILL_TOKEN_THRESHOLD": "4",
            "NUM_REPLICAS": "1",
        }
    )
    return env


def _load_metrics(output_root: Path, model_type: str, workload_type: str, run_id: str) -> tuple[pd.DataFrame, dict]:
    run_dir = output_root / model_type / workload_type / run_id
    request_metrics = run_dir / "request_metrics.csv"
    system_metrics = run_dir / "system_metrics.json"
    assert request_metrics.is_file(), request_metrics
    assert system_metrics.is_file(), system_metrics
    return pd.read_csv(request_metrics), json.loads(system_metrics.read_text(encoding="utf-8"))


@pytest.mark.skipif(
    os.environ.get("FRONTIER_RUN_COLOCATION_EXAMPLES_E2E") != "1",
    reason="Set FRONTIER_RUN_COLOCATION_EXAMPLES_E2E=1 to run the full co-location examples runtime suite.",
)
@pytest.mark.parametrize(
    ("case_id", "relative_script", "workload_type", "model_type", "expected_requests"),
    CASES,
)
def test_colocation_example_runtime_case(
    tmp_path: Path,
    case_id: str,
    relative_script: str,
    workload_type: str,
    model_type: str,
    expected_requests: int,
) -> None:
    configured_output_root = os.environ.get("FRONTIER_COLOCATION_RUNTIME_OUTPUT_ROOT")
    output_root = Path(configured_output_root) if configured_output_root else tmp_path / "metrics"
    script = COLOCATION_ROOT / relative_script

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=_runtime_env(case_id, output_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout[-4000:]
    request_df, system_metrics = _load_metrics(output_root, model_type, workload_type, case_id)

    assert len(request_df) == expected_requests
    assert int(system_metrics["simulation_metadata"]["total_requests"]) == expected_requests
    assert int(system_metrics["simulation_metadata"]["completed_requests"]) == expected_requests
    assert request_df["request_num_tokens"].sum() > 0
    assert request_df["request_num_decode_tokens"].sum() > 0

    for column in ["ttft", "request_e2e_time", "request_execution_time"]:
        assert column in request_df.columns
        assert request_df[column].notna().all()
        assert (request_df[column] >= 0).all()

    throughput = system_metrics["throughput_metrics"]
    assert throughput["requests_per_second"] > 0
    assert throughput["tokens_per_second"] > 0

    summary_jsonl = os.environ.get("FRONTIER_COLOCATION_RUNTIME_SUMMARY_JSONL")
    if summary_jsonl:
        summary_path = Path(summary_jsonl)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "case_id": case_id,
            "script": relative_script,
            "workload_type": workload_type,
            "model_type": model_type,
            "expected_requests": expected_requests,
            "actual_request_rows": int(len(request_df)),
            "completed_requests": int(system_metrics["simulation_metadata"]["completed_requests"]),
            "total_input_tokens": int(request_df["request_num_prefill_tokens"].sum()),
            "total_decode_tokens": int(request_df["request_num_decode_tokens"].sum()),
            "mean_ttft_ms": float(request_df["ttft"].mean()),
            "mean_e2e_ms": float(request_df["request_e2e_time"].mean()),
            "requests_per_second": float(throughput["requests_per_second"]),
            "tokens_per_second": float(throughput["tokens_per_second"]),
            "metrics_dir": str((output_root / model_type / workload_type / case_id).resolve()),
        }
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

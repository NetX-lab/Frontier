from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "tests" / "e2e" / "pdd_scripts_cross_validate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pdd_scripts_cross_validate", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pdd_cross_validator_normalizes_request_and_transfer_metrics(tmp_path: Path) -> None:
    runner = _load_runner()
    metrics_run_dir = tmp_path / "metrics" / "dense_case"
    metrics_run_dir.mkdir(parents=True)
    (metrics_run_dir / "request_metrics.csv").write_text(
        "Request Id,request_e2e_time,ttft,tpot,transfer_kv_cache,request_num_prefill_tokens,request_num_decode_tokens,request_thinking_round_count,request_spec_total_iterations,request_spec_accepted_drafts,request_spec_rejected_drafts,request_spec_committed_tokens,request_spec_acceptance_ratio,request_spec_committed_per_iteration,request_cached_prefill_tokens,request_prefix_cache_query_blocks,request_prefix_cache_hit_blocks\n"
        "0,1432.66777216,716.0,358.0,0.66777216,8,2,0,1,2,0,2,1.0,2,4,6,4\n",
        encoding="utf-8",
    )
    (metrics_run_dir / "system_metrics.json").write_text(
        json.dumps(
            {
                "simulation_metadata": {"total_requests": 1, "completed_requests": 1},
                "kv_cache_transfer_statistics": {
                    "total_transfers": 1,
                    "total_data_transferred_bytes": 4194304,
                    "total_transfer_time_ms": 0.66777216,
                },
                "kv_cache_transfer_total_bytes": 4194304,
                "prefix_cache_statistics": {"total_queries": 6, "total_hits": 4},
                "spec_decode_statistics": {"total_iterations": 1, "accepted_drafts": 2},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary = runner.load_metrics_summary(metrics_run_dir, case_id="dense_case", role="candidate")

    assert summary["case_id"] == "dense_case"
    assert summary["role"] == "candidate"
    assert summary["counts"] == {
        "request_rows": 1,
        "total_requests": 1,
        "completed_requests": 1,
    }
    assert summary["request_rows"][0] == {
        "request_id": "0",
        "request_e2e_time": "1432.66777216",
        "request_waiting_time_total": "",
        "ttft": "716.0",
        "tpot": "358.0",
        "transfer_kv_cache": "0.66777216",
        "transfer_kv_cache_request_start_ts": "",
        "transfer_kv_cache_request_end_ts": "",
        "request_num_prefill_tokens": "8",
        "request_num_decode_tokens": "2",
        "request_thinking_round_count": "0",
        "request_session_id": "",
        "request_spec_total_iterations": "1",
        "request_spec_accepted_drafts": "2",
        "request_spec_rejected_drafts": "0",
        "request_spec_committed_tokens": "2",
        "request_spec_acceptance_ratio": "1.0",
        "request_spec_committed_per_iteration": "2",
        "request_cached_prefill_tokens": "4",
        "request_prefix_cache_query_blocks": "6",
        "request_prefix_cache_hit_blocks": "4",
    }
    assert summary["kv_cache_transfer_statistics"] == {
        "total_transfers": 1,
        "total_data_transferred_bytes": 4194304,
        "total_transfer_time_ms": 0.66777216,
    }
    assert summary["kv_cache_transfer_total_bytes"] == 4194304
    assert summary["prefix_cache_statistics"] == {"total_queries": 6, "total_hits": 4}
    assert summary["spec_decode_statistics"] == {"total_iterations": 1, "accepted_drafts": 2}


def test_pdd_cross_validator_reports_strict_nested_mismatches() -> None:
    runner = _load_runner()
    candidate = {
        "case_id": "dense_case",
        "counts": {"request_rows": 1, "completed_requests": 1, "total_requests": 1},
        "request_rows": [{"request_id": "0", "ttft": "716.0"}],
        "kv_cache_transfer_statistics": {"total_transfers": 1},
    }
    reference = {
        "case_id": "dense_case",
        "counts": {"request_rows": 1, "completed_requests": 1, "total_requests": 1},
        "request_rows": [{"request_id": "0", "ttft": "717.0"}],
        "kv_cache_transfer_statistics": {"total_transfers": 1},
    }

    diff = runner.compare_summaries(candidate, reference)

    assert diff["status"] == "FAIL"
    assert diff["mismatch_count"] == 1
    assert diff["mismatches"] == [
        {
            "field": "request_rows",
            "candidate": [{"request_id": "0", "ttft": "716.0"}],
            "reference": [{"request_id": "0", "ttft": "717.0"}],
        }
    ]

    assert runner.compare_summaries(candidate, candidate)["status"] == "PASS"

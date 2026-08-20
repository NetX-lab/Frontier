from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.metrics.metrics_store import MetricsStore
from frontier.types import ClusterType
from tests.e2e import transfer_metrics_contract as transfer_contract
from tests.e2e.transfer_metrics_contract import (
    main,
    validate_request_metrics,
    validate_stage_transfer_alignment,
    validate_transfer_rows,
)


def _m2n_row(
    *,
    transfer_id: str = "m2n-1",
    source_cluster: str = "DECODE_ATTN",
    target_cluster: str = "DECODE_FFN",
    source_replica_id: int = 0,
    target_replica_id: int = 1,
    batch_id: int = 11,
    batch_global_id: int = 11,
    request_ids: list[int] | None = None,
    layer_id: int = 4,
    pipeline_stage: str | None = None,
    start: float = 1.0,
    end: float = 1.002,
    ffn_source_lane: int | None = 0,
) -> dict[str, object]:
    if request_ids is None:
        request_ids = [7]
    if pipeline_stage is None:
        pipeline_stage = (
            "attn_to_ffn" if source_cluster == "DECODE_ATTN" else "ffn_to_attn"
        )
    source_replica_local_id = (
        ffn_source_lane if source_cluster == "DECODE_FFN" else None
    )
    attention_owner_replica_id = (
        source_replica_id
        if source_cluster == "DECODE_ATTN"
        else target_replica_id
    )
    attention_owner_replica_local_id = (
        source_replica_local_id
        if source_cluster == "DECODE_ATTN"
        else None
    )
    return {
        "transfer_id": transfer_id,
        "transfer_kind": "m2n",
        "request_ids": request_ids,
        "request_runtime_epochs": [0 for _ in request_ids],
        "batch_id": batch_id,
        "batch_global_id": batch_global_id,
        "source_cluster": source_cluster,
        "target_cluster": target_cluster,
        "source_replica_id": source_replica_id,
        "source_replica_local_id": source_replica_local_id,
        "attention_owner_replica_id": attention_owner_replica_id,
        "attention_owner_replica_local_id": attention_owner_replica_local_id,
        "target_replica_id": target_replica_id,
        "target_replica_local_id": None,
        "layer_id": layer_id,
        "afd_stage_idx": 0,
        "iteration_ids": [0],
        "pipeline_stage": pipeline_stage,
        "bytes": 4096,
        "start_ts_s": start,
        "end_ts_s": end,
        "duration_ms": (end - start) * 1000.0,
        "target_bound": True,
        "status": "completed",
    }


def test_transfer_validator_accepts_complete_bidirectional_lineage() -> None:
    rows = [
        _m2n_row(),
        _m2n_row(
            transfer_id="m2n-2",
            source_cluster="DECODE_FFN",
            target_cluster="DECODE_ATTN",
            source_replica_id=1,
            target_replica_id=0,
            start=1.003,
            end=1.005,
        ),
    ]

    result = validate_transfer_rows(rows)

    assert result["status"] == "PASS"
    assert result["transfer_count"] == 2


def test_transfer_validator_uses_explicit_attention_owner_identity() -> None:
    rows = [
        _m2n_row(),
        _m2n_row(
            transfer_id="m2n-2",
            source_cluster="DECODE_FFN",
            target_cluster="DECODE_ATTN",
            source_replica_id=1,
            target_replica_id=0,
            start=1.003,
            end=1.005,
        ),
    ]
    result = validate_transfer_rows(rows)

    assert result["status"] == "PASS"


def test_transfer_validator_requires_configured_pdaf_transfer_kinds() -> None:
    rows = [
        _m2n_row(),
        _m2n_row(
            transfer_id="m2n-2",
            source_cluster="DECODE_FFN",
            target_cluster="DECODE_ATTN",
            source_replica_id=1,
            target_replica_id=0,
            start=1.003,
            end=1.005,
        ),
    ]

    result = validate_transfer_rows(
        rows,
        required_transfer_kinds={"kv_cache", "m2n"},
    )

    assert result["status"] == "FAIL"
    assert any("required transfer kind" in error for error in result["errors"])


def test_transfer_validator_requires_one_transfer_per_direction() -> None:
    result = validate_transfer_rows([_m2n_row()])

    assert result["status"] == "FAIL"


def test_transfer_validator_rejects_non_inverse_replica_lineage() -> None:
    result = validate_transfer_rows(
        [
            _m2n_row(),
            _m2n_row(
                transfer_id="m2n-2",
                source_cluster="DECODE_FFN",
                target_cluster="DECODE_ATTN",
                source_replica_id=3,
                target_replica_id=0,
                start=1.003,
                end=1.005,
            ),
        ]
    )

    assert result["status"] == "FAIL"


def test_transfer_validator_rejects_bidirectional_byte_mismatch() -> None:
    first = _m2n_row()
    second = _m2n_row(
        transfer_id="m2n-2",
        source_cluster="DECODE_FFN",
        target_cluster="DECODE_ATTN",
        source_replica_id=1,
        target_replica_id=0,
        start=1.003,
        end=1.005,
    )
    second["bytes"] = 8192

    result = validate_transfer_rows([first, second])

    assert result["status"] == "FAIL"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("target_replica_id"),
        lambda row: row.update(transfer_id="m2n-2"),
        lambda row: row.update(
            source_cluster="DECODE_FFN",
            target_cluster="DECODE_ATTN",
        ),
        lambda row: row.update(start_ts_s=2.0, end_ts_s=1.0),
        lambda row: row.update(duration_ms=999.0),
    ],
    ids=["missing-target", "duplicate-id", "wrong-direction", "reverse-time", "duration"],
)
def test_transfer_validator_fails_closed_on_mutations(mutation) -> None:
    first = _m2n_row()
    second = _m2n_row(
        transfer_id="m2n-2",
        source_cluster="DECODE_FFN",
        target_cluster="DECODE_ATTN",
        source_replica_id=1,
        target_replica_id=0,
        start=1.003,
        end=1.005,
    )
    mutation(first)

    result = validate_transfer_rows([first, second])

    assert result["status"] == "FAIL"


def _write_request_metric_fixture(
    tmp_path: Path,
    *,
    truth_overrides: dict[str, object] | None = None,
    arrival_overrides: dict[str, object] | None = None,
    csv_overrides: dict[str, object] | None = None,
    system_overrides: dict[str, object] | None = None,
    remove_system_keys: tuple[str, ...] = (),
) -> Path:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    system_metrics: dict[str, object] = {
        "simulation_metadata": {"total_requests": 1, "completed_requests": 1},
        "ttft_statistics": {
            "count": 1,
            "mean": 10.0,
            "median": 10.0,
            "std": 0.0,
            "min": 10.0,
            "max": 10.0,
            "p50": 10.0,
            "p90": 10.0,
            "p95": 10.0,
            "p99": 10.0,
            "unit": "ms",
        },
        "tpot_statistics": {
            "count": 1,
            "mean": 10.0,
            "median": 10.0,
            "std": 0.0,
            "min": 10.0,
            "max": 10.0,
            "p50": 10.0,
            "p90": 10.0,
            "p95": 10.0,
            "p99": 10.0,
            "unit": "ms",
            "note": "TPOT is only computed for requests with num_decode_tokens > 1. Computed from 1 out of 1 requests.",
        },
        "request_e2e_time_statistics": {
            "count": 1,
            "mean": 50.0,
            "median": 50.0,
            "std": 0.0,
            "min": 50.0,
            "max": 50.0,
            "p50": 50.0,
            "p90": 50.0,
            "p95": 50.0,
            "p99": 50.0,
            "unit": "ms",
        },
        "throughput_metrics": {
            "total_duration_ms": 50.0,
            "total_duration_seconds": 0.05,
            "requests_per_second": 20.0,
            "total_tokens_processed": 5,
            "total_decode_tokens_generated": 3,
            "tokens_per_second": 100.0,
            "decode_tokens_per_second": 60.0,
        },
        "kv_cache_transfer_statistics": {
            "total_transfers": 1,
            "total_transfer_time_ms": 4.0,
            "total_data_transferred_bytes": 8192,
        },
        "m2n_transfer_statistics": {
            "total_transfers": 2,
            "total_transfer_time_ms": 4.0,
            "total_data_transferred_bytes": 8192,
            "attn_to_ffn_transfers": 1,
            "ffn_to_attn_transfers": 1,
        },
    }
    if system_overrides:
        system_metrics.update(system_overrides)
    for key in remove_system_keys:
        system_metrics.pop(key, None)
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps(system_metrics),
        encoding="utf-8",
    )
    truth: dict[str, object] = {
        "event_type": "request_completion",
        "request_id": 7,
        "arrived_at": 1.0,
        "prefill_completed_at": 1.01,
        "first_decode_token_completed_at": 1.03,
        "completed_at": 1.05,
        "num_prefill_tokens": 2,
        "num_decode_tokens": 3,
        "request_e2e_time_s": 0.05,
        "ttft_s": 0.01,
        "tpot_s": 0.01,
    }
    if truth_overrides:
        truth.update(truth_overrides)
    arrival = {
        "event_type": "request_arrival",
        "request_id": 7,
        "arrived_at": 1.0,
        "num_prefill_tokens": 2,
        "num_decode_tokens": 3,
    }
    if arrival_overrides:
        arrival.update(arrival_overrides)
    (metrics_dir / "metrics_ground_truth.jsonl").write_text(
        json.dumps(arrival) + "\n" + json.dumps(truth) + "\n",
        encoding="utf-8",
    )
    csv_row: dict[str, object] = {
        "Request Id": "7",
        "request_e2e_time": "50.0",
        "ttft": "10.0",
        "tpot": "10.0",
        "decode_e2e_time": "40.0",
        "request_num_tokens": "5",
        "request_num_prefill_tokens": "2",
        "request_num_decode_tokens": "3",
        "tpot_computation": "8.0",
        "tpot_transfer": "2.0",
        "transfer_kv_cache": "4.0",
        "transfer_m2n_total": "4.0",
        "transfer_m2n_attn_to_ffn": "2.0",
        "transfer_m2n_ffn_to_attn": "2.0",
    }
    if csv_overrides:
        csv_row.update(csv_overrides)
    with (metrics_dir / "request_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_row))
        writer.writeheader()
        writer.writerow(csv_row)
    return metrics_dir


def _request_transfer_rows() -> list[dict[str, object]]:
    return [
        {
            "transfer_id": "kv_cache-0",
            "transfer_kind": "kv_cache",
            "request_ids": [7],
            "request_runtime_epochs": [0],
            "bytes": 8192,
            "duration_ms": 4.0,
            "source_cluster": "PREFILL",
            "target_cluster": "DECODE_ATTN",
        },
        _m2n_row(request_ids=[7], start=1.01, end=1.012),
        _m2n_row(
            transfer_id="m2n-2",
            source_cluster="DECODE_FFN",
            target_cluster="DECODE_ATTN",
            source_replica_id=1,
            target_replica_id=0,
            request_ids=[7],
            start=1.013,
            end=1.015,
        ),
    ]


def test_request_validator_checks_cardinality_units_and_decomposition(tmp_path: Path) -> None:
    metrics_dir = _write_request_metric_fixture(tmp_path)

    result = validate_request_metrics(
        metrics_dir,
        expected_request_count=1,
        transfer_rows=_request_transfer_rows(),
    )

    assert result["status"] == "PASS"
    assert result["request_count"] == 1


def test_request_validator_rejects_csv_unit_mutation(tmp_path: Path) -> None:
    metrics_dir = _write_request_metric_fixture(
        tmp_path,
        csv_overrides={"request_e2e_time": "0.05"},
    )

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "FAIL"


def test_request_validator_rejects_common_mode_tpot_mutation(tmp_path: Path) -> None:
    metrics_dir = _write_request_metric_fixture(
        tmp_path,
        truth_overrides={"tpot_s": 0.02},
        csv_overrides={"tpot": "20.0", "tpot_computation": "18.0"},
    )

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "FAIL"


def test_request_validator_rejects_missing_system_ttft_statistics(
    tmp_path: Path,
) -> None:
    metrics_dir = _write_request_metric_fixture(
        tmp_path,
        remove_system_keys=("ttft_statistics",),
    )

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("ttft_statistics" in error for error in result["errors"])


def test_request_validator_rejects_wrong_system_ttft_mean(tmp_path: Path) -> None:
    metrics_dir = _write_request_metric_fixture(
        tmp_path,
        system_overrides={
            "ttft_statistics": {
                "count": 1,
                "mean": 11.0,
                "median": 10.0,
                "std": 0.0,
                "min": 10.0,
                "max": 10.0,
                "p50": 10.0,
                "p90": 10.0,
                "p95": 10.0,
                "p99": 10.0,
                "unit": "ms",
            }
        },
    )

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "FAIL"
    assert any("ttft_statistics.mean" in error for error in result["errors"])


def test_request_validator_reports_missing_aggregate_unit_as_insufficient(
    tmp_path: Path,
) -> None:
    metrics_dir = _write_request_metric_fixture(tmp_path)
    system_path = metrics_dir / "system_metrics.json"
    system_metrics = json.loads(system_path.read_text(encoding="utf-8"))
    system_metrics["ttft_statistics"].pop("unit")
    system_path.write_text(json.dumps(system_metrics), encoding="utf-8")

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("ttft_statistics: missing unit" in error for error in result["errors"])


def test_request_validator_rejects_wrong_aggregate_unit(tmp_path: Path) -> None:
    metrics_dir = _write_request_metric_fixture(
        tmp_path,
        system_overrides={
            "ttft_statistics": {
                "count": 1,
                "mean": 10.0,
                "median": 10.0,
                "std": 0.0,
                "min": 10.0,
                "max": 10.0,
                "p50": 10.0,
                "p90": 10.0,
                "p95": 10.0,
                "p99": 10.0,
                "unit": "s",
            }
        },
    )

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "FAIL"
    assert any("ttft_statistics.unit mismatch" in error for error in result["errors"])


def test_request_validator_rejects_missing_throughput_total_tokens(
    tmp_path: Path,
) -> None:
    metrics_dir = _write_request_metric_fixture(tmp_path)
    system_path = metrics_dir / "system_metrics.json"
    system_metrics = json.loads(system_path.read_text(encoding="utf-8"))
    system_metrics["throughput_metrics"].pop("total_tokens_processed")
    system_path.write_text(json.dumps(system_metrics), encoding="utf-8")

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("total_tokens_processed" in error for error in result["errors"])


def test_request_validator_rejects_wrong_throughput_total_tokens(
    tmp_path: Path,
) -> None:
    metrics_dir = _write_request_metric_fixture(
        tmp_path,
        system_overrides={
            "throughput_metrics": {
                "total_duration_ms": 50.0,
                "total_duration_seconds": 0.05,
                "requests_per_second": 20.0,
                "total_tokens_processed": 6,
                "total_decode_tokens_generated": 3,
                "tokens_per_second": 120.0,
                "decode_tokens_per_second": 60.0,
            }
        },
    )

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "FAIL"
    assert any("total_tokens_processed" in error for error in result["errors"])


def test_request_validator_accepts_documented_single_token_tpot_error(
    tmp_path: Path,
) -> None:
    metrics_dir = _write_request_metric_fixture(
        tmp_path,
        truth_overrides={"num_decode_tokens": 1, "tpot_s": 0.0},
        arrival_overrides={"num_decode_tokens": 1},
        csv_overrides={
            "request_num_tokens": "3",
            "request_num_decode_tokens": "1",
            "tpot": "0.0",
            "tpot_computation": "0.0",
            "tpot_transfer": "0.0",
        },
        system_overrides={
            "tpot_statistics": {
                "error": (
                    "No TPOT data available "
                    "(all requests may have num_decode_tokens=1)"
                )
            },
            "throughput_metrics": {
                "total_duration_ms": 50.0,
                "total_duration_seconds": 0.05,
                "requests_per_second": 20.0,
                "total_tokens_processed": 3,
                "total_decode_tokens_generated": 1,
                "tokens_per_second": 60.0,
                "decode_tokens_per_second": 20.0,
            },
        },
    )

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "PASS"
    assert result["recomputed_tpot_statistics"]["count"] == 0


def test_request_validator_rejects_non_documented_single_token_tpot_error(
    tmp_path: Path,
) -> None:
    metrics_dir = _write_request_metric_fixture(
        tmp_path,
        truth_overrides={"num_decode_tokens": 1, "tpot_s": 0.0},
        arrival_overrides={"num_decode_tokens": 1},
        csv_overrides={
            "request_num_tokens": "3",
            "request_num_decode_tokens": "1",
            "tpot": "0.0",
            "tpot_computation": "0.0",
            "tpot_transfer": "0.0",
        },
        system_overrides={
            "tpot_statistics": {"error": "missing TPOT"},
            "throughput_metrics": {
                "total_duration_ms": 50.0,
                "total_duration_seconds": 0.05,
                "requests_per_second": 20.0,
                "total_tokens_processed": 3,
                "total_decode_tokens_generated": 1,
                "tokens_per_second": 60.0,
                "decode_tokens_per_second": 20.0,
            },
        },
    )

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "FAIL"
    assert any("unexpected no-data error" in error for error in result["errors"])


def test_request_validator_rejects_transfer_decomposition_mutation(tmp_path: Path) -> None:
    metrics_dir = _write_request_metric_fixture(
        tmp_path,
        csv_overrides={"transfer_m2n_total": "5.0"},
    )

    result = validate_request_metrics(
        metrics_dir,
        expected_request_count=1,
        transfer_rows=_request_transfer_rows(),
    )

    assert result["status"] == "FAIL"


def test_request_validator_rejects_missing_configured_pdaf_kv_evidence(
    tmp_path: Path,
) -> None:
    metrics_dir = _write_request_metric_fixture(
        tmp_path,
        csv_overrides={"transfer_kv_cache": "0.0"},
        remove_system_keys=("kv_cache_transfer_statistics",),
    )
    transfer_rows = [
        row
        for row in _request_transfer_rows()
        if row["transfer_kind"] == "m2n"
    ]

    result = validate_request_metrics(
        metrics_dir,
        expected_request_count=1,
        transfer_rows=transfer_rows,
        required_transfer_kinds={"kv_cache", "m2n"},
    )

    assert result["status"] == "FAIL"
    assert any("required transfer kind" in error for error in result["errors"])


def test_request_validator_rejects_duplicate_completion_rows(
    tmp_path: Path,
) -> None:
    metrics_dir = _write_request_metric_fixture(tmp_path)
    ground_truth_path = metrics_dir / "metrics_ground_truth.jsonl"
    completion_row = ground_truth_path.read_text(encoding="utf-8")
    ground_truth_path.write_text(
        completion_row + completion_row,
        encoding="utf-8",
    )

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "FAIL"
    assert any("duplicate request_completion" in error for error in result["errors"])


def test_request_validator_requires_system_completed_request_count(
    tmp_path: Path,
) -> None:
    metrics_dir = _write_request_metric_fixture(tmp_path)
    system_path = metrics_dir / "system_metrics.json"
    system_metrics = json.loads(system_path.read_text(encoding="utf-8"))
    system_metrics["simulation_metadata"].pop("completed_requests")
    system_path.write_text(json.dumps(system_metrics), encoding="utf-8")

    result = validate_request_metrics(metrics_dir, expected_request_count=1)

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("completed_requests" in error for error in result["errors"])


def _aligned_kv_stage_and_transfer_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    stage_rows = [
        {
            "batch_id": 10,
            "cluster_type": "PREFILL",
            "execution_scope": "FULL_STAGE_WORLD",
            "request_ids": ["7"],
            "request_runtime_epochs": [3],
            "iteration_ids": [0],
            "replica_id": 0,
            "replica_local_id": None,
            "stage_id": 0,
            "stage_start_ts": 0.9,
            "stage_end_ts": 1.0,
        },
        {
            "batch_id": 12,
            "cluster_type": "DECODE",
            "execution_scope": "FULL_STAGE_WORLD",
            "request_ids": ["7"],
            "request_runtime_epochs": [3],
            "iteration_ids": [0],
            "replica_id": 2,
            "replica_local_id": None,
            "stage_id": 0,
            "stage_start_ts": 1.004,
            "stage_end_ts": 1.014,
        },
    ]
    transfer_rows = [
        {
            "transfer_id": "kv_cache-0",
            "transfer_kind": "kv_cache",
            "request_ids": [7],
            "request_runtime_epochs": [3],
            "iteration_ids": [0],
            "batch_id": 11,
            "batch_global_id": -1,
            "source_cluster": "PREFILL",
            "target_cluster": "DECODE",
            "source_replica_id": 0,
            "source_replica_local_id": None,
            "target_replica_id": 2,
            "target_replica_local_id": None,
            "target_bound": True,
            "bytes": 8192,
            "start_ts_s": 1.0,
            "end_ts_s": 1.004,
            "duration_ms": 4.0,
            "status": "completed",
        }
    ]
    return stage_rows, transfer_rows


def test_kv_stage_alignment_accepts_exact_pdd_lineage() -> None:
    stage_rows, transfer_rows = _aligned_kv_stage_and_transfer_rows()

    result = transfer_contract.validate_kv_stage_alignment(
        stage_rows,
        transfer_rows,
    )

    assert result["status"] == "PASS"


def test_kv_stage_alignment_accepts_legal_target_queue_delay() -> None:
    stage_rows, transfer_rows = _aligned_kv_stage_and_transfer_rows()
    stage_rows[1]["stage_start_ts"] = 1.006
    stage_rows[1]["stage_end_ts"] = 1.016
    later_target_stage = dict(stage_rows[1])
    later_target_stage["batch_id"] = 13
    later_target_stage["stage_start_ts"] = 1.016
    later_target_stage["stage_end_ts"] = 1.026
    stage_rows.append(later_target_stage)

    result = transfer_contract.validate_kv_stage_alignment(
        stage_rows,
        transfer_rows,
    )

    assert result["status"] == "PASS"


def test_kv_stage_alignment_rejects_premature_matching_target_stage() -> None:
    stage_rows, transfer_rows = _aligned_kv_stage_and_transfer_rows()
    premature_target_stage = dict(stage_rows[1])
    premature_target_stage["batch_id"] = 11
    premature_target_stage["stage_start_ts"] = 1.001
    premature_target_stage["stage_end_ts"] = 1.003
    stage_rows.append(premature_target_stage)

    result = transfer_contract.validate_kv_stage_alignment(
        stage_rows,
        transfer_rows,
    )

    assert result["status"] == "FAIL"


def test_kv_stage_alignment_rejects_transfer_before_latest_source_completion() -> None:
    stage_rows, transfer_rows = _aligned_kv_stage_and_transfer_rows()
    later_source_stage = dict(stage_rows[0])
    later_source_stage["batch_id"] = 11
    later_source_stage["stage_start_ts"] = 1.0
    later_source_stage["stage_end_ts"] = 1.02
    stage_rows.append(later_source_stage)

    result = transfer_contract.validate_kv_stage_alignment(
        stage_rows,
        transfer_rows,
    )

    assert result["status"] == "FAIL"


def test_kv_stage_alignment_requires_terminal_source_pipeline_stage() -> None:
    stage_rows, transfer_rows = _aligned_kv_stage_and_transfer_rows()
    first_source_stage = stage_rows[0]
    first_source_stage["stage_end_ts"] = 0.99
    terminal_source_stage = dict(first_source_stage)
    terminal_source_stage["stage_id"] = 1
    terminal_source_stage["stage_start_ts"] = 0.99
    terminal_source_stage["stage_end_ts"] = 1.0
    next_chunk_entry = dict(first_source_stage)
    next_chunk_entry["batch_id"] = 11
    next_chunk_entry["stage_start_ts"] = 1.0
    next_chunk_entry["stage_end_ts"] = 1.02
    stage_rows.extend([terminal_source_stage, next_chunk_entry])
    transfer_rows[0].update(
        start_ts_s=1.02,
        end_ts_s=1.024,
        duration_ms=4.0,
    )
    stage_rows[1].update(
        stage_start_ts=1.024,
        stage_end_ts=1.034,
    )

    result = transfer_contract.validate_kv_stage_alignment(
        stage_rows,
        transfer_rows,
    )

    assert result["status"] == "FAIL"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda stages, transfers: transfers[0].update(target_replica_id=9),
        lambda stages, transfers: stages[1].update(request_runtime_epochs=[4]),
    ],
    ids=["target-replica", "target-runtime-epoch"],
)
def test_kv_stage_alignment_rejects_target_identity_mutation(
    mutation,
) -> None:
    stage_rows, transfer_rows = _aligned_kv_stage_and_transfer_rows()
    mutation(stage_rows, transfer_rows)

    result = transfer_contract.validate_kv_stage_alignment(
        stage_rows,
        transfer_rows,
    )

    assert result["status"] == "FAIL"


def _aligned_stage_and_transfer_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    stage_rows = [
        {
            "batch_id": 34,
            "cluster_type": "DECODE_ATTN",
            "execution_scope": "FULL_STAGE_WORLD",
            "request_ids": ["0"],
            "request_runtime_epochs": [0],
            "iteration_ids": [0],
            "layer_id": 4,
            "afd_stage_idx": 0,
            "replica_id": 0,
            "replica_local_id": None,
            "stage_start_ts": 0.99,
            "stage_end_ts": 1.0,
            "operation_kind": "full_stage",
        },
        {
            "batch_id": 35,
            "cluster_type": "DECODE_FFN",
            "execution_scope": "EP_WAVE_LANE",
            "request_ids": ["513"],
            "request_runtime_epochs": [0],
            "iteration_ids": [0],
            "layer_id": 4,
            "afd_stage_idx": 0,
            "replica_id": 1,
            "replica_local_id": 0,
            "stage_id": 0,
            "operation_id": 34,
            "schedule_epoch": 0,
            "source_request_ids": ["0"],
            "source_request_runtime_epochs": [0],
            "source_batch_ids": [34],
            "source_batch_arrival_times": [1.002],
            "source_group_ready_ts": 1.002,
            "stage_start_ts": 1.002,
            "stage_end_ts": 1.003,
            "stage_completion_observed_ts": 1.003,
            "stage_completion_observed_source": "ep_alltoall_combine_collective",
            "operation_kind": "ep_ffn",
        },
    ]
    transfer_rows = [
        _m2n_row(batch_id=34, batch_global_id=34, request_ids=[0]),
        _m2n_row(
            transfer_id="m2n-2",
            source_cluster="DECODE_FFN",
            target_cluster="DECODE_ATTN",
            source_replica_id=1,
            target_replica_id=0,
            batch_id=34,
            batch_global_id=34,
            request_ids=[0],
            start=1.003,
            end=1.005,
        ),
    ]
    return stage_rows, transfer_rows


def test_stage_transfer_alignment_joins_source_request_lineage() -> None:
    stage_rows, transfer_rows = _aligned_stage_and_transfer_rows()

    result = validate_stage_transfer_alignment(
        stage_rows,
        transfer_rows,
        expected_layer_protocols={4: "moe"},
        expected_moe_ep_size=1,
    )

    assert result["status"] == "PASS"


def test_stage_transfer_alignment_uses_observed_ep_completion_for_f2a() -> None:
    stage_rows, transfer_rows = _aligned_stage_and_transfer_rows()
    stage_rows[1]["stage_end_ts"] = 1.0028

    result = validate_stage_transfer_alignment(
        stage_rows,
        transfer_rows,
        expected_layer_protocols={4: "moe"},
        expected_moe_ep_size=1,
    )

    assert result["status"] == "PASS"


def test_stage_transfer_alignment_requires_expected_moe_ep_size() -> None:
    stage_rows, transfer_rows = _aligned_stage_and_transfer_rows()

    result = validate_stage_transfer_alignment(
        stage_rows,
        transfer_rows,
        expected_layer_protocols={4: "moe"},
    )

    assert result["status"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize("duplicate_lane", [False, True], ids=["missing-lane", "duplicate-lane"])
def test_stage_transfer_alignment_rejects_invalid_moe_ep_participants(
    duplicate_lane: bool,
) -> None:
    stage_rows, transfer_rows = _aligned_stage_and_transfer_rows()
    if duplicate_lane:
        duplicate = dict(stage_rows[1])
        duplicate["batch_id"] = 36
        stage_rows.append(duplicate)

    result = validate_stage_transfer_alignment(
        stage_rows,
        transfer_rows,
        expected_layer_protocols={4: "moe"},
        expected_moe_ep_size=2,
    )

    assert result["status"] == "FAIL"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda stages, transfers: stages[0].update(
            request_runtime_epochs=[1]
        ),
        lambda stages, transfers: [
            row.update(request_runtime_epochs=[1]) for row in transfers
        ],
        lambda stages, transfers: stages[1].update(
            source_request_runtime_epochs=[1]
        ),
    ],
    ids=["attention-stage", "transfers", "ffn-source"],
)
def test_stage_transfer_alignment_rejects_runtime_epoch_mismatch(
    mutation,
) -> None:
    stage_rows, transfer_rows = _aligned_stage_and_transfer_rows()
    mutation(stage_rows, transfer_rows)

    result = validate_stage_transfer_alignment(
        stage_rows,
        transfer_rows,
        expected_layer_protocols={4: "moe"},
        expected_moe_ep_size=1,
    )

    assert result["status"] == "FAIL"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda stages, transfers: [
            row.update(layer_id=5) for row in transfers
        ],
        lambda stages, transfers: [
            row.update(iteration_ids=[1]) for row in transfers
        ],
        lambda stages, transfers: stages[1].update(replica_id=3),
        lambda stages, transfers: stages[1].update(stage_start_ts=1.0025),
    ],
    ids=["layer", "iteration", "ffn-replica", "a2f-to-ffn-boundary"],
)
def test_stage_transfer_alignment_fails_closed_on_mutations(mutation) -> None:
    stage_rows, transfer_rows = _aligned_stage_and_transfer_rows()
    mutation(stage_rows, transfer_rows)

    result = validate_stage_transfer_alignment(
        stage_rows,
        transfer_rows,
        expected_layer_protocols={4: "moe"},
        expected_moe_ep_size=1,
    )

    assert result["status"] == "FAIL"


def _dense_stage_and_transfer_rows(
    *,
    layer_id: int,
    batch_id: int,
    request_id: int,
    attn_start: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_batch_id = batch_id
    synthetic_ffn_batch_id = batch_id + 1000
    attn_end = attn_start + 0.001
    a2f_end = attn_end + 0.002
    ffn_end = a2f_end + 0.001
    return (
        [
            {
                "batch_id": batch_id,
                "cluster_type": "DECODE_ATTN",
                "execution_scope": "FULL_STAGE_WORLD",
                "request_ids": [str(request_id)],
                "request_runtime_epochs": [0],
                "iteration_ids": [0],
                "layer_id": layer_id,
                "afd_stage_idx": 0,
                "replica_id": 0,
                "replica_local_id": None,
                "stage_start_ts": attn_start,
                "stage_end_ts": attn_end,
                "operation_kind": "full_stage",
            },
            {
                "batch_id": synthetic_ffn_batch_id,
                "cluster_type": "DECODE_FFN",
                "execution_scope": "FULL_STAGE_WORLD",
                "request_ids": [str(request_id)],
                "request_runtime_epochs": [0],
                "iteration_ids": [0],
                "source_request_ids": [str(request_id)],
                "source_request_runtime_epochs": [0],
                "source_batch_ids": [source_batch_id],
                "layer_id": layer_id,
                "afd_stage_idx": 0,
                "replica_id": 1,
                "replica_local_id": None,
                "stage_id": 0,
                "operation_id": source_batch_id,
                "schedule_epoch": 0,
                "stage_start_ts": a2f_end,
                "stage_end_ts": ffn_end,
                "source_batch_arrival_times": [a2f_end],
                "source_group_ready_ts": a2f_end,
                "operation_kind": "full_stage",
            },
        ],
        [
            _m2n_row(
                transfer_id=f"m2n-{batch_id}-a2f",
                batch_id=source_batch_id,
                batch_global_id=source_batch_id,
                request_ids=[request_id],
                layer_id=layer_id,
                start=attn_end,
                end=a2f_end,
            ),
            _m2n_row(
                transfer_id=f"m2n-{batch_id}-f2a",
                source_cluster="DECODE_FFN",
                target_cluster="DECODE_ATTN",
                source_replica_id=1,
                target_replica_id=0,
                batch_id=source_batch_id,
                batch_global_id=source_batch_id,
                request_ids=[request_id],
                layer_id=layer_id,
                start=ffn_end,
                end=ffn_end + 0.002,
                ffn_source_lane=None,
            ),
        ],
    )


def test_stage_transfer_alignment_accepts_dense_pdaf_layer() -> None:
    stage_rows, transfer_rows = _dense_stage_and_transfer_rows(
        layer_id=0,
        batch_id=34,
        request_id=0,
        attn_start=0.99,
    )

    result = validate_stage_transfer_alignment(
        stage_rows,
        transfer_rows,
        expected_layer_protocols={0: "dense"},
    )

    assert result["status"] == "PASS"


def test_stage_transfer_alignment_accepts_ffn_resource_queue_delay() -> None:
    first_stages, first_transfers = _dense_stage_and_transfer_rows(
        layer_id=0,
        batch_id=34,
        request_id=0,
        attn_start=0.99,
    )
    second_stages, second_transfers = _dense_stage_and_transfer_rows(
        layer_id=0,
        batch_id=36,
        request_id=1,
        attn_start=0.99,
    )
    first_ffn = first_stages[1]
    second_ffn = second_stages[1]
    for operation_id, stage in enumerate((first_ffn, second_ffn), start=1):
        stage.update(
            stage_id=0,
            operation_id=operation_id,
            schedule_epoch=0,
            source_group_ready_ts=0.993,
        )
    first_ffn.update(stage_start_ts=0.993, stage_end_ts=0.994)
    second_ffn.update(stage_start_ts=0.994, stage_end_ts=0.995)
    first_transfers[1].update(start_ts_s=0.994, end_ts_s=0.996)
    second_transfers[1].update(start_ts_s=0.995, end_ts_s=0.997)

    result = validate_stage_transfer_alignment(
        first_stages + second_stages,
        first_transfers + second_transfers,
        expected_layer_protocols={0: "dense"},
    )

    assert result["status"] == "PASS"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda first, second, first_transfers, second_transfers: second.update(
            source_batch_arrival_times=[0.992]
        ),
        lambda first, second, first_transfers, second_transfers: second.update(
            source_group_ready_ts=0.992
        ),
        lambda first, second, first_transfers, second_transfers: first.update(
            stage_start_ts=0.992
        ),
        lambda first, second, first_transfers, second_transfers: (
            second.update(stage_start_ts=0.9945, stage_end_ts=0.9955),
            second_transfers[1].update(start_ts_s=0.9955, end_ts_s=0.9975),
        ),
        lambda first, second, first_transfers, second_transfers: second.update(
            stage_start_ts=0.9935
        ),
    ],
    ids=[
        "source-arrival",
        "group-ready",
        "early-start",
        "unexplained-idle-gap",
        "resource-overlap",
    ],
)
def test_stage_transfer_alignment_rejects_invalid_ffn_queue_timing(
    mutation,
) -> None:
    first_stages, first_transfers = _dense_stage_and_transfer_rows(
        layer_id=0,
        batch_id=34,
        request_id=0,
        attn_start=0.99,
    )
    second_stages, second_transfers = _dense_stage_and_transfer_rows(
        layer_id=0,
        batch_id=36,
        request_id=1,
        attn_start=0.99,
    )
    first_ffn = first_stages[1]
    second_ffn = second_stages[1]
    for operation_id, stage in enumerate((first_ffn, second_ffn), start=1):
        stage.update(
            stage_id=0,
            operation_id=operation_id,
            schedule_epoch=0,
            source_group_ready_ts=0.993,
        )
    first_ffn.update(stage_start_ts=0.993, stage_end_ts=0.994)
    second_ffn.update(stage_start_ts=0.994, stage_end_ts=0.995)
    first_transfers[1].update(start_ts_s=0.994, end_ts_s=0.996)
    second_transfers[1].update(start_ts_s=0.995, end_ts_s=0.997)
    mutation(first_ffn, second_ffn, first_transfers, second_transfers)

    result = validate_stage_transfer_alignment(
        first_stages + second_stages,
        first_transfers + second_transfers,
        expected_layer_protocols={0: "dense"},
    )

    assert result["status"] == "FAIL"


def test_cli_threads_required_kinds_and_expected_layer_protocols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stage_rows, transfer_rows = _dense_stage_and_transfer_rows(
        layer_id=0,
        batch_id=34,
        request_id=0,
        attn_start=0.99,
    )
    stage_path = tmp_path / "stage.jsonl"
    transfer_path = tmp_path / "transfer.jsonl"
    stage_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in stage_rows),
        encoding="utf-8",
    )
    transfer_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in transfer_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transfer_metrics_contract.py",
            "--transfer-ledger",
            str(transfer_path),
            "--stage-ledger",
            str(stage_path),
            "--required-transfer-kind",
            "m2n",
            "--expected-layer-protocols-json",
            '["dense"]',
        ],
    )

    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["transfer"]["status"] == "PASS"
    assert result["stage_transfer"]["status"] == "PASS"


def test_cli_validates_pdd_kv_against_stage_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stage_rows, transfer_rows = _aligned_kv_stage_and_transfer_rows()
    stage_path = tmp_path / "stage.jsonl"
    transfer_path = tmp_path / "transfer.jsonl"
    stage_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in stage_rows),
        encoding="utf-8",
    )
    transfer_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in transfer_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transfer_metrics_contract.py",
            "--transfer-ledger",
            str(transfer_path),
            "--stage-ledger",
            str(stage_path),
            "--required-transfer-kind",
            "kv_cache",
        ],
    )

    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["transfer"]["status"] == "PASS"
    assert result["kv_stage"]["status"] == "PASS"


@pytest.mark.parametrize(
    "stage_dependent_args",
    [
        ["--required-transfer-kind", "kv_cache"],
        ["--expected-layer-protocols-json", '["dense"]'],
    ],
    ids=["required-kv", "expected-layer-protocols"],
)
def test_cli_requires_stage_ledger_for_stage_dependent_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stage_dependent_args: list[str],
) -> None:
    _, transfer_rows = _aligned_kv_stage_and_transfer_rows()
    transfer_path = tmp_path / "transfer.jsonl"
    transfer_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in transfer_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transfer_metrics_contract.py",
            "--transfer-ledger",
            str(transfer_path),
            *stage_dependent_args,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "--stage-ledger" in capsys.readouterr().err


def test_cli_requires_transfer_ledger_for_required_transfer_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transfer_metrics_contract.py",
            "--metrics-dir",
            str(tmp_path),
            "--required-transfer-kind",
            "m2n",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "--transfer-ledger" in capsys.readouterr().err


def test_cli_requires_metrics_dir_for_expected_request_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, transfer_rows = _aligned_kv_stage_and_transfer_rows()
    transfer_path = tmp_path / "transfer.jsonl"
    transfer_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in transfer_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transfer_metrics_contract.py",
            "--transfer-ledger",
            str(transfer_path),
            "--expected-request-count",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "--metrics-dir" in capsys.readouterr().err


def test_cli_requires_transfer_ledger_for_expected_layer_protocols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stage_rows, _ = _dense_stage_and_transfer_rows(
        layer_id=0,
        batch_id=34,
        request_id=0,
        attn_start=0.99,
    )
    stage_path = tmp_path / "stage.jsonl"
    stage_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in stage_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transfer_metrics_contract.py",
            "--stage-ledger",
            str(stage_path),
            "--expected-layer-protocols-json",
            '["dense"]',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "--transfer-ledger" in capsys.readouterr().err


def test_cli_rejects_expected_moe_ep_size_without_moe_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stage_rows, transfer_rows = _dense_stage_and_transfer_rows(
        layer_id=0,
        batch_id=34,
        request_id=0,
        attn_start=0.99,
    )
    stage_path = tmp_path / "stage.jsonl"
    transfer_path = tmp_path / "transfer.jsonl"
    stage_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in stage_rows),
        encoding="utf-8",
    )
    transfer_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in transfer_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transfer_metrics_contract.py",
            "--transfer-ledger",
            str(transfer_path),
            "--stage-ledger",
            str(stage_path),
            "--expected-layer-protocols-json",
            '["dense"]',
            "--expected-moe-ep-size",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "MoE" in capsys.readouterr().err


def test_stage_transfer_alignment_fails_when_expected_dense_chain_is_absent() -> None:
    stage_rows, transfer_rows = _aligned_stage_and_transfer_rows()

    result = validate_stage_transfer_alignment(
        stage_rows,
        transfer_rows,
        expected_layer_protocols={3: "dense", 4: "moe"},
        expected_moe_ep_size=1,
    )

    assert result["status"] == "FAIL"


def test_stage_transfer_alignment_accepts_multi_source_ffn_aggregate() -> None:
    stage_rows = [
        {
            "batch_id": 34,
            "cluster_type": "DECODE_ATTN",
            "execution_scope": "FULL_STAGE_WORLD",
            "request_ids": ["0"],
            "request_runtime_epochs": [0],
            "iteration_ids": [0],
            "layer_id": 4,
            "afd_stage_idx": 0,
            "replica_id": 0,
            "replica_local_id": None,
            "stage_start_ts": 0.99,
            "stage_end_ts": 1.0,
            "operation_kind": "full_stage",
        },
        {
            "batch_id": 36,
            "cluster_type": "DECODE_ATTN",
            "execution_scope": "FULL_STAGE_WORLD",
            "request_ids": ["1"],
            "request_runtime_epochs": [0],
            "iteration_ids": [0],
            "layer_id": 4,
            "afd_stage_idx": 0,
            "replica_id": 2,
            "replica_local_id": None,
            "stage_start_ts": 0.991,
            "stage_end_ts": 1.001,
            "operation_kind": "full_stage",
        },
    ]
    for lane_id in (0, 1):
        stage_rows.append(
            {
                "batch_id": 40 + lane_id,
                "cluster_type": "DECODE_FFN",
                "execution_scope": "EP_WAVE_LANE",
                "request_ids": [str(500 + lane_id)],
                "request_runtime_epochs": [0],
                "iteration_ids": [0, 0],
                "layer_id": 4,
                "afd_stage_idx": 0,
                "replica_id": 1,
                "replica_local_id": lane_id,
                "stage_id": 0,
                "operation_id": 40,
                "schedule_epoch": 0,
                "source_request_ids": ["0", "1"],
                "source_request_runtime_epochs": [0, 0],
                "source_batch_ids": [34, 36],
                "source_batch_arrival_times": [1.002, 1.003],
                "source_group_ready_ts": 1.003,
                "stage_start_ts": 1.003,
                "stage_end_ts": 1.004,
                "stage_completion_observed_ts": 1.004,
                "stage_completion_observed_source": (
                    "ep_alltoall_combine_collective"
                ),
                "operation_kind": "ep_ffn",
            }
        )
    transfer_rows = [
        _m2n_row(
            transfer_id="m2n-34-a2f",
            batch_id=34,
            batch_global_id=34,
            request_ids=[0],
            start=1.0,
            end=1.002,
        ),
        _m2n_row(
            transfer_id="m2n-34-f2a",
            source_cluster="DECODE_FFN",
            target_cluster="DECODE_ATTN",
            source_replica_id=1,
            target_replica_id=0,
            batch_id=34,
            batch_global_id=34,
            request_ids=[0],
            start=1.004,
            end=1.006,
            ffn_source_lane=0,
        ),
        _m2n_row(
            transfer_id="m2n-36-a2f",
            source_replica_id=2,
            target_replica_id=1,
            batch_id=36,
            batch_global_id=36,
            request_ids=[1],
            start=1.001,
            end=1.003,
        ),
        _m2n_row(
            transfer_id="m2n-36-f2a",
            source_cluster="DECODE_FFN",
            target_cluster="DECODE_ATTN",
            source_replica_id=1,
            target_replica_id=2,
            batch_id=36,
            batch_global_id=36,
            request_ids=[1],
            start=1.004,
            end=1.006,
            ffn_source_lane=1,
        ),
    ]

    result = validate_stage_transfer_alignment(
        stage_rows,
        transfer_rows,
        expected_layer_protocols={4: "moe"},
        expected_moe_ep_size=2,
    )

    assert result["status"] == "PASS"


class _LedgerConfig:
    write_metrics = True
    enable_op_level_tracing = False
    subsamples = None
    save_table_to_wandb = False
    store_plots = False
    output_dir = "."


class _Series:
    def put(self, *_args, **_kwargs) -> None:
        return None


def _ledger_store(tmp_path: Path) -> MetricsStore:
    store = object.__new__(MetricsStore)
    store._config = _LedgerConfig()
    store._config.output_dir = str(tmp_path)
    store._transfer_ledger_next_id = 0
    store._transfer_ledger_rows = []
    store._transfer_ledger_rows_by_info_id = {}
    store._transfer_info_objects = {}
    store._trace_store = None
    store._kv_cache_transfer_metrics = {
        "transfer_count": 0,
        "total_transfer_time": 0.0,
        "total_data_transferred": 0,
        "transfer_times": _Series(),
        "transfer_sizes": _Series(),
    }
    return store


def _request_and_batch() -> tuple[SimpleNamespace, SimpleNamespace]:
    request = SimpleNamespace(id=7, current_decode_token_index=1, runtime_epoch=3)
    batch = SimpleNamespace(
        id=11,
        global_id=22,
        requests=[request],
        request_runtime_epochs=[3],
    )
    return request, batch


def test_m2n_ledger_binds_target_after_end_without_guessing(tmp_path: Path) -> None:
    _, batch = _request_and_batch()
    transfer_info = SimpleNamespace(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=0,
        source_replica_local_id=None,
        source_execution_replica_id=0,
        source_execution_replica_local_id=None,
        target_execution_replica_id=None,
        target_execution_replica_local_id=None,
        target_ffn_replica_id=None,
        layer_id=4,
        afd_stage_idx=0,
        pipeline_stage="attn_to_ffn",
        activation_size_bytes=4096,
    )
    store = _ledger_store(tmp_path)

    store.on_m2n_transfer_start(
        1.0,
        0,
        ClusterType.DECODE_ATTN,
        ClusterType.DECODE_FFN,
        4096,
        transfer_info,
    )
    store.on_m2n_transfer_end(
        1.002,
        2.0,
        4096,
        ClusterType.DECODE_ATTN,
        ClusterType.DECODE_FFN,
        transfer_info,
    )

    row = store._transfer_ledger_rows[0]
    assert row["status"] == "pending_target"
    assert row["target_replica_id"] is None
    assert row["target_bound"] is False

    transfer_info.target_execution_replica_id = 1
    store.on_m2n_transfer_target_bound(transfer_info)

    assert row["target_replica_id"] == 1
    assert row["target_bound"] is True
    assert row["status"] == "completed"
    assert row["request_runtime_epochs"] == [3]
    assert id(transfer_info) not in store._transfer_ledger_rows_by_info_id
    assert id(transfer_info) not in store._transfer_info_objects
    store._write_frontier_transfer_ledger()
    persisted = json.loads(
        (tmp_path / "frontier_transfer_ledger.jsonl").read_text().splitlines()[0]
    )
    assert persisted["target_replica_id"] == 1
    assert persisted["target_bound"] is True


def test_m2n_ledger_rejects_scheduler_mapping_without_physical_target(
    tmp_path: Path,
) -> None:
    _, batch = _request_and_batch()
    transfer_info = SimpleNamespace(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=0,
        source_replica_local_id=None,
        source_execution_replica_id=0,
        source_execution_replica_local_id=None,
        target_execution_replica_id=None,
        target_execution_replica_local_id=None,
        target_ffn_replica_id=1,
        layer_id=4,
        afd_stage_idx=0,
        pipeline_stage="attn_to_ffn",
        activation_size_bytes=4096,
    )
    store = _ledger_store(tmp_path)

    store.on_m2n_transfer_start(
        1.0,
        0,
        ClusterType.DECODE_ATTN,
        ClusterType.DECODE_FFN,
        4096,
        transfer_info,
    )

    with pytest.raises(ValueError, match="explicit physical target Replica"):
        store.on_m2n_transfer_target_bound(transfer_info)


@pytest.mark.parametrize(
    "target_ffn_replica_id",
    [None, 9],
    ids=["batch-fallback", "wrong-direction-ffn-target"],
)
def test_m2n_ledger_rejects_missing_explicit_target_identity(
    tmp_path: Path,
    target_ffn_replica_id: int | None,
) -> None:
    _, batch = _request_and_batch()
    batch.decode_attn_original_replica_id = 9
    batch.decode_attn_original_replica_local_id = None
    transfer_info = SimpleNamespace(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        source_replica_id=0,
        source_replica_local_id=None,
        source_execution_replica_id=1,
        source_execution_replica_local_id=0,
        target_execution_replica_id=None,
        target_execution_replica_local_id=None,
        target_ffn_replica_id=target_ffn_replica_id,
        layer_id=4,
        afd_stage_idx=0,
        pipeline_stage="ffn_to_attn",
        activation_size_bytes=4096,
    )
    store = _ledger_store(tmp_path)

    store.on_m2n_transfer_start(
        1.0,
        1,
        ClusterType.DECODE_FFN,
        ClusterType.DECODE_ATTN,
        4096,
        transfer_info,
    )

    row = store._transfer_ledger_rows[0]
    assert row["target_replica_id"] is None
    with pytest.raises(ValueError, match="explicit physical target Replica"):
        store.on_m2n_transfer_target_bound(transfer_info)


def test_transfer_ledger_preserves_full_duration_precision(tmp_path: Path) -> None:
    _, batch = _request_and_batch()
    transfer_info = SimpleNamespace(
        batch=batch,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=0,
        source_replica_local_id=None,
        source_execution_replica_id=0,
        source_execution_replica_local_id=None,
        target_execution_replica_id=None,
        target_execution_replica_local_id=None,
        target_ffn_replica_id=None,
        layer_id=4,
        afd_stage_idx=0,
        pipeline_stage="attn_to_ffn",
        activation_size_bytes=4096,
    )
    store = _ledger_store(tmp_path)
    duration_ms = 0.123456789491

    store._register_transfer_ledger_start(
        transfer_info=transfer_info,
        transfer_kind="m2n",
        source_replica_id=0,
        source_replica_local_id=None,
        target_cluster_type=ClusterType.DECODE_FFN,
        byte_count=4096,
        start_time=1.0,
    )
    store._complete_transfer_ledger_end(
        transfer_info=transfer_info,
        end_time=1.0 + duration_ms / 1000.0,
        duration_ms=duration_ms,
    )

    assert store._transfer_ledger_rows[0]["duration_ms"] == duration_ms


def test_kv_ledger_binds_target_on_first_target_stage(tmp_path: Path) -> None:
    _, batch = _request_and_batch()
    transfer_info = SimpleNamespace(
        batch=batch,
        source_cluster_type=ClusterType.PREFILL,
        target_cluster_type=ClusterType.DECODE,
        source_replica_id=0,
        source_replica_local_id=None,
        target_replica_id=None,
        target_replica_local_id=None,
        kv_cache_size_bytes=8192,
    )
    store = _ledger_store(tmp_path)

    store.on_kv_cache_transfer_start(
        1.0,
        0,
        None,
        ClusterType.DECODE,
        8192,
        transfer_info,
    )
    store.on_kv_cache_transfer_end(
        1.004,
        4.0,
        8192,
        ClusterType.DECODE,
        transfer_info,
    )
    row = store._transfer_ledger_rows[0]
    assert row["status"] == "pending_target"

    store._bind_pending_kv_transfer_target(
        time=1.005,
        batch_stage=SimpleNamespace(
            request_ids=[7],
            request_runtime_epochs=[3],
        ),
        cluster_type=ClusterType.DECODE,
        replica_id=2,
        replica_local_id=None,
    )

    assert row["target_replica_id"] == 2
    assert row["target_replica_local_id"] is None
    assert row["status"] == "completed"


def test_kv_ledger_does_not_bind_different_runtime_epoch(
    tmp_path: Path,
) -> None:
    _, batch = _request_and_batch()
    transfer_info = SimpleNamespace(
        batch=batch,
        source_cluster_type=ClusterType.PREFILL,
        target_cluster_type=ClusterType.DECODE,
        source_replica_id=0,
        source_replica_local_id=None,
        target_replica_id=None,
        target_replica_local_id=None,
        kv_cache_size_bytes=8192,
    )
    store = _ledger_store(tmp_path)
    store.on_kv_cache_transfer_start(
        1.0,
        0,
        None,
        ClusterType.DECODE,
        8192,
        transfer_info,
    )
    store.on_kv_cache_transfer_end(
        1.004,
        4.0,
        8192,
        ClusterType.DECODE,
        transfer_info,
    )
    row = store._transfer_ledger_rows[0]

    store._bind_pending_kv_transfer_target(
        time=1.005,
        batch_stage=SimpleNamespace(
            request_ids=[7],
            request_runtime_epochs=[4],
        ),
        cluster_type=ClusterType.DECODE,
        replica_id=2,
        replica_local_id=None,
    )

    assert row["target_replica_id"] is None
    assert row["target_bound"] is False
    assert row["status"] == "pending_target"


def test_kv_ledger_rejects_ambiguous_duplicate_request_cohort(
    tmp_path: Path,
) -> None:
    """A target stage must not bind two transfers with the same request epoch."""

    _, first_batch = _request_and_batch()
    _, second_batch = _request_and_batch()
    second_batch.id = 12
    second_batch.global_id = 23
    transfers = [
        SimpleNamespace(
            batch=batch,
            source_cluster_type=ClusterType.PREFILL,
            target_cluster_type=ClusterType.DECODE,
            source_replica_id=index,
            source_replica_local_id=None,
            target_replica_id=None,
            target_replica_local_id=None,
            kv_cache_size_bytes=8192,
        )
        for index, batch in enumerate((first_batch, second_batch))
    ]
    store = _ledger_store(tmp_path)

    for transfer_info in transfers:
        store.on_kv_cache_transfer_start(
            1.0,
            transfer_info.source_replica_id,
            None,
            ClusterType.DECODE,
            8192,
            transfer_info,
        )
        store.on_kv_cache_transfer_end(
            1.004,
            4.0,
            8192,
            ClusterType.DECODE,
            transfer_info,
        )

    with pytest.raises(ValueError, match="ambiguous KV transfer target"):
        store._bind_pending_kv_transfer_target(
            time=1.005,
            batch_stage=SimpleNamespace(
                request_ids=[7],
                request_runtime_epochs=[3],
            ),
            cluster_type=ClusterType.DECODE,
            replica_id=2,
            replica_local_id=None,
        )

    assert all(row["target_bound"] is False for row in store._transfer_ledger_rows)

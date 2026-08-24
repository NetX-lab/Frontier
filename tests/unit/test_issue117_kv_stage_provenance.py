from __future__ import annotations

from copy import deepcopy

import pytest

from tests.e2e.transfer_metrics_contract import validate_kv_stage_alignment


def _provenance_bound_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    stage_rows = [
        {
            "batch_id": 10,
            "batch_stage_id": 101,
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
            "batch_stage_id": 202,
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
            "source_batch_stage_id": 101,
            "target_batch_stage_id": 202,
            "target_bound": True,
            "bytes": 8192,
            "start_ts_s": 1.0,
            "end_ts_s": 1.004,
            "duration_ms": 4.0,
            "status": "completed",
        }
    ]
    return stage_rows, transfer_rows


def test_kv_alignment_requires_explicit_stage_foreign_keys() -> None:
    stage_rows, transfer_rows = _provenance_bound_rows()
    stage_rows = deepcopy(stage_rows)
    transfer_rows = deepcopy(transfer_rows)
    del stage_rows[0]["batch_stage_id"]
    del transfer_rows[0]["source_batch_stage_id"]

    result = validate_kv_stage_alignment(stage_rows, transfer_rows)

    assert result["status"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda stages, transfers: transfers[0].update(source_batch_stage_id=999),
        lambda stages, transfers: transfers[0].update(target_batch_stage_id=999),
        lambda stages, transfers: stages[1].update(batch_stage_id=999),
    ],
    ids=["source-transfer-mismatch", "target-transfer-mismatch", "target-stage-mismatch"],
)
def test_kv_alignment_rejects_stage_foreign_key_mutation(mutation) -> None:
    stage_rows, transfer_rows = _provenance_bound_rows()
    mutation(stage_rows, transfer_rows)

    result = validate_kv_stage_alignment(stage_rows, transfer_rows)

    assert result["status"] == "FAIL"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda stages, transfers: stages[0].update(stage_start_ts=1.1),
        lambda stages, transfers: stages[1].update(stage_end_ts=1.003),
    ],
    ids=["negative-source-duration", "negative-target-duration"],
)
def test_kv_alignment_rejects_negative_stage_interval(mutation) -> None:
    stage_rows, transfer_rows = _provenance_bound_rows()
    mutation(stage_rows, transfer_rows)

    result = validate_kv_stage_alignment(stage_rows, transfer_rows)

    assert result["status"] == "FAIL"

"""Simulator regression for attention-DP lanes sharing pipeline stages."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.e2e.stage_admission_matrix import ATTN_DP_LANE, SUCCESS, build_cases, read_ledger, run_case_in_child

REPO_ROOT = Path(__file__).resolve().parents[2]
SET_NAME = "test"


def run_case(root, case_id):
    """Run one matrix case in its own process, because ``IS_MOE`` is process-global."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(root), case_id],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "WANDB_DISABLED": "true",
             "VIDUR_DISABLE_WANDB": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
    )
    (root / "run.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-15000:]
    case_dir = root / SET_NAME / case_id
    outcome = json.loads((case_dir / "outcome.json").read_text())
    assert outcome["outcome"] == SUCCESS, outcome
    return case_dir / "metrics"


@pytest.mark.parametrize("case_id, expected", [
    # (requests, prefill tokens, decode tokens): 16 prompt tokens and one output token each.
    ("G3a-moe-dp2-pp2-n4", (4, 64, 4)),
    ("G3a-moe-dp4-pp2-n8", (8, 128, 8)),
])
def test_moe_lanes_complete_every_request(tmp_path, case_id, expected):
    metrics_dir = run_case(tmp_path, case_id)
    with next(metrics_dir.rglob("request_metrics.csv")).open() as handle:
        rows = list(csv.DictReader(handle))
    observed = (
        len(rows),
        sum(int(float(row["request_num_prefill_tokens"])) for row in rows),
        sum(int(float(row["request_num_decode_tokens"])) for row in rows),
    )
    assert observed == expected


def test_dense_lanes_start_in_the_same_first_forward(tmp_path):
    """Every request arrives at t=0 and the stage has room for both lanes."""
    rows = read_ledger(run_case(tmp_path, "G4-dense-dp2-pp2-n8"))
    first_start = {}
    for row in sorted(rows, key=lambda row: row["stage_start_ts"]):
        if row["execution_scope"] == ATTN_DP_LANE and row["stage_id"] == 0:
            first_start.setdefault(row["replica_local_id"], row["stage_start_ts"])
    assert sorted(first_start) == [0, 1]
    assert first_start[0] == first_start[1]


if __name__ == "__main__":
    cases = {case.case_id: case for case in build_cases()}
    run_case_in_child(cases[sys.argv[2]], Path(sys.argv[1]), SET_NAME)

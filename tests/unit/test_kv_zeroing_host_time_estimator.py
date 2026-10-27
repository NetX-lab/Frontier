"""The KV side-storage reset host-time estimator on synthetic host-step probe records."""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.config import ReplicaConfig, global_vars
from frontier.execution_time_predictor.kv_block_zeroing_host_time import KVBlockZeroingHostTime
from frontier.types import ClusterType
from tests.integration.step5_pd import kv_zeroing_host_time as estimator

REPO = Path(__file__).resolve().parents[2]
MS = 1_000_000
WALL = 1_000 * 10**9
COST = 0.2 * MS
POST = 1 * MS


def collective(processes, ordinal, engines, tp_size, start):
    """One DP all-reduce per TP rank: engine `dp` arrives `arrive` ms after `start`.

    `engines` maps dp to (kind, arrive_ms, pre_ms per TP rank, reqs); kind is "reset", "plain"
    or "dummy". The collective ends `COST` after the last arrival, the forward launches `POST`
    later, and a resetting rank's first `.tolist()` returns `pre` before its arrival.
    """
    for tp in range(tp_size):
        arrivals = {dp: start + arrive * MS for dp, (_, arrive, _, _) in engines.items()}
        exit_ = max(arrivals.values()) + COST
        for dp, (kind, _, pre, reqs) in engines.items():
            records = processes[dp, tp]
            record = {"kind": "step", "seq": len(records) + 1, "wall_ns": WALL + start,
                      "dummy": kind == "dummy", "profile": False, "reqs": reqs, "tokens": reqs,
                      "sync": {"ordinal": ordinal, "enter": arrivals[dp], "exit": exit_},
                      "launch": exit_ + POST, "gap_ms": 9.0, "span_ms": 40.0}
            if kind == "reset":
                record["zero"] = {"hooks": 23, "groups": [2] * 23, "prior_pending": True,
                                  "first_sync": arrivals[dp] - pre[tp] * MS}
            records.append(record)


def empty(dp_size, tp_size):
    return {(dp, tp): [] for dp in range(dp_size) for tp in range(tp_size)}


def test_local_host_time_excludes_the_wait_for_other_engines():
    processes = empty(2, 2)
    # Engine 0 resets for 5 ms (TP rank 1: 5.5 ms); engine 1 reaches the collective 8 ms later.
    collective(processes, 1, {0: ("reset", 0, (5, 5.5), 4), 1: ("plain", 8, None, 4)}, 2, 0)
    # Engine 0 arrives last this time; its wait is 0 and its value is unchanged.
    collective(processes, 2, {0: ("reset", 3, (5, 5.5), 4), 1: ("dummy", 0, None, 0)}, 2, 50 * MS)
    steps, join = estimator.engine_steps(
        {key: {"records": records} for key, records in processes.items()}, 2, 2, [(0, 2 * WALL / 1e9)])
    assert join == {"joined_all_reduces": 2, "unjoined_all_reduces": 0}
    # The dummy participant leaves no step; the plain step carries no host time.
    assert [(step["dp"], step["ordinal"], step["resetting"]) for step in steps] == [
        (0, 1, True), (1, 1, False), (0, 2, True)]
    for step in steps[0], steps[2]:
        assert step["host_ms"] == pytest.approx(5.5 + 0.2 + 1)
        assert step["rank0_host_ms"] == pytest.approx(5 + 0.2 + 1)
        assert step["tp_spread_ms"] == pytest.approx(0.5)
        assert step["hooks"] == [23]
    assert steps[0]["wait_ms"] == pytest.approx(8)
    assert steps[2]["wait_ms"] == pytest.approx(0)
    # The device gap subtracts the same wait; the first step has no previous span.
    assert "device_gap_ms" not in steps[0]
    assert steps[2]["device_gap_ms"] == pytest.approx(9.0)


def test_unmatched_all_reduces_windows_and_inconsistent_ranks():
    processes = empty(2, 1)
    collective(processes, 1, {0: ("reset", 0, (5,), 4), 1: ("plain", 1, None, 4)}, 1, 0)
    collective(processes, 2, {0: ("reset", 0, (5,), 4), 1: ("plain", 1, None, 4)}, 1, 10**9)
    collective(processes, 3, {0: ("reset", 0, (5,), 4), 1: ("plain", 1, None, 4)}, 1, 2 * 10**9)
    del processes[1, 0][1]
    steps, join = estimator.engine_steps(
        {key: {"records": records} for key, records in processes.items()}, 2, 1,
        [(WALL / 1e9 + 1.5, WALL / 1e9 + 2.5)])
    assert join == {"joined_all_reduces": 2, "unjoined_all_reduces": 1}
    assert [step["ordinal"] for step in steps] == [3, 3]

    processes = empty(1, 2)
    collective(processes, 1, {0: ("reset", 0, (5, 5), 4)}, 2, 0)
    processes[0, 1][0]["reqs"] = 3
    with pytest.raises(ValueError, match="TP ranks disagree"):
        estimator.engine_steps({key: {"records": records} for key, records in processes.items()},
                               1, 2, [(0, 2 * WALL / 1e9)])


def test_a_role_needs_one_clean_process_per_engine_and_rank():
    header = {"kind": "header", "dp_size": 1, "tp_size": 2}
    process = {"path": "p", "header": header, "records": []}
    complete = {("decode", 0, 0): [process], ("decode", 0, 1): [process]}
    assert set(estimator.role_processes(complete, "decode", 1, 2)) == {(0, 0), (0, 1)}
    with pytest.raises(ValueError, match="differ from DP1 x TP2"):
        estimator.role_processes({("decode", 0, 0): [process]}, "decode", 1, 2)
    with pytest.raises(ValueError, match="more than one process generation"):
        estimator.role_processes(complete | {("decode", 0, 1): [process, process]}, "decode", 1, 2)
    failed = {"path": "q", "header": header, "records": [{"kind": "error"}]}
    with pytest.raises(ValueError, match="disabled itself"):
        estimator.role_processes(complete | {("decode", 0, 1): [failed]}, "decode", 1, 2)


def summary(ttft, tpot):
    return {"repeat_metric_means": {"ttft_ms": ttft, "tpot_ms": tpot},
            "repeat_variation": {"ttft_ms": {"min": 907.0, "max": 915.0},
                                 "tpot_ms": {"min": 53.2, "max": 53.4}}}


def test_perturbation_compares_with_the_clean_repeat_range():
    clean = summary(912.0, 53.3)
    assert estimator.perturbation(summary(919.0, 53.4), clean)["status"] == "PASS"
    result = estimator.perturbation(summary(912.0, 53.6), clean)
    assert result["status"] == "PERTURBED"
    assert result["metrics"]["tpot_ms"]["within"] is False


def stage(tmp_path, steps_per_role):
    """An h200_32 host_probe stage: DP2 x TP8 per role, each step resetting on engine 0."""
    root = tmp_path / "stage"
    config = {"canonical_replay": {"kind": "host_probe"}, "case_id": "h200_32",
              "source": {"commit": "fixture"},
              "tool_hashes": {"host_probe_bootstrap/host_step_probe.py": "fixture"},
              "engine": {"tensor_parallel_size": 8, "data_parallel_size": 2, "pipeline_parallel_size": 1,
                         "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"}}}
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    for role, (count, pre) in steps_per_role.items():
        processes = empty(2, 8)
        for ordinal in range(1, count + 1):
            collective(processes, ordinal, {0: ("reset", 0, (pre,) * 8, 4), 1: ("plain", 2, None, 4)},
                       8, ordinal * 50 * MS)
        for (dp, tp), records in processes.items():
            directory = root / "evidence/nodes" / f"{role}-{dp}/host_probe"
            directory.mkdir(parents=True, exist_ok=True)
            header = {"kind": "header", "role": role, "dp_rank": dp, "tp_rank": tp, "dp_size": 2, "tp_size": 8}
            with gzip.open(directory / f"{role}_{tp}.jsonl.gz", "wt") as stream:
                stream.writelines(json.dumps(line) + "\n" for line in [header, *records])
    formal = {"origin_monotonic": 0.0, "origin_wall": WALL / 1e9}
    probe = summary(912.0, 53.3) | {"repetitions": [{"phases": {"formal": formal}, "metrics": {
        "first_submission_monotonic": 0.0, "last_output_monotonic": 100.0}}]}
    (root / "evidence/replay").mkdir(parents=True)
    (root / "evidence/replay/canonical_summary.json").write_text(json.dumps(probe))
    (tmp_path / "clean.json").write_text(json.dumps(summary(912.0, 53.3)))
    return root


def run(tmp_path, root):
    return subprocess.run(
        [sys.executable, str(REPO / "tests/integration/step5_pd/kv_zeroing_host_time.py"),
         "--case", "h200_32", "--stage-dir", str(root), "--clean-summary", str(tmp_path / "clean.json"),
         "--job-id", "fixture", "--min-samples", "3", "--output", str(tmp_path / "profile.json"),
         "--report", str(tmp_path / "report.json")],
        capture_output=True, text=True, env=os.environ | {"PYTHONPATH": str(REPO)}, check=False)


def test_the_profile_loads_for_the_run_it_was_measured_on(tmp_path, monkeypatch):
    result = run(tmp_path, stage(tmp_path, {"prefill": (3, 9), "decode": (4, 4)}))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["perturbation"]["status"] == "PASS"
    assert report["roles"]["decode"]["steps"] == {"resetting": 4, "non_resetting": 4}
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
    replica = ReplicaConfig(model_name="stepfun-ai/Step-5-Preview", device="h200", network_device="h200_dgx",
                            attn_tensor_parallel_size=8, attn_dp=2, moe_tensor_parallel_size=1,
                            moe_expert_parallel_size=16, attention_backend="STEP5_DSA", kv_cache_dtype="auto")
    for cluster_type, expected in ((ClusterType.PREFILL, 9 + 0.2 + 1), (ClusterType.DECODE, 4 + 0.2 + 1)):
        host_time = KVBlockZeroingHostTime.from_file(
            tmp_path / "profile.json", replica_config=replica, cluster_type=cluster_type,
            predictor_config=SimpleNamespace(skip_cpu_overhead_modeling=True))
        assert host_time.time_ms(4) == pytest.approx(expected)


def test_too_few_resetting_steps_write_the_report_but_no_profile(tmp_path):
    result = run(tmp_path, stage(tmp_path, {"prefill": (2, 9), "decode": (4, 4)}))
    assert result.returncode != 0
    assert "{'prefill': 2} are fewer than 3; no profile written" in result.stderr
    assert (tmp_path / "report.json").exists()
    assert not (tmp_path / "profile.json").exists()

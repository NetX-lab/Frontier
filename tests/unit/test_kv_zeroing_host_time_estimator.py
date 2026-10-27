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
COST = 0.2 * MS  # the all_reduce's own cost
PREP = 0.3 * MS  # local tensor preparation before the all_reduce
POST = 1 * MS  # input preparation from the sync's return to the launch


def collective(processes, ordinal, engines, tp_size, start):
    """One DP sync per TP rank, at `start` ns after the probe clock's origin.

    `engines` maps dp to (kind, arrive_ms, pre_ms, reqs, dispatch_ms). kind is "reset",
    "plain" or "dummy"; the engine reaches the all_reduce `arrive_ms` (a number or one per TP
    rank) after `start` and runs `dispatch_ms` of local work after it. The all_reduce ends
    `COST` after the last arrival. A resetting rank's first `.tolist()` returns `pre_ms` (per
    TP rank) before its sync begins.
    """
    for tp in range(tp_size):
        arrivals = {dp: start + (arrive[tp] if isinstance(arrive, tuple) else arrive) * MS
                    for dp, (_, arrive, _, _, _) in engines.items()}
        reduce_exit = max(arrivals.values()) + COST
        for dp, (kind, _, pre, reqs, dispatch) in engines.items():
            records = processes[dp, tp]
            enter, exit_ = arrivals[dp] - PREP, reduce_exit + dispatch * MS
            record = {"kind": "step", "seq": len(records) + 1, "wall_ns": WALL + start,
                      "dummy": kind == "dummy", "profile": False, "reqs": reqs, "tokens": reqs,
                      "sync": {"ordinal": ordinal, "enter": enter, "reduce_enter": arrivals[dp],
                               "reduce_exit": reduce_exit, "exit": exit_},
                      "launch": exit_ + POST, "gap_ms": 9.0, "span_ms": 40.0}
            if kind == "reset":
                record["zero"] = {"hooks": 23, "groups": [2] * 23, "prior_pending": True,
                                  "first_sync": enter - pre[tp] * MS}
            records.append(record)


def empty(dp_size, tp_size):
    return {(dp, tp): [] for dp in range(dp_size) for tp in range(tp_size)}


def joined(processes):
    return {key: {"records": records} for key, records in processes.items()}


def steps_of(processes, dp_size, tp_size, windows=((0, 2 * WALL / 1e9),)):
    return estimator.engine_steps(joined(processes), dp_size, tp_size, list(windows))


def test_local_host_time_excludes_the_wait_for_other_engines():
    processes = empty(2, 2)
    # Engine 0 resets for 5 ms (TP rank 1: 5.5 ms); engine 1 reaches the all_reduce 8 ms later.
    collective(processes, 1, {0: ("reset", 0, (5, 5.5), 4, 0), 1: ("plain", 8, None, 4, 0)}, 2, 0)
    # Engine 0 arrives last this time; its wait is 0 and its value is unchanged.
    collective(processes, 2, {0: ("reset", 3, (5, 5.5), 4, 0), 1: ("dummy", 0, None, 0, 0)}, 2, 50 * MS)
    steps, join = steps_of(processes, 2, 2)
    assert join == {"joined_all_reduces": 2, "unjoined_all_reduces": 0}
    # The dummy participant leaves no step; the plain step carries no host time.
    assert [(step["dp"], step["ordinal"], step["resetting"]) for step in steps] == [
        (0, 1, True), (1, 1, False), (0, 2, True)]
    for step in steps[0], steps[2]:
        assert step["host_ms"] == pytest.approx(5.5 + 0.3 + 0.2 + 1)
        assert step["rank_host_ms"] == pytest.approx(5.5 + 0.3 + 0.2 + 1)
        assert step["rank0_host_ms"] == pytest.approx(5 + 0.3 + 0.2 + 1)
        assert step["first_sync_skew_ms"] == pytest.approx(0.5)
        assert step["hooks"] == [23] and step["waited"]
    assert steps[0]["wait_ms"] == pytest.approx(8)
    assert steps[2]["wait_ms"] == pytest.approx(0)
    # The device gap subtracts the same wait; the first step has no previous span.
    assert "device_gap_ms" not in steps[0]
    assert steps[2]["device_gap_ms"] == pytest.approx(9.0)


def test_local_work_around_the_all_reduce_stays_in_the_host_time():
    processes = empty(2, 1)
    # Both engines reach the all_reduce together; engine 0 dispatches for 2 ms after it.
    collective(processes, 1, {0: ("reset", 0, (5,), 4, 2), 1: ("reset", 0, (5,), 4, 0)}, 1, 0)
    # Equal local work, engine 1 arrives 3 ms later: only engine 0 waited.
    collective(processes, 2, {0: ("reset", 0, (5,), 4, 0), 1: ("reset", 3, (5,), 4, 0)}, 1, 50 * MS)
    steps, _ = steps_of(processes, 2, 1)
    assert [(step["ordinal"], step["dp"]) for step in steps] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert [step["wait_ms"] for step in steps] == pytest.approx([0, 0, 3, 0])
    assert [step["host_ms"] for step in steps] == pytest.approx([8.5, 6.5, 6.5, 6.5])


def test_the_engine_interval_spans_its_earliest_first_sync_to_its_last_launch():
    processes = empty(2, 2)
    # Engine 0: TP rank 0 waits for the device and reaches the all_reduce at 0 ms; TP rank 1's host
    # returns from its first .tolist() 5 ms later and reaches the all_reduce at 2 ms, with engine 1.
    collective(processes, 1, {0: ("reset", (0, 2), (3, 0), 4, 0), 1: ("plain", 2, None, 4, 0)}, 2, 0)
    [step, _] = steps_of(processes, 2, 2)[0]
    # The device idles from -3.3 ms until TP rank 1 launches at 3.2 ms; rank 0's 2 ms is a wait.
    assert step["host_ms"] == pytest.approx(6.5)
    assert step["rank_host_ms"] == pytest.approx(4.5)
    assert step["first_sync_skew_ms"] == pytest.approx(5.0)
    assert step["wait_ms"] == pytest.approx(2)


def test_unmatched_all_reduces_windows_and_inconsistent_ranks():
    processes = empty(2, 1)
    for ordinal in (1, 2, 3):
        collective(processes, ordinal, {0: ("reset", 0, (5,), 4, 0), 1: ("plain", 1, None, 4, 0)}, 1,
                   (ordinal - 1) * 10**9)
    del processes[1, 0][1]
    steps, join = steps_of(processes, 2, 1, [(WALL / 1e9 + 1.5, WALL / 1e9 + 2.5)])
    assert join == {"joined_all_reduces": 2, "unjoined_all_reduces": 1}
    assert [step["ordinal"] for step in steps] == [3, 3]

    processes = empty(1, 2)
    collective(processes, 1, {0: ("reset", 0, (5, 5), 4, 0)}, 2, 0)
    processes[0, 1][0]["reqs"] = 3
    with pytest.raises(ValueError, match="TP ranks disagree"):
        steps_of(processes, 1, 2)


def process(path, dp, tp, records, host="node0", steps=None):
    header = {"kind": "header", "host": host, "dp_size": 1, "tp_size": 2, "dp_rank": dp, "tp_rank": tp}
    end = [] if steps is False else [{"kind": "end", "steps": len(records) if steps is None else steps}]
    return {"path": path, "header": header, "records": records + end}


def test_a_role_needs_one_complete_process_per_engine_and_rank_on_one_node():
    step = {"kind": "step", "seq": 1}
    complete = {("decode", 0, tp): [process(f"p{tp}", 0, tp, [step])] for tp in range(2)}
    assert set(estimator.role_processes(complete, "decode", 1, 2)) == {(0, 0), (0, 1)}
    with pytest.raises(ValueError, match="differ from DP1 x TP2"):
        estimator.role_processes({("decode", 0, 0): complete["decode", 0, 0]}, "decode", 1, 2)
    with pytest.raises(ValueError, match="more than one process generation"):
        estimator.role_processes(complete | {("decode", 0, 1): complete["decode", 0, 1] * 2}, "decode", 1, 2)
    failed = process("q", 0, 1, [step, {"kind": "error"}])
    with pytest.raises(ValueError, match="disabled itself"):
        estimator.role_processes(complete | {("decode", 0, 1): [failed]}, "decode", 1, 2)
    for truncated in (process("q", 0, 1, [step], steps=False), process("q", 0, 1, [step], steps=2)):
        with pytest.raises(ValueError, match="q is truncated"):
            estimator.role_processes(complete | {("decode", 0, 1): [truncated]}, "decode", 1, 2)
    elsewhere = process("q", 0, 1, [step], host="node1")
    with pytest.raises(ValueError, match="clocks are not comparable"):
        estimator.role_processes(complete | {("decode", 0, 1): [elsewhere]}, "decode", 1, 2)


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


def idle(name, monotonic, verified=True):
    return {"boundary": name, "verified": verified, "observations": [{"monotonic": monotonic}]}


def probe_summary(boundaries):
    formal = {"origin_monotonic": 0.0, "origin_wall": WALL / 1e9}
    return summary(912.0, 53.3) | {"repetitions": [{"repeat": 1, "phases": {"formal": formal}}],
                                   "idle_boundaries": boundaries}


def test_formal_windows_lie_between_the_verified_idle_boundaries():
    boundaries = [idle("after_length_sweep", -9.0), idle("after_repeat01_warmup", -1.0),
                  idle("after_repeat01_formal", 100.0)]
    windows = estimator.formal_windows(probe_summary(boundaries))
    assert windows == [(WALL / 1e9 - 1.0, WALL / 1e9 + 100.0)]
    records = [{"kind": "step", "dummy": False, "wall_ns": WALL + 2 * 10**9},
               {"kind": "step", "dummy": True, "wall_ns": WALL - 10**9}]
    assert estimator.edge_clearance_s({"p": [{"records": records}]}, windows) == pytest.approx(3.0)
    boundaries[2] = idle("after_repeat01_formal", 100.0, verified=False)
    with pytest.raises(ValueError, match="after_repeat01_formal is unverified"):
        estimator.formal_windows(probe_summary(boundaries))


def stage(tmp_path, steps_per_role):
    """An h200_32 host_probe stage: DP2 x TP8 per role; engine 0 resets in every listed
    (pre_ms, reqs) step and reaches the all_reduce 2 ms before engine 1."""
    root = tmp_path / "stage"
    config = {"canonical_replay": {"kind": "host_probe"}, "case_id": "h200_32",
              "source": {"commit": "fixture"},
              "tool_hashes": {"host_probe_bootstrap/host_step_probe.py": "fixture"},
              "engine": {"tensor_parallel_size": 8, "data_parallel_size": 2, "pipeline_parallel_size": 1,
                         "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"}}}
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    for role, role_steps in steps_per_role.items():
        processes = empty(2, 8)
        for ordinal, (pre, reqs) in enumerate(role_steps, 1):
            collective(processes, ordinal, {0: ("reset", 0, (pre,) * 8, reqs, 0), 1: ("plain", 2, None, reqs, 0)},
                       8, ordinal * 50 * MS)
        for (dp, tp), records in processes.items():
            directory = root / "evidence/nodes" / f"{role}-{dp}/host_probe"
            directory.mkdir(parents=True, exist_ok=True)
            header = {"kind": "header", "role": role, "host": f"{role}-{dp}", "dp_rank": dp, "tp_rank": tp,
                      "dp_size": 2, "tp_size": 8}
            with gzip.open(directory / f"{role}_{tp}.jsonl.gz", "wt") as stream:
                stream.writelines(json.dumps(line) + "\n" for line in [
                    header, *records, {"kind": "end", "steps": len(records)}])
    probe = probe_summary([idle("after_repeat01_warmup", -1.0), idle("after_repeat01_formal", 100.0)])
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


def host_times(tmp_path, monkeypatch):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
    replica = ReplicaConfig(model_name="stepfun-ai/Step-5-Preview", device="h200", network_device="h200_dgx",
                            attn_tensor_parallel_size=8, attn_dp=2, moe_tensor_parallel_size=1,
                            moe_expert_parallel_size=16, attention_backend="STEP5_DSA", kv_cache_dtype="auto")
    return {cluster_type: KVBlockZeroingHostTime.from_file(
        tmp_path / "profile.json", replica_config=replica, cluster_type=cluster_type,
        predictor_config=SimpleNamespace(skip_cpu_overhead_modeling=True))
        for cluster_type in (ClusterType.PREFILL, ClusterType.DECODE)}


def test_the_profile_loads_for_the_run_it_was_measured_on(tmp_path, monkeypatch):
    # A spread of 0.9 ms over 4-40 requests stays below the trend limit: one value.
    decode = [(4, 4), (4.9, 40), (4, 4), (4.9, 40)]
    result = run(tmp_path, stage(tmp_path, {"prefill": [(9, 4)] * 3, "decode": decode}))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["perturbation"]["status"] == "PASS"
    assert report["window_edge_clearance_s"] == pytest.approx(1.05)
    assert report["roles"]["decode"]["steps"] == {"resetting": 4, "non_resetting": 4}
    assert report["roles"]["decode"]["request_trend_ms"] == pytest.approx(0.9)
    times = host_times(tmp_path, monkeypatch)
    assert times[ClusterType.PREFILL].time_ms(4) == pytest.approx(9 + 1.5)
    assert times[ClusterType.DECODE].time_ms(4) == pytest.approx(4.45 + 1.5)


def test_a_request_count_trend_above_the_limit_writes_a_table(tmp_path, monkeypatch):
    decode = [(4, 4), (8, 40), (4, 4), (8, 40)]
    result = run(tmp_path, stage(tmp_path, {"prefill": [(9, 4)] * 3, "decode": decode}))
    assert result.returncode == 0, result.stderr
    profile = json.loads((tmp_path / "profile.json").read_text())
    assert profile["roles"]["prefill"] == {"time_ms": pytest.approx(10.5)}
    assert profile["roles"]["decode"] == {"by_num_requests": [[4, pytest.approx(5.5)], [40, pytest.approx(9.5)]]}
    decode_time = host_times(tmp_path, monkeypatch)[ClusterType.DECODE]
    assert decode_time.time_ms(22) == pytest.approx(7.5)
    with pytest.raises(ValueError, match="covers 4-40 requests, not 41"):
        decode_time.time_ms(41)


def test_too_few_resetting_steps_write_the_report_but_no_profile(tmp_path):
    result = run(tmp_path, stage(tmp_path, {"prefill": [(9, 4)] * 2, "decode": [(4, 4)] * 4}))
    assert result.returncode != 0
    assert "{'prefill': 2} are fewer than 3; no profile written" in result.stderr
    assert (tmp_path / "report.json").exists()
    assert not (tmp_path / "profile.json").exists()

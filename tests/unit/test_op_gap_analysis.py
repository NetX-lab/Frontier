"""Join, peer-state, cycle-split and alignment rules of `tests/comparison/calibration/op_gap_analysis.py`."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.comparison.calibration.op_gap_analysis import (
    batch_windows,
    decode_cycles,
    first_formal_admission,
    measurement_cost,
    peer_states,
    schedule_alignment,
)


def boundary_row(batch_id: int, forward: tuple[float, float], request_ids: list[str]) -> dict:
    """A first-rank pp_boundary row on (pp_rank 0, dp_rank 0); times in seconds."""
    return {
        "pp_rank": 0, "tp_rank": 0, "dp_rank": 0, "batch_id": batch_id, "batch_size": len(request_ids),
        "num_prefill_tokens": 0, "num_decode_tokens": len(request_ids), "request_ids": request_ids,
        "preprocess_start_ts": forward[0] - 0.001, "preprocess_end_ts": forward[0],
        "forward_start_ts": forward[0], "forward_end_ts": forward[1],
        "send_start_ts": forward[1], "send_end_ts": forward[1] + 0.0005,
        "recv_start_ts": None, "recv_end_ts": None, "execute_end_ts": None,
    }


def log_row(batch_id: int, request_ids: list[str]) -> dict:
    return {
        "pp_rank": 0, "tp_rank": 0, "dp_rank": 0, "batch_id": batch_id, "batch_size": len(request_ids),
        "batch_num_prefill_tokens": 0, "batch_num_decode_tokens": len(request_ids),
        "request_ids": request_ids, "batch_execution_time_ms": 10.0,
    }


def scope_row(batch_id: int, op_name: str, scope_seq: int, cuda_time_ms: float) -> dict:
    return {
        "pp_rank": 0, "tp_rank": 0, "dp_rank": 0, "batch_id": batch_id, "batch_size": 1, "batch_num_prefill_tokens": 0,
        "batch_num_decode_tokens": 1, "op_name": op_name, "scope_seq": scope_seq,
        "cuda_time_ms": cuda_time_ms, "timestamp": 100.0 + batch_id,
    }


def test_batch_window_join_reports_duplicate_key():
    ids = ["cmpl-r1-0"]
    boundary = pd.DataFrame([boundary_row(0, (1.000, 1.010), ids), boundary_row(1, (1.020, 1.030), ["cmpl-w0-0"])])
    log = pd.DataFrame([log_row(0, ids), log_row(1, ["cmpl-w0-0"]), log_row(1, ["cmpl-w0-0"])])
    records = pd.DataFrame([
        scope_row(0, "input_layernorm", 0, 2.0), scope_row(0, "input_layernorm", 1, 1.0),
        scope_row(0, "expert_parallel_alltoall_dispatch", 0, 3.0), scope_row(1, "input_layernorm", 0, 1.0),
    ])
    windows, problems = batch_windows(records, log, boundary, warmup_ids={"w0"})

    first, second = windows.iloc[0], windows.iloc[1]
    assert first.forward_ms == pytest.approx(10.0)
    assert (first.scope_sum_ms, first.collective_sum_ms) == pytest.approx((3.0, 3.0))
    assert first.uncovered_ms == pytest.approx(4.0)
    assert first.n_scope_records == 3 and first.shape_class == "decode_b1"
    assert first.op_record_counts == '{"expert_parallel_alltoall_dispatch": 1, "input_layernorm": 2}'
    assert first.join_problems == "" and not first.has_warmup_request
    assert second.has_warmup_request
    assert second.join_problems == "duplicate key in batch_log"
    assert pd.isna(second.batch_execution_time_ms)
    assert problems == [{
        "source": "batch_log", "problem": "duplicate key", "keys": 1, "rows": 2,
        "examples": [{"pp_rank": 0, "dp_rank": 0, "batch_id": 1}],
    }]


def test_tp_allreduce_nested_in_projection_counts_once():
    ids = ["cmpl-r1-0"]
    boundary = pd.DataFrame([boundary_row(0, (1.000, 1.010), ids)])
    log = pd.DataFrame([log_row(0, ids)])
    records = pd.DataFrame([
        scope_row(0, "attn_post_proj", 0, 2.0), scope_row(0, "attn_post_proj_tp_allreduce", 0, 0.5),
        scope_row(0, "moe_tensor_parallel_allreduce", 0, 1.0),
    ])
    windows, problems = batch_windows(records, log, boundary, warmup_ids=set())

    first = windows.iloc[0]
    assert (first.scope_sum_ms, first.collective_sum_ms) == pytest.approx((1.5, 1.5))
    assert first.uncovered_ms == pytest.approx(7.0)
    assert problems == []


def window(dp_rank: int, batch_id: int, start: float, end: float, shape: str = "decode_b1") -> dict:
    return {
        "pp_rank": 0, "dp_rank": dp_rank, "batch_id": batch_id, "shape_class": shape, "has_warmup_request": False,
        "join_problems": "", "forward_start_ts": start, "forward_end_ts": end,
        "forward_ms": (end - start) * 1000.0, "collective_sum_ms": 1.0,
    }


def test_peer_state_classifies_batch_dummy_and_idle():
    windows = pd.DataFrame([
        window(0, 0, 0.0, 10.0), window(0, 1, 20.0, 30.0), window(0, 2, 40.0, 50.0),
        window(1, 0, 5.0, 15.0, shape="decode_b2"),
    ])
    # Engine 0 serves dp_rank 1, so its dummy passes are the peer of dp_rank 0.
    dummies = pd.DataFrame({"engine": [0, 1], "start_monotonic": [22.0, 41.0], "monotonic": [28.0, 49.0]})
    states = peer_states(windows, dummies, dp_of_engine={0: 1, 1: 0}).set_index(["dp_rank", "batch_id"])

    assert states.peer_state.to_dict() == {(0, 0): "batch", (0, 1): "dummy", (0, 2): "idle", (1, 0): "batch"}
    assert states.loc[(0, 0), "peer_shape_class"] == "decode_b2"
    assert states.loc[(0, 0), "peer_batch_id"] == 0
    assert states.loc[(0, 0), "overlap_fraction"] == pytest.approx(0.5)
    assert states.loc[(0, 1), "overlap_fraction"] == pytest.approx(0.6)
    assert states.loc[(0, 2), "overlap_fraction"] == 0.0
    assert states.loc[(1, 0), "peer_shape_class"] == "decode_b1"


def iteration(engine: int, seq: int, monotonic: float, branch: str, running: int, tokens: int, new: str = "[]") -> dict:
    return {
        "engine": engine, "seq": seq, "monotonic": monotonic, "branch": branch, "running": running,
        "num_scheduled_tokens": tokens, "scheduled_new_req_ids": new,
    }


def engine_sequence(scale: float) -> pd.DataFrame:
    """Engine 0 runs two decode cycles; engine 1 has two running requests from 0.105 s."""
    applied = "applied_after_scheduling"
    return pd.DataFrame([
        iteration(0, 0, 0.000, "scheduled_without_applying", 1, 64, '["cmpl-a-0"]'),
        iteration(0, 1, 0.100 * scale, applied, 1, 0),
        iteration(0, 2, 0.110 * scale, applied, 1, 1),
        iteration(0, 3, 0.130 * scale, applied, 1, 0),
        iteration(0, 4, 0.140 * scale, applied, 1, 1),
        iteration(0, 5, 0.160 * scale, applied, 1, 0),
        iteration(1, 0, 0.105 * scale, "scheduled_without_applying", 2, 2, '["cmpl-b-0", "cmpl-c-0"]'),
    ])


def test_cycle_split_and_measurement_cost():
    clean = decode_cycles(engine_sequence(scale=1.0))
    assert clean.seq.tolist() == [2, 4]
    assert clean.schedule_step_ms.tolist() == pytest.approx([10.0, 10.0])
    assert clean.blocking_step_ms.tolist() == pytest.approx([20.0, 20.0])
    assert clean.cycle_ms.tolist() == pytest.approx([30.0, 30.0])
    assert clean.other_running.tolist() == [2, 2]

    cost = measurement_cost(clean, decode_cycles(engine_sequence(scale=2.0)), min_count=2).set_index("metric")
    assert cost.loc["cycle_ms", ["running", "other_running", "clean_count", "op_count"]].tolist() == [1, 2, 2, 2]
    assert cost.loc["cycle_ms", "delta"] == pytest.approx(30.0)
    assert cost.loc["schedule_step_ms", "delta"] == pytest.approx(10.0)
    assert cost.loc["blocking_step_ms", "delta"] == pytest.approx(20.0)
    assert measurement_cost(clean, clean, min_count=3).empty
    assert first_formal_admission(engine_sequence(scale=1.0), warmup_ids={"a"}) == pytest.approx(0.105)


def test_alignment_reports_first_divergence():
    applied = "applied_after_scheduling"
    clean = pd.DataFrame([
        iteration(0, 0, 0.0, applied, 1, 32, '["cmpl-a-0"]'), iteration(0, 1, 0.1, applied, 1, 0),
        iteration(0, 2, 0.2, applied, 1, 1), iteration(0, 3, 0.3, applied, 2, 33, '["cmpl-b-0"]'),
        iteration(0, 4, 0.4, applied, 2, 2),
    ])
    op = pd.DataFrame([
        iteration(0, 0, 0.0, applied, 1, 32, '["cmpl-a-0"]'), iteration(0, 1, 0.1, applied, 1, 1),
        iteration(0, 2, 0.2, applied, 1, 1), iteration(0, 3, 0.3, applied, 2, 33, '["cmpl-b-0"]'),
        iteration(0, 4, 0.4, applied, 2, 2),
    ])
    summaries, opcodes = schedule_alignment(clean, op)

    assert summaries[0] == {
        "clean_length": 4, "op_length": 5, "matched_prefix": 2,
        "first_divergence": {"index": 2, "clean": ['["cmpl-b-0"]', 33, 2], "op": ["[]", 1, 1]},
        "equal_block_positions": 4,
    }
    assert opcodes.tag.tolist() == ["equal", "insert", "equal"]

from dataclasses import replace
from itertools import product
from types import SimpleNamespace
import json

import pytest

from frontier.validation.batch_record import BatchRecord, read_batches, validate_rank_cohorts
from frontier.validation.replay import replay_batches, summarize_errors
from frontier.validation.sglang_capture import decode_trajectory, snapshot_forward_batch
from frontier.validation.trace import extract_batch_kernels, interval_union_us


def record(**updates):
    fields = dict(batch_id="b", rank=0, phase="decode", request_ids=("a", "b", "c"),
                  query_lens=(1, 1, 1), context_lens=(1024, 2048, 4096),
                  prefill_mask=(False,) * 3, graph_mode="FULL", capture_size=4,
                  forward_gpu_ms=20.0, step_wall_ms=22.0)
    fields.update(updates)
    return BatchRecord(**fields)


def test_record_roundtrip_and_duplicate_detection(tmp_path):
    path = tmp_path / "batches.jsonl"
    path.write_text(json.dumps(record().to_dict()) + "\n")
    assert read_batches(path) == [record()]
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate batch/rank"):
        read_batches(path)


@pytest.mark.parametrize("change", [
    {"rank": -1}, {"rank": True}, {"schema_version": 2},
    {"query_lens": (0, 1, 1)}, {"query_lens": (2, 1, 1)},
    {"context_lens": (0, 1, 1)}, {"context_lens": (-1, 1, 1)},
    {"prefill_mask": (False, False)}, {"prefill_mask": (0, 0, 0)},
    {"phase": "prefill"}, {"capture_size": 2},
    {"graph_mode": "NONE"}, {"graph_mode": "PIECEWISE"},
    {"request_ids": ("a", "a", "c")}, {"request_ids": (1, 2, 3)},
    {"forward_gpu_ms": float("nan")}, {"forward_gpu_ms": 0},
    {"step_wall_ms": -1}, {"profiled": "false"},
    {"decode_input_sha256": "short"}, {"decode_input_sha256": "G" * 64},
    {"decode_input_sha256": 0},
])
def test_record_rejects_invalid_runtime_contract(change):
    with pytest.raises(ValueError):
        record(**change)


def test_tp_cohort_requires_all_ranks_and_matching_shapes():
    with pytest.raises(ValueError, match="Incomplete"):
        validate_rank_cohorts([record()], tensor_parallel_size=2)
    with pytest.raises(ValueError, match="disagree"):
        validate_rank_cohorts([record(), record(rank=1, capture_size=8)], tensor_parallel_size=2)
    assert len(validate_rank_cohorts([record(), record(rank=1)], tensor_parallel_size=2)) == 1
    with pytest.raises(ValueError, match="disagree"):
        validate_rank_cohorts([record(), record(rank=1, decode_input_sha256="a" * 64)],
                              tensor_parallel_size=2)


def test_fixed_decode_trajectory_excludes_last_output_and_retains_step_order():
    rows, digest = decode_trajectory([11, 12, 21, 22, 31, 32], 2, 3)
    assert rows == ((11, 12), (21, 22))
    assert digest == decode_trajectory([11, 12, 21, 22, 99, 99], 2, 3)[1]
    assert digest != decode_trajectory([11, 12, 99, 22, 31, 32], 2, 3)[1]
    assert record(decode_input_sha256=digest).decode_input_sha256 == digest


@pytest.mark.parametrize("tokens,bs,steps", [([1], 1, 2), ([1, -1], 1, 2),
    ([True, 2], 1, 2), ([], 0, 2), ([1], 1, 1)])
def test_incomplete_or_invalid_decode_trajectory_fails(tokens, bs, steps):
    with pytest.raises(ValueError, match="trajectory"):
        decode_trajectory(tokens, bs, steps)


# 64 concrete snapshots across dense/MoE, prefill/decode, lengths, request
# counts, and graph toggles. Scheduling mode/QPS cannot alter locked shapes.
@pytest.mark.parametrize("is_moe,phase,size,context,graphs", list(product(
    (False, True), ("prefill", "decode"), (1, 3, 8, 17), (128, 4096), (False, True)
)))
def test_locked_replay_shape_matrix(is_moe, phase, size, context, graphs):
    prefill = phase == "prefill"
    full = graphs and not prefill
    padded = ((size + 3) // 4) * 4
    row = record(phase=phase, request_ids=tuple(str(i) for i in range(size)),
                 query_lens=(16 if prefill else 1,) * size,
                 context_lens=(context,) * size, prefill_mask=(prefill,) * size,
                 graph_mode="FULL" if full else "NONE", capture_size=padded if full else 0)
    batch = row.to_frontier_batch(is_moe=is_moe)
    assert batch.is_moe == is_moe
    assert batch.size == size
    assert batch.total_num_tokens == size * (16 if prefill else 1)
    assert batch.num_prefill_tokens == (size * 16 if prefill else 0)
    assert [r.num_processed_tokens for r in batch.requests] == [context] * size
    assert batch.get_decode_cuda_graph_runtime_mode() == row.graph_mode
    from frontier.types import ClusterType
    assert batch.get_effective_total_tokens_for_compute(ClusterType.MONOLITHIC) == (
        padded if full else batch.total_num_tokens
    )
    from frontier.gdn.predictor import GDNBatchFeatures
    features = GDNBatchFeatures.from_batch(batch)
    assert features.batch_size == (padded if full else size)
    assert features.batch_num_tokens == (padded if full else batch.total_num_tokens)


def test_mixed_snapshot_preserves_continuation_prefill_and_decode():
    batch = record(phase="mixed", prefill_mask=(True, False, True),
                   query_lens=(128, 1, 16), context_lens=(0, 100, 256),
                   graph_mode="NONE", capture_size=0).to_frontier_batch(is_moe=True)
    assert batch.num_prefill_tokens == 144
    assert batch.num_decode_tokens == 1
    assert [r.is_prefill_complete for r in batch.requests] == [False, True, False]


def test_snapshot_preserves_original_prompt_and_decode_progress():
    batch = record(prompt_lens=(1024, 1024, 1024)).to_frontier_batch(is_moe=True)
    assert [r.num_processed_decode_tokens for r in batch.requests] == [0, 1024, 3072]
    with pytest.raises(ValueError, match="progress"):
        record(prompt_lens=(2000, 2000, 2000))


def test_replay_uses_model_milliseconds_and_slowest_rank_excludes_profiled_rows():
    class Predictor:
        def predict_stage_execution_time(self, batch, **kwargs):
            assert not hasattr(batch, "forward_gpu_ms")
            assert kwargs["num_layers"] == 92
            return SimpleNamespace(model_time_ms=22.0, total_time=999.0)
    rows = replay_batches([
        record(), record(rank=1, forward_gpu_ms=25.0),
        record(batch_id="trace", profiled=True), record(batch_id="trace", rank=1, profiled=True)
    ], Predictor(), model_config=SimpleNamespace(is_moe=True, num_layers=92), tensor_parallel_size=2)
    assert len(rows) == 1
    assert rows[0]["observed_forward_gpu_ms"] == 25
    assert rows[0]["signed_error_pct"] == pytest.approx(-12)
    assert summarize_errors(rows)["all"]["mape_pct"] == pytest.approx(12)


def test_dummy_and_missing_observations_cannot_report_parity():
    with pytest.raises(ValueError, match="Dummy"):
        replay_batches([record()], SimpleNamespace(_enable_dummy_mode=True),
                       model_config=None, tensor_parallel_size=1)
    with pytest.raises(ValueError, match="Missing forward"):
        replay_batches([record(forward_gpu_ms=None)], object(),
                       model_config=None, tensor_parallel_size=1)


def test_sglang_cpu_snapshot_decode_context_excludes_current_token():
    batch = SimpleNamespace(batch_size=3, seq_lens_cpu=[1025, 2049, 4097],
                            forward_mode=SimpleNamespace(is_decode=lambda: True))
    result = snapshot_forward_batch(batch)
    assert result["context_lens"] == (1024, 2048, 4096)


def test_sglang_cpu_snapshot_continuation_and_no_device_copy():
    batch = SimpleNamespace(batch_size=2, extend_seq_lens_cpu=[1, 128],
                            extend_prefix_lens_cpu=[128, 0], forward_mode=SimpleNamespace(
                                is_decode=lambda: False, is_extend=lambda: True,
                                is_mixed=lambda: False))
    result = snapshot_forward_batch(batch)
    assert result["prefill_mask"] == (True, True)  # a one-token extend is still prefill
    batch.extend_prefix_lens_cpu = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    with pytest.raises(ValueError, match="CPU-resident"):
        snapshot_forward_batch(batch)


def event(name, ts, dur, cat="kernel", device=0):
    return dict(name=name, ts=ts, dur=dur, cat=cat, ph="X", args={"device": device})


def test_trace_groups_by_synchronized_marker_and_counts_overlap_once():
    kernels = [event("gemm", 10, 10), event("reduce", 15, 15), event("sample", 40, 5)]
    assert interval_union_us(kernels) == 25
    trace = {"traceEvents": [event("frontier.batch:b", 0, 50, "user_annotation"),
                            event("frontier.batch:b", 5, 42, "gpu_user_annotation"), *kernels]}
    assert extract_batch_kernels(trace) == {"b": kernels}
    trace["traceEvents"].append(event("other GPU", 20, 5, device=1))
    with pytest.raises(ValueError, match="exactly one GPU"):
        extract_batch_kernels(trace)


def test_trace_rejects_missing_or_overlapping_markers():
    with pytest.raises(ValueError, match="No frontier.batch"):
        extract_batch_kernels({"traceEvents": [event("gemm", 0, 10)]})
    with pytest.raises(ValueError, match="overlap"):
        extract_batch_kernels({"traceEvents": [event("gemm", 5, 5),
            event("frontier.batch:a", 0, 20, "user_annotation"),
            event("frontier.batch:b", 10, 20, "user_annotation")]})

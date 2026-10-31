"""Task-local A11 observer, native admission, and diagnostic arithmetic checks."""

import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/analysis"))
sys.path.insert(0, str(ROOT / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10"))
import a10_observer as observer
import v4_replay_census as replay
import v4_runtime_census as native
from frontier.types import ClusterType
from test_dsv4_dp_sync import decode_lane, dummy_lane, prefill_lane, native_graphs, policy, lane_scheduler
from frontier.scheduler.utils.dp_dummy_forward import _synchronize_dp_descriptors


@pytest.mark.parametrize("shape,expected,step", [
    ("tail", "first_tail_alone", (1, 1, 0, 0)),
    ("mixed", "first_tail_with_decodes", (1, 1, 3, 3)),
    ("decode", "decode", (0, 0, 5, 5)),
    ("dummy", "dummy", (0, 0, 1, 1)),
    ("prefill", "prefill", (1, 6144, 0, 0)),
])
def test_frontier_and_r1_step_class(native_graphs, shape, expected, step):
    batches = {"tail": lambda: decode_lane(0, first_tails=1), "mixed": lambda: decode_lane(3, first_tails=1),
               "decode": lambda: decode_lane(5), "dummy": dummy_lane, "prefill": lambda: prefill_lane(6144)}
    batch = batches[shape]()
    batch._cluster_type = ClusterType.PREFILL if shape == "prefill" else ClusterType.DECODE
    role = batch._cluster_type.name.lower()
    record = {"real": shape != "dummy", "step": dict(zip(
        ("context_requests", "context_tokens", "generation_requests", "generation_tokens"), step))}
    assert observer.frontier_step_class(role, batch) == native.step_class(role, record) == expected
    assert observer.batch_shape(batch)["step"] == record["step"]


def counter_row(phase, count, kind="serving", dp=0, prefill=False, requests=1):
    return {"kind": kind, "phase": phase, "successful_forwards": count, "successful_replays": 999,
            "scheduled_num_requests": requests, "dispatch_evidence": {
                "classifier": {"inputs": {"has_prefill": prefill, "num_reqs": requests}},
                "dp_post_all_reduce": [[1, 1], [2, 2], [1, 1], [1, 1]],
                "dispatch": {"inputs": {"dp_rank": dp}, "descriptor": {"cg_mode": "FULL"},
                             "num_tokens_across_dp": [8, 8]}}}


def test_counter_phases_weights_and_all_nodes(tmp_path):
    for dp in (0, 1):
        path = tmp_path / f"nodes/host{dp}/serving/rank0.json"
        path.parent.mkdir(parents=True)
        rows = [counter_row("formal", 3 + dp, dp=dp), counter_row("warmup", 7, dp=dp),
                counter_row(None, 11, "dummy", dp), counter_row("formal", 5, "dummy", dp),
                {"kind": "capture", "phase": None, "successful_forwards": 2},
                {"kind": "profile", "phase": None, "successful_forwards": 1}]
        path.write_text(json.dumps({"role": "decode", "counters": {"rows": rows}}))
    result = replay.read_counters(tmp_path)
    assert len(result["sources"]) == 2
    assert sum(r["count"] for r in result["records"]) == 17
    assert sum(r["count"] for r in result["unassigned_phase_records"]) == 22
    assert sum(r["count"] for r in result["excluded_records"]) == 20
    assert sum(r["count"] for r in result["phase_counts"]) == 59
    counts = replay.census_counts(result["records"], "forwards")
    assert sum(counts.values()) == 17
    assert counts[("decode", 1, "decode", 1, "real", "FULL")] == 4


def test_counter_tail_ambiguity_is_preserved():
    one = replay.counter_record("decode", counter_row("formal", 2, prefill=True))
    many = replay.counter_record("decode", counter_row("formal", 3, prefill=True, requests=2))
    assert one["step_class"] == "first_tail_alone"
    assert many["step_class"] == "first_tail_unresolved"


def test_comparison_arithmetic_and_undefined_relative_error():
    assert replay.error(12, 10) == {"signed_error": 2, "absolute_error": 2, "relative_error": .2}
    assert replay.error(2, 0) == {"signed_error": 2, "absolute_error": 2, "relative_error": None}
    assert replay.error(0, 0)["relative_error"] is None
    assert replay.error(None, 0)["absolute_error"] is None
    stat = replay.time_stat([1, 2, 3, 4], [0, 0])
    assert stat["frontier"] == {"n": 4, "median": 2.5, "p90": 4}
    assert stat["errors"]["median"]["relative_error"] is None
    assert replay.count_comparison(Counter({("decode",): 4}), Counter({("decode",): 6}))[0]["difference"] == -2


def test_collective_brackets_exact_unequal_and_shares():
    from frontier.cc_backend.backends.measured_cc_backend import _Curve
    backend = SimpleNamespace(cluster_type=ClusterType.DECODE,
                              _curve_key=lambda *args: ("all_reduce", "tp", 8),
                              _curves={("all_reduce", "tp", 8): _Curve((16, 64), (2., 8.), ())})
    exact = observer.collective_request(backend, None, None, 16, 8, None, None, .002)
    between = observer.collective_request(backend, None, None, 32, 8, None, None, .004)
    assert exact["exact"] and exact["lower_bytes"] == exact["upper_bytes"] == 16
    assert exact["log_distance_ratio"] is None
    assert not between["exact"] and between["log_distance_ratio"] == .5
    rows = [{**exact, "call_count": 2, "predicted_total_ms": .004},
            {**between, "call_count": 1, "predicted_total_ms": .004}]
    stat = replay.collective_shares(rows)[0]
    assert stat["exact"]["call_share"] == 2 / 3
    assert stat["exact"]["time_share"] == .5
    backend._curve_key = lambda *args: ("reduce_scatterv", "ep", 16)
    backend._unequal_combine_times = {(("reduce_scatterv", "ep", 16), (1, 2), 16): 4.}
    row = observer.collective_request(backend, None, None, 16, 16, None, (1, 2), .004)
    assert row["unequal_rows"] == (1, 2) and row["exact"]


def test_observer_restores_hooks_and_preserves_failure(tmp_path):
    from frontier.cc_backend.backends.measured_cc_backend import MeasuredCCBackend, MeasuredCCOutOfRangeError, _Curve
    backend = object.__new__(MeasuredCCBackend)
    backend._cluster_type = ClusterType.PREFILL
    key = ("allreduce", None, 8)
    backend._curves = {key: _Curve((16, 64), (2., 8.), ())}
    predict = MeasuredCCBackend.predict_operator_collective
    with observer.ForwardObserver(tmp_path) as record:
        assert backend.predict_operator_collective("allreduce", None, 32, 8, None) == .004
        with pytest.raises(MeasuredCCOutOfRangeError) as failure:
            backend.predict_operator_collective("allreduce", None, 100, 8, None)
        assert failure.value.requested_bytes == 100
    assert MeasuredCCBackend.predict_operator_collective is predict
    receipt = record.receipt()
    assert receipt["forwards"]["rows"] == 0
    assert receipt["collectives"]["calls"] == 1
    rows = json.loads((tmp_path / "collectives.json").read_text())["rows"]
    assert len(rows) == 1 and rows[0]["call_count"] == 1


def test_scope_mapping_retains_composition_and_unmapped_costs(tmp_path):
    layer = SimpleNamespace(attention_family_id="deepseek_v4_attention", _dsv4_operator_gap_ms={"dsv4_fused_qkv_projection": .2},
                            _dsv4_attention_schedule=SimpleNamespace(start_ms={"dsv4_fused_qkv_projection": 0., "dsv4_compressor_projection": 0.}, elapsed_ms=2.2))
    timing = SimpleNamespace(layer_execution_times=[layer], op_times={"dsv4_fused_qkv_projection": 2., "dsv4_compressor_projection": 1.,
                            "dsv4_input_preparation": .4, "kv_cache_block_zero": .3, "dsv4_sampling": .1, "unknown_scope": .8})
    record = observer.ForwardObserver(tmp_path)
    record._add_scopes(("decode", 0, 0), timing)
    parts = record._segments[("decode", 0, 0)]
    assert math.isclose(parts["dispatch"], 2.2)
    assert parts["prepare"] == .4 and parts["post"] == .1 and parts["kv_cache_block_zero"] == .3
    assert record._scope_times[("decode", 0, 0)]["dsv4_fused_qkv_projection"] == 2.2
    assert parts["unknown_scope"] == .8


def test_observer_snapshots_final_descriptor_and_records_end_once(tmp_path, native_graphs):
    import gzip
    from frontier.entities import BatchStage
    from frontier.events.batch_stage_end_event import BatchStageEndEvent
    from frontier.scheduler.utils import dp_dummy_forward
    from unittest.mock import patch

    batch, dummy = decode_lane(3, first_tails=1), dummy_lane()
    batch._cluster_type = dummy._cluster_type = ClusterType.DECODE
    owner = policy()
    for lane in (batch, dummy):
        lane.decode_cuda_graph_metadata = owner._build_decode_cuda_graph_metadata(lane)
    scheduler = lane_scheduler(owner)
    scheduler._cluster_type = ClusterType.DECODE
    # Keep the test focused on the observer; scheduler completion itself is
    # exercised by the existing DP sync regression and the CPU identity run.
    with patch.object(BatchStageEndEvent, "handle_event", return_value=[]) as original:
        with observer.ForwardObserver(tmp_path) as record:
            # The batch skips `on_schedule`, so seed the formation record it would write.
            formed = {"formed_s": .5, "formation_index": 0, "queued_ahead_at_formation": 0,
                      "request_ids": [request.id for request in batch.requests]}
            record._formed[record._key(ClusterType.DECODE, batch)] = formed
            dp_dummy_forward._synchronize_dp_descriptors(scheduler, 0, {0: batch, 1: dummy})
            stage = BatchStage(batch.id, 0, 0, .01, .01, batch.requests, batch.num_tokens,
                               batch.request_is_decoding, ClusterType.DECODE)
            stage.on_schedule(1.)
            event = BatchStageEndEvent(1.01, 0, 0, True, batch, stage, ClusterType.DECODE, 0)
            event.handle_event(None, None)
        assert original.call_count == 1
    with gzip.open(tmp_path / "forwards.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == record.row_count == record.stage_end_events == 1
    row = rows[0]
    assert row["step_class"] == "first_tail_with_decodes" and row["mode"] == "PIECEWISE"
    assert row["peer_modes"] == ["PIECEWISE"] and row["peer_step_classes"] == ["dummy"]
    assert row["num_tokens_across_dp"] == [4, 4]
    assert row["classifier"]["has_prefill"] and row["classifier"]["uniform_token_count"] is None
    assert row["start_s"] == 1. and row["end_s"] == 1.01
    assert row["composed_ms"] == 10.
    assert {name: row[name] for name in formed} == formed


@pytest.mark.parametrize("status", ["STARTED", "INITIALIZED", "FAILED"])
def test_comparison_rejects_incomplete_run(tmp_path, status):
    (tmp_path / "receipt.json").write_text(json.dumps({"status": status}))
    with pytest.raises(ValueError, match=f"A11 needs a finished replay: .*status={status}"):
        replay.compare_run(tmp_path, tmp_path / "native")


def test_null_phase_dummy_count_is_unknown_in_formal_comparison(tmp_path, monkeypatch):
    import gzip
    run, artifacts = tmp_path / "run", tmp_path / "native"
    run.mkdir()
    source = artifacts / "nodes/host0/serving/rank0.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"role": "decode", "counters": {"rows": [counter_row(None, 11, "dummy")]}}))
    row = {"role": "decode", "dp": 0, "phase": "serving", "real": False,
           "step_class": "dummy", "mode": "FULL", "synchronized_width": 8,
           "peer_modes": ["FULL"], "composed_ms": 2.,
           "segment_ms": {"prepare": 0., "dispatch": 2., "post": 0.}}
    with gzip.open(run / "forwards.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(row) + "\n")
    (run / "collectives.json").write_text(json.dumps({"rows": []}))
    receipt = {"status": "FINISHED", "case": "fixture", "a11": {
        "scope_mapping": {}, "forwards": {"rows": 1}, "collectives": {}}}
    for name, filename in (("forwards", "forwards.jsonl.gz"), ("collectives", "collectives.json")):
        receipt["a11"][name]["sha256"] = replay.digest(run / filename)
    (run / "receipt.json").write_text(json.dumps(receipt))
    reference = {**row, "phase": "profiler_window", "composed_ms": 0.}
    monkeypatch.setattr(replay, "runtime_records", lambda _: ([reference], []))
    result = replay.compare_run(run, artifacts)
    for dimension in ("forwards", "widths", "dummy_peers"):
        formal = result["census"][dimension][0]
        assert formal["frontier"] == 1
        assert formal["native"] is None and formal["difference"] is None
        unknown = result["unassigned_phase_census"][dimension][0]
        assert unknown["native"] == 11 and unknown["difference"] is None
    composed = result["time_by_step_class"][0]["time_ms"]["composed_ms"]
    assert composed["native"]["n"] == 1
    assert composed["errors"]["median"]["relative_error"] is None


def test_source_scopes_follow_the_actual_disaggregation_moe_override(tmp_path, native_graphs):
    from unittest.mock import patch
    from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import SklearnDisaggregationExecutionTimePredictor
    from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import SklearnMoEExecutionTimePredictor

    class ConcretePredictor(SklearnDisaggregationExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("unused")

        def _get_grid_search_params(self):
            raise AssertionError("unused")

    predictor = object.__new__(ConcretePredictor)
    predictor._model_config = SimpleNamespace(get_model_architecture_profile=lambda: SimpleNamespace(profile_id="deepseek_v4"))
    batch = decode_lane(1)
    batch._cluster_type = ClusterType.DECODE
    layer = SimpleNamespace(global_layer_id=0, attention_family_id="deepseek_v4_attention")
    timing = SimpleNamespace(num_layers=1, layer_execution_times=[layer], op_times={
        "dsv4_q_projection": 2., "dsv4_input_preparation": .4, "kv_cache_block_zero": .3})
    with patch.object(SklearnMoEExecutionTimePredictor, "predict_stage_execution_time", return_value=timing) as original:
        with observer.ForwardObserver(tmp_path) as record:
            for _ in range(2):
                assert predictor.predict_stage_execution_time(batch, 0, ClusterType.DECODE, include_ffn=False) is timing
            key = record._key(ClusterType.DECODE, batch)
            assert record._scope_times[key]["dsv4_q_projection"] == 2.
            assert record._segments[key]["prepare"] == .4
            assert record._segments[key]["kv_cache_block_zero"] == .3
        assert original.call_count == 2

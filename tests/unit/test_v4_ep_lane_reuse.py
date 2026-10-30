"""V4 EP phase reuse preserves numeric timings and per-participant metadata."""

from dataclasses import fields, make_dataclass, replace
from functools import partial

import pytest

from frontier.entities import Batch, DummyForwardBatch, EPBatchGroup, Request
from frontier.execution_time_predictor.deepseek_v4 import source_sp_token_widths
from frontier.moe_ep_workload import EPLaneWorkload, materialize_balanced_ep_workload
from frontier.scheduler.utils.batch_builders import build_ep_lane_batch
from frontier.scheduler.utils.expert_parallel import (
    _v4_ep_lane_prediction_key, get_ep_phase_times_ms, predict_ep_wave_phase_times,
)
from frontier.types import ClusterType
from test_dsv4_operator_consumer import batch, trained  # Reuse the trained CPU fixture.


@pytest.mark.parametrize("dp", [2, 4])
@pytest.mark.parametrize("source_kind", ["prefill_with_dummies", "decode"])
@pytest.mark.parametrize("layer_id", [0, 3, 42])
def test_trained_v4_distinct_lanes_preserve_phases(trained, monkeypatch, dp, source_kind, layer_id):
    predictor, _ = trained
    replica = predictor._replica_config
    monkeypatch.setattr(replica, "attn_dp", dp)
    monkeypatch.setattr(replica, "moe_expert_parallel_size", 8 * dp)
    if source_kind == "prefill_with_dummies":
        sources = {0: batch("NONE")}
        for lane in range(1, dp):
            sources[lane] = DummyForwardBatch(0, 0, lane)
            sources[lane]._cluster_type = ClusterType.MONOLITHIC
    else:
        # Different source identities/request counts, with the same FULL
        # descriptor width. Partial SP ranks have different valid-row metadata.
        sources = {}
        for lane, count in enumerate((1, 3, 9, 16)[:dp]):
            source = Batch(0, [Request(0, 6144, 48) for _ in range(count)], [1] * count,
                           is_moe=True, num_context_tokens=[6144] * count,
                           request_is_decoding=[True] * count)
            source._cluster_type = ClusterType.MONOLITHIC
            source.decode_cuda_graph_metadata = replace(
                batch("FULL").decode_cuda_graph_metadata,
                original_total_tokens=count, original_decode_batch_size=count,
            )
            sources[lane] = source
    widths = source_sp_token_widths(predictor, sources)
    workload = materialize_balanced_ep_workload(
        routing_token_count=widths.gathered_tokens, router_topk=6,
        total_expert_num=256, moe_expert_parallel_size=8 * dp,
    )
    keys = [_v4_ep_lane_prediction_key(rank // 8, workload.lane(rank)) for rank in range(8 * dp)]
    expected_first = [rank for rank in range(8 * dp) if keys[rank] not in keys[:rank]]
    original_predict = predictor._predict_dsv4_structural_execution
    original_assemble = predictor._assemble_stage
    predictions, assemblies, phase_calls, logs, built = [], [], [], [], []

    def predict(*args, **kwargs):
        predictions.append(kwargs["lane_workload"].ep_id)
        return original_predict(*args, **kwargs)

    def assemble(*args, **kwargs):
        assemblies.append(1)
        return original_assemble(*args, **kwargs)

    def phase_getter(*args, **kwargs):
        phase_calls.append(kwargs["ep_id"])
        return get_ep_phase_times_ms(*args, **kwargs)

    def create_group(requests, tokens, replica_id, rank, time, source_ids, lane_workload):
        return EPBatchGroup(requests, tokens, replica_id, rank, time, source_ids,
                            lane_workload, ClusterType.MONOLITHIC, is_moe=True)

    lane_builder = partial(build_ep_lane_batch, create_batch_group=create_group,
                           cluster_type=ClusterType.MONOLITHIC)

    def build(**kwargs):
        lane = lane_builder(**kwargs)
        built.append(lane)
        return lane

    monkeypatch.setattr(predictor, "_predict_dsv4_structural_execution", predict)
    monkeypatch.setattr(predictor, "_assemble_stage", assemble)
    results = []
    for capture in (False, True, False):
        predictions.clear()
        assemblies.clear()
        phase_calls.clear()
        logs.clear()
        built.clear()
        result = predict_ep_wave_phase_times(
            layer_workload=workload, source_batch=sources[0], stage_id=0, layer_id=layer_id,
            cluster_type=ClusterType.MONOLITHIC, predictor=predictor, lane_builder=build,
            phase_getter=phase_getter, workload_logger=lambda **kwargs: logs.append(kwargs),
            trace_identity=None, batch_id=sources[0].id, source_batches=sources,
            sp_token_widths=widths, capture_lane_timings=capture,
        )
        results.append(result)
        expected = list(range(8 * dp)) if capture else expected_first
        assert predictions == phase_calls == expected
        assert len(assemblies) == len(expected)
        assert len(built) == len(logs) == 8 * dp
        for rank, (lane, log) in enumerate(zip(built, logs)):
            assert log["lane_workload"] is lane.lane_workload
            assert log["lane_workload"].ep_id == rank
            assert lane.moe_sequence_parallel_token_widths == widths
            assert lane.moe_pre_routing_effective_total_tokens == widths.local_tokens[rank // 8]
        if capture:
            assert [record.ep_id for record in result.lane_records] == list(range(8 * dp))
            for record in result.lane_records:
                metadata = record.execution_time.layer_execution_times[0]._dsv4_source_rows
                assert metadata["source_tp_rank"] == record.ep_id % 8
                assert metadata["source_batch_id"] == sources[record.ep_id // 8].id
        else:
            assert result.lane_records == ()
    assert results[0][:6] == results[1][:6] == results[2][:6]
    assert len(set(keys)) < 8 * dp


def test_v4_lane_key_rejects_an_uncovered_dataclass_field():
    extended = make_dataclass("ExtendedLane", [("new_phase_input", int, 0)],
                              bases=(EPLaneWorkload,), frozen=True)
    lane = materialize_balanced_ep_workload(
        routing_token_count=32, router_topk=6, total_expert_num=256,
        moe_expert_parallel_size=16,
    ).lane(0)
    changed = extended(**{field.name: getattr(lane, field.name) for field in fields(lane)})
    with pytest.raises(ValueError, match="V4 EP lane prediction key.*uncovered=.*new_phase_input"):
        _v4_ep_lane_prediction_key(0, changed)

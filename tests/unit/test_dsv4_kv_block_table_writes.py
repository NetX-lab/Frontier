"""Native V2 add/update/remove table coordinates on real Frontier KV state."""

from math import ceil
from dataclasses import FrozenInstanceError

import pytest

from frontier.config import global_vars
from frontier.entities import Request
from frontier.types import ClusterType
from test_dsv4_structural_pd import _harness, _lane, _Metrics, _start_event, _nixl_context
from test_grouped_kv_cache_scheduler import _v4_schedulers


def forward(lane, time):
    batch, = lane._schedule_iteration(time)
    return batch


def finish(lane, batch, time):
    batch.on_batch_end(time, lane._cluster_type)
    lane.on_batch_end(batch)


def test_first_schedule_null_entries_reclamation_and_free():
    source, _ = _lane(ClusterType.PREFILL)
    request = Request(0, 6144, 1)
    source.add_request(request)
    batch = forward(source, 0)
    record = batch.kv_block_table_writes
    assert (record.w_c, record.r) == (0, 1)
    assert tuple(length for _, length in record.new_rows[0][1]) == (24, 96, 96, 1536, 768)
    with pytest.raises(FrozenInstanceError):
        record.new_rows = ()
    # Reclaimed blocks become null entries; the table's length is unchanged.
    source._kv_cache_manager.reclaim_skipped_blocks(request, settled_num_computed_tokens=6144)
    tables = source._kv_cache_manager.get_group_block_ids(request.id)
    assert any(None in blocks for blocks in tables.values())
    repeated = source._record_kv_block_table_writes([request])
    assert (repeated.w_c, repeated.r) == (0, 0)
    finish(source, batch, 1)
    source._free_request_resources(request)
    assert request.id not in source._group_block_table_lengths
    assert tuple(length for _, length in record.new_rows[0][1]) == (24, 96, 96, 1536, 768)


def test_decode_nixl_first_step_and_c1_rules_for_short_pdd_run():
    source, target, scheduler, _ = _harness()
    request = Request(0, 6144, 17)
    source.add_request(request)
    p = forward(source, 0)
    sizes = [group.logical_block_size for group in source._kv_cache_capacity.groups]
    assert len(p.cache_block_zeroing.block_ids) == sum(ceil(6144 / size) for size in sizes)
    finish(source, p, 1)
    start = _start_event(request, source)
    end, = start.handle_event(scheduler, _Metrics())
    assert target._pending_cache_zero_block_ids == {}
    assert request.id not in target._group_block_table_lengths
    end.handle_event(scheduler, _Metrics())
    for step in range(17):
        batch = forward(target, end.time + step * 2)
        assert batch.num_tokens == [1]
        assert batch.num_context_tokens == [6143 + step]
        from frontier.attention.dsv4_features import dsv4_request_is_decoding
        from frontier.execution_time_predictor.dsv4_runner import preparation_coordinate
        assert dsv4_request_is_decoding(batch) == [step != 0]
        assert preparation_coordinate(batch)["phase"] == ("first_tail" if step == 0 else "decode")
        old, new, is_new = 6143 + step, 6144 + step, step == 0
        # C1 c1_workload.py:316-326: native old/new/is_new tuples.
        differences = [ceil(new / size) - ceil(old / size) for size in sizes]
        zero = 0 if batch.cache_block_zeroing is None else len(batch.cache_block_zeroing.block_ids)
        assert zero == sum(differences)
        writes = batch.kv_block_table_writes
        assert (writes.w_c, writes.r) == (sum(d > 0 for d in differences) if not is_new else 0, int(is_new))
        entries = sum(count for _, _, count in writes.continuing_entries) + sum(
            length for _, groups in writes.new_rows for _, length in groups)
        assert entries == sum(ceil(new / size) - (0 if is_new else ceil(old / size)) for size in sizes)
        if step == 0:
            assert zero == 0
            assert tuple(length for _, length in writes.new_rows[0][1]) == (24, 96, 96, 1536, 768)
        finish(target, batch, end.time + step * 2 + 1)
    assert request.completed
    assert target._group_block_table_lengths == {}
    assert source._group_block_table_lengths == {}


def test_preempted_forward_resumes_as_new_row(monkeypatch):
    monkeypatch.setattr(global_vars, 'KV_CONNECTOR', 'none')
    lane = _v4_schedulers(blocks=200, chunk=64)[0]
    request = Request(0, 64, 20)
    lane.add_request(request)
    first = forward(lane, 0)
    finish(lane, first, 1)
    assert request.id in lane._group_block_table_lengths
    preempted = []
    lane._preempt_request(request, preempted)
    assert preempted == [request]
    assert request.id not in lane._group_block_table_lengths
    resumed = forward(lane, 2)
    assert (resumed.kv_block_table_writes.w_c, resumed.kv_block_table_writes.r) == (0, 1)


def test_non_grouped_forward_and_free_create_no_table_state(monkeypatch):
    from frontier.config import ReplicaConfig, SyntheticRequestGeneratorConfig, VllmV1SchedulerConfig
    from frontier.entities import Replica
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import VLLMv1EngineReplicaScheduler
    from test_grouped_kv_cache_scheduler import _ClusterScheduler

    monkeypatch.setattr(global_vars, 'KV_CONNECTOR', 'none')
    config = ReplicaConfig(model_name='meta-llama/Llama-2-7b-hf', device='h200', network_device='h200_dgx')
    requests = SyntheticRequestGeneratorConfig()
    replica = Replica(config, requests, ClusterType.MONOLITHIC)
    lane = VLLMv1EngineReplicaScheduler(
        replica_config=config,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=100, num_blocks_mode='explicit', max_tokens_in_batch=8,
            long_prefill_token_threshold=8, enable_chunked_prefill=True, batch_size_cap=8),
        request_generator_config=requests, replica=replica, predictor=object(),
        cluster_scheduler=_ClusterScheduler(replica.id), cluster_type=ClusterType.MONOLITHIC)
    assert not hasattr(lane, '_group_block_table_lengths')
    request = Request(0, 8, 1)
    lane.add_request(request)
    batch = forward(lane, 0)
    assert batch.kv_block_table_writes is None
    finish(lane, batch, 1)
    lane._free_request_resources(request)
    assert not hasattr(lane, '_group_block_table_lengths')

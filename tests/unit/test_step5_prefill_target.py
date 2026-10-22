"""Real scheduler/event PD lifetimes with explicit CPU costs, not native profiles."""

from collections import Counter

import pytest

from frontier.entities import Request
from frontier.entities.batch import DummyForwardBatch
from frontier.events.cluster_batch_end_event import ClusterBatchEndEvent
from frontier.events.kv_cache_transfer_start_event import KVCacheTransferStartEvent
from frontier.types import ClusterType
from tests.unit.test_step5_measured_pd import _FixtureCosts, _measured_cluster
from tests.unit.test_step5_structural_pd import _GlobalScheduler, _Metrics, simulation_globals


class _BatchMetrics(_Metrics):
    def on_batch_end(self, *args):
        pass


@pytest.mark.parametrize("mode", ["offline", "online"])
@pytest.mark.parametrize("prompt,prefix,outputs", [(6143, 6112, 1024), (6144, 6128, 1024), (6145, 6128, 1023)])
def test_scheduler_prefill_transfer_tail_and_outputs_preserve_client_lengths(monkeypatch, mode, prompt, prefix, outputs):
    from frontier.config import global_vars

    monkeypatch.setattr(global_vars, "SIMULATION_MODE", mode)
    costs = _FixtureCosts(changes={"num_prefill_tokens": prompt, "num_computed_tokens": prefix})
    source_cluster = _measured_cluster(ClusterType.PREFILL, costs, attn_dp=2)
    target_cluster = _measured_cluster(ClusterType.DECODE, costs, attn_dp=2)
    scheduler = _GlobalScheduler(source_cluster, target_cluster)
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    # Keep the native semantic bound, independently of the P compute target.
    for cluster in (source_cluster, target_cluster):
        for lane in cluster._replica_schedulers.values():
            lane._max_model_len = 7168
    request = Request(0, prompt, outputs)
    request.on_arrival(0, ClusterType.PREFILL)
    source.add_request(request)
    [batch] = source.on_schedule(0.1)
    assert batch.num_tokens == [prefix]
    assert batch.num_context_tokens == [0]
    assert source._allocation_map[request.id] == 4 * (prefix // 16)
    assert source._get_grouped_admission_blocks(request) == source._kv_cache_manager.get_num_admission_blocks(
        prefix, max_model_len=7168, max_in_flight_tokens=8192,
    )
    assert request.num_prefill_tokens == request.user_facing_num_prefill_tokens == prompt
    metrics = _BatchMetrics()
    event = ClusterBatchEndEvent(0.2, source._replica_id, batch, ClusterType.PREFILL, 0, source_batch_stage_id=0)
    events = event.handle_event(scheduler, metrics)
    [start] = [event for event in events if isinstance(event, KVCacheTransferStartEvent)]
    assert start._batch.num_tokens == [prefix]
    assert request.num_processed_tokens == prefix
    assert request.is_prefill_complete and request.is_finished_for_cluster(ClusterType.PREFILL)
    assert request.num_emitted_decode_tokens == 0
    assert source.get_kv_transfer_source_prompt_common_blocks(request) == prefix // 16
    assert request.id in source._allocation_map
    [end] = start.handle_event(scheduler, metrics)
    info = end._transfer_info
    target = target_cluster.get_replica_scheduler(info.target_replica_id, info.target_replica_local_id)
    assert target.on_schedule(end.time - 0.001) == []
    assert request.num_emitted_decode_tokens == 0
    end.handle_event(scheduler, metrics)
    assert info.target_ready_time == info.source_release_time == end.time
    assert source._allocation_map == {}
    assert request.num_processed_tokens == prompt and request.num_context_tokens == prefix
    with pytest.raises(RuntimeError):
        start.handle_event(scheduler, metrics)
    with pytest.raises(RuntimeError):
        end.handle_event(scheduler, metrics)
    widths = []
    contexts = []
    for output_index in range(outputs):
        now = end.time + .01 * (output_index + 1)
        for lane in target._dp_group():
            if lane is target:
                continue
            for idle in lane.on_schedule(now):
                assert isinstance(idle, DummyForwardBatch)
                assert idle.requests == []
                lane.on_dummy_forward_end(now + .005)
        batches = target.on_schedule(now)
        if not batches:
            assert target._waits_at_dp_sync
            for lane in target._dp_group():
                if lane is not target:
                    for idle in lane.on_schedule(now):
                        assert idle.requests == []
                        lane.on_dummy_forward_end(now + .005)
            batches = target.on_schedule(now)
        [decode] = batches
        widths.extend(decode.num_tokens)
        contexts.extend(decode.num_context_tokens)
        decode.on_batch_end(now + .005, ClusterType.DECODE)
        target.on_batch_end(decode)
        target_cluster.on_replica_batch_end(now + .005, info.target_replica_id, info.target_replica_local_id, decode)
        assert request.num_emitted_decode_tokens == output_index + 1
    assert Counter(widths) == Counter({prompt - prefix: 1, 1: outputs - 1})
    assert contexts == [prefix, *range(prompt, prompt + outputs - 1)]
    assert request.completed and request.num_processed_tokens == prompt + outputs
    assert request.num_emitted_decode_tokens == outputs
    assert request.first_decode_token_completed_at > end.time > request.prefill_completed_at
    assert target._allocation_map == {}
    assert source._pending_kv_transfer_requests == set()
    assert target_cluster._load_balancer.request_owners == {}
    assert len(metrics.starts) == len(metrics.ends) == 1


@pytest.mark.parametrize("qualifier", ["dummy_timing", "no_connector", "co_location", "prefix_cache"])
def test_nonqualifying_prefill_keeps_the_full_semantic_prompt(monkeypatch, qualifier):
    from frontier.config import global_vars

    source_cluster = _measured_cluster(ClusterType.PREFILL, _FixtureCosts())
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    if qualifier == "dummy_timing":
        source_cluster._config.execution_time_predictor_config.enable_dummy_mode = True
    elif qualifier == "no_connector":
        monkeypatch.setattr(global_vars, "KV_CONNECTOR", "none")
    elif qualifier == "co_location":
        monkeypatch.setattr(global_vars, "SYS_ARCH", "co-location")
    else:
        monkeypatch.setattr(source, "_is_prefix_caching_enabled", lambda: True)
    request = Request(0, 6144, 1024)
    source.add_request(request)
    assert request.prefill_compute_target == request.num_prefill_tokens == 6144


def test_only_step_registers_this_native_prefix_contract():
    from frontier.attention.families import get_attention_family, STEP5_ATTENTION_FAMILY, DSA_ATTENTION_FAMILY

    assert STEP5_ATTENTION_FAMILY.kv_transfer_prefill_target_resolver is not None
    # Sliding retention alone must not select the Step replay rule.
    assert DSA_ATTENTION_FAMILY.kv_transfer_prefill_target_resolver is None
    assert get_attention_family("dense_attention").kv_transfer_prefill_target_resolver is None


def test_mismatched_exact_transfer_coverage_fails_before_read_or_placement():
    costs = _FixtureCosts(changes={"num_prefill_tokens": 6144, "num_computed_tokens": 6112})
    source_cluster = _measured_cluster(ClusterType.PREFILL, costs)
    target_cluster = _measured_cluster(ClusterType.DECODE, costs)
    scheduler = _GlobalScheduler(source_cluster, target_cluster)
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    request = Request(0, 6144, 1024)
    source.add_request(request)
    [batch] = source.on_schedule(.1)
    metrics = _BatchMetrics()
    events = ClusterBatchEndEvent(.2, source._replica_id, batch, ClusterType.PREFILL, 0,
                                 source_batch_stage_id=0).handle_event(scheduler, metrics)
    [start] = [event for event in events if isinstance(event, KVCacheTransferStartEvent)]
    with pytest.raises(ValueError, match="prefix does not match the P compute target"):
        start.handle_event(scheduler, metrics)
    assert request.id in source._allocation_map
    assert not source._kv_transfer_start_claimed_request_ids
    assert not target_cluster._load_balancer.request_owners
    assert all(not lane._allocation_map for lane in target_cluster._replica_schedulers.values())
    assert metrics.starts == metrics.ends == []


def test_partial_prefill_completes_at_target_and_preemption_retains_target():
    source_cluster = _measured_cluster(ClusterType.PREFILL, _FixtureCosts())
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    request = Request(0, 6144, 1024)
    source.add_request(request)
    source._allocate_request(request, 2048)
    source._running_requests.append(request)
    source._request_queue.remove(request)
    request.on_batch_end(.1, 2048, ClusterType.PREFILL)
    assert not request.is_prefill_complete
    assert source._get_request_next_num_tokens(request) == 4080
    source._preempt_request(request, [])
    assert request.prefill_compute_target == 6128 and request.num_processed_tokens == 0
    assert source._get_request_next_num_tokens(request) == 6128
    [batch] = source.on_schedule(.2)
    assert batch.num_tokens == [6128]
    batch.on_batch_end(.3, ClusterType.PREFILL)
    source.on_batch_end(batch)
    assert request.is_prefill_complete and request.num_processed_tokens == 6128
    assert request.num_emitted_decode_tokens == 0

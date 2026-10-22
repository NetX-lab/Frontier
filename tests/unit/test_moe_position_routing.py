"""Actual count/materializer and request-owner source/tensor contracts."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from frontier.entities import Batch, Request
from frontier.entities.batch import DummyForwardBatch
from frontier.moe_ep_workload import build_contiguous_expert_ownership, materialize_layer_ep_workload
from frontier.moe_routing_policy import (
    SourceLaneRouting, TokenInterval, WaveRoutingSource, build_wave_routing_source,
    position_cyclic_expert_counts, resolve_routing_count_policy,
)
from frontier.profiling.moe.inputs import MoESourceFrame, source_frame_tensors
from frontier.types import ClusterType


def source(start=6128, length=16, padding=1):
    return WaveRoutingSource("position_cyclic", (
        SourceLaneRouting(0, 2, -1, 7, "NONE", "standalone_legacy", length, length, (TokenInterval(start, length),)),
        SourceLaneRouting(1, 3, -1, 7, "NONE", "standalone_legacy", padding, 0, ()),
    ), length + padding)


def materialize(routing_source):
    return materialize_layer_ep_workload(
        policy_id="position_cyclic", routing_source=routing_source,
        target_replica_id=0, global_layer_id=3, routing_token_count=routing_source.gathered_physical_rows,
        router_topk=8, total_expert_num=352, moe_expert_parallel_size=16,
        expert_to_ep=build_contiguous_expert_ownership(352, 16),
    )


def test_full_pd_chunk_counts_use_logical_positions_not_wave_hamilton():
    full = position_cyclic_expert_counts((TokenInterval(0, 6144),), 8, 352)
    split = position_cyclic_expert_counts((TokenInterval(0, 6128), TokenInterval(6128, 16)), 8, 352)
    chunks = position_cyclic_expert_counts(tuple(TokenInterval(a, n) for a, n in ((0, 2048), (2048, 2048), (4096, 2032), (6128, 16))), 8, 352)
    assert full == split == chunks
    tail = materialize(source())
    assert tail.routing_token_count == 17 and tail.live_routing_token_count == 16
    assert tail.padding_row_count == 1 and tail.total_routed_assignments == 128
    assert [e for e, count in tail.global_per_expert_tokens.items() if count] == list(range(96, 224))
    assert sum(tail.per_ep_routed_tokens.values()) == 128
    assert tail.lane(0).routed_token_count == 0 and tail.lane(0).routing_source == source()
    q1 = position_cyclic_expert_counts((TokenInterval(6144, 1),), 8, 352)
    assert [e for e, count in q1.items() if count] == list(range(224, 232))


def test_source_immutable_padding_and_repeated_positions():
    original = source()
    padded = replace(original, source_lanes=(original.source_lanes[0], replace(original.source_lanes[1], physical_rows=8)), gathered_physical_rows=24)
    assert materialize(original).global_per_expert_tokens == materialize(padded).global_per_expert_tokens
    with pytest.raises(FrozenInstanceError):
        original.source_lanes[0].physical_rows = 5
    repeated = position_cyclic_expert_counts((TokenInterval(6144, 1),) * 4, 8, 352)
    assert max(repeated.values()) == 4 and sum(repeated.values()) == 32
    all_padding = replace(original, source_lanes=tuple(replace(lane, live_rows=0, intervals=()) for lane in original.source_lanes))
    assert materialize(all_padding).total_routed_assignments == 0
    assert materialize(all_padding).routing_token_count == 17


def test_actual_batches_freeze_schedule_context_not_later_request_progress():
    from frontier.scheduler.utils.forward_start_agreement import ForwardStartDescriptor
    request = Request(arrived_at=0, num_prefill_tokens=6144, num_decode_tokens=1024)
    real = Batch(0, [request], [16], is_moe=True, num_context_tokens=[6128])
    dummy = DummyForwardBatch(0, 0, 0)
    for batch, width in ((real, 16), (dummy, 1)):
        batch.forward_start_descriptor = ForwardStartDescriptor(9, "NONE", width, 1, None, None)
    frozen = build_wave_routing_source({0: real, 1: dummy}, policy_id="position_cyclic", cluster_type=ClusterType.DECODE, gathered_physical_rows=17)
    assert frozen.source_lanes[0].intervals == (TokenInterval(6128, 16),)
    assert frozen.source_lanes[1].intervals == ()
    real._num_context_tokens[0] = 7000
    assert frozen.source_lanes[0].intervals[0].start == 6128
    real.forward_start_descriptor = None
    with pytest.raises(ValueError, match="forward_start_descriptor"):
        build_wave_routing_source({0: real}, policy_id="position_cyclic", cluster_type=ClusterType.DECODE, gathered_physical_rows=16)
    with pytest.raises(ValueError, match="request-owner"):
        build_wave_routing_source({0: SimpleNamespace()}, policy_id="position_cyclic", cluster_type=ClusterType.DECODE, gathered_physical_rows=1)


def test_registry_owns_the_complete_count_call_and_strict_source():
    policy = resolve_routing_count_policy(" position_cyclic ")
    assert policy.requires_source and policy.runtime_path == "position_cyclic"
    with pytest.raises(ValueError, match="matching source"):
        materialize_layer_ep_workload(policy_id="position_cyclic", target_replica_id=0, global_layer_id=0,
                                     routing_token_count=17, router_topk=8, total_expert_num=352,
                                     moe_expert_parallel_size=16, expert_to_ep=build_contiguous_expert_ownership(352, 16))
    with pytest.raises(ValueError, match="explicit source"):
        policy.count_experts(routing_source=None, routing_ratios=None, routing_token_count=1, router_topk=8, total_expert_num=352)


def test_one_source_tensor_interface_preserves_masks_and_request_intervals():
    pytest.importorskip("torch")
    frame = MoESourceFrame(0, (2, 1), (6128, 6144), 5, "standalone_legacy")
    positions, mask = source_frame_tensors(frame, device="cpu")
    assert positions.tolist()[:3] == [6128, 6129, 6144]
    assert mask.tolist() == [False, False, False, True, True]
    positions, mask = source_frame_tensors(MoESourceFrame(1, (), (), 2, "standalone_legacy"), device="cpu")
    assert mask.tolist() == [True, True]
    assert str(positions.dtype) == "torch.int64" and str(mask.dtype) == "torch.bool"

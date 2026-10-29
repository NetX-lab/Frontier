"""V2 graph dispatch on semantic DECODE requests with native first-tail rows."""

import pytest

from frontier.attention.deepseek_v4 import CSA_VARIANT
from frontier.attention.dsv4_features import build_dsv4_features, dsv4_request_is_decoding
from frontier.config import global_vars
from frontier.scheduler.utils.dp_dummy_forward import _synchronize_dp_descriptors
from frontier.types import ClusterType
from test_dsv4_dp_sync import (
    decode_lane, dummy_lane, lane_scheduler, native_graphs,
    padded_num_reqs, policy, prefill_lane,
)


def features(batch):
    return build_dsv4_features(
        batch.num_tokens, batch.num_context_tokens, dsv4_request_is_decoding(batch),
        variant=CSA_VARIANT, compress_ratio=4, sliding_window=128, index_topk=512,
        padded_rows=batch.decode_cuda_graph_metadata.padded_total_tokens,
    )


def test_runtime_first_tail_with_decodes_dispatches_piecewise(native_graphs):
    """RED on e52ecec4: first_tail attention must share its PIECEWISE descriptor."""
    batch, peer = decode_lane(11, first_tails=1), decode_lane(8)
    assert all(request.is_decoding for request in batch.requests)
    assert batch.request_is_decoding == [True] * 12
    tail = batch.requests[0]
    assert tail.num_processed_tokens == tail.num_prefill_tokens
    assert batch.num_context_tokens[0] == tail.num_prefill_tokens - 1
    assert batch.kv_block_table_writes.new_rows == ((tail.id, ()),)
    assert dsv4_request_is_decoding(batch) == [False] + [True] * 11
    owner = policy()
    for lane in (batch, peer):
        lane.decode_cuda_graph_metadata = owner._build_decode_cuda_graph_metadata(lane)
    _synchronize_dp_descriptors(lane_scheduler(owner), 0, {0: batch, 1: peer})
    assert features(batch).phase == "first_tail"
    assert batch.get_decode_cuda_graph_runtime_mode() == "PIECEWISE"
    assert peer.get_decode_cuda_graph_runtime_mode() == "PIECEWISE"
    assert [padded_num_reqs(lane) for lane in (batch, peer)] == [12, 8]


@pytest.mark.parametrize("decodes,with_dummy", [(0, False), (3, False), (0, True), (3, True)])
def test_first_tail_clears_the_forward_uniform_count(native_graphs, decodes, with_dummy):
    batch = decode_lane(decodes, first_tails=1)
    lanes = {0: batch, 1: dummy_lane() if with_dummy else decode_lane(5)}
    owner = policy()
    calls = []
    dispatch = owner.dispatch_full_and_piecewise

    def record_dispatch(num_reqs, tokens, uniform):
        calls.append(uniform)
        return dispatch(num_reqs, tokens, uniform)

    owner.dispatch_full_and_piecewise = record_dispatch
    for lane in lanes.values():
        lane._cluster_type = ClusterType.DECODE
        lane.decode_cuda_graph_metadata = owner._build_decode_cuda_graph_metadata(lane)
    assert batch.decode_cuda_graph_metadata.is_mixed_batch
    assert batch.decode_cuda_graph_metadata.original_decode_batch_size == decodes
    assert batch.get_decode_cuda_graph_runtime_mode() == "PIECEWISE"
    calls.clear()
    _synchronize_dp_descriptors(lane_scheduler(owner), 0, lanes)
    assert calls == [None, None]
    assert [lane.get_decode_cuda_graph_runtime_mode() for lane in lanes.values()] == ["PIECEWISE"] * 2
    assert padded_num_reqs(batch) == batch.size
    assert features(batch).phase == "first_tail"


@pytest.mark.parametrize("role", [ClusterType.MONOLITHIC, ClusterType.PREFILL])
@pytest.mark.parametrize("tokens,expected", [(1, "PIECEWISE"), (6144, "NONE")])
def test_initial_prefill_chunk_keeps_its_mode(native_graphs, role, tokens, expected):
    batch = prefill_lane(tokens)
    batch._cluster_type = role
    metadata = policy(role)._build_decode_cuda_graph_metadata(batch)
    assert metadata.runtime_mode == expected
    assert metadata.is_mixed_batch
    assert metadata.original_decode_batch_size == metadata.padded_decode_batch_size == 0


@pytest.mark.parametrize("mode,expected", [("full_and_piecewise", "FULL"),
                                          ("full_decode_only", "FULL"),
                                          ("piecewise", "PIECEWISE")])
def test_steady_decodes_keep_their_mode(native_graphs, monkeypatch, mode, expected):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", mode)
    metadata = policy()._build_decode_cuda_graph_metadata(decode_lane(5))
    assert metadata.runtime_mode == expected
    assert not metadata.is_mixed_batch
    assert metadata.original_decode_batch_size == 5
    assert metadata.padded_decode_batch_size == metadata.padded_total_tokens == 8


@pytest.mark.parametrize("mode,expected", [("full_decode_only", "FULL"), ("piecewise", "PIECEWISE")])
@pytest.mark.parametrize("decodes", [0, 3])
def test_other_runners_keep_request_level_first_tail_flags(native_graphs, monkeypatch, mode, expected, decodes):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", mode)
    batch = decode_lane(decodes, first_tails=1)
    metadata = policy()._build_decode_cuda_graph_metadata(batch)
    assert metadata.runtime_mode == expected
    assert not metadata.is_mixed_batch
    assert metadata.original_decode_batch_size == batch.size
    assert metadata.padded_decode_batch_size == metadata.padded_total_tokens


@pytest.mark.parametrize("mode", ["full_decode_only", "piecewise"])
def test_other_runners_keep_prefill_role_excluded(native_graphs, monkeypatch, mode):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", mode)
    assert policy(ClusterType.PREFILL)._build_decode_cuda_graph_metadata(prefill_lane(1)) is None

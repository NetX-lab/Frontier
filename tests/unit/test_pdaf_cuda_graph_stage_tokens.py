from collections import deque
from types import MethodType

import pytest

from frontier.config import global_vars
from frontier.entities.batch import AFDStageMetadata, Batch
from frontier.entities.request import Request
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


@pytest.fixture(autouse=True)
def _reset_cuda_graph_globals():
    global_vars.reset_global_vars()
    global_vars.set_cuda_graph_config(
        use_cuda_graph=False,
        cudagraph_capture_sizes=None,
    )
    yield
    global_vars.reset_global_vars()


def _make_decode_request(request_index: int) -> Request:
    request = Request(
        arrived_at=float(request_index),
        num_prefill_tokens=32,
        num_decode_tokens=3,
        num_processed_tokens=32,
    )
    request._is_prefill_complete = True
    request._current_decode_token_index = 1
    return request


def _make_batch(*, num_requests: int = 1, is_moe: bool = False) -> Batch:
    requests = [_make_decode_request(index) for index in range(num_requests)]
    return Batch(
        replica_id=0,
        requests=requests,
        num_tokens=[1] * num_requests,
        is_moe=is_moe,
    )


def _make_decode_attn_scheduler(
    requests: list[Request],
    *,
    logical_stage_count: int,
    is_moe: bool,
) -> VLLMv1EngineReplicaScheduler:
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._replica_id = 0
    scheduler._replica_local_id = 0
    scheduler._replica_is_moe = is_moe
    scheduler._num_stages = 1
    scheduler._micro_batch_size = len(requests)
    scheduler._af_pipeline_num_micro_batch = logical_stage_count
    scheduler._af_pending_micro_batches = deque()
    scheduler._running_requests = list(requests)
    scheduler._waiting_requests = []
    scheduler._continuation_request_ids = set()
    scheduler._batch_creation_counter = 0
    scheduler._active_batch_request_counts = {}

    scheduler._can_allocate_request = lambda _request, _num_tokens: True
    scheduler._allocate_request = lambda _request, _num_tokens: None

    def create_batch(
        self: VLLMv1EngineReplicaScheduler,
        batch_requests: list[Request],
        batch_tokens: list[int],
    ) -> Batch:
        batch = Batch(
            replica_id=self._replica_id,
            requests=batch_requests,
            num_tokens=batch_tokens,
            is_moe=self._replica_is_moe,
        )
        batch.set_global_id(self._batch_creation_counter)
        self._batch_creation_counter += 1
        return batch

    scheduler._create_batch = MethodType(create_batch, scheduler)
    return scheduler


def test_metadata_factory_records_stage_local_cuda_graph_tokens() -> None:
    metadata = AFDStageMetadata.from_batch_params(
        num_reqs=5,
        num_tokens_per_req=[1, 1, 1, 1, 1],
        num_stages=2,
        use_cuda_graph=True,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        ffn_use_cuda_graph=True,
        ffn_cudagraph_capture_sizes=[1, 2, 4, 8, 16],
    )

    assert metadata.original_stage_token_lens == [2, 3]
    assert metadata.padded_stage_token_lens == [2, 4]
    assert metadata.ffn_compute_stage_token_lens == [2, 4]
    assert metadata.original_total_tokens == 5
    assert metadata.padded_total_tokens == 6
    assert metadata.ffn_compute_total_tokens == 8


def test_physical_stage_batch_uses_stage_local_compute_and_transfer_tokens() -> None:
    batch = _make_batch(num_requests=2)
    batch.afd_stage_idx = 1
    batch.afd_stage_metadata = AFDStageMetadata(
        num_stages=2,
        original_total_tokens=6,
        padded_total_tokens=10,
        ffn_compute_total_tokens=12,
        original_stage_token_lens=[2, 4],
        padded_stage_token_lens=[2, 8],
        ffn_compute_stage_token_lens=[4, 8],
    )

    assert (
        batch.get_effective_total_tokens_for_compute(ClusterType.DECODE_ATTN)
        == 8
    )
    assert (
        batch.get_effective_total_tokens_for_transfer(ClusterType.DECODE_ATTN)
        == 8
    )
    assert (
        batch.get_effective_total_tokens_for_compute(ClusterType.DECODE_FFN)
        == 8
    )


def test_macro_wave_uses_aggregate_compute_and_max_stage_transfer_tokens() -> None:
    batch = _make_batch()
    batch.afd_stage_idx = 0
    batch.afd_stage_represents_all_stages = True
    batch.afd_stage_metadata = AFDStageMetadata(
        num_stages=4,
        original_total_tokens=256,
        padded_total_tokens=1024,
        ffn_compute_total_tokens=1024,
        original_stage_token_lens=[64, 64, 64, 64],
        padded_stage_token_lens=[256, 256, 256, 256],
        ffn_compute_stage_token_lens=[256, 256, 256, 256],
    )

    assert (
        batch.get_effective_total_tokens_for_compute(ClusterType.DECODE_ATTN)
        == 1024
    )
    assert (
        batch.get_effective_total_tokens_for_compute(ClusterType.DECODE_FFN)
        == 1024
    )
    assert (
        batch.get_effective_total_tokens_for_transfer(ClusterType.DECODE_ATTN)
        == 256
    )
    assert (
        batch.get_effective_total_tokens_for_transfer(ClusterType.DECODE_FFN)
        == 256
    )


def test_dense_pp1_scheduler_uses_one_metadata_macro_wave() -> None:
    requests = [_make_decode_request(index) for index in range(6)]
    scheduler = _make_decode_attn_scheduler(
        requests,
        logical_stage_count=3,
        is_moe=False,
    )

    first_batch = scheduler._schedule_decode_attn_only(is_micro_batch=True)

    assert first_batch is not None
    assert len(scheduler._af_pending_micro_batches) == 0
    assert first_batch.afd_stage_idx == 0
    assert first_batch.afd_stage_represents_all_stages is True
    assert first_batch.afd_stage_metadata is not None
    assert first_batch.afd_stage_metadata.original_stage_token_lens == [2, 2, 2]
    assert [request.id for request in first_batch.requests] == [
        request.id for request in requests
    ]
    assert (
        first_batch.get_effective_total_tokens_for_compute(
            ClusterType.DECODE_ATTN
        )
        == 6
    )
    assert (
        first_batch.get_effective_total_tokens_for_transfer(
            ClusterType.DECODE_ATTN
        )
        == 2
    )


def test_moe_pp1_scheduler_keeps_physical_stage_siblings() -> None:
    requests = [_make_decode_request(index) for index in range(6)]
    scheduler = _make_decode_attn_scheduler(
        requests,
        logical_stage_count=3,
        is_moe=True,
    )

    first_batch = scheduler._schedule_decode_attn_only(is_micro_batch=True)

    assert first_batch is not None
    stage_batches = [first_batch, *scheduler._af_pending_micro_batches]
    assert len(stage_batches) == 3
    assert [len(batch.requests) for batch in stage_batches] == [2, 2, 2]
    assert [batch.afd_stage_idx for batch in stage_batches] == [0, 1, 2]
    assert all(
        batch.afd_stage_represents_all_stages is False
        for batch in stage_batches
    )


@pytest.mark.parametrize(
    ("helper_name", "cluster_type", "metadata_field"),
    [
        (
            "get_effective_total_tokens_for_compute",
            ClusterType.DECODE_ATTN,
            "padded_stage_token_lens",
        ),
        (
            "get_effective_total_tokens_for_compute",
            ClusterType.DECODE_FFN,
            "ffn_compute_stage_token_lens",
        ),
        (
            "get_effective_total_tokens_for_transfer",
            ClusterType.DECODE_ATTN,
            "padded_stage_token_lens",
        ),
    ],
)
def test_stage_token_helpers_fail_fast_for_out_of_range_stage_index(
    helper_name: str,
    cluster_type: ClusterType,
    metadata_field: str,
) -> None:
    batch = _make_batch()
    batch.afd_stage_idx = 2
    batch.afd_stage_metadata = AFDStageMetadata(
        num_stages=2,
        original_total_tokens=2,
        padded_total_tokens=2,
        ffn_compute_total_tokens=2,
        original_stage_token_lens=[1, 1],
        padded_stage_token_lens=[1, 1],
        ffn_compute_stage_token_lens=[1, 1],
    )

    helper = getattr(batch, helper_name)
    with pytest.raises(ValueError, match=metadata_field):
        helper(cluster_type)

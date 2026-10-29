"""Runner coordinates shared by CPU promotion, training and V4 runtime pricing."""

from types import MappingProxyType

from frontier.types import ClusterType
from frontier.entities.batch import DummyForwardBatch
from frontier.attention.deepseek_v4 import compute_rows
from frontier.attention.dsv4_features import (
    PHASE_DECODE, PHASE_FIRST_TAIL, PHASE_PREFILL, PHASE_DUMMY, dsv4_request_is_decoding,
)
from frontier.operators.families import KV_CACHE_FAMILY, DEEPSEEK_V4_RUNNER_FAMILY

ZERO_SCOPE, WRITE_SCOPE = (op.name for op in KV_CACHE_FAMILY.operators)
INPUT_SCOPE, SAMPLING_SCOPE = (op.name for op in DEEPSEEK_V4_RUNNER_FAMILY.operators)
BLOCK_TABLE = "model_runner_block_kernel_only.csv"
REPLAY_TABLE = "model_runner_replay_kernel_only.csv"
PREPARATION_COLUMNS = (
    "phase", "graph_runtime_mode", "padded_rows", "dispatch_num_requests", "batch_size",
    "num_prefill_seqs", "num_decode_seqs", "total_tokens",
)
ACTIVITY_COLUMNS = ("kernel_count", "memcpy_count", "memset_count")
NEW_ROW_ENTRIES = "new_row_entries"
SAMPLING_COLUMNS = ("needs_logits_processing", "b")
LOGITS_PROCESSING_OFF, LOGITS_PROCESSING_ON = 0, 1
SAMPLING_BRANCH_KERNEL = "_bias_kernel"
# vLLM 0.28 sampler.py:75-87,192-199 selects a batch-level processing branch.
# Frontier's fixed output lengths use native min_tokens=max_tokens (EOS bias)
# for MONOLITHIC and unified DECODE; the 0.28 PD proxy removes min_tokens on
# PREFILL (pd_proxy.py:103-108). Native P/D traces and G2h-02/04 confirm these
# branches. This is the simulated request contract, not a tuning parameter.
SAMPLING_LOGITS_PROCESSING_BY_ROLE = MappingProxyType({
    ClusterType.MONOLITHIC: LOGITS_PROCESSING_ON,
    ClusterType.DECODE: LOGITS_PROCESSING_ON,
    ClusterType.PREFILL: LOGITS_PROCESSING_OFF,
})


def sampling_coordinate(batch):
    return dict(zip(SAMPLING_COLUMNS, (
        SAMPLING_LOGITS_PROCESSING_BY_ROLE[batch._cluster_type], len(batch.requests),
    )))


def preparation_coordinate(batch):
    """The local real geometry and the already synchronized V2 descriptor."""
    mode = batch.get_decode_cuda_graph_runtime_mode()
    padded = compute_rows(batch)
    dummy = isinstance(batch, DummyForwardBatch)
    if dummy:
        phase = PHASE_DUMMY
        prefill, decode = ("", "")
        real_requests = ""
    else:
        decode = sum(dsv4_request_is_decoding(batch))
        prefill = len(batch.requests) - decode
        real_requests = len(batch.requests)
        phase = (PHASE_FIRST_TAIL if prefill and max(batch.num_tokens) == 1
                 else PHASE_PREFILL if prefill else PHASE_DECODE)
    # FULL pads requests; PIECEWISE pads only tokens. V2 dummies retain one
    # request outside FULL, independently of their attention query geometry.
    dispatched = (padded if mode == "FULL" else "" if mode == "PIECEWISE"
                  else 1 if dummy else real_requests)
    return dict(zip(PREPARATION_COLUMNS, (
        phase, mode, padded, dispatched, real_requests, prefill, decode,
        "" if dummy else batch.total_num_tokens,
    )))

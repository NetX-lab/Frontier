from __future__ import annotations

import pytest

from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
)
from frontier.attention.memory import (
    AttentionRuntimeKVLayout,
    get_attention_runtime_kv_layout,
)
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType


def test_dense_kv_layout_matches_vllm_page_bytes_for_gqa_mha_mqa() -> None:
    layout = get_attention_runtime_kv_layout(
        DENSE_ATTENTION_FAMILY,
        runtime_num_kv_heads_per_worker=4,
        runtime_head_size=96,
    )

    assert layout == AttentionRuntimeKVLayout(
        family_id="dense_attention",
        kv_factor=2,
        runtime_num_kv_heads_per_worker=4,
        runtime_head_size=96,
        bytes_per_element=2,
    )
    assert layout.elements_per_token_per_worker == 2 * 4 * 96
    assert layout.page_bytes(block_size=16) == 24576


def test_latent_mla_layout_matches_vllm_runtime_groundtruth_page_bytes() -> None:
    layout = get_attention_runtime_kv_layout(
        LATENT_MLA_ATTENTION_FAMILY,
        runtime_num_kv_heads_per_worker=1,
        runtime_head_size=576,
    )

    assert layout.kv_factor == 1
    assert layout.elements_per_token_per_worker == 1 * 1 * 576
    assert layout.page_bytes(block_size=64) == 73728


def test_attention_runtime_layout_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="runtime_head_size"):
        get_attention_runtime_kv_layout(
            DENSE_ATTENTION_FAMILY,
            runtime_num_kv_heads_per_worker=4,
            runtime_head_size=0,
        )
    with pytest.raises(ValueError, match="runtime_num_kv_heads_per_worker"):
        get_attention_runtime_kv_layout(
            DENSE_ATTENTION_FAMILY,
            runtime_num_kv_heads_per_worker=0,
            runtime_head_size=96,
        )


def test_frozen_dsa_layout_fails_fast() -> None:
    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        get_attention_runtime_kv_layout(
            DSA_ATTENTION_FAMILY,
            runtime_num_kv_heads_per_worker=1,
            runtime_head_size=128,
        )


def test_decode_ffn_memory_planner_keeps_zero_kv_elements_without_binding() -> None:
    planner = object.__new__(MemoryPlanner)
    planner._cluster_type = ClusterType.DECODE_FFN
    planner._replica_config = object()
    planner._replica = object()

    assert planner._get_kv_cache_elements_per_token_per_worker() == 0

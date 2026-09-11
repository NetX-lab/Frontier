from __future__ import annotations

from frontier.profiling.attention.backends.vllm_rocm_attention_wrapper import (
    build_rocm_sequence_plan,
)
from frontier.profiling.attention.sequence_proxy import SequenceMetadataProxy


def test_rocm_sequence_plan_orders_prefill_then_decode_and_maps_slots() -> None:
    prefill = SequenceMetadataProxy(
        is_prompt=True,
        total_len=20,
        processed_len=16,
        block_table=[3, 4],
    )
    decode = SequenceMetadataProxy(
        is_prompt=False,
        total_len=18,
        processed_len=17,
        block_table=[8, 9],
    )

    plan = build_rocm_sequence_plan([prefill, decode], block_size=16)

    assert plan.query_lengths == [4, 1]
    assert plan.sequence_lengths == [20, 18]
    assert plan.block_tables == [[3, 4], [8, 9]]
    assert plan.slot_mapping == [64, 65, 66, 67, 145]


def test_rocm_sequence_plan_rejects_short_block_table() -> None:
    sequence = SequenceMetadataProxy(
        is_prompt=True,
        total_len=33,
        processed_len=32,
        block_table=[0, 1],
    )

    try:
        build_rocm_sequence_plan([sequence], block_size=16)
    except ValueError as exc:
        assert "Block table is too short" in str(exc)
    else:
        raise AssertionError("Expected a short block table to be rejected")

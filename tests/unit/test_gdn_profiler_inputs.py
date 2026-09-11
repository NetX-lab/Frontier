from argparse import Namespace

import pytest

from frontier.profiling.gdn.inputs import GDNProfileInput
from frontier.profiling.gdn.main import build_profile_inputs


def test_gdn_profile_input_classifies_prefill_decode_and_mixed() -> None:
    prefill = GDNProfileInput.prefill(seq_len=128)
    decode = GDNProfileInput.decode(batch_size=8, context_len=512)
    mixed = GDNProfileInput.mixed(
        decode_batch_size=4,
        decode_context_len=512,
        prefill_seq_len=128,
    )

    assert prefill.phase == "prefill"
    assert prefill.num_prefill_tokens == 128
    assert not prefill.has_initial_state
    assert decode.phase == "decode"
    assert decode.num_decode_tokens == 8
    assert decode.has_initial_state
    assert mixed.phase == "mixed"
    assert mixed.query_lens == (1, 1, 1, 1, 128)
    assert mixed.num_decode_tokens == 4
    assert mixed.num_prefill_tokens == 128
    assert prefill.state_block_ids == (1,)
    assert decode.state_block_ids == tuple(range(1, 9))
    assert mixed.state_block_ids == (1, 2, 3, 4, 5)


def test_gdn_profile_input_rejects_prefill_before_decode() -> None:
    with pytest.raises(ValueError, match="decode-first"):
        GDNProfileInput(
            query_lens=(128, 1),
            context_lens=(0, 512),
        )


def test_gdn_cli_plan_includes_cold_hot_decode_and_mixed() -> None:
    args = Namespace(
        prefill_seq_lens=[16, 128],
        prefill_batch_sizes=[1],
        decode_batch_sizes=[1, 8],
        decode_context_len=512,
        continuation_context_len=256,
        include_continuation_prefill=True,
        include_mixed=True,
    )

    inputs = build_profile_inputs(args)

    assert [profile_input.phase for profile_input in inputs] == [
        "prefill",
        "prefill",
        "prefill",
        "prefill",
        "decode",
        "decode",
        "mixed",
    ]
    assert [
        profile_input.has_initial_state for profile_input in inputs[:4]
    ] == [False, False, True, True]


def test_gdn_cli_plan_crosses_prefill_lengths_and_batch_sizes() -> None:
    args = Namespace(
        prefill_seq_lens=[128, 1024],
        prefill_batch_sizes=[1, 16],
        decode_batch_sizes=[],
        decode_context_len=1024,
        continuation_context_len=1024,
        include_continuation_prefill=False,
        include_mixed=False,
    )

    inputs = build_profile_inputs(args)

    assert [(item.batch_size, item.num_tokens) for item in inputs] == [
        (1, 128),
        (1, 1024),
        (16, 2048),
        (16, 16384),
    ]

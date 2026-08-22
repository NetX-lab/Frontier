import pytest

from frontier.profiling.attention.attention_input import AttentionInput
from frontier.profiling.utils import (
    get_attention_batch_sizes_to_profile,
    get_attention_input_combinations,
    get_attention_prefill_chunk_sizes_to_profile,
    get_num_tokens_to_profile,
    get_mixed_prefill_input_combinations,
    get_online_grid_mixed_prefill_input_combinations,
    get_seq_lengths_to_profile,
    get_true_mixed_attention_input_combinations,
)
from frontier.profiling.moe.moe_input import get_default_moe_profiling_config


def _attention_input_tuples(inputs):
    return [
        (
            item.prefill_chunk_size,
            item.kv_cache_size,
            item.batch_size,
            item.is_prefill,
        )
        for item in inputs
    ]


def test_extra_num_tokens_are_unioned_even_when_above_default_grid_limit():
    values = get_num_tokens_to_profile(10, extra_num_tokens=[11, 11, 7])

    assert values == sorted(set(values), reverse=True)
    assert 11 in values
    assert 7 in values


def test_default_token_grid_reaches_requested_endpoint():
    values = get_num_tokens_to_profile(10)

    assert max(values) == 10
    assert all(value <= 10 for value in values)


@pytest.mark.parametrize("extra_value", [0, -1])
def test_extra_num_tokens_reject_non_positive_values(extra_value):
    with pytest.raises(ValueError, match="extra_num_tokens"):
        get_num_tokens_to_profile(10, extra_num_tokens=[extra_value])


def test_sequence_length_grid_reaches_requested_endpoint():
    values = get_seq_lengths_to_profile(1000)

    assert values == sorted(set(values))
    assert values
    assert max(values) == 1000
    assert all(value <= 1000 for value in values)


def test_prefill_chunk_grid_reaches_requested_endpoint():
    values = get_attention_prefill_chunk_sizes_to_profile(1000)

    assert values == sorted(set(values))
    assert values
    assert max(values) == 1000
    assert all(value <= 1000 for value in values)


def test_attention_batch_grid_reaches_requested_endpoints():
    values = get_attention_batch_sizes_to_profile(129, 130)

    assert values == [129, 130]


def test_default_moe_token_grid_reaches_requested_endpoint():
    config = get_default_moe_profiling_config(max_tokens=5000)

    assert max(config.num_tokens_list) == 5000
    assert config.num_tokens_list == sorted(set(config.num_tokens_list))


def test_attention_combinations_cover_prefill_and_decode_endpoints_deterministically():
    kwargs = dict(
        max_seq_len=1000,
        min_batch_size=1,
        max_batch_size=2,
        profile_only_prefill=False,
        profile_only_decode=False,
    )
    first = get_attention_input_combinations(**kwargs)
    second = get_attention_input_combinations(**kwargs)
    first_tuples = _attention_input_tuples(first)

    assert first_tuples == _attention_input_tuples(second)
    assert (1000, 0, 1, True) in first_tuples
    assert (0, 999, 1, False) in first_tuples
    assert all(item.is_valid(1000) for item in first)


def test_decode_attention_input_reserves_the_current_token():
    assert AttentionInput(0, 999, 1, False).is_valid(1000)
    assert not AttentionInput(0, 1000, 1, False).is_valid(1000)


def test_decode_memory_limit_reserves_the_current_token():
    decode_input = AttentionInput(0, 32, 1, False)

    assert not decode_input.is_under_memory_limit(32)
    assert decode_input.is_under_memory_limit(33)


def test_explicit_decode_cache_endpoint_remains_bounded():
    inputs = get_attention_input_combinations(
        max_seq_len=1000,
        min_batch_size=1,
        max_batch_size=1,
        profile_only_prefill=False,
        profile_only_decode=True,
        decode_kv_cache_size_list=[999],
    )

    assert _attention_input_tuples(inputs) == [(0, 999, 1, False)]

    with pytest.raises(ValueError, match="decode_kv_cache_size_list"):
        get_attention_input_combinations(
            max_seq_len=1000,
            min_batch_size=1,
            max_batch_size=1,
            profile_only_prefill=False,
            profile_only_decode=True,
            decode_kv_cache_size_list=[1000],
        )


def test_legacy_mixed_prefill_retains_high_kv_endpoint_with_legal_sequence():
    inputs = get_mixed_prefill_input_combinations(
        max_seq_len=1000,
        min_batch_size=2,
        max_batch_size=2,
        mode="even",
        kv_cache_sizes=[0, 992],
    )

    high_kv_inputs = [item for item in inputs if item.kv_cache_size == 992]
    assert high_kv_inputs
    assert max(item.max_seq_len + item.kv_cache_size for item in high_kv_inputs) == 1000
    assert all(item.is_valid(1000, max_batch_size=128) for item in high_kv_inputs)


def test_legacy_mixed_prefill_emits_unique_structural_workloads():
    inputs = get_mixed_prefill_input_combinations(
        max_seq_len=1000,
        min_batch_size=2,
        max_batch_size=8,
        mode="both",
        kv_cache_sizes=[0],
    )

    identities = {
        (tuple(item.seq_lens), item.kv_cache_size, item.mode) for item in inputs
    }
    assert len(inputs) == len(identities)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_batch_size": 129, "max_batch_size": 129}, "batch_size"),
        ({"kv_cache_sizes": [1000]}, "kv_cache_sizes"),
        ({"mode": "unsupported"}, "mode"),
    ],
)
def test_legacy_mixed_prefill_rejects_runtime_invalid_axes(kwargs, message):
    base_kwargs = {
        "max_seq_len": 1000,
        "min_batch_size": 2,
        "max_batch_size": 2,
        "mode": "even",
        "kv_cache_sizes": [0],
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        get_mixed_prefill_input_combinations(**base_kwargs)


def test_online_grid_explicit_lists_extend_default_envelope():
    inputs = get_online_grid_mixed_prefill_input_combinations(
        max_seq_len=1000,
        min_batch_size=2,
        max_batch_size=3,
        min_total_tokens=10,
        max_total_tokens=11,
        shapes_per_point=1,
        batch_size_list=[4],
        total_tokens_list=[12],
    )

    points = {(item.batch_size, item.total_tokens) for item in inputs}
    assert {(2, 10), (3, 11), (4, 12)} <= points


def test_online_grid_rejects_runtime_invalid_explicit_batch_size():
    with pytest.raises(ValueError, match="batch_size_list"):
        get_online_grid_mixed_prefill_input_combinations(
            max_seq_len=1000,
            min_batch_size=2,
            max_batch_size=2,
            min_total_tokens=10,
            max_total_tokens=10,
            shapes_per_point=1,
            batch_size_list=[129],
            total_tokens_list=[129],
        )


def test_online_grid_rejects_invalid_explicit_axis_cross_product():
    with pytest.raises(ValueError, match="total_tokens"):
        get_online_grid_mixed_prefill_input_combinations(
            max_seq_len=1000,
            min_batch_size=2,
            max_batch_size=2,
            min_total_tokens=10,
            max_total_tokens=10,
            shapes_per_point=1,
            batch_size_list=[4],
            total_tokens_list=[3],
        )


def test_online_grid_skips_invalid_automatic_points_but_keeps_explicit_points():
    inputs = get_online_grid_mixed_prefill_input_combinations(
        max_seq_len=256,
        max_model_len=512,
        min_batch_size=2,
        max_batch_size=2,
        min_total_tokens=1025,
        max_total_tokens=1055,
        shapes_per_point=1,
        batch_size_list=[2],
        total_tokens_list=[64, 128],
    )

    points = {(item.batch_size, item.total_tokens) for item in inputs}
    assert points == {(2, 64), (2, 128)}


def test_online_grid_rejects_explicit_point_without_a_legal_pair():
    with pytest.raises(ValueError, match="total_tokens"):
        get_online_grid_mixed_prefill_input_combinations(
            max_seq_len=256,
            max_model_len=512,
            min_batch_size=2,
            max_batch_size=2,
            min_total_tokens=1025,
            max_total_tokens=1055,
            shapes_per_point=1,
            batch_size_list=[2],
            total_tokens_list=[1025],
        )


def test_online_grid_explicit_kv_can_extend_beyond_profile_context():
    inputs = get_online_grid_mixed_prefill_input_combinations(
        max_seq_len=256,
        max_model_len=512,
        min_batch_size=2,
        max_batch_size=2,
        min_total_tokens=64,
        max_total_tokens=64,
        shapes_per_point=1,
        kv_cache_sizes=[256],
    )

    assert len(inputs) == 1
    assert inputs[0].kv_cache_size == 256


def test_online_grid_rejects_model_capacity_below_profile_context():
    with pytest.raises(ValueError, match="max_model_len must be >= max_seq_len"):
        get_online_grid_mixed_prefill_input_combinations(
            max_seq_len=256,
            max_model_len=128,
            min_batch_size=2,
            max_batch_size=2,
            min_total_tokens=64,
            max_total_tokens=64,
            shapes_per_point=1,
        )


def test_online_grid_keeps_legal_kv_siblings_when_an_automatic_pair_is_invalid():
    inputs = get_online_grid_mixed_prefill_input_combinations(
        max_seq_len=256,
        max_model_len=512,
        min_batch_size=2,
        max_batch_size=32,
        min_total_tokens=64,
        max_total_tokens=64,
        shapes_per_point=1,
        kv_cache_sizes=[0, 500],
    )

    point_kv = {
        (item.batch_size, item.total_tokens, item.kv_cache_size)
        for item in inputs
    }
    assert (2, 64, 0) in point_kv
    assert (2, 64, 500) not in point_kv
    assert (32, 64, 500) in point_kv


def test_online_grid_keeps_a_valid_shape_when_its_sibling_shape_is_invalid():
    inputs = get_online_grid_mixed_prefill_input_combinations(
        max_seq_len=256,
        max_model_len=512,
        min_batch_size=2,
        max_batch_size=32,
        min_total_tokens=64,
        max_total_tokens=64,
        shapes_per_point=2,
        kv_cache_sizes=[0, 480],
    )

    point_modes = {
        (item.batch_size, item.total_tokens, item.kv_cache_size, item.mode)
        for item in inputs
    }
    assert (2, 64, 480, "online_grid_balanced") in point_modes
    assert (2, 64, 480, "online_grid_skewed") not in point_modes


def test_true_mixed_grid_reaches_prefill_and_decode_endpoints():
    inputs = get_true_mixed_attention_input_combinations(
        max_seq_len=1000,
        prefill_batch_sizes=[1],
        prefill_chunk_sizes=[64],
        decode_batch_sizes=[1],
        decode_kv_cache_sizes=[128],
        prefill_kv_cache_size=0,
    )

    assert max(item.prefill_seq_lens[0] for item in inputs) == 1000
    assert max(item.decode_kv_cache_sizes[0] for item in inputs) == 999
    assert all(item.is_valid(max_seq_len=1000, max_batch_size=128) for item in inputs)


@pytest.mark.parametrize(
    ("max_seq_len", "expected_prefill_chunks", "expected_decode_kv"),
    [
        (128, {64, 128, 256}, {127, 128}),
        (256, {64, 128, 256, 512}, {128, 255, 256}),
        (
            1024,
            {64, 128, 256, 512, 1024, 2048},
            {128, 256, 512, 1023, 1024},
        ),
    ],
)
def test_true_mixed_auto_axes_follow_profile_context(
    max_seq_len, expected_prefill_chunks, expected_decode_kv
):
    inputs = get_true_mixed_attention_input_combinations(
        max_seq_len=max_seq_len,
        max_model_len=max_seq_len * 2,
        prefill_batch_sizes=[1, 2, 4],
        prefill_chunk_sizes=None,
        decode_batch_sizes=[1, 2, 4, 8],
        decode_kv_cache_sizes=None,
        prefill_kv_cache_size=0,
    )

    prefill_chunks = {item.prefill_seq_lens[0] for item in inputs}
    decode_kv = {item.decode_kv_cache_sizes[0] for item in inputs}
    assert prefill_chunks == expected_prefill_chunks
    assert decode_kv == expected_decode_kv


def test_true_mixed_auto_axes_use_physical_boundary_when_next_anchor_does_not_fit():
    inputs = get_true_mixed_attention_input_combinations(
        max_seq_len=1000,
        max_model_len=1010,
        prefill_batch_sizes=[1],
        prefill_chunk_sizes=None,
        decode_batch_sizes=[1],
        decode_kv_cache_sizes=None,
        prefill_kv_cache_size=0,
    )

    prefill_chunks = {item.prefill_seq_lens[0] for item in inputs}
    decode_kv = {item.decode_kv_cache_sizes[0] for item in inputs}
    assert prefill_chunks == {64, 128, 256, 512, 1000, 1010}
    assert decode_kv == {128, 256, 512, 999, 1009}


def test_true_mixed_explicit_axes_can_extend_beyond_profile_context():
    inputs = get_true_mixed_attention_input_combinations(
        max_seq_len=256,
        max_model_len=512,
        prefill_batch_sizes=[1],
        prefill_chunk_sizes=[512],
        decode_batch_sizes=[1],
        decode_kv_cache_sizes=[256],
        prefill_kv_cache_size=0,
    )

    extended = [
        item
        for item in inputs
        if item.prefill_seq_lens == [512]
        and item.decode_kv_cache_sizes == [256]
    ]
    assert len(extended) == 1
    assert len(inputs) > len(extended)


def test_true_mixed_explicit_axes_beyond_profile_capacity_fail_fast():
    with pytest.raises(ValueError):
        get_true_mixed_attention_input_combinations(
            max_seq_len=256,
            max_model_len=512,
            prefill_batch_sizes=[1],
            prefill_chunk_sizes=[513],
            decode_batch_sizes=[1],
            decode_kv_cache_sizes=[256],
            prefill_kv_cache_size=0,
        )


def test_true_mixed_rejects_model_capacity_below_profile_context():
    with pytest.raises(ValueError, match="max_model_len must be >= max_seq_len"):
        get_true_mixed_attention_input_combinations(
            max_seq_len=256,
            max_model_len=128,
            prefill_batch_sizes=[1],
            prefill_chunk_sizes=None,
            decode_batch_sizes=[1],
            decode_kv_cache_sizes=None,
            prefill_kv_cache_size=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prefill_chunk_sizes": [1001]}, "prefill_chunk_sizes"),
        ({"decode_kv_cache_sizes": [1000]}, "decode_kv_cache_sizes"),
        (
            {
                "prefill_batch_sizes": [100],
                "decode_batch_sizes": [29],
            },
            "total batch size",
        ),
        ({"prefill_kv_cache_size": 1000}, "prefill_kv_cache_size"),
    ],
)
def test_true_mixed_rejects_runtime_invalid_explicit_axes(kwargs, message):
    base_kwargs = {
        "max_seq_len": 1000,
        "prefill_batch_sizes": [1],
        "prefill_chunk_sizes": [64],
        "decode_batch_sizes": [1],
        "decode_kv_cache_sizes": [128],
        "prefill_kv_cache_size": 0,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        get_true_mixed_attention_input_combinations(**base_kwargs)

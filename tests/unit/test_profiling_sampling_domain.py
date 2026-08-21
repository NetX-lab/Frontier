import pytest

from frontier.profiling.utils import (
    get_attention_batch_sizes_to_profile,
    get_attention_input_combinations,
    get_attention_prefill_chunk_sizes_to_profile,
    get_num_tokens_to_profile,
    get_seq_lengths_to_profile,
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
    assert (0, 1000, 1, False) in first_tuples
    assert all(item.is_valid(1000) for item in first)


def test_explicit_decode_cache_endpoint_remains_bounded():
    inputs = get_attention_input_combinations(
        max_seq_len=1000,
        min_batch_size=1,
        max_batch_size=1,
        profile_only_prefill=False,
        profile_only_decode=True,
        decode_kv_cache_size_list=[1000],
    )

    assert _attention_input_tuples(inputs) == [(0, 1000, 1, False)]

    with pytest.raises(ValueError, match="decode_kv_cache_size_list"):
        get_attention_input_combinations(
            max_seq_len=1000,
            min_batch_size=1,
            max_batch_size=1,
            profile_only_prefill=False,
            profile_only_decode=True,
            decode_kv_cache_size_list=[1001],
        )

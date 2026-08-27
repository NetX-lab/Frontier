import pytest

from frontier.profiling.attention.main import (
    _filter_standard_attention_inputs_by_memory,
    _validate_explicit_decode_kv_coverage,
)
from frontier.profiling.utils import get_attention_input_combinations


def _explicit_decode_inputs(kv_cache_size: int):
    return get_attention_input_combinations(
        max_seq_len=256,
        max_model_len=512,
        min_batch_size=1,
        max_batch_size=1,
        profile_only_prefill=False,
        profile_only_decode=True,
        decode_kv_cache_size_list=[kv_cache_size],
    )


def test_memory_filter_warns_when_explicit_kv_is_dropped_for_target():
    inputs = _explicit_decode_inputs(511)

    with pytest.warns(RuntimeWarning, match="explicit decode KV values.*511"):
        filtered, retained_explicit_kv = _filter_standard_attention_inputs_by_memory(
            inputs,
            max_num_tokens=256,
            model="test-model",
            tensor_parallel_size=1,
            explicit_decode_kv_cache_sizes=[511],
        )

    assert filtered == []
    assert retained_explicit_kv == set()


def test_explicit_kv_coverage_allows_target_specific_subsets():
    inputs = _explicit_decode_inputs(511)

    with pytest.warns(RuntimeWarning):
        filtered_small, retained_small = _filter_standard_attention_inputs_by_memory(
            inputs,
            max_num_tokens=256,
            model="small-model",
            tensor_parallel_size=1,
            explicit_decode_kv_cache_sizes=[511],
        )
    filtered_large, retained_large = _filter_standard_attention_inputs_by_memory(
        inputs,
        max_num_tokens=512,
        model="large-model",
        tensor_parallel_size=1,
        explicit_decode_kv_cache_sizes=[511],
    )

    assert filtered_small == []
    assert retained_small == set()
    assert len(filtered_large) == 1
    assert retained_large == {511}

    _validate_explicit_decode_kv_coverage(
        explicit_decode_kv_cache_sizes=[511],
        retained_explicit_kv={511},
        target_capacities={
            ("small-model", 1): 256,
            ("large-model", 1): 512,
        },
    )


def test_explicit_kv_coverage_fails_when_all_targets_drop_value():
    with pytest.raises(ValueError, match="no physically legal pairing.*511"):
        _validate_explicit_decode_kv_coverage(
            explicit_decode_kv_cache_sizes=[511],
            retained_explicit_kv=set(),
            target_capacities={
                ("small-model", 1): 256,
                ("medium-model", 1): 384,
            },
        )

"""CPU contracts for DeepSeek-V4 phase classification and causal closed forms."""

from dataclasses import FrozenInstanceError
import random

import pytest

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError, deepseek_v4_padded_q_heads
from frontier.attention.dsv4_features import (
    DSV4_FEATURE_COLUMNS, PHASE_DECODE, PHASE_FIRST_TAIL, PHASE_PREFILL, PHASE_DUMMY,
    build_dsv4_features, build_dsv4_dummy_features, dsv4_feature_columns,
)
from frontier.attention.families import DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT


def build(q, c, decoding, *, variant="csa", ratio=4):
    return build_dsv4_features(q, c, decoding, variant=variant, compress_ratio=ratio,
                               sliding_window=16, index_topk=8, padded_rows=sum(q) + 3)


@pytest.mark.parametrize("q,c,decoding,phase", [
    ([1, 1], [10, 20], [True, True], PHASE_DECODE),
    ([1], [10], [False], PHASE_FIRST_TAIL),
    ([1, 1], [10, 20], [False, True], PHASE_FIRST_TAIL),
    ([2, 3], [0, 10], [False, False], PHASE_PREFILL),
])
def test_phase_uses_scheduled_queries_and_decoding(q, c, decoding, phase):
    assert build(q, c, decoding).phase == phase


@pytest.mark.parametrize("q,decoding", [([1, 2], [False, False]), ([1, 2], [True, False])])
def test_unmeasured_mix_names_query_and_decoding_flags(q, decoding):
    with pytest.raises(DeepSeekV4CoverageError, match="mix") as error:
        build(q, [1, 2], decoding)
    assert str(list(zip(q, decoding))) in str(error.value)


def test_decode_requires_one_query():
    with pytest.raises(ValueError, match="one query row"):
        build([2], [10], [True])


@pytest.mark.parametrize("variant,ratio", [("csa", 4), ("hca", 128), ("uncompressed", 1)])
def test_closed_forms_equal_token_loops_across_boundaries(variant, ratio):
    rng = random.Random(741)
    batches = [([3, 4], [ratio - 1, ratio * 8 - 1]), ([7], [15]), ([8], [ratio * 8 + 20])]
    batches += [([rng.randint(2, 40) for _ in range(3)],
                 [rng.randint(0, ratio * 12 + 40) for _ in range(3)]) for _ in range(100)]
    for q, c in batches:
        values = build(q, c, [False] * len(q), variant=variant, ratio=ratio).values
        keys = [context + token for query, context in zip(q, c) for token in range(1, query + 1)]
        compressed = [key // ratio for key in keys] if variant != "uncompressed" else []
        assert values["sum_swa_key_pairs"] == sum(min(key, 16) for key in keys)
        assert values["sum_compressed_key_pairs"] == sum(compressed)
        expected_selected = sum(min(key, 8) for key in compressed) if variant == "csa" else sum(compressed)
        assert values["sum_selected_key_pairs"] == expected_selected
        assert values["num_compressed_writes"] == (sum((context + query) // ratio - context // ratio
                                                        for query, context in zip(q, c)) if compressed else 0)
        assert values["num_sequences"] == len(q) and values["num_query_tokens"] == sum(q)
        assert values["max_key_len"] == max(context + query for query, context in zip(q, c))
        assert values["min_query_len"] == min(q) and values["max_query_len"] == max(q)
        assert values["min_context_len"] == min(c) and values["max_context_len"] == max(c)
        assert values["padded_rows"] == sum(q) + 3


@pytest.mark.parametrize("variant", ["csa", "hca"])
@pytest.mark.parametrize("ratio", [0, 1, -4])
def test_compressed_variants_require_ratio_above_one(variant, ratio):
    with pytest.raises(ValueError, match="compress_ratio"):
        build([2], [0], [False], variant=variant, ratio=ratio)


def test_uncompressed_does_not_use_ratio():
    assert build([2], [4], [False], variant="uncompressed", ratio=0).values["sum_compressed_key_pairs"] == 0


def test_dummy_has_only_dispatch_rows_and_frozen_phase():
    features = build_dsv4_dummy_features(64)
    assert features.phase == PHASE_DUMMY and features.values == {"padded_rows": 64}
    with pytest.raises(FrozenInstanceError):
        features.phase = PHASE_DECODE


def test_table_matches_csa_family_and_dummy_schema():
    assert set(DSV4_FEATURE_COLUMNS) == {op.name for op in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT["csa"]}
    for family in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT.values():
        assert {op.name for op in family} <= DSV4_FEATURE_COLUMNS.keys()
    for scope, columns in DSV4_FEATURE_COLUMNS.items():
        assert dsv4_feature_columns(scope, PHASE_PREFILL) == columns
        assert dsv4_feature_columns(scope, PHASE_DUMMY) == ("padded_rows",)


@pytest.mark.parametrize("local,expected", [(8, 64), (64, 64), (65, 128), (128, 128)])
def test_shared_padded_query_heads(local, expected):
    assert deepseek_v4_padded_q_heads(local) == expected

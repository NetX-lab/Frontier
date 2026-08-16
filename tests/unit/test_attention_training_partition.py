"""Regression tests for the shared dense-attention training-row contract."""

from __future__ import annotations

import pandas as pd
import pytest

from frontier.attention.training_partition import (
    CACHE_WRITE_FEATURE_COLUMNS,
    DENSE_MIXED_PREFILL_FEATURES,
    build_dense_mixed_prefill_training_rows,
    prepare_dense_cache_write_training_rows,
    validate_cache_write_target_consistency,
    partition_dense_attention_rows,
)


def _true_mixed_row(**overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "is_decode": False,
        "is_mixed_batch": False,
        "is_true_mixed_batch": True,
        "batch_size": 3,
        "prefill_seq_lens": "[8, 12]",
        "prefill_kv_cache_sizes": "[128, 256]",
        "time_stats.attn_prefill.median": 0.3,
    }
    row.update(overrides)
    return row


def test_true_mixed_prefill_prefers_raw_list_contract_over_stale_prefixed_values() -> None:
    """Raw profiler lists are the source of truth when both forms are present."""

    row = _true_mixed_row(
        prefill_mixed_batch_size=99,
        prefill_mixed_kv_cache_size=999,
        prefill_mixed_total_tokens=999,
        prefill_mixed_avg_seq_len=999.0,
        prefill_mixed_min_seq_len=1,
        prefill_mixed_max_seq_len=999,
        prefill_mixed_total_tokens_squared=998001,
        prefill_mixed_seq_len_variance=0.0,
        prefill_mixed_seq_len_cv=0.0,
        prefill_mixed_seq_len_range=998,
        prefill_mixed_batch_variance_interaction=0.0,
        prefill_mixed_batch_cv_interaction=0.0,
    )
    partitions = partition_dense_attention_rows(pd.DataFrame([row]))

    mixed = build_dense_mixed_prefill_training_rows(
        partitions,
        kv_cache_prediction_granularity=64,
        target_col="time_stats.attn_prefill.median",
    )

    actual = mixed.iloc[0]
    assert actual["batch_size"] == 2
    assert actual["kv_cache_size"] == 192
    assert actual["total_tokens"] == 20
    assert actual["avg_seq_len"] == 10.0


def test_true_mixed_prefill_handles_list_valued_prefixed_columns() -> None:
    """A list-valued metadata cell must not make missingness checks ambiguous."""

    row = _true_mixed_row(prefill_mixed_total_tokens=[20, 20])
    partitions = partition_dense_attention_rows(pd.DataFrame([row]))

    mixed = build_dense_mixed_prefill_training_rows(
        partitions,
        kv_cache_prediction_granularity=64,
        target_col="time_stats.attn_prefill.median",
    )

    assert list(mixed.iloc[0][list(DENSE_MIXED_PREFILL_FEATURES)]) == [
        10.0,
        0.4,
        2,
        8.0,
        192,
        12,
        8,
        0.2,
        4,
        4.0,
        20,
        400,
    ]


@pytest.mark.parametrize("granularity", [0, -1, 1.5, True])
def test_mixed_prefill_rejects_non_positive_or_non_integer_granularity(granularity) -> None:
    partitions = partition_dense_attention_rows(pd.DataFrame([_true_mixed_row()]))

    with pytest.raises(ValueError, match="positive integer"):
        build_dense_mixed_prefill_training_rows(
            partitions,
            kv_cache_prediction_granularity=granularity,
            target_col="time_stats.attn_prefill.median",
        )


def test_partition_derives_mixed_prefill_when_metadata_column_is_absent() -> None:
    frame = pd.DataFrame(
        {
            "is_decode": [False, False, True],
            "batch_size": [1, 3, 3],
        }
    )

    partitions = partition_dense_attention_rows(frame)

    assert len(partitions["mixed_prefill"]) == 1
    assert partitions["mixed_prefill"].iloc[0]["batch_size"] == 3
    assert len(partitions["prefill"]) == 1
    assert len(partitions["decode"]) == 1


def test_true_mixed_without_prefill_representation_is_skipped() -> None:
    frame = pd.DataFrame(
        {
            "is_decode": [False],
            "is_true_mixed_batch": [True],
            "batch_size": [3],
            "total_tokens": [1026],
            "time_stats.attn_prefill.median": [1.1],
        }
    )
    partitions = partition_dense_attention_rows(frame)

    mixed = build_dense_mixed_prefill_training_rows(
        partitions,
        kv_cache_prediction_granularity=64,
        target_col="time_stats.attn_prefill.median",
    )

    assert mixed.empty


def test_true_mixed_partial_prefill_representation_fails_fast() -> None:
    frame = pd.DataFrame(
        {
            "is_decode": [False],
            "is_true_mixed_batch": [True],
            "batch_size": [3],
            "prefill_mixed_batch_size": [2],
            "time_stats.attn_prefill.median": [1.1],
        }
    )
    partitions = partition_dense_attention_rows(frame)

    with pytest.raises(ValueError, match="complete prefill-side feature contract"):
        build_dense_mixed_prefill_training_rows(
            partitions,
            kv_cache_prediction_granularity=64,
            target_col="time_stats.attn_prefill.median",
        )


def test_cache_write_training_rows_use_true_mixed_decode_context_and_total_batch() -> None:
    frame = pd.DataFrame(
        [
            {
                "is_true_mixed_batch": False,
                "total_tokens": 32,
                "kv_cache_size": 0,
                "batch_size": 1,
            },
            {
                "is_true_mixed_batch": True,
                "total_tokens": 18,
                "kv_cache_size": 0,
                "batch_size": 4,
                "decode_avg_kv_cache_size": 64,
                "total_batch_size": 4,
            },
        ]
    )

    prepared = prepare_dense_cache_write_training_rows(frame)

    assert CACHE_WRITE_FEATURE_COLUMNS == (
        "total_tokens",
        "kv_cache_size",
        "batch_size",
    )
    assert tuple(prepared.loc[1, list(CACHE_WRITE_FEATURE_COLUMNS)]) == (18, 64, 4)
    assert tuple(prepared.loc[0, list(CACHE_WRITE_FEATURE_COLUMNS)]) == (32, 0, 1)


def test_cache_write_training_rows_reject_true_mixed_without_runtime_context() -> None:
    frame = pd.DataFrame(
        [
            {
                "is_true_mixed_batch": True,
                "total_tokens": 18,
                "kv_cache_size": 0,
                "batch_size": 4,
            }
        ]
    )

    with pytest.raises(ValueError, match="decode_avg_kv_cache_size.*total_batch_size"):
        prepare_dense_cache_write_training_rows(frame)


def test_cache_write_training_rows_use_zero_context_for_pure_prefill() -> None:
    frame = pd.DataFrame(
        [
            {
                "is_prefill": True,
                "is_true_mixed_batch": False,
                "total_tokens": 64,
                "kv_cache_size": 64,
                "batch_size": 1,
            }
        ]
    )

    prepared = prepare_dense_cache_write_training_rows(frame)

    assert prepared.loc[0, "kv_cache_size"] == 0


@pytest.mark.parametrize(
    "column,value",
    [("total_tokens", 1.5), ("kv_cache_size", 0.5), ("batch_size", 1.5)],
)
def test_cache_write_training_rows_require_integer_shape_axes(column: str, value: float) -> None:
    frame = pd.DataFrame(
        [
            {
                "is_true_mixed_batch": False,
                "total_tokens": 8.0,
                "kv_cache_size": 0.0,
                "batch_size": 1.0,
            }
        ]
    )
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match="integers"):
        prepare_dense_cache_write_training_rows(frame)


def test_cache_write_target_conflict_after_normalization_fails_fast() -> None:
    frame = pd.DataFrame(
        {
            "total_tokens": [8, 8],
            "kv_cache_size": [0, 0],
            "batch_size": [1, 1],
            "time_stats.attn_kv_cache_save.median": [0.1, 0.3],
        }
    )

    with pytest.raises(ValueError, match="conflicting targets"):
        validate_cache_write_target_consistency(
            frame,
            target_col="time_stats.attn_kv_cache_save.median",
        )


def test_cache_write_repeated_measurements_require_explicit_opt_in() -> None:
    frame = pd.DataFrame(
        {
            "total_tokens": [8, 8],
            "kv_cache_size": [0, 0],
            "batch_size": [1, 1],
            "time_stats.attn_kv_cache_save.median": [0.1, 0.3],
        }
    )

    validate_cache_write_target_consistency(
        frame,
        target_col="time_stats.attn_kv_cache_save.median",
        allow_repeated_measurements=True,
    )

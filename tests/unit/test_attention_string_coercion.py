"""Unit tests for the shared truthy-string coercion helper (cluster C4).

The attention profiling / training / prediction pipelines historically repeated the
same inline boolean normalization
``series.astype(str).str.strip().str.lower().isin({"1","true","t","yes","y"})``
in 11 places. ``frontier.attention.string_coercion`` is the single source of truth;
these tests pin its truth table AND prove byte-for-byte equivalence to the historical
inline expression so the de-duplication cannot change dense/MLA numerics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frontier.attention.string_coercion import (
    _TRUTHY_STRINGS,
    coerce_truthy_bool,
    coerce_truthy_int,
)


def _inline_reference_bool(series: pd.Series) -> pd.Series:
    """The exact historical inline expression, replicated for the equivalence proof."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def test_truthy_token_set_is_exactly_the_historical_tokens():
    assert set(_TRUTHY_STRINGS) == {"1", "true", "t", "yes", "y"}


def test_coerce_truthy_bool_matches_documented_truth_table():
    s = pd.Series(["1", "true", "t", "yes", "y", "0", "false", ""])
    assert coerce_truthy_bool(s).tolist() == [
        True, True, True, True, True, False, False, False
    ]


def test_coerce_truthy_int_matches_documented_truth_table():
    s = pd.Series(["1", "true", "t", "yes", "y", "0", "false", ""])
    assert coerce_truthy_int(s).tolist() == [1, 1, 1, 1, 1, 0, 0, 0]


def test_coerce_handles_case_and_whitespace_variants():
    s = pd.Series([" TRUE ", "Yes", "T", "  y", "N", "no", " 0 "])
    assert coerce_truthy_bool(s).tolist() == [
        True, True, True, True, False, False, False
    ]


@pytest.mark.parametrize(
    "values",
    [
        ["1", "true", "t", "yes", "y", "0", "false", "", "True", " Yes ", "NO", "1.0"],
        [1, 0, 1, True, False],
        [np.nan, None, "nan", "none", "yes"],
    ],
)
def test_coerce_truthy_bool_is_byte_identical_to_inline(values):
    s = pd.Series(values)
    pd.testing.assert_series_equal(coerce_truthy_bool(s), _inline_reference_bool(s))


def test_coerce_truthy_int_is_byte_identical_to_inline_astype_int():
    s = pd.Series(["1", "true", "t", "yes", "y", "0", "false", "", np.nan])
    pd.testing.assert_series_equal(
        coerce_truthy_int(s), _inline_reference_bool(s).astype(int)
    )


def test_coerce_preserves_index():
    s = pd.Series(["yes", "no"], index=[7, 11])
    out = coerce_truthy_bool(s)
    assert out.index.tolist() == [7, 11]

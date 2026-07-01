"""Shared truthy-string coercion for profiling / training dataframes (cluster C4).

The attention profiling, training, and prediction pipelines repeatedly normalize
free-form boolean-ish columns (``is_prefill``, ``is_mixed_batch``,
``is_true_mixed_batch``) that arrive as heterogeneous strings (``"1"``, ``"True"``,
``"t"``, ``"yes"``, ``"y"`` and their cased / whitespace variants). This module is the
single source of truth for that coercion so the historical inline copies cannot drift.

The functions reproduce the historical inline expression byte-for-byte:
``series.astype(str).str.strip().str.lower().isin(_TRUTHY_STRINGS)``.
"""

from __future__ import annotations

import pandas as pd

# Canonical truthy tokens (already lower-cased and stripped). Anything else -> False.
# Kept verbatim from the 11 historical inline copies; changing this set changes the
# dense/MLA training truth table, so it is treated as a frozen contract.
_TRUTHY_STRINGS = frozenset({"1", "true", "t", "yes", "y"})


def coerce_truthy_bool(series: pd.Series) -> pd.Series:
    """Coerce a heterogeneous-string Series to a boolean truthiness Series.

    Equivalent (byte-for-byte) to the historical inline expression
    ``series.astype(str).str.strip().str.lower().isin({"1","true","t","yes","y"})``.
    ``NaN`` / ``None`` stringify to ``"nan"`` / ``"none"`` via ``astype(str)`` and
    therefore map to ``False``; callers needing a NaN-aware default apply ``.where``
    on the result themselves (preserving the prior per-call-site behavior).
    """
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(_TRUTHY_STRINGS)
    )


def coerce_truthy_int(series: pd.Series) -> pd.Series:
    """Integer (0 / 1) form of :func:`coerce_truthy_bool`."""
    return coerce_truthy_bool(series).astype(int)

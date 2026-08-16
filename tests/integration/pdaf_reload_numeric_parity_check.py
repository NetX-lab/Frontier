"""Compare first-run and cache-reload request metrics without coercion bugs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _read_row(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"expected exactly one request row in {path}, got {len(frame)}")
    return frame.iloc[0]


def _compare(first: pd.Series, reload: pd.Series) -> list[dict[str, float | str]]:
    differences: list[dict[str, float | str]] = []
    for column in sorted(set(first.index) & set(reload.index)):
        try:
            first_value = float(first[column])
            reload_value = float(reload[column])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(first_value) or not math.isfinite(reload_value):
            continue
        if first_value != reload_value:
            differences.append(
                {
                    "field": column,
                    "first": first_value,
                    "reload": reload_value,
                    "abs_diff": abs(first_value - reload_value),
                }
            )
    return differences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-request-metrics", type=Path, required=True)
    parser.add_argument("--reload-request-metrics", type=Path, required=True)
    args = parser.parse_args()

    first = _read_row(args.first_request_metrics)
    reload = _read_row(args.reload_request_metrics)
    common_fields = sorted(set(first.index) & set(reload.index))
    numeric_fields = []
    for field in common_fields:
        try:
            first_value = float(first[field])
            reload_value = float(reload[field])
        except (TypeError, ValueError):
            continue
        if math.isfinite(first_value) and math.isfinite(reload_value):
            numeric_fields.append(field)

    differences = _compare(first, reload)
    if differences:
        raise AssertionError(
            f"reload request metrics differ in {len(differences)} fields: {differences}"
        )

    key_metrics = {
        field: float(first[field])
        for field in common_fields
        if field in {"ttft", "request_e2e_time", "throughput", "transfer_kv_cache"}
    }
    print(
        json.dumps(
            {
                "first_request_metrics": str(args.first_request_metrics),
                "reload_request_metrics": str(args.reload_request_metrics),
                "common_numeric_fields": len(numeric_fields),
                "nonzero_differences": len(differences),
                "max_abs_diff": 0.0,
                "key_metrics": key_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

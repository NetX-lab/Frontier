"""Build a per-operator calibration table without retaining obsolete expert costs."""

import argparse
import json
from pathlib import Path

import pandas as pd
from frontier.types import MeasurementType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    args = parser.parse_args()
    base = pd.read_csv(args.base, float_precision="round_trip")
    corrected = pd.read_csv(args.corrected, float_precision="round_trip")
    selected = corrected.loc[
        corrected.num_tokens.isin(args.tokens) & corrected.calibration_split.eq("anchor")
    ].copy()
    if selected.empty:
        raise ValueError("No measured anchor rows match the requested global token counts")
    gg_columns = [c for c in base if c.startswith("time_stats.moe_grouped_gemm.")]
    other_columns = [c for c in selected if c.startswith("time_stats.") and c not in gg_columns]
    base[gg_columns] = float("nan")
    selected[other_columns] = float("nan")
    for frame, source, owner in [(base, args.base, "unchanged_gating_shuffling"),
                                  (selected, args.corrected, "corrected_expert_path")]:
        frame["measurement_type"] = frame["measurement_type"].map(
            lambda value: MeasurementType.from_string(str(value)).value
        )
        frame["calibration_source_file"] = str(source.resolve())
        frame["calibration_source_csv_line"] = frame.index + 2
        frame["calibration_target_owner"] = owner
    result = pd.concat([base, selected], ignore_index=True)
    target = "time_stats.moe_grouped_gemm.median"
    assert result.loc[result.calibration_target_owner.eq("unchanged_gating_shuffling"), target].isna().all()
    assert result[target].notna().sum() == len(selected)
    assert result.loc[result[target].notna(), "num_tokens"].isin(args.tokens).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    receipt = {
        "status": "PER_OPERATOR_TABLE_PREPARED_RUNTIME_TRAINING_PENDING",
        "base_source": str(args.base.resolve()),
        "corrected_source": str(args.corrected.resolve()),
        "output": str(args.output.resolve()),
        "global_token_anchors": args.tokens,
        "retained_unchanged_rows": len(base),
        "corrected_expert_rows": len(selected),
        "obsolete_expert_targets_retained": 0,
        "holdout_rows_used_for_corrected_expert_training": 0,
        "operation_target_counts": {
            c: int(result[c].notna().sum()) for c in result
            if c.startswith("time_stats.") and c.endswith(".median")
        },
        "table_contract": "Per-operator sparse training input; not a dense canonical profiling export",
    }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

"""Merge the PREFILL and DECODE Step profile tables into one predictor input set.

Frontier's measured predictor reads one table per family and measurement type for both P/D
roles. Each role package measures its own cells, so the merged table is the union of the role
rows; rows of both roles at the same operator key are averaged by the predictor's exact lookup.
A role may have several packages, for example a later package that adds MoE rows.
The script also writes the predictor options that select the merged tables, the kernel gap and
the NIXL READ profile.

Usage: merge_role_profiles.py --prefill <dir>... --decode <dir>... --output <dir>
           --kernel-gap <csv> --nixl-read-profile <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Predictor option -> (profile family directory, table name).
TABLES = {
    "linear_op_input_file": ("linear", "linear_op.csv"),
    "linear_op_kernel_only_input_file": ("linear", "linear_op_kernel_only.csv"),
    "moe_input_file": ("moe", "moe.csv"),
    "moe_kernel_only_input_file": ("moe", "moe_kernel_only.csv"),
    "atten_input_file": ("attention", "step5_attention.csv"),
    "atten_kernel_only_input_file": ("attention", "step5_attention_kernel_only.csv"),
}


def role_table(root: Path, family: str, name: str) -> pd.DataFrame | None:
    paths = sorted((root / family).rglob(name))
    if len(paths) > 1:
        raise ValueError(f"{root / family} holds {len(paths)} tables named {name}")
    if not paths:
        return None
    frame = pd.read_csv(paths[0])
    # The MoE profiler measured EP rank 0 before it recorded the rank of generated routing.
    if family == "moe" and "moe_ep_rank" not in frame:
        frame["moe_ep_rank"] = 0
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", type=Path, nargs="+", required=True)
    parser.add_argument("--decode", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-gap", type=Path, required=True)
    parser.add_argument("--nixl-read-profile", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    options = dict(enable_dummy_mode=False, skip_cpu_overhead_modeling=True,
                   kernel_gap_input_file=str(args.kernel_gap.resolve()),
                   nixl_read_profile=str(args.nixl_read_profile.resolve()))
    counts = {}
    for option, (family, name) in TABLES.items():
        frames = [(role, frame) for role, roots in (("PREFILL", args.prefill), ("DECODE", args.decode))
                  for root in roots if (frame := role_table(root, family, name)) is not None]
        if not frames:
            raise ValueError(f"Neither role produced {family}/{name}")
        columns = {frozenset(frame.columns) for _, frame in frames}
        if len(columns) != 1:
            raise ValueError(f"The role tables {family}/{name} have different columns")
        path = args.output / name
        pd.concat([frame for _, frame in frames], ignore_index=True).to_csv(path, index=False)
        options[option] = str(path.resolve())
        counts[name] = [[role, len(frame)] for role, frame in frames]
    (args.output / "predictor_options.json").write_text(json.dumps(options, indent=2) + "\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Promote path-B block rows and accepted replay cost rows with hashed provenance.

G2h-02 replay is a structure fixture only. Pass a scratch output directory and
manifest for it; the lead promotes the accepted G2h-04 replay separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontier.execution_time_predictor.dsv4_runner import BLOCK_TABLE, REPLAY_TABLE, WRITE_SCOPE
from frontier.profiling.common.runtime_identity import file_sha256
from frontier.profiling.model_runner.dsv4_runner_tables import (
    reduce_block_rows, reduce_replay_rows, write_table, read_rows,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TABLES = ROOT / "data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash"
GRID = HERE.parent / "g2h_runner/block_operator_grid.json"


def source_row_count(path):
    """CSV data rows, streamed JSONL records, or top-level JSON records."""
    path = Path(path)
    if path.suffix == ".csv":
        return len(read_rows(path))
    if path.suffix == ".jsonl":
        with path.open() as handle:
            return sum(1 for _ in handle)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return len(payload) if isinstance(payload, list) else 1
    raise ValueError(f"No runner source row-count format for {path}")


def promote_runner_tables(*, output_dir, manifest_path, block_dir=None,
                          replay_raw_dir=None, attention_table=TABLES / "attention_graph_replay_kernel_only.csv",
                          block_run_tag=None, replay_run_tag=None, grid_path=GRID):
    inputs, outputs, comparisons = [], [], []
    if block_dir is not None:
        if not block_run_tag:
            raise ValueError("Block promotion needs its source run tag")
        rows, sources = reduce_block_rows(block_dir, grid_path, block_run_tag)
        target = Path(output_dir) / BLOCK_TABLE
        write_table(target, rows)
        inputs.extend(sources)
        outputs.append(target)
    if replay_raw_dir is not None:
        if not replay_run_tag:
            raise ValueError("Replay promotion needs its source run tag")
        if "g2h_runner_02" in str(replay_raw_dir) and Path(output_dir).resolve() == TABLES.resolve():
            raise ValueError("G2h-02 replay is a structure fixture; use scratch output")
        rows, sources, comparisons = reduce_replay_rows(replay_raw_dir, attention_table, replay_run_tag)
        target = Path(output_dir) / REPLAY_TABLE
        write_table(target, rows)
        inputs.extend(sources)
        outputs.append(target)
        block_path = Path(output_dir) / BLOCK_TABLE
        if not block_path.exists():
            block_path = TABLES / BLOCK_TABLE
        if block_path.exists():
            block_costs = [float(row[f"kernel_stats.{WRITE_SCOPE}.median"]) +
                           float(row[f"memcpy_stats.{WRITE_SCOPE}.median"])
                           for row in read_rows(block_path) if row["operator"] == WRITE_SCOPE]
            for row in comparisons:
                if "kernel_median_ms" in row:
                    row["cost_median_ms"] = row["kernel_median_ms"] + row["memcpy_median_ms"]
                    row["path_b_cost_range_ms"] = [min(block_costs), max(block_costs)]
    if not outputs:
        raise ValueError("Supply block and/or replay inputs")
    manifest = {
        "inputs": [{"path": str(path), "sha256": file_sha256(path), "row_count": source_row_count(path)}
                   for path in dict.fromkeys(inputs)],
        "outputs": [{"path": str(path), "sha256": file_sha256(path), "row_count": len(read_rows(path))}
                    for path in outputs],
        "source_run_tags": {"block": block_run_tag, "replay": replay_run_tag},
        "replay_write_comparison": comparisons,
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-dir", type=Path)
    parser.add_argument("--replay-raw-dir", type=Path)
    parser.add_argument("--block-run-tag")
    parser.add_argument("--replay-run-tag")
    parser.add_argument("--grid-path", type=Path, default=GRID)
    parser.add_argument("--attention-table", type=Path, default=TABLES / "attention_graph_replay_kernel_only.csv")
    parser.add_argument("--output-dir", type=Path, default=TABLES)
    parser.add_argument("--manifest-path", type=Path, default=HERE / "runner_manifest.json")
    args = parser.parse_args()
    print(json.dumps(promote_runner_tables(**vars(args))["outputs"], indent=2))


if __name__ == "__main__":
    main()

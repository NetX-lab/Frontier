"""Merge the CSVs of one published profiling run into the tables the predictor reads.

A profiler invocation writes its own files, so one table can be spread over
several of them: the attention profiler writes standard and true-mixed rows to
separate files, and each MoE gating context is a separate invocation. The
predictor reads one file per table (`linear_op.csv`, `attention.csv`,
`moe.csv`). This tool writes each table from the files named for it.

The run directory is one that `run_profiling_plan.py` published. Its `COMPLETE`
marker must read `status=0`, and every named source must match the SHA-256 that
the run's `manifest.json` recorded for it.

Merge rules, per table:

- Columns are the union of the sources' columns in first-seen order; a cell a
  source does not have is left empty.
- Rows are every source row, in the order the sources are named, so an output
  row's provenance is its position (the receipt records each source's range).
- A row's identity is every column except the `time_stats.` measurements. Two
  rows with the same identity are rejected, and so is a row without any
  measurement.

With `--base-dir`, the output extends an earlier supplement that this tool
wrote. Each base table must match the SHA-256 its receipt recorded; it keeps
every row, first and in order, and a base table the run does not name is
carried over unchanged. A run row whose identity equals a base row is not
added: the base keeps its measurement, and the receipt lists the run row's
location and measurements under `repeated_base_rows`.

The output directory must not exist. It is assembled beside its target and
renamed into place, with `merge_receipt.json` and then `COMPLETE` written last.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

MEASUREMENT_PREFIX = "time_stats."


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, help="earlier supplement written by this tool")
    parser.add_argument(
        "--table", nargs="+", action="append", required=True, metavar=("NAME", "SOURCE"),
        help="output file name, then its sources as paths relative to --run-dir",
    )
    args = parser.parse_args(argv)
    for table in args.table:
        if len(table) < 2:
            parser.error(f"--table {table[0]} names no source")
    return args


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def published_csvs(run_dir: Path) -> dict[str, str]:
    marker = (run_dir / "COMPLETE").read_text().strip()
    if marker != "status=0":
        raise ValueError(f"{run_dir} is not a passing run: COMPLETE reads {marker!r}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return {
        entry["path"]: entry["sha256"]
        for invocation in manifest["invocations"]
        for entry in invocation["csvs"]
    }


def supplement_tables(base_dir: Path) -> dict[str, str]:
    marker = (base_dir / "COMPLETE").read_text().strip()
    if marker != "status=0":
        raise ValueError(f"{base_dir} is not a complete supplement: COMPLETE reads {marker!r}")
    receipt = json.loads((base_dir / "merge_receipt.json").read_text())
    return {name: table["sha256"] for name, table in receipt["tables"].items()}


def identity(row: dict) -> tuple:
    return tuple(sorted(
        (key, value) for key, value in row.items()
        if value and not key.startswith(MEASUREMENT_PREFIX)
    ))


def measurements(row: dict) -> dict:
    return {key: value for key, value in row.items() if key.startswith(MEASUREMENT_PREFIX) and value}


def merge_table(run_dir: Path, sources: list[str], recorded: dict[str, str], base: Path | None, base_sha256: str | None):
    columns: list[str] = []
    rows: list[dict] = []
    provenance: dict = {"sources": []}
    base_rows: dict[tuple, int] = {}
    if base is not None:
        if sha256(base) != base_sha256:
            raise ValueError(f"{base} is not the table its supplement recorded")
        with base.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames)
            rows = list(reader)
        base_rows = {identity(row): index for index, row in enumerate(rows)}
        provenance["base"] = {"path": str(base), "sha256": base_sha256, "output_rows": [0, len(rows)]}
        provenance["repeated_base_rows"] = []
    first_seen: dict[tuple, str] = {}
    for source in sources:
        path = run_dir / source
        digest = sha256(path)
        if recorded.get(source) != digest:
            raise ValueError(f"{source} is not the file the run published")
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns += [column for column in reader.fieldnames if column not in columns]
            start = len(rows)
            for line, row in enumerate(reader, start=2):
                location = f"{source}:{line}"
                if not measurements(row):
                    raise ValueError(f"{location} has no measurement")
                key = identity(row)
                if key in first_seen:
                    raise ValueError(f"{location} repeats the identity of {first_seen[key]}")
                first_seen[key] = location
                if key in base_rows:
                    provenance["repeated_base_rows"].append({
                        "location": location, "base_row": base_rows[key], "measurements": measurements(row),
                    })
                    continue
                rows.append(row)
        provenance["sources"].append({"path": source, "sha256": digest, "output_rows": [start, len(rows)]})
    return columns, rows, provenance


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists():
        print(f"output already exists: {args.output_dir}", file=sys.stderr)
        return 2
    recorded = published_csvs(args.run_dir)
    named = {name: sources for name, *sources in args.table}
    base = supplement_tables(args.base_dir) if args.base_dir else {}
    merged = {
        name: merge_table(
            args.run_dir, named.get(name, []), recorded,
            args.base_dir / name if name in base else None, base.get(name),
        )
        for name in [*base, *(name for name in named if name not in base)]
    }
    staging = args.output_dir.with_name(args.output_dir.name + ".partial")
    staging.mkdir(parents=True)
    tables = {}
    for name, (columns, rows, provenance) in merged.items():
        with (staging / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(rows)
        tables[name] = {
            "rows": len(rows),
            "columns": len(columns),
            "sha256": sha256(staging / name),
            **provenance,
        }
    receipt = {"run_dir": str(args.run_dir), "tables": tables}
    if args.base_dir:
        receipt = {"base_dir": str(args.base_dir), **receipt}
    (staging / "merge_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    staging.rename(args.output_dir)
    (args.output_dir / "COMPLETE").write_text("status=0\n")
    print(json.dumps({name: table["rows"] for name, table in tables.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

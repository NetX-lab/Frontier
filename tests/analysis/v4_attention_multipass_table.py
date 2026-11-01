"""Write scratch attention tables whose kernel-time targets use several independent runs.

The published table holds one measurement run per workload. For the workloads that
the reference run and two independent runs all measured, write three tables for a
controlled A6 comparison:

- ``original_shared``: only those workloads, with the published values unchanged;
- ``median3_shared``: the same rows, each ``time_stats.<scope>.median`` replaced by
  the median of the three runs;
- ``median3_all``: every published row, with the shared workloads replaced as above.

Only kernel-time medians change; gap statistics, counts and identities are copied
unchanged. Each table keeps the published file name, which the V4 loader requires.
The tables are diagnostic and never replace the published input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import resource
import statistics

from v4_attention_repeats import read_rows

TABLE_NAME = "attention_graph_replay_kernel_only.csv"


def row_key(row):
    return row["case_id"], row["attention_variant"], str(row["plans_tile_schedule"])


def render(fieldnames, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-table", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
    published_bytes = args.published_table.read_bytes()
    reader = csv.DictReader(io.StringIO(published_bytes.decode()))
    fieldnames, published = reader.fieldnames, list(reader)
    if render(fieldnames, published) != published_bytes:
        raise ValueError("Rewriting the published table does not reproduce its bytes")
    reference, reference_input = read_rows(args.reference)
    first, first_input = read_rows(args.comparison[0])
    second, second_input = read_rows(args.comparison[1])
    if set(reference) != set(first) or set(reference) != set(second):
        raise ValueError("The three runs do not measure the same workload/layer identities")
    medians = {}
    for key, row in reference.items():
        values = {}
        for column, value in row.items():
            if column.startswith("time_stats.") and column.endswith(".median"):
                runs = (value, first[key][column], second[key][column])
                if min(runs) <= 0:
                    raise ValueError(f"Non-positive measured duration: {row['case_id']} {column}")
                values[column] = statistics.median(runs)
        medians[row_key(row)] = (row, values)
    shared = [row for row in published if row_key(row) in medians]
    if len(shared) != len(medians):
        raise ValueError(f"{len(medians)} measured rows match {len(shared)} published rows")
    for row in shared:
        measured, values = medians[row_key(row)]
        for column in values:
            if abs(float(row[column]) - measured[column]) > 1e-12:
                raise ValueError(f"Published median differs from the reference run: {row['case_id']} {column}")
        if {c for c in fieldnames if c.startswith("time_stats.") and c.endswith(".median") and row[c]} != set(values):
            raise ValueError(f"Measured scopes differ from the published row: {row['case_id']}")

    def replace(row):
        if row_key(row) not in medians:
            return row
        return {**row, **{column: repr(value) for column, value in medians[row_key(row)][1].items()}}

    tables = {"original_shared": shared, "median3_shared": [replace(row) for row in shared],
              "median3_all": [replace(row) for row in published]}
    replaced_values = sum(len(values) for _, values in medians.values())
    manifest = {"classification": "DIAGNOSTIC_MULTIPASS_TARGET_TABLES",
                "published_table": {"path": str(args.published_table.resolve()),
                                    "sha256": hashlib.sha256(published_bytes).hexdigest(),
                                    "rows": len(published), "rewrite_byte_identical": True},
                "runs": {"reference": reference_input, "comparisons": [first_input, second_input]},
                "shared_rows": len(shared), "shared_cases": len({row["case_id"] for row in shared}),
                "replaced_kernel_time_medians": replaced_values,
                "unchanged_columns": "all except time_stats.<scope>.median of shared rows", "tables": {}}
    for name, rows in tables.items():
        payload = render(fieldnames, rows)
        path = args.output_dir / name / TABLE_NAME
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        manifest["tables"][name] = {"path": str(path.resolve()), "rows": len(rows),
                                    "sha256": hashlib.sha256(payload).hexdigest()}
        print(f"{name}: {len(rows)} rows -> {path}", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(shared)} shared rows, {replaced_values} kernel-time medians replaced", flush=True)


if __name__ == "__main__":
    main()

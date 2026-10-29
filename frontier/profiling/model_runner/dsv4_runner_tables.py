"""Torch-free reduction of native runner scope rows into exact consumer tables.

Raw replay JSONL is streamed; only step records and scope activity counters are
retained. Scope medians remain producer values, never a whole-step residual.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from frontier.execution_time_predictor.dsv4_runner import (
    ZERO_SCOPE, WRITE_SCOPE, INPUT_SCOPE, SAMPLING_SCOPE, PREPARATION_COLUMNS,
    ACTIVITY_COLUMNS, NEW_ROW_ENTRIES,
)
from frontier.profiling.model_runner.dsv4_block_operators import BlockGrid
from frontier.profiling.model_runner.dsv4_runner_rows import COST

csv.field_size_limit(10**9)


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_table(path, rows):
    if not rows:
        raise ValueError(f"No runner rows for {path}")
    columns = list(dict.fromkeys(column for row in rows for column in row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _counts(counter):
    return tuple(counter[name] for name in ("kernel", "memcpy", "memset"))


def _constant_counts(coordinate, samples, seen):
    values = set(map(_counts, samples))
    if len(values) != 1:
        raise ValueError(f"Runner activity count varies at {coordinate}: {sorted(values)}")
    counts = values.pop()
    if coordinate in seen and seen[coordinate] != counts:
        raise ValueError(f"Runner activity count varies at {coordinate}: {seen[coordinate]} vs {counts}")
    seen[coordinate] = counts
    return dict(zip(ACTIVITY_COLUMNS, counts)) | {"activity_count": sum(counts)}


def reduce_block_rows(block_dir, grid_path, run_tag):
    block_dir = Path(block_dir)
    layout = json.loads((block_dir / "layout.json").read_text())
    grid = BlockGrid(json.loads(Path(grid_path).read_text()))
    entries = grid.entries(layout["block_sizes"])
    result, seen, inputs = [], {}, [block_dir / "block_operators.csv", block_dir / "layout.json", Path(grid_path)]
    for row in read_rows(inputs[0]):
        if row["profile_method"] != "record_function" or row["row_role"] != COST:
            continue
        name = row["operator"]
        coordinate = ((name, int(row["n"])) if name == ZERO_SCOPE
                      else (name, int(row["w_c"]), int(row["r"])))
        suffix = (f"n-{coordinate[1]}" if name == ZERO_SCOPE
                  else f"w_c-{coordinate[1]}_r-{coordinate[2]}")
        trace = block_dir / f"operator-{name}_{suffix}_profile_method-record_function.trace.intervals.json"
        inputs.append(trace)
        by_step = defaultdict(Counter)
        for interval in json.loads(trace.read_text()):
            if interval.get("owner") == [-1, name] and interval["activity"] != "hold":
                by_step[interval["step_index"]][interval["activity"]] += 1
        result.append(row | _constant_counts(coordinate, by_step.values(), seen) | {
            NEW_ROW_ENTRIES: json.dumps(entries, separators=(",", ":")),
            "first_chunk_tokens": grid.payload["first_chunk_tokens"],
            "source_row_id": trace.name.removesuffix(".trace.intervals.json"),
            "engine_index": "native_layout", "run_tag": run_tag,
        })
    return result, inputs


def descriptor_index(attention_path):
    descriptors = defaultdict(set)
    for row in read_rows(attention_path):
        descriptors[row["case_id"]].add(tuple(row[c] for c in PREPARATION_COLUMNS))
    for case, values in descriptors.items():
        if len(values) != 1:
            raise ValueError(f"Runner case {case} has more than one descriptor: {sorted(values)}")
    return {case: dict(zip(PREPARATION_COLUMNS, next(iter(values)))) for case, values in descriptors.items()}


def _replay_records(path):
    counts, steps, writes = defaultdict(Counter), {}, defaultdict(list)
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            kind = row["kind"]
            if kind == "step":
                steps[(row["case_id"], row["step_index"])] = row
            elif kind == "interval" and row.get("scope") in (INPUT_SCOPE, SAMPLING_SCOPE, WRITE_SCOPE):
                if row["activity"] != "hold":
                    counts[(row["case_id"], row["scope"], row["step_index"])][row["activity"]] += 1
                    if row["scope"] == WRITE_SCOPE:
                        writes[(row["case_id"], row["step_index"])].append(row)
    return counts, steps, writes


def reduce_replay_rows(raw_dir, attention_path, run_tag):
    descriptors = descriptor_index(attention_path)
    result, seen, comparisons, inputs = [], {}, [], [Path(attention_path)]
    for raw in sorted(Path(raw_dir).glob("dsv4_raw_e*.jsonl")):
        table = raw.with_suffix(".runner.csv")
        inputs.extend((raw, table))
        counts, steps, writes = _replay_records(raw)
        for row in read_rows(table):
            case, name = row["case_id"], row["operator"]
            if name == WRITE_SCOPE:
                comparisons.append({"case_id": case, "engine_index": row["engine_index"],
                                    "kernel_median_ms": float(row[f"kernel_stats.{name}.median"]),
                                    "memcpy_median_ms": float(row[f"memcpy_stats.{name}.median"])})
            if row["row_role"] != COST:
                continue
            if case not in descriptors:
                raise ValueError(f"Runner case {case} has no attention descriptor")
            descriptor = descriptors[case]
            if name == INPUT_SCOPE:
                coordinate = {c: descriptor[c] for c in PREPARATION_COLUMNS}
            elif name == SAMPLING_SCOPE:
                coordinate = {"b": int(descriptor["batch_size"])}
                # Native sample() uses one logits index per real request,
                # never graph request padding (model_runner.py:1122-1130,1339-1353).
                for (step_case, _), step in steps.items():
                    if step_case == case and len(step["requests"]) != coordinate["b"]:
                        raise ValueError(f"Sampling b disagrees with real requests for {case}")
            else:
                raise ValueError(f"Unexpected cost runner scope {name}")
            key = (name, *coordinate.values())
            samples = [counter for (sample_case, scope, _), counter in counts.items()
                       if sample_case == case and scope == name]
            result.append(row | coordinate | _constant_counts(key, samples, seen) | {"run_tag": run_tag})
        for key, step in steps.items():
            if key not in writes:
                comparisons.append({"case_id": key[0], "step_index": key[1], "write_activity": 0,
                                    "staging_record": {name: value for name, value in step.items()
                                                       if "stag" in name or "block_id" in name}})
    return result, inputs, comparisons

"""Measure the native side of the operator gap of one DP x PP case from its clean, op and kernel runs.

The three runs replay one trace on one engine. `clean` has no instrumentation.
`op` times each operator scope with CUDA events as it runs (scope mode
`default`, launch gaps inside a scope included). `kernel` syncs the device
before each scope and after each collective (scope mode `kernel_only`), so only
its scope times are read, never its step times. Each run directory holds the
`run/` and `extraction/` directories the case launcher and extractor wrote.

The tool computes numbers and draws no conclusions. Into `--output-dir` it
writes:

- `batch_windows_<run>.csv` (op, kernel): one row per (pp_rank, dp_rank,
  batch_id) joining `op_timing.jsonl`, `batch_log.jsonl` and
  `pp_boundary.jsonl`. `uncovered_ms` is the forward window less the summed
  non-collective scope time and the summed collective scope time.
- `scope_by_shape.csv`, `window_by_shape.csv`: per run, pp_rank and shape
  class, the scope times and the parts of the batch window.
- `host_gap_by_shape.csv`: per scope, the op-run per-batch time less the
  kernel-run per-batch time.
- `peer_state_<run>.csv`, `peer_state_summary.csv`: what the other dp_rank at
  the same pp_rank did over each batch's forward window.
- `dummy_pass_summary.csv`: DP dummy pass durations per run and engine.
- `decode_cycles_<run>.csv` (clean, op) and `measurement_cost.csv`: the decode
  cycle split and, per cycle class, the op-run median less the clean-run median.
  Cycles before the run's first admission of a formal request are warmup and
  left out; the summary counts them.
- `alignment.csv`: the `difflib` opcodes of the op run's scheduling iterations
  against the clean run's, per engine.
- `summary.json`: input hashes and row counts, the engine to dp_rank mapping,
  the clock check, join problems, the alignment summary and the written files.

A batch that holds a warmup request, or whose key has a join problem, stays in
`batch_windows_<run>.csv` with its flag and is left out of every other batch
table. Join problems (a key missing from a source, a key repeated in a source,
a batch attribute that differs between sources) are listed with counts in the
summary's `problems`, never dropped silently.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BATCH_KEY = ["pp_rank", "dp_rank", "batch_id"]
COLLECTIVE_SCOPES = ("expert_parallel_alltoall_dispatch", "expert_parallel_alltoall_combine")
WINDOW_PARTS = (
    "forward_ms", "scope_sum_ms", "collective_sum_ms", "uncovered_ms",
    "preprocess_ms", "send_ms", "recv_ms", "execute_tail_ms",
)
CYCLE_PARTS = ("cycle_ms", "schedule_step_ms", "blocking_step_ms")
APPLIED = "applied_after_scheduling"
PROBLEM_EXAMPLES = 10
# Batch attributes that batch_log and op_timing repeat from pp_boundary, by their pp_boundary name.
BATCH_LOG_ATTRIBUTES = {
    "batch_size": "batch_size", "batch_num_prefill_tokens": "num_prefill_tokens",
    "batch_num_decode_tokens": "num_decode_tokens", "request_ids": "request_ids",
}
OP_TIMING_ATTRIBUTES = {
    "batch_size": "batch_size", "batch_num_prefill_tokens": "num_prefill_tokens",
    "batch_num_decode_tokens": "num_decode_tokens",
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean-run", type=Path, required=True, help="run directory without instrumentation")
    parser.add_argument("--op-run", type=Path, required=True, help="run directory with scope mode default")
    parser.add_argument("--kernel-run", type=Path, required=True, help="run directory with scope mode kernel_only")
    parser.add_argument(
        "--request-ids", type=Path, required=True, help="trace request_ids.json naming the warmup requests",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--clock-window-ms", type=float, default=50.0,
        help="largest delay from a scheduling engine iteration to a PP0 forward start of its dp_rank",
    )
    parser.add_argument(
        "--min-count", type=int, default=5, help="fewest batches or cycles on each side of a compared class",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path, inputs: dict) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        with path.open() as handle:
            frame = pd.DataFrame.from_records([json.loads(line) for line in handle])
    else:
        frame = pd.read_csv(path)
    inputs[str(path)] = {"sha256": sha256(path), "rows": len(frame)}
    return frame


def read_json(path: Path, inputs: dict) -> dict:
    inputs[str(path)] = {"sha256": sha256(path)}
    return json.loads(path.read_text())


@dataclass
class NativeRun:
    label: str
    iterations: pd.DataFrame
    placement: pd.DataFrame
    op_timing: pd.DataFrame | None = None
    batch_log: pd.DataFrame | None = None
    pp_boundary: pd.DataFrame | None = None
    dummy_passes: pd.DataFrame | None = None
    layers_per_stage: int | None = None


def stage_layers(model_config: dict, pp_boundary: pd.DataFrame) -> int:
    pp_sizes = pp_boundary.pp_world_size.unique()
    layers = model_config["num_hidden_layers"]
    if len(pp_sizes) != 1 or layers % pp_sizes[0]:
        raise ValueError(f"{layers} layers do not split evenly over pp_world_size {sorted(pp_sizes)}")
    return int(layers // pp_sizes[0])


def load_run(label: str, run_dir: Path, inputs: dict, timed: bool) -> NativeRun:
    run = NativeRun(
        label,
        read_table(run_dir / "extraction" / "engine_iterations.csv", inputs),
        read_table(run_dir / "extraction" / "placement.csv", inputs),
    )
    if timed:
        run.op_timing = read_table(run_dir / "run" / "op_timing.jsonl", inputs)
        run.batch_log = read_table(run_dir / "run" / "batch_log.jsonl", inputs)
        run.pp_boundary = read_table(run_dir / "run" / "pp_boundary.jsonl", inputs)
        run.dummy_passes = read_table(run_dir / "extraction" / "dummy_passes.csv", inputs)
        model_config = read_json(run_dir / "run" / "model" / "config.json", inputs)
        run.layers_per_stage = stage_layers(model_config, run.pp_boundary)
    return run


def engine_request_id(name: str) -> str:
    """Return the trace request id of the vLLM completions engine request `cmpl-<request_id>-0`."""
    if not (name.startswith("cmpl-") and name.endswith("-0")):
        raise ValueError(f"{name!r} is not an engine request name cmpl-<request_id>-0")
    return name[len("cmpl-"):-len("-0")]


def listed(request_ids) -> bool:
    return isinstance(request_ids, list)


def shape_class(batch_size: int, num_prefill_tokens: int, num_decode_tokens: int) -> str:
    """`decode_b<k>` for a batch of k decodes; else `prefill_t<u>`, u the power of two >= its prefill tokens."""
    if num_prefill_tokens == 0:
        return f"decode_b{batch_size}"
    bucket = 1 << (num_prefill_tokens - 1).bit_length()
    return f"prefill_t{bucket}" + ("_mixed" if num_decode_tokens > 0 else "")


def shape_rank(shape: str) -> tuple:
    kind, rest = shape.split("_", 1)
    return kind != "decode", int(rest[1:].split("_")[0]), rest.endswith("_mixed")


def sort_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.sort_values(
        columns, key=lambda column: column.map(shape_rank) if column.name == "shape_class" else column,
    ).reset_index(drop=True)


def key_problem(source: str, problem: str, keys: pd.DataFrame, rows: int | None = None) -> dict:
    unique = keys.drop_duplicates()
    entry = {"source": source, "problem": problem, "keys": len(unique)}
    if rows is not None:
        entry["rows"] = rows
    entry["examples"] = unique.head(PROBLEM_EXAMPLES).to_dict("records")
    return entry


def scope_totals(records: pd.DataFrame) -> pd.DataFrame:
    """Per batch key: summed non-collective and collective scope time and record counts."""
    collective = records.op_name.isin(COLLECTIVE_SCOPES)
    timed = records.assign(
        scope_ms=records.cuda_time_ms.where(~collective, 0.0),
        collective_ms=records.cuda_time_ms.where(collective, 0.0),
    )
    grouped = timed.groupby(BATCH_KEY)
    totals = grouped.agg(
        scope_sum_ms=("scope_ms", "sum"),
        collective_sum_ms=("collective_ms", "sum"),
        n_scope_records=("op_name", "size"),
        **{column: (column, "first") for column in OP_TIMING_ATTRIBUTES},
    )
    totals["op_record_counts"] = grouped.op_name.agg(lambda names: json.dumps(dict(sorted(Counter(names).items()))))
    return totals.reset_index()


def attribute_mismatches(boundary: pd.DataFrame, other: pd.DataFrame, source: str, attributes: dict[str, str]):
    """Yield (problem, keys) for each batch attribute of `other` that differs from pp_boundary."""
    renamed = other[BATCH_KEY + list(attributes)].rename(columns={name: f"{source}.{name}" for name in attributes})
    joined = boundary.merge(renamed, on=BATCH_KEY)
    comparable = lambda value: tuple(value) if listed(value) else value
    for name, boundary_name in attributes.items():
        differs = joined[boundary_name].map(comparable) != joined[f"{source}.{name}"].map(comparable)
        if differs.any():
            yield f"{name} differs from pp_boundary {boundary_name}", joined.loc[differs, BATCH_KEY]


def batch_windows(
    op_timing: pd.DataFrame, batch_log: pd.DataFrame, pp_boundary: pd.DataFrame, warmup_ids: set[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Join the three per-batch sources on (pp_rank, dp_rank, batch_id); return the windows and join problems.

    Batch attributes and request ids come from pp_boundary. A source whose key
    is repeated contributes nothing to that key's row; the row keeps the key and
    names the problem in `join_problems`.
    """
    problems: list[dict] = []
    notes: list[pd.DataFrame] = []

    def report(source: str, problem: str, keys: pd.DataFrame, rows: int | None = None) -> None:
        problems.append(key_problem(source, problem, keys[BATCH_KEY], rows))
        notes.append(keys[BATCH_KEY].drop_duplicates().assign(join_problem=f"{problem} in {source}"))

    unique = {}
    for source, frame in (("pp_boundary", pp_boundary), ("batch_log", batch_log)):
        repeated = frame.duplicated(BATCH_KEY, keep=False)
        if repeated.any():
            report(source, "duplicate key", frame[repeated], rows=int(repeated.sum()))
        unique[source] = frame[~repeated]
    # One op_timing key is one batch flush: one timestamp and each (op_name, scope_seq) once.
    flushes = op_timing.groupby(BATCH_KEY).timestamp.transform("nunique")
    repeated = (flushes > 1) | op_timing.duplicated(BATCH_KEY + ["op_name", "scope_seq"], keep=False)
    repeated_keys = op_timing.loc[repeated, BATCH_KEY].drop_duplicates()
    if len(repeated_keys):
        in_repeated = op_timing[BATCH_KEY].merge(repeated_keys, how="left", indicator=True)._merge.eq("both").to_numpy()
        report("op_timing", "duplicate key", op_timing[in_repeated], rows=int(in_repeated.sum()))
        records = op_timing[~in_repeated]
    else:
        records = op_timing
    unique["op_timing"] = scope_totals(records)

    boundary = unique["pp_boundary"]
    windows = boundary[BATCH_KEY + ["batch_size", "num_prefill_tokens", "num_decode_tokens", "request_ids",
                                    "forward_start_ts", "forward_end_ts"]].copy()
    windows["preprocess_ms"] = (boundary.preprocess_end_ts - boundary.preprocess_start_ts) * 1000.0
    windows["forward_ms"] = (boundary.forward_end_ts - boundary.forward_start_ts) * 1000.0
    windows["send_ms"] = (boundary.send_end_ts - boundary.send_start_ts) * 1000.0
    windows["recv_ms"] = (boundary.recv_end_ts - boundary.recv_start_ts) * 1000.0
    windows["execute_tail_ms"] = (boundary.execute_end_ts - boundary.forward_end_ts) * 1000.0
    for problem, keys in attribute_mismatches(windows, unique["batch_log"], "batch_log", BATCH_LOG_ATTRIBUTES):
        report("batch_log", problem, keys)
    for problem, keys in attribute_mismatches(windows, unique["op_timing"], "op_timing", OP_TIMING_ATTRIBUTES):
        report("op_timing", problem, keys)

    originals = {"pp_boundary": pp_boundary, "batch_log": batch_log, "op_timing": op_timing}
    keys = pd.concat([frame[BATCH_KEY] for frame in originals.values()]).drop_duplicates()
    for source, frame in originals.items():
        present = keys.merge(frame[BATCH_KEY].drop_duplicates(), how="left", indicator=True)._merge.eq("both")
        if not present.all():
            report(source, "missing key", keys[~present.to_numpy()])
    table = (
        keys.merge(windows, on=BATCH_KEY, how="left")
        .merge(unique["batch_log"][BATCH_KEY + ["batch_execution_time_ms"]], on=BATCH_KEY, how="left")
        .merge(unique["op_timing"].drop(columns=list(OP_TIMING_ATTRIBUTES)), on=BATCH_KEY, how="left")
    )
    table["uncovered_ms"] = table.forward_ms - table.scope_sum_ms - table.collective_sum_ms
    table["has_warmup_request"] = table.request_ids.map(
        lambda ids: any(engine_request_id(name) in warmup_ids for name in ids) if listed(ids) else None
    )
    table["shape_class"] = [
        shape_class(int(size), int(prefill), int(decode)) if listed(ids) else None
        for size, prefill, decode, ids in zip(
            table.batch_size, table.num_prefill_tokens, table.num_decode_tokens, table.request_ids,
        )
    ]
    if notes:
        labels = pd.concat(notes).groupby(BATCH_KEY).join_problem.agg(lambda names: "; ".join(sorted(set(names))))
        table = table.merge(labels.rename("join_problems").reset_index(), on=BATCH_KEY, how="left")
    else:
        table["join_problems"] = None
    table["join_problems"] = table.join_problems.fillna("")
    columns = [
        *BATCH_KEY, "batch_size", "num_prefill_tokens", "num_decode_tokens", "request_ids", "has_warmup_request",
        "shape_class", "forward_start_ts", "forward_end_ts", "preprocess_ms", "forward_ms", "send_ms", "recv_ms",
        "execute_tail_ms", "batch_execution_time_ms", "scope_sum_ms", "collective_sum_ms", "uncovered_ms",
        "n_scope_records", "op_record_counts", "join_problems",
    ]
    return table[columns].sort_values(BATCH_KEY).reset_index(drop=True), problems


def formal_batches(windows: pd.DataFrame) -> pd.DataFrame:
    return windows[windows.has_warmup_request.eq(False) & windows.join_problems.eq("")]


def spread(grouped, column: str, name: str) -> dict[str, pd.Series]:
    return {
        f"{name}_median": grouped[column].median(),
        f"{name}_p10": grouped[column].quantile(0.1),
        f"{name}_p90": grouped[column].quantile(0.9),
    }


def scope_batches(records: pd.DataFrame, formal: pd.DataFrame) -> pd.DataFrame:
    """Per formal batch and scope: the number of records and their summed time."""
    per_batch = records.groupby(BATCH_KEY + ["op_name"]).cuda_time_ms.agg(records="size", time_ms="sum").reset_index()
    return per_batch.merge(formal[BATCH_KEY + ["shape_class"]], on=BATCH_KEY)


def scope_by_shape(label: str, records: pd.DataFrame, formal: pd.DataFrame) -> pd.DataFrame:
    group = ["pp_rank", "shape_class", "op_name"]
    grouped = scope_batches(records, formal).groupby(group)
    per_record = records.merge(formal[BATCH_KEY + ["shape_class"]], on=BATCH_KEY).groupby(group).cuda_time_ms
    table = pd.DataFrame({
        "batches": grouped.size(),
        "records_per_batch_median": grouped.records.median(),
        **spread(grouped, "time_ms", "batch_time_ms"),
        "record_time_ms_median": per_record.median(),
    }).reset_index()
    return table.assign(run=label)[["run", *table.columns]]


def window_by_shape(label: str, formal: pd.DataFrame) -> pd.DataFrame:
    grouped = formal.groupby(["pp_rank", "shape_class"])
    parts = {name: series for part in WINDOW_PARTS for name, series in spread(grouped, part, part).items()}
    table = pd.DataFrame({"batches": grouped.size(), **parts}).reset_index()
    return table.assign(run=label)[["run", *table.columns]]


def host_gap_by_shape(op_batches: pd.DataFrame, kernel_batches: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Per scope present in both runs: the op-run median per-batch time less the kernel-run median."""
    group = ["pp_rank", "shape_class", "op_name"]
    sides = []
    for label, per_batch in (("op", op_batches), ("kernel", kernel_batches)):
        grouped = per_batch.groupby(group).time_ms
        sides.append(pd.DataFrame({
            f"{label}_batches": grouped.size(), f"{label}_batch_time_ms_median": grouped.median(),
        }))
    table = sides[0].join(sides[1], how="inner").reset_index()
    table = table[(table.op_batches >= min_count) & (table.kernel_batches >= min_count)].copy()
    table["op_minus_kernel_ms"] = table.op_batch_time_ms_median - table.kernel_batch_time_ms_median
    table.insert(3, "kind", np.where(
        table.op_name.isin(COLLECTIVE_SCOPES), "wait-inclusive collective", "host gap per scope",
    ))
    return table


def overlaps(starts: np.ndarray, ends: np.ndarray, other_starts: np.ndarray, other_ends: np.ndarray) -> np.ndarray:
    """Overlap of each interval with each other interval, in the intervals' unit."""
    return np.clip(
        np.minimum(ends[:, None], other_ends[None, :]) - np.maximum(starts[:, None], other_starts[None, :]), 0.0, None,
    )


def peer_states(windows: pd.DataFrame, dummy_passes: pd.DataFrame, dp_of_engine: dict[int, int]) -> pd.DataFrame:
    """Classify the peer (dp_rank 1-d, same pp_rank) over each formal batch's forward window.

    `batch`: a peer forward window overlaps; the peer batch with the largest
    overlap is recorded with that overlap as a fraction of this forward window.
    `dummy`: otherwise a DP dummy pass of the peer's engine overlaps; the summed
    overlap of those passes is recorded as the fraction. `idle`: neither.
    Peer windows are every batch window without a join problem, warmup included.
    """
    dp_ranks = sorted(windows.dp_rank.unique())
    if dp_ranks != [0, 1]:
        raise ValueError(f"the peer of dp_rank d is 1-d, which needs dp_ranks [0, 1]; found {dp_ranks}")
    engine_of_dp = {dp_rank: engine for engine, dp_rank in dp_of_engine.items()}
    valid = windows[windows.join_problems.eq("")]
    parts = []
    for (pp_rank, dp_rank), subject in formal_batches(windows).groupby(["pp_rank", "dp_rank"]):
        peers = valid[(valid.pp_rank == pp_rank) & (valid.dp_rank == 1 - dp_rank)]
        dummies = dummy_passes[dummy_passes.engine == engine_of_dp[1 - dp_rank]]
        start, end = subject.forward_start_ts.to_numpy(), subject.forward_end_ts.to_numpy()
        batch_overlap = overlaps(start, end, peers.forward_start_ts.to_numpy(), peers.forward_end_ts.to_numpy())
        dummy_overlap = overlaps(start, end, dummies.start_monotonic.to_numpy(), dummies.monotonic.to_numpy())
        largest = batch_overlap.argmax(axis=1)
        in_batch = (batch_overlap > 0).any(axis=1)
        in_dummy = ~in_batch & (dummy_overlap > 0).any(axis=1)
        state = subject[BATCH_KEY + ["shape_class", "forward_ms", "collective_sum_ms"]].copy()
        state["peer_state"] = np.where(in_batch, "batch", np.where(in_dummy, "dummy", "idle"))
        largest_peer = peers.iloc[largest].set_index(state.index)
        state["peer_batch_id"] = largest_peer.batch_id.where(in_batch).astype("Int64")
        state["peer_shape_class"] = largest_peer.shape_class.where(in_batch)
        overlap = np.where(in_batch, batch_overlap[np.arange(len(subject)), largest], dummy_overlap.sum(axis=1))
        state["overlap_fraction"] = overlap / (end - start)
        state["overlapping_count"] = np.where(
            in_batch, (batch_overlap > 0).sum(axis=1), (dummy_overlap > 0).sum(axis=1),
        )
        parts.append(state)
    return pd.concat(parts).sort_values(BATCH_KEY).reset_index(drop=True)


def peer_state_summary(label: str, states: pd.DataFrame) -> pd.DataFrame:
    grouped = states.groupby(["pp_rank", "shape_class", "peer_state"])
    table = pd.DataFrame({
        "batches": grouped.size(), "collective_sum_ms_median": grouped.collective_sum_ms.median(),
    }).reset_index()
    return table.assign(run=label)[["run", *table.columns]]


def dummy_pass_summary(label: str, dummy_passes: pd.DataFrame) -> pd.DataFrame:
    durations = dummy_passes.assign(duration_ms=(dummy_passes.monotonic - dummy_passes.start_monotonic) * 1000.0)
    grouped = durations.groupby("engine")
    table = pd.DataFrame({"count": grouped.size(), **spread(grouped, "duration_ms", "duration_ms")}).reset_index()
    return table.assign(run=label)[["run", *table.columns]]


def decode_cycles(iterations: pd.DataFrame) -> pd.DataFrame:
    """Split each engine's decode token cycles into the schedule step and the blocking step.

    A decode cycle is an iteration that schedules one token per running request
    and applies the previous batch (`applied_after_scheduling`, no new request),
    followed by an empty `applied_after_scheduling` iteration that blocks on the
    scheduled batch. `schedule_step_ms` ends at the cycle iteration and starts at
    the previous iteration; `blocking_step_ms` runs to the next iteration.
    `other_running` is the other engine's `running` at its latest iteration no
    later than the cycle iteration, 0 before its first.
    """
    engines = sorted(iterations.engine.unique())
    if len(engines) != 2:
        raise ValueError(f"the cycle split compares two engines; found {engines}")
    parts = []
    for engine, frame in iterations.groupby("engine"):
        frame = frame.sort_values("seq").reset_index(drop=True)
        step_ms = frame.monotonic.diff() * 1000.0
        after = frame.shift(-1)
        is_cycle = (
            (frame.branch == APPLIED) & (frame.num_scheduled_tokens > 0)
            & (frame.num_scheduled_tokens == frame.running) & (frame.scheduled_new_req_ids == "[]")
            & (after.branch == APPLIED) & (after.num_scheduled_tokens == 0)
            & (frame.index > 0)
        )
        cycles = pd.DataFrame({
            "engine": engine, "seq": frame.seq, "running": frame.running, "t_sched_end": frame.monotonic,
            "schedule_step_ms": step_ms, "blocking_step_ms": step_ms.shift(-1),
        })[is_cycle]
        cycles["cycle_ms"] = cycles.schedule_step_ms + cycles.blocking_step_ms
        other = iterations[iterations.engine != engine].sort_values("monotonic")
        latest = np.searchsorted(other.monotonic.to_numpy(), cycles.t_sched_end.to_numpy(), side="right") - 1
        cycles["other_running"] = np.where(latest >= 0, other.running.to_numpy()[np.maximum(latest, 0)], 0)
        parts.append(cycles)
    return pd.concat(parts).reset_index(drop=True)


def first_formal_admission(iterations: pd.DataFrame, warmup_ids: set[str]) -> float:
    """The monotonic time of the first iteration that admits a request outside the warmup set."""
    admitted = iterations.scheduled_new_req_ids.map(json.loads)
    formal = admitted.map(lambda names: any(engine_request_id(name) not in warmup_ids for name in names))
    return float(iterations.loc[formal, "monotonic"].min())


def measurement_cost(clean_cycles: pd.DataFrame, op_cycles: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Per (running, other_running) class with enough cycles on both sides: op median less clean median."""
    classes = ["running", "other_running"]
    clean_groups, op_groups = clean_cycles.groupby(classes), op_cycles.groupby(classes)
    rows = []
    for running, other_running in sorted(set(clean_groups.groups) & set(op_groups.groups)):
        clean = clean_groups.get_group((running, other_running))
        op = op_groups.get_group((running, other_running))
        if len(clean) < min_count or len(op) < min_count:
            continue
        for part in CYCLE_PARTS:
            rows.append({
                "running": running, "other_running": other_running, "metric": part,
                "clean_median": clean[part].median(), "clean_count": len(clean),
                "op_median": op[part].median(), "op_count": len(op),
                "delta": op[part].median() - clean[part].median(),
            })
    return pd.DataFrame(rows)


def scheduled_keys(iterations: pd.DataFrame, engine: int) -> list[tuple]:
    """The engine's iterations that schedule tokens, in seq order, as (new request ids, tokens, running)."""
    frame = iterations[(iterations.engine == engine) & (iterations.num_scheduled_tokens > 0)].sort_values("seq")
    return list(zip(frame.scheduled_new_req_ids, frame.num_scheduled_tokens.map(int), frame.running.map(int)))


def sequence_alignment(clean: list[tuple], op: list[tuple]) -> tuple[dict, list[tuple]]:
    """Summarize how the op sequence follows the clean one; also return the `difflib` opcodes."""
    prefix = 0
    while prefix < min(len(clean), len(op)) and clean[prefix] == op[prefix]:
        prefix += 1
    divergence = None
    if clean != op:
        divergence = {
            "index": prefix,
            "clean": list(clean[prefix]) if prefix < len(clean) else None,
            "op": list(op[prefix]) if prefix < len(op) else None,
        }
    opcodes = difflib.SequenceMatcher(None, clean, op, autojunk=False).get_opcodes()
    summary = {
        "clean_length": len(clean), "op_length": len(op), "matched_prefix": prefix, "first_divergence": divergence,
        "equal_block_positions": sum(end - start for tag, start, end, _, _ in opcodes if tag == "equal"),
    }
    return summary, opcodes


def schedule_alignment(clean_iterations: pd.DataFrame, op_iterations: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    summaries, rows = {}, []
    for engine in sorted(set(clean_iterations.engine) | set(op_iterations.engine)):
        clean, op = scheduled_keys(clean_iterations, engine), scheduled_keys(op_iterations, engine)
        summaries[int(engine)], opcodes = sequence_alignment(clean, op)
        for tag, clean_start, clean_end, op_start, op_end in opcodes:
            rows.append({
                "engine": engine, "tag": tag, "clean_start": clean_start, "clean_end": clean_end,
                "op_start": op_start, "op_end": op_end,
                "clean_first_key": json.dumps(list(clean[clean_start])) if clean_end > clean_start else "",
                "op_first_key": json.dumps(list(op[op_start])) if op_end > op_start else "",
            })
    return summaries, pd.DataFrame(rows)


def placement_comparison(clean: pd.DataFrame, other: pd.DataFrame, source: str) -> tuple[dict, list[dict]]:
    """Compare each request's engine between the clean placement and another run's placement."""
    problems = []
    unique = []
    for label, frame in (("clean", clean), (source, other)):
        repeated = frame.request_id.duplicated(keep=False)
        if repeated.any():
            problems.append(key_problem(f"{label} placement.csv", "duplicate request_id",
                                        frame.loc[repeated, ["request_id"]], rows=int(repeated.sum())))
        unique.append(frame.loc[~repeated, ["request_id", "engine"]])
    joined = unique[0].merge(unique[1], on="request_id", how="outer", suffixes=("_clean", "_other"), indicator=True)
    both = joined[joined._merge == "both"]
    differing = both[both.engine_clean != both.engine_other]
    for side, label in (("left_only", "clean"), ("right_only", source)):
        missing = joined.loc[joined._merge == side, ["request_id"]]
        if len(missing):
            problems.append(key_problem(f"{label} placement.csv", "request_id missing from the other run", missing))
    return {
        "equal": len(both) - len(differing),
        "different": len(differing),
        "differing": [
            {"request_id": request_id, "clean_engine": int(clean_engine), f"{source}_engine": int(other_engine)}
            for request_id, clean_engine, other_engine in zip(
                differing.request_id, differing.engine_clean, differing.engine_other,
            )
        ],
    }, problems


def engine_dp_mapping(placement: pd.DataFrame, windows: pd.DataFrame) -> dict:
    """Map each engine to a dp_rank through its placed requests and the dp_rank of the batches holding them."""
    held = windows.loc[windows.request_ids.map(listed), ["dp_rank", "request_ids"]].explode("request_ids")
    held = pd.DataFrame({
        "request_id": held.request_ids.map(engine_request_id), "dp_rank": held.dp_rank,
    }).drop_duplicates()
    joined = placement[["request_id", "engine"]].merge(held, on="request_id", how="outer", indicator=True)
    pairs = (
        joined[joined._merge == "both"].astype({"engine": int, "dp_rank": int})
        .groupby(["engine", "dp_rank"]).request_id.nunique().rename("requests").reset_index()
    )
    conflicts = []
    for column, other in (("engine", "dp_rank"), ("dp_rank", "engine")):
        counts = pairs.groupby(column)[other].nunique()
        for value in counts[counts > 1].index:
            partners = sorted(pairs.loc[pairs[column] == value, other].tolist())
            conflicts.append({column: int(value), f"{other}s": partners})
    split = held.groupby("request_id").dp_rank.nunique()
    if (split > 1).any():
        conflicts.append({"requests_in_several_dp_ranks": sorted(split[split > 1].index)})
    return {
        "mapping": {} if conflicts else dict(zip(pairs.engine.tolist(), pairs.dp_rank.tolist())),
        "requests_per_pair": pairs.to_dict("records"),
        "conflicts": conflicts,
        "placed_without_batches": sorted(joined.loc[joined._merge == "left_only", "request_id"]),
        "batched_without_placement": sorted(joined.loc[joined._merge == "right_only", "request_id"]),
    }


def clock_check(
    iterations: pd.DataFrame, windows: pd.DataFrame, dp_of_engine: dict[int, int], window_ms: float,
) -> dict:
    """Test that engine `monotonic` and pp_boundary `*_ts` share one clock.

    Each engine iteration that schedules tokens should be followed, within
    `window_ms`, by a PP0 forward start of a batch on the engine's dp_rank.
    """
    first_stage = windows[(windows.pp_rank == 0) & windows.forward_start_ts.notna()]
    engines = []
    for engine, frame in iterations[iterations.num_scheduled_tokens > 0].groupby("engine"):
        starts = np.sort(first_stage.loc[first_stage.dp_rank == dp_of_engine[engine], "forward_start_ts"].to_numpy())
        times = frame.monotonic.to_numpy()
        following = np.searchsorted(starts, times, side="left")
        found = following < len(starts)
        delay_ms = np.full(len(times), np.inf)
        delay_ms[found] = (starts[following[found]] - times[found]) * 1000.0
        within = int((delay_ms <= window_ms).sum())
        engines.append({
            "engine": int(engine), "dp_rank": dp_of_engine[engine], "iterations": len(times), "within_window": within,
            "fraction": within / len(times),
            "delay_ms_median": float(np.median(delay_ms[found])),
            "delay_ms_p90": float(np.quantile(delay_ms[found], 0.9)),
            "without_following_forward": int((~found).sum()),
        })
    iterations_total = sum(engine["iterations"] for engine in engines)
    within_total = sum(engine["within_window"] for engine in engines)
    return {"window_ms": window_ms, "iterations": iterations_total, "within_window": within_total,
            "fraction": within_total / iterations_total, "engines": engines}


def records_per_batch(records: pd.DataFrame, layers_per_stage: int) -> list[dict]:
    """Distribution of each scope's records per batch, flagged when not a multiple of the stage's layers."""
    per_batch = records.groupby(BATCH_KEY + ["op_name"]).size().rename("records").reset_index()
    table = per_batch.groupby(["pp_rank", "op_name", "records"]).size().rename("batches").reset_index()
    table["multiple_of_stage_layers"] = table.records % layers_per_stage == 0
    return table.to_dict("records")


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs: dict = {}
    warmup_ids = set(read_json(args.request_ids, inputs)["warmup_request_ids"])
    clean = load_run("clean", args.clean_run, inputs, timed=False)
    timed = {
        label: load_run(label, path, inputs, timed=True)
        for label, path in (("op", args.op_run), ("kernel", args.kernel_run))
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def write(name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(args.output_dir / name, index=False)
        written.append({"file": name, "rows": len(frame)})

    problems, windows, formal, mappings, clocks = [], {}, {}, {}, {}
    for label, run in timed.items():
        windows[label], run_problems = batch_windows(run.op_timing, run.batch_log, run.pp_boundary, warmup_ids)
        problems += [{"run": label, **problem} for problem in run_problems]
        formal[label] = formal_batches(windows[label])
        write(f"batch_windows_{label}.csv", windows[label].assign(
            request_ids=windows[label].request_ids.map(lambda ids: json.dumps(ids) if listed(ids) else "")))
        mappings[label] = engine_dp_mapping(run.placement, windows[label])
        if mappings[label]["conflicts"]:
            raise ValueError(f"{label} run: engine to dp_rank conflicts {mappings[label]['conflicts']}")
        clocks[label] = clock_check(run.iterations, windows[label], mappings[label]["mapping"], args.clock_window_ms)

    per_batch = {label: scope_batches(run.op_timing, formal[label]) for label, run in timed.items()}
    write("scope_by_shape.csv", sort_rows(pd.concat(
        [scope_by_shape(label, run.op_timing, formal[label]) for label, run in timed.items()]),
        ["run", "pp_rank", "shape_class", "op_name"]))
    write("window_by_shape.csv", sort_rows(pd.concat(
        [window_by_shape(label, formal[label]) for label in timed]), ["run", "pp_rank", "shape_class"]))
    write("host_gap_by_shape.csv", sort_rows(
        host_gap_by_shape(per_batch["op"], per_batch["kernel"], args.min_count), ["pp_rank", "shape_class", "op_name"]))

    summaries = []
    for label, run in timed.items():
        states = peer_states(windows[label], run.dummy_passes, mappings[label]["mapping"])
        write(f"peer_state_{label}.csv", states)
        summaries.append(peer_state_summary(label, states))
    write("peer_state_summary.csv", sort_rows(pd.concat(summaries), ["run", "pp_rank", "shape_class", "peer_state"]))
    write("dummy_pass_summary.csv", pd.concat(
        [dummy_pass_summary(label, run.dummy_passes) for label, run in timed.items()]))

    cycles, warmup_cycles = {}, {}
    for label, iterations in (("clean", clean.iterations), ("op", timed["op"].iterations)):
        frame = decode_cycles(iterations)
        formal_cycles = frame.t_sched_end >= first_formal_admission(iterations, warmup_ids)
        cycles[label], warmup_cycles[label] = frame[formal_cycles], int((~formal_cycles).sum())
        write(f"decode_cycles_{label}.csv", cycles[label])
    write("measurement_cost.csv", measurement_cost(cycles["clean"], cycles["op"], args.min_count))

    alignment, opcodes = schedule_alignment(clean.iterations, timed["op"].iterations)
    write("alignment.csv", opcodes)
    placement = {}
    for label, run in timed.items():
        placement[f"clean_vs_{label}"], placement_problems = placement_comparison(clean.placement, run.placement, label)
        problems += placement_problems

    summary = {
        "runs": {"clean": str(args.clean_run), "op": str(args.op_run), "kernel": str(args.kernel_run)},
        "inputs": inputs,
        "record_modes": {
            label: {column: sorted(run.op_timing[column].unique().tolist())
                    for column in ("scope_mode", "timing_mode", "aggregation_mode", "count")}
            for label, run in timed.items()
        },
        "layers_per_stage": {label: run.layers_per_stage for label, run in timed.items()},
        "batches": {
            label: {
                "windows": len(table), "with_warmup_request": int(table.has_warmup_request.eq(True).sum()),
                "with_join_problem": int(table.join_problems.ne("").sum()), "formal": len(formal[label]),
            }
            for label, table in windows.items()
        },
        "engine_dp_mapping": mappings,
        "clock_check": clocks,
        "problems": problems,
        "records_per_batch": {
            label: records_per_batch(run.op_timing, run.layers_per_stage) for label, run in timed.items()
        },
        "decode_cycles": {
            label: {"formal": len(frame), "before_first_formal_admission": warmup_cycles[label]}
            for label, frame in cycles.items()
        },
        "alignment": {"engines": alignment, "placement": placement},
        "written_files": written,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=native) + "\n")
    print(json.dumps({entry["file"]: entry["rows"] for entry in written}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

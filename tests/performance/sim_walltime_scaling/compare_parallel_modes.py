"""Summarize paired sequential/parallel Frontier wall-clock results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_WORKLOAD_FIELDS = (
    "schema_version",
    "git_sha",
    "runner_sha256",
    "python_executable",
    "host",
    "seed",
    "model",
    "total_gpus",
    "simulation_mode",
    "shape",
    "replicas_per_cluster",
    "requests",
    "qps",
    "prefill_tokens",
    "decode_tokens",
)
_TIMING_FIELDS = ("init_s", "sim_wallclock_s", "total_proc_s")
_IGNORED_MODE_FLAGS = frozenset(
    {"--enable_parallel_clusters", "--no-enable_parallel_clusters"}
)
_IGNORED_VALUE_OPTIONS = frozenset(
    {"--metrics_config_output_dir", "--metrics_config_run_id"}
)


def load_result_files(results_dir: Path) -> list[dict[str, Any]]:
    """Load only direct-child JSON result files from a benchmark directory."""

    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise ValueError(f"results directory does not exist: {results_dir}")
    records = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Result file must contain an object: {path}")
        records.append(payload)
    return records


def _workload_key(record: Mapping[str, Any]) -> str:
    missing = [field for field in _WORKLOAD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Result is missing workload fields: {missing}")
    payload = {field: record[field] for field in _WORKLOAD_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _positive_float(record: Mapping[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a positive number, got {value!r}")
    return value


def _normalized_runtime_command(record: Mapping[str, Any]) -> tuple[str, ...]:
    command = record.get("command")
    if not isinstance(command, list) or any(
        not isinstance(argument, str) for argument in command
    ):
        raise ValueError(f"command must be a list of strings, got {command!r}")

    normalized = []
    index = 0
    while index < len(command):
        argument = command[index]
        if argument in _IGNORED_MODE_FLAGS:
            index += 1
            continue
        if argument in _IGNORED_VALUE_OPTIONS:
            if index + 1 >= len(command):
                raise ValueError(f"command option is missing a value: {argument}")
            index += 2
            continue
        normalized.append(argument)
        index += 1
    return tuple(normalized)


def _validate_success(record: Mapping[str, Any]) -> None:
    if record.get("status") != "success":
        raise ValueError(
            f"comparison requires success results, got status={record.get('status')!r}"
        )
    requests = record.get("requests")
    expected = record.get("expected_requests")
    completed = record.get("completed_requests")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (requests, expected, completed)
    ):
        raise ValueError(
            "request counts must be positive integers: "
            f"requests={requests!r}, completed_requests={completed!r}, "
            f"expected_requests={expected!r}"
        )
    if expected != completed:
        raise ValueError(
            "incomplete result: "
            f"completed_requests={completed!r}, expected_requests={expected!r}"
        )
    if requests != expected:
        raise ValueError(
            "request counts must match: "
            f"requests={requests!r}, completed_requests={completed!r}, "
            f"expected_requests={expected!r}"
        )
    effective_parallel_mode = record.get("effective_parallel_mode")
    if not isinstance(effective_parallel_mode, bool):
        raise ValueError(
            "effective_parallel_mode must be a boolean, "
            f"got {effective_parallel_mode!r}"
        )
    expected_parallel_mode = record.get("mode") == "parallel"
    if effective_parallel_mode != expected_parallel_mode:
        raise ValueError(
            "effective parallel mode mismatch: "
            f"mode={record.get('mode')!r}, "
            f"effective_parallel_mode={effective_parallel_mode!r}"
        )
    for field in _TIMING_FIELDS:
        _positive_float(record, field)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "cv": statistics.stdev(values) / mean,
    }


def _timing_summary(
    sequential_values: Sequence[float],
    parallel_values: Sequence[float],
) -> dict[str, Any]:
    paired_speedups = [
        sequential / parallel
        for sequential, parallel in zip(
            sequential_values,
            parallel_values,
            strict=True,
        )
    ]
    paired_slowdowns_pct = [
        ((parallel / sequential) - 1.0) * 100.0
        for sequential, parallel in zip(
            sequential_values,
            parallel_values,
            strict=True,
        )
    ]
    paired_speedup_median = statistics.median(paired_speedups)
    return {
        "sequential": _distribution(sequential_values),
        "parallel": _distribution(parallel_values),
        "median_ratio_speedup": (
            statistics.median(sequential_values)
            / statistics.median(parallel_values)
        ),
        "paired_speedup_median": paired_speedup_median,
        "paired_speedup_min": min(paired_speedups),
        "paired_speedup_max": max(paired_speedups),
        "parallel_slowdown_pct_median": statistics.median(
            paired_slowdowns_pct
        ),
    }


def summarize_paired_results(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate paired runs and return robust wall-clock summaries.

    Pairing uses ``attempt_index`` as the repetition identifier. The caller is
    responsible for assigning that index to runs from the same paired execution;
    this analyzer does not infer dispatch adjacency from timestamps. Each
    repetition must contain exactly one successful sequential result and one
    successful parallel result for the same workload. Event counts must match so
    a timing ratio never compares different simulator work. The paired medians
    are the primary comparison; ``median_ratio_speedup`` is retained only as a
    secondary ratio-of-medians diagnostic. ``cv`` is a dimensionless fraction,
    not a percentage. Because speedup and slowdown are independently summarized
    after a nonlinear transform, their medians need not be reciprocal for an
    even number of repetitions.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_workload_key(record)].append(record)
    if not grouped:
        raise ValueError("comparison requires at least one workload")

    summaries = []
    for workload_key in sorted(grouped):
        workload_records = grouped[workload_key]
        repetitions: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for record in workload_records:
            repetition = record.get("attempt_index")
            if isinstance(repetition, bool) or not isinstance(repetition, int):
                raise ValueError(
                    f"attempt_index must be an integer, got {repetition!r}"
                )
            mode = record.get("mode")
            if mode not in {"sequential", "parallel"}:
                raise ValueError(f"unsupported comparison mode: {mode!r}")
            if mode in repetitions[repetition]:
                raise ValueError(
                    "exactly one sequential and one parallel result are required "
                    f"for repetition {repetition}"
                )
            repetitions[repetition][mode] = record

        sequential_by_metric = {field: [] for field in _TIMING_FIELDS}
        parallel_by_metric = {field: [] for field in _TIMING_FIELDS}
        event_counts = set()

        for repetition in sorted(repetitions):
            pair = repetitions[repetition]
            if set(pair) != {"sequential", "parallel"}:
                raise ValueError(
                    "exactly one sequential and one parallel result are required "
                    f"for repetition {repetition}"
                )
            sequential = pair["sequential"]
            parallel = pair["parallel"]
            _validate_success(sequential)
            _validate_success(parallel)
            sequential_command = _normalized_runtime_command(sequential)
            parallel_command = _normalized_runtime_command(parallel)
            if sequential_command != parallel_command:
                raise ValueError(
                    "runtime command mismatch after removing mode and output "
                    f"identity options: repetition={repetition}"
                )

            sequential_events = sequential.get("event_count")
            parallel_events = parallel.get("event_count")
            if sequential_events != parallel_events:
                raise ValueError(
                    "event_count mismatch: "
                    f"repetition={repetition}, sequential={sequential_events!r}, "
                    f"parallel={parallel_events!r}"
                )
            if (
                isinstance(sequential_events, bool)
                or not isinstance(sequential_events, int)
                or sequential_events <= 0
            ):
                raise ValueError(
                    f"event_count must be a positive integer, got {sequential_events!r}"
                )
            event_counts.add(sequential_events)

            for field in _TIMING_FIELDS:
                sequential_by_metric[field].append(
                    _positive_float(sequential, field)
                )
                parallel_by_metric[field].append(_positive_float(parallel, field))

        if len(repetitions) < 3:
            raise ValueError(
                "comparison requires at least 3 paired repetitions, "
                f"got {len(repetitions)}"
            )

        if len(event_counts) != 1:
            raise ValueError(
                f"event_count changed across repetitions: {sorted(event_counts)}"
            )

        metadata = json.loads(workload_key)
        summary = {
            **metadata,
            "repetitions": len(repetitions),
            "event_count": next(iter(event_counts)),
        }
        for field in _TIMING_FIELDS:
            summary[field] = _timing_summary(
                sequential_by_metric[field],
                parallel_by_metric[field],
            )
        summaries.append(summary)

    return summaries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize paired sequential/parallel wall-clock results."
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summaries = summarize_paired_results(load_result_files(args.results_dir))
    output = json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    if args.output_json is None:
        print(output, end="")
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

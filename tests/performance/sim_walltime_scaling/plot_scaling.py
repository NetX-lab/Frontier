"""Aggregate and plot Frontier wall-clock scaling results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from tests.performance.sim_walltime_scaling.run_case import CaseSpec
from tests.performance.sim_walltime_scaling.sweep import (
    DEFAULT_SCALES,
    TERMINAL_STATUSES,
    _validate_result,
)


MODEL_ORDER = {"dense": 0, "moe": 1}
TIMEOUT_LOWER_BOUND_S = 14_400.0
SHAPE_FIELDS = ("attn_tp", "attn_dp", "moe_tp", "moe_ep", "pp")
CSV_FIELDS = (
    "schema_version",
    "model",
    "model_name",
    "total_gpus",
    "status",
    "mode",
    "effective_parallel_mode",
    "simulation_mode",
    "host",
    "worker_job_id",
    "shape",
    *SHAPE_FIELDS,
    "replicas_per_cluster",
    "requests",
    "qps",
    "prefill_tokens",
    "decode_tokens",
    "sim_wallclock_s",
    "init_s",
    "total_proc_s",
    "peak_rss_mb",
    "expected_requests",
    "completed_requests",
    "event_count",
    "events_per_s",
    "git_sha",
    "runner_sha256",
    "python_executable",
    "seed",
    "case_id",
    "attempt_id",
    "attempt_index",
    "case_fingerprint",
    "command",
    "started_at",
    "completed_at",
    "exit_code",
    "signal",
    "failure_reason",
    "oom_evidence",
    "notes",
    "stderr_tail",
    "stdout_tail",
)
STATUS_STYLES = {
    "simulated-oom": {"color": "#d97706", "marker": "s"},
    "host-oom": {"color": "#dc2626", "marker": "X"},
    "bug": {"color": "#7c3aed", "marker": "P"},
}


def _case_from_result(row: dict[str, Any]) -> CaseSpec:
    payload = {
        "model": row.get("model"),
        "total_gpus": row.get("total_gpus"),
        "mode": row.get("mode"),
        "attempt_index": row.get("attempt_index"),
        "shape": row.get("shape"),
        "num_requests": row.get("requests"),
        "qps": row.get("qps"),
        "prefill_tokens": row.get("prefill_tokens"),
        "decode_tokens": row.get("decode_tokens"),
        "seed": row.get("seed"),
        "simulation_mode": row.get("simulation_mode"),
    }
    try:
        return CaseSpec.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Result cannot reconstruct its CaseSpec: {exc}") from exc


def _attempts_directory(results_dir: Path) -> Path:
    nested = results_dir / "results"
    return nested if nested.is_dir() else results_dir


def _row_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        MODEL_ORDER[row["model"]],
        int(row["total_gpus"]),
        int(row["attempt_index"]),
    )


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load and strictly validate immutable attempt JSONs."""

    results_dir = Path(results_dir)
    attempts_dir = _attempts_directory(results_dir)
    if not attempts_dir.is_dir():
        raise FileNotFoundError(f"Attempt results directory does not exist: {attempts_dir}")

    rows: list[dict[str, Any]] = []
    seen_attempt_ids: set[str] = set()
    for path in sorted(attempts_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Result JSON must contain an object: {path}")
        attempt_id = payload.get("attempt_id")
        if attempt_id in seen_attempt_ids:
            raise ValueError(f"duplicate attempt_id {attempt_id!r} in {attempts_dir}")
        case = _case_from_result(payload)
        row = _validate_result(path, case)
        attempt_id = row["attempt_id"]
        seen_attempt_ids.add(attempt_id)
        rows.append(row)

    return sorted(rows, key=_row_sort_key)


def select_summary_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the first success, or the final terminal attempt, per point."""

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        status = row.get("status")
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Cannot summarize non-terminal status: {status!r}")
        groups[(row["model"], int(row["total_gpus"]))].append(row)

    selected: list[dict[str, Any]] = []
    for attempts in groups.values():
        ordered = sorted(attempts, key=lambda row: int(row["attempt_index"]))
        selected.append(next((row for row in ordered if row["status"] == "success"), ordered[-1]))
    return sorted(selected, key=_row_sort_key)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    shape = row.get("shape")
    if not isinstance(shape, dict):
        raise ValueError(f"Result shape must be an object: {shape!r}")
    flattened = dict(row)
    for field_name in SHAPE_FIELDS:
        flattened[field_name] = shape.get(field_name)
    return {field_name: _csv_value(flattened.get(field_name)) for field_name in CSV_FIELDS}


def write_summary_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write attempt or selected-summary rows with deterministic nested values."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))


def validate_formal_completeness(rows: Iterable[dict[str, Any]]) -> None:
    """Require at least one terminal attempt for every formal model/scale point."""

    observed = {(row.get("model"), row.get("total_gpus")) for row in rows}
    expected = {
        (model, total_gpus)
        for model in MODEL_ORDER
        for total_gpus in DEFAULT_SCALES
    }
    missing = sorted(expected - observed, key=lambda value: (MODEL_ORDER[value[0]], value[1]))
    extra = sorted(observed - expected, key=lambda value: (str(value[0]), str(value[1])))
    if missing or extra:
        raise ValueError(
            "Formal result completeness failure: "
            f"missing={missing}, extra={extra}"
        )


def _positive_wallclock(row: dict[str, Any]) -> float:
    value = row.get("sim_wallclock_s")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"Invalid success sim_wallclock_s: {value!r}")
    return float(value)


def _format_seconds(value: float) -> str:
    return f"{value:g} s"


def _deduplicate_legend(axis: Any) -> None:
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="best", fontsize=8)


def plot_scaling(rows: Iterable[dict[str, Any]], png_path: Path, pdf_path: Path) -> Figure:
    """Render dense and MoE wall-clock panels without fabricating failure times."""

    all_rows = list(rows)
    selected = select_summary_rows(all_rows)
    failure_rows = [row for row in all_rows if row["status"] != "success"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    model_metadata = {
        "dense": (axes[0], "Dense: llama3.3-70b", "#2563eb"),
        "moe": (axes[1], "MoE: Qwen3-235B-A22B", "#059669"),
    }

    for model, (axis, title, success_color) in model_metadata.items():
        model_rows = sorted(
            (row for row in selected if row["model"] == model),
            key=lambda row: int(row["total_gpus"]),
        )
        success_rows = [row for row in model_rows if row["status"] == "success"]
        success_x = [int(row["total_gpus"]) for row in success_rows]
        success_y = [_positive_wallclock(row) for row in success_rows]
        if success_rows:
            axis.plot(
                success_x,
                success_y,
                color=success_color,
                marker="o",
                linewidth=1.8,
                label="sequential success",
            )
            for x_value, y_value in zip(success_x, success_y):
                axis.annotate(
                    _format_seconds(y_value),
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                )

        for row in sorted(
            (row for row in failure_rows if row["model"] == model),
            key=lambda row: int(row["total_gpus"]),
        ):
            x_value = int(row["total_gpus"])
            status = row["status"]
            if status == "timeout":
                axis.plot(
                    [x_value],
                    [TIMEOUT_LOWER_BOUND_S],
                    color="#111827",
                    marker="^",
                    linestyle="None",
                    markersize=8,
                    label="timeout lower bound",
                )
                axis.annotate(
                    ">= 14400 s",
                    (x_value, TIMEOUT_LOWER_BOUND_S),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                )
            elif status in STATUS_STYLES:
                style = STATUS_STYLES[status]
                axis.plot(
                    [x_value],
                    [0.04],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="None",
                    markersize=8,
                    transform=axis.get_xaxis_transform(),
                    clip_on=False,
                    label=status,
                )
                axis.text(
                    x_value,
                    0.085,
                    status,
                    color=style["color"],
                    fontsize=7,
                    ha="center",
                    rotation=25,
                    transform=axis.get_xaxis_transform(),
                )

        axis.plot(
            [DEFAULT_SCALES[0], DEFAULT_SCALES[-1]],
            [0.015, 0.015],
            color="#9ca3af",
            linestyle=":",
            linewidth=0.8,
            transform=axis.get_xaxis_transform(),
            clip_on=False,
            label="status band",
        )
        modes = sorted({str(row["mode"]) for row in model_rows}) or ["sequential"]
        axis.text(
            0.02,
            0.96,
            f"Mode: {', '.join(modes)}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(DEFAULT_SCALES)
        axis.get_xaxis().set_major_formatter(ScalarFormatter())
        axis.set_xlabel("Total simulated GPUs")
        axis.grid(True, which="both", alpha=0.25)
        _deduplicate_legend(axis)

    axes[0].set_ylabel("Simulator.run() wall-clock time (seconds)")
    figure.suptitle("Frontier PDD wall-clock weak scaling (sequential mode)")
    figure.subplots_adjust(bottom=0.2, top=0.86, wspace=0.08)

    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    return figure


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate Frontier wall-clock attempts and render scaling plots."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = load_results(args.results_dir)
    validate_formal_completeness(rows)
    selected = select_summary_rows(rows)
    write_summary_csv(rows, args.results_dir / "attempts.csv")
    write_summary_csv(selected, args.results_dir / "summary.csv")
    figure = plot_scaling(
        rows,
        args.results_dir / "sim_walltime_scaling.png",
        args.results_dir / "sim_walltime_scaling.pdf",
    )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

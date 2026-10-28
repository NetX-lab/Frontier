"""Run the P1h V4 graph bridge through the accepted bootstrap and report on CPU.

Each graph mode receives the same workload flags and an unchanged copy of the
bootstrap identity fields. Saved captures and traces let the report repeat clone
attribution without a GPU. A passing bridge requires successful runs, complete
trace attribution, and observed FULL and PIECEWISE dispatch steps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from frontier.profiling.attention.dsa_kernel_trace import load_trace_events
from frontier.profiling.attention.dsa_plan import GRAPH_REPLAY_CUDAGRAPH_MODES, ENGINE_POLICY_GRAPH_REPLAY
from frontier.profiling.attention.dsv4_graph_replay_runner import (
    BREAKABLE_CUDAGRAPH_ENV,
    CONTAINER_SCOPES,
    CUDAGRAPH_DISPATCH_KEY,
    V2_MODEL_RUNNER_ENV,
)
from frontier.profiling.attention.graph_capture import CaptureRecord
from frontier.profiling.attention.graph_replay_trace import attribute_replay_trace

_NONE_MODE = GRAPH_REPLAY_CUDAGRAPH_MODES[-1]
_FULL_DISPATCH_MODE = "FULL"
_PIECEWISE_DISPATCH_MODE = "PIECEWISE"
_REQUIRED_DISPATCH_MODES = (_FULL_DISPATCH_MODE, _PIECEWISE_DISPATCH_MODE)
_ENV_ENABLED = "1"
_PASS = "PASS"
_FAIL = "FAIL"
_RUN_FILE = "bridge_run.json"
_REPORT_FILE = "bridge_report.json"
_BOOTSTRAP_FILE = "bootstrap_config.json"
_GRAPH_INTERVAL_KIND = "graph"
_EAGER_INTERVAL_KIND = "eager"
_KERNEL_ACTIVITY = "kernel"
_ARTIFACT_ERRORS = (OSError, ValueError, RuntimeError, KeyError, TypeError)


@dataclass(frozen=True)
class BridgeConfig:
    bootstrap: dict[str, Any]
    plan_args: tuple[str, ...]
    run_args: tuple[str, ...]
    engine_index: int
    case_ids: tuple[str, ...]
    modes: tuple[str, ...]
    cudagraph_capture_sizes: tuple[int, ...]
    output_root: Path


def load_config(path: Path) -> BridgeConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    modes = tuple(payload["modes"])
    if not modes or len(set(modes)) != len(modes) or any(
        mode not in GRAPH_REPLAY_CUDAGRAPH_MODES for mode in modes
    ):
        raise ValueError(f"Bridge modes must be a non-empty ordered subset of {GRAPH_REPLAY_CUDAGRAPH_MODES}, got {modes}")
    return BridgeConfig(
        bootstrap=payload["bootstrap"],
        plan_args=tuple(payload["plan_args"]), run_args=tuple(payload["run_args"]),
        engine_index=payload["engine_index"], case_ids=tuple(payload["case_ids"]), modes=modes,
        cudagraph_capture_sizes=tuple(payload["cudagraph_capture_sizes"]),
        output_root=Path(payload["output_root"]).resolve(),
    )


def profile_args(config: BridgeConfig, mode: str) -> list[str]:
    """Build one bootstrap invocation, retaining the shared workload flags."""

    mode_root = config.output_root / mode
    return [
        "run", *config.plan_args, *graph_mode_flags(config, mode),
        "--engine_index", str(config.engine_index), *config.run_args,
        "--raw_dir", str(mode_root / "raw"), "--trace_dir", str(mode_root / "traces"),
        "--keep_traces", "--case_ids", *config.case_ids,
    ]


def graph_mode_flags(config: BridgeConfig, mode: str) -> list[str]:
    """Flags shared by profiling and CPU plan reconstruction for one graph mode."""
    capture_flags = [] if mode == _NONE_MODE else [
        "--cudagraph_capture_sizes", *(str(size) for size in config.cudagraph_capture_sizes),
    ]
    return [
        "--engine_policy", ENGINE_POLICY_GRAPH_REPLAY, "--cudagraph_mode", mode,
        *capture_flags,
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_bridge(config: BridgeConfig) -> int:
    """Run every mode, saving failures while allowing the next mode to run."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, **{
        BREAKABLE_CUDAGRAPH_ENV: _ENV_ENABLED, V2_MODEL_RUNNER_ENV: _ENV_ENABLED,
    })
    runs = {}
    for mode in config.modes:
        mode_root = config.output_root / mode
        mode_root.mkdir(parents=True, exist_ok=True)
        bootstrap_path = mode_root / _BOOTSTRAP_FILE
        _write_json(bootstrap_path, dict(config.bootstrap, profile_args=profile_args(config, mode)))
        command = [sys.executable, config.bootstrap["script"], "--config", str(bootstrap_path)]
        start = datetime.now(timezone.utc).isoformat()
        tick = time.monotonic()
        exit_code = None
        error = None
        with (mode_root / "run.log").open("w", encoding="utf-8") as log:
            try:
                exit_code = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            except OSError as failure:
                error = str(failure)
                log.write(error + "\n")
        runs[mode] = {
            "exit_code": exit_code, "start": start,
            "end": datetime.now(timezone.utc).isoformat(), "duration_seconds": time.monotonic() - tick,
            "error": error,
        }
        _write_json(config.output_root / _RUN_FILE, {"modes": runs})
        print(f"[dsv4_bridge] {mode}: exit {exit_code}", flush=True)
    return int(any(run["exit_code"] != 0 for run in runs.values()))


def report_bridge(config: BridgeConfig) -> int:
    """Repeat saved attribution and write an explicit bridge verdict on CPU."""

    reasons = []
    try:
        runs = json.loads((config.output_root / _RUN_FILE).read_text(encoding="utf-8"))["modes"]
    except _ARTIFACT_ERRORS as error:
        runs = {}
        reasons.append(f"Run record: {error}")
    reports = {}
    dispatch_counts = Counter()
    for mode in config.modes:
        mode_root = config.output_root / mode
        exit_code = runs.get(mode, {}).get("exit_code")
        if exit_code != 0:
            reasons.append(f"{mode}: run exit code {exit_code!r}")
        record = None
        capture_summary = None
        artifact_errors = []
        try:
            capture = json.loads((mode_root / "raw" / f"dsv4_capture_e{config.engine_index}.json").read_text(encoding="utf-8"))
            record = CaptureRecord.from_json(capture["record"])
            capture_summary = capture["summary"]
        except _ARTIFACT_ERRORS as error:
            artifact_errors.append(f"Capture: {error}")
        mode_counts = Counter()
        try:
            raw_path = mode_root / "raw" / f"dsv4_raw_e{config.engine_index}.jsonl"
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                step = json.loads(line)
                if step["kind"] == "step":
                    mode_counts[step[CUDAGRAPH_DISPATCH_KEY]["cg_mode"]] += 1
        except _ARTIFACT_ERRORS as error:
            artifact_errors.append(f"Raw records: {error}")
        dispatch_counts.update(mode_counts)
        traces = []
        trace_paths = sorted((mode_root / "traces").glob("*.json"))
        if not trace_paths:
            artifact_errors.append("No replay traces")
        for path in trace_paths:
            trace = {
                "path": str(path), "graph_kernel_interval_count": None,
                "eager_kernel_interval_count": None, "outside_kernel_count": None,
                "unattributed_launch_count": None, "error": None,
            }
            try:
                if record is None:
                    raise ValueError("Capture record is unavailable")
                attributed = attribute_replay_trace(load_trace_events(path), record, CONTAINER_SCOPES)
                counts = Counter(row.kind for row in attributed.intervals if row.activity == _KERNEL_ACTIVITY)
                trace.update({
                    "graph_kernel_interval_count": counts[_GRAPH_INTERVAL_KIND],
                    "eager_kernel_interval_count": counts[_EAGER_INTERVAL_KIND],
                    "outside_kernel_count": sum(attributed.outside_kernel_counts.values()),
                    "unattributed_launch_count": attributed.unattributed_launch_count,
                })
            except _ARTIFACT_ERRORS as error:
                trace["error"] = str(error)
                reasons.append(f"{mode}: attribution failed for {path.name}: {error}")
            traces.append(trace)
        reasons.extend(f"{mode}: {error}" for error in artifact_errors)
        reports[mode] = {
            "run_exit_code": exit_code, "capture_summary": capture_summary,
            "cg_mode_counts": dict(sorted(mode_counts.items())), "traces": traces,
            "artifact_errors": artifact_errors,
        }
    for mode in _REQUIRED_DISPATCH_MODES:
        if dispatch_counts[mode] == 0:
            reasons.append(f"No {mode} steps observed")
    verdict = _FAIL if reasons else _PASS
    report = {"verdict": verdict, "reasons": reasons, "modes": reports}
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_root / _REPORT_FILE, report)
    print(f"[dsv4_bridge] {verdict}: {len(reports)} modes, FULL={dispatch_counts[_FULL_DISPATCH_MODE]}, PIECEWISE={dispatch_counts[_PIECEWISE_DISPATCH_MODE]}")
    for reason in reasons:
        print(f"[dsv4_bridge] {reason}")
    return int(verdict == _FAIL)


def main(argv: list[str] | None = None) -> int:
    # The accepted bootstrap invokes the driver with exactly ``--config PATH``.
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", action="store_true", help="Report saved artifacts instead of running.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return report_bridge(config) if args.report else run_bridge(config)


__all__ = ["BridgeConfig", "graph_mode_flags", "load_config", "main", "profile_args", "report_bridge", "run_bridge"]

if __name__ == "__main__":
    raise SystemExit(main())

"""Run the P1h V4 graph bridge through the accepted bootstrap and report on CPU.

Each (mode, engine) run receives its planned cases and an unchanged copy of the
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
    BREAKABLE_CUDAGRAPH_ENV, CONTAINER_SCOPES, CUDAGRAPH_DISPATCH_KEY, V2_MODEL_RUNNER_ENV,
)
from frontier.profiling.attention.graph_capture import CaptureRecord
from frontier.profiling.attention.graph_replay_trace import HOLD_ACTIVITY, attribute_replay_trace

_NONE_MODE = GRAPH_REPLAY_CUDAGRAPH_MODES[-1]
_FULL_DISPATCH_MODE, _PIECEWISE_DISPATCH_MODE = "FULL", "PIECEWISE"
_REQUIRED_DISPATCH_MODES = (_FULL_DISPATCH_MODE, _PIECEWISE_DISPATCH_MODE)
_ENV_ENABLED = "1"
_PASS, _FAIL = "PASS", "FAIL"
_RUN_FILE, _REPORT_FILE = "bridge_run.json", "bridge_report.json"
_GRAPH_INTERVAL_KIND, _EAGER_INTERVAL_KIND, _KERNEL_ACTIVITY = "graph", "eager", "kernel"
_RUNS_KEY = "runs"
_MODE_KEY = 'mode'
_ENGINE_INDEX_KEY = 'engine_index'
_CASE_IDS_KEY = 'case_ids'
_HOLD_INTERVAL_COUNT_KEY = 'hold_interval_count'

_OLD_KEYS = frozenset({"modes", _ENGINE_INDEX_KEY, _CASE_IDS_KEY})
_RUN_KEYS = frozenset({_MODE_KEY, _ENGINE_INDEX_KEY, _CASE_IDS_KEY})
_ARTIFACT_ERRORS = (OSError, ValueError, RuntimeError, KeyError, TypeError)


@dataclass(frozen=True)
class BridgeRun:
    mode: str
    engine_index: int
    case_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.mode}/e{self.engine_index}"


@dataclass(frozen=True)
class BridgeConfig:
    bootstrap: dict[str, Any]
    plan_args: tuple[str, ...]
    run_args: tuple[str, ...]
    runs: tuple[BridgeRun, ...]
    cudagraph_capture_sizes: tuple[int, ...]
    output_root: Path


def load_config(path: Path) -> BridgeConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    old = _OLD_KEYS & payload.keys()
    if old:
        raise ValueError(f"Bridge config rejects old top-level keys {sorted(old)}; use runs with mode, engine_index and case_ids")
    entries = payload[_RUNS_KEY]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Bridge runs must be a non-empty JSON list")
    runs, pairs = [], set()
    for row in entries:
        if not isinstance(row, dict) or row.keys() != _RUN_KEYS:
            raise ValueError(f"Bridge run requires exactly {sorted(_RUN_KEYS)}, got {row!r}")
        mode, engine, cases = row[_MODE_KEY], row[_ENGINE_INDEX_KEY], row[_CASE_IDS_KEY]
        if mode not in GRAPH_REPLAY_CUDAGRAPH_MODES:
            raise ValueError(f"Bridge run mode must be in {GRAPH_REPLAY_CUDAGRAPH_MODES}, got {mode!r}")
        if type(engine) is not int or engine < 0:
            raise ValueError(f"Bridge run engine_index must be a non-negative int, got {engine!r}")
        if (not isinstance(cases, list) or not cases
                or any(not isinstance(case, str) or not case for case in cases)
                or len(set(cases)) != len(cases)):
            raise ValueError("Bridge run case_ids must be a non-empty list of unique non-empty strings")
        if (mode, engine) in pairs:
            raise ValueError(f"Duplicate bridge run pair {(mode, engine)!r}")
        pairs.add((mode, engine))
        runs.append(BridgeRun(mode, engine, tuple(cases)))
    return BridgeConfig(payload["bootstrap"], tuple(payload["plan_args"]), tuple(payload["run_args"]),
                        tuple(runs), tuple(payload["cudagraph_capture_sizes"]),
                        Path(payload["output_root"]).resolve())


def graph_mode_flags(mode: str, capture_sizes: tuple[int, ...]) -> list[str]:
    """Flags shared by profiling and CPU plan reconstruction for one graph mode."""
    capture_flags = [] if mode == _NONE_MODE else [
        "--cudagraph_capture_sizes", *(str(size) for size in capture_sizes),
    ]
    return ["--engine_policy", ENGINE_POLICY_GRAPH_REPLAY, "--cudagraph_mode", mode, *capture_flags]


def profile_args(config: BridgeConfig, run: BridgeRun) -> list[str]:
    """Build one bootstrap invocation, retaining the shared workload flags."""
    root = config.output_root / run.mode
    return ["run", *config.plan_args, *graph_mode_flags(run.mode, config.cudagraph_capture_sizes),
            "--engine_index", str(run.engine_index), *config.run_args,
            "--raw_dir", str(root / "raw"), "--trace_dir", str(root / "traces"),
            "--keep_traces", "--case_ids", *run.case_ids]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_bridge(config: BridgeConfig) -> int:
    """Run every engine, saving failures while allowing the next run to start."""
    config.output_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, **{BREAKABLE_CUDAGRAPH_ENV: _ENV_ENABLED, V2_MODEL_RUNNER_ENV: _ENV_ENABLED})
    results = {}
    for run in config.runs:
        root = config.output_root / run.mode
        root.mkdir(parents=True, exist_ok=True)
        bootstrap_path = root / f"bootstrap_config_e{run.engine_index}.json"
        _write_json(bootstrap_path, dict(config.bootstrap, profile_args=profile_args(config, run)))
        command = [sys.executable, config.bootstrap["script"], "--config", str(bootstrap_path)]
        start, tick = datetime.now(timezone.utc).isoformat(), time.monotonic()
        exit_code, error = None, None
        with (root / f"run_e{run.engine_index}.log").open("w", encoding="utf-8") as log:
            try:
                exit_code = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            except OSError as failure:
                error = str(failure)
                log.write(error + "\n")
        results[run.key] = {"exit_code": exit_code, "start": start,
                            "end": datetime.now(timezone.utc).isoformat(),
                            "duration_seconds": time.monotonic() - tick, "error": error}
        _write_json(config.output_root / _RUN_FILE, {_RUNS_KEY: results})
        print(f"[dsv4_bridge] {run.key}: exit {exit_code}", flush=True)
    return int(any(result["exit_code"] != 0 for result in results.values()))


def report_bridge(config: BridgeConfig) -> int:
    """Repeat each run's listed traces with that engine's capture record."""
    reasons = []
    try:
        results = json.loads((config.output_root / _RUN_FILE).read_text(encoding="utf-8"))[_RUNS_KEY]
    except _ARTIFACT_ERRORS as error:
        results = {}
        reasons.append(f"Run record: {error}")
    reports, dispatch_counts = {}, Counter()
    for run in config.runs:
        root = config.output_root / run.mode
        exit_code = results.get(run.key, {}).get("exit_code")
        if exit_code != 0:
            reasons.append(f"{run.key}: run exit code {exit_code!r}")
        record, capture_summary = None, None
        artifact_errors = []
        try:
            capture = json.loads((root / "raw" / f"dsv4_capture_e{run.engine_index}.json").read_text(encoding="utf-8"))
            record, capture_summary = CaptureRecord.from_json(capture["record"]), capture["summary"]
        except _ARTIFACT_ERRORS as error:
            artifact_errors.append(f"Capture: {error}")
        counts = Counter()
        try:
            path = root / "raw" / f"dsv4_raw_e{run.engine_index}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["kind"] == "step" and row["case_id"] in run.case_ids:
                    counts[row[CUDAGRAPH_DISPATCH_KEY]["cg_mode"]] += 1
        except _ARTIFACT_ERRORS as error:
            artifact_errors.append(f"Raw records: {error}")
        dispatch_counts.update(counts)
        traces = []
        for case_id in run.case_ids:
            path = root / "traces" / f"{case_id}.json"
            trace = {"path": str(path), "graph_kernel_interval_count": None,
                     "eager_kernel_interval_count": None, _HOLD_INTERVAL_COUNT_KEY: None,
                     "outside_kernel_count": None, "unattributed_launch_count": None, "error": None}
            try:
                if record is None:
                    raise ValueError("Capture record is unavailable")
                attributed = attribute_replay_trace(load_trace_events(path), record, CONTAINER_SCOPES)
                kinds = Counter(row.kind for row in attributed.intervals if row.activity == _KERNEL_ACTIVITY)
                trace.update(graph_kernel_interval_count=kinds[_GRAPH_INTERVAL_KIND],
                             eager_kernel_interval_count=kinds[_EAGER_INTERVAL_KIND],
                             hold_interval_count=sum(row.activity == HOLD_ACTIVITY for row in attributed.intervals),
                             outside_kernel_count=sum(attributed.outside_kernel_counts.values()),
                             unattributed_launch_count=attributed.unattributed_launch_count)
            except _ARTIFACT_ERRORS as error:
                trace["error"] = str(error)
                reasons.append(f"{run.key}: attribution failed for {path.name}: {error}")
            traces.append(trace)
        reasons.extend(f"{run.key}: {error}" for error in artifact_errors)
        reports[run.key] = {"run_exit_code": exit_code, "capture_summary": capture_summary,
                            "cg_mode_counts": dict(sorted(counts.items())), "traces": traces,
                            "artifact_errors": artifact_errors}
    for mode in _REQUIRED_DISPATCH_MODES:
        if dispatch_counts[mode] == 0:
            reasons.append(f"No {mode} steps observed")
    verdict = _FAIL if reasons else _PASS
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_root / _REPORT_FILE, {"verdict": verdict, "reasons": reasons, _RUNS_KEY: reports})
    print(f"[dsv4_bridge] {verdict}: {len(reports)} runs, FULL={dispatch_counts[_FULL_DISPATCH_MODE]}, PIECEWISE={dispatch_counts[_PIECEWISE_DISPATCH_MODE]}")
    for reason in reasons:
        print(f"[dsv4_bridge] {reason}")
    return int(verdict == _FAIL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", action="store_true", help="Report saved artifacts instead of running.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return report_bridge(config) if args.report else run_bridge(config)


__all__ = ["BridgeConfig", "BridgeRun", "graph_mode_flags", "load_config", "main", "profile_args", "report_bridge", "run_bridge"]

if __name__ == "__main__":
    raise SystemExit(main())

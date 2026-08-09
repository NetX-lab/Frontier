"""Run immutable Frontier wall-clock scaling attempts with bounded concurrency."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.performance.sim_walltime_scaling.run_case import (
    PRIMARY_SHAPES,
    REQUIRED_RESULT_FIELDS,
    SCHEMA_VERSION,
    CaseSpec,
    ParallelShape,
    write_json_atomic,
)


DEFAULT_SCALES = (32, 64, 128, 256, 512, 1024, 4096)
DEFAULT_TIMEOUT_S = 14_400.0
OUTPUT_TAIL_BYTES = 20_000
TERMINATION_GRACE_S = 5.0
DEFAULT_TEMP_ROOT = Path("/data/ycfeng/tmp")
TEMP_ROOT_ENV = "FRONTIER_WALLTIME_TMPDIR"
RUN_CASE_PATH = Path(__file__).with_name("run_case.py").resolve()
DENSE_MASTER_WRAPPER = (
    "systemd-run",
    "--user",
    "--scope",
    "-p",
    "MemoryMax=4G",
    "--quiet",
)
TERMINAL_STATUSES = frozenset(
    {"success", "simulated-oom", "timeout", "host-oom", "bug"}
)


@dataclass(frozen=True)
class ChildOutcome:
    """Bounded evidence returned by one child-process execution."""

    returncode: int
    elapsed_s: float
    stdout_tail: str
    stderr_tail: str
    timed_out: bool


AttemptRunner = Callable[[list[str], float], ChildOutcome]


def build_cases(
    model: str,
    scales: Sequence[int],
    mode: str,
    shape_overrides: Mapping[str, int] | None,
    attempt_index: int = 0,
) -> list[CaseSpec]:
    """Build primary weak-scaling attempts while holding replica shape fixed."""

    if model not in PRIMARY_SHAPES:
        raise ValueError(
            f"model must be one of {sorted(PRIMARY_SHAPES)}, got {model!r}"
        )
    shape_values = dict(PRIMARY_SHAPES[model])
    overrides = dict(shape_overrides or {})
    unknown_fields = set(overrides) - set(shape_values)
    if unknown_fields:
        raise ValueError(f"Unknown parallel shape fields: {sorted(unknown_fields)}")
    shape_values.update(overrides)
    shape = ParallelShape(**shape_values)

    return [
        CaseSpec.for_scale(
            model=model,
            total_gpus=scale,
            mode=mode,
            attempt_index=attempt_index,
            shape=shape,
        )
        for scale in scales
    ]


def _case_path(output_root: Path, case: CaseSpec) -> Path:
    return output_root / "cases" / f"{case.attempt_id}.json"


def _result_path(output_root: Path, case: CaseSpec) -> Path:
    return output_root / "results" / f"{case.attempt_id}.json"


def _load_json_object(path: Path, artifact_name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} JSON must contain an object: {path}")
    return payload


def _ensure_case_artifact(path: Path, case: CaseSpec) -> None:
    if not path.exists():
        write_json_atomic(path, case.to_dict())
        return

    payload = _load_json_object(path, "Case")
    try:
        existing_case = CaseSpec.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CaseSpec artifact at {path}: {exc}") from exc
    if existing_case != case:
        raise ValueError(
            f"CaseSpec mismatch for immutable artifact {path}: "
            f"existing={existing_case.to_dict()}, expected={case.to_dict()}"
        )


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    )


def _signal_from_returncode(returncode: int) -> int | None:
    if returncode < 0:
        return -returncode
    if 128 < returncode <= 255:
        return returncode - 128
    return None


def _validate_result(path: Path, case: CaseSpec) -> dict[str, Any]:
    payload = _load_json_object(path, "Result")
    missing_fields = REQUIRED_RESULT_FIELDS - set(payload)
    if missing_fields:
        raise ValueError(
            f"Result JSON is incomplete at {path}; missing fields: "
            f"{sorted(missing_fields)}"
        )

    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "attempt_id": case.attempt_id,
        "attempt_index": case.attempt_index,
        "case_fingerprint": case.case_fingerprint,
        "seed": case.seed,
        "model": case.model,
        "model_name": case.model_name,
        "total_gpus": case.total_gpus,
        "simulation_mode": case.simulation_mode,
        "shape": case.to_dict()["shape"],
        "replicas_per_cluster": case.replicas_per_cluster,
        "mode": case.mode,
        "requests": case.num_requests,
        "qps": case.qps,
        "prefill_tokens": case.prefill_tokens,
        "decode_tokens": case.decode_tokens,
    }
    for field_name, expected_value in expected_identity.items():
        if payload[field_name] != expected_value:
            raise ValueError(
                f"Result identity mismatch at {path}: field {field_name!r} "
                f"is {payload[field_name]!r}, expected {expected_value!r}"
            )

    status = payload["status"]
    if not isinstance(status, str) or status not in TERMINAL_STATUSES:
        raise ValueError(
            f"Result status is not terminal at {path}: {status!r}"
        )
    if status == "success":
        request_counts = (
            payload["requests"],
            payload["expected_requests"],
            payload["completed_requests"],
        )
        if request_counts != (case.num_requests,) * 3:
            raise ValueError(
                f"Invalid success request counts at {path}: {request_counts!r}"
            )
        for field_name in ("sim_wallclock_s", "event_count", "events_per_s"):
            if not _positive_number(payload[field_name]):
                raise ValueError(
                    f"Invalid success field {field_name!r} at {path}: "
                    f"{payload[field_name]!r}"
                )
    elif status == "timeout":
        exit_code = payload["exit_code"]
        valid_timeout = (
            payload["failure_reason"] == "parent_timeout"
            and payload["sim_wallclock_s"] is None
            and _positive_number(payload["total_proc_s"])
            and isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and payload["signal"] == _signal_from_returncode(exit_code)
        )
        if not valid_timeout:
            raise ValueError(f"Invalid timeout result contract at {path}")
    elif status == "host-oom":
        exit_code = payload["exit_code"]
        valid_host_oom = (
            payload["failure_reason"] == "parent_host_oom"
            and payload["sim_wallclock_s"] is None
            and _positive_number(payload["total_proc_s"])
            and isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and exit_code in (137, -signal.SIGKILL)
            and payload["signal"] == signal.SIGKILL
            and bool(payload["oom_evidence"])
        )
        if not valid_host_oom:
            raise ValueError(f"Invalid host-oom result contract at {path}")
    elif status == "simulated-oom":
        exit_code = payload["exit_code"]
        failure_reason = payload["failure_reason"]
        valid_simulated_oom = (
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and exit_code == 2
            and payload["signal"] is None
            and isinstance(failure_reason, str)
            and bool(failure_reason.strip())
            and _positive_number(payload["total_proc_s"])
            and (
                payload["sim_wallclock_s"] is None
                or _positive_number(payload["sim_wallclock_s"])
            )
            and isinstance(payload["oom_evidence"], Mapping)
        )
        if not valid_simulated_oom:
            raise ValueError(f"Invalid simulated-oom result contract at {path}")
    elif status == "bug":
        exit_code = payload["exit_code"]
        failure_reason = payload["failure_reason"]
        valid_bug = (
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and payload["signal"] == _signal_from_returncode(exit_code)
            and isinstance(failure_reason, str)
            and bool(failure_reason.strip())
            and payload["oom_evidence"] is None
            and _positive_number(payload["total_proc_s"])
            and (
                payload["sim_wallclock_s"] is None
                or _positive_number(payload["sim_wallclock_s"])
            )
        )
        if not valid_bug:
            raise ValueError(f"Invalid bug result contract at {path}")
    return payload


def _read_tail(handle: BinaryIO) -> str:
    handle.flush()
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(max(0, size - OUTPUT_TAIL_BYTES))
    return handle.read(OUTPUT_TAIL_BYTES).decode("utf-8", errors="replace")


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _resolve_temp_root() -> Path:
    configured_value = os.environ.get(TEMP_ROOT_ENV)
    temp_root = (
        Path(configured_value)
        if configured_value is not None
        else DEFAULT_TEMP_ROOT
    )
    if not temp_root.is_absolute():
        raise ValueError(
            f"{TEMP_ROOT_ENV} must be an absolute path, got {configured_value!r}"
        )
    resolved_default_root = DEFAULT_TEMP_ROOT.resolve(strict=False)
    resolved_temp_root = temp_root.resolve(strict=False)
    try:
        resolved_temp_root.relative_to(resolved_default_root)
    except ValueError as exc:
        raise ValueError(
            f"{TEMP_ROOT_ENV} must resolve to {resolved_default_root} or one of "
            f"its descendants, got {resolved_temp_root}"
        ) from exc
    temp_root = resolved_temp_root
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:
        raise NotADirectoryError(
            f"Configured wall-time temp root is not a directory: {temp_root}"
        ) from exc
    if not temp_root.is_dir():
        raise NotADirectoryError(
            f"Configured wall-time temp root is not a directory: {temp_root}"
        )
    return temp_root


def _execute_child(command: list[str], timeout_s: float) -> ChildOutcome:
    """Execute one attempt in a new session and terminate its process group."""

    started = time.perf_counter()
    temp_root = _resolve_temp_root()
    with tempfile.TemporaryFile(
        mode="w+b", dir=temp_root
    ) as stdout_file, tempfile.TemporaryFile(
        mode="w+b", dir=temp_root
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = process.wait(timeout=TERMINATION_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = process.wait()
            else:
                if _process_group_exists(process.pid):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

        return ChildOutcome(
            returncode=returncode,
            elapsed_s=time.perf_counter() - started,
            stdout_tail=_read_tail(stdout_file),
            stderr_tail=_read_tail(stderr_file),
            timed_out=timed_out,
        )


_OOM_PATTERNS = (
    ("oom-kill", re.compile(r"oom-kill[^\r\n]*", re.IGNORECASE)),
    (
        "memory cgroup out of memory",
        re.compile(r"memory cgroup out of memory[^\r\n]*", re.IGNORECASE),
    ),
    (
        "out of memory: killed process",
        re.compile(r"out of memory: killed process[^\r\n]*", re.IGNORECASE),
    ),
    (
        "memory.events oom_kill",
        re.compile(
            r"memory\.events:.{0,4096}?oom_kill\s+[1-9]\d*",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


def _find_oom_evidence(
    stdout_tail: str, stderr_tail: str
) -> dict[str, str] | None:
    for source, text in (("stdout", stdout_tail), ("stderr", stderr_tail)):
        for pattern_name, pattern in _OOM_PATTERNS:
            match = pattern.search(text)
            if match:
                return {
                    "source": source,
                    "pattern": pattern_name,
                    "matched_text": match.group(0),
                }
    return None


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    sha = result.stdout.strip()
    if len(sha) != 40:
        raise RuntimeError(f"Unexpected git SHA from rev-parse: {sha!r}")
    return sha


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_command(
    case_path: Path,
    result_path: Path,
    python_executable: str,
    dense_master_wrapper: bool,
) -> list[str]:
    command = [
        python_executable,
        str(RUN_CASE_PATH),
        "--case-json",
        str(case_path),
        "--result-json",
        str(result_path),
    ]
    if dense_master_wrapper:
        return [*DENSE_MASTER_WRAPPER, *command]
    return command


def _parent_result(
    *,
    case: CaseSpec,
    command: list[str],
    python_executable: str,
    git_sha: str,
    started_at: str,
    completed_at: str,
    outcome: ChildOutcome,
) -> dict[str, Any]:
    oom_evidence = None
    if outcome.timed_out:
        status = "timeout"
        failure_reason = "parent_timeout"
    elif outcome.returncode in {137, -signal.SIGKILL}:
        oom_evidence = _find_oom_evidence(
            outcome.stdout_tail, outcome.stderr_tail
        )
        if oom_evidence is not None:
            status = "host-oom"
            failure_reason = "parent_host_oom"
        else:
            status = "bug"
            failure_reason = "child_missing_result"
    else:
        status = "bug"
        failure_reason = "child_missing_result"

    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "attempt_id": case.attempt_id,
        "attempt_index": case.attempt_index,
        "case_fingerprint": case.case_fingerprint,
        "git_sha": git_sha,
        "python_executable": python_executable,
        "seed": case.seed,
        "model": case.model,
        "model_name": case.model_name,
        "total_gpus": case.total_gpus,
        "simulation_mode": case.simulation_mode,
        "shape": case.to_dict()["shape"],
        "replicas_per_cluster": case.replicas_per_cluster,
        "mode": case.mode,
        "host": socket.gethostname(),
        "worker_job_id": os.environ.get("FRONTIER_WORKER_JOB_ID"),
        "status": status,
        "sim_wallclock_s": None,
        "init_s": None,
        "total_proc_s": outcome.elapsed_s,
        "peak_rss_mb": None,
        "requests": case.num_requests,
        "qps": case.qps,
        "prefill_tokens": case.prefill_tokens,
        "decode_tokens": case.decode_tokens,
        "expected_requests": case.num_requests,
        "completed_requests": None,
        "event_count": None,
        "events_per_s": None,
        "command": command,
        "started_at": started_at,
        "completed_at": completed_at,
        "exit_code": outcome.returncode,
        "signal": _signal_from_returncode(outcome.returncode),
        "failure_reason": failure_reason,
        "oom_evidence": oom_evidence,
        "notes": [],
        "stderr_tail": outcome.stderr_tail,
        "stdout_tail": outcome.stdout_tail,
    }


def _run_pending_attempt(
    *,
    case: CaseSpec,
    case_path: Path,
    result_path: Path,
    python_executable: str,
    dense_master_wrapper: bool,
    timeout_s: float,
    attempt_runner: AttemptRunner,
    git_sha: str,
) -> Path:
    command = _build_command(
        case_path,
        result_path,
        python_executable,
        dense_master_wrapper,
    )
    started_at = _utc_now()
    outcome = attempt_runner(command, timeout_s)

    if result_path.exists():
        _validate_result(result_path, case)
        return result_path

    parent_result = _parent_result(
        case=case,
        command=command,
        python_executable=python_executable,
        git_sha=git_sha,
        started_at=started_at,
        completed_at=_utc_now(),
        outcome=outcome,
    )
    try:
        write_json_atomic(result_path, parent_result)
    except FileExistsError:
        _validate_result(result_path, case)
        return result_path
    _validate_result(result_path, case)
    return result_path


def run_sweep(
    cases: Sequence[CaseSpec],
    output_root: Path,
    *,
    max_concurrency: int = 1,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    python_executable: str | os.PathLike[str] | None = None,
    dense_master_wrapper: bool = False,
    attempt_runner: AttemptRunner | None = None,
) -> list[Path]:
    """Run or strictly resume immutable attempts in input order."""

    if (
        not isinstance(max_concurrency, int)
        or isinstance(max_concurrency, bool)
        or max_concurrency <= 0
    ):
        raise ValueError(
            f"max_concurrency must be a positive integer, got {max_concurrency!r}"
        )
    if (
        not isinstance(timeout_s, (int, float))
        or isinstance(timeout_s, bool)
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0
    ):
        raise ValueError(
            f"timeout_s must be a positive finite number, got {timeout_s!r}"
        )

    case_list = list(cases)
    attempt_ids = [case.attempt_id for case in case_list]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("Each CaseSpec attempt_id must be unique within a sweep")

    output_root = Path(output_root).resolve(strict=False)
    resolved_python = os.fspath(python_executable or sys.executable)
    runner = attempt_runner if attempt_runner is not None else _execute_child
    result_paths: list[Path] = []
    pending: list[tuple[int, CaseSpec, Path, Path]] = []

    for index, case in enumerate(case_list):
        case_path = _case_path(output_root, case)
        result_path = _result_path(output_root, case)
        _ensure_case_artifact(case_path, case)
        result_paths.append(result_path)
        if result_path.exists():
            _validate_result(result_path, case)
        else:
            pending.append((index, case, case_path, result_path))

    if not pending:
        return result_paths

    git_sha = _git_sha()
    with ThreadPoolExecutor(
        max_workers=min(max_concurrency, len(pending))
    ) as executor:
        futures = {
            index: executor.submit(
                _run_pending_attempt,
                case=case,
                case_path=case_path,
                result_path=result_path,
                python_executable=resolved_python,
                dense_master_wrapper=dense_master_wrapper,
                timeout_s=float(timeout_s),
                attempt_runner=runner,
                git_sha=git_sha,
            )
            for index, case, case_path, result_path in pending
        }
        for index, _case, _case_path_value, _result_path_value in pending:
            result_paths[index] = futures[index].result()

    return result_paths


def _parse_shape_overrides(values: Sequence[str]) -> dict[str, int]:
    valid_names = set(PRIMARY_SHAPES["dense"])
    overrides: dict[str, int] = {}
    for value in values:
        name, separator, raw_value = value.partition("=")
        if not separator or not name or not raw_value:
            raise ValueError(
                f"shape override must use NAME=INT syntax, got {value!r}"
            )
        if name not in valid_names:
            raise ValueError(
                f"unknown shape override {name!r}; expected one of "
                f"{sorted(valid_names)}"
            )
        if name in overrides:
            raise ValueError(f"duplicate shape override: {name!r}")
        try:
            overrides[name] = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"shape override {name!r} must be an integer, got {raw_value!r}"
            ) from exc
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Frontier wall-clock weak-scaling sweep."
    )
    parser.add_argument("--model", choices=sorted(PRIMARY_SHAPES), required=True)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=DEFAULT_SCALES,
        metavar="TOTAL_GPUS",
    )
    parser.add_argument(
        "--mode",
        choices=("sequential", "parallel"),
        default="sequential",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dense-master-wrapper", action="store_true")
    parser.add_argument(
        "--shape-override",
        action="append",
        default=[],
        metavar="NAME=INT",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    sweep_runner: Callable[..., list[Path]] | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dense_master_wrapper and args.model != "dense":
        parser.error("--dense-master-wrapper is valid only for --model dense")
    try:
        shape_overrides = _parse_shape_overrides(args.shape_override)
        cases = build_cases(
            args.model,
            args.scales,
            args.mode,
            shape_overrides,
            attempt_index=args.attempt_index,
        )
    except ValueError as exc:
        parser.error(str(exc))

    runner = sweep_runner if sweep_runner is not None else run_sweep
    result_paths = runner(
        cases,
        args.output_root,
        max_concurrency=args.max_concurrency,
        timeout_s=args.timeout_s,
        python_executable=args.python_executable,
        dense_master_wrapper=args.dense_master_wrapper,
    )
    for result_path in result_paths:
        print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only old-version replay for the non-dummy MoE EP matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from tests.e2e.moe_ep_non_dummy_matrix import (
    MatrixCase,
    _finite_metric_values,
    _find_metrics_dir,
    _load_manifest,
    _load_result_rows,
    _merge_result_rows,
    _script_for_case,
    _write_jsonl,
    validate_profile_inputs,
)
from tests.scratch_root import SCRATCH_ROOT_ENV, resolve_scratch_root


EXPECTED_BASELINE_COMMIT = "5cb8dda2ed6aafeaa02a480685d34014ae43e4f9"
EXPECTED_ARCHITECTURE_COUNTS = Counter(
    {
        "co-location": 50,
        "pd-disaggregation": 50,
        "pd-af-disaggregation": 10,
    }
)
SUPPORTED_ROUTING_DISTRIBUTIONS = frozenset(
    {"balanced", "random", "skewed", "zipf"}
)
_CACHE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def translate_routing_selector(distribution: str) -> tuple[str, str]:
    """Translate the current selector into fields owned by the old runtime."""

    normalized = str(distribution).strip()
    if normalized not in SUPPORTED_ROUTING_DISTRIBUTIONS:
        raise ValueError(f"unsupported routing distribution: {distribution!r}")
    mode = "uniform_random" if normalized == "random" else "simulation"
    return mode, normalized


def tmp_root() -> Path:
    """Return the scratch root that bounds every replay write path."""

    return resolve_scratch_root()


def _require_tmp_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    scratch_root = tmp_root()
    if not resolved.is_relative_to(scratch_root.resolve()):
        raise ValueError(
            f"{label} must be under {scratch_root}: {resolved} "
            f"(set {SCRATCH_ROOT_ENV} to relocate the scratch root)"
        )
    return resolved


def validate_manifest_cases(cases: Sequence[MatrixCase]) -> None:
    """Validate the immutable 110-case comparison surface."""

    if len(cases) != 110:
        raise ValueError(f"baseline manifest must contain 110 cases, got {len(cases)}")
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("baseline manifest contains duplicate case IDs")
    architecture_counts = Counter(case.architecture for case in cases)
    if architecture_counts != EXPECTED_ARCHITECTURE_COUNTS:
        raise ValueError(
            "baseline manifest architecture counts do not match the current matrix: "
            f"{dict(architecture_counts)}"
        )
    for case in cases:
        if case.total_cards <= 0:
            raise ValueError(f"{case.case_id} has a non-positive card count")
        if case.total_cards > 32:
            raise ValueError(
                f"{case.case_id} exceeds 32 cards: total_cards={case.total_cards}"
            )
        translate_routing_selector(case.routing_distribution)


def load_and_validate_manifest(path: Path) -> list[MatrixCase]:
    cases = _load_manifest(path)
    validate_manifest_cases(cases)
    return cases


def prepare_data_only_cwd(current_data_root: Path, data_cwd: Path) -> Path:
    """Expose current profiling data without exposing the current source tree."""

    current_data_root = current_data_root.resolve()
    if not current_data_root.is_dir():
        raise FileNotFoundError(current_data_root)
    data_cwd = _require_tmp_path(data_cwd, label="data-only cwd")
    data_cwd.mkdir(parents=True, exist_ok=True)
    link = data_cwd / "data"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(current_data_root, target_is_directory=True)
    validate_data_only_cwd(current_data_root, data_cwd)
    return data_cwd


def validate_data_only_cwd(current_data_root: Path, data_cwd: Path) -> None:
    current_data_root = current_data_root.resolve()
    data_cwd = _require_tmp_path(data_cwd, label="data-only cwd")
    entries = sorted(path.name for path in data_cwd.iterdir())
    if entries != ["data"]:
        raise RuntimeError(
            f"data-only cwd must contain only the data link, found {entries!r}"
        )
    link = data_cwd / "data"
    if not link.is_symlink() or link.resolve() != current_data_root:
        raise RuntimeError(
            "data-only cwd data link does not target the current profiling data tree"
        )


def validate_baseline_worktree(
    repo_root: Path,
    *,
    expected_commit: str = EXPECTED_BASELINE_COMMIT,
) -> str:
    """Require the exact clean detached comparison commit."""

    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != expected_commit:
        raise RuntimeError(
            f"baseline commit mismatch: expected={expected_commit}, actual={head}"
        )
    branch = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if branch.returncode == 0:
        raise RuntimeError(
            f"baseline worktree must be detached, found branch={branch.stdout.strip()!r}"
        )
    if branch.returncode != 1:
        raise RuntimeError(
            f"failed to inspect baseline detached state: {branch.stderr.strip()}"
        )
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    )
    if status.strip():
        raise RuntimeError(f"baseline worktree is not clean:\n{status.rstrip()}")
    return head


def build_baseline_shell_command(
    case: MatrixCase,
    baseline_repo_root: Path,
    output_root: Path,
    *,
    python_executable: Path,
    cache_token: str,
) -> tuple[str, dict[str, str]]:
    """Build one old-wrapper command without adding current source to sys.path."""

    if not _CACHE_TOKEN_RE.fullmatch(cache_token):
        raise ValueError(f"invalid cache token: {cache_token!r}")
    baseline_repo_root = baseline_repo_root.resolve()
    output_root = _require_tmp_path(output_root, label="baseline output root")
    script = _script_for_case(case, baseline_repo_root)
    case_dir = output_root / case.case_id
    metrics_root = case_dir / "metrics"
    predictor_cache_dir = case_dir / f"predictor_cache_{cache_token}"
    routing_mode, routing_distribution = translate_routing_selector(
        case.routing_distribution
    )

    scratch_root = tmp_root()
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(baseline_repo_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(scratch_root),
            "TEMP": str(scratch_root),
            "TMP": str(scratch_root),
            "PYTHON_BIN": str(python_executable),
            "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1",
            "MODEL_NAME": case.model_name,
            "ENABLE_DUMMY_MODE": "false",
            "DECODE_CUDA_GRAPH_MODE": "none",
            "ENABLE_CUDA_GRAPH": "false",
            "ENABLE_CHUNKED_PREFILL": "false",
            "NUM_REQUESTS": str(case.num_requests),
            "PREFILL_TOKENS": str(case.prefill_tokens),
            "DECODE_TOKENS": str(case.decode_tokens),
            "QPS": "1.0",
            "RUN_ID": case.case_id,
            "METRICS_OUTPUT_DIR": str(metrics_root),
            "MOE_ROUTING_MODE": routing_mode,
            "MOE_ROUTING_DISTRIBUTION_TYPE": routing_distribution,
            "MOE_ROUTING_SEED": str(case.seed),
            "TOTAL_EXPERTS": str(case.total_experts),
            "ROUTER_TOPK": str(case.router_topk),
            "MAX_TOKENS_IN_BATCH": "64",
            "LONG_PREFILL_TOKEN_THRESHOLD": "0",
        }
    )
    if case.architecture == "co-location":
        env.update(
            {
                "SYS_ARCH": "co-location",
                "NUM_REPLICAS": str(case.replica_count),
                "ATTN_TP": str(case.attn_tensor_parallel_size),
                "MOE_TP": str(case.moe_tensor_parallel_size),
                "MOE_EP": str(case.ep_size),
                "PP": str(case.pipeline_stages),
                "DP": "1",
                "DEVICE": case.device,
            }
        )
    elif case.architecture == "pd-disaggregation":
        env.update(
            {
                "SYS_ARCH": "pd-disaggregation",
                "PREFILL_REPLICAS": str(case.prefill_replicas),
                "DECODE_REPLICAS": str(case.decode_replicas),
                "PREFILL_ATTN_TP": str(case.prefill_attn_tensor_parallel_size),
                "PREFILL_ATTN_DP": "1",
                "PREFILL_MOE_TP": str(case.prefill_moe_tensor_parallel_size),
                "PREFILL_MOE_EP": str(case.prefill_moe_expert_parallel_size),
                "DECODE_ATTN_TP": str(case.decode_attn_tensor_parallel_size),
                "DECODE_ATTN_DP": "1",
                "DECODE_MOE_TP": str(case.decode_moe_tensor_parallel_size),
                "DECODE_MOE_EP": str(case.decode_moe_expert_parallel_size),
                "PREFILL_DEVICE": case.device,
                "DECODE_DEVICE": case.device,
                "PREFILL_PP": str(case.pipeline_stages),
                "DECODE_PP": str(case.pipeline_stages),
            }
        )
    elif case.architecture == "pd-af-disaggregation":
        env.update(
            {
                "SYS_ARCH": "pd-af-disaggregation",
                "PREFILL_REPLICAS": str(case.prefill_replicas),
                "DECODE_ATTN_REPLICAS": str(case.decode_attn_replicas),
                "DECODE_FFN_REPLICAS": str(case.decode_ffn_replicas),
                "PREFILL_ATTN_TP": str(case.prefill_attn_tensor_parallel_size),
                "PREFILL_ATTN_DP": "1",
                "PREFILL_MOE_TP": str(case.prefill_moe_tensor_parallel_size),
                "PREFILL_MOE_EP": str(case.prefill_moe_expert_parallel_size),
                "DECODE_ATTN_TP": str(case.decode_attn_tensor_parallel_size),
                "DECODE_ATTN_DP": "1",
                "DECODE_FFN_MOE_TP": str(case.decode_moe_tensor_parallel_size),
                "DECODE_FFN_MOE_EP": str(case.decode_moe_expert_parallel_size),
                "PREFILL_DEVICE": case.device,
                "DECODE_ATTN_DEVICE": case.device,
                "DECODE_FFN_DEVICE": case.device,
                "PREFILL_PP": str(case.pipeline_stages),
                "DECODE_ATTN_PP": "1",
                "DECODE_FFN_PP": str(case.pipeline_stages),
            }
        )
    else:
        raise ValueError(f"unsupported architecture: {case.architecture}")

    command_parts = [
        "bash",
        str(script),
        "--",
        "--replica_config_moe_routing_distribution_type",
        routing_distribution,
    ]
    if case.architecture == "co-location" and case.is_moe:
        command_parts.extend(["--replica_config_device", case.device])
    command_parts.extend(
        ["--metrics_config_cache_dir", str(predictor_cache_dir.resolve())]
    )
    return shlex.join(command_parts), env


def check_baseline_case(
    case: MatrixCase,
    log_path: Path,
    metrics_dir: Path | None,
) -> dict[str, Any]:
    """Classify execution and current-schema workflow evidence independently."""

    errors: list[str] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Traceback" in text:
        errors.append("Traceback")
    if "Simulation completed successfully." not in text:
        errors.append("missing success marker")
    if "Dummy Mode: false" not in text:
        errors.append("dummy mode was not explicitly disabled")

    metrics: dict[str, Any] = {}
    numeric_metric_count = 0
    if metrics_dir is None:
        errors.append("missing fresh metrics directory")
    else:
        metric_path = metrics_dir / "system_metrics.json"
        if not metric_path.is_file():
            errors.append(f"missing metrics file: {metric_path}")
        else:
            try:
                metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                numeric_metric_count = sum(1 for _ in _finite_metric_values(metrics))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"invalid metrics: {exc}")

    def _stat_value(name: str) -> float | None:
        value = metrics.get(name, {})
        if not isinstance(value, Mapping):
            return None
        candidate = value.get("mean")
        if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
            return None
        result = float(candidate)
        if not math.isfinite(result) or result < 0:
            errors.append(f"invalid {name} mean: {candidate!r}")
            return None
        return result

    ttft_mean_ms = _stat_value("ttft_statistics")
    tpot_mean_ms = _stat_value("tpot_statistics")
    e2e_mean_ms = _stat_value("request_e2e_time_statistics")
    if ttft_mean_ms is None:
        errors.append("missing finite ttft_statistics mean")
    if e2e_mean_ms is None:
        errors.append("missing finite request_e2e_time_statistics mean")

    ep_workload_records = text.count("[EP-WORKLOAD]")
    dispatch_barrier_records = len(
        re.findall(r"\[EP-BARRIER\].*phase=dispatch", text)
    )
    combine_barrier_records = len(
        re.findall(r"\[EP-BARRIER\].*phase=combine", text)
    )
    ep_conservation_records = text.count("[EP-CONSERVATION]")
    current_counts = (
        ep_workload_records,
        dispatch_barrier_records,
        combine_barrier_records,
        ep_conservation_records,
    )
    if not case.is_moe:
        workflow_status = "NOT_APPLICABLE_DENSE"
    elif all(count > 0 for count in current_counts):
        workflow_status = "COMPLETE_CURRENT_SCHEMA"
    elif any(count > 0 for count in current_counts):
        workflow_status = "PARTIAL_CURRENT_SCHEMA"
    else:
        workflow_status = "MISSING_CURRENT_SCHEMA"

    old_moe_op_trace_count = sum(
        "[OP-TRACE]" in line and "[MOE]" in line for line in text.splitlines()
    )
    layer_ids = sorted({int(value) for value in re.findall(r"layer_id=(\d+)", text)})
    return {
        "execution_status": "PASS" if not errors else "FAIL",
        "execution_errors": "; ".join(errors),
        "workflow_evidence_status": workflow_status,
        "layer_ids": layer_ids,
        "old_op_trace_count": text.count("[OP-TRACE]"),
        "old_moe_op_trace_count": old_moe_op_trace_count,
        "old_per_expert_map_records": text.count("per_expert_tokens extracted:"),
        "ep_workload_records": ep_workload_records,
        "dispatch_barrier_records": dispatch_barrier_records,
        "combine_barrier_records": combine_barrier_records,
        "ep_conservation_records": ep_conservation_records,
        "legacy_scaling_wording_records": len(
            re.findall(
                r"(?i)(scaling factor|visibility multiplier|calibration scale)",
                text,
            )
        ),
        "numeric_metric_count": numeric_metric_count,
        "ttft_mean_ms": ttft_mean_ms,
        "tpot_mean_ms": tpot_mean_ms,
        "e2e_mean_ms": e2e_mean_ms,
    }


def validate_baseline_ledger_provenance(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_repo_root: Path,
    baseline_commit: str,
    data_cwd: Path,
    output_root: Path,
    results_path: Path,
) -> None:
    """Reject rows created by another baseline campaign or source commit."""

    expected_paths = {
        "baseline_repo_root": baseline_repo_root.resolve(),
        "data_cwd": data_cwd.resolve(),
        "output_root": output_root.resolve(),
        "results_path": results_path.resolve(),
    }
    expected_output = expected_paths["output_root"]
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("baseline result row has no non-empty case_id")
        for field, expected in expected_paths.items():
            value = row.get(field)
            if not isinstance(value, str):
                raise ValueError(
                    "baseline result ledger row is missing canonical provenance: "
                    f"case_id={case_id!r}, field={field!r}"
                )
            if Path(value).resolve() != expected:
                raise ValueError(
                    f"baseline result ledger {field} provenance mismatch: "
                    f"case_id={case_id!r}, row={value!r}, expected={str(expected)!r}"
                )
        if row.get("baseline_commit") != baseline_commit:
            raise ValueError(
                "baseline result ledger commit provenance mismatch: "
                f"case_id={case_id!r}, row={row.get('baseline_commit')!r}, "
                f"expected={baseline_commit!r}"
            )

        status = row.get("execution_status")
        if status not in {"PASS", "FAIL"}:
            raise ValueError(
                "baseline result ledger execution_status must be PASS or FAIL: "
                f"case_id={case_id!r}, status={status!r}"
            )
        case_root = expected_output / case_id
        log_path = row.get("log_path")
        metrics_path = row.get("metrics_path")
        if not isinstance(log_path, str) or not log_path:
            raise ValueError(
                f"baseline result ledger row has no log_path: case_id={case_id!r}"
            )
        if not Path(log_path).resolve().is_relative_to(case_root):
            raise ValueError(
                "baseline result ledger log_path is outside its canonical case directory: "
                f"case_id={case_id!r}, path={log_path!r}"
            )
        if not isinstance(metrics_path, str):
            raise ValueError(
                f"baseline result ledger metrics_path must be a string: case_id={case_id!r}"
            )
        if metrics_path and not Path(metrics_path).resolve().is_relative_to(case_root):
            raise ValueError(
                "baseline result ledger metrics_path is outside its canonical case directory: "
                f"case_id={case_id!r}, path={metrics_path!r}"
            )
        if status == "PASS" and not metrics_path:
            raise ValueError(
                "baseline result ledger PASS row has no canonical metrics_path: "
                f"case_id={case_id!r}"
            )


def _run_baseline_process(
    command: str,
    *,
    cwd: Path,
    env: Mapping[str, str],
    stream: TextIO,
    timeout_seconds: int,
) -> int:
    """Run one case and terminate its whole process group on timeout."""

    process = subprocess.Popen(
        shlex.split(command),
        cwd=cwd,
        env=dict(env),
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        return int(process.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        else:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        return 124


def run_baseline_cases(
    cases: Sequence[MatrixCase],
    baseline_repo_root: Path,
    profile_repo_root: Path,
    data_cwd: Path,
    output_root: Path,
    results_path: Path,
    *,
    python_executable: Path,
    expected_commit: str = EXPECTED_BASELINE_COMMIT,
    start: int = 0,
    limit: int | None = None,
    timeout_seconds: int = 600,
) -> list[dict[str, Any]]:
    """Run baseline cases while continuing only after old-runtime failures."""

    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive when provided, got {limit}")
    if timeout_seconds <= 0:
        raise ValueError(
            f"timeout_seconds must be positive, got {timeout_seconds}"
        )

    baseline_repo_root = baseline_repo_root.resolve()
    profile_repo_root = profile_repo_root.resolve()
    data_cwd = _require_tmp_path(data_cwd, label="data-only cwd")
    output_root = _require_tmp_path(output_root, label="baseline output root")
    results_path = results_path.resolve()
    python_executable = python_executable.resolve()
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise FileNotFoundError(
            f"baseline Python executable is not runnable: {python_executable}"
        )

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("baseline runner cases contain duplicate case IDs")
    for case in cases:
        if case.total_cards <= 0 or case.total_cards > 32:
            raise ValueError(
                f"{case.case_id} has invalid total_cards={case.total_cards}"
            )
        translate_routing_selector(case.routing_distribution)
    selected = list(cases[start : (start + limit) if limit is not None else None])
    if not selected:
        raise ValueError("baseline runner selected no cases")

    baseline_commit = validate_baseline_worktree(
        baseline_repo_root,
        expected_commit=expected_commit,
    )
    current_data_root = profile_repo_root / "data"
    prepare_data_only_cwd(current_data_root, data_cwd)
    output_root.mkdir(parents=True, exist_ok=True)

    existing_rows = _load_result_rows(results_path)
    validate_baseline_ledger_provenance(
        existing_rows,
        baseline_repo_root=baseline_repo_root,
        baseline_commit=baseline_commit,
        data_cwd=data_cwd,
        output_root=output_root,
        results_path=results_path,
    )
    persisted = _merge_result_rows(
        existing_rows,
        (),
        expected_case_ids=case_ids,
    )

    results: list[dict[str, Any]] = []
    for offset, case in enumerate(selected):
        validate_profile_inputs(case, profile_repo_root)
        validate_baseline_worktree(
            baseline_repo_root,
            expected_commit=baseline_commit,
        )
        validate_data_only_cwd(current_data_root, data_cwd)

        cache_token = f"attempt-{start + offset}-{time.time_ns()}"
        command, env = build_baseline_shell_command(
            case,
            baseline_repo_root,
            output_root,
            python_executable=python_executable,
            cache_token=cache_token,
        )
        case_dir = output_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        log_path = case_dir / f"{case.case_id}.log"
        metadata_path = case_dir / "baseline_case_metadata.json"
        run_started_at_s = time.time()
        metadata_path.write_text(
            json.dumps(
                {
                    "case": asdict(case),
                    "command": command,
                    "environment": {
                        key: env[key]
                        for key in (
                            "PYTHONPATH",
                            "PYTHONDONTWRITEBYTECODE",
                            "MODEL_NAME",
                            "ENABLE_DUMMY_MODE",
                            "MOE_ROUTING_MODE",
                            "MOE_ROUTING_DISTRIBUTION_TYPE",
                            "MOE_ROUTING_SEED",
                        )
                    },
                    "baseline_repo_root": str(baseline_repo_root),
                    "baseline_commit": baseline_commit,
                    "profile_repo_root": str(profile_repo_root),
                    "data_cwd": str(data_cwd),
                    "output_root": str(output_root),
                    "results_path": str(results_path),
                    "run_started_at_unix_s": run_started_at_s,
                    "trace_schema_version": 1,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        run_freshness_marker_ns = metadata_path.stat().st_mtime_ns

        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as stream:
            stream.write(f"BASELINE_COMMAND: {command}\n")
            stream.write(f"BASELINE_COMMIT: {baseline_commit}\n")
            stream.write(f"BASELINE_DATA_CWD: {data_cwd}\n")
            stream.flush()
            exit_code = _run_baseline_process(
                command,
                cwd=data_cwd,
                env=env,
                stream=stream,
                timeout_seconds=timeout_seconds,
            )
            if exit_code == 124:
                stream.write(f"BASELINE_TIMEOUT after {timeout_seconds}s\n")
        elapsed = time.monotonic() - started
        run_finished_at_s = time.time()

        validate_baseline_worktree(
            baseline_repo_root,
            expected_commit=baseline_commit,
        )
        validate_data_only_cwd(current_data_root, data_cwd)

        metrics_path = ""
        metrics_error = ""
        try:
            metrics_dir = _find_metrics_dir(
                output_root,
                case,
                started_at_ns=run_freshness_marker_ns,
            )
            metrics_path = str(metrics_dir)
        except (FileNotFoundError, OSError) as exc:
            metrics_dir = None
            metrics_error = str(exc)
        check = check_baseline_case(case, log_path, metrics_dir)
        if metrics_error:
            existing_error = str(check.get("execution_errors", "")).strip()
            check["execution_errors"] = "; ".join(
                part for part in (existing_error, metrics_error) if part
            )
            check["execution_status"] = "FAIL"

        execution_status = (
            "PASS"
            if exit_code == 0 and check.get("execution_status") == "PASS"
            else "FAIL"
        )
        result = {
            "case_id": case.case_id,
            "architecture": case.architecture,
            "model_kind": case.model_kind,
            "total_cards": case.total_cards,
            "exit_code": exit_code,
            "elapsed_seconds": round(elapsed, 3),
            "log_path": str(log_path),
            "metrics_path": metrics_path,
            "baseline_repo_root": str(baseline_repo_root),
            "baseline_commit": baseline_commit,
            "profile_repo_root": str(profile_repo_root),
            "data_cwd": str(data_cwd),
            "output_root": str(output_root),
            "results_path": str(results_path),
            "run_started_at_unix_s": run_started_at_s,
            "run_finished_at_unix_s": run_finished_at_s,
            "trace_schema_version": 1,
            "execution_status": execution_status,
            "workflow_evidence_status": check["workflow_evidence_status"],
            "check": check,
        }
        validate_baseline_ledger_provenance(
            (result,),
            baseline_repo_root=baseline_repo_root,
            baseline_commit=baseline_commit,
            data_cwd=data_cwd,
            output_root=output_root,
            results_path=results_path,
        )
        results.append(result)
        persisted = _merge_result_rows(
            persisted,
            (result,),
            expected_case_ids=case_ids,
        )
        _write_jsonl(results_path, persisted)

    validate_baseline_worktree(
        baseline_repo_root,
        expected_commit=baseline_commit,
    )
    validate_data_only_cwd(current_data_root, data_cwd)
    validate_baseline_ledger_provenance(
        persisted,
        baseline_repo_root=baseline_repo_root,
        baseline_commit=baseline_commit,
        data_cwd=data_cwd,
        output_root=output_root,
        results_path=results_path,
    )
    return results


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    task_dir = (
        repo_root
        / "task_memory"
        / "task_2026-08-12_moe_ep_rank_stragger_analysis"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=task_dir / "observable_v3" / "moe_ep_non_dummy_matrix_manifest.jsonl",
    )
    parser.add_argument(
        "--baseline-repo-root",
        type=Path,
        default=repo_root.parent / "baseline-replay",
    )
    parser.add_argument("--profile-repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--data-cwd",
        type=Path,
        default=tmp_root() / "frontier_baseline_replay_data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=tmp_root() / "frontier_baseline_replay_observable_v1",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=task_dir / "moe_ep_baseline_results_observable_v1.jsonl",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = args.manifest_path.resolve()
    baseline_repo_root = args.baseline_repo_root.resolve()
    profile_repo_root = args.profile_repo_root.resolve()
    data_cwd = args.data_cwd.resolve()
    output_root = args.output_root.resolve()
    results_path = args.results_path.resolve()
    python_executable = args.python_executable.resolve()

    cases = load_and_validate_manifest(manifest_path)
    results = run_baseline_cases(
        cases,
        baseline_repo_root,
        profile_repo_root,
        data_cwd,
        output_root,
        results_path,
        python_executable=python_executable,
        expected_commit=EXPECTED_BASELINE_COMMIT,
        start=args.start,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
    )
    passed = sum(row["execution_status"] == "PASS" for row in results)
    failed = len(results) - passed
    workflow_counts = Counter(
        str(row.get("workflow_evidence_status", "UNKNOWN")) for row in results
    )
    print(
        f"results={results_path} passed={passed} failed={failed} "
        f"workflow={dict(workflow_counts)}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

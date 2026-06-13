#!/usr/bin/env python3
"""Cross-validate PDD example scripts against a reference Frontier worktree.

The candidate run executes each public PDD shell script directly. The reference
run executes the same script but replaces the Python import root through a small
PYTHON_BIN wrapper, so both roles use identical shell defaults, environment
knobs, trace fixtures, and CLI arguments while importing different Frontier
implementations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScriptCase:
    case_id: str
    script_relpath: str
    expected_requests: int
    expected_transfers: int
    env: dict[str, str] = field(default_factory=dict)


SCRIPT_CASES: tuple[ScriptCase, ...] = (
    ScriptCase(
        "compat_dense",
        "examples/architecture/pdd/dense_model_basic.sh",
        expected_requests=1,
        expected_transfers=1,
        env={"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"},
    ),
    ScriptCase(
        "offline_dense",
        "examples/architecture/pdd/offline/dense_model_basic.sh",
        expected_requests=1,
        expected_transfers=1,
        env={"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"},
    ),
    ScriptCase(
        "offline_moe",
        "examples/architecture/pdd/offline/moe_model_basic.sh",
        expected_requests=1,
        expected_transfers=1,
        env={"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"},
    ),
    ScriptCase(
        "offline_thinking",
        "examples/architecture/pdd/offline/thinking_mode_basic.sh",
        expected_requests=1,
        expected_transfers=2,
        env={"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"},
    ),
    ScriptCase(
        "offline_spec_dec",
        "examples/architecture/pdd/offline/moe_spec_dec.sh",
        expected_requests=1,
        expected_transfers=1,
        env={"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"},
    ),
    ScriptCase(
        "offline_prefix",
        "examples/architecture/pdd/offline/moe_prefix_caching.sh",
        expected_requests=2,
        expected_transfers=2,
    ),
    ScriptCase(
        "online_dense",
        "examples/architecture/pdd/online/dense_model_basic_online.sh",
        expected_requests=1,
        expected_transfers=1,
        env={"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"},
    ),
    ScriptCase(
        "online_moe",
        "examples/architecture/pdd/online/moe_model_basic_online.sh",
        expected_requests=1,
        expected_transfers=1,
        env={"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"},
    ),
    ScriptCase(
        "online_thinking",
        "examples/architecture/pdd/online/thinking_mode_basic_online.sh",
        expected_requests=1,
        expected_transfers=2,
        env={"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"},
    ),
    ScriptCase(
        "online_spec_dec",
        "examples/architecture/pdd/online/moe_spec_dec_online.sh",
        expected_requests=1,
        expected_transfers=1,
        env={"NUM_REQUESTS": "1", "PREFILL_TOKENS": "8", "DECODE_TOKENS": "2"},
    ),
    ScriptCase(
        "online_prefix",
        "examples/architecture/pdd/online/moe_prefix_caching_online.sh",
        expected_requests=2,
        expected_transfers=2,
    ),
)

REQUEST_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("request_id", ("request_id", "Request Id")),
    ("request_e2e_time", ("request_e2e_time", "request_e2e_time_ms")),
    ("request_waiting_time_total", ("request_waiting_time_total", "request_waiting_time_total_ms")),
    ("ttft", ("ttft", "ttft_ms")),
    ("tpot", ("tpot", "tpot_ms")),
    ("transfer_kv_cache", ("transfer_kv_cache",)),
    ("transfer_kv_cache_request_start_ts", ("transfer_kv_cache_request_start_ts",)),
    ("transfer_kv_cache_request_end_ts", ("transfer_kv_cache_request_end_ts",)),
    ("request_num_prefill_tokens", ("request_num_prefill_tokens", "num_prefill_tokens")),
    ("request_num_decode_tokens", ("request_num_decode_tokens", "num_decode_tokens")),
    ("request_thinking_round_count", ("request_thinking_round_count",)),
    ("request_session_id", ("request_session_id", "session_id")),
    ("request_spec_total_iterations", ("request_spec_total_iterations",)),
    ("request_spec_accepted_drafts", ("request_spec_accepted_drafts",)),
    ("request_spec_rejected_drafts", ("request_spec_rejected_drafts",)),
    ("request_spec_committed_tokens", ("request_spec_committed_tokens",)),
    ("request_spec_acceptance_ratio", ("request_spec_acceptance_ratio",)),
    ("request_spec_committed_per_iteration", ("request_spec_committed_per_iteration",)),
    ("request_cached_prefill_tokens", ("request_cached_prefill_tokens",)),
    ("request_prefix_cache_query_blocks", ("request_prefix_cache_query_blocks",)),
    ("request_prefix_cache_hit_blocks", ("request_prefix_cache_hit_blocks",)),
)

SYSTEM_STAT_FIELDS: tuple[str, ...] = (
    "ttft_statistics",
    "tpot_statistics",
    "e2e_latency_statistics",
    "request_e2e_time_statistics",
    "throughput_metrics",
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_metrics_run_dir(metrics_root: Path, run_id: str | None = None) -> Path:
    if not metrics_root.exists():
        raise FileNotFoundError(f"Metrics root does not exist: {metrics_root}")

    if run_id:
        run_id_dirs = [path for path in metrics_root.rglob(run_id) if path.is_dir()]
        candidates = [path for path in run_id_dirs if (path / "request_metrics.csv").is_file()]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(f"Found multiple metrics run dirs for run_id={run_id}: {candidates}")

    request_metric_files = list(metrics_root.rglob("request_metrics.csv"))
    candidates = [path.parent for path in request_metric_files]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one metrics run dir under {metrics_root}, found {len(candidates)}: {candidates}")
    return candidates[0]


def _request_value(row: dict[str, str], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        if alias in row:
            return row[alias]
    return ""


def _request_sort_key(row: dict[str, str]) -> tuple[int, str]:
    request_id = row.get("request_id", "")
    try:
        return (int(request_id), request_id)
    except ValueError:
        return (0, request_id)


def _normalize_request_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append({field: _request_value(row, aliases) for field, aliases in REQUEST_FIELDS})
    return sorted(normalized, key=_request_sort_key)


def load_metrics_summary(metrics_run_dir: Path, case_id: str, role: str) -> dict[str, Any]:
    request_metrics_path = metrics_run_dir / "request_metrics.csv"
    system_metrics_path = metrics_run_dir / "system_metrics.json"
    if not request_metrics_path.is_file():
        raise FileNotFoundError(f"Missing request metrics: {request_metrics_path}")
    if not system_metrics_path.is_file():
        raise FileNotFoundError(f"Missing system metrics: {system_metrics_path}")

    request_rows = _read_csv_rows(request_metrics_path)
    system_metrics = _load_json(system_metrics_path)
    metadata = system_metrics.get("simulation_metadata", {})
    transfer_stats = system_metrics.get("kv_cache_transfer_statistics", {})

    stable_system_stats: dict[str, Any] = {}
    for field_name in SYSTEM_STAT_FIELDS:
        if field_name in system_metrics:
            stable_system_stats[field_name] = system_metrics[field_name]

    return {
        "case_id": case_id,
        "role": role,
        "artifacts": {
            "request_metrics_csv": str(request_metrics_path),
            "system_metrics_json": str(system_metrics_path),
        },
        "counts": {
            "request_rows": len(request_rows),
            "total_requests": metadata.get("total_requests"),
            "completed_requests": metadata.get("completed_requests"),
        },
        "request_rows": _normalize_request_rows(request_rows),
        "kv_cache_transfer_statistics": transfer_stats,
        "kv_cache_transfer_total_bytes": system_metrics.get("kv_cache_transfer_total_bytes"),
        "prefix_cache_statistics": system_metrics.get("prefix_cache_statistics", {}),
        "spec_decode_statistics": system_metrics.get("spec_decode_statistics", {}),
        "system_statistics": stable_system_stats,
    }


def _canonical_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "counts": summary.get("counts"),
        "request_rows": summary.get("request_rows"),
        "kv_cache_transfer_statistics": summary.get("kv_cache_transfer_statistics"),
        "kv_cache_transfer_total_bytes": summary.get("kv_cache_transfer_total_bytes"),
        "prefix_cache_statistics": summary.get("prefix_cache_statistics"),
        "spec_decode_statistics": summary.get("spec_decode_statistics"),
        "system_statistics": summary.get("system_statistics"),
    }


def compare_summaries(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    candidate_canonical = _canonical_summary(candidate)
    reference_canonical = _canonical_summary(reference)
    mismatches: list[dict[str, Any]] = []
    for field_name in sorted(set(candidate_canonical) | set(reference_canonical)):
        candidate_value = candidate_canonical.get(field_name)
        reference_value = reference_canonical.get(field_name)
        if candidate_value != reference_value:
            mismatches.append(
                {
                    "field": field_name,
                    "candidate": candidate_value,
                    "reference": reference_value,
                }
            )
    return {
        "case_id": candidate.get("case_id"),
        "status": "PASS" if not mismatches else "FAIL",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _write_reference_wrapper(wrapper_path: Path, python_bin: Path, reference_root: Path) -> None:
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f"REFERENCE_ROOT={json.dumps(str(reference_root.resolve()))}\n"
        f"PYTHON_REAL={json.dumps(str(python_bin.resolve()))}\n"
        "if [ -n \"${PYTHONPATH:-}\" ]; then\n"
        "  export PYTHONPATH=\"$REFERENCE_ROOT:$PYTHONPATH\"\n"
        "else\n"
        "  export PYTHONPATH=\"$REFERENCE_ROOT\"\n"
        "fi\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        "exec \"$PYTHON_REAL\" \"$@\"\n",
        encoding="utf-8",
    )
    mode = wrapper_path.stat().st_mode
    wrapper_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_script_role(
    case: ScriptCase,
    candidate_root: Path,
    output_root: Path,
    role: str,
    python_bin: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    role_dir = output_root / case.case_id / role
    metrics_root = role_dir / "metrics"
    role_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(candidate_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1",
            "PYTHON_BIN": str(python_bin),
            "METRICS_OUTPUT_DIR": str(metrics_root),
            "RUN_ID": case.case_id,
            "QPS": "1000.0",
            "MAX_TOKENS_IN_BATCH": "128",
            "LONG_PREFILL_TOKEN_THRESHOLD": "4",
        }
    )
    env.update(case.env)

    script_path = candidate_root / case.script_relpath
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=candidate_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    (role_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (role_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
    _write_json(
        role_dir / "run_metadata.json",
        {
            "case_id": case.case_id,
            "role": role,
            "script_relpath": case.script_relpath,
            "returncode": result.returncode,
            "expected_requests": case.expected_requests,
            "expected_transfers": case.expected_transfers,
            "env_overrides": {key: env[key] for key in sorted(case.env | {"RUN_ID": "", "METRICS_OUTPUT_DIR": ""}) if key in env},
        },
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{role} run failed for {case.case_id} with exit code {result.returncode}. "
            f"See {role_dir / 'stdout.log'} and {role_dir / 'stderr.log'}"
        )

    metrics_run_dir = _find_metrics_run_dir(metrics_root, case.case_id)
    summary = load_metrics_summary(metrics_run_dir, case.case_id, role)
    transfer_count = summary["kv_cache_transfer_statistics"].get("total_transfers")
    if summary["counts"]["request_rows"] != case.expected_requests:
        raise AssertionError(
            f"{role} {case.case_id}: request row count {summary['counts']['request_rows']} != expected {case.expected_requests}"
        )
    if summary["counts"]["total_requests"] != case.expected_requests:
        raise AssertionError(
            f"{role} {case.case_id}: total_requests {summary['counts']['total_requests']} != expected {case.expected_requests}"
        )
    if summary["counts"]["completed_requests"] != case.expected_requests:
        raise AssertionError(
            f"{role} {case.case_id}: completed_requests {summary['counts']['completed_requests']} != expected {case.expected_requests}"
        )
    if transfer_count != case.expected_transfers:
        raise AssertionError(
            f"{role} {case.case_id}: total_transfers {transfer_count} != expected {case.expected_transfers}"
        )
    _write_json(role_dir / "summary.normalized.json", summary)
    return summary


def run_case_pair(
    case: ScriptCase,
    candidate_root: Path,
    reference_root: Path,
    output_root: Path,
    python_bin: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    wrapper_path = output_root / "_bin" / "python_reference_frontier.sh"
    _write_reference_wrapper(wrapper_path, python_bin, reference_root)

    candidate_summary = _run_script_role(
        case=case,
        candidate_root=candidate_root,
        output_root=output_root,
        role="candidate",
        python_bin=python_bin,
        timeout_seconds=timeout_seconds,
    )
    reference_summary = _run_script_role(
        case=case,
        candidate_root=candidate_root,
        output_root=output_root,
        role="reference",
        python_bin=wrapper_path,
        timeout_seconds=timeout_seconds,
    )
    diff = compare_summaries(candidate_summary, reference_summary)
    _write_json(output_root / case.case_id / "diff.json", diff)
    return diff


def _select_cases(case_filter: str | None) -> list[ScriptCase]:
    if not case_filter:
        return list(SCRIPT_CASES)
    selected = [case for case in SCRIPT_CASES if case_filter in case.case_id or case_filter in case.script_relpath]
    if not selected:
        supported = ", ".join(case.case_id for case in SCRIPT_CASES)
        raise ValueError(f"case filter matched no cases: {case_filter}. Supported cases: {supported}")
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=Path.cwd())
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-filter", default=None)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_root = args.candidate_root.resolve()
    reference_root = args.reference_root.resolve()
    output_root = args.output_root.resolve()
    python_bin = args.python_bin.resolve()

    if not (candidate_root / "frontier" / "main.py").is_file():
        raise FileNotFoundError(f"Candidate root does not look like Frontier: {candidate_root}")
    if not (reference_root / "frontier" / "main.py").is_file():
        raise FileNotFoundError(f"Reference root does not look like Frontier: {reference_root}")
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python binary does not exist: {python_bin}")

    output_root.mkdir(parents=True, exist_ok=True)
    selected_cases = _select_cases(args.case_filter)
    results: list[dict[str, Any]] = []
    for case in selected_cases:
        print(f"[PDD-CROSS] Running {case.case_id} ({case.script_relpath})")
        try:
            result = run_case_pair(
                case=case,
                candidate_root=candidate_root,
                reference_root=reference_root,
                output_root=output_root,
                python_bin=python_bin,
                timeout_seconds=args.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - runner must preserve per-case failure artifacts.
            result = {
                "case_id": case.case_id,
                "status": "ERROR",
                "mismatch_count": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_json(output_root / case.case_id / "diff.json", result)
        results.append(result)
        print(f"[PDD-CROSS] {case.case_id}: {result['status']}")

    failed_results = [result for result in results if result.get("status") != "PASS"]
    aggregate = {
        "status": "PASS" if not failed_results else "FAIL",
        "candidate_root": str(candidate_root),
        "reference_root": str(reference_root),
        "output_root": str(output_root),
        "case_count": len(results),
        "failed_case_count": len(failed_results),
        "results": results,
    }
    _write_json(output_root / "aggregate_report.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0 if aggregate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

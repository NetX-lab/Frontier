"""Unit tests for the wall-clock scaling result aggregator and plotter."""

from __future__ import annotations

import csv
import json
import signal
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pytest

matplotlib.use("Agg")

from tests.performance.sim_walltime_scaling import plot_scaling as plot_module
from tests.performance.sim_walltime_scaling.run_case import (
    REQUIRED_RESULT_FIELDS,
    SCHEMA_VERSION,
    CaseSpec,
)


SCALES = (32, 64, 128, 256, 512, 1024, 4096)


def _case(model: str, total_gpus: int, attempt_index: int) -> CaseSpec:
    from tests.performance.sim_walltime_scaling.sweep import build_cases

    return build_cases(model, (total_gpus,), "sequential", None)[0].__class__(
        **{
            **build_cases(model, (total_gpus,), "sequential", None)[0].to_dict(),
            "attempt_index": attempt_index,
            "shape": build_cases(model, (total_gpus,), "sequential", None)[0].shape,
        }
    )


def _result(
    model: str,
    total_gpus: int,
    attempt_index: int = 0,
    *,
    status: str = "success",
    sim_wallclock_s: float | None = 12.5,
) -> dict[str, Any]:
    case = _case(model, total_gpus, attempt_index)
    if status != "success":
        sim_wallclock_s = None
    signal_value = {
        "timeout": signal.SIGTERM,
        "host-oom": signal.SIGKILL,
    }.get(status)
    exit_code = {
        "success": 0,
        "simulated-oom": 2,
        "timeout": -signal.SIGTERM,
        "host-oom": 137,
        "bug": 1,
    }[status]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "attempt_id": case.attempt_id,
        "attempt_index": attempt_index,
        "case_fingerprint": case.case_fingerprint,
        "git_sha": "a" * 40,
        "python_executable": sys.executable,
        "seed": case.seed,
        "model": model,
        "model_name": case.model_name,
        "total_gpus": total_gpus,
        "simulation_mode": case.simulation_mode,
        "shape": case.to_dict()["shape"],
        "replicas_per_cluster": case.replicas_per_cluster,
        "mode": case.mode,
        "host": "plot-test-host",
        "worker_job_id": "worker-1",
        "status": status,
        "sim_wallclock_s": sim_wallclock_s,
        "init_s": 0.5,
        "total_proc_s": 13.0,
        "peak_rss_mb": 256.0,
        "requests": case.num_requests,
        "qps": case.qps,
        "prefill_tokens": case.prefill_tokens,
        "decode_tokens": case.decode_tokens,
        "expected_requests": case.num_requests,
        "completed_requests": case.num_requests if status == "success" else None,
        "event_count": 125 if status == "success" else None,
        "events_per_s": 10.0 if status == "success" else None,
        "command": [sys.executable, "run_case.py", "--seed", "42"],
        "started_at": "2026-08-06T00:00:00+00:00",
        "completed_at": "2026-08-06T00:00:13+00:00",
        "exit_code": exit_code,
        "signal": signal_value,
        "failure_reason": (
            None
            if status == "success"
            else {
                "timeout": "parent_timeout",
                "host-oom": "parent_host_oom",
            }.get(status, f"test_{status}")
        ),
        "oom_evidence": (
            {"source": "stderr", "reason": "test"}
            if status in {"simulated-oom", "host-oom"}
            else None
        ),
        "notes": {"fixture": True, "attempt": attempt_index},
        "stderr_tail": "",
        "stdout_tail": "",
    }
    assert REQUIRED_RESULT_FIELDS <= payload.keys()
    return payload


def _write_result(results_dir: Path, payload: dict[str, Any], name: str | None = None) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / (name or f"{payload['attempt_id']}.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _complete_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("dense", "moe"):
        for index, scale in enumerate(SCALES):
            status = "success"
            wallclock = 2.0 + index if model == "dense" else 3.0 + index
            if model == "dense" and scale == 1024:
                status, wallclock = "timeout", None
            if model == "moe" and scale == 512:
                status, wallclock = "simulated-oom", None
            rows.append(
                _result(
                    model,
                    scale,
                    status=status,
                    sim_wallclock_s=wallclock,
                )
            )
    return rows


def test_load_results_accepts_terminal_attempts_and_ignores_case_json(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    success = _result("dense", 32)
    timeout = _result("moe", 32, status="timeout")
    _write_result(results_dir, success)
    _write_result(results_dir, timeout)
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "not-an-attempt.json").write_text("{not json", encoding="utf-8")

    rows = plot_module.load_results(results_dir)

    assert [row["attempt_id"] for row in rows] == [
        success["attempt_id"],
        timeout["attempt_id"],
    ]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update({"status": "unknown"}),
        lambda payload: payload.pop("event_count"),
        lambda payload: payload.update({"sim_wallclock_s": 0.0}),
        lambda payload: payload.update({"model_name": "wrong-model"}),
    ],
)
def test_load_results_rejects_corrupt_incomplete_or_mismatched_rows(
    tmp_path: Path, mutator: Any
) -> None:
    results_dir = tmp_path / "results"
    payload = _result("dense", 32)
    mutator(payload)
    _write_result(results_dir, payload)

    with pytest.raises((ValueError, json.JSONDecodeError)):
        plot_module.load_results(results_dir)


def test_load_results_rejects_corrupt_json_and_duplicate_attempt_ids(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    first = _result("dense", 32)
    second = _result("moe", 32)
    second["attempt_id"] = first["attempt_id"]
    _write_result(results_dir, first, "first.json")
    _write_result(results_dir, second, "second.json")

    with pytest.raises(ValueError, match="duplicate.*attempt_id"):
        plot_module.load_results(results_dir)

    corrupt_dir = tmp_path / "corrupt"
    _write_result(corrupt_dir, first)
    (corrupt_dir / "broken.json").write_text("{broken", encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        plot_module.load_results(corrupt_dir)


def test_summary_selection_uses_first_success_or_final_terminal_attempt() -> None:
    rows = [
        _result("dense", 32, 0, status="bug"),
        _result("dense", 32, 1, status="success", sim_wallclock_s=7.0),
        _result("dense", 32, 2, status="success", sim_wallclock_s=8.0),
        _result("moe", 32, 0, status="bug"),
        _result("moe", 32, 2, status="timeout"),
    ]

    selected = plot_module.select_summary_rows(rows)

    assert [(row["model"], row["total_gpus"], row["attempt_index"]) for row in selected] == [
        ("dense", 32, 1),
        ("moe", 32, 2),
    ]


def test_csv_serializes_nested_values_deterministically_and_keeps_numbers(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    row = _result("dense", 32)

    plot_module.write_summary_csv([row], path)

    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert records[0]["total_gpus"] == "32"
    assert records[0]["sim_wallclock_s"] == "12.5"
    assert records[0]["shape"] == json.dumps(row["shape"], sort_keys=True, separators=(",", ":"))
    assert records[0]["command"] == json.dumps(row["command"], sort_keys=True, separators=(",", ":"))
    assert records[0]["oom_evidence"] == ""
    assert "model_name" in records[0]
    assert "events_per_s" in records[0]


def test_formal_completeness_requires_both_models_at_all_scales() -> None:
    rows = _complete_rows()
    missing = [row for row in rows if not (row["model"] == "moe" and row["total_gpus"] == 4096)]

    with pytest.raises(ValueError, match="completeness"):
        plot_module.validate_formal_completeness(missing)
    assert plot_module.validate_formal_completeness(rows) is None


def test_plot_has_two_log_subplots_status_band_and_annotations(tmp_path: Path) -> None:
    png_path = tmp_path / "nested" / "scaling.png"
    pdf_path = tmp_path / "nested" / "scaling.pdf"

    figure = plot_module.plot_scaling(_complete_rows(), png_path, pdf_path)

    assert len(figure.axes) == 2
    dense_ax, moe_ax = figure.axes
    assert "dense" in dense_ax.get_title().lower()
    assert "moe" in moe_ax.get_title().lower()
    assert all(ax.get_xscale() == "log" for ax in figure.axes)
    assert all(ax.get_yscale() == "log" for ax in figure.axes)
    assert list(dense_ax.get_xticks()) == list(SCALES)
    assert list(moe_ax.get_xticks()) == list(SCALES)
    all_text = " ".join(text.get_text() for ax in figure.axes for text in ax.texts)
    assert "sequential" in all_text.lower()
    assert ">= 14400 s" in all_text
    assert any("2 s" in text.get_text() for text in dense_ax.texts)
    assert any("simulated-oom" in text.get_text() for text in moe_ax.texts)
    for ax in figure.axes:
        assert any(artist.get_transform() == ax.get_xaxis_transform() for artist in ax.lines + ax.collections)

    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF-")


def test_pdf_export_is_byte_stable_for_identical_rows(tmp_path: Path) -> None:
    rows = _complete_rows()
    first_png = tmp_path / "first.png"
    first_pdf = tmp_path / "first.pdf"
    second_png = tmp_path / "second.png"
    second_pdf = tmp_path / "second.pdf"

    first_figure = plot_module.plot_scaling(rows, first_png, first_pdf)
    matplotlib.pyplot.close(first_figure)
    second_figure = plot_module.plot_scaling(rows, second_png, second_pdf)
    matplotlib.pyplot.close(second_figure)

    assert first_pdf.read_bytes() == second_pdf.read_bytes()


def test_plot_marks_a_nonselected_bug_attempt_alongside_success(tmp_path: Path) -> None:
    rows = [
        _result("moe", 32, 0, status="bug"),
        _result("moe", 32, 1, status="success", sim_wallclock_s=72.0),
    ]

    figure = plot_module.plot_scaling(
        rows,
        tmp_path / "scaling.png",
        tmp_path / "scaling.pdf",
    )

    all_text = " ".join(
        text.get_text() for axis in figure.axes for text in axis.texts
    )
    assert "bug" in all_text


def test_cli_passes_nonselected_failures_to_plotter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    rows = _complete_rows()
    rows.append(_result("moe", 32, 2, status="bug"))
    for row in rows:
        _write_result(results_dir, row)

    observed_statuses: list[str] = []
    original_plotter = plot_module.plot_scaling

    def spy_plotter(rows: Any, png_path: Path, pdf_path: Path) -> Any:
        observed_statuses.extend(row["status"] for row in rows)
        return original_plotter(rows, png_path, pdf_path)

    monkeypatch.setattr(plot_module, "plot_scaling", spy_plotter)
    assert plot_module.main(["--results-dir", str(results_dir)]) == 0
    assert "bug" in observed_statuses


def test_cli_writes_all_four_artifacts_after_formal_validation(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for row in _complete_rows():
        _write_result(results_dir, row)

    assert plot_module.main(["--results-dir", str(results_dir)]) == 0
    for name in (
        "attempts.csv",
        "summary.csv",
        "sim_walltime_scaling.png",
        "sim_walltime_scaling.pdf",
    ):
        artifact = results_dir / name
        assert artifact.exists() and artifact.stat().st_size > 0

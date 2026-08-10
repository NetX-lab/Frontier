"""Tests for paired sequential/parallel wall-clock result analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.performance.sim_walltime_scaling import compare_parallel_modes


def _result(
    *,
    repetition: int,
    mode: str,
    sim_wallclock_s: float,
    total_proc_s: float,
    event_count: int = 100,
    status: str = "success",
    completed_requests: int = 8,
    seed: int = 42,
) -> dict:
    return {
        "schema_version": 2,
        "runner_sha256": "runner-sha256",
        "attempt_index": repetition,
        "seed": seed,
        "git_sha": "abc123",
        "python_executable": "/opt/frontier/bin/python",
        "host": "benchmark-host",
        "model": "moe",
        "total_gpus": 32,
        "simulation_mode": "online",
        "shape": {
            "attn_tp": 4,
            "attn_dp": 2,
            "moe_tp": 1,
            "moe_ep": 8,
            "pp": 2,
        },
        "replicas_per_cluster": 1,
        "mode": mode,
        "effective_parallel_mode": mode == "parallel",
        "status": status,
        "sim_wallclock_s": sim_wallclock_s,
        "init_s": 2.0,
        "total_proc_s": total_proc_s,
        "requests": 8,
        "qps": 8.0,
        "prefill_tokens": 16,
        "decode_tokens": 2,
        "expected_requests": 8,
        "completed_requests": completed_requests,
        "event_count": event_count,
        "command": [
            "/opt/frontier/bin/python",
            "-m",
            "frontier.main",
            (
                "--enable_parallel_clusters"
                if mode == "parallel"
                else "--no-enable_parallel_clusters"
            ),
            "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
            "1.0",
            "--metrics_config_output_dir",
            "/data/ycfeng/tmp/test-simulator-configs",
            "--metrics_config_run_id",
            f"{mode}-attempt-{repetition:02d}",
        ],
    }


def _three_pairs() -> list[dict]:
    return [
        _result(
            repetition=repetition,
            mode=mode,
            sim_wallclock_s=1.0 if mode == "sequential" else 2.0,
            total_proc_s=3.0 if mode == "sequential" else 4.0,
        )
        for repetition in range(3)
        for mode in ("sequential", "parallel")
    ]


def test_summarize_paired_results_reports_center_variability_and_speedup() -> None:
    records = []
    for repetition, sequential_s, parallel_s in (
        (0, 10.0, 20.0),
        (1, 12.0, 24.0),
        (2, 14.0, 28.0),
    ):
        records.extend(
            [
                _result(
                    repetition=repetition,
                    mode="sequential",
                    sim_wallclock_s=sequential_s,
                    total_proc_s=sequential_s + 2.0,
                ),
                _result(
                    repetition=repetition,
                    mode="parallel",
                    sim_wallclock_s=parallel_s,
                    total_proc_s=parallel_s + 2.0,
                ),
            ]
        )

    summaries = compare_parallel_modes.summarize_paired_results(records)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["repetitions"] == 3
    assert summary["event_count"] == 100
    assert summary["sim_wallclock_s"]["sequential"]["median"] == 12.0
    assert summary["sim_wallclock_s"]["sequential"]["min"] == 10.0
    assert summary["sim_wallclock_s"]["sequential"]["max"] == 14.0
    assert summary["sim_wallclock_s"]["parallel"]["median"] == 24.0
    assert summary["sim_wallclock_s"]["paired_speedup_median"] == 0.5
    assert summary["sim_wallclock_s"]["parallel_slowdown_pct_median"] == 100.0
    assert summary["sim_wallclock_s"]["sequential"]["cv"] == pytest.approx(
        1.0 / 6.0
    )
    assert summary["total_proc_s"]["paired_speedup_median"] == pytest.approx(
        14.0 / 26.0
    )


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                _result(
                    repetition=0,
                    mode="sequential",
                    sim_wallclock_s=1.0,
                    total_proc_s=3.0,
                )
            ],
            "exactly one sequential and one parallel result",
        ),
        (
            [
                _result(
                    repetition=0,
                    mode="sequential",
                    sim_wallclock_s=1.0,
                    total_proc_s=3.0,
                ),
                _result(
                    repetition=0,
                    mode="parallel",
                    sim_wallclock_s=2.0,
                    total_proc_s=4.0,
                    event_count=101,
                ),
            ],
            "event_count mismatch",
        ),
        (
            [
                _result(
                    repetition=0,
                    mode="sequential",
                    sim_wallclock_s=1.0,
                    total_proc_s=3.0,
                ),
                _result(
                    repetition=0,
                    mode="parallel",
                    sim_wallclock_s=2.0,
                    total_proc_s=4.0,
                    completed_requests=7,
                ),
            ],
            "incomplete result",
        ),
    ],
)
def test_summarize_paired_results_rejects_invalid_comparisons(
    records: list[dict],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_parallel_modes.summarize_paired_results(records)


def test_summarize_paired_results_does_not_pair_different_seeds() -> None:
    records = [
        _result(
            repetition=0,
            mode="sequential",
            sim_wallclock_s=1.0,
            total_proc_s=3.0,
            seed=42,
        ),
        _result(
            repetition=0,
            mode="parallel",
            sim_wallclock_s=2.0,
            total_proc_s=4.0,
            seed=43,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="exactly one sequential and one parallel result",
    ):
        compare_parallel_modes.summarize_paired_results(records)


def test_summarize_paired_results_rejects_missing_request_counts() -> None:
    records = [
        _result(
            repetition=0,
            mode=mode,
            sim_wallclock_s=1.0 if mode == "sequential" else 2.0,
            total_proc_s=3.0 if mode == "sequential" else 4.0,
        )
        for mode in ("sequential", "parallel")
    ]
    for record in records:
        record["expected_requests"] = None
        record["completed_requests"] = None

    with pytest.raises(ValueError, match="request counts must be positive integers"):
        compare_parallel_modes.summarize_paired_results(records)


def test_summarize_paired_results_rejects_empty_result_set() -> None:
    with pytest.raises(ValueError, match="at least one workload"):
        compare_parallel_modes.summarize_paired_results([])


def test_summarize_paired_results_requires_declared_request_count_to_match() -> None:
    records = []
    for repetition in range(3):
        for mode in ("sequential", "parallel"):
            record = _result(
                repetition=repetition,
                mode=mode,
                sim_wallclock_s=1.0 if mode == "sequential" else 2.0,
                total_proc_s=3.0 if mode == "sequential" else 4.0,
            )
            record["requests"] = 9
            records.append(record)

    with pytest.raises(ValueError, match="request counts must match"):
        compare_parallel_modes.summarize_paired_results(records)


def test_summarize_paired_results_rejects_schema_revision_mismatch() -> None:
    records = _three_pairs()
    for record in records:
        if record["mode"] == "parallel":
            record["schema_version"] = 1

    with pytest.raises(
        ValueError,
        match="exactly one sequential and one parallel result",
    ):
        compare_parallel_modes.summarize_paired_results(records)


def test_summarize_paired_results_rejects_effective_mode_mismatch() -> None:
    records = _three_pairs()
    for record in records:
        if record["mode"] == "parallel":
            record["effective_parallel_mode"] = False

    with pytest.raises(ValueError, match="effective parallel mode mismatch"):
        compare_parallel_modes.summarize_paired_results(records)


def test_summarize_paired_results_rejects_runtime_command_mismatch() -> None:
    records = _three_pairs()
    for record in records:
        if record["mode"] == "parallel":
            dummy_time_index = record["command"].index("1.0")
            record["command"][dummy_time_index] = "2.0"

    with pytest.raises(ValueError, match="runtime command mismatch"):
        compare_parallel_modes.summarize_paired_results(records)


def test_summarize_paired_results_requires_three_repetitions() -> None:
    records = [
        _result(
            repetition=0,
            mode=mode,
            sim_wallclock_s=1.0 if mode == "sequential" else 2.0,
            total_proc_s=3.0 if mode == "sequential" else 4.0,
        )
        for mode in ("sequential", "parallel")
    ]

    with pytest.raises(ValueError, match="at least 3 paired repetitions"):
        compare_parallel_modes.summarize_paired_results(records)


def test_slowdown_median_is_computed_from_paired_slowdowns() -> None:
    records = []
    for repetition, parallel_s in enumerate((1.0, 2.0, 4.0, 8.0)):
        records.extend(
            [
                _result(
                    repetition=repetition,
                    mode="sequential",
                    sim_wallclock_s=1.0,
                    total_proc_s=3.0,
                ),
                _result(
                    repetition=repetition,
                    mode="parallel",
                    sim_wallclock_s=parallel_s,
                    total_proc_s=parallel_s + 2.0,
                ),
            ]
        )

    summary = compare_parallel_modes.summarize_paired_results(records)[0]

    assert summary["sim_wallclock_s"]["paired_speedup_median"] == 0.375
    assert summary["sim_wallclock_s"]["parallel_slowdown_pct_median"] == 200.0


def test_summarize_paired_results_does_not_pair_different_git_revisions() -> None:
    records = [
        _result(
            repetition=0,
            mode="sequential",
            sim_wallclock_s=1.0,
            total_proc_s=3.0,
        ),
        _result(
            repetition=0,
            mode="parallel",
            sim_wallclock_s=2.0,
            total_proc_s=4.0,
        ),
    ]
    records[1]["git_sha"] = "different"

    with pytest.raises(
        ValueError,
        match="exactly one sequential and one parallel result",
    ):
        compare_parallel_modes.summarize_paired_results(records)


def test_load_result_files_uses_only_direct_json_children(tmp_path) -> None:
    direct = tmp_path / "direct.json"
    direct.write_text("{}\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("{}\n", encoding="utf-8")

    assert compare_parallel_modes.load_result_files(tmp_path) == [{}]


def test_load_result_files_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(ValueError, match="results directory does not exist"):
        compare_parallel_modes.load_result_files(missing)


def test_summarize_paired_results_rejects_non_success_status() -> None:
    records = _three_pairs()
    records[0]["status"] = "bug"

    with pytest.raises(ValueError, match="comparison requires success results"):
        compare_parallel_modes.summarize_paired_results(records)


def test_summarize_paired_results_rejects_non_integer_attempt_index() -> None:
    records = _three_pairs()
    records[0]["attempt_index"] = "0"

    with pytest.raises(ValueError, match="attempt_index must be an integer"):
        compare_parallel_modes.summarize_paired_results(records)


def test_load_result_files_rejects_non_object_json(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Result file must contain an object"):
        compare_parallel_modes.load_result_files(tmp_path)


def test_main_writes_summary_json(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for index, record in enumerate(_three_pairs()):
        (results_dir / f"result-{index}.json").write_text(
            json.dumps(record),
            encoding="utf-8",
        )
    output_path = tmp_path / "summary.json"

    exit_code = compare_parallel_modes.main(
        [str(results_dir), "--output-json", str(output_path)]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["repetitions"] == 3

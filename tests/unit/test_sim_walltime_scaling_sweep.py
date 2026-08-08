from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.performance.sim_walltime_scaling.run_case import (
    PRIMARY_SHAPES,
    REQUIRED_RESULT_FIELDS,
    SCHEMA_VERSION,
    CaseSpec,
)


SWEEP_MODULE = "tests.performance.sim_walltime_scaling.sweep"


def _load_sweep() -> Any:
    try:
        return importlib.import_module(SWEEP_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == SWEEP_MODULE:
            pytest.fail(f"Required sweep module is missing: {SWEEP_MODULE}")
        raise


def _case_path(output_root: Path, case: CaseSpec) -> Path:
    return output_root / "cases" / f"{case.attempt_id}.json"


def _result_path(output_root: Path, case: CaseSpec) -> Path:
    return output_root / "results" / f"{case.attempt_id}.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _terminal_result(case: CaseSpec, status: str) -> dict[str, Any]:
    success = status == "success"
    exit_codes = {
        "success": 0,
        "simulated-oom": 2,
        "timeout": -signal.SIGTERM,
        "host-oom": 137,
        "bug": 1,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "attempt_id": case.attempt_id,
        "attempt_index": case.attempt_index,
        "case_fingerprint": case.case_fingerprint,
        "git_sha": "a" * 40,
        "python_executable": sys.executable,
        "seed": case.seed,
        "model": case.model,
        "model_name": case.model_name,
        "total_gpus": case.total_gpus,
        "simulation_mode": case.simulation_mode,
        "shape": case.to_dict()["shape"],
        "replicas_per_cluster": case.replicas_per_cluster,
        "mode": case.mode,
        "host": "test-host",
        "worker_job_id": None,
        "status": status,
        "sim_wallclock_s": 2.0 if success else None,
        "init_s": 0.25,
        "total_proc_s": 2.25,
        "peak_rss_mb": 128.0,
        "requests": case.num_requests,
        "qps": case.qps,
        "prefill_tokens": case.prefill_tokens,
        "decode_tokens": case.decode_tokens,
        "expected_requests": case.num_requests,
        "completed_requests": case.num_requests if success else None,
        "event_count": 20 if success else None,
        "events_per_s": 10.0 if success else None,
        "command": [sys.executable, "run_case.py"],
        "started_at": "2026-08-06T00:00:00+00:00",
        "completed_at": "2026-08-06T00:00:02+00:00",
        "exit_code": exit_codes[status],
        "signal": (
            signal.SIGTERM
            if status == "timeout"
            else signal.SIGKILL if status == "host-oom" else None
        ),
        "failure_reason": (
            None
            if success
            else {
                "timeout": "parent_timeout",
                "host-oom": "parent_host_oom",
            }.get(status, f"test_{status}")
        ),
        "oom_evidence": (
            {"reason": "test"}
            if status in {"simulated-oom", "host-oom"}
            else None
        ),
        "notes": [],
        "stderr_tail": "",
    }


def _case_and_result_from_command(command: list[str]) -> tuple[CaseSpec, Path]:
    case_path = Path(command[command.index("--case-json") + 1])
    result_path = Path(command[command.index("--result-json") + 1])
    case = CaseSpec.from_dict(json.loads(case_path.read_text()))
    return case, result_path


@pytest.mark.parametrize(
    ("model", "expected_shape", "expected_replicas"),
    [
        (
            "dense",
            PRIMARY_SHAPES["dense"],
            (2, 4, 8, 16, 32, 64, 256),
        ),
        (
            "moe",
            PRIMARY_SHAPES["moe"],
            (1, 2, 4, 8, 16, 32, 128),
        ),
    ],
)
def test_build_cases_has_exact_seven_point_weak_scaling_matrix(
    model: str,
    expected_shape: dict[str, int],
    expected_replicas: tuple[int, ...],
) -> None:
    sweep = _load_sweep()

    cases = sweep.build_cases(
        model,
        sweep.DEFAULT_SCALES,
        "sequential",
        shape_overrides=None,
    )

    assert sweep.DEFAULT_SCALES == (32, 64, 128, 256, 512, 1024, 4096)
    assert [case.total_gpus for case in cases] == list(sweep.DEFAULT_SCALES)
    assert [case.replicas_per_cluster for case in cases] == list(
        expected_replicas
    )
    assert all(case.shape == cases[0].shape for case in cases)
    for case in cases:
        actual_shape = {
            "attn_tp": case.shape.attn_tp,
            "attn_dp": case.shape.attn_dp,
            "moe_tp": case.shape.moe_tp,
            "moe_ep": case.shape.moe_ep,
            "pp": case.shape.pp,
        }
        assert actual_shape == expected_shape
        assert case.simulation_mode == "online"
        assert case.mode == "sequential"
        assert case.attempt_index == 0
        assert case.num_requests == 2 * case.total_gpus
        assert case.qps == 0.25 * case.total_gpus
        assert case.prefill_tokens == 512
        assert case.decode_tokens == 128
        assert case.seed == 42


def test_build_cases_applies_one_fixed_shape_override_to_every_scale() -> None:
    sweep = _load_sweep()

    cases = sweep.build_cases(
        "dense",
        (32, 64),
        "sequential",
        shape_overrides={"attn_tp": 2, "pp": 4},
    )

    assert [(case.shape.attn_tp, case.shape.pp) for case in cases] == [
        (2, 4),
        (2, 4),
    ]
    assert cases[0].shape == cases[1].shape


def test_build_cases_propagates_attempt_index_to_every_scale() -> None:
    sweep = _load_sweep()

    cases = sweep.build_cases(
        "moe",
        (32, 64),
        "sequential",
        shape_overrides={"pp": 1},
        attempt_index=3,
    )

    assert [case.attempt_index for case in cases] == [3, 3]


def test_build_cases_rejects_negative_attempt_index() -> None:
    sweep = _load_sweep()

    with pytest.raises(ValueError, match="attempt_index"):
        sweep.build_cases(
            "dense",
            (32,),
            "sequential",
            shape_overrides=None,
            attempt_index=-1,
        )


def test_run_sweep_uses_requested_bounded_concurrency(tmp_path: Path) -> None:
    sweep = _load_sweep()
    cases = sweep.build_cases(
        "dense",
        (32, 64, 128, 256),
        "sequential",
        shape_overrides=None,
    )
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=2.0)
    active = 0
    maximum_active = 0

    def runner(command: list[str], timeout_s: float) -> Any:
        nonlocal active, maximum_active
        assert timeout_s == 17.0
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            barrier.wait()
            case, result_path = _case_and_result_from_command(command)
            _write_json(result_path, _terminal_result(case, "success"))
            return sweep.ChildOutcome(
                returncode=0,
                elapsed_s=0.01,
                stdout_tail="",
                stderr_tail="",
                timed_out=False,
            )
        finally:
            with lock:
                active -= 1

    paths = sweep.run_sweep(
        cases,
        tmp_path,
        max_concurrency=2,
        timeout_s=17.0,
        attempt_runner=runner,
    )

    assert maximum_active == 2
    assert paths == [_result_path(tmp_path, case) for case in cases]
    assert all(path.exists() for path in paths)


def test_relative_output_root_is_resolved_before_child_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep = _load_sweep()
    monkeypatch.chdir(tmp_path)
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]
    observed: dict[str, Any] = {}

    def runner(command: list[str], _timeout_s: float) -> Any:
        observed["command"] = command
        return sweep.ChildOutcome(
            returncode=137,
            elapsed_s=0.01,
            stdout_tail="",
            stderr_tail="",
            timed_out=False,
        )

    [result_path] = sweep.run_sweep(
        [case],
        Path("relative-output"),
        max_concurrency=1,
        attempt_runner=runner,
    )

    command = observed["command"]
    assert Path(command[3]).is_absolute()
    assert Path(command[5]).is_absolute()
    assert result_path.is_absolute()
    assert result_path == (tmp_path / "relative-output" / "results" / f"{case.attempt_id}.json").resolve()


def test_falsey_callable_injections_are_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]

    class FalseyAttemptRunner:
        called = False

        def __bool__(self) -> bool:
            return False

        def __call__(self, command: list[str], _timeout_s: float) -> Any:
            self.called = True
            parsed_case, result_path = _case_and_result_from_command(command)
            _write_json(result_path, _terminal_result(parsed_case, "success"))
            return sweep.ChildOutcome(
                returncode=0,
                elapsed_s=0.01,
                stdout_tail="",
                stderr_tail="",
                timed_out=False,
            )

    attempt_runner = FalseyAttemptRunner()
    monkeypatch.setattr(
        sweep,
        "_execute_child",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default attempt runner was selected")
        ),
    )
    sweep.run_sweep(
        [case],
        tmp_path / "attempts",
        max_concurrency=1,
        attempt_runner=attempt_runner,
    )
    assert attempt_runner.called is True

    class FalseySweepRunner:
        called = False

        def __bool__(self) -> bool:
            return False

        def __call__(self, *_args: Any, **_kwargs: Any) -> list[Path]:
            self.called = True
            return []

    sweep_runner = FalseySweepRunner()
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default sweep runner was selected")
        ),
    )
    assert (
        sweep.main(
            ["--model", "dense", "--scales", "32", "--output-root", str(tmp_path / "cli")],
            sweep_runner=sweep_runner,
        )
        == 0
    )
    assert sweep_runner.called is True


@pytest.mark.parametrize(
    "status", ["success", "simulated-oom", "timeout", "host-oom", "bug"]
)
def test_matching_valid_existing_terminal_result_is_skipped_without_launch(
    tmp_path: Path,
    status: str,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]
    result_path = _result_path(tmp_path, case)
    _write_json(result_path, _terminal_result(case, status))
    original = result_path.read_bytes()

    def unexpected_runner(_command: list[str], _timeout_s: float) -> Any:
        raise AssertionError("A valid immutable result must not be launched again")

    paths = sweep.run_sweep(
        [case], tmp_path, max_concurrency=1, attempt_runner=unexpected_runner
    )

    assert paths == [result_path]
    assert result_path.read_bytes() == original


@pytest.mark.parametrize(
    ("status", "field", "invalid_value"),
    [
        ("timeout", "failure_reason", "wrong_timeout_reason"),
        ("timeout", "sim_wallclock_s", 1.0),
        ("timeout", "total_proc_s", 0.0),
        ("timeout", "exit_code", None),
        ("timeout", "signal", signal.SIGKILL),
        ("host-oom", "failure_reason", "wrong_host_oom_reason"),
        ("host-oom", "sim_wallclock_s", 1.0),
        ("host-oom", "oom_evidence", None),
        ("host-oom", "exit_code", 1),
        ("host-oom", "exit_code", []),
        ("host-oom", "signal", None),
        ("host-oom", "total_proc_s", 0.0),
    ],
)
def test_invalid_parent_terminal_contract_is_not_resumed(
    tmp_path: Path,
    status: str,
    field: str,
    invalid_value: Any,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]
    result_path = _result_path(tmp_path, case)
    payload = _terminal_result(case, status)
    payload[field] = invalid_value
    _write_json(result_path, payload)
    launched = False

    def unexpected_runner(_command: list[str], _timeout_s: float) -> Any:
        nonlocal launched
        launched = True
        raise AssertionError("Invalid parent terminal record must not launch")

    with pytest.raises(ValueError, match=status):
        sweep.run_sweep(
            [case], tmp_path, max_concurrency=1, attempt_runner=unexpected_runner
        )

    assert launched is False
    assert result_path.read_bytes() == (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode()


@pytest.mark.parametrize(
    "existing_kind",
    ["corrupt", "incomplete", "fingerprint-mismatch", "identity-mismatch"],
)
def test_invalid_existing_result_fails_fast_without_launch(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]
    result_path = _result_path(tmp_path, case)
    result_path.parent.mkdir(parents=True)
    if existing_kind == "corrupt":
        result_path.write_text("{not-json\n")
    elif existing_kind == "incomplete":
        _write_json(result_path, {})
    else:
        payload = _terminal_result(case, "success")
        if existing_kind == "fingerprint-mismatch":
            payload["case_fingerprint"] = "0" * 64
        else:
            payload["total_gpus"] = 64
        _write_json(result_path, payload)
    launched = False

    def unexpected_runner(_command: list[str], _timeout_s: float) -> Any:
        nonlocal launched
        launched = True
        raise AssertionError("Invalid resume state must fail before launch")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        sweep.run_sweep(
            [case], tmp_path, max_concurrency=1, attempt_runner=unexpected_runner
        )

    assert launched is False


def test_case_json_is_reused_only_when_it_exactly_matches(tmp_path: Path) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]
    case_path = _case_path(tmp_path, case)
    mismatched = case.to_dict()
    mismatched["qps"] = case.qps + 1.0
    _write_json(case_path, mismatched)

    with pytest.raises(ValueError, match="CaseSpec"):
        sweep.run_sweep([case], tmp_path, max_concurrency=1)

    assert not _result_path(tmp_path, case).exists()


@pytest.mark.parametrize(
    (
        "leader_exits_after_sigterm",
        "descendant_group_remains",
        "expected_signals",
        "expected_returncode",
    ),
    [
        (True, False, [signal.SIGTERM], -signal.SIGTERM),
        (False, True, [signal.SIGTERM, signal.SIGKILL], -signal.SIGKILL),
        (True, True, [signal.SIGTERM, signal.SIGKILL], -signal.SIGTERM),
    ],
)
def test_execute_child_kills_the_process_group_and_escalates_only_if_needed(
    monkeypatch: pytest.MonkeyPatch,
    leader_exits_after_sigterm: bool,
    descendant_group_remains: bool,
    expected_signals: list[signal.Signals],
    expected_returncode: int,
) -> None:
    sweep = _load_sweep()
    command = [sys.executable, "-c", "pass"]
    timeout_error = subprocess.TimeoutExpired(command, 0.01)

    class FakeProcess:
        pid = 4242

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.responses: list[object] = [timeout_error]
            if not leader_exits_after_sigterm:
                self.responses.append(timeout_error)
            self.responses.append(expected_returncode)

        def wait(self, timeout: float | None = None) -> int:
            response = self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            self.returncode = int(response)
            return self.returncode

    process = FakeProcess()
    popen_kwargs: dict[str, Any] = {}
    signals: list[signal.Signals] = []

    def fake_popen(_command: list[str], **kwargs: Any) -> FakeProcess:
        popen_kwargs.update(kwargs)
        return process

    def fake_killpg(pid: int, sent_signal: signal.Signals) -> None:
        assert pid == process.pid
        if sent_signal == 0:
            if descendant_group_remains:
                return
            raise ProcessLookupError
        signals.append(sent_signal)

    monkeypatch.setattr(sweep.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sweep.os, "killpg", fake_killpg)

    outcome = sweep._execute_child(command, timeout_s=0.01)

    assert outcome.timed_out is True
    assert outcome.returncode == expected_returncode
    assert signals == expected_signals
    assert popen_kwargs["start_new_session"] is True


def test_execute_child_records_only_bounded_output_tails() -> None:
    sweep = _load_sweep()
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.stdout.write('x' * 25000 + 'OUT_END'); "
            "sys.stderr.write('y' * 25000 + 'ERR_END')"
        ),
    ]

    outcome = sweep._execute_child(command, timeout_s=5.0)

    assert outcome.returncode == 0
    assert len(outcome.stdout_tail.encode("utf-8")) <= sweep.OUTPUT_TAIL_BYTES
    assert len(outcome.stderr_tail.encode("utf-8")) <= sweep.OUTPUT_TAIL_BYTES
    assert outcome.stdout_tail.endswith("OUT_END")
    assert outcome.stderr_tail.endswith("ERR_END")


def test_execute_child_uses_explicit_default_temp_root_and_ignores_tmpdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep = _load_sweep()
    real_temporary_file = sweep.tempfile.TemporaryFile
    observed_dirs: list[Path | None] = []

    def recording_temporary_file(*args: Any, **kwargs: Any) -> Any:
        configured_dir = kwargs.get("dir")
        observed_dirs.append(
            Path(configured_dir) if configured_dir is not None else None
        )
        return real_temporary_file(*args, **kwargs)

    monkeypatch.delenv("FRONTIER_WALLTIME_TMPDIR", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "must-not-be-used"))
    monkeypatch.setattr(sweep.tempfile, "TemporaryFile", recording_temporary_file)

    outcome = sweep._execute_child([sys.executable, "-c", "pass"], 5.0)

    assert outcome.returncode == 0
    assert observed_dirs == [Path("/data/ycfeng/tmp")] * 2


def test_execute_child_creates_and_uses_task_specific_temp_root_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep = _load_sweep()
    configured_root = tmp_path / "explicit" / "walltime-temp"
    real_temporary_file = sweep.tempfile.TemporaryFile
    observed_dirs: list[Path | None] = []

    def recording_temporary_file(*args: Any, **kwargs: Any) -> Any:
        configured_dir = kwargs.get("dir")
        observed_dirs.append(
            Path(configured_dir) if configured_dir is not None else None
        )
        return real_temporary_file(*args, **kwargs)

    monkeypatch.setenv("FRONTIER_WALLTIME_TMPDIR", str(configured_root))
    monkeypatch.setattr(sweep.tempfile, "TemporaryFile", recording_temporary_file)

    outcome = sweep._execute_child([sys.executable, "-c", "pass"], 5.0)

    assert outcome.returncode == 0
    assert configured_root.is_dir()
    assert observed_dirs == [configured_root] * 2


@pytest.mark.parametrize("configured_value", ["relative/path", ""])
def test_execute_child_rejects_non_absolute_temp_root(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str,
) -> None:
    sweep = _load_sweep()
    monkeypatch.setenv("FRONTIER_WALLTIME_TMPDIR", configured_value)

    with pytest.raises(ValueError, match="absolute"):
        sweep._execute_child([sys.executable, "-c", "pass"], 5.0)


def test_execute_child_rejects_temp_root_that_is_not_a_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep = _load_sweep()
    configured_file = tmp_path / "not-a-directory"
    configured_file.write_text("occupied")
    monkeypatch.setenv("FRONTIER_WALLTIME_TMPDIR", str(configured_file))

    with pytest.raises(NotADirectoryError, match="directory"):
        sweep._execute_child([sys.executable, "-c", "pass"], 5.0)


def test_temp_root_override_rejects_absolute_path_outside_default_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep = _load_sweep()
    monkeypatch.setenv("FRONTIER_WALLTIME_TMPDIR", "/tmp")

    with pytest.raises(ValueError, match="/data/ycfeng/tmp"):
        sweep._resolve_temp_root()


def test_temp_root_override_rejects_symlink_escape_from_default_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep = _load_sweep()
    escape_link = tmp_path / "escape-to-system-tmp"
    escape_link.symlink_to("/tmp", target_is_directory=True)
    monkeypatch.setenv("FRONTIER_WALLTIME_TMPDIR", str(escape_link))

    with pytest.raises(ValueError, match="/data/ycfeng/tmp"):
        sweep._resolve_temp_root()


def test_timeout_creates_a_complete_parent_owned_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]
    monkeypatch.setenv("FRONTIER_WORKER_JOB_ID", "worker-123")

    def runner(_command: list[str], _timeout_s: float) -> Any:
        return sweep.ChildOutcome(
            returncode=-signal.SIGTERM,
            elapsed_s=0.25,
            stdout_tail="stdout evidence",
            stderr_tail="stderr evidence",
            timed_out=True,
        )

    [result_path] = sweep.run_sweep(
        [case],
        tmp_path,
        max_concurrency=1,
        timeout_s=0.01,
        python_executable="/opt/frontier-python",
        attempt_runner=runner,
    )
    result = json.loads(result_path.read_text())

    assert REQUIRED_RESULT_FIELDS <= result.keys()
    assert result["status"] == "timeout"
    assert result["failure_reason"] == "parent_timeout"
    assert result["sim_wallclock_s"] is None
    assert result["total_proc_s"] == 0.25
    assert result["exit_code"] == -signal.SIGTERM
    assert result["signal"] == signal.SIGTERM
    assert result["stderr_tail"] == "stderr evidence"
    assert result["stdout_tail"] == "stdout evidence"
    assert result["worker_job_id"] == "worker-123"
    assert result["python_executable"] == "/opt/frontier-python"
    assert result["case_fingerprint"] == case.case_fingerprint
    assert result["shape"] == case.to_dict()["shape"]


@pytest.mark.parametrize(
    ("returncode", "stdout_tail", "stderr_tail", "expected_status"),
    [
        (137, "oom-kill: constraint=CONSTRAINT_MEMCG", "", "host-oom"),
        (-9, "", "memory cgroup out of memory", "host-oom"),
        (-9, "", "Out of memory: Killed process 1234 (python)", "host-oom"),
        (
            137,
            "memory.events:\nlow 0\noom 1\noom_kill 1\n",
            "",
            "host-oom",
        ),
        (
            137,
            "memory.events:\nlow 0\noom 0\noom_kill 0\n",
            "",
            "bug",
        ),
        (137, "Killed", "", "bug"),
        (-9, "", "", "bug"),
    ],
)
def test_sigkill_is_host_oom_only_with_explicit_evidence(
    tmp_path: Path,
    returncode: int,
    stdout_tail: str,
    stderr_tail: str,
    expected_status: str,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]

    def runner(_command: list[str], _timeout_s: float) -> Any:
        return sweep.ChildOutcome(
            returncode=returncode,
            elapsed_s=0.1,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            timed_out=False,
        )

    [result_path] = sweep.run_sweep(
        [case], tmp_path, max_concurrency=1, attempt_runner=runner
    )
    result = json.loads(result_path.read_text())

    assert result["status"] == expected_status
    if expected_status == "host-oom":
        assert result["oom_evidence"]
        assert result["failure_reason"] == "parent_host_oom"
    else:
        assert result["oom_evidence"] is None


@pytest.mark.parametrize("returncode", [0, 1, 2, -signal.SIGTERM])
def test_child_exit_without_a_result_is_recorded_as_bug(
    tmp_path: Path,
    returncode: int,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]

    def runner(_command: list[str], _timeout_s: float) -> Any:
        return sweep.ChildOutcome(
            returncode=returncode,
            elapsed_s=0.1,
            stdout_tail="child stdout",
            stderr_tail="child stderr",
            timed_out=False,
        )

    [result_path] = sweep.run_sweep(
        [case], tmp_path, max_concurrency=1, attempt_runner=runner
    )
    result = json.loads(result_path.read_text())

    assert result["status"] == "bug"
    assert result["failure_reason"] == "child_missing_result"
    assert result["exit_code"] == returncode

    def should_not_launch(_command: list[str], _timeout_s: float) -> Any:
        raise AssertionError("valid parent bug result must be resumable")

    [resumed_path] = sweep.run_sweep(
        [case], tmp_path, max_concurrency=1, attempt_runner=should_not_launch
    )
    assert resumed_path == result_path


def test_parent_never_clobbers_a_child_result(tmp_path: Path) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]
    original_bytes: bytes | None = None

    def runner(command: list[str], _timeout_s: float) -> Any:
        nonlocal original_bytes
        child_case, result_path = _case_and_result_from_command(command)
        _write_json(result_path, _terminal_result(child_case, "success"))
        original_bytes = result_path.read_bytes()
        return sweep.ChildOutcome(
            returncode=1,
            elapsed_s=0.1,
            stdout_tail="",
            stderr_tail="late parent error",
            timed_out=False,
        )

    [result_path] = sweep.run_sweep(
        [case], tmp_path, max_concurrency=1, attempt_runner=runner
    )

    assert original_bytes is not None
    assert result_path.read_bytes() == original_bytes
    assert json.loads(original_bytes)["status"] == "success"


@pytest.mark.parametrize(
    ("status", "returncode"),
    [("success", 0), ("simulated-oom", 2), ("bug", 1)],
)
def test_child_produced_terminal_records_are_validated_and_kept(
    tmp_path: Path,
    status: str,
    returncode: int,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]

    def runner(command: list[str], _timeout_s: float) -> Any:
        child_case, result_path = _case_and_result_from_command(command)
        _write_json(result_path, _terminal_result(child_case, status))
        return sweep.ChildOutcome(
            returncode=returncode,
            elapsed_s=0.1,
            stdout_tail="",
            stderr_tail="",
            timed_out=False,
        )

    [result_path] = sweep.run_sweep(
        [case], tmp_path, max_concurrency=1, attempt_runner=runner
    )

    assert json.loads(result_path.read_text())["status"] == status


@pytest.mark.parametrize(
    ("status", "field", "value"),
    [
        ("bug", "exit_code", True),
        ("bug", "failure_reason", None),
        ("bug", "oom_evidence", {"unexpected": "evidence"}),
        ("bug", "signal", signal.SIGTERM),
        ("bug", "total_proc_s", 0.0),
        ("bug", "sim_wallclock_s", "not-a-number"),
        ("simulated-oom", "exit_code", 0),
        ("simulated-oom", "failure_reason", None),
        ("simulated-oom", "oom_evidence", None),
        ("simulated-oom", "oom_evidence", []),
        ("simulated-oom", "signal", signal.SIGTERM),
        ("simulated-oom", "total_proc_s", 0.0),
        ("simulated-oom", "sim_wallclock_s", "not-a-number"),
    ],
)
def test_invalid_failure_terminal_record_is_rejected_without_resume(
    tmp_path: Path,
    status: str,
    field: str,
    value: Any,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]

    def runner(command: list[str], _timeout_s: float) -> Any:
        child_case, result_path = _case_and_result_from_command(command)
        payload = _terminal_result(child_case, status)
        payload[field] = value
        _write_json(result_path, payload)
        return sweep.ChildOutcome(
            returncode=payload["exit_code"],
            elapsed_s=0.1,
            stdout_tail="",
            stderr_tail="",
            timed_out=False,
        )

    with pytest.raises(ValueError, match="Invalid"):
        sweep.run_sweep(
            [case], tmp_path, max_concurrency=1, attempt_runner=runner
        )


def test_empty_simulated_oom_evidence_is_a_valid_terminal_record(
    tmp_path: Path,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]

    def runner(command: list[str], _timeout_s: float) -> Any:
        child_case, result_path = _case_and_result_from_command(command)
        payload = _terminal_result(child_case, "simulated-oom")
        payload["oom_evidence"] = {}
        _write_json(result_path, payload)
        return sweep.ChildOutcome(
            returncode=2,
            elapsed_s=0.1,
            stdout_tail="",
            stderr_tail="",
            timed_out=False,
        )

    [result_path] = sweep.run_sweep(
        [case], tmp_path, max_concurrency=1, attempt_runner=runner
    )
    assert json.loads(result_path.read_text())["status"] == "simulated-oom"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_requests", 63),
        ("sim_wallclock_s", 0.0),
        ("event_count", 0),
        ("events_per_s", 0.0),
    ],
)
def test_invalid_child_success_record_is_rejected_without_replacement(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]
    original: bytes | None = None

    def runner(command: list[str], _timeout_s: float) -> Any:
        nonlocal original
        child_case, result_path = _case_and_result_from_command(command)
        payload = _terminal_result(child_case, "success")
        payload[field] = value
        _write_json(result_path, payload)
        original = result_path.read_bytes()
        return sweep.ChildOutcome(
            returncode=0,
            elapsed_s=0.1,
            stdout_tail="",
            stderr_tail="",
            timed_out=False,
        )

    with pytest.raises(ValueError, match="success"):
        sweep.run_sweep(
            [case], tmp_path, max_concurrency=1, attempt_runner=runner
        )

    result_path = _result_path(tmp_path, case)
    assert original is not None
    assert result_path.read_bytes() == original


def test_dense_master_wrapper_is_an_exact_argument_vector_prefix(
    tmp_path: Path,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]
    seen: dict[str, Any] = {}

    def runner(command: list[str], timeout_s: float) -> Any:
        seen["command"] = command
        seen["timeout_s"] = timeout_s
        child_case, result_path = _case_and_result_from_command(command)
        _write_json(result_path, _terminal_result(child_case, "success"))
        return sweep.ChildOutcome(
            returncode=0,
            elapsed_s=0.1,
            stdout_tail="",
            stderr_tail="",
            timed_out=False,
        )

    sweep.run_sweep(
        [case],
        tmp_path,
        max_concurrency=1,
        python_executable="/opt/frontier-python",
        dense_master_wrapper=True,
        attempt_runner=runner,
    )

    command = seen["command"]
    assert command[:6] == [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        "MemoryMax=4G",
        "--quiet",
    ]
    assert command[6] == "/opt/frontier-python"
    assert Path(command[7]).name == "run_case.py"
    assert seen["timeout_s"] == 14_400.0


@pytest.mark.parametrize(
    ("max_concurrency", "timeout_s", "message"),
    [(0, 1.0, "max_concurrency"), (1, 0.0, "timeout_s")],
)
def test_run_sweep_rejects_invalid_execution_bounds(
    tmp_path: Path,
    max_concurrency: int,
    timeout_s: float,
    message: str,
) -> None:
    sweep = _load_sweep()
    case = sweep.build_cases(
        "dense", (32,), "sequential", shape_overrides=None
    )[0]

    with pytest.raises(ValueError, match=message):
        sweep.run_sweep(
            [case],
            tmp_path,
            max_concurrency=max_concurrency,
            timeout_s=timeout_s,
        )


def test_cli_main_uses_required_defaults_and_prints_result_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sweep = _load_sweep()
    output_root = tmp_path / "artifacts"
    expected_paths = [output_root / "first.json", output_root / "second.json"]
    observed: dict[str, Any] = {}

    def sweep_runner(
        cases: list[CaseSpec],
        received_output_root: Path,
        **kwargs: Any,
    ) -> list[Path]:
        observed["cases"] = cases
        observed["output_root"] = received_output_root
        observed["kwargs"] = kwargs
        return expected_paths

    returncode = sweep.main(
        ["--model", "moe", "--output-root", str(output_root)],
        sweep_runner=sweep_runner,
    )

    assert returncode == 0
    assert [case.total_gpus for case in observed["cases"]] == list(
        sweep.DEFAULT_SCALES
    )
    assert all(case.model == "moe" for case in observed["cases"])
    assert all(case.mode == "sequential" for case in observed["cases"])
    assert observed["output_root"] == output_root
    assert observed["kwargs"] == {
        "max_concurrency": 1,
        "timeout_s": 14_400.0,
        "python_executable": sys.executable,
        "dense_master_wrapper": False,
    }
    assert capsys.readouterr().out.splitlines() == [
        str(path) for path in expected_paths
    ]


def test_cli_main_passes_explicit_options_and_repeated_shape_overrides(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sweep = _load_sweep()
    output_root = tmp_path / "artifacts"
    result_path = output_root / "result.json"
    observed: dict[str, Any] = {}

    def sweep_runner(
        cases: list[CaseSpec],
        received_output_root: Path,
        **kwargs: Any,
    ) -> list[Path]:
        observed["cases"] = cases
        observed["output_root"] = received_output_root
        observed["kwargs"] = kwargs
        return [result_path]

    returncode = sweep.main(
        [
            "--model",
            "dense",
            "--scales",
            "32",
            "64",
            "--mode",
            "parallel",
            "--output-root",
            str(output_root),
            "--max-concurrency",
            "3",
            "--timeout-s",
            "9.5",
            "--attempt-index",
            "1",
            "--python-executable",
            "/opt/frontier-python",
            "--dense-master-wrapper",
            "--shape-override",
            "attn_tp=2",
            "--shape-override",
            "pp=4",
        ],
        sweep_runner=sweep_runner,
    )

    cases = observed["cases"]
    assert returncode == 0
    assert [case.total_gpus for case in cases] == [32, 64]
    assert all(case.mode == "parallel" for case in cases)
    assert all(case.attempt_index == 1 for case in cases)
    assert [(case.shape.attn_tp, case.shape.pp) for case in cases] == [
        (2, 4),
        (2, 4),
    ]
    assert observed["output_root"] == output_root
    assert observed["kwargs"] == {
        "max_concurrency": 3,
        "timeout_s": 9.5,
        "python_executable": "/opt/frontier-python",
        "dense_master_wrapper": True,
    }
    assert capsys.readouterr().out == f"{result_path}\n"


def test_cli_rejects_dense_master_wrapper_for_moe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sweep = _load_sweep()

    def unexpected_runner(*_args: Any, **_kwargs: Any) -> list[Path]:
        raise AssertionError("Rejected CLI input must not start a sweep")

    with pytest.raises(SystemExit) as exc_info:
        sweep.main(
            [
                "--model",
                "moe",
                "--output-root",
                str(tmp_path),
                "--dense-master-wrapper",
            ],
            sweep_runner=unexpected_runner,
        )

    assert exc_info.value.code == 2
    assert "dense" in capsys.readouterr().err


@pytest.mark.parametrize(
    "override_args",
    [
        ["attn_tp"],
        ["attn_tp=not-an-int"],
        ["unknown=1"],
        ["attn_tp=2", "attn_tp=4"],
    ],
)
def test_cli_rejects_invalid_shape_overrides(
    tmp_path: Path,
    override_args: list[str],
) -> None:
    sweep = _load_sweep()
    argv = ["--model", "dense", "--output-root", str(tmp_path)]
    for override in override_args:
        argv.extend(["--shape-override", override])

    with pytest.raises(SystemExit) as exc_info:
        sweep.main(argv, sweep_runner=lambda *_args, **_kwargs: [])

    assert exc_info.value.code == 2


def test_cli_script_exposes_reproducible_help_entry_point() -> None:
    sweep = _load_sweep()
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(Path(sweep.__file__).resolve()), "--help"],
        cwd=sweep.REPO_ROOT,
        env=clean_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    for option in (
        "--model",
        "--scales",
        "--mode",
        "--output-root",
        "--max-concurrency",
        "--timeout-s",
        "--python-executable",
        "--dense-master-wrapper",
        "--shape-override",
    ):
        assert option in result.stdout


def test_run_case_script_exposes_reproducible_help_entry_point() -> None:
    sweep = _load_sweep()
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(sweep.RUN_CASE_PATH), "--help"],
        cwd=sweep.REPO_ROOT,
        env=clean_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--case-json" in result.stdout
    assert "--result-json" in result.stdout


def test_lifecycle_reproducer_exposes_reproducible_help_entry_point() -> None:
    sweep = _load_sweep()
    reproducer_path = sweep.RUN_CASE_PATH.with_name(
        "pd_moe_lifecycle_reproducer.py"
    )
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(reproducer_path), "--help"],
        cwd=sweep.REPO_ROOT,
        env=clean_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--case-json" in result.stdout
    assert "--state-json" in result.stdout


def test_sweep_child_runner_executes_run_case_without_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep = _load_sweep()
    monkeypatch.delenv("PYTHONPATH", raising=False)

    outcome = sweep._execute_child(
        [sys.executable, str(sweep.RUN_CASE_PATH), "--help"],
        timeout_s=30.0,
    )

    assert outcome.timed_out is False
    assert outcome.returncode == 0, outcome.stderr_tail
    assert "--case-json" in outcome.stdout_tail
    assert "--result-json" in outcome.stdout_tail

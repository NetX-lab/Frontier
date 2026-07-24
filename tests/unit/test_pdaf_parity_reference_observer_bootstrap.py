import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.e2e.pd_af_parity import reference_observer_bootstrap


BOOTSTRAP_MODULE = (
    Path(__file__).parents[1]
    / "e2e"
    / "pd_af_parity"
    / "reference_observer_bootstrap.py"
)


def _safe_simulator_argv(
    tmp_path: Path,
    *extra: str,
) -> tuple[str, ...]:
    return (
        "--simulation_mode",
        "offline",
        "--metrics_config_output_dir",
        str(tmp_path / "metrics"),
        "--metrics_config_cache_dir",
        str(tmp_path / "cache"),
        "--cluster_config_cc_backend_config_type",
        "analytical",
        *extra,
    )


class _ClusterType:
    name = "DECODE_ATTN"


class _Request:
    def __init__(self) -> None:
        self.id = 0
        self.arrived_at = 0.0
        self.prefill_completed_at = 1.0
        self.num_processed_decode_tokens = 0


class _Batch:
    def __init__(self, request: _Request) -> None:
        self.id = 1
        self.global_id = 2
        self.requests = [request]


class _Scheduler:
    def __init__(self) -> None:
        self._cluster_type = _ClusterType()

    def resolve_decode_attn_boundary_first_mixed_global_end_time(
        self,
        time: float,
        batch: _Batch,
    ) -> float:
        return time


class _GlobalBatchEndEvent:
    def __init__(self, time: float, batch: _Batch) -> None:
        self.time = time
        self._batch = batch
        self._cluster_type = _ClusterType()

    def handle_event(self) -> None:
        self._batch.requests[0].num_processed_decode_tokens += 1


def test_reference_observer_bootstrap_module_exists() -> None:
    assert BOOTSTRAP_MODULE.is_file()


def test_reference_observer_bootstrap_exposes_run_entrypoint() -> None:
    assert callable(reference_observer_bootstrap.run_reference_with_observer)


def test_reference_observer_bootstrap_rejects_non_pinned_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="pinned Reference repo root"):
        reference_observer_bootstrap.run_reference_with_observer(
            tmp_path,
            tmp_path / "lifecycle.json",
            (),
            (0,),
        )


def test_reference_observer_bootstrap_validates_pinned_identity() -> None:
    identity = reference_observer_bootstrap._validate_reference_identity(
        reference_observer_bootstrap.REFERENCE_REPO_ROOT
    )

    assert identity == {
        "reference_git_head": (
            "dcb1cc8ee160a9c3c5412293d93b64042960aa4d"
        ),
        "request_source_sha256": (
            "4cff6da775a1b04ba4c252ccc679a3f2919ed5bfc98f1c039dff1519b9bc42b0"
        ),
        "cluster_scheduler_source_sha256": (
            "5a28d18a7cfdcfc04b2848a9861973c652947005b356fa9b837b90356329fb6d"
        ),
        "global_batch_end_event_source_sha256": (
            "5366bd739c9765ef57b06448ce719d013795273535fd623aa17ed064279021b0"
        ),
    }


def test_reference_observer_bootstrap_reads_complete_git_status_without_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(
        argv: list[str],
        *,
        check: bool,
        capture_output: bool,
    ) -> object:
        captured.update(
            argv=argv,
            check=check,
            capture_output=capture_output,
        )
        return SimpleNamespace(
            stdout=(
                b" M frontier/entities/batch.py\0"
                b"?? frontier/new_runtime.py\0"
            )
        )

    monkeypatch.setattr(
        reference_observer_bootstrap.subprocess,
        "run",
        fake_run,
    )

    status = reference_observer_bootstrap._read_git_status(
        reference_observer_bootstrap.REFERENCE_REPO_ROOT
    )

    assert status == (
        b" M frontier/entities/batch.py",
        b"?? frontier/new_runtime.py",
    )
    assert captured == {
        "argv": [
            "git",
            "--no-optional-locks",
            "-C",
            str(reference_observer_bootstrap.REFERENCE_REPO_ROOT),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        "check": True,
        "capture_output": True,
    }


def test_reference_observer_bootstrap_propagates_git_status_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = subprocess.CalledProcessError(128, ["git", "status"])
    monkeypatch.setattr(
        reference_observer_bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(subprocess.CalledProcessError) as captured:
        reference_observer_bootstrap._read_git_status(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT
        )

    assert captured.value is error


@pytest.mark.parametrize(
    "status_entry",
    [
        b" M frontier/entities/batch.py",
        b"M  frontier/entities/batch.py",
        b"?? frontier/new_runtime.py",
    ],
)
def test_reference_observer_bootstrap_rejects_dirty_reference_identity(
    monkeypatch: pytest.MonkeyPatch,
    status_entry: bytes,
) -> None:
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_read_git_status",
        lambda _root: (status_entry,),
        raising=False,
    )

    with pytest.raises(ValueError, match="worktree and index must be clean"):
        reference_observer_bootstrap._validate_reference_identity(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT
        )


def test_reference_observer_bootstrap_rejects_wrong_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_read_git_head",
        lambda _root: "0" * 40,
        raising=False,
    )

    with pytest.raises(ValueError, match="Reference git HEAD"):
        reference_observer_bootstrap._validate_reference_identity(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT
        )


def test_reference_observer_bootstrap_rejects_wrong_source_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_sha256_file",
        lambda _path: "0" * 64,
        raising=False,
    )

    with pytest.raises(ValueError, match="frontier/entities/request.py"):
        reference_observer_bootstrap._validate_reference_identity(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT
        )


def test_reference_observer_bootstrap_rejects_existing_sidecar(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "lifecycle.json"
    sidecar.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar already exists"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            ("--simulation_mode", "offline"),
            (0,),
        )


def test_reference_observer_bootstrap_rejects_missing_sidecar_parent(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="parent is not a directory"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            tmp_path / "missing" / "lifecycle.json",
            ("--simulation_mode", "offline"),
            (0,),
        )


@pytest.mark.parametrize(
    "request_ids",
    [(), (True,), (-1,), (0, 0), (0, 1.5)],
)
def test_reference_observer_bootstrap_rejects_invalid_expected_request_ids(
    tmp_path: Path,
    request_ids: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="expected_request_ids"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            tmp_path / "lifecycle.json",
            ("--simulation_mode", "offline"),
            request_ids,
        )


@pytest.mark.parametrize("argv", [("--flag", 1), ("bad\x00token",)])
def test_reference_observer_bootstrap_rejects_invalid_simulator_argv(
    tmp_path: Path,
    argv: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="simulator_argv"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            tmp_path / "lifecycle.json",
            argv,
            (0,),
        )


def test_reference_observer_bootstrap_rejects_initial_dirty_reference_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original_argv = list(sys.argv)
    original_sys_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode

    def unexpected_import_check() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_read_git_status",
        lambda _root: (b" M frontier/entities/batch.py",),
        raising=False,
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_require_fresh_frontier_import",
        unexpected_import_check,
    )
    sidecar = tmp_path / "lifecycle.json"

    with pytest.raises(ValueError, match="worktree and index must be clean"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert calls == 0
    assert not sidecar.exists()
    assert sys.argv == original_argv
    assert sys.path == original_sys_path
    assert sys.dont_write_bytecode is original_dont_write_bytecode


def test_reference_observer_bootstrap_rejects_preimported_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "frontier", object())

    with pytest.raises(ValueError, match="already imported frontier"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            tmp_path / "lifecycle.json",
            _safe_simulator_argv(tmp_path),
            (0,),
        )


def test_reference_observer_bootstrap_rejects_wrong_imported_module_path(
    tmp_path: Path,
) -> None:
    wrong_path = tmp_path / "frontier" / "main.py"
    wrong_path.parent.mkdir()
    wrong_path.write_text("# wrong module\n", encoding="utf-8")
    module = ModuleType("frontier.main")
    module.__file__ = str(wrong_path)

    with pytest.raises(ValueError, match="imported from wrong path"):
        reference_observer_bootstrap._require_module_path(
            module,
            BOOTSTRAP_MODULE,
        )


def test_reference_observer_bootstrap_rejects_non_callable_reference_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = {
        "frontier.main": SimpleNamespace(main=object()),
        "frontier.scheduler.cluster_scheduler.base_cluster_scheduler": (
            SimpleNamespace(BaseClusterScheduler=_Scheduler)
        ),
        "frontier.events.global_batch_end_event": SimpleNamespace(
            GlobalBatchEndEvent=_GlobalBatchEndEvent
        ),
    }
    monkeypatch.setattr(
        reference_observer_bootstrap.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_require_module_path",
        lambda *_args: None,
    )

    with pytest.raises(ValueError, match="main is not callable"):
        reference_observer_bootstrap._import_reference_runtime(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT
        )


def test_reference_observer_bootstrap_runs_main_and_writes_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator_argv = _safe_simulator_argv(tmp_path)
    observed_argv: list[str] = []
    original_dont_write_bytecode = sys.dont_write_bytecode
    identity_calls = 0
    validate_identity = reference_observer_bootstrap._validate_reference_identity

    def tracking_validate_identity(root: Path) -> dict[str, str]:
        nonlocal identity_calls
        identity_calls += 1
        return validate_identity(root)

    def fake_main() -> None:
        assert sys.dont_write_bytecode is True
        observed_argv.extend(sys.argv)
        request = _Request()
        batch = _Batch(request)
        scheduler = _Scheduler()
        resolved = scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
            2.0,
            batch,
        )
        _GlobalBatchEndEvent(resolved, batch).handle_event()

    runtime = SimpleNamespace(
        base_cluster_scheduler_class=_Scheduler,
        global_batch_end_event_class=_GlobalBatchEndEvent,
        main=fake_main,
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_import_reference_runtime",
        lambda _root: runtime,
        raising=False,
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_require_fresh_frontier_import",
        lambda: None,
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_validate_reference_identity",
        tracking_validate_identity,
    )
    sidecar = tmp_path / "lifecycle.json"
    original_argv = list(sys.argv)
    original_sys_path = list(sys.path)

    payload = reference_observer_bootstrap.run_reference_with_observer(
        reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        sidecar,
        simulator_argv,
        (0,),
    )

    assert observed_argv == ["frontier.main", *simulator_argv]
    assert identity_calls == 2
    assert sys.argv == original_argv
    assert sys.path == original_sys_path
    assert sys.dont_write_bytecode is original_dont_write_bytecode
    assert payload == json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["request_count"] == 1
    assert payload["requests"][0]["raw_decode_execution_completed_at_s"] == 2.0
    producer = payload["producer"]
    expected_argv_hash = hashlib.sha256(
        json.dumps(
            list(simulator_argv),
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert producer["argv_sha256"] == expected_argv_hash
    assert producer["observer_source_sha256"] == (
        reference_observer_bootstrap._sha256_file(
            BOOTSTRAP_MODULE.with_name("reference_lifecycle_observer.py")
        )
    )
    assert producer["bootstrap_source_sha256"] == (
        reference_observer_bootstrap._sha256_file(BOOTSTRAP_MODULE)
    )
    assert _Scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time.__name__ == (
        "resolve_decode_attn_boundary_first_mixed_global_end_time"
    )
    assert _GlobalBatchEndEvent.handle_event.__name__ == "handle_event"


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    main: object,
) -> None:
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_require_fresh_frontier_import",
        lambda: None,
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_import_reference_runtime",
        lambda _root: SimpleNamespace(
            base_cluster_scheduler_class=_Scheduler,
            global_batch_end_event_class=_GlobalBatchEndEvent,
            main=main,
        ),
    )


def test_reference_observer_bootstrap_preserves_simulator_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = RuntimeError("simulator failure")

    def failing_main() -> None:
        raise original_error

    _patch_runtime(monkeypatch, failing_main)
    sidecar = tmp_path / "lifecycle.json"

    with pytest.raises(RuntimeError, match="simulator failure") as captured:
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert captured.value is original_error
    assert not sidecar.exists()
    assert _Scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time.__name__ == (
        "resolve_decode_attn_boundary_first_mixed_global_end_time"
    )
    assert _GlobalBatchEndEvent.handle_event.__name__ == "handle_event"


def test_reference_observer_bootstrap_rejects_unexpected_main_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch, lambda: 1)
    sidecar = tmp_path / "lifecycle.json"

    with pytest.raises(ValueError, match="must return None"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert not sidecar.exists()


def test_reference_observer_bootstrap_rejects_incomplete_request_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch, lambda: None)
    sidecar = tmp_path / "lifecycle.json"

    with pytest.raises(ValueError, match="request set.*expected_request_ids"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert not sidecar.exists()


def test_reference_observer_bootstrap_rejects_extra_observed_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def main_with_extra_request() -> None:
        for request_id in (0, 1):
            request = _Request()
            request.id = request_id
            batch = _Batch(request)
            resolved = _Scheduler().resolve_decode_attn_boundary_first_mixed_global_end_time(
                2.0 + request_id,
                batch,
            )
            _GlobalBatchEndEvent(resolved, batch).handle_event()

    _patch_runtime(monkeypatch, main_with_extra_request)
    sidecar = tmp_path / "lifecycle.json"

    with pytest.raises(ValueError, match=r"actual=\(0, 1\)"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert not sidecar.exists()


def test_reference_observer_bootstrap_rejects_pending_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def pending_main() -> None:
        request = _Request()
        batch = _Batch(request)
        _Scheduler().resolve_decode_attn_boundary_first_mixed_global_end_time(
            2.0,
            batch,
        )

    _patch_runtime(monkeypatch, pending_main)
    sidecar = tmp_path / "lifecycle.json"

    with pytest.raises(ValueError, match="pending candidates"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert not sidecar.exists()


def test_reference_observer_bootstrap_rejects_sidecar_inside_reference_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch, lambda: None)
    sidecar = (
        reference_observer_bootstrap.REFERENCE_REPO_ROOT
        / "_forbidden_reference_lifecycle.json"
    )

    with pytest.raises(ValueError, match="outside.*Reference"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            ("--simulation_mode", "offline"),
            (0,),
        )

    assert not sidecar.exists()


@pytest.mark.parametrize(
    ("write_argument", "expected_flag"),
    [
        (
            (
                "--metrics_config_output_dir",
                str(reference_observer_bootstrap.REFERENCE_REPO_ROOT / "outputs"),
            ),
            "metrics_config_output_dir",
        ),
        (
            (
                "--metrics_config_cache_dir="
                f"{reference_observer_bootstrap.REFERENCE_REPO_ROOT / 'cache'}",
            ),
            "metrics_config_cache_dir",
        ),
    ],
)
def test_reference_observer_bootstrap_rejects_write_path_inside_reference_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_argument: tuple[str, ...],
    expected_flag: str,
) -> None:
    _patch_runtime(monkeypatch, lambda: None)

    with pytest.raises(
        ValueError,
        match=rf"{expected_flag}.*must resolve outside",
    ):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            tmp_path / "lifecycle.json",
            _safe_simulator_argv(tmp_path, *write_argument),
            (0,),
        )


@pytest.mark.parametrize(
    "missing_flag",
    [
        "--metrics_config_output_dir",
        "--metrics_config_cache_dir",
    ],
)
def test_reference_observer_bootstrap_requires_explicit_external_metrics_paths(
    tmp_path: Path,
    missing_flag: str,
) -> None:
    values = {
        "--metrics_config_output_dir": str(tmp_path / "metrics"),
        "--metrics_config_cache_dir": str(tmp_path / "cache"),
    }
    argv = tuple(
        token
        for flag, value in values.items()
        if flag != missing_flag
        for token in (flag, value)
    ) + (
        "--cluster_config_cc_backend_config_type",
        "analytical",
    )

    with pytest.raises(ValueError, match=f"required.*{missing_flag}"):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


@pytest.mark.parametrize(
    "event_log_argument",
    [
        (
            "--cluster_event_log_dir",
            str(
                reference_observer_bootstrap.REFERENCE_REPO_ROOT
                / "logs"
                / "cluster_events"
            ),
        ),
        (
            "--cluster_event_log_dir="
            f"{reference_observer_bootstrap.REFERENCE_REPO_ROOT / 'events'}",
        ),
    ],
)
def test_reference_observer_bootstrap_rejects_event_log_path_inside_reference(
    tmp_path: Path,
    event_log_argument: tuple[str, ...],
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--enable_cluster_event_logging",
        *event_log_argument,
    )

    with pytest.raises(ValueError, match="cluster_event_log_dir"):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


def test_reference_observer_bootstrap_requires_event_dir_when_logging_enabled(
    tmp_path: Path,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--enable_cluster_event_logging",
    )

    with pytest.raises(ValueError, match="required.*cluster_event_log_dir"):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


def test_reference_observer_bootstrap_rejects_profile_path_inside_reference(
    tmp_path: Path,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--enable_performance_profiling",
        "--performance_profiling_output_file",
        str(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT
            / "performance_profile.json"
        ),
    )

    with pytest.raises(ValueError, match="performance_profiling_output_file"):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


def test_reference_observer_bootstrap_accepts_external_metrics_paths(
    tmp_path: Path,
) -> None:
    assert (
        reference_observer_bootstrap._validate_simulator_write_paths(
            _safe_simulator_argv(tmp_path),
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )
        is None
    )


def test_reference_observer_bootstrap_accepts_external_event_log_path(
    tmp_path: Path,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--enable_cluster_event_logging",
        "--cluster_event_log_dir",
        str(tmp_path / "events"),
    )

    assert (
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )
        is None
    )


def test_reference_observer_bootstrap_accepts_disabled_event_logging_without_dir(
    tmp_path: Path,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--enable_cluster_event_logging",
        "--no-enable_cluster_event_logging",
    )

    assert (
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )
        is None
    )


def test_reference_observer_bootstrap_accepts_relative_profile_output(
    tmp_path: Path,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--enable_performance_profiling",
        "--performance_profiling_output_file",
        "performance_profile.json",
    )

    assert (
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )
        is None
    )


def test_reference_observer_bootstrap_accepts_default_profile_output(
    tmp_path: Path,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--enable_performance_profiling",
    )

    assert (
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )
        is None
    )


def test_reference_observer_bootstrap_rejects_trace_path_inside_reference(
    tmp_path: Path,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--metrics_config_trace_output_file",
        str(reference_observer_bootstrap.REFERENCE_REPO_ROOT / "trace.json"),
    )

    with pytest.raises(ValueError, match="metrics_config_trace_output_file"):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
    )


def test_reference_observer_bootstrap_rejects_relative_trace_traversal_into_reference(
    tmp_path: Path,
) -> None:
    metrics_run_root = tmp_path / "metrics" / "<run>"
    target = reference_observer_bootstrap.REFERENCE_REPO_ROOT / "trace.json"
    relative_target = os.path.relpath(target, start=metrics_run_root)
    argv = _safe_simulator_argv(
        tmp_path,
        "--metrics_config_trace_output_file",
        relative_target,
    )

    with pytest.raises(
        ValueError,
        match=r"metrics_config_trace_output_file.*must resolve outside",
    ):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


def test_reference_observer_bootstrap_rejects_unsafe_direct_path_before_safe_duplicate(
    tmp_path: Path,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--cluster_event_log_dir",
        str(reference_observer_bootstrap.REFERENCE_REPO_ROOT / "events"),
        "--cluster_event_log_dir",
        str(tmp_path / "events"),
    )

    with pytest.raises(
        ValueError,
        match=r"cluster_event_log_dir.*must resolve outside",
    ):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


def test_reference_observer_bootstrap_rejects_unsafe_relative_path_before_safe_duplicate(
    tmp_path: Path,
) -> None:
    metrics_run_root = tmp_path / "metrics" / "<run>"
    target = reference_observer_bootstrap.REFERENCE_REPO_ROOT / "trace.json"
    relative_target = os.path.relpath(target, start=metrics_run_root)
    argv = _safe_simulator_argv(
        tmp_path,
        "--metrics_config_trace_output_file",
        relative_target,
        "--metrics_config_trace_output_file",
        "trace.json",
    )

    with pytest.raises(
        ValueError,
        match=r"metrics_config_trace_output_file.*must resolve outside",
    ):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


@pytest.mark.parametrize(
    "missing_value_argument",
    [
        ("--cluster_event_log_dir",),
        ("--cluster_event_log_dir", ""),
        ("--cluster_event_log_dir", "--enable_cluster_event_logging"),
        ("--metrics_config_trace_output_file=",),
    ],
)
def test_reference_observer_bootstrap_rejects_path_flag_without_value(
    tmp_path: Path,
    missing_value_argument: tuple[str, ...],
) -> None:
    argv = _safe_simulator_argv(tmp_path, *missing_value_argument)

    with pytest.raises(ValueError, match="requires a path value"):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


@pytest.mark.parametrize(
    "backend_type",
    [
        "vidur",
        "collective_sim",
        "aiconfigurator",
        "astra_sim_analytical",
    ],
)
def test_reference_observer_bootstrap_rejects_non_analytical_base_cc_backend(
    tmp_path: Path,
    backend_type: str,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--cluster_config_cc_backend_config_type",
        backend_type,
    )

    with pytest.raises(
        ValueError,
        match=r"cluster_config_cc_backend_config_type.*must be analytical",
    ):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


def test_reference_observer_bootstrap_requires_explicit_base_cc_backend(
    tmp_path: Path,
) -> None:
    argv = (
        "--simulation_mode",
        "offline",
        "--metrics_config_output_dir",
        str(tmp_path / "metrics"),
        "--metrics_config_cache_dir",
        str(tmp_path / "cache"),
    )

    with pytest.raises(
        ValueError,
        match=r"required.*cluster_config_cc_backend_config_type",
    ):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


@pytest.mark.parametrize(
    "override_flag",
    [
        "--cluster_config_prefill_cc_backend_config_type",
        "--cluster_config_decode_cc_backend_config_type",
        "--cluster_config_decode_attn_cc_backend_config_type",
        "--cluster_config_decode_ffn_cc_backend_config_type",
    ],
)
def test_reference_observer_bootstrap_rejects_non_analytical_cluster_cc_override(
    tmp_path: Path,
    override_flag: str,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        override_flag,
        "vidur",
    )

    with pytest.raises(
        ValueError,
        match=rf"{override_flag.removeprefix('--')}.*must be analytical",
    ):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


def test_reference_observer_bootstrap_accepts_analytical_cluster_cc_overrides(
    tmp_path: Path,
) -> None:
    argv = _safe_simulator_argv(
        tmp_path,
        "--cluster_config_prefill_cc_backend_config_type=analytical",
        "--cluster_config_decode_attn_cc_backend_config_type",
        "analytical",
        "--cluster_config_decode_ffn_cc_backend_config_type=analytical",
    )

    assert (
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )
        is None
    )


def test_reference_observer_bootstrap_uses_last_cc_backend_value(
    tmp_path: Path,
) -> None:
    safe_argv = _safe_simulator_argv(
        tmp_path,
        "--cluster_config_cc_backend_config_type",
        "vidur",
        "--cluster_config_cc_backend_config_type=analytical",
    )
    unsafe_argv = _safe_simulator_argv(
        tmp_path,
        "--cluster_config_cc_backend_config_type=analytical",
        "--cluster_config_cc_backend_config_type",
        "vidur",
    )

    assert (
        reference_observer_bootstrap._validate_simulator_write_paths(
            safe_argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )
        is None
    )
    with pytest.raises(
        ValueError,
        match=r"cluster_config_cc_backend_config_type.*must be analytical",
    ):
        reference_observer_bootstrap._validate_simulator_write_paths(
            unsafe_argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


@pytest.mark.parametrize(
    "missing_value_argument",
    [
        ("--cluster_config_cc_backend_config_type",),
        ("--cluster_config_cc_backend_config_type", ""),
        (
            "--cluster_config_decode_attn_cc_backend_config_type",
            "--enable_cluster_event_logging",
        ),
        ("--cluster_config_decode_attn_cc_backend_config_type=",),
    ],
)
def test_reference_observer_bootstrap_rejects_cc_backend_flag_without_value(
    tmp_path: Path,
    missing_value_argument: tuple[str, ...],
) -> None:
    argv = _safe_simulator_argv(tmp_path, *missing_value_argument)

    with pytest.raises(ValueError, match="requires a backend value"):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


@pytest.mark.parametrize(
    "abbreviated_argument",
    [
        ("--cluster_config_cc_backend_config_ty", "vidur"),
        ("--cluster_config_decode_attn_cc_backend_config_ty=vidur",),
        (
            "--metrics_config_output_di",
            str(reference_observer_bootstrap.REFERENCE_REPO_ROOT / "metrics"),
        ),
        (
            "--metrics_config_trace_output_fil",
            str(reference_observer_bootstrap.REFERENCE_REPO_ROOT / "trace.json"),
        ),
        ("--enable_cluster_event_loggin",),
    ],
)
def test_reference_observer_bootstrap_rejects_protected_option_abbreviation(
    tmp_path: Path,
    abbreviated_argument: tuple[str, ...],
) -> None:
    argv = _safe_simulator_argv(tmp_path, *abbreviated_argument)

    with pytest.raises(
        ValueError,
        match="abbreviated protected simulator option",
    ):
        reference_observer_bootstrap._validate_simulator_write_paths(
            argv,
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
        )


@pytest.mark.parametrize("failure_stage", ["import", "install"])
def test_reference_observer_bootstrap_restores_globals_after_setup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    original_argv = list(sys.argv)
    original_sys_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    original_error = RuntimeError(f"{failure_stage} failure")

    if failure_stage == "import":
        monkeypatch.setattr(
            reference_observer_bootstrap,
            "_require_fresh_frontier_import",
            lambda: None,
        )
        monkeypatch.setattr(
            reference_observer_bootstrap,
            "_import_reference_runtime",
            lambda _root: (_ for _ in ()).throw(original_error),
        )
    else:
        class FailingInstallObserver:
            def install(self, *_args: object) -> None:
                raise original_error

        observer_module = SimpleNamespace(
            ReferenceLifecycleObserver=FailingInstallObserver,
        )
        monkeypatch.setattr(
            reference_observer_bootstrap,
            "_load_observer_module",
            lambda: (observer_module, BOOTSTRAP_MODULE, "0" * 64),
        )
        _patch_runtime(monkeypatch, lambda: None)

    with pytest.raises(RuntimeError, match=f"{failure_stage} failure") as captured:
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            tmp_path / "lifecycle.json",
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert captured.value is original_error
    assert sys.argv == original_argv
    assert sys.path == original_sys_path
    assert sys.dont_write_bytecode is original_dont_write_bytecode


def test_reference_observer_bootstrap_restores_globals_when_uninstall_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingObserver:
        def install(self, *_args: object) -> None:
            return None

        def uninstall(self) -> None:
            raise RuntimeError("uninstall failure")

    observer_module = SimpleNamespace(
        ReferenceLifecycleObserver=FailingObserver,
        CANDIDATE_HOOK="candidate",
        TRANSITION_HOOK="transition",
        TRANSITION_CONTRACT="contract",
        TIMESTAMP_CONTRACT="timestamp",
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_load_observer_module",
        lambda: (observer_module, BOOTSTRAP_MODULE, "0" * 64),
    )
    _patch_runtime(monkeypatch, lambda: None)
    original_argv = list(sys.argv)
    original_sys_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    observed_state: tuple[list[str], list[str], bool] | None = None

    try:
        with pytest.raises(RuntimeError, match="uninstall failure"):
            reference_observer_bootstrap.run_reference_with_observer(
                reference_observer_bootstrap.REFERENCE_REPO_ROOT,
                tmp_path / "lifecycle.json",
                _safe_simulator_argv(tmp_path),
                (0,),
            )
    finally:
        observed_state = (
            list(sys.argv),
            list(sys.path),
            sys.dont_write_bytecode,
        )
        sys.argv[:] = original_argv
        sys.path[:] = original_sys_path
        sys.dont_write_bytecode = original_dont_write_bytecode

    assert observed_state == (
        original_argv,
        original_sys_path,
        original_dont_write_bytecode,
    )


def test_reference_observer_bootstrap_preserves_primary_error_when_uninstall_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = RuntimeError("simulator failure")
    uninstall_error = RuntimeError("uninstall failure")

    class FailingUninstallObserver:
        def install(self, *_args: object) -> None:
            return None

        def uninstall(self) -> None:
            raise uninstall_error

    observer_module = SimpleNamespace(
        ReferenceLifecycleObserver=FailingUninstallObserver,
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_load_observer_module",
        lambda: (observer_module, BOOTSTRAP_MODULE, "0" * 64),
    )
    _patch_runtime(
        monkeypatch,
        lambda: (_ for _ in ()).throw(original_error),
    )

    with pytest.raises(RuntimeError, match="simulator failure") as captured:
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            tmp_path / "lifecycle.json",
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert captured.value is original_error
    assert captured.value.__notes__ == [
        "Reference observer uninstall also failed: "
        "RuntimeError('uninstall failure')"
    ]


def test_reference_observer_bootstrap_audits_identity_after_simulator_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = RuntimeError("simulator failure")
    identities = iter(
        (
            {"reference_git_head": "a" * 40},
            {"reference_git_head": "b" * 40},
        )
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_validate_reference_identity",
        lambda _root: next(identities),
    )
    _patch_runtime(
        monkeypatch,
        lambda: (_ for _ in ()).throw(original_error),
    )

    with pytest.raises(RuntimeError, match="simulator failure") as captured:
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            tmp_path / "lifecycle.json",
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert captured.value is original_error
    assert captured.value.__notes__ == [
        "Reference identity audit also failed: "
        "ReferenceObserverBootstrapError('Reference identity changed during "
        "observed simulation')"
    ]


def test_reference_observer_bootstrap_rejects_reference_identity_change_after_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_main() -> None:
        request = _Request()
        batch = _Batch(request)
        resolved = _Scheduler().resolve_decode_attn_boundary_first_mixed_global_end_time(
            2.0,
            batch,
        )
        _GlobalBatchEndEvent(resolved, batch).handle_event()

    identities = iter(
        (
            {"reference_git_head": "a" * 40},
            {"reference_git_head": "b" * 40},
        )
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_validate_reference_identity",
        lambda _root: next(identities),
    )
    _patch_runtime(monkeypatch, fake_main)
    sidecar = tmp_path / "lifecycle.json"

    with pytest.raises(ValueError, match="identity changed"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert not sidecar.exists()


def test_reference_observer_bootstrap_rejects_dirty_reference_after_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = iter(((), (b" M frontier/entities/batch.py",)))

    def fake_main() -> None:
        request = _Request()
        batch = _Batch(request)
        resolved = _Scheduler().resolve_decode_attn_boundary_first_mixed_global_end_time(
            2.0,
            batch,
        )
        _GlobalBatchEndEvent(resolved, batch).handle_event()

    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_read_git_status",
        lambda _root: next(statuses),
        raising=False,
    )
    _patch_runtime(monkeypatch, fake_main)
    sidecar = tmp_path / "lifecycle.json"

    with pytest.raises(ValueError, match="worktree and index must be clean"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert not sidecar.exists()
    assert _Scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time.__name__ == (
        "resolve_decode_attn_boundary_first_mixed_global_end_time"
    )
    assert _GlobalBatchEndEvent.handle_event.__name__ == "handle_event"


def test_reference_observer_bootstrap_preserves_simulator_error_when_reference_dirties(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = RuntimeError("simulator failure")
    statuses = iter(((), (b"?? frontier/new_runtime.py",)))
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_read_git_status",
        lambda _root: next(statuses),
        raising=False,
    )
    _patch_runtime(
        monkeypatch,
        lambda: (_ for _ in ()).throw(original_error),
    )
    sidecar = tmp_path / "lifecycle.json"

    with pytest.raises(RuntimeError, match="simulator failure") as captured:
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert captured.value is original_error
    assert captured.value.__notes__ is not None
    assert len(captured.value.__notes__) == 1
    note = captured.value.__notes__[0]
    assert note.startswith(
        "Reference identity audit also failed: ReferenceObserverBootstrapError("
    )
    assert "Reference worktree and index must be clean" in note
    assert "?? frontier/new_runtime.py" in note
    assert not sidecar.exists()


def test_reference_observer_bootstrap_preserves_uninstall_error_when_reference_dirties(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninstall_error = RuntimeError("uninstall failure")
    statuses = iter(((), (b"M  frontier/entities/batch.py",)))

    class FailingUninstallObserver:
        def install(self, *_args: object) -> None:
            return None

        def uninstall(self) -> None:
            raise uninstall_error

    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_read_git_status",
        lambda _root: next(statuses),
        raising=False,
    )
    monkeypatch.setattr(
        reference_observer_bootstrap,
        "_load_observer_module",
        lambda: (
            SimpleNamespace(ReferenceLifecycleObserver=FailingUninstallObserver),
            BOOTSTRAP_MODULE,
            "0" * 64,
        ),
    )
    _patch_runtime(monkeypatch, lambda: None)
    sidecar = tmp_path / "lifecycle.json"

    with pytest.raises(RuntimeError, match="uninstall failure") as captured:
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            sidecar,
            _safe_simulator_argv(tmp_path),
            (0,),
        )

    assert captured.value is uninstall_error
    assert captured.value.__notes__ is not None
    assert len(captured.value.__notes__) == 1
    note = captured.value.__notes__[0]
    assert note.startswith(
        "Reference identity audit also failed: ReferenceObserverBootstrapError("
    )
    assert "Reference worktree and index must be clean" in note
    assert "M  frontier/entities/batch.py" in note
    assert not sidecar.exists()


def test_reference_observer_bootstrap_cli_routes_control_and_simulator_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(
        reference_repo_root: str,
        sidecar_path: str,
        simulator_argv: tuple[str, ...],
        expected_request_ids: tuple[int, ...],
    ) -> dict[str, object]:
        captured.update(
            reference_repo_root=reference_repo_root,
            sidecar_path=sidecar_path,
            simulator_argv=simulator_argv,
            expected_request_ids=expected_request_ids,
        )
        return {}

    monkeypatch.setattr(
        reference_observer_bootstrap,
        "run_reference_with_observer",
        fake_run,
    )
    sidecar = tmp_path / "lifecycle.json"

    exit_code = reference_observer_bootstrap.main(
        (
            "--reference-repo-root",
            str(reference_observer_bootstrap.REFERENCE_REPO_ROOT),
            "--sidecar-path",
            str(sidecar),
            "--expected-request-count",
            "2",
            "--",
            "--simulation_mode",
            "offline",
        )
    )

    assert exit_code == 0
    assert captured == {
        "reference_repo_root": str(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT
        ),
        "sidecar_path": str(sidecar),
        "simulator_argv": ("--simulation_mode", "offline"),
        "expected_request_ids": (0, 1),
    }


def test_reference_observer_bootstrap_cli_requires_delimiter() -> None:
    with pytest.raises(SystemExit) as captured:
        reference_observer_bootstrap.main(
            (
                "--reference-repo-root",
                str(reference_observer_bootstrap.REFERENCE_REPO_ROOT),
                "--sidecar-path",
                "/tmp/lifecycle.json",
                "--expected-request-count",
                "1",
            )
        )

    assert captured.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ("--reference-repo-root", "/reference", "--", "--", "--flag"),
        (
            "--reference-repo-root",
            "/reference",
            "--sidecar-path",
            "/tmp/lifecycle.json",
            "--expected-request-count",
            "1",
            "--",
        ),
        (
            "--reference-repo-root",
            "/reference",
            "--sidecar-path",
            "/tmp/lifecycle.json",
            "--expected-request-count",
            "0",
            "--",
            "--flag",
        ),
    ],
)
def test_reference_observer_bootstrap_cli_rejects_invalid_boundaries(
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit) as captured:
        reference_observer_bootstrap.main(argv)

    assert captured.value.code == 2


def test_reference_observer_bootstrap_rejects_empty_simulator_argv(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="simulator_argv.*empty"):
        reference_observer_bootstrap.run_reference_with_observer(
            reference_observer_bootstrap.REFERENCE_REPO_ROOT,
            tmp_path / "lifecycle.json",
            (),
            (0,),
        )

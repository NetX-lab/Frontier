"""Controlled bootstrap for the pinned Reference lifecycle observer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence


REFERENCE_REPO_ROOT = Path(
    "/data/ycfeng/stepfun-performance-optimization/Frontier/"
    "worktrees/ref-afd-readonly"
)
REFERENCE_GIT_HEAD = "dcb1cc8ee160a9c3c5412293d93b64042960aa4d"
REFERENCE_SOURCE_IDENTITIES = {
    "request_source_sha256": (
        "frontier/entities/request.py",
        "4cff6da775a1b04ba4c252ccc679a3f2919ed5bfc98f1c039dff1519b9bc42b0",
    ),
    "cluster_scheduler_source_sha256": (
        "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py",
        "5a28d18a7cfdcfc04b2848a9861973c652947005b356fa9b837b90356329fb6d",
    ),
    "global_batch_end_event_source_sha256": (
        "frontier/events/global_batch_end_event.py",
        "5366bd739c9765ef57b06448ce719d013795273535fd623aa17ed064279021b0",
    ),
}
_REQUIRED_EXTERNAL_WRITE_PATH_FLAGS = frozenset(
    {
        "--metrics_config_output_dir",
        "--metrics_config_cache_dir",
    }
)
_DIRECT_WRITE_PATH_FLAGS = frozenset(
    {
        *_REQUIRED_EXTERNAL_WRITE_PATH_FLAGS,
        "--cluster_event_log_dir",
    }
)
_METRICS_OUTPUT_RELATIVE_WRITE_PATH_FLAGS = frozenset(
    {
        "--metrics_config_trace_output_file",
        "--performance_profiling_output_file",
    }
)
_REFERENCE_WRITE_PATH_FLAGS = (
    _DIRECT_WRITE_PATH_FLAGS | _METRICS_OUTPUT_RELATIVE_WRITE_PATH_FLAGS
)
_BASE_REFERENCE_CC_BACKEND_TYPE_FLAG = (
    "--cluster_config_cc_backend_config_type"
)
_REFERENCE_CC_BACKEND_TYPE_FLAGS = frozenset(
    {
        _BASE_REFERENCE_CC_BACKEND_TYPE_FLAG,
        "--cluster_config_prefill_cc_backend_config_type",
        "--cluster_config_decode_cc_backend_config_type",
        "--cluster_config_decode_attn_cc_backend_config_type",
        "--cluster_config_decode_ffn_cc_backend_config_type",
    }
)
_PROTECTED_SIMULATOR_OPTION_FLAGS = frozenset(
    {
        *_REFERENCE_WRITE_PATH_FLAGS,
        *_REFERENCE_CC_BACKEND_TYPE_FLAGS,
        "--enable_cluster_event_logging",
        "--no-enable_cluster_event_logging",
    }
)


class ReferenceObserverBootstrapError(ValueError):
    """Raised when the controlled Reference bootstrap contract is violated."""


@dataclass(frozen=True)
class _ReferenceRuntime:
    base_cluster_scheduler_class: type
    global_batch_end_event_class: type
    main: object


def _require_reference_root(value: str | Path) -> Path:
    root = Path(value).resolve(strict=True)
    expected = REFERENCE_REPO_ROOT.resolve(strict=True)
    if root != expected:
        raise ReferenceObserverBootstrapError(
            "reference_repo_root must equal the pinned Reference repo root: "
            f"expected={expected}, actual={root}"
        )
    return root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _read_git_status(root: Path) -> tuple[bytes, ...]:
    completed = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    )
    return tuple(entry for entry in completed.stdout.split(b"\0") if entry)


def _validate_reference_identity(root: Path) -> dict[str, str]:
    head = _read_git_head(root)
    if head != REFERENCE_GIT_HEAD:
        raise ReferenceObserverBootstrapError(
            "Reference git HEAD mismatch: "
            f"expected={REFERENCE_GIT_HEAD}, actual={head}"
        )
    identity = {"reference_git_head": head}
    for field_name, (relative_path, expected_sha256) in (
        REFERENCE_SOURCE_IDENTITIES.items()
    ):
        actual_sha256 = _sha256_file(root / relative_path)
        if actual_sha256 != expected_sha256:
            raise ReferenceObserverBootstrapError(
                f"Reference source hash mismatch for {relative_path}: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        identity[field_name] = actual_sha256
    status = _read_git_status(root)
    if status:
        raise ReferenceObserverBootstrapError(
            "Reference worktree and index must be clean: "
            f"status={status!r}"
        )
    return identity


def _require_new_sidecar_path(value: str | Path) -> Path:
    path = Path(value)
    if os.path.lexists(path):
        raise ReferenceObserverBootstrapError(
            f"Reference lifecycle sidecar already exists: {path}"
        )
    if not path.parent.is_dir():
        raise ReferenceObserverBootstrapError(
            f"Reference lifecycle sidecar parent is not a directory: {path.parent}"
        )
    return path


def _is_within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=True)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _require_external_write_path(
    path: Path,
    root: Path,
    context: str,
) -> None:
    if _is_within(path, root):
        raise ReferenceObserverBootstrapError(
            f"{context} must resolve outside the Reference repo root: {path}"
        )


def _reject_protected_option_abbreviations(argv: Sequence[str]) -> None:
    for token in argv:
        if not token.startswith("--") or token == "--":
            continue
        option_name = token.split("=", 1)[0]
        matching_flag = next(
            (
                flag
                for flag in _PROTECTED_SIMULATOR_OPTION_FLAGS
                if option_name != flag and flag.startswith(option_name)
            ),
            None,
        )
        if matching_flag is not None:
            raise ReferenceObserverBootstrapError(
                "abbreviated protected simulator option is not allowed: "
                f"actual={option_name}, required={matching_flag}"
            )


def _validate_reference_cc_backend_contract(argv: Sequence[str]) -> None:
    backend_values: dict[str, list[str]] = {
        flag: [] for flag in _REFERENCE_CC_BACKEND_TYPE_FLAGS
    }
    for index, token in enumerate(argv):
        matching_flag = next(
            (
                flag
                for flag in _REFERENCE_CC_BACKEND_TYPE_FLAGS
                if token == flag or token.startswith(f"{flag}=")
            ),
            None,
        )
        if matching_flag is None:
            continue
        if token == matching_flag:
            if (
                index + 1 >= len(argv)
                or not argv[index + 1]
                or argv[index + 1].startswith("--")
            ):
                raise ReferenceObserverBootstrapError(
                    f"{matching_flag} requires a backend value"
                )
            backend_type = argv[index + 1]
        else:
            backend_type = token.split("=", 1)[1]
            if not backend_type:
                raise ReferenceObserverBootstrapError(
                    f"{matching_flag} requires a backend value"
                )
        backend_values[matching_flag].append(backend_type)

    if not backend_values[_BASE_REFERENCE_CC_BACKEND_TYPE_FLAG]:
        raise ReferenceObserverBootstrapError(
            "required Reference CC backend selector is missing: "
            f"{_BASE_REFERENCE_CC_BACKEND_TYPE_FLAG}"
        )

    for flag, values in backend_values.items():
        if values and values[-1] != "analytical":
            raise ReferenceObserverBootstrapError(
                f"{flag.removeprefix('--')} must be analytical"
            )


def _validate_simulator_write_paths(
    argv: Sequence[str],
    root: Path,
) -> None:
    _reject_protected_option_abbreviations(argv)
    _validate_reference_cc_backend_contract(argv)

    path_values: dict[str, list[str]] = {
        flag: [] for flag in _REFERENCE_WRITE_PATH_FLAGS
    }
    for index, token in enumerate(argv):
        matching_flag = next(
            (
                flag
                for flag in _REFERENCE_WRITE_PATH_FLAGS
                if token == flag or token.startswith(f"{flag}=")
            ),
            None,
        )
        if matching_flag is None:
            continue
        if token == matching_flag:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ReferenceObserverBootstrapError(
                    f"{matching_flag} requires a path value"
                )
            raw_path = argv[index + 1]
        else:
            raw_path = token.split("=", 1)[1]
        if not raw_path:
            raise ReferenceObserverBootstrapError(
                f"{matching_flag} requires a path value"
            )
        path_values[matching_flag].append(raw_path)

    for required_flag in sorted(_REQUIRED_EXTERNAL_WRITE_PATH_FLAGS):
        if not path_values[required_flag]:
            raise ReferenceObserverBootstrapError(
                f"required external write path is missing: {required_flag}"
            )

    for flag in sorted(_DIRECT_WRITE_PATH_FLAGS):
        for raw_path in path_values[flag]:
            _require_external_write_path(
                Path(raw_path),
                root,
                flag.removeprefix("--"),
            )

    metrics_output_root = Path(
        path_values["--metrics_config_output_dir"][-1]
    )
    for flag in sorted(_METRICS_OUTPUT_RELATIVE_WRITE_PATH_FLAGS):
        for raw_path in path_values[flag]:
            configured_path = Path(raw_path)
            effective_path = (
                configured_path
                if configured_path.is_absolute()
                else metrics_output_root / "<run>" / configured_path
            )
            _require_external_write_path(
                effective_path,
                root,
                flag.removeprefix("--"),
            )

    cluster_event_logging_enabled = False
    for token in argv:
        if token == "--enable_cluster_event_logging":
            cluster_event_logging_enabled = True
        elif token == "--no-enable_cluster_event_logging":
            cluster_event_logging_enabled = False
    if (
        cluster_event_logging_enabled
        and not path_values["--cluster_event_log_dir"]
    ):
        raise ReferenceObserverBootstrapError(
            "required external write path is missing: --cluster_event_log_dir"
        )


def _normalize_expected_request_ids(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ReferenceObserverBootstrapError(
            "expected_request_ids must not be empty"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in result
    ):
        raise ReferenceObserverBootstrapError(
            "expected_request_ids must contain only non-negative integers"
        )
    if len(set(result)) != len(result):
        raise ReferenceObserverBootstrapError(
            "expected_request_ids must not contain duplicates"
        )
    return tuple(sorted(result))


def _normalize_simulator_argv(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result:
        raise ReferenceObserverBootstrapError(
            "simulator_argv must not be empty"
        )
    if any(
        not isinstance(value, str) or "\x00" in value
        for value in result
    ):
        raise ReferenceObserverBootstrapError(
            "simulator_argv must contain only strings without NUL bytes"
        )
    return result


def _require_fresh_frontier_import() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name == "frontier" or name.startswith("frontier.")
    )
    if imported:
        raise ReferenceObserverBootstrapError(
            f"process already imported frontier modules: {imported}"
        )


def _require_module_path(module: ModuleType, expected_path: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise ReferenceObserverBootstrapError(
            f"Reference module {module.__name__} has no source path"
        )
    actual_path = Path(module_file).resolve(strict=True)
    expected = expected_path.resolve(strict=True)
    if actual_path != expected:
        raise ReferenceObserverBootstrapError(
            f"Reference module {module.__name__} imported from wrong path: "
            f"expected={expected}, actual={actual_path}"
        )


def _import_reference_runtime(root: Path) -> _ReferenceRuntime:
    main_module = importlib.import_module("frontier.main")
    scheduler_module = importlib.import_module(
        "frontier.scheduler.cluster_scheduler.base_cluster_scheduler"
    )
    event_module = importlib.import_module(
        "frontier.events.global_batch_end_event"
    )
    _require_module_path(main_module, root / "frontier/main.py")
    _require_module_path(
        scheduler_module,
        root
        / "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py",
    )
    _require_module_path(
        event_module,
        root / "frontier/events/global_batch_end_event.py",
    )
    main = getattr(main_module, "main", None)
    if not callable(main):
        raise ReferenceObserverBootstrapError(
            "pinned Reference frontier.main.main is not callable"
        )
    return _ReferenceRuntime(
        base_cluster_scheduler_class=getattr(
            scheduler_module,
            "BaseClusterScheduler",
        ),
        global_batch_end_event_class=getattr(
            event_module,
            "GlobalBatchEndEvent",
        ),
        main=main,
    )


def _load_observer_module() -> tuple[ModuleType, Path, str]:
    path = Path(__file__).with_name("reference_lifecycle_observer.py").resolve(
        strict=True
    )
    source_sha256 = _sha256_file(path)
    module_name = f"_frontier_pdaf_reference_observer_{source_sha256}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing, path, source_sha256
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ReferenceObserverBootstrapError(
            f"cannot load Reference lifecycle observer from {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module, path, source_sha256


def _sha256_argv(argv: Sequence[str]) -> str:
    encoded = json.dumps(
        list(argv),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_producer(
    root: Path,
    reference_identity: dict[str, str],
    simulator_argv: Sequence[str],
    observer_module: ModuleType,
    observer_source_sha256: str,
) -> dict[str, object]:
    return {
        "branch_kind": "reference",
        "reference_repo_root": str(root),
        **reference_identity,
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "argv_sha256": _sha256_argv(simulator_argv),
        "observer_source_sha256": observer_source_sha256,
        "bootstrap_source_sha256": _sha256_file(Path(__file__)),
        "candidate_hook": observer_module.CANDIDATE_HOOK,
        "transition_hook": observer_module.TRANSITION_HOOK,
        "transition_contract": observer_module.TRANSITION_CONTRACT,
        "timestamp_contract": observer_module.TIMESTAMP_CONTRACT,
    }


def run_reference_with_observer(
    reference_repo_root: str | Path,
    sidecar_path: str | Path,
    simulator_argv: Sequence[str],
    expected_request_ids: Sequence[int],
) -> dict[str, object]:
    """Run the pinned Reference simulator with direct lifecycle observation."""
    root = _require_reference_root(reference_repo_root)
    reference_identity = _validate_reference_identity(root)
    sidecar = _require_new_sidecar_path(sidecar_path)
    _require_external_write_path(sidecar, root, "sidecar_path")
    expected_ids = _normalize_expected_request_ids(expected_request_ids)
    normalized_argv = _normalize_simulator_argv(simulator_argv)
    _validate_simulator_write_paths(normalized_argv, root)
    _require_fresh_frontier_import()
    observer_module, _, observer_source_sha256 = _load_observer_module()
    observer = observer_module.ReferenceLifecycleObserver()

    original_argv = list(sys.argv)
    original_sys_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    installed = False
    primary_error: BaseException | None = None
    uninstall_error: BaseException | None = None
    identity_error: BaseException | None = None
    result: object = None
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        runtime = _import_reference_runtime(root)
        observer.install(
            runtime.base_cluster_scheduler_class,
            runtime.global_batch_end_event_class,
        )
        installed = True
        sys.argv[:] = ["frontier.main", *normalized_argv]
        result = runtime.main()
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if installed:
            try:
                observer.uninstall()
            except BaseException as error:
                uninstall_error = error
                if primary_error is not None:
                    primary_error.add_note(
                        "Reference observer uninstall also failed: "
                        f"{error!r}"
                    )
        sys.argv[:] = original_argv
        sys.path[:] = original_sys_path
        sys.dont_write_bytecode = original_dont_write_bytecode
        try:
            final_reference_identity = _validate_reference_identity(root)
            if final_reference_identity != reference_identity:
                raise ReferenceObserverBootstrapError(
                    "Reference identity changed during observed simulation"
                )
        except BaseException as error:
            identity_error = error
        if primary_error is not None and identity_error is not None:
            primary_error.add_note(
                "Reference identity audit also failed: "
                f"{identity_error!r}"
            )
        if primary_error is None and uninstall_error is not None:
            if identity_error is not None:
                uninstall_error.add_note(
                    "Reference identity audit also failed: "
                    f"{identity_error!r}"
                )
            raise uninstall_error
        if primary_error is None and identity_error is not None:
            raise identity_error

    if result is not None:
        raise ReferenceObserverBootstrapError(
            f"frontier.main.main must return None, got {result!r}"
        )
    if observer.pending_count:
        raise ReferenceObserverBootstrapError(
            f"cannot finalize with {observer.pending_count} pending candidates"
        )
    observed_ids = tuple(
        int(record["request_id"])
        for record in observer.records
    )
    if observed_ids != expected_ids:
        raise ReferenceObserverBootstrapError(
            "observed first-real-decode request set does not match "
            f"expected_request_ids: expected={expected_ids}, "
            f"actual={observed_ids}"
        )
    producer = _build_producer(
        root,
        reference_identity,
        normalized_argv,
        observer_module,
        observer_source_sha256,
    )
    return observer.write_sidecar(sidecar, producer)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned Reference simulator with direct lifecycle "
            "observation."
        )
    )
    parser.add_argument("--reference-repo-root", required=True)
    parser.add_argument("--sidecar-path", required=True)
    parser.add_argument("--expected-request-count", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse bootstrap controls and forward the remaining simulator argv."""
    parser = _build_argument_parser()
    tokens = tuple(sys.argv[1:] if argv is None else argv)
    if tokens.count("--") != 1:
        parser.error("exactly one -- delimiter is required")
    delimiter_index = tokens.index("--")
    namespace = parser.parse_args(tokens[:delimiter_index])
    simulator_argv = tokens[delimiter_index + 1 :]
    try:
        expected_request_count = int(namespace.expected_request_count)
    except ValueError:
        parser.error("--expected-request-count must be an integer")
    if expected_request_count <= 0:
        parser.error("--expected-request-count must be positive")
    if not simulator_argv:
        parser.error("simulator argv must not be empty")

    run_reference_with_observer(
        namespace.reference_repo_root,
        namespace.sidecar_path,
        simulator_argv,
        tuple(range(expected_request_count)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

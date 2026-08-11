"""Keep all explicitly selected CUDA devices busy with a sustained GEMM loop.

This module is intentionally independent from Frontier's simulator.  It is a
foreground utility used while long-running wall-clock measurements execute on
GPU workers.  The parent validates the explicit CUDA visibility contract,
starts one spawned process per local CUDA device, waits for a post-synchronise
ready barrier, and then monitors every child until a signal or child failure.

``torch`` is imported only inside the execution paths that need it.  The
orchestration APIs accept process, queue, event, clock, output, and torch
dependencies so CPU-only unit tests can exercise the complete contract without
initialising CUDA.
"""

from __future__ import annotations

import argparse
import importlib
import math
import multiprocessing
import os
import queue as queue_module
import re
import signal as signal_module
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


DEFAULT_EXPECTED_DEVICE_COUNT = 8
DEFAULT_MATRIX_SIZE = 4096
DEFAULT_DTYPE = "float16"
DEFAULT_STARTUP_TIMEOUT_S = 30.0
DEFAULT_SYNC_EVERY = 16
DEFAULT_SHUTDOWN_GRACE_S = 5.0
POLL_INTERVAL_S = 0.1


class KeepaliveError(RuntimeError):
    """A validation, worker-startup, or worker-monitoring failure."""


_CLEANUP_ERROR_ATTRIBUTE = "_keepalive_cleanup_error"


def _attach_cleanup_error(
    original_error: BaseException, cleanup_error: BaseException
) -> None:
    setattr(original_error, _CLEANUP_ERROR_ATTRIBUTE, cleanup_error)
    original_error.__cause__ = cleanup_error
    original_error.__suppress_context__ = True


def _get_cleanup_error(error: BaseException) -> BaseException | None:
    value = getattr(error, _CLEANUP_ERROR_ATTRIBUTE, None)
    return value if isinstance(value, BaseException) else None


@dataclass(frozen=True)
class KeepaliveConfig:
    """Validated runtime options for the keepalive utility."""

    expected_device_count: int = DEFAULT_EXPECTED_DEVICE_COUNT
    matrix_size: int = DEFAULT_MATRIX_SIZE
    dtype: str = DEFAULT_DTYPE
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S
    sync_every: int = DEFAULT_SYNC_EVERY
    shutdown_grace_s: float = DEFAULT_SHUTDOWN_GRACE_S

    def __post_init__(self) -> None:
        _require_positive_int(
            self.expected_device_count, "expected_device_count", minimum=1
        )
        _require_positive_int(self.matrix_size, "matrix_size", minimum=2)
        if self.dtype not in {"float16", "bfloat16"}:
            raise ValueError(
                f"dtype must be 'float16' or 'bfloat16', got {self.dtype!r}"
            )
        _require_positive_float(self.startup_timeout_s, "startup_timeout_s")
        _require_positive_int(self.sync_every, "sync_every", minimum=1)
        _require_positive_float(self.shutdown_grace_s, "shutdown_grace_s")


@dataclass(frozen=True)
class WorkerGroup:
    """Processes and IPC handles that passed the startup ready barrier."""

    config: KeepaliveConfig
    visible_devices: tuple[str, ...]
    actual_device_count: int
    processes: tuple[Any, ...]
    ready_queue: Any
    stop_event: Any
    ready_messages: tuple[dict[str, Any], ...]

    @property
    def ready_indices(self) -> tuple[int, ...]:
        return tuple(message["local_index"] for message in self.ready_messages)


def _require_positive_int(value: Any, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _require_positive_float(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")


# CUDA supports decimal ordinals, GPU/MIG UUIDs, and legacy MIG instance paths.
# Numeric selectors are what the H800 launcher uses, while strict UUID parsing
# rejects malformed values before any child process is created.
_UUID_PATTERN = (
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
_DEVICE_TOKEN_RE = re.compile(
    r"^(?:"
    r"[0-9]+|"
    rf"GPU-{_UUID_PATTERN}|"
    rf"MIG-{_UUID_PATTERN}|"
    rf"MIG-GPU-{_UUID_PATTERN}/[0-9]+/[0-9]+"
    r")$"
)


def _print_stdout_line(message: str) -> None:
    print(message, flush=True)


def _print_stderr_line(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _canonical_device_token(token: str) -> str:
    if token.isdecimal():
        # ``0`` and ``00`` identify the same CUDA ordinal and must not bypass
        # duplicate detection.
        return str(int(token, 10))
    if token.startswith("MIG-GPU-"):
        uuid_token, gpu_instance, compute_instance = token.split("/")
        return (
            f"MIG-GPU-{uuid_token.removeprefix('MIG-GPU-').lower()}"
            f"/{int(gpu_instance, 10)}/{int(compute_instance, 10)}"
        )
    prefix, uuid_token = token.split("-", maxsplit=1)
    return f"{prefix.upper()}-{uuid_token.lower()}"


def parse_visible_devices(
    raw: str | None,
    *,
    expected_device_count: int = DEFAULT_EXPECTED_DEVICE_COUNT,
) -> tuple[str, ...]:
    """Parse and validate an explicit ``CUDA_VISIBLE_DEVICES`` value.

    The variable must be present, non-empty, contain no empty fields, and have
    exactly ``expected_device_count`` unique CUDA selectors.  Input order is
    retained because it defines the local index mapping used by workers.
    """

    _require_positive_int(expected_device_count, "expected_device_count", minimum=1)
    if raw is None:
        raise ValueError("CUDA_VISIBLE_DEVICES must be set explicitly")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("CUDA_VISIBLE_DEVICES must be a non-empty list")

    tokens = [part.strip() for part in raw.split(",")]
    if any(not token for token in tokens):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES contains an empty device selector; "
            "provide a comma-separated list"
        )

    canonical: list[str] = []
    for token in tokens:
        if _DEVICE_TOKEN_RE.fullmatch(token) is None:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES contains malformed selector {token!r}"
            )
        canonical_token = _canonical_device_token(token)
        if canonical_token in canonical:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES contains duplicate selector {token!r}"
            )
        canonical.append(canonical_token)

    if len(canonical) != expected_device_count:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES count mismatch: "
            f"expected {expected_device_count}, actual {len(canonical)}"
        )
    return tuple(canonical)


def validate_visible_devices(
    visible_devices: Sequence[str] | str | None,
    *,
    expected_device_count: int = DEFAULT_EXPECTED_DEVICE_COUNT,
) -> tuple[str, ...]:
    """Validate an already materialised device list or parse a raw value."""

    if isinstance(visible_devices, str) or visible_devices is None:
        return parse_visible_devices(
            visible_devices, expected_device_count=expected_device_count
        )
    try:
        raw = ",".join(str(item) for item in visible_devices)
    except TypeError as exc:
        raise ValueError("visible_devices must be a sequence of selectors") from exc
    return parse_visible_devices(raw, expected_device_count=expected_device_count)


def _import_torch() -> Any:
    """Import torch lazily so parsing/unit tests remain CPU-safe."""

    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - exercised on GPU hosts
        raise KeepaliveError(f"unable to import torch: {exc}") from exc


def validate_cuda_environment(
    torch_module: Any,
    visible_devices: Sequence[str] | str,
    *,
    expected_device_count: int = DEFAULT_EXPECTED_DEVICE_COUNT,
) -> int:
    """Validate CUDA availability and the device-count contract.

    No tensor is allocated here; GEMM capability is checked in each worker so
    a failing device is associated with its local index and sibling cleanup is
    deterministic.
    """

    devices = validate_visible_devices(
        visible_devices, expected_device_count=expected_device_count
    )
    if torch_module is None:
        torch_module = _import_torch()
    try:
        cuda = torch_module.cuda
        available = bool(cuda.is_available())
    except Exception as exc:
        raise KeepaliveError(f"unable to query CUDA availability: {exc}") from exc
    if not available:
        raise KeepaliveError("CUDA is unavailable")

    try:
        actual_count = int(cuda.device_count())
    except Exception as exc:
        raise KeepaliveError(f"unable to query CUDA device count: {exc}") from exc
    if actual_count != expected_device_count or actual_count != len(devices):
        raise KeepaliveError(
            "CUDA device count mismatch: "
            f"expected {expected_device_count}, actual {actual_count}, "
            f"visible_ids={len(devices)}"
        )
    return actual_count


def _resolve_dtype(torch_module: Any, dtype_name: str) -> Any:
    try:
        return getattr(torch_module, dtype_name)
    except AttributeError as exc:
        raise KeepaliveError(
            f"torch backend does not expose supported dtype {dtype_name!r}"
        ) from exc


def _worker_error_message(
    *, local_index: int, device_id: str, exc: BaseException
) -> dict[str, Any]:
    return {
        "kind": "error",
        "local_index": local_index,
        "device_id": device_id,
        "pid": os.getpid(),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def worker_main(
    *,
    local_index: int,
    device_id: str,
    matrix_size: int,
    dtype_name: str,
    sync_every: int,
    ready_queue: Any,
    stop_event: Any,
    torch_module: Any | None = None,
) -> int:
    """Run one device's warm-up and sustained GEMM loop.

    The return value is a process exit status.  The spawned entry wrapper turns
    a non-zero value into ``SystemExit`` so the parent observes a failing child.
    """

    try:
        _require_positive_int(local_index, "local_index", minimum=0)
        _require_positive_int(matrix_size, "matrix_size", minimum=2)
        _require_positive_int(sync_every, "sync_every", minimum=1)
        if dtype_name not in {"float16", "bfloat16"}:
            raise ValueError(f"unsupported dtype {dtype_name!r}")
        if torch_module is None:
            torch_module = _import_torch()

        torch_module.cuda.set_device(local_index)
        dtype = _resolve_dtype(torch_module, dtype_name)
        device = f"cuda:{local_index}"
        left = torch_module.randn(
            (matrix_size, matrix_size), device=device, dtype=dtype
        )
        right = torch_module.randn(
            (matrix_size, matrix_size), device=device, dtype=dtype
        )

        # The first GEMM is deliberately synchronised before the ready message;
        # process creation alone is never treated as evidence of GPU work.
        result = left @ right
        torch_module.cuda.synchronize(local_index)
        ready_queue.put(
            {
                "kind": "ready",
                "local_index": local_index,
                "device_id": device_id,
                "pid": os.getpid(),
                "synchronized": True,
            }
        )

        iteration = 0
        while not stop_event.is_set():
            result = left @ right
            iteration += 1
            if iteration % sync_every == 0:
                torch_module.cuda.synchronize(local_index)
        # Keep the result alive until the process exits, preventing an eager
        # backend from discarding the sustained operation's output.
        del result
        return 0
    except Exception as exc:
        try:
            ready_queue.put(_worker_error_message(local_index=local_index, device_id=device_id, exc=exc))
        except Exception:
            # If IPC itself is broken, the non-zero process exit still lets the
            # parent fail fast rather than waiting for an idle worker.
            pass
        return 1


def _worker_process_entry(
    local_index: int,
    device_id: str,
    matrix_size: int,
    dtype_name: str,
    sync_every: int,
    ready_queue: Any,
    stop_event: Any,
) -> None:
    """Spawn-safe wrapper that imports torch inside the child process."""

    status = worker_main(
        local_index=local_index,
        device_id=device_id,
        matrix_size=matrix_size,
        dtype_name=dtype_name,
        sync_every=sync_every,
        ready_queue=ready_queue,
        stop_event=stop_event,
    )
    if status:
        raise SystemExit(status)


def _default_runtime_factories() -> tuple[Callable[..., Any], Callable[[], Any], Callable[[], Any]]:
    context = multiprocessing.get_context("spawn")
    return (
        lambda target, args: context.Process(target=target, args=args),
        context.Queue,
        context.Event,
    )


class KeepaliveRunner:
    """Orchestrate workers while keeping all side effects injectable."""

    def __init__(
        self,
        config: KeepaliveConfig,
        *,
        visible_devices: Sequence[str] | str | None = None,
        environ: Mapping[str, str] | None = None,
        torch_module: Any | None = None,
        process_factory: Callable[[Callable[..., Any], tuple[Any, ...]], Any] | None = None,
        queue_factory: Callable[[], Any] | None = None,
        event_factory: Callable[[], Any] | None = None,
        worker_entrypoint: Callable[..., Any] = _worker_process_entry,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        output: Callable[[str], None] = _print_stdout_line,
        signal_backend: Any = signal_module,
    ) -> None:
        if not isinstance(config, KeepaliveConfig):
            raise TypeError("config must be a KeepaliveConfig")
        self.config = config
        env = os.environ if environ is None else environ
        raw_devices: Sequence[str] | str | None = (
            visible_devices
            if visible_devices is not None
            else env.get("CUDA_VISIBLE_DEVICES")
        )
        self.visible_devices = validate_visible_devices(
            raw_devices, expected_device_count=config.expected_device_count
        )
        self._torch_module = torch_module
        default_process, default_queue, default_event = _default_runtime_factories()
        self._process_factory = process_factory or default_process
        self._queue_factory = queue_factory or default_queue
        self._event_factory = event_factory or default_event
        self._worker_entrypoint = worker_entrypoint
        self._clock = clock
        self._sleep = sleep
        self._output = output
        self._signal_backend = signal_backend
        self._old_signal_handlers: dict[int, Any] = {}
        self._signal_number: int | None = None
        self._active_processes: list[Any] = []
        self._active_stop_event: Any | None = None

    @property
    def signal_number(self) -> int | None:
        return self._signal_number

    def _install_signal_handlers(self, stop_event: Any) -> None:
        if self._signal_backend is None:
            return

        def handle(signum: int, _frame: Any) -> None:
            self._signal_number = signum
            stop_event.set()

        for signum in (
            self._signal_backend.SIGINT,
            self._signal_backend.SIGTERM,
        ):
            self._old_signal_handlers[signum] = self._signal_backend.getsignal(signum)
            self._signal_backend.signal(signum, handle)

    def _restore_signal_handlers(self) -> None:
        if self._signal_backend is None:
            return
        for signum, handler in self._old_signal_handlers.items():
            self._signal_backend.signal(signum, handler)
        self._old_signal_handlers.clear()

    def _new_ipc(self) -> tuple[Any, Any]:
        ready_queue = self._queue_factory()
        stop_event = self._event_factory()
        self._active_stop_event = stop_event
        return ready_queue, stop_event

    def _process_args(
        self, local_index: int, device_id: str, ready_queue: Any, stop_event: Any
    ) -> tuple[Any, ...]:
        return (
            local_index,
            device_id,
            self.config.matrix_size,
            self.config.dtype,
            self.config.sync_every,
            ready_queue,
            stop_event,
        )

    def _join_processes(
        self,
        processes: Sequence[Any],
        *,
        phase: str,
        errors: list[str],
    ) -> None:
        deadline = self._clock() + self.config.shutdown_grace_s
        for index, process in enumerate(processes):
            try:
                if not process.is_alive():
                    continue
                remaining = max(0.0, deadline - self._clock())
                process.join(timeout=remaining)
            except Exception as exc:
                errors.append(f"worker {index} {phase} join failed: {exc}")

    @staticmethod
    def _alive_processes(
        processes: Sequence[Any],
        *,
        phase: str,
        errors: list[str],
    ) -> list[tuple[int, Any]]:
        alive: list[tuple[int, Any]] = []
        for index, process in enumerate(processes):
            try:
                if process.is_alive():
                    alive.append((index, process))
            except Exception as exc:
                errors.append(f"worker {index} {phase} liveness check failed: {exc}")
        return alive

    def _cleanup_processes(self, processes: Sequence[Any], stop_event: Any) -> None:
        """Stop all workers with three group-bounded escalation phases.

        Each phase has one shared ``shutdown_grace_s`` deadline for the whole
        process group, so the total join budget is at most three grace periods
        rather than one grace period per worker.
        """

        errors: list[str] = []
        try:
            stop_event.set()
        except Exception as exc:
            errors.append(f"stop event failed: {exc}")

        self._join_processes(processes, phase="graceful", errors=errors)
        for index, process in self._alive_processes(
            processes, phase="pre-terminate", errors=errors
        ):
            try:
                process.terminate()
            except Exception as exc:
                errors.append(f"worker {index} terminate failed: {exc}")

        self._join_processes(processes, phase="terminate", errors=errors)
        for index, process in self._alive_processes(
            processes, phase="pre-kill", errors=errors
        ):
            try:
                process.kill()
            except Exception as exc:
                errors.append(f"worker {index} kill failed: {exc}")

        self._join_processes(processes, phase="kill", errors=errors)
        residual = self._alive_processes(
            processes, phase="final", errors=errors
        )
        if residual:
            residual_details = ", ".join(
                f"index={index} pid={getattr(process, 'pid', None)}"
                for index, process in residual
            )
            errors.append(f"workers still alive after kill: {residual_details}")
        if errors:
            raise KeepaliveError("; ".join(errors))

    def _raise_worker_failure(self, message: Mapping[str, Any]) -> None:
        local_index = message.get("local_index", "?")
        detail = message.get("error", "unknown worker error")
        raise KeepaliveError(f"worker {local_index} failed: {detail}")

    def _validate_ready_message(
        self,
        message: Any,
        ready_by_index: dict[int, dict[str, Any]],
    ) -> None:
        if not isinstance(message, Mapping):
            raise KeepaliveError(f"invalid worker IPC message: {message!r}")
        if message.get("kind") == "error":
            self._raise_worker_failure(message)
        if message.get("kind") != "ready":
            raise KeepaliveError(f"unexpected worker IPC message: {message!r}")
        local_index = message.get("local_index")
        if (
            isinstance(local_index, bool)
            or not isinstance(local_index, int)
            or not 0 <= local_index < len(self.visible_devices)
        ):
            raise KeepaliveError(f"invalid worker local index in ready message: {message!r}")
        if local_index in ready_by_index:
            raise KeepaliveError(f"duplicate ready message for worker {local_index}")
        expected_device = self.visible_devices[local_index]
        if message.get("device_id") != expected_device:
            raise KeepaliveError(
                f"worker {local_index} reported device {message.get('device_id')!r}; "
                f"expected {expected_device!r}"
            )
        if message.get("synchronized") is not True:
            raise KeepaliveError(
                f"worker {local_index} reported ready before synchronized GEMM"
            )
        ready_by_index[local_index] = dict(message)

    def _check_exited_before_ready(self, processes: Sequence[Any], ready_by_index: Mapping[int, Any]) -> None:
        for index, process in enumerate(processes):
            try:
                alive = bool(process.is_alive())
            except Exception as exc:
                raise KeepaliveError(f"unable to inspect worker {index}: {exc}") from exc
            if not alive and index not in ready_by_index:
                exitcode = getattr(process, "exitcode", None)
                raise KeepaliveError(
                    f"worker {index} exited before ready (exitcode={exitcode!r})"
                )

    def start(self) -> WorkerGroup:
        """Launch children and wait for every post-GEMM ready message."""

        torch_module = self._torch_module if self._torch_module is not None else _import_torch()
        actual_count = validate_cuda_environment(
            torch_module,
            self.visible_devices,
            expected_device_count=self.config.expected_device_count,
        )
        ready_queue, stop_event = self._new_ipc()
        processes: list[Any] = []
        self._active_processes = processes
        try:
            for local_index, device_id in enumerate(self.visible_devices):
                if stop_event.is_set():
                    raise KeepaliveError(
                        "startup interrupted by termination signal"
                    )
                process = self._process_factory(
                    self._worker_entrypoint,
                    self._process_args(local_index, device_id, ready_queue, stop_event),
                )
                processes.append(process)
                process.start()

            deadline = self._clock() + self.config.startup_timeout_s
            ready_by_index: dict[int, dict[str, Any]] = {}
            while len(ready_by_index) < self.config.expected_device_count:
                if stop_event.is_set():
                    raise KeepaliveError("startup interrupted by termination signal")
                self._check_exited_before_ready(processes, ready_by_index)
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise KeepaliveError(
                        "startup timeout waiting for all workers to report "
                        f"ready ({len(ready_by_index)}/{self.config.expected_device_count})"
                    )
                try:
                    message = ready_queue.get(timeout=min(POLL_INTERVAL_S, remaining))
                except queue_module.Empty:
                    continue
                self._validate_ready_message(message, ready_by_index)

            ordered_messages = tuple(ready_by_index[index] for index in range(len(processes)))
            return WorkerGroup(
                config=self.config,
                visible_devices=self.visible_devices,
                actual_device_count=actual_count,
                processes=tuple(processes),
                ready_queue=ready_queue,
                stop_event=stop_event,
                ready_messages=ordered_messages,
            )
        except BaseException as original_error:
            cleanup_error: BaseException | None = None
            try:
                self._cleanup_processes(processes, stop_event)
            except BaseException as exc:
                cleanup_error = exc
            finally:
                self._active_processes = []
                self._active_stop_event = None
            if cleanup_error is not None:
                _attach_cleanup_error(original_error, cleanup_error)
            raise

    def emit_ready_line(self, group: WorkerGroup) -> None:
        pids = ",".join(str(getattr(process, "pid", "?")) for process in group.processes)
        self._output(
            "GPU keepalive ready: "
            f"expected_device_count={group.config.expected_device_count} "
            f"actual_device_count={group.actual_device_count} "
            f"visible_devices={','.join(group.visible_devices)} "
            f"matrix_size={group.config.matrix_size} "
            f"dtype={group.config.dtype} "
            f"worker_pids={pids}"
        )

    def monitor(self, group: WorkerGroup) -> int:
        """Remain in the foreground until deliberate stop or child failure."""

        while True:
            if group.stop_event.is_set():
                self.stop(group)
                if self._signal_number is not None:
                    return 128 + int(self._signal_number)
                return 0
            # Drain all immediately available messages so a post-start worker
            # failure is surfaced without waiting for the next polling interval.
            while True:
                try:
                    message = group.ready_queue.get(timeout=0)
                except queue_module.Empty:
                    break
                if isinstance(message, Mapping) and message.get("kind") == "error":
                    self._raise_worker_failure(message)
                raise KeepaliveError(f"unexpected worker IPC message: {message!r}")
            for index, process in enumerate(group.processes):
                if not process.is_alive():
                    raise KeepaliveError(
                        f"worker {index} exited unexpectedly "
                        f"(exitcode={getattr(process, 'exitcode', None)!r})"
                    )
            self._sleep(POLL_INTERVAL_S)

    def stop(self, group: WorkerGroup) -> None:
        try:
            self._cleanup_processes(group.processes, group.stop_event)
        finally:
            self._active_processes = []
            self._active_stop_event = None

    def run(self) -> int:
        """Run until a signal or worker failure, cleaning siblings on errors."""

        group: WorkerGroup | None = None
        original_new_ipc = self._new_ipc
        # Build the IPC event before installing handlers so a signal received
        # during startup can always request child shutdown.  ``start`` creates
        # the actual event; install handlers immediately after that point.
        try:
            # ``start`` itself is responsible for validation and process launch.
            # A short wrapper lets us install handlers before the first child is
            # started by pre-creating no external resources or CUDA state.
            def create_ipc() -> tuple[Any, Any]:
                handles = original_new_ipc()
                self._install_signal_handlers(handles[1])
                return handles

            self._new_ipc = create_ipc  # type: ignore[method-assign]
            group = self.start()
            self.emit_ready_line(group)
            return self.monitor(group)
        except BaseException as original_error:
            cleanup_error: BaseException | None = None
            try:
                if group is not None:
                    self.stop(group)
                elif self._active_processes and self._active_stop_event is not None:
                    try:
                        self._cleanup_processes(
                            self._active_processes, self._active_stop_event
                        )
                    finally:
                        self._active_processes = []
                        self._active_stop_event = None
            except BaseException as exc:
                cleanup_error = exc
            if cleanup_error is not None:
                _attach_cleanup_error(original_error, cleanup_error)
            if (
                isinstance(original_error, KeepaliveError)
                and self._signal_number is not None
                and _get_cleanup_error(original_error) is None
            ):
                return 128 + int(self._signal_number)
            raise
        finally:
            self._new_ipc = original_new_ipc  # type: ignore[method-assign]
            self._restore_signal_handlers()


def run_keepalive(
    config: KeepaliveConfig,
    *,
    visible_devices: Sequence[str] | str | None = None,
    environ: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
    process_factory: Callable[[Callable[..., Any], tuple[Any, ...]], Any] | None = None,
    queue_factory: Callable[[], Any] | None = None,
    event_factory: Callable[[], Any] | None = None,
    worker_entrypoint: Callable[..., Any] = _worker_process_entry,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    output: Callable[[str], None] = _print_stdout_line,
    error_output: Callable[[str], None] = _print_stderr_line,
    signal_backend: Any = signal_module,
) -> int:
    """Run the utility and translate expected failures into a non-zero code."""

    try:
        runner = KeepaliveRunner(
            config,
            visible_devices=visible_devices,
            environ=environ,
            torch_module=torch_module,
            process_factory=process_factory,
            queue_factory=queue_factory,
            event_factory=event_factory,
            worker_entrypoint=worker_entrypoint,
            clock=clock,
            sleep=sleep,
            output=output,
            signal_backend=signal_backend,
        )
        return runner.run()
    except (KeepaliveError, ValueError) as exc:
        # Environment/list validation is intentionally pure and raises
        # ``ValueError``; the CLI contract still translates it to fail-fast
        # non-zero status at this boundary.
        cleanup_error = _get_cleanup_error(exc)
        message = str(exc)
        if cleanup_error is not None:
            message = f"{message}; cleanup failed: {cleanup_error}"
        error_output(message)
        return 1


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _matrix_size_arg(value: str) -> int:
    parsed = _positive_int_arg(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def _positive_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sustain one explicit GEMM worker on every visible CUDA device."
    )
    parser.add_argument(
        "--expected-device-count",
        type=_positive_int_arg,
        default=DEFAULT_EXPECTED_DEVICE_COUNT,
        help="Required number of explicit CUDA_VISIBLE_DEVICES entries (default: 8).",
    )
    parser.add_argument(
        "--matrix-size",
        type=_matrix_size_arg,
        default=DEFAULT_MATRIX_SIZE,
        help="Square GEMM dimension (default: 4096).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default=DEFAULT_DTYPE,
        help="GEMM dtype (default: float16).",
    )
    parser.add_argument(
        "--startup-timeout-s",
        type=_positive_float_arg,
        default=DEFAULT_STARTUP_TIMEOUT_S,
        help="Bounded worker-ready startup timeout (default: 30 seconds).",
    )
    parser.add_argument(
        "--sync-every",
        type=_positive_int_arg,
        default=DEFAULT_SYNC_EVERY,
        help="Synchronize every N sustained GEMMs (default: 16).",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
    process_factory: Callable[[Callable[..., Any], tuple[Any, ...]], Any] | None = None,
    queue_factory: Callable[[], Any] | None = None,
    event_factory: Callable[[], Any] | None = None,
    worker_entrypoint: Callable[..., Any] = _worker_process_entry,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    output: Callable[[str], None] = _print_stdout_line,
    error_output: Callable[[str], None] = _print_stderr_line,
    signal_backend: Any = signal_module,
) -> int:
    """CLI entry point; no finite-iteration option is intentionally exposed."""

    parser = build_arg_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        config = KeepaliveConfig(
            expected_device_count=args.expected_device_count,
            matrix_size=args.matrix_size,
            dtype=args.dtype,
            startup_timeout_s=args.startup_timeout_s,
            sync_every=args.sync_every,
        )
    except SystemExit as exc:
        # argparse uses SystemExit for both --help and invalid arguments.  Keep
        # the conventional zero help status while returning non-zero failures
        # to callers that invoke ``main`` directly in tests.
        return int(exc.code)
    except (TypeError, ValueError, KeepaliveError) as exc:
        error_output(str(exc))
        return 1

    return run_keepalive(
        config,
        environ=environ,
        torch_module=torch_module,
        process_factory=process_factory,
        queue_factory=queue_factory,
        event_factory=event_factory,
        worker_entrypoint=worker_entrypoint,
        clock=clock,
        sleep=sleep,
        output=output,
        error_output=error_output,
        signal_backend=signal_backend,
    )


if __name__ == "__main__":  # pragma: no cover - exercised by worker launch
    raise SystemExit(main())

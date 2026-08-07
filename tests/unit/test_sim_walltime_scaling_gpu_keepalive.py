"""CPU-safe contract tests for the eight-device GEMM keepalive utility."""

from __future__ import annotations

import builtins
import multiprocessing
import queue
import signal
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import pytest

from tests.performance.sim_walltime_scaling import gpu_keepalive_gemm as keepalive


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        count: int = 8,
        trace: list[str] | None = None,
    ) -> None:
        self.available = available
        self.count = count
        self.trace = trace
        self.selected: list[int] = []
        self.sync_calls: list[int | None] = []

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def set_device(self, index: int) -> None:
        self.selected.append(index)

    def synchronize(self, device: int | None = None) -> None:
        self.sync_calls.append(device)
        if self.trace is not None:
            self.trace.append(f"synchronize:{device}")


class FakeTensor:
    def __init__(self, trace: list[str], name: str) -> None:
        self.trace = trace
        self.name = name

    def __matmul__(self, other: "FakeTensor") -> "FakeTensor":
        del other
        self.trace.append(f"gemm:{self.name}")
        return FakeTensor(self.trace, "result")


class FakeTorch:
    float16 = "float16-object"
    bfloat16 = "bfloat16-object"

    def __init__(
        self,
        cuda: FakeCuda | None = None,
        *,
        trace: list[str] | None = None,
    ) -> None:
        self.trace = trace if trace is not None else []
        self.cuda = cuda or FakeCuda(trace=self.trace)

    def randn(self, shape: tuple[int, int], *, device: str, dtype: Any) -> FakeTensor:
        self.trace.append(f"randn:{shape}:{device}:{dtype}")
        return FakeTensor(self.trace, device)


class FakeStopEvent:
    def __init__(self, *, set_after_checks: int | None = None) -> None:
        self.set_calls = 0
        self.checks = 0
        self._set = False
        self.set_after_checks = set_after_checks

    def set(self) -> None:
        self.set_calls += 1
        self._set = True

    def is_set(self) -> bool:
        self.checks += 1
        if self.set_after_checks is not None and self.checks > self.set_after_checks:
            self._set = True
        return self._set


class FakeQueue:
    def __init__(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        trace: list[str] | None = None,
    ) -> None:
        self.messages = deque(messages or [])
        self.put_messages: list[dict[str, Any]] = []
        self.trace = trace

    def put(self, message: dict[str, Any]) -> None:
        self.put_messages.append(message)
        self.messages.append(message)
        if self.trace is not None:
            self.trace.append(f"queue:{message.get('kind')}")

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        del timeout
        if not self.messages:
            raise queue.Empty
        return self.messages.popleft()


class FakeClock:
    def __init__(self, values: list[float] | None = None) -> None:
        self.values = deque(values or [0.0])
        self.last = 0.0

    def __call__(self) -> float:
        if self.values:
            self.last = self.values.popleft()
        else:
            self.last += 1.0
        return self.last


@dataclass
class FakeProcess:
    target: Any
    args: tuple[Any, ...]
    pid: int
    alive: bool = True
    exitcode: int | None = None
    ignore_terminate: bool = False
    ignore_kill: bool = False

    def __post_init__(self) -> None:
        self.started = False
        self.terminated = False
        self.killed = False
        self.join_calls = 0

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.join_calls += 1

    def terminate(self) -> None:
        self.terminated = True
        if self.ignore_terminate:
            return
        self.alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self.killed = True
        if self.ignore_kill:
            return
        self.alive = False
        self.exitcode = -9


class FlushTrackingStream:
    def __init__(self) -> None:
        self.contents: list[str] = []
        self.flush_calls = 0

    def write(self, value: str) -> int:
        self.contents.append(value)
        return len(value)

    def flush(self) -> None:
        self.flush_calls += 1


class ProcessFactory:
    def __init__(self) -> None:
        self.processes: list[FakeProcess] = []

    def __call__(self, target: Any, args: tuple[Any, ...]) -> FakeProcess:
        process = FakeProcess(target=target, args=args, pid=10_000 + len(self.processes))
        self.processes.append(process)
        return process


class FakeSignalModule:
    SIGINT = 2
    SIGTERM = 15

    def __init__(self) -> None:
        self.handlers: dict[int, Any] = {}

    def getsignal(self, signum: int) -> Any:
        return self.handlers.get(signum, "default")

    def signal(self, signum: int, handler: Any) -> None:
        self.handlers[signum] = handler


def _ignore_sigterm_until_killed(ready: Any) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(0.1)


def _visible(count: int = 8) -> tuple[str, ...]:
    return tuple(str(index) for index in range(count))


def _uuid(index: int) -> str:
    return f"00000000-0000-0000-0000-{index:012x}"


@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", "0,,1", "0,x,1", "-1,0", "0,1,0", "0,00"],
)
def test_parse_visible_devices_rejects_missing_empty_malformed_and_duplicate(raw: str | None) -> None:
    with pytest.raises(ValueError):
        keepalive.parse_visible_devices(raw)


def test_parse_visible_devices_preserves_explicit_local_order() -> None:
    assert keepalive.parse_visible_devices("7, 3, 5, 1, 0, 2, 6, 4") == (
        "7",
        "3",
        "5",
        "1",
        "0",
        "2",
        "6",
        "4",
    )


@pytest.mark.parametrize(
    "devices",
    [
        tuple(f"GPU-{_uuid(index)}" for index in range(8)),
        tuple(f"MIG-{_uuid(index)}" for index in range(8)),
        tuple(f"MIG-GPU-{_uuid(index)}/1/2" for index in range(8)),
    ],
)
def test_parse_visible_devices_accepts_cuda_uuid_forms(
    devices: tuple[str, ...],
) -> None:
    assert keepalive.parse_visible_devices(",".join(devices)) == devices


@pytest.mark.parametrize(
    "prefix",
    ["GPU-not-a-real-uuid-", "MIG-not-a-real-uuid-"],
)
def test_parse_visible_devices_rejects_malformed_uuid_forms(prefix: str) -> None:
    devices = tuple(f"{prefix}{index}" for index in range(8))

    with pytest.raises(ValueError, match="malformed selector"):
        keepalive.parse_visible_devices(",".join(devices))


def test_parse_visible_devices_requires_expected_count() -> None:
    with pytest.raises(ValueError, match="expected 8.*actual 2"):
        keepalive.parse_visible_devices("0,1")


def test_module_import_does_not_import_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []
    original_import = builtins.__import__

    def tracking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            imported.append(name)
            raise AssertionError("unit test must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)
    keepalive.parse_visible_devices("0,1,2,3,4,5,6,7")
    assert imported == []


@pytest.mark.parametrize(
    "cuda,match",
    [
        (FakeCuda(available=False), "CUDA is unavailable"),
        (FakeCuda(available=True, count=7), "expected 8.*actual 7"),
    ],
)
def test_validate_cuda_environment_fails_fast(cuda: FakeCuda, match: str) -> None:
    with pytest.raises(keepalive.KeepaliveError, match=match):
        keepalive.validate_cuda_environment(
            FakeTorch(cuda), _visible(), expected_device_count=8
        )


def test_worker_binds_gemm_synchronizes_then_signals_ready() -> None:
    trace: list[str] = []
    torch = FakeTorch(trace=trace)
    stop_event = FakeStopEvent(set_after_checks=2)
    messages = FakeQueue(trace=trace)

    keepalive.worker_main(
        local_index=3,
        device_id="17",
        matrix_size=8,
        dtype_name="bfloat16",
        sync_every=1,
        ready_queue=messages,
        stop_event=stop_event,
        torch_module=torch,
    )

    assert torch.cuda.selected == [3]
    assert torch.trace[:3] == [
        "randn:(8, 8):cuda:3:bfloat16-object",
        "randn:(8, 8):cuda:3:bfloat16-object",
        "gemm:cuda:3",
    ]
    assert torch.cuda.sync_calls[0] == 3
    assert messages.put_messages[0]["kind"] == "ready"
    assert messages.put_messages[0]["synchronized"] is True
    assert trace.index("synchronize:3") < trace.index("queue:ready")
    assert torch.trace.index("gemm:cuda:3") < len(torch.trace)


def test_start_launches_one_worker_per_explicit_device_and_waits_for_all_ready() -> None:
    factory = ProcessFactory()
    messages = FakeQueue(
        [
            {
                "kind": "ready",
                "local_index": index,
                "device_id": device,
                "pid": 10_000 + index,
                "synchronized": True,
            }
            for index, device in enumerate(_visible())
        ]
    )
    event = FakeStopEvent()
    output: list[str] = []
    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8, startup_timeout_s=2.0),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        process_factory=factory,
        queue_factory=lambda: messages,
        event_factory=lambda: event,
        clock=FakeClock([0.0] * 20),
        sleep=lambda _: None,
        output=output.append,
    )

    group = runner.start()

    assert len(factory.processes) == 8
    assert [process.args[0] for process in factory.processes] == list(range(8))
    assert [process.args[1] for process in factory.processes] == list(_visible())
    assert output == []
    assert group.ready_indices == tuple(range(8))
    runner.stop(group)


def test_ready_line_is_emitted_only_after_all_workers_are_ready() -> None:
    factory = ProcessFactory()
    messages = FakeQueue(
        [
            {
                "kind": "ready",
                "local_index": index,
                "device_id": device,
                "pid": 20_000 + index,
                "synchronized": True,
            }
            for index, device in enumerate(_visible())
        ]
    )
    output: list[str] = []
    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=16, dtype="float16"),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        process_factory=factory,
        queue_factory=lambda: messages,
        event_factory=FakeStopEvent,
        clock=FakeClock([0.0] * 20),
        sleep=lambda _: None,
        output=output.append,
    )
    group = runner.start()
    assert output == []
    runner.emit_ready_line(group)
    assert len(output) == 1
    assert "expected_device_count=8" in output[0]
    assert "actual_device_count=8" in output[0]
    assert "visible_devices=0,1,2,3,4,5,6,7" in output[0]
    assert "matrix_size=16" in output[0]
    assert "dtype=float16" in output[0]
    assert "worker_pids=" in output[0]
    runner.stop(group)


def test_default_ready_output_is_flushed(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FlushTrackingStream()
    monkeypatch.setattr(keepalive.sys, "stdout", stream)
    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        signal_backend=None,
    )
    group = keepalive.WorkerGroup(
        config=runner.config,
        visible_devices=_visible(),
        actual_device_count=8,
        processes=tuple(
            FakeProcess(target=None, args=(), pid=40_000 + index, alive=False)
            for index in range(8)
        ),
        ready_queue=FakeQueue(),
        stop_event=FakeStopEvent(),
        ready_messages=tuple(
            {
                "kind": "ready",
                "local_index": index,
                "device_id": device,
                "pid": 40_000 + index,
                "synchronized": True,
            }
            for index, device in enumerate(_visible())
        ),
    )

    runner.emit_ready_line(group)

    assert "GPU keepalive ready:" in "".join(stream.contents)
    assert stream.flush_calls == 1


def test_cleanup_kills_worker_that_ignores_terminate() -> None:
    factory = ProcessFactory()
    messages = FakeQueue(
        [
            {
                "kind": "ready",
                "local_index": index,
                "device_id": device,
                "pid": 50_000 + index,
                "synchronized": True,
            }
            for index, device in enumerate(_visible())
        ]
    )
    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8, shutdown_grace_s=0.01),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        process_factory=factory,
        queue_factory=lambda: messages,
        event_factory=FakeStopEvent,
        clock=FakeClock([0.0] * 20),
        sleep=lambda _: None,
        signal_backend=None,
    )
    group = runner.start()
    stubborn = factory.processes[0]
    stubborn.ignore_terminate = True

    runner.stop(group)

    assert stubborn.terminated is True
    assert stubborn.killed is True
    assert stubborn.is_alive() is False


def test_cleanup_deadline_is_bounded_per_phase_for_the_whole_group() -> None:
    class MutableClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = MutableClock()
    processes = [
        FakeProcess(
            target=None,
            args=(),
            pid=55_000 + index,
            ignore_terminate=True,
        )
        for index in range(8)
    ]
    for process in processes:
        original_join = process.join

        def consume_timeout(
            timeout: float | None = None,
            *,
            original_join: Any = original_join,
        ) -> None:
            original_join(timeout)
            clock.now += float(timeout or 0.0)

        process.join = consume_timeout  # type: ignore[method-assign]

    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8, shutdown_grace_s=5.0),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        clock=clock,
        signal_backend=None,
    )

    runner._cleanup_processes(processes, FakeStopEvent())

    assert all(process.killed for process in processes)
    assert clock.now <= 3 * runner.config.shutdown_grace_s


def test_cleanup_fails_if_worker_survives_kill() -> None:
    process = FakeProcess(
        target=None,
        args=(),
        pid=60_000,
        ignore_terminate=True,
        ignore_kill=True,
    )
    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(
            expected_device_count=1,
            matrix_size=8,
            shutdown_grace_s=0.01,
        ),
        visible_devices=("0",),
        torch_module=FakeTorch(FakeCuda(count=1)),
        signal_backend=None,
    )
    group = keepalive.WorkerGroup(
        config=runner.config,
        visible_devices=("0",),
        actual_device_count=1,
        processes=(process,),
        ready_queue=FakeQueue(),
        stop_event=FakeStopEvent(),
        ready_messages=(
            {
                "kind": "ready",
                "local_index": 0,
                "device_id": "0",
                "pid": process.pid,
                "synchronized": True,
            },
        ),
    )

    with pytest.raises(keepalive.KeepaliveError, match="still alive.*60000"):
        runner.stop(group)


def test_start_cleanup_failure_preserves_original_base_exception() -> None:
    process = FakeProcess(
        target=None,
        args=(),
        pid=61_000,
        ignore_terminate=True,
        ignore_kill=True,
    )

    def start() -> None:
        process.started = True
        raise KeyboardInterrupt("original interrupt")

    process.start = start  # type: ignore[method-assign]
    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(
            expected_device_count=1,
            matrix_size=8,
            shutdown_grace_s=0.01,
        ),
        visible_devices=("0",),
        torch_module=FakeTorch(FakeCuda(count=1)),
        process_factory=lambda _target, _args: process,
        queue_factory=FakeQueue,
        event_factory=FakeStopEvent,
        signal_backend=None,
    )

    with pytest.raises(KeyboardInterrupt, match="original interrupt") as caught:
        runner.start()

    assert isinstance(caught.value.__cause__, keepalive.KeepaliveError)
    assert "still alive after kill" in str(caught.value.__cause__)


def test_run_cleanup_failure_preserves_original_output_exception() -> None:
    factory = ProcessFactory()
    messages = FakeQueue(
        [
            {
                "kind": "ready",
                "local_index": index,
                "device_id": device,
                "pid": 62_000 + index,
                "synchronized": True,
            }
            for index, device in enumerate(_visible())
        ]
    )

    def stubborn_factory(target: Any, args: tuple[Any, ...]) -> FakeProcess:
        process = factory(target, args)
        if len(factory.processes) == 1:
            process.ignore_terminate = True
            process.ignore_kill = True
        return process

    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8, shutdown_grace_s=0.01),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        process_factory=stubborn_factory,
        queue_factory=lambda: messages,
        event_factory=FakeStopEvent,
        clock=FakeClock([0.0] * 40),
        sleep=lambda _: None,
        output=lambda _message: (_ for _ in ()).throw(
            BrokenPipeError("original output failure")
        ),
        signal_backend=None,
    )

    with pytest.raises(BrokenPipeError, match="original output failure") as caught:
        runner.run()

    assert isinstance(caught.value.__cause__, keepalive.KeepaliveError)
    assert "still alive after kill" in str(caught.value.__cause__)


def test_run_keepalive_reports_original_and_cleanup_failures_once() -> None:
    errors: list[str] = []
    process = FakeProcess(
        target=None,
        args=(),
        pid=63_000,
        ignore_terminate=True,
        ignore_kill=True,
    )

    def start() -> None:
        process.started = True
        raise keepalive.KeepaliveError("original worker launch failure")

    process.start = start  # type: ignore[method-assign]

    status = keepalive.run_keepalive(
        keepalive.KeepaliveConfig(
            expected_device_count=1,
            matrix_size=8,
            shutdown_grace_s=0.01,
        ),
        visible_devices=("0",),
        torch_module=FakeTorch(FakeCuda(count=1)),
        process_factory=lambda _target, _args: process,
        queue_factory=FakeQueue,
        event_factory=FakeStopEvent,
        error_output=errors.append,
        signal_backend=None,
    )

    assert status == 1
    assert errors == [
        "original worker launch failure; cleanup failed: "
        "workers still alive after kill: index=0 pid=63000"
    ]


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="requires POSIX signals")
def test_cleanup_kills_real_process_that_ignores_sigterm() -> None:
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    process = context.Process(target=_ignore_sigterm_until_killed, args=(ready,))
    process.start()
    assert ready.wait(timeout=2.0)
    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(
            expected_device_count=1,
            matrix_size=8,
            shutdown_grace_s=0.5,
        ),
        visible_devices=("0",),
        torch_module=FakeTorch(FakeCuda(count=1)),
        signal_backend=None,
    )
    group = keepalive.WorkerGroup(
        config=runner.config,
        visible_devices=("0",),
        actual_device_count=1,
        processes=(process,),
        ready_queue=FakeQueue(),
        stop_event=context.Event(),
        ready_messages=(
            {
                "kind": "ready",
                "local_index": 0,
                "device_id": "0",
                "pid": process.pid,
                "synchronized": True,
            },
        ),
    )

    try:
        runner.stop(group)
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)

    assert process.is_alive() is False
    assert process.exitcode == -signal.SIGKILL


def test_output_failure_still_cleans_all_workers() -> None:
    factory = ProcessFactory()
    messages = FakeQueue(
        [
            {
                "kind": "ready",
                "local_index": index,
                "device_id": device,
                "pid": 70_000 + index,
                "synchronized": True,
            }
            for index, device in enumerate(_visible())
        ]
    )

    def fail_output(_message: str) -> None:
        raise BrokenPipeError("ready consumer closed")

    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        process_factory=factory,
        queue_factory=lambda: messages,
        event_factory=FakeStopEvent,
        clock=FakeClock([0.0] * 20),
        sleep=lambda _: None,
        output=fail_output,
        signal_backend=None,
    )

    with pytest.raises(BrokenPipeError, match="ready consumer closed"):
        runner.run()
    assert all(not process.is_alive() for process in factory.processes)


def test_signal_during_startup_returns_signal_exit_code_and_cleans_workers() -> None:
    signals = FakeSignalModule()
    factory = ProcessFactory()
    messages = FakeQueue([])
    first_process = True

    def signalling_factory(target: Any, args: tuple[Any, ...]) -> FakeProcess:
        nonlocal first_process
        process = factory(target, args)
        original_start = process.start
        should_signal = first_process
        first_process = False

        def start() -> None:
            original_start()
            if should_signal:
                signals.handlers[signals.SIGTERM](signals.SIGTERM, None)

        process.start = start  # type: ignore[method-assign]
        return process

    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        process_factory=signalling_factory,
        queue_factory=lambda: messages,
        event_factory=FakeStopEvent,
        clock=FakeClock([0.0] * 20),
        sleep=lambda _: None,
        signal_backend=signals,
    )

    status = runner.run()

    assert status == 128 + signals.SIGTERM
    assert len(factory.processes) == 1
    assert all(not process.is_alive() for process in factory.processes)
    assert signals.handlers[signals.SIGINT] == "default"
    assert signals.handlers[signals.SIGTERM] == "default"


def test_signal_after_ready_returns_signal_exit_code_and_cleans_workers() -> None:
    signals = FakeSignalModule()
    factory = ProcessFactory()
    messages = FakeQueue(
        [
            {
                "kind": "ready",
                "local_index": index,
                "device_id": device,
                "pid": 80_000 + index,
                "synchronized": True,
            }
            for index, device in enumerate(_visible())
        ]
    )

    def signal_after_ready(_message: str) -> None:
        signals.handlers[signals.SIGINT](signals.SIGINT, None)

    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        process_factory=factory,
        queue_factory=lambda: messages,
        event_factory=FakeStopEvent,
        clock=FakeClock([0.0] * 20),
        sleep=lambda _: None,
        output=signal_after_ready,
        signal_backend=signals,
    )

    status = runner.run()

    assert status == 128 + signals.SIGINT
    assert all(not process.is_alive() for process in factory.processes)


def test_post_ready_worker_error_cleans_siblings() -> None:
    factory = ProcessFactory()
    messages = FakeQueue(
        [
            *[
                {
                    "kind": "ready",
                    "local_index": index,
                    "device_id": device,
                    "pid": 90_000 + index,
                    "synchronized": True,
                }
                for index, device in enumerate(_visible())
            ],
            {
                "kind": "error",
                "local_index": 3,
                "device_id": "3",
                "pid": 90_003,
                "error": "RuntimeError: CUDA launch failed",
            },
        ]
    )
    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        process_factory=factory,
        queue_factory=lambda: messages,
        event_factory=FakeStopEvent,
        clock=FakeClock([0.0] * 20),
        sleep=lambda _: None,
        signal_backend=None,
    )

    with pytest.raises(keepalive.KeepaliveError, match="worker 3 failed"):
        runner.run()

    assert all(not process.is_alive() for process in factory.processes)


def test_startup_timeout_terminates_all_workers() -> None:
    factory = ProcessFactory()
    messages = FakeQueue([])
    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8, startup_timeout_s=0.5),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        process_factory=factory,
        queue_factory=lambda: messages,
        event_factory=FakeStopEvent,
        clock=FakeClock([0.0, 1.0]),
        sleep=lambda _: None,
    )

    with pytest.raises(keepalive.KeepaliveError, match="startup timeout"):
        runner.start()
    assert all(process.terminated for process in factory.processes)


def test_unexpected_worker_exit_terminates_siblings() -> None:
    factory = ProcessFactory()
    processes = factory.processes
    messages = FakeQueue(
        [
            {
                "kind": "ready",
                "local_index": index,
                "device_id": device,
                "pid": 30_000 + index,
                "synchronized": True,
            }
            for index, device in enumerate(_visible())
        ]
    )

    def process_factory(target: Any, args: tuple[Any, ...]) -> FakeProcess:
        process = factory(target, args)
        if len(factory.processes) == 1:
            process.alive = False
            process.exitcode = 1
        return process

    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8),
        visible_devices=_visible(),
        torch_module=FakeTorch(),
        process_factory=process_factory,
        queue_factory=lambda: messages,
        event_factory=FakeStopEvent,
        clock=FakeClock([0.0, 0.1]),
        sleep=lambda _: None,
    )
    with pytest.raises(keepalive.KeepaliveError, match="worker"):
        runner.run()
    assert len(processes) == 8
    assert all(process.terminated or not process.alive for process in processes)


@pytest.mark.parametrize(
    "argv",
    [
        ["--expected-device-count", "0"],
        ["--matrix-size", "1"],
        ["--startup-timeout-s", "0"],
        ["--sync-every", "0"],
        ["--dtype", "float32"],
    ],
)
def test_cli_rejects_invalid_numeric_or_dtype_inputs(argv: list[str]) -> None:
    assert keepalive.main(argv, environ={"CUDA_VISIBLE_DEVICES": ",".join(_visible())}) != 0


def test_run_keepalive_returns_nonzero_for_missing_cuda_visible_devices() -> None:
    assert keepalive.main([], environ={}) != 0


def test_run_keepalive_reports_keepalive_error_once() -> None:
    errors: list[str] = []

    status = keepalive.run_keepalive(
        keepalive.KeepaliveConfig(matrix_size=8),
        visible_devices=_visible(),
        torch_module=FakeTorch(FakeCuda(available=False)),
        error_output=errors.append,
        signal_backend=None,
    )

    assert status == 1
    assert errors == ["CUDA is unavailable"]


def test_default_error_output_is_flushed(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FlushTrackingStream()
    monkeypatch.setattr(keepalive.sys, "stderr", stream)

    status = keepalive.run_keepalive(
        keepalive.KeepaliveConfig(matrix_size=8),
        visible_devices=_visible(),
        torch_module=FakeTorch(FakeCuda(available=False)),
        signal_backend=None,
    )

    assert status == 1
    assert "CUDA is unavailable" in "".join(stream.contents)
    assert stream.flush_calls == 1


def test_run_restores_ipc_factory_when_validation_fails_before_ipc_creation() -> None:
    runner = keepalive.KeepaliveRunner(
        keepalive.KeepaliveConfig(matrix_size=8),
        visible_devices=_visible(),
        torch_module=FakeTorch(FakeCuda(available=False)),
        signal_backend=None,
    )
    original_new_ipc = runner._new_ipc

    with pytest.raises(keepalive.KeepaliveError, match="CUDA is unavailable"):
        runner.run()

    assert runner._new_ipc == original_new_ipc

"""CPU lifecycle checks for profiling timer ownership and failed measurements."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from frontier.profiling.common.device_timer import DeviceTimer
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.utils import ProfileMethod
from frontier.profiling.utils.singleton import Singleton


METHODS = (
    ProfileMethod.CUDA_EVENT,
    ProfileMethod.DEVICE_EVENT,
    ProfileMethod.PERF_COUNTER,
    ProfileMethod.RECORD_FUNCTION,
    ProfileMethod.KINETO,
)


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    monkeypatch.setattr(Singleton, "_instances", {})


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = []
    exits = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            calls.append("event_create")

        def record(self):
            calls.append("event_record")

        def elapsed_time(self, other):
            assert other is not self
            return 2.5

    class Context:
        def __init__(self, kind, callback=None):
            self.kind = kind
            self.callback = callback

        def __enter__(self):
            calls.append(f"{self.kind}_enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append(f"{self.kind}_exit")
            exits.append((self.kind, exc_type, exc_value, traceback))
            if self.callback:
                self.callback(SimpleNamespace(events=lambda: [
                    SimpleNamespace(name="keep_kernel", device_time_total=2500.0),
                    SimpleNamespace(name="other_kernel", device_time_total=9000.0),
                ]))

    def profile(*, activities, on_trace_ready):
        assert activities == ["cuda"]
        calls.append("profiler_create")
        return Context("profiler", on_trace_ready)

    def record_function(name):
        assert name == "vidur_operation"
        return Context("record_function")

    profiler = ModuleType("torch.profiler")
    profiler.ProfilerActivity = SimpleNamespace(CUDA="cuda")
    profiler.profile = profile
    profiler.record_function = record_function
    torch = ModuleType("torch")
    torch.profiler = profiler
    torch.cuda = SimpleNamespace(Event=Event, synchronize=lambda: calls.append("synchronize"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.profiler", profiler)
    monkeypatch.setattr(DeviceTimer, "_torch", staticmethod(lambda: torch))
    ticks = iter((10.0, 10.004, 20.0, 20.004))
    monkeypatch.setattr(
        "frontier.profiling.common.device_timer.time.perf_counter", lambda: next(ticks)
    )
    return SimpleNamespace(calls=calls, exits=exits)


def test_store_accessor_does_not_create_a_store():
    assert TimerStatsStore.get_existing() is None
    store = TimerStatsStore(profile_method="device_event")
    assert TimerStatsStore.get_existing() is store


def test_unnamed_timer_leaves_named_timer_enabled(fake_runtime):
    with DeviceTimer(None):
        pass
    assert TimerStatsStore.get_existing() is None
    assert fake_runtime.calls == []
    timer = DeviceTimer("operation")
    with timer:
        pass
    assert not timer.disabled
    assert timer.timer_stats_store.get_times() == {"operation": [2.5]}


@pytest.mark.parametrize("method", METHODS)
def test_existing_store_owns_method_and_success_lifecycle(fake_runtime, method):
    store = TimerStatsStore(profile_method=method.value)
    explicit_method = (
        ProfileMethod.CUDA_EVENT if method is not ProfileMethod.CUDA_EVENT
        else ProfileMethod.DEVICE_EVENT
    )
    timer = DeviceTimer(
        "operation", profile_method=explicit_method.value, filter_str="keep_"
    )
    assert timer.timer_stats_store is store
    assert store.profile_method is method
    with timer:
        fake_runtime.calls.append("body")

    if method in {ProfileMethod.CUDA_EVENT, ProfileMethod.DEVICE_EVENT}:
        assert fake_runtime.calls == [
            "event_create", "event_record", "body", "event_create", "event_record"
        ]
        assert store.get_times() == {"operation": [2.5]}
    elif method is ProfileMethod.PERF_COUNTER:
        assert fake_runtime.calls == ["synchronize", "body", "synchronize"]
        assert store.get_times()["operation"] == pytest.approx([4.0])
    elif method is ProfileMethod.RECORD_FUNCTION:
        assert fake_runtime.calls == ["record_function_enter", "body", "record_function_exit"]
        assert store.get_times() == {}
    else:
        assert fake_runtime.calls == ["profiler_create", "profiler_enter", "body", "profiler_exit"]
        assert store.get_times() == {"operation": [2.5]}


@pytest.mark.parametrize("method", METHODS)
def test_standalone_named_timer_honors_explicit_method(fake_runtime, method):
    timer = DeviceTimer("operation", profile_method=method.value, filter_str="keep_")
    assert not timer.disabled
    assert timer.timer_stats_store.profile_method is method
    with timer:
        pass
    assert bool(fake_runtime.calls)


@pytest.mark.parametrize("alias, expected", [
    ("cuda", ProfileMethod.CUDA_EVENT),
    ("kernel_only", ProfileMethod.RECORD_FUNCTION),
])
def test_legacy_method_aliases_normalize_before_timing(fake_runtime, alias, expected):
    timer = DeviceTimer("operation", profile_method=alias)
    assert timer.timer_stats_store.profile_method is expected
    with timer:
        pass


@pytest.mark.parametrize("method", METHODS)
def test_existing_disabled_store_remains_disabled(monkeypatch, method):
    store = TimerStatsStore(profile_method=method.value, disabled=True)

    def unexpected_runtime():
        raise AssertionError("Disabled timers must not construct or enter GPU contexts")

    monkeypatch.setattr(DeviceTimer, "_torch", staticmethod(unexpected_runtime))
    timer = DeviceTimer("operation", profile_method="device_event")
    assert timer.disabled
    assert timer.timer_stats_store is store
    with timer:
        pass
    assert store.profile_method is method
    assert store.get_times() == {}


@pytest.mark.parametrize("method", METHODS)
def test_failed_measurement_unwinds_without_retaining_sample(fake_runtime, method):
    store = TimerStatsStore(profile_method=method.value)
    store.record_time("operation", 7.0)
    failure = RuntimeError("injected operator failure")
    timer = DeviceTimer("operation", filter_str="keep_")
    with pytest.raises(RuntimeError, match="injected operator failure") as caught:
        with timer:
            raise failure
    assert caught.value is failure
    assert store.get_times() == {"operation": [7.0]}
    if method in {ProfileMethod.RECORD_FUNCTION, ProfileMethod.KINETO}:
        assert len(fake_runtime.exits) == 1
        _, exc_type, exc_value, traceback = fake_runtime.exits[0]
        assert exc_type is RuntimeError
        assert exc_value is failure
        assert traceback is not None


@pytest.mark.parametrize("method,disabled", [
    (None, False), ("device_event", False), ("perf_counter", False),
    ("kineto", False), ("device_event", True),
])
def test_gdn_campaign_validates_effective_owner_before_native_work(monkeypatch, method, disabled):
    import builtins
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.gdn.vllm_wrapper import VllmQwen35GDNWrapper

    model = ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")
    owner = TimerStatsStore(profile_method=method, disabled=disabled) if method else None
    if owner:
        owner.record_time("previous", 1.25)
    original_import = builtins.__import__

    class NativeBoundaryReached(RuntimeError):
        pass

    def stop_native(name, *args, **kwargs):
        if name in {"torch", "triton", "vllm"}:
            raise NativeBoundaryReached("native boundary reached")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stop_native)
    incompatible = disabled or method in {"perf_counter", "kineto"}
    expected_error = ValueError if incompatible else NativeBoundaryReached
    with pytest.raises(expected_error, match="GDN timer owner" if incompatible else "native boundary"):
        VllmQwen35GDNWrapper(
            frontier_model_config=model, model_path="unused", device_name="mi355x",
            profile_method="DEVICE_EVENT",
        )
    effective = Singleton._instances[TimerStatsStore]
    if owner:
        assert effective is owner
        assert effective.get_times() == {"previous": [1.25]}
    else:
        assert effective.profile_method is ProfileMethod.DEVICE_EVENT
        assert not effective.disabled

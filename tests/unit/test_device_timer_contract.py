from __future__ import annotations

from types import SimpleNamespace

from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.profiling.common.device_timer import DeviceTimer
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.utils import (
    ProfileMethod,
    build_profile_method_output_path,
    normalize_profile_method,
    profile_method_to_measurement_type,
    validate_profile_method_platform,
)
from frontier.profiling.utils.singleton import Singleton
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.types import ClusterType, MeasurementType


def _reset_timer_store() -> None:
    Singleton._instances.pop(TimerStatsStore, None)  # noqa: SLF001


class _FakeEvent:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def record(self) -> None:
        self._calls.append("record")

    def elapsed_time(self, other: "_FakeEvent") -> float:
        assert other is not self
        return 2.5


class _FakeProfiler:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def __enter__(self):
        self._calls.append("profiler_enter")
        return self

    def __exit__(self, *_args) -> None:
        self._calls.append("profiler_exit")


def test_device_timer_records_device_events_without_importing_gpu_runtime(monkeypatch) -> None:
    _reset_timer_store()
    calls: list[str] = []

    class _FakeCuda:
        @staticmethod
        def Event(*, enable_timing: bool):
            assert enable_timing is True
            return _FakeEvent(calls)

    fake_torch = SimpleNamespace(cuda=_FakeCuda())
    monkeypatch.setattr(DeviceTimer, "_torch", staticmethod(lambda: fake_torch))

    timer = DeviceTimer("attention")
    with timer:
        calls.append("body")

    assert calls == ["record", "body", "record"]
    assert timer.timer_stats_store.get_times()["attention"] == [2.5]


def test_cuda_timer_reuses_existing_kineto_store(monkeypatch) -> None:
    _reset_timer_store()
    store = TimerStatsStore(profile_method="kineto")
    calls: list[str] = []

    class _FakeProfilerFactory:
        ProfilerActivity = SimpleNamespace(CUDA="cuda")

        @staticmethod
        def profile(*, activities, on_trace_ready):
            assert activities == ["cuda"]
            assert callable(on_trace_ready)
            return _FakeProfiler(calls)

    fake_torch = SimpleNamespace(profiler=_FakeProfilerFactory)
    monkeypatch.setattr(DeviceTimer, "_torch", staticmethod(lambda: fake_torch))

    timer = CudaTimer("collective")
    assert timer.timer_stats_store is store
    with timer:
        calls.append("body")

    assert calls == ["profiler_enter", "body", "profiler_exit"]


def test_device_event_output_path_is_separate_from_cuda_event() -> None:
    cuda_path = build_profile_method_output_path(
        output_root="/tmp/profiling",
        profiling_type="compute",
        hardware="mi355x",
        model_name="fixture",
        op_name="attention",
        profile_method="cuda_event",
    )
    device_path = build_profile_method_output_path(
        output_root="/tmp/profiling",
        profiling_type="compute",
        hardware="mi355x",
        model_name="fixture",
        op_name="attention",
        profile_method="device_event",
    )

    assert cuda_path.name == "attention.csv"
    assert device_path.name == "attention_device_event.csv"


def test_device_event_mapping_is_strict_and_keeps_cuda_aliases_unchanged() -> None:
    assert normalize_profile_method("cuda") == ProfileMethod.CUDA_EVENT.value
    assert normalize_profile_method("device_event") == ProfileMethod.DEVICE_EVENT.value
    assert profile_method_to_measurement_type("device_event").value == "DEVICE_EVENT"

    validate_profile_method_platform("device_event", "rocm")
    validate_profile_method_platform("cuda_event", "cuda")

    try:
        validate_profile_method_platform("cuda_event", "rocm")
    except ValueError as exc:
        assert "ROCm" in str(exc)
    else:  # pragma: no cover - defensive assertion for the strict contract
        raise AssertionError("CUDA events must be rejected on ROCm")


def test_shared_manager_keeps_device_event_models_in_a_separate_registry(tmp_path) -> None:
    manager = ExecutionTimePredictionModelManager(
        {}, SimpleNamespace(cache_dir=str(tmp_path))
    )
    manager._active_measurement_type = MeasurementType.DEVICE_EVENT
    model = object()

    manager._store_model_precision("attn_prefill", "BF16", model)

    assert manager._trained_models_device_event["attn_prefill"] is model
    assert manager._trained_models_eager == {}
    assert manager._models_by_precision_device_event["BF16"]["attn_prefill"] is model


def test_rocm_replica_selects_device_event_without_host_detection() -> None:
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    replica = SimpleNamespace(device="mi355x")

    assert manager._get_measurement_types_for_cluster(
        ClusterType.PREFILL, replica
    ) == [MeasurementType.DEVICE_EVENT]
    assert (
        manager._measurement_family_name(MeasurementType.DEVICE_EVENT)
        == "device_event"
    )

"""Lazy, platform-neutral device-event timing for CUDA and ROCm."""

from __future__ import annotations

import time
from typing import Any

from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.utils import ProfileMethod


class DeviceTimer:
    """Record one operation with a CUDA-compatible device event.

    PyTorch exposes the ``torch.cuda.Event`` API on ROCm. The import stays
    inside construction/entry so CPU-only schema and planner code remains
    importable without a GPU package.
    """

    def __init__(
        self,
        name: str | None,
        layer_id: int = 0,
        aggregation_fn=sum,
        filter_str: str | None = None,
        profile_method: str | None = None,
    ) -> None:
        del layer_id
        normalized_name = str(name).replace("OperationMetrics.", "").lower() if name else None
        self.name = f"vidur_{normalized_name}" if normalized_name else None
        # The existing campaign store owns both method and enabled state.
        # An unnamed standalone timer is inert and must not create global state.
        self.timer_stats_store = TimerStatsStore.get_existing()
        if self.timer_stats_store is None and normalized_name is not None:
            self.timer_stats_store = TimerStatsStore(
                profile_method=profile_method or ProfileMethod.DEVICE_EVENT.value,
            )
        self.disabled = normalized_name is None or self.timer_stats_store.disabled
        self.aggregation_fn = aggregation_fn
        self.filter_str = filter_str
        self.start_event: Any = None
        self.end_event: Any = None
        self.start_time: float | None = None
        self._record_function_context: Any = None
        self.profiler: Any = None
        self._failed = False

        if not self.disabled and self.timer_stats_store.profile_method is ProfileMethod.KINETO:
            torch = self._torch()
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=self.handle_trace,
            )

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Device-event profiling requires PyTorch in the GPU profiling environment"
            ) from exc
        return torch

    def __enter__(self):
        if self.disabled:
            return self
        self._failed = False
        method = self.timer_stats_store.profile_method
        torch = self._torch()
        if method in {ProfileMethod.CUDA_EVENT, ProfileMethod.DEVICE_EVENT}:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        elif method is ProfileMethod.PERF_COUNTER:
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()
        elif method is ProfileMethod.RECORD_FUNCTION:
            from torch.profiler import record_function

            self._record_function_context = record_function(self.name)
            self._record_function_context.__enter__()
        elif method is ProfileMethod.KINETO:
            self.profiler.__enter__()
        else:
            raise ValueError(f"Unsupported DeviceTimer profile method: {method}")
        return self

    def __exit__(self, *args: Any) -> None:
        if self.disabled:
            return
        method = self.timer_stats_store.profile_method
        torch = self._torch()
        if method in {ProfileMethod.CUDA_EVENT, ProfileMethod.DEVICE_EVENT}:
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.end_event.record()
            if args[0] is None:
                self.timer_stats_store.record_time(self.name, [self.start_event, self.end_event])
        elif method is ProfileMethod.PERF_COUNTER:
            torch.cuda.synchronize()
            if args[0] is None:
                self.timer_stats_store.record_time(
                    self.name, (time.perf_counter() - self.start_time) * 1e3
                )
        elif method is ProfileMethod.RECORD_FUNCTION:
            self._record_function_context.__exit__(*args)
        elif method is ProfileMethod.KINETO:
            self._failed = args[0] is not None
            self.profiler.__exit__(*args)
        else:
            raise ValueError(f"Unsupported DeviceTimer profile method: {method}")

    def handle_trace(self, trace: Any) -> None:
        """Record filtered profiler device time for the legacy KINETO mode."""

        if self._failed:
            return
        events = trace.events()
        if self.filter_str:
            events = [event for event in events if event.name.startswith(self.filter_str)]
        if not events:
            return
        if hasattr(events[0], "device_time_total"):
            total_device_time = self.aggregation_fn(
                [event.device_time_total for event in events]
            )
        else:
            total_device_time = self.aggregation_fn(
                [event.cuda_time_total for event in events]
            )
        self.timer_stats_store.record_time(self.name, total_device_time * 1e-3)

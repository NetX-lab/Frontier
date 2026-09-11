"""HIP event-record graph nodes, without Kineto or PyTorch external events.

The public HIP graph API lets an event node become the current capture stream's
dependency. Existing dependencies are preserved, so the next captured kernel
runs after the event. Handles must outlive every executable graph using them.
This module imports neither torch nor the HIP library until explicitly used.
"""

from __future__ import annotations

import ctypes as C
from ctypes.util import find_library
import math


class HipGraphEvents:
    def __init__(self, library_path: str | None = None):
        self.lib = C.CDLL(library_path or find_library("amdhip64") or "libamdhip64.so")
        pointer = C.c_void_p
        pointer_array = C.POINTER(pointer)
        signatures = {
            "hipStreamGetCaptureInfo_v2": [pointer, C.POINTER(C.c_int), C.POINTER(C.c_ulonglong),
                C.POINTER(pointer), C.POINTER(pointer_array), C.POINTER(C.c_size_t)],
            "hipStreamUpdateCaptureDependencies": [pointer, pointer_array, C.c_size_t, C.c_uint],
            "hipGraphAddEventRecordNode": [C.POINTER(pointer), pointer, pointer_array, C.c_size_t, pointer],
            "hipEventCreateWithFlags": [C.POINTER(pointer), C.c_uint],
            "hipEventElapsedTime": [C.POINTER(C.c_float), pointer, pointer],
            "hipEventDestroy": [pointer],
        }
        for name, signature in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = signature, C.c_int
        self.events = []

    def _call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with HIP status {status}")

    def capture_info(self, stream):
        status, capture_id = C.c_int(), C.c_ulonglong()
        graph, dependencies, count = C.c_void_p(), C.POINTER(C.c_void_p)(), C.c_size_t()
        self._call("hipStreamGetCaptureInfo_v2", C.c_void_p(stream.cuda_stream),
                   C.byref(status), C.byref(capture_id), C.byref(graph),
                   C.byref(dependencies), C.byref(count))
        if status.value == 0:
            return None
        if status.value != 1 or not graph.value:
            raise ValueError("HIP stream capture is invalidated or has no graph")
        return capture_id.value, graph, dependencies, count.value

    def create_event(self):
        handle = C.c_void_p()
        self._call("hipEventCreateWithFlags", C.byref(handle), 0)
        self.events.append(handle)
        return _GraphEvent(self, handle)

    def record(self, event, stream):
        info = self.capture_info(stream)
        if info is None:
            raise ValueError("Graph timing nodes require an active capture")
        capture_id, graph, dependencies, count = info
        node = C.c_void_p()
        self._call("hipGraphAddEventRecordNode", C.byref(node), graph, dependencies, count, event)
        # hipStreamSetCaptureDependencies == 1. The new node already depends on
        # ALL prior stream sinks; replacing the sink set does not discard work.
        self._call("hipStreamUpdateCaptureDependencies", C.c_void_p(stream.cuda_stream),
                   C.byref(node), 1, 1)
        return capture_id

    def elapsed_ms(self, start, end):
        result = C.c_float()
        self._call("hipEventElapsedTime", C.byref(result), start, end)
        value = float(result.value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid HIP graph event duration")
        return value

    def close_after_graphs_destroyed(self):
        """Caller must synchronize and destroy all graphs referencing the events."""
        while self.events:
            self._call("hipEventDestroy", self.events[-1])
            self.events.pop()


class _GraphEvent:
    def __init__(self, api, handle):
        self.api, self.handle = api, handle
        self.capture_id = None

    def record(self, stream):
        self.capture_id = self.api.record(self.handle, stream)

    def elapsed_time(self, other):
        if self.api is not other.api or self.capture_id is None or self.capture_id != other.capture_id:
            raise ValueError("Timing endpoints must belong to the same HIP capture")
        return self.api.elapsed_ms(self.handle, other.handle)

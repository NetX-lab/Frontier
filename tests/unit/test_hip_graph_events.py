"""Exercise public HIP ABI/dependency handling without importing torch or HIP."""

import ctypes as C
from types import SimpleNamespace

import pytest

from frontier.validation.hip_graph_events import HipGraphEvents


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def put(pointer, kind, value):
    C.cast(pointer, C.POINTER(kind))[0] = value


class Library:
    def __init__(self):
        self.capture_status, self.capture_id, self.graph = 1, 42, 7
        self.dependencies = (C.c_void_p * 2)(10, 11)
        self.event_id, self.duration = 100, 0.125
        self.calls, self.destroyed = [], []
        self.failure = None
        for name in ("hipStreamGetCaptureInfo_v2", "hipStreamUpdateCaptureDependencies",
                     "hipGraphAddEventRecordNode", "hipEventCreateWithFlags",
                     "hipEventElapsedTime", "hipEventDestroy"):
            setattr(self, name, Function(lambda *args, name=name: self.call(name, *args)))

    def call(self, name, *args):
        if name == self.failure:
            return 99
        if name == "hipStreamGetCaptureInfo_v2":
            stream, status, capture_id, graph, dependencies, count = args
            assert stream.value == 3
            put(status, C.c_int, self.capture_status)
            put(capture_id, C.c_ulonglong, self.capture_id)
            put(graph, C.c_void_p, self.graph)
            put(dependencies, C.POINTER(C.c_void_p), C.cast(self.dependencies, C.POINTER(C.c_void_p)))
            put(count, C.c_size_t, len(self.dependencies))
        elif name == "hipEventCreateWithFlags":
            handle, flags = args
            assert flags == 0  # timing must remain enabled
            put(handle, C.c_void_p, self.event_id)
            self.event_id += 1
        elif name == "hipGraphAddEventRecordNode":
            node, graph, dependencies, count, event = args
            self.calls.append(("add", graph.value, [dependencies[i] for i in range(count)], event.value))
            put(node, C.c_void_p, 200)
        elif name == "hipStreamUpdateCaptureDependencies":
            stream, dependencies, count, flags = args
            self.calls.append(("update", stream.value,
                               [C.cast(dependencies, C.POINTER(C.c_void_p))[i] for i in range(count)], flags))
        elif name == "hipEventElapsedTime":
            put(args[0], C.c_float, self.duration)
        elif name == "hipEventDestroy":
            self.destroyed.append(args[0].value)
        return 0


@pytest.fixture
def api(monkeypatch):
    library = Library()
    monkeypatch.setattr(C, "CDLL", lambda path: library)
    return HipGraphEvents("test-library"), library, SimpleNamespace(cuda_stream=3)


def test_explicit_nodes_preserve_all_dependencies_and_set_the_new_stream_sink(api):
    events, library, stream = api
    start, end = events.create_event(), events.create_event()
    start.record(stream)
    end.record(stream)
    assert library.calls[:2] == [("add", 7, [10, 11], 100), ("update", 3, [200], 1)]
    assert start.elapsed_time(end) == pytest.approx(0.125)
    assert len(events.events) == 2
    events.close_after_graphs_destroyed()
    events.close_after_graphs_destroyed()
    assert library.destroyed == [101, 100]


def test_capture_status_and_hip_errors_do_not_fall_back_to_other_timing_methods(api):
    events, library, stream = api
    event = events.create_event()
    library.capture_status = 0
    assert events.capture_info(stream) is None
    with pytest.raises(ValueError, match="active capture"):
        event.record(stream)
    library.capture_status = 2
    with pytest.raises(ValueError, match="invalidated"):
        events.capture_info(stream)
    library.capture_status, library.graph = 1, 0
    with pytest.raises(ValueError, match="no graph"):
        events.capture_info(stream)
    library.graph, library.failure = 7, "hipGraphAddEventRecordNode"
    with pytest.raises(RuntimeError, match="HIP status 99"):
        event.record(stream)
    assert library.calls == []  # never update dependencies after a failed insertion


@pytest.mark.parametrize("duration", [-1.0, float("inf"), float("nan")])
def test_invalid_durations_fail(api, duration):
    events, library, stream = api
    start, end = events.create_event(), events.create_event()
    start.record(stream)
    end.record(stream)
    library.duration = duration
    with pytest.raises(ValueError, match="Invalid"):
        start.elapsed_time(end)


def test_event_endpoints_must_share_a_capture_and_owner(api):
    events, library, stream = api
    start, end = events.create_event(), events.create_event()
    with pytest.raises(ValueError, match="same HIP capture"):
        start.elapsed_time(end)
    start.record(stream)
    library.capture_id += 1
    end.record(stream)
    with pytest.raises(ValueError, match="same HIP capture"):
        start.elapsed_time(end)
    start.capture_id = end.capture_id
    end.api = object()
    with pytest.raises(ValueError, match="same HIP capture"):
        start.elapsed_time(end)

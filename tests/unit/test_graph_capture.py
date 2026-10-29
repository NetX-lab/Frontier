"""CPU capture tests with stream graphs and native CUPTI payload structures."""

from __future__ import annotations

import ctypes
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from frontier.profiling.attention import graph_capture as capture
from frontier.profiling.attention.graph_capture import (
    CUPTI_CB_DOMAIN_RESOURCE,
    CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED,
    CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED,
    CaptureRecord,
    CuptiCloneRecorder,
    CuptiGraphData,
    CuptiResourceData,
    EagerBreak,
    GraphCaptureRecorder,
    KernelLaunch,
    parse_clone_payload,
)

LAYER = (4, "layer")
INNER = (4, "attention")
SIBLING = (4, "projection")


class FakeQueries:
    """Launches depend on the last node on their stream, within one graph."""

    def __init__(self):
        self.stream = 7
        self.active = None
        self.graphs = {}
        self.nodes = {}
        self.last = {}
        self.sweeps = []
        self.eager_launches = []
        self.ended = []
        self.before_end = lambda: None

    def current_stream(self):
        return self.stream

    def capture_graph(self, stream):
        assert stream == 7
        if self.active is None:
            raise ValueError("Stream is not capturing")
        self.sweeps.append(self.active)
        return self.active

    def graph_nodes(self, graph):
        # Reverse order exercises dependency-id assignment before node recording.
        return list(reversed(self.graphs[graph]))

    def node_type(self, node):
        return self.nodes[node].kind

    def kernel_launch(self, node):
        return self.nodes[node].launch

    def dependencies(self, node):
        return self.nodes[node].dependencies

    def node_id(self, node):
        return node + 10000

    def demangle(self, name):
        return name.removeprefix("_Z") + "()" if name.startswith("_Z") else None

    def launch(self, name, kind="kernel"):
        if self.active is None:
            self.eager_launches.append(name)
            return name
        node = len(self.nodes) + 1
        previous = self.last.get((self.active, self.stream))
        self.nodes[node] = SimpleNamespace(
            kind=kind, launch=KernelLaunch(name, (2, 1, 1), (32, 1, 1)),
            dependencies=[] if previous is None else [previous],
        )
        self.graphs[self.active].append(node)
        self.last[self.active, self.stream] = node
        return name


@pytest.fixture
def fake_classes():
    queries = FakeQueries()

    class Graph:
        def capture_begin(self, *args, **kwargs):
            assert queries.active is None
            self.graph = len(queries.graphs) + 1
            queries.graphs[self.graph] = []
            queries.active = self.graph
            self.arguments = args, kwargs

        def capture_end(self):
            queries.before_end()
            assert queries.active == self.graph
            queries.ended.append(self.graph)
            queries.active = None  # Source graph is unavailable to a later sweep.

        def replay(self):
            return self.graph

    class Breakable:
        _tls = threading.local()

        def __init__(self, pool=None):
            self.pool = pool
            self.segments = []
            self._capturing = False
            self._current_graph = None

        @classmethod
        def current(cls):
            return getattr(cls._tls, "active", None)

        def __enter__(self):
            assert self.current() is None
            self._tls.active = self
            self._begin_segment()
            return self

        def __exit__(self, exc_type, exc, traceback):
            try:
                self._end_segment()
            finally:
                self._tls.active = None

        def _begin_segment(self):
            graph = Graph()
            graph.capture_begin(pool=self.pool)
            self._current_graph = graph
            self._capturing = True

        def _end_segment(self):
            if not self._capturing:
                return
            self._current_graph.capture_end()
            self.segments.append(self._current_graph.replay)
            self._current_graph = None
            self._capturing = False

        def add_eager(self, fn):
            self._end_segment()
            result = fn()
            self.segments.append(fn)
            self._begin_segment()
            return result

        def replay(self):
            for segment in self.segments:
                segment()

    return queries, Graph, Breakable


def _recorder(queries, calls=None):
    def eager_scope_runner(owner, fn):
        if calls is not None:
            calls.append(owner)
        return fn()
    return GraphCaptureRecorder(queries, eager_scope_runner)


def test_full_nested_scopes_sweep_before_original_end(fake_classes):
    queries, Graph, Breakable = fake_classes
    recorder = _recorder(queries)
    with recorder.install(Graph, Breakable):
        graph = Graph()
        graph.capture_begin("mode", pool=42)
        queries.launch("outside_before")
        with recorder.scope(LAYER):
            with recorder.scope(INNER):
                queries.launch("_Zattention")
                queries.launch("second")
            with recorder.scope(SIBLING):
                queries.launch("projection")
        queries.launch("outside_after", kind="memset")
        def before_end():
            assert recorder._open is None
            assert len(recorder._nodes) == 5
        queries.before_end = before_end
        graph.capture_end()
    record = recorder.record([])
    assert graph.arguments == (("mode",), {"pool": 42})
    assert [(row.ordinal, row.kind, row.breakable_capture, row.origin_stream)
            for row in record.captures] == [(0, "graph", None, 7)]
    nodes = {row.node_id: row for row in record.nodes}
    assert [nodes[index].owner for index in sorted(nodes)] == [None, INNER, INNER, SIBLING, None]
    assert nodes[10002].kernel_name_demangled == "attention()"
    assert nodes[10003].dependencies == (10002,)
    assert nodes[10005].node_type == "memset"
    assert nodes[10005].kernel_name is None
    assert [(row.owner, row.parent) for row in record.scope_entries] == [(LAYER, None), (INNER, LAYER), (SIBLING, LAYER)]


def test_warmup_has_no_sweeps_or_entries(fake_classes):
    queries, Graph, Breakable = fake_classes
    recorder = _recorder(queries)
    with recorder.install(Graph, Breakable), recorder.scope(LAYER):
        with recorder.scope(INNER):
            queries.launch("warmup")
    assert queries.sweeps == []
    assert recorder.record([]) == CaptureRecord()


def test_breakable_segments_keep_registration_owner_only_at_replay(fake_classes):
    queries, Graph, Breakable = fake_classes
    calls = []
    recorder = _recorder(queries, calls)
    fn = lambda: queries.launch("eager")
    with recorder.install(Graph, Breakable):
        with Breakable(pool=42) as breakable:
            with recorder.scope(LAYER):
                with recorder.scope(INNER):
                    queries.launch("segment0")
                assert breakable.add_eager(fn) == "eager"
                with recorder.scope(SIBLING):
                    queries.launch("segment1")
        assert calls == []
        assert queries.eager_launches == ["eager"]
        assert breakable.segments[1] is not fn
    # Wrappers remain usable after class patches are removed.
    breakable.replay()
    assert calls == [LAYER]
    assert queries.eager_launches == ["eager", "eager"]
    record = recorder.record([(11001, 10001), (11002, 10002)])
    assert [(row.kind, row.breakable_capture) for row in record.captures] == [("breakable_segment", 0)] * 2
    assert [(row.capture, row.owner) for row in record.nodes] == [(0, INNER), (1, SIBLING)]
    assert record.eager_breaks == (EagerBreak(0, 1, LAYER),)
    assert record.clones[0].original_node_id == 10001


@pytest.mark.parametrize("fail", [False, True])
def test_install_restores_all_patches(fake_classes, fail):
    queries, Graph, Breakable = fake_classes
    originals = Graph.capture_begin, Graph.capture_end, Breakable.add_eager
    recorder = _recorder(queries)
    try:
        with recorder.install(Graph, Breakable):
            assert (Graph.capture_begin, Graph.capture_end, Breakable.add_eager) != originals
            if fail:
                raise RuntimeError("body failure")
    except RuntimeError as error:
        assert str(error) == "body failure"
    assert (Graph.capture_begin, Graph.capture_end, Breakable.add_eager) == originals


def test_scope_stack_pops_when_sweep_fails(fake_classes):
    queries, Graph, Breakable = fake_classes
    recorder = _recorder(queries)
    with pytest.raises(ValueError, match="Unsupported captured node type"):
        with recorder.install(Graph, Breakable):
            Graph().capture_begin()
            with recorder.scope(INNER):
                queries.launch("unsupported", "host")
    assert recorder._stack == []


def test_nested_capture_is_rejected(fake_classes):
    queries, Graph, Breakable = fake_classes
    with _recorder(queries).install(Graph, Breakable):
        graph = Graph()
        graph.capture_begin()
        with pytest.raises(ValueError, match="another capture is open"):
            Graph().capture_begin()
        graph.capture_end()


def test_json_round_trip_and_duplicate_ids(fake_classes):
    queries, Graph, Breakable = fake_classes
    recorder = _recorder(queries)
    with recorder.install(Graph, Breakable):
        graph = Graph()
        graph.capture_begin()
        with recorder.scope(INNER):
            queries.launch("kernel")
        graph.capture_end()
    record = recorder.record([(11001, 10001)])
    raw = record.to_json()
    assert raw["nodes"][0]["owner"] == [4, "attention"]
    assert raw["nodes"][0]["grid"] == [2, 1, 1]
    assert CaptureRecord.from_json(json.loads(json.dumps(raw))) == record
    with pytest.raises(ValueError, match="Duplicate node id 10001"):
        replace(record, nodes=record.nodes * 2)
    raw["nodes"].append(raw["nodes"][0])
    with pytest.raises(ValueError, match="Duplicate node id 10001"):
        CaptureRecord.from_json(raw)
    with pytest.raises(ValueError, match="Duplicate node id 11001"):
        recorder.record([(11001, 10001), (11001, 10001)])


def test_duplicate_query_ids_fail_during_sweep(fake_classes):
    queries, Graph, Breakable = fake_classes
    queries.node_id = lambda node: 10
    with pytest.raises(ValueError, match="Duplicate node id 10"):
        with _recorder(queries).install(Graph, Breakable):
            graph = Graph()
            graph.capture_begin()
            queries.launch("one")
            queries.launch("two")
            graph.capture_end()


class FakeFunction:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class FakeCupti:
    def __init__(self, subscribe_code=0, id_code=0, enable_code=0, unsubscribe_code=0):
        self.calls = []
        self.callback = None
        self.subscribe_code, self.id_code = subscribe_code, id_code
        self.enable_code, self.unsubscribe_code = enable_code, unsubscribe_code
        self.cuptiSubscribe = FakeFunction(self.subscribe)
        self.cuptiEnableCallback = FakeFunction(self.enable)
        self.cuptiUnsubscribe = FakeFunction(self.unsubscribe)
        self.cuptiGetGraphNodeId = FakeFunction(self.node_id)

    def subscribe(self, subscriber, callback, userdata):
        self.calls.append(("subscribe",))
        if not self.subscribe_code:
            ctypes.cast(subscriber, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(123)
            self.callback = callback
        return self.subscribe_code

    def enable(self, enabled, subscriber, domain, callback_id):
        self.calls.append(("enable", enabled, subscriber.value, domain, callback_id))
        return self.enable_code if enabled else 0

    def unsubscribe(self, subscriber):
        self.calls.append(("unsubscribe", subscriber.value))
        return self.unsubscribe_code

    def node_id(self, node, result):
        ctypes.cast(result, ctypes.POINTER(ctypes.c_uint64))[0] = node.value + 1000
        return self.id_code


def _payload(node=22, original=11):
    graph = CuptiGraphData(node=node, originalNode=original)
    resource = CuptiResourceData(context=999, resourceHandle=888, resourceDescriptor=ctypes.addressof(graph))
    return graph, resource


def test_payload_reader_uses_resource_descriptor_and_native_layout():
    graph, resource = _payload()
    assert ctypes.sizeof(CuptiResourceData) == 24
    assert ctypes.sizeof(CuptiGraphData) == 56
    assert [getattr(CuptiGraphData, field).offset for field, _ in CuptiGraphData._fields_] == [0, 8, 16, 24, 32, 40, 48]
    assert parse_clone_payload(ctypes.byref(resource), lambda node: node + 1000) == (1022, 1011)
    assert graph.node == 22


def test_clone_recorder_reads_payload_and_ignores_other_callbacks():
    library = FakeCupti()
    graph, resource = _payload()
    with CuptiCloneRecorder(library) as recorder:
        library.callback(None, 0, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED, None)
        library.callback(None, CUPTI_CB_DOMAIN_RESOURCE, 0, None)
        library.callback(None, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED, ctypes.byref(resource))
        assert recorder.clones == [(1022, 1011)]
    assert library.calls == [
        ("subscribe",), ("enable", 1, 123, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED),
        ("enable", 0, 123, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED), ("unsubscribe", 123),
    ]


@pytest.mark.parametrize("fault", ["descriptor", "node", "original", "payload", "id_query"])
def test_callback_errors_are_stored_until_exit(fault):
    library = FakeCupti(id_code=7 if fault == "id_query" else 0)
    graph, resource = _payload(node=0 if fault == "node" else 22, original=0 if fault == "original" else 11)
    if fault == "descriptor":
        resource.resourceDescriptor = None
    message = "cuptiGetGraphNodeId.*7" if fault == "id_query" else "null"
    with pytest.raises((ValueError, RuntimeError), match=message):
        with CuptiCloneRecorder(library) as recorder:
            library.callback(None, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED,
                             None if fault == "payload" else ctypes.byref(resource))
            assert recorder.clones == []
            assert recorder._errors
    assert library.calls[-1] == ("unsubscribe", 123)


def test_multiple_subscriber_error_never_unsubscribes_another_slot():
    library = FakeCupti(subscribe_code=CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED)
    with pytest.raises(RuntimeError, match="another CUPTI subscriber.*torch.profiler/Kineto.*before any profiler session"):
        with CuptiCloneRecorder(library):
            pytest.fail("Must not enter on subscription failure")
    assert library.calls == [("subscribe",)]


def test_callback_enable_failure_releases_owned_subscriber():
    library = FakeCupti(enable_code=7)
    with pytest.raises(RuntimeError, match="cuptiEnableCallback.*7"):
        with CuptiCloneRecorder(library):
            pytest.fail("Must not enter on enable failure")
    assert library.calls[-1] == ("unsubscribe", 123)


def test_unsubscribe_failure_keeps_callback_alive_until_retry():
    library = FakeCupti(unsubscribe_code=7)
    recorder = CuptiCloneRecorder(library)
    with pytest.raises(RuntimeError, match="cuptiUnsubscribe.*7"):
        with recorder:
            pass
    assert recorder in capture._ACTIVE_SUBSCRIPTIONS
    assert recorder._owned
    library.unsubscribe_code = 0
    recorder.__exit__(None, None, None)
    assert recorder not in capture._ACTIVE_SUBSCRIPTIONS


@pytest.mark.parametrize("paths", [[], ["/one/libcupti.so.13", "/two/libcupti.so.13"]])
def test_library_resolution_requires_one_mapped_path(monkeypatch, paths):
    maps = "\n".join(f"100-200 r--p 0000 00:00 1 {path}" for path in paths)
    monkeypatch.setattr(capture.Path, "read_text", lambda self: maps)
    with pytest.raises(RuntimeError, match="exactly one mapped libcupti.so"):
        capture.resolve_cupti_library()


def test_library_resolution_deduplicates_mapped_segments(monkeypatch):
    maps = "\n".join(["100-200 r--p 0000 00:00 1 /one/libcupti.so.13"] * 2)
    monkeypatch.setattr(capture.Path, "read_text", lambda self: maps)
    monkeypatch.setattr(capture.ctypes, "CDLL", lambda path: path)
    assert capture.resolve_cupti_library() == "/one/libcupti.so.13"

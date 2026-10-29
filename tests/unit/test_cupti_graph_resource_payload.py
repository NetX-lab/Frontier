"""Allocated CPU payload controls; no native CUPTI calls or runtime proof."""

import ctypes
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(os.environ.get("FRONTIER_GRAPH_PROBE_SOURCE", str(
    Path(__file__).parents[2] / "task_memory/task_2026-10-23_v4_remaining_measurements"
    / "profile01/graph_probe/probe_diagnostic.py")))


class ResourceHandle(ctypes.Union):
    _fields_ = [("stream", ctypes.c_void_p)]


class ResourceData(ctypes.Structure):
    _fields_ = [("context", ctypes.c_void_p), ("resourceHandle", ResourceHandle),
                ("resourceDescriptor", ctypes.c_void_p)]


@pytest.fixture
def probe():
    return SimpleNamespace(**runpy.run_path(str(SOURCE)))


@pytest.fixture
def capture(probe):
    state = {"calls": [], "queries": [], "results": {}, "in_callback": False}

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            state["calls"].append((self.name, args[-1] if self.name == "cuptiEnableCallback" else None))
            result = state["results"].get(self.name, 0)
            if self.name == "cuptiSubscribe" and not result:
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 1234
            return result

    def query(kind, handle):
        assert state["in_callback"]
        state["queries"].append((kind, handle))
        if state.get("query_error"):
            raise KeyboardInterrupt("injected query failure")
        return {11: 2**60 + 17, 12: 2**59 + 23, 701: 101, 702: 202}.get(handle, 99)

    library = SimpleNamespace(**{name: Function(name) for name in
        ("cuptiSubscribe", "cuptiEnableCallback", "cuptiUnsubscribe")})
    ids = SimpleNamespace(library=library, node_id=lambda h: query("node", h),
                          graph_id=lambda h: query("graph", h), exec_id=lambda h: query("exec", h))
    value = probe.CuptiGraphCallbacks(ids)
    yield value, state
    state["results"]["cuptiUnsubscribe"] = 0
    assert value.close()
    assert value not in probe._OWNED_CALLBACKS


def envelope(probe, **fields):
    native = probe.CuptiGraphData(**fields)
    storage = ctypes.create_string_buffer(ctypes.sizeof(probe.CuptiGraphData))
    resource = ResourceData.from_buffer(storage)
    resource.context, resource.resourceHandle.stream = 901, 902
    resource.resourceDescriptor = ctypes.addressof(native)
    for offset in (24, 48):
        ctypes.c_void_p.from_buffer(storage, offset).value = 903 + offset
    return native, storage, resource


def invoke(probe, capture, callback_id, pointer, domain=None):
    value, state = capture
    state["in_callback"] = True
    try:
        value.callback(None, probe.RESOURCE if domain is None else domain, callback_id, pointer)
    finally:
        state["in_callback"] = False


@pytest.mark.parametrize("kind", ["clone", "exec"])
def test_resource_envelope_exact_handles_ids_and_copy(probe, capture, kind):
    value, state = capture
    value.start()
    native, storage, resource = envelope(probe, node=11, originalNode=12, graph=701, graphExec=702)
    callback_id = probe.NODE_CLONED if kind == "clone" else probe.EXEC_CREATED
    invoke(probe, capture, callback_id, ctypes.byref(resource))
    expected = ({"callback_id": 20, "node": 11, "original_node": 12,
                 "node_id": 2**60 + 17, "original_node_id": 2**59 + 23} if kind == "clone" else
                {"callback_id": 18, "graph": 701, "graph_exec": 702, "graph_id": 101, "exec_graph_id": 202})
    assert state["queries"] == ([("node", 11), ("node", 12)] if kind == "clone" else
                                [("graph", 701), ("exec", 702)])
    native.node = native.originalNode = native.graph = native.graphExec = 999
    resource.resourceDescriptor = None
    assert value.receipt["events"] == [expected] and not value.receipt["errors"]
    assert value.close() and value.receipt["unsubscribed"]


@pytest.mark.parametrize("fault", ["null_pointer", "null_descriptor", "node", "originalNode",
                                   "graph", "graphExec", "domain", "callback_id"])
def test_malformed_resource_has_no_queries(probe, capture, fault):
    native, storage, resource = envelope(probe, node=11, originalNode=12, graph=701, graphExec=702)
    callback_id = probe.EXEC_CREATED if fault in ("graph", "graphExec") else probe.NODE_CLONED
    if fault == "null_descriptor":
        resource.resourceDescriptor = None
    elif fault in ("node", "originalNode", "graph", "graphExec"):
        setattr(native, fault, None)
    invoke(probe, capture, 999 if fault == "callback_id" else callback_id,
           None if fault == "null_pointer" else ctypes.byref(resource), 999 if fault == "domain" else None)
    value, state = capture
    assert not state["queries"] and len(value.receipt["errors"]) == 1
    if fault == "null_descriptor":
        assert value.receipt["errors"][0]["message"] == "Graph resource callback has no descriptor"


def test_query_exception_isolated_and_later_callback_works(probe, capture):
    native, storage, resource = envelope(probe, node=11, originalNode=12)
    value, state = capture
    state["query_error"] = True
    invoke(probe, capture, probe.NODE_CLONED, ctypes.byref(resource))
    assert value.receipt["errors"][0]["error_type"] == "KeyboardInterrupt"
    state["query_error"] = False
    invoke(probe, capture, probe.NODE_CLONED, ctypes.byref(resource))
    assert value.receipt["events"][-1]["original_node_id"] == 2**59 + 23


@pytest.mark.parametrize("failure", [None, "cuptiSubscribe", "cuptiEnableCallback", "cuptiUnsubscribe"])
def test_subscription_lifecycle_and_selected_callbacks(probe, capture, failure):
    value, state = capture
    if failure:
        state["results"][failure] = 7
    value.start()
    assert value.owned == (failure != "cuptiSubscribe")
    assert value.close() == (failure != "cuptiUnsubscribe")
    if failure == "cuptiSubscribe":
        assert state["calls"] == [("cuptiSubscribe", None)]
    elif failure == "cuptiEnableCallback":
        assert state["calls"] == [("cuptiSubscribe", None), ("cuptiEnableCallback", 20), ("cuptiUnsubscribe", None)]
    else:
        assert state["calls"][1:3] == [("cuptiEnableCallback", 20), ("cuptiEnableCallback", 18)]
    assert not state["queries"]


def test_resource_and_graph_abi(probe):
    abi = probe.callback_abi()
    assert abi["resource_data"] == {"size": 24, "alignment": 8,
        "offsets": {"context": 0, "resourceHandle": 8, "resourceDescriptor": 16}}
    assert abi["resource_handle"] == {"size": 8, "alignment": 8, "offsets": {"stream": 0}}
    assert (abi["size"], abi["alignment"]) == (56, 8)
    assert issubclass(probe.CuptiResourceHandle, ctypes.Union)
    assert dict(probe.CuptiResourceData._fields_)["resourceHandle"] is probe.CuptiResourceHandle

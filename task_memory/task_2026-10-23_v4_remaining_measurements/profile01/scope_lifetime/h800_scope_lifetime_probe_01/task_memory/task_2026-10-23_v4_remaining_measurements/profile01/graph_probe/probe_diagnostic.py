"""Probe retained CUDA graph identities using CUDA 13.0 topology and CUPTI APIs.

This is a diagnostic, not a benchmark or a kernel-only profiling producer.
Only ``run`` initializes CUDA. ``inspect-api`` is safe in a CPU-only container.
"""

from __future__ import annotations

# DIAGNOSTIC_PROLOGUE_BEGIN
import faulthandler as _diagnostic_faults
if __name__ == "__main__":
    _diagnostic_faults.enable(file=2, all_threads=True)
import json as _diagnostic_json
import os as _diagnostic_os
import time as _diagnostic_time

_DIAGNOSTIC_STAGE_FD = None
_DIAGNOSTIC_FAULT_FD = None
_DIAGNOSTIC_SEQUENCE = 0


def _diagnostic_stage(event, result=None):
    global _DIAGNOSTIC_SEQUENCE
    if _DIAGNOSTIC_STAGE_FD is None:
        return  # Imported CPU tests do not initialize the CLI diagnostic sink.
    _DIAGNOSTIC_SEQUENCE += 1
    row = {"sequence": _DIAGNOSTIC_SEQUENCE, "event": event, "result": result,
           "pid": _diagnostic_os.getpid(), "monotonic_ns": _diagnostic_time.monotonic_ns()}
    payload = (_diagnostic_json.dumps(row) + "\n").encode()
    if _diagnostic_os.write(_DIAGNOSTIC_STAGE_FD, payload) != len(payload):
        raise OSError("Incomplete diagnostic stage write")
    _diagnostic_os.fsync(_DIAGNOSTIC_STAGE_FD)


def _diagnostic_start():
    global _DIAGNOSTIC_STAGE_FD, _DIAGNOSTIC_FAULT_FD
    directory = _diagnostic_os.environ["FRONTIER_GRAPH_DIAGNOSTICS_DIR"]
    flags = _diagnostic_os.O_WRONLY | _diagnostic_os.O_CREAT | _diagnostic_os.O_EXCL
    _DIAGNOSTIC_STAGE_FD = _diagnostic_os.open(_diagnostic_os.path.join(directory, "stages.jsonl"), flags, 0o600)
    _DIAGNOSTIC_FAULT_FD = _diagnostic_os.open(_diagnostic_os.path.join(directory, "fault.log"), flags, 0o600)
    _diagnostic_faults.enable(file=_DIAGNOSTIC_FAULT_FD, all_threads=True)
    _diagnostic_stage("copy.enter")


def _diagnostic_api_inspection(torch, runtime):
    _diagnostic_stage("api_inspection.enter")
    value = api_inspection(torch, runtime)
    _diagnostic_stage("api_inspection.exit")
    return value


if __name__ == "__main__":
    _diagnostic_start()
# DIAGNOSTIC_PROLOGUE_END

import argparse
import ctypes
import hashlib
import importlib.metadata
import inspect
import json
import traceback
from pathlib import Path

TARGET = "frontier.graph_scope_probe.target_replay"
SCOPE_NAMES = ("scope_a", "scope_b")
BOUNDARY_NAMES = ("start", "end")
KERNELS_PER_SCOPE, UNSCOPED_KERNELS = 2, 1
_OWNED_CALLBACKS = []  # Native subscriptions must outlive every registered callback pointer.


class CuptiResourceHandle(ctypes.Union):
    """CUpti_ResourceData resourceHandle union from CUPTI13.0.85."""

    _fields_ = [("stream", ctypes.c_void_p)]


class CuptiResourceData(ctypes.Structure):
    """Resource callback envelope; graph payload is resourceDescriptor."""

    _fields_ = [("context", ctypes.c_void_p), ("resourceHandle", CuptiResourceHandle),
                ("resourceDescriptor", ctypes.c_void_p)]


class CuptiGraphData(ctypes.Structure):
    """CUpti_GraphData from the locked CUDA13.0 cupti_callbacks.h."""

    _fields_ = [(name, ctypes.c_void_p) for name in
                ("graph", "originalGraph", "node", "originalNode")] + [
        ("nodeType", ctypes.c_int), ("dependency", ctypes.c_void_p), ("graphExec", ctypes.c_void_p)]


RESOURCE, NODE_CLONED, EXEC_CREATED = 3, 20, 18
CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p)


def callback_abi():
    return {"size": ctypes.sizeof(CuptiGraphData), "alignment": ctypes.alignment(CuptiGraphData),
            "offsets": {name: getattr(CuptiGraphData, name).offset for name, _ in CuptiGraphData._fields_},
            "resource_domain": RESOURCE, "node_cloned": NODE_CLONED, "exec_created": EXEC_CREATED,
            "resource_data": {"size": ctypes.sizeof(CuptiResourceData),
                "alignment": ctypes.alignment(CuptiResourceData), "offsets": {
                    name: getattr(CuptiResourceData, name).offset for name, _ in CuptiResourceData._fields_}},
            "resource_handle": {"size": ctypes.sizeof(CuptiResourceHandle),
                "alignment": ctypes.alignment(CuptiResourceHandle), "offsets": {
                    name: getattr(CuptiResourceHandle, name).offset for name, _ in CuptiResourceHandle._fields_}}}


def checked(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA binding returned {result[0]}")
    return result[1:]


def cupti_library():
    distribution = importlib.metadata.distribution("nvidia-cuda-cupti")
    libraries = [distribution.locate_file(p) for p in distribution.files if p.name == "libcupti.so.13"]
    if len(libraries) != 1:
        raise RuntimeError("Expected exactly one installed CUDA 13 CUPTI library")
    return libraries[0]


class CuptiGraphIds:
    """Only the three graph identity queries declared by CUDA 13.0 cupti_activity.h."""

    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        for name, value_type in (("cuptiGetGraphNodeId", ctypes.c_uint64),
                                 ("cuptiGetGraphId", ctypes.c_uint32),
                                 ("cuptiGetGraphExecId", ctypes.c_uint32)):
            function = getattr(self.library, name)
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(value_type)]
            function.restype = ctypes.c_int

    @staticmethod
    def _read(function, handle, value_type):
        value = value_type()
        result = function(ctypes.c_void_p(int(handle)), ctypes.byref(value))
        if result != 0:
            raise RuntimeError(f"{function.__name__} returned CUptiResult={result}")
        return int(value.value)

    def node_id(self, handle):
        return self._read(self.library.cuptiGetGraphNodeId, handle, ctypes.c_uint64)

    def graph_id(self, handle):
        return self._read(self.library.cuptiGetGraphId, handle, ctypes.c_uint32)

    def exec_id(self, handle):
        return self._read(self.library.cuptiGetGraphExecId, handle, ctypes.c_uint32)


class CuptiGraphCallbacks:
    """One isolated resource subscription; no activity tracing or subscriber replacement."""

    def __init__(self, ids):
        self.ids, self.handle, self.owned = ids, ctypes.c_void_p(), False
        self.receipt = {"subscribed": False, "unsubscribed": False, "subscription_active": False,
                     "events": [], "errors": [], "results": []}
        self.callback = CALLBACK(self._callback)
        signatures = {"cuptiSubscribe": [ctypes.POINTER(ctypes.c_void_p), CALLBACK, ctypes.c_void_p],
                      "cuptiEnableCallback": [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32],
                      "cuptiUnsubscribe": [ctypes.c_void_p]}
        for name, arguments in signatures.items():
            function = getattr(ids.library, name)
            function.argtypes, function.restype = arguments, ctypes.c_int

    def _call(self, name, *arguments):
        result = int(getattr(self.ids.library, name)(*arguments))
        row = {"api": name, "result": result}
        if name == "cuptiEnableCallback":
            row["callback_id"] = int(arguments[-1])
        self.receipt["results"].append(row)
        if result:
            self.receipt["errors"].append(dict(row))
        return result == 0

    def start(self):
        _diagnostic_stage("subscriber.cuptiSubscribe.enter")
        if not self._call("cuptiSubscribe", ctypes.byref(self.handle), self.callback, None):
            _diagnostic_stage("subscriber.cuptiSubscribe.exit", self.receipt["results"][-1]["result"])
            return
        self.owned = True
        _OWNED_CALLBACKS.append(self)
        self.receipt.update(subscribed=True, subscription_active=True, subscriber_handle=int(self.handle.value))
        _diagnostic_stage("subscriber.cuptiSubscribe.exit", self.receipt["results"][-1]["result"])
        for callback_id in (NODE_CLONED, EXEC_CREATED):
            _diagnostic_stage("subscriber.cuptiEnableCallback.enter", callback_id)
            if not self._call("cuptiEnableCallback", 1, self.handle, RESOURCE, callback_id):
                _diagnostic_stage("subscriber.cuptiEnableCallback.exit", self.receipt["results"][-1]["result"])
                return
            _diagnostic_stage("subscriber.cuptiEnableCallback.exit", self.receipt["results"][-1]["result"])

    def close(self):
        if not self.owned:
            return True  # Never unsubscribe somebody else's occupied slot.
        if not self._call("cuptiUnsubscribe", self.handle):
            return False  # Keep the callback/library alive and do not start Kineto.
        self.owned = False
        self.receipt.update(unsubscribed=True, subscription_active=False)
        _OWNED_CALLBACKS.remove(self)
        return True

    def _callback(self, userdata, domain, callback_id, pointer):
        try:
            if domain != RESOURCE or callback_id not in (NODE_CLONED, EXEC_CREATED) or not pointer:
                raise ValueError("Unexpected or empty graph resource callback")
            resource = ctypes.cast(pointer, ctypes.POINTER(CuptiResourceData)).contents
            if not resource.resourceDescriptor:
                raise ValueError("Graph resource callback has no descriptor")
            native = ctypes.cast(resource.resourceDescriptor, ctypes.POINTER(CuptiGraphData)).contents
            row = {"callback_id": callback_id}
            if callback_id == NODE_CLONED:
                row.update(node=int(native.node or 0), original_node=int(native.originalNode or 0))
                self.receipt["events"].append(row)
                if not row["node"] or not row["original_node"]:
                    raise ValueError("Clone callback has no original/node handle")
                row.update(node_id=self.ids.node_id(row["node"]),
                           original_node_id=self.ids.node_id(row["original_node"]))
            else:
                row.update(graph=int(native.graph or 0), graph_exec=int(native.graphExec or 0))
                self.receipt["events"].append(row)
                if not row["graph"] or not row["graph_exec"]:
                    raise ValueError("Exec callback has no graph/exec handle")
                row.update(graph_id=self.ids.graph_id(row["graph"]),
                           exec_graph_id=self.ids.exec_id(row["graph_exec"]))
        except BaseException as error:
            self.receipt["errors"].append({"callback_id": callback_id,
                "error_type": type(error).__name__, "message": str(error)})


def api_inspection(torch, runtime):
    import cuda.bindings.driver as driver

    names = ("cudaGraphGetNodes", "cudaGraphGetEdges", "cudaGraphNodeGetType",
             "cudaGraphEventRecordNodeGetEvent")
    library = cupti_library()
    cupti = CuptiGraphIds(library)  # Resolves typed exports only; calls no CUDA/CUPTI function.
    callbacks = CuptiGraphCallbacks(cupti)  # Resolves callback exports without subscribing.
    header = library.parent.parent / "include" / "cupti_activity.h"
    declarations = [{"line": i + 1, "declaration": line} for i, line in enumerate(header.read_text().splitlines())
                    if line.startswith("CUptiResult CUPTIAPI cuptiGetGraph")]
    runtime_header = header.with_name("cuda_runtime_api.h")
    runtime_declarations = [{"line": i + 1, "declaration": line.strip()}
        for i, line in enumerate(runtime_header.read_text().splitlines())
        if line.strip().startswith("extern") and any(name + "(" in line for name in names)]
    return {"python_api": {name: hasattr(torch.cuda.CUDAGraph, name) for name in
                          ("raw_cuda_graph", "raw_cuda_graph_exec", "instantiate")},
            "mapping_api": "CUDA13_topology_and_CUPTI_ID_queries",
            "cuda_bindings_version": importlib.metadata.version("cuda-bindings"),
            "runtime_api": {name: hasattr(runtime, name) for name in names},
            "cuda_runtime_version": importlib.metadata.version("nvidia-cuda-runtime"),
            "runtime_header": str(runtime_header), "runtime_declarations": runtime_declarations,
            "cupti_version": importlib.metadata.version("nvidia-cuda-cupti"),
            "cupti_library": str(library), "cupti_header": str(header),
            "cupti_declarations": declarations,
            "cupti_missing_python_bindings": {name: not hasattr(runtime, name) and not hasattr(driver, name)
                for name in ("cuptiGetGraphNodeId", "cuptiGetGraphId", "cuptiGetGraphExecId")},
            "cupti_typed_exports": {name: hasattr(cupti.library, name)
                for name in ("cuptiGetGraphNodeId", "cuptiGetGraphId", "cuptiGetGraphExecId")},
            "callback_abi": callback_abi(),
            "callback_missing_python_bindings": {name: not hasattr(runtime, name) and not hasattr(driver, name)
                for name in ("cuptiSubscribe", "cuptiEnableCallback", "cuptiUnsubscribe")},
            "callback_typed_exports": {name: hasattr(callbacks.ids.library, name)
                for name in ("cuptiSubscribe", "cuptiEnableCallback", "cuptiUnsubscribe")},
            "mapped_cupti_libraries": sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                              if "libcupti.so" in line}),
            "event_cuda_event_descriptor": hasattr(torch.cuda.Event, "cuda_event"),
            "rejected_get_graph_data_source": inspect.getsource(torch.cuda.CUDAGraph.get_graph_data),
            "cuda_initialized": torch.cuda.is_initialized()}


def target_kernels(events):
    markers = [e for e in events if e.get("name") == TARGET and e.get("ph") == "X"
               and e.get("cat") == "user_annotation"]
    if len(markers) != 1:
        raise ValueError(f"Expected one target replay annotation, got {len(markers)}")
    marker = markers[0]
    launches = [e for e in events if e.get("cat") == "cuda_runtime"
                and e.get("name", "").startswith("cudaGraphLaunch")
                and e.get("pid") == marker.get("pid") and e.get("tid") == marker.get("tid")
                and marker["ts"] <= e["ts"]
                and e["ts"] + e.get("dur", 0) <= marker["ts"] + marker["dur"]]
    if len(launches) != 1:
        raise ValueError(f"Expected one target cudaGraphLaunch, got {len(launches)}")
    launch = launches[0]
    correlation = launch["args"]["correlation"]
    kernels = [e for e in events if e.get("cat") == "kernel"
               and e.get("args", {}).get("correlation") == correlation]
    if not kernels:
        raise ValueError("Target graph launch has no correlated kernels")
    for kernel in kernels:
        if any(int(kernel["args"][key]) <= 0 for key in ("graph id", "graph node id")):
            raise ValueError("Target replay kernel has no positive graph/node identity")
    return launch, kernels


def match_replay_nodes(dag, kernels):
    """Exact-source check retained for the immutable attempt1 CPU reproducer only."""
    nodes = [n for n in dag["nodes"] if n["node_type"] == "kernel"]
    graph_ids = {int(k["args"]["graph id"]) for k in kernels}
    matches = [kind for kind in ("source_graph_id", "exec_graph_id") if graph_ids == {int(dag[kind])}]
    if not matches:
        raise ValueError("Replay graph IDs match neither the exact CUPTI source nor exec graph ID")
    lookup = {int(n["cupti_node_id"]): n["index"] for n in nodes}
    keys = [int(k["args"]["graph node id"]) for k in kernels]
    if len(lookup) != len(nodes) or len(keys) != len(set(keys)) or set(keys) != set(lookup):
        raise ValueError("Replay uint64 node IDs do not exactly match the retained source DAG")
    return {"trace_graph_id": next(iter(graph_ids)), "graph_id_matches": matches,
            "node_identity": "cupti_uint64_exact", "kernel_node_indices": [lookup[key] for key in keys]}


def require_probe_graph(dag, event_handles):
    labels = {f"{scope}.{edge}" for scope in SCOPE_NAMES for edge in BOUNDARY_NAMES}
    kernel_count = len(SCOPE_NAMES) * KERNELS_PER_SCOPE + UNSCOPED_KERNELS
    nodes = dag["nodes"]
    kernels = [n for n in nodes if n["node_type"] == "kernel"]
    events = [n for n in nodes if n["node_type"] == "event_record"]
    if (len(nodes) != kernel_count + len(labels) or len(kernels) != kernel_count
            or len(events) != len(labels)):
        raise ValueError("Frozen probe graph must contain exactly five kernels and four event nodes")
    handles = [n.get("event_handle") for n in events]
    if (set(event_handles) != labels or len(set(event_handles.values())) != len(labels)
            or any(not h for h in handles) or len(set(handles)) != len(labels)
            or set(handles) != set(event_handles.values())):
        raise ValueError("Frozen probe graph requires four unique named event boundaries")
    return kernel_count


def match_witnessed_replay(report, kernels):
    capture, dag = report["callback_capture"], report["dag"]
    kernel_count = require_probe_graph(dag, report["event_handles"])
    if (not capture["subscribed"] or not capture["unsubscribed"]
            or capture["subscription_active"] or capture["errors"]):
        raise ValueError("CUPTI resource subscription or callback collection failed")
    before, after = report["id_snapshots"]["before"], report["id_snapshots"]["after"]
    expected = {"source_graph_id": dag["source_graph_id"], "exec_graph_id": dag["exec_graph_id"],
                "node_ids": {str(n["handle"]): n["cupti_node_id"] for n in dag["nodes"]}}
    if before != expected or after != before:
        raise ValueError("Full CUPTI IDs changed or snapshots do not match the retained DAG")
    execs = [r for r in capture["events"] if r["callback_id"] == EXEC_CREATED]
    clones = [r for r in capture["events"] if r["callback_id"] == NODE_CLONED]
    relation = {"graph": report["raw_cuda_graph"], "graph_exec": report["raw_cuda_graph_exec"],
                "graph_id": before["source_graph_id"], "exec_graph_id": before["exec_graph_id"]}
    if (len(execs) != 1 or any(execs[0][k] != v for k, v in relation.items())
            or len(capture["events"]) != len(execs) + len(clones)):
        raise ValueError("No unique directly witnessed source/exec graph relation")
    if report["U4"] != "PASS":
        raise ValueError("Retained-source scope boundaries are not validated")
    by_handle = {n["handle"]: n for n in dag["nodes"]}
    links = {}
    for row in clones:
        node = by_handle.get(row["original_node"])
        if node is None or node["index"] in links:
            raise ValueError("Foreign or duplicate original node in clone callbacks")
        if (row["original_node_id"] != node["cupti_node_id"] or not row["node"]
                or row["node"] in by_handle or row["node_id"] <= 0):
            raise ValueError("Stale or invalid directly witnessed clone relation")
        links[node["index"]] = row
    if (len(links) != len(dag["nodes"])
            or len({r["node"] for r in links.values()}) != len(links)
            or len({r["node_id"] for r in links.values()}) != len(links)):
        raise ValueError("Incomplete or ambiguous clone coverage, including event nodes")
    lookup = {links[n["index"]]["node_id"]: n["index"]
              for n in dag["nodes"] if n["node_type"] == "kernel"}
    keys = [int(k["args"]["graph node id"]) for k in kernels]
    if ({int(k["args"]["graph id"]) for k in kernels} != {before["exec_graph_id"]}
            or len(keys) != kernel_count or len(set(keys)) != len(keys) or set(keys) != set(lookup)):
        raise ValueError("Replay kernels do not exactly match the directly witnessed exec nodes")
    transfer = {scope: {"start_event": links[v["start_node_index"]],
                        "end_event": links[v["end_node_index"]],
                        "kernel_node_ids": [links[i]["node_id"] for i in v["kernel_node_indices"]]}
                for scope, v in report["scope_mapping"]["scopes"].items()}
    return {"node_identity": "direct_cupti_clone_callbacks", "graph_relation": execs[0],
            "kernel_node_indices": [lookup[key] for key in keys], "scope_transfer": transfer,
            "unscoped_kernel_ids": [links[i]["node_id"]
                for i in report["scope_mapping"]["unscoped_kernel_indices"]],
            "clone_links": list(links.values())}


def scope_nodes(dag, event_handles):
    nodes = {n["index"]: n for n in dag["nodes"]}
    if (len(nodes) != len(dag["nodes"])
            or len({n["cupti_node_id"] for n in nodes.values()}) != len(nodes)
            or len({n["handle"] for n in nodes.values()}) != len(nodes)):
        raise ValueError("Retained DAG node IDs are not unique")
    for index, node in nodes.items():
        for other in node["dependents"]:
            if other not in nodes or index not in nodes[other]["dependencies"]:
                raise ValueError("DAG edges are missing or inconsistent")
        for other in node["dependencies"]:
            if other not in nodes or index not in nodes[other]["dependents"]:
                raise ValueError("DAG edges are missing or inconsistent")

    indegrees = {i: len(n["dependencies"]) for i, n in nodes.items()}
    pending, visited = [i for i, count in indegrees.items() if count == 0], 0
    while pending:
        current = pending.pop()
        visited += 1
        for other in nodes[current]["dependents"]:
            indegrees[other] -= 1
            if indegrees[other] == 0:
                pending.append(other)
    if visited != len(nodes):
        raise ValueError("Retained graph contains a cycle")

    def reachable(index, field):
        found, pending = set(), list(nodes[index][field])
        while pending:
            current = pending.pop()
            if current == index:
                raise ValueError("Retained graph contains a cycle")
            if current not in found:
                found.add(current)
                pending.extend(nodes[current][field])
        return found

    bounds = {}
    for label, handle in event_handles.items():
        matched = [n for n in nodes.values() if n.get("event_handle") == handle]
        if not handle or len(matched) != 1:
            raise ValueError(f"Event handle does not identify exactly one node: {label}")
        index = matched[0]["index"]
        if nodes[index]["node_type"] != "event_record":
            raise ValueError("Scope boundary is not an event record node")
        bounds[label] = index
    if len(set(bounds.values())) != len(bounds):
        raise ValueError("Scope boundaries share an event record node")
    kernel_indices = {i for i, n in nodes.items() if n["node_type"] == "kernel"}
    owners, scopes = {}, {}
    for scope in sorted({label.rsplit(".", 1)[0] for label in bounds}):
        start, end = bounds[f"{scope}.start"], bounds[f"{scope}.end"]
        after_start = reachable(start, "dependents")
        if end not in after_start:
            raise ValueError("Scope end is not reachable from its start")
        members = kernel_indices & after_start & reachable(end, "dependencies")
        if not members or any(i in owners for i in members):
            raise ValueError("Scope ownership is empty or ambiguous")
        owners.update(dict.fromkeys(members, scope))
        scopes[scope] = {"start_node_index": start, "end_node_index": end,
                         "kernel_node_indices": sorted(members)}
    return {"scopes": scopes, "unscoped_kernel_indices": sorted(kernel_indices - owners.keys())}


def retained_graph(graph, runtime, events, result):
    raw = graph.raw_cuda_graph()
    result["raw_cuda_graph"] = raw
    result["raw_cuda_graph_exec"] = graph.raw_cuda_graph_exec()
    _, count = checked(runtime.cudaGraphGetNodes(raw, numNodes=0))
    handles, _ = checked(runtime.cudaGraphGetNodes(raw, numNodes=count))
    result["event_handles"] = {label: int(event.cuda_event) for label, event in events.items()}
    dag = {"nodes": []}
    result["dag"] = dag
    types = {runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel: "kernel",
             runtime.cudaGraphNodeType.cudaGraphNodeTypeEventRecord: "event_record"}
    for index, handle in enumerate(handles):
        node_type, = checked(runtime.cudaGraphNodeGetType(handle))
        row = {"index": index, "handle": int(handle), "node_type": types.get(node_type, str(node_type)),
               "dependencies": [], "dependents": []}
        dag["nodes"].append(row)
        if node_type == runtime.cudaGraphNodeType.cudaGraphNodeTypeEventRecord:
            event, = checked(runtime.cudaGraphEventRecordNodeGetEvent(handle))
            row["event_handle"] = int(event)
    by_handle = {n["handle"]: n for n in dag["nodes"]}
    _, _, _, count = checked(runtime.cudaGraphGetEdges(raw, numEdges=0))
    if count:
        sources, targets, _, _ = checked(runtime.cudaGraphGetEdges(raw, numEdges=count))
        for source, target in zip(sources, targets):
            by_handle[int(source)]["dependents"].append(by_handle[int(target)]["index"])
            by_handle[int(target)]["dependencies"].append(by_handle[int(source)]["index"])
    cupti = CuptiGraphIds(cupti_library())
    dag["source_graph_id"] = cupti.graph_id(raw)
    dag["exec_graph_id"] = cupti.exec_id(result["raw_cuda_graph_exec"])
    for node in dag["nodes"]:
        node["cupti_node_id"] = cupti.node_id(node["handle"])


def id_snapshot(graph, nodes, ids):
    return {"source_graph_id": ids.graph_id(graph.raw_cuda_graph()),
            "exec_graph_id": ids.exec_id(graph.raw_cuda_graph_exec()),
            "node_ids": {str(n["handle"]): ids.node_id(n["handle"]) for n in nodes}}


def run_probe(torch, runtime, output, elements, warmup, report):
    _diagnostic_stage("allocation.enter")
    stream = torch.cuda.Stream()
    x = torch.ones(elements, device="cuda")
    a, b, c, d, outside, prime = [torch.empty_like(x) for _ in range(6)]
    events = {f"{scope}.{edge}": torch.cuda.Event(enable_timing=True, external=True)
              for scope in SCOPE_NAMES for edge in BOUNDARY_NAMES}
    _diagnostic_stage("allocation.exit")

    def operations():
        events["scope_a.start"].record()
        torch.add(x, x, out=a)
        torch.mul(a, x, out=b)
        events["scope_a.end"].record()
        events["scope_b.start"].record()
        torch.add(b, x, out=c)
        torch.mul(c, x, out=d)
        events["scope_b.end"].record()
        torch.sub(d, x, out=outside)

    _diagnostic_stage("warmup.enter")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            operations()  # Initializes events and kernel dispatch before capture.
    torch.cuda.synchronize()
    _diagnostic_stage("warmup.exit")
    _diagnostic_stage("capture_setup.enter")
    ids = CuptiGraphIds(cupti_library())
    capture = CuptiGraphCallbacks(ids)
    report["callback_capture"] = capture.receipt
    _diagnostic_stage("capture_setup.exit")
    try:
        capture.start()
        _diagnostic_stage("graph_constructor.enter")
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        _diagnostic_stage("graph_constructor.exit")
        _diagnostic_stage("capture.enter")
        with torch.cuda.graph(graph, stream=stream):
            operations()
        _diagnostic_stage("capture.exit")
        _diagnostic_stage("instantiate.enter")
        graph.instantiate()
        _diagnostic_stage("instantiate.exit")
    finally:
        try:
            _diagnostic_stage("subscriber.cuptiUnsubscribe.enter")
        finally:
            released = capture.close()
        _diagnostic_stage("subscriber.cuptiUnsubscribe.exit", capture.receipt["unsubscribed"])
    try:
        _diagnostic_stage("retained_graph.enter")
        retained_graph(graph, runtime, events, report)
        _diagnostic_stage("retained_graph.exit")
        report.update(U2="PASS", U2_basis="retained_source_topology_and_IDs_before_Kineto")
        analyze_retained_scopes(report)
    except Exception:
        report.update(U2="FAIL", api_error=traceback.format_exc())
    if not released:
        report.update(U3="FAIL", profiler_blocked="Owned CUPTI subscriber could not be released")
        return
    _diagnostic_stage("warmup_replay.enter")
    graph.replay()
    torch.cuda.synchronize()
    _diagnostic_stage("warmup_replay.exit")
    report["id_snapshots"] = {}
    _diagnostic_stage("kineto.enter")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        _diagnostic_stage("kineto.entered")
        torch.add(x, x, out=prime)  # Deliberately the same kernel class, outside the graph launch.
        torch.cuda.synchronize()
        if report["U2"] == "PASS":
            try:
                _diagnostic_stage("snapshot_before.enter")
                report["id_snapshots"]["before"] = id_snapshot(graph, report["dag"]["nodes"], ids)
                _diagnostic_stage("snapshot_before.exit")
            except Exception:
                report["snapshot_error_before"] = traceback.format_exc()
        _diagnostic_stage("target_replay.enter")
        with torch.profiler.record_function(TARGET):
            graph.replay()
        torch.cuda.synchronize()
        _diagnostic_stage("target_replay.exit")
        if report["U2"] == "PASS":
            try:
                _diagnostic_stage("snapshot_after.enter")
                report["id_snapshots"]["after"] = id_snapshot(graph, report["dag"]["nodes"], ids)
                _diagnostic_stage("snapshot_after.exit")
            except Exception:
                report["snapshot_error"] = traceback.format_exc()
    _diagnostic_stage("kineto.exit")
    trace = output / "replay_trace.json"
    _diagnostic_stage("trace_export.enter")
    profile.export_chrome_trace(str(trace))
    _diagnostic_stage("trace_export.exit")
    report["trace_path"] = str(trace)
    trace_events = json.loads(trace.read_text())["traceEvents"]
    _diagnostic_stage("trace_analysis.enter")
    analyze_trace(report, trace_events)
    _diagnostic_stage("trace_analysis.exit")


def analyze_retained_scopes(report):
    if report["U2"] == "PASS":
        report["U4_basis"] = "retained_source_graph_only"
        try:
            require_probe_graph(report["dag"], report["event_handles"])
            report["scope_mapping"] = scope_nodes(report["dag"], report["event_handles"])
            if (any(len(v["kernel_node_indices"]) != KERNELS_PER_SCOPE
                    for v in report["scope_mapping"]["scopes"].values())
                    or len(report["scope_mapping"]["unscoped_kernel_indices"]) != UNSCOPED_KERNELS):
                raise ValueError("Tiny graph must have two kernels per scope and one unscoped kernel")
            report["U4"] = "PASS"
        except Exception:
            report.update(U4="FAIL", scope_error=traceback.format_exc())


def analyze_trace(report, trace_events):
    """Inspect a captured trace without CUDA calls or modifying its raw artifacts."""
    analyze_retained_scopes(report)
    report["all_kernel_count"] = sum(e.get("cat") == "kernel" for e in trace_events)
    try:
        launch, kernels = target_kernels(trace_events)
        report.update(U1="PASS", target_launch=launch, target_kernels=kernels)
    except Exception:
        report.update(U1="FAIL", trace_error=traceback.format_exc())
        return
    if report["U2"] != "PASS":
        return
    try:
        report["replay_node_mapping"] = match_witnessed_replay(report, kernels)
        report["U3"] = "PASS"
    except Exception:
        report.update(U3="FAIL", id_error=traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect-api", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.elements <= 0 or args.warmup <= 0:
        parser.error("elements and warmup must be positive")
    _diagnostic_stage("import_torch.enter")
    import torch
    _diagnostic_stage("import_torch.exit")
    _diagnostic_stage("import_cuda.enter")
    import cuda.bindings.runtime as runtime
    _diagnostic_stage("import_cuda.exit")

    _diagnostic_stage("output_mkdir.enter")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _diagnostic_stage("output_mkdir.exit")
    report = {"schema_version": "cuda_graph_scope_probe.v3", "command": args.command,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "image_digest_from_launcher": args.image_digest, "torch_version": torch.__version__,
              "torch_cuda_version": torch.version.cuda, "api": _diagnostic_api_inspection(torch, runtime),
              "performance_acceptance": False, "consumer_admission": False,
              "U1": "NOT_RUN", "U2": "NOT_RUN", "U3": "NOT_RUN", "U4": "NOT_RUN"}
    if args.command == "run":
        try:
            with torch.inference_mode():
                _diagnostic_stage("device_name.enter")
                report["device_name"] = torch.cuda.get_device_name()
                _diagnostic_stage("device_name.exit")
                _diagnostic_stage("driver_version.enter")
                report["driver_version"], = checked(runtime.cudaDriverGetVersion())
                _diagnostic_stage("driver_version.exit")
                run_probe(torch, runtime, args.output_dir, args.elements, args.warmup, report)
        except Exception:
            report["probe_error"] = traceback.format_exc()
    path = args.output_dir / "probe_report.json"
    _diagnostic_stage("report_write.enter")
    path.write_text(json.dumps(report, indent=2) + "\n")
    _diagnostic_stage("report_write.exit")
    print(json.dumps({"report": str(path), **{k: report[k] for k in ("U1", "U2", "U3", "U4")}}))
    if args.command == "run" and any(report[k] != "PASS" for k in ("U1", "U2", "U3", "U4")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Record operator ownership while CUDA graphs are being captured.

Scope boundaries sweep the graph on its origin stream before changing the
scope stack. The final sweep precedes graph instantiation, whose CUPTI clone
callbacks bridge captured node ids to the ids seen in Kineto replay traces.
Breakable graphs retain eager functions with their registering scope for replay.
CUDA and torch are imported only when constructing the real query implementation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol

ScopeOwner = tuple[int, str]
# CUDA 13 cupti_callbacks.h and cupti_result.h.
CUPTI_CB_DOMAIN_RESOURCE = 3
CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED = 20
CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED = 39
_CUPTI_SUCCESS = 0
# CUDA runtime/driver success enums in driver_types.h and cuda.h.
_CUDA_SUCCESS = 0
_CALLBACK_ENABLED = 1
_CALLBACK_DISABLED = 0
_CUPTI_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p
)
_NODE_TYPES = frozenset({"kernel", "memcpy", "memset", "event_record", "event_wait", "empty"})
# Retain native callback pointers if an unsubscribe fails.
_ACTIVE_SUBSCRIPTIONS: list[CuptiCloneRecorder] = []


@dataclass(frozen=True)
class KernelLaunch:
    name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


@dataclass(frozen=True)
class GraphCapture:
    ordinal: int
    kind: str
    breakable_capture: int | None
    origin_stream: int


@dataclass(frozen=True)
class GraphNode:
    capture: int
    node_id: int
    node_type: str
    kernel_name: str | None
    kernel_name_demangled: str | None
    grid: tuple[int, int, int] | None
    block: tuple[int, int, int] | None
    dependencies: tuple[int, ...]
    owner: ScopeOwner | None


@dataclass(frozen=True)
class ScopeEntry:
    capture: int
    owner: ScopeOwner | None
    parent: ScopeOwner | None


@dataclass(frozen=True)
class EagerBreak:
    breakable_capture: int
    segment_index: int
    owner: ScopeOwner | None


@dataclass(frozen=True)
class NodeClone:
    node_id: int
    original_node_id: int


def _json_lists(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_lists(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_lists(item) for item in value]
    return value


def _owner(value: Any) -> ScopeOwner | None:
    if value is None:
        return None
    if len(value) != 2 or not isinstance(value[0], int) or not isinstance(value[1], str):
        raise ValueError(f"Invalid scope owner {value!r}; expected [layer, scope]")
    return value[0], value[1]


def _dimensions(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if len(value) != 3 or any(not isinstance(item, int) for item in value):
        raise ValueError(f"Invalid launch dimensions {value!r}; expected three integers")
    return tuple(value)


@dataclass(frozen=True)
class CaptureRecord:
    captures: tuple[GraphCapture, ...] = ()
    nodes: tuple[GraphNode, ...] = ()
    scope_entries: tuple[ScopeEntry, ...] = ()
    eager_breaks: tuple[EagerBreak, ...] = ()
    clones: tuple[NodeClone, ...] = ()

    def __post_init__(self) -> None:
        ids: set[int] = set()
        for node in (*self.nodes, *self.clones):
            if node.node_id in ids:
                raise ValueError(f"Duplicate node id {node.node_id} in capture record")
            ids.add(node.node_id)

    def to_json(self) -> dict[str, Any]:
        return {
            field: [_json_lists(asdict(row)) for row in getattr(self, field)]
            for field in ("captures", "nodes", "scope_entries", "eager_breaks", "clones")
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> CaptureRecord:
        return cls(
            captures=tuple(GraphCapture(**row) for row in value["captures"]),
            nodes=tuple(GraphNode(
                **{key: item for key, item in row.items()
                   if key not in {"owner", "grid", "block", "dependencies"}},
                owner=_owner(row["owner"]), grid=_dimensions(row["grid"]),
                block=_dimensions(row["block"]), dependencies=tuple(row["dependencies"]),
            ) for row in value["nodes"]),
            scope_entries=tuple(ScopeEntry(
                capture=row["capture"], owner=_owner(row["owner"]), parent=_owner(row["parent"]),
            ) for row in value["scope_entries"]),
            eager_breaks=tuple(EagerBreak(
                breakable_capture=row["breakable_capture"], segment_index=row["segment_index"],
                owner=_owner(row["owner"]),
            ) for row in value["eager_breaks"]),
            clones=tuple(NodeClone(**row) for row in value["clones"]),
        )


class GraphQueries(Protocol):
    def current_stream(self) -> int: ...
    def capture_graph(self, stream: int) -> int: ...
    def graph_nodes(self, graph: int) -> list[int]: ...
    def node_type(self, node: int) -> str: ...
    def kernel_launch(self, node: int) -> KernelLaunch: ...
    def dependencies(self, node: int) -> list[int]: ...
    def node_id(self, node: int) -> int: ...
    def demangle(self, name: str) -> str | None: ...


def resolve_cupti_library() -> Any:
    """Load the single CUPTI library already mapped by torch/Kineto."""
    paths = {
        parts[5] for line in Path("/proc/self/maps").read_text().splitlines()
        if len(parts := line.split(maxsplit=5)) == 6
        and Path(parts[5]).name.startswith("libcupti.so")
    }
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one mapped libcupti.so path, found {sorted(paths)}")
    return ctypes.CDLL(paths.pop())


def _checked(api: str, result: tuple[Any, ...]) -> tuple[Any, ...]:
    code = int(result[0])
    if code != _CUDA_SUCCESS:
        raise RuntimeError(f"{api} failed with error code {code}")
    return result[1:]


def _node_id(library: Any, node: int) -> int:
    result = ctypes.c_uint64()
    code = int(library.cuptiGetGraphNodeId(ctypes.c_void_p(node), ctypes.byref(result)))
    if code != _CUPTI_SUCCESS:
        raise RuntimeError(f"cuptiGetGraphNodeId failed with error code {code}")
    return result.value


def _configure_node_ids(library: Any) -> None:
    library.cuptiGetGraphNodeId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    library.cuptiGetGraphNodeId.restype = ctypes.c_int


class CudaGraphQueries:
    """CUDA graph queries with explicit error handling and no ownership transfer."""

    def __init__(self) -> None:
        import torch
        from cuda.bindings import driver, runtime

        self.torch, self.runtime, self.driver = torch, runtime, driver
        self.cupti = resolve_cupti_library()
        _configure_node_ids(self.cupti)
        path = ctypes.util.find_library("stdc++")
        if path is None:
            raise RuntimeError("Cannot resolve libstdc++ for __cxa_demangle")
        self._stdcpp = ctypes.CDLL(path)
        self._demangle = getattr(self._stdcpp, "__cxa_demangle")
        self._demangle.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._demangle.restype = ctypes.c_void_p
        self._libc = ctypes.CDLL(None)
        self._libc.free.argtypes = [ctypes.c_void_p]
        self._libc.free.restype = None

    def current_stream(self) -> int:
        return int(self.torch.cuda.current_stream().cuda_stream)

    def capture_graph(self, stream: int) -> int:
        status, _, graph, *_ = _checked(
            "cudaStreamGetCaptureInfo", self.runtime.cudaStreamGetCaptureInfo(stream)
        )
        if status != self.runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
            raise RuntimeError(f"cudaStreamGetCaptureInfo: stream {stream} is not capturing (status {int(status)})")
        return int(graph)

    def graph_nodes(self, graph: int) -> list[int]:
        _, count = _checked("cudaGraphGetNodes", self.runtime.cudaGraphGetNodes(graph, numNodes=0))
        nodes, count = _checked("cudaGraphGetNodes", self.runtime.cudaGraphGetNodes(graph, numNodes=count))
        return [int(node) for node in nodes[:count]]

    def node_type(self, node: int) -> str:
        kind, = _checked("cudaGraphNodeGetType", self.runtime.cudaGraphNodeGetType(node))
        # Names from CUDA driver_types.h; use bindings' enum values directly.
        enum = self.runtime.cudaGraphNodeType
        names = {
            enum.cudaGraphNodeTypeKernel: "kernel", enum.cudaGraphNodeTypeMemcpy: "memcpy",
            enum.cudaGraphNodeTypeMemset: "memset", enum.cudaGraphNodeTypeEmpty: "empty",
            enum.cudaGraphNodeTypeEventRecord: "event_record", enum.cudaGraphNodeTypeWaitEvent: "event_wait",
        }
        if kind not in names:
            raise ValueError(f"cudaGraphNodeGetType returned unsupported node type {int(kind)}")
        return names[kind]

    def kernel_launch(self, node: int) -> KernelLaunch:
        params, = _checked("cuGraphKernelNodeGetParams", self.driver.cuGraphKernelNodeGetParams(node))
        if int(params.func):
            name, = _checked("cuFuncGetName", self.driver.cuFuncGetName(params.func))
        elif int(params.kern):
            name, = _checked("cuKernelGetName", self.driver.cuKernelGetName(params.kern))
        else:
            raise ValueError(f"cuGraphKernelNodeGetParams: node {node} has neither func nor kern")
        return KernelLaunch(
            name=name.decode("utf-8"),
            grid=(int(params.gridDimX), int(params.gridDimY), int(params.gridDimZ)),
            block=(int(params.blockDimX), int(params.blockDimY), int(params.blockDimZ)),
        )

    def dependencies(self, node: int) -> list[int]:
        _, _, count = _checked(
            "cudaGraphNodeGetDependencies", self.runtime.cudaGraphNodeGetDependencies(node, pNumDependencies=0)
        )
        nodes, _, count = _checked(
            "cudaGraphNodeGetDependencies", self.runtime.cudaGraphNodeGetDependencies(node, pNumDependencies=count)
        )
        return [int(item) for item in nodes[:count]]

    def node_id(self, node: int) -> int:
        return _node_id(self.cupti, node)

    def demangle(self, name: str) -> str | None:
        status = ctypes.c_int()
        pointer = self._demangle(name.encode("utf-8"), None, None, ctypes.byref(status))
        try:
            return ctypes.string_at(pointer).decode("utf-8") if status.value == 0 else None
        finally:
            if pointer:
                self._libc.free(pointer)


class CuptiResourceData(ctypes.Structure):
    """CUpti_ResourceData in CUDA 13 cupti_callbacks.h (three pointer fields)."""

    _fields_ = [(name, ctypes.c_void_p) for name in ("context", "resourceHandle", "resourceDescriptor")]


class CuptiGraphData(ctypes.Structure):
    """CUpti_GraphData in CUDA 13 cupti_callbacks.h, with native padding."""

    _fields_ = [(name, ctypes.c_void_p) for name in ("graph", "originalGraph", "node", "originalNode")] + [
        ("nodeType", ctypes.c_int), ("dependency", ctypes.c_void_p), ("graphExec", ctypes.c_void_p),
    ]


def parse_clone_payload(cbdata: Any, id_query: Callable[[int], int]) -> tuple[int, int]:
    """Read a resource descriptor while its callback payload remains valid."""
    if not cbdata:
        raise ValueError("GRAPHNODE_CLONED has a null resource payload")
    resource = ctypes.cast(cbdata, ctypes.POINTER(CuptiResourceData)).contents
    if not resource.resourceDescriptor:
        raise ValueError("GRAPHNODE_CLONED has a null resourceDescriptor")
    graph = ctypes.cast(resource.resourceDescriptor, ctypes.POINTER(CuptiGraphData)).contents
    if not graph.node or not graph.originalNode:
        raise ValueError("GRAPHNODE_CLONED has a null node or originalNode")
    return id_query(graph.node), id_query(graph.originalNode)


class CuptiCloneRecorder:
    """Own one CUPTI callback subscription, stopped before any profiler session."""

    def __init__(self, library: Any = None) -> None:
        self.library = resolve_cupti_library() if library is None else library
        _configure_node_ids(self.library)
        signatures = {
            "cuptiSubscribe": [ctypes.POINTER(ctypes.c_void_p), _CUPTI_CALLBACK, ctypes.c_void_p],
            "cuptiEnableCallback": [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32],
            "cuptiUnsubscribe": [ctypes.c_void_p],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        self.clones: list[tuple[int, int]] = []
        self._errors: list[BaseException] = []
        self._subscriber = ctypes.c_void_p()
        self._owned = False
        self._callback_pointer = _CUPTI_CALLBACK(self._callback)

    def _call(self, api: str, *arguments: Any) -> None:
        code = int(getattr(self.library, api)(*arguments))
        if code == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED and api == "cuptiSubscribe":
            raise RuntimeError(
                f"cuptiSubscribe failed with error code {code}: another CUPTI subscriber "
                "(for example an active torch.profiler/Kineto session) holds the slot; "
                "the clone recorder must run before any profiler session"
            )
        if code != _CUPTI_SUCCESS:
            raise RuntimeError(f"{api} failed with error code {code}")

    def __enter__(self) -> CuptiCloneRecorder:
        if self._owned:
            raise ValueError("CuptiCloneRecorder already owns a subscriber")
        self._call("cuptiSubscribe", ctypes.byref(self._subscriber), self._callback_pointer, None)
        self._owned = True
        _ACTIVE_SUBSCRIPTIONS.append(self)
        try:
            self._call("cuptiEnableCallback", _CALLBACK_ENABLED, self._subscriber,
                       CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._owned:
            try:
                self._call("cuptiEnableCallback", _CALLBACK_DISABLED, self._subscriber,
                           CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED)
            finally:
                self._call("cuptiUnsubscribe", self._subscriber)
                self._owned = False
                _ACTIVE_SUBSCRIPTIONS.remove(self)
        if self._errors:
            raise self._errors[0]

    def _callback(self, userdata: Any, domain: int, callback_id: int, cbdata: Any) -> None:
        if domain != CUPTI_CB_DOMAIN_RESOURCE or callback_id != CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED:
            return
        try:
            self.clones.append(parse_clone_payload(cbdata, lambda node: _node_id(self.library, node)))
        except BaseException as error:
            self._errors.append(error)


class GraphCaptureRecorder:
    """Scope stack and capture sweeps, independent of the CUDA query mechanism."""

    def __init__(self, queries: GraphQueries, eager_scope_runner: Callable[[ScopeOwner | None, Callable[[], Any]], Any]) -> None:
        self.queries, self.eager_scope_runner = queries, eager_scope_runner
        self._captures: list[GraphCapture] = []
        self._nodes: list[GraphNode] = []
        self._scope_entries: list[ScopeEntry] = []
        self._eager_breaks: list[EagerBreak] = []
        self._stack: list[ScopeOwner | None] = []
        self._open: GraphCapture | None = None
        self._seen: dict[int, int] = {}
        self._ids: set[int] = set()
        self._breakable_captures: list[Any] = []

    @property
    def current_owner(self) -> ScopeOwner | None:
        """Return the innermost scope owner, or the outside owner."""

        return self._stack[-1] if self._stack else None

    def _breakable_ordinal(self, capture: Any) -> int:
        for ordinal, existing in enumerate(self._breakable_captures):
            if existing is capture:
                return ordinal
        self._breakable_captures.append(capture)
        return len(self._breakable_captures) - 1

    @contextmanager
    def install(self, graph_class: Any, breakable_class: Any) -> Iterator[GraphCaptureRecorder]:
        originals = [(graph_class, name, getattr(graph_class, name), name in vars(graph_class))
                     for name in ("capture_begin", "capture_end")]
        originals.append((breakable_class, "add_eager", breakable_class.add_eager, "add_eager" in vars(breakable_class)))
        begin, end, add_eager = [row[2] for row in originals]

        @wraps(begin)
        def capture_begin(graph_self: Any, *args: Any, **kwargs: Any) -> Any:
            if self._open is not None:
                raise ValueError("Cannot open a CUDA graph capture while another capture is open")
            breakable = breakable_class.current()
            capture = GraphCapture(
                len(self._captures), "graph" if breakable is None else "breakable_segment",
                None if breakable is None else self._breakable_ordinal(breakable), self.queries.current_stream(),
            )
            self._captures.append(capture)
            self._open, self._seen = capture, {}
            try:
                return begin(graph_self, *args, **kwargs)
            except BaseException:
                self._open = None
                raise

        @wraps(end)
        def capture_end(graph_self: Any) -> Any:
            self._sweep()
            self._open = None
            return end(graph_self)

        @wraps(add_eager)
        def eager(capture_self: Any, fn: Callable[[], Any]) -> Any:
            owner = self._stack[-1] if self._stack else None
            ordinal = self._breakable_ordinal(capture_self)
            result = add_eager(capture_self, fn)
            if capture_self.segments[-1] is not fn:
                raise ValueError("Breakable add_eager did not retain the registered function as its last segment")
            index = len(capture_self.segments) - 1

            @wraps(fn)
            def replay() -> Any:
                return self.eager_scope_runner(owner, fn)

            capture_self.segments[index] = replay
            self._eager_breaks.append(EagerBreak(ordinal, index, owner))
            return result

        try:
            graph_class.capture_begin, graph_class.capture_end = capture_begin, capture_end
            breakable_class.add_eager = eager
            yield self
        finally:
            for cls, name, original, local in originals:
                if local:
                    setattr(cls, name, original)
                else:
                    delattr(cls, name)
            self._open = None

    @contextmanager
    def scope(self, owner: ScopeOwner | None) -> Iterator[None]:
        if self._open is not None:
            self._sweep()
            self._scope_entries.append(ScopeEntry(self._open.ordinal, owner, self._stack[-1] if self._stack else None))
        self._stack.append(owner)
        try:
            yield
        finally:
            try:
                if self._open is not None:
                    self._sweep()
            finally:
                self._stack.pop()

    def _sweep(self) -> None:
        capture = self._open
        if capture is None:
            raise ValueError("Cannot sweep without an open CUDA graph capture")
        graph = self.queries.capture_graph(capture.origin_stream)
        new = [node for node in self.queries.graph_nodes(graph) if node not in self._seen]
        for node in new:
            node_id = self.queries.node_id(node)
            if node_id in self._ids:
                raise ValueError(f"Duplicate node id {node_id} in capture record")
            self._ids.add(node_id)
            self._seen[node] = node_id
        for node in new:
            kind = self.queries.node_type(node)
            if kind not in _NODE_TYPES:
                raise ValueError(f"Unsupported captured node type {kind!r}")
            launch = self.queries.kernel_launch(node) if kind == "kernel" else None
            self._nodes.append(GraphNode(
                capture.ordinal, self._seen[node], kind,
                launch.name if launch else None, self.queries.demangle(launch.name) if launch else None,
                launch.grid if launch else None, launch.block if launch else None,
                tuple(self._seen[dependency] for dependency in self.queries.dependencies(node)),
                self._stack[-1] if self._stack else None,
            ))

    def record(self, clones: Iterable[tuple[int, int] | NodeClone]) -> CaptureRecord:
        if self._open is not None:
            raise ValueError("Cannot build a record while a CUDA graph capture is open")
        return CaptureRecord(
            tuple(self._captures), tuple(self._nodes), tuple(self._scope_entries), tuple(self._eager_breaks),
            tuple(row if isinstance(row, NodeClone) else NodeClone(*row) for row in clones),
        )


__all__ = [
    "CaptureRecord", "CudaGraphQueries", "CuptiCloneRecorder", "CuptiGraphData", "CuptiResourceData",
    "CUPTI_CB_DOMAIN_RESOURCE", "CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED",
    "CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED", "EagerBreak", "GraphCapture", "GraphCaptureRecorder",
    "GraphNode", "GraphQueries", "KernelLaunch", "NodeClone", "ScopeEntry", "ScopeOwner",
    "parse_clone_payload", "resolve_cupti_library",
]

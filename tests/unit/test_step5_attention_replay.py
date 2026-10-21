"""CPU module-boundary contracts, not native CUDA execution or timing claims."""

import builtins
import runpy
import sys
from contextlib import contextmanager
from itertools import product
from types import ModuleType, SimpleNamespace

import pytest

from frontier.profiling.step5 import replay
from frontier.profiling.step5.replay import capture_native_replay, choose_graph_mode


@pytest.mark.parametrize("phase,tokens,expected", [
    ("decode", 1, "full"), ("decode", 2, "full"),
    ("prefill", 2, "piecewise"), ("mixed", 2, "piecewise"),
    *((phase, tokens, "none") for phase, tokens in product(
        ("decode", "prefill", "mixed"), (3, 16, 64),
    )),
])
def test_phase_dispatch_capture_ceiling_and_stage04_shapes(phase, tokens, expected):
    assert choose_graph_mode(phase, tokens) == expected


@pytest.mark.parametrize("phase", ["decode", "prefill", "mixed"])
@pytest.mark.parametrize("sizes", [(1, 2, 4, 8), [1, 2, 4, 8]])
def test_caller_capture_sizes_control_exact_matches_and_eager_misses(phase, sizes):
    assert choose_graph_mode(phase, 8, sizes) == ("full" if phase == "decode" else "piecewise")
    assert choose_graph_mode(phase, 9, sizes) == "none"
    assert list(sizes) == [1, 2, 4, 8]


@pytest.mark.parametrize("phase", ["decode", "prefill", "mixed"])
@pytest.mark.parametrize("tokens,sizes,padded", [(1, (2,), 2), (3, (1, 2, 4), 4), (5, (1, 2, 8), 8)])
def test_native_padding_is_rejected_instead_of_capturing_unpadded_rows(phase, tokens, sizes, padded):
    with pytest.raises(ValueError, match=f"pads {tokens} tokens to capture size {padded}.*padded physical rows"):
        choose_graph_mode(phase, tokens, sizes)


@pytest.mark.parametrize("sizes", [
    (), [], None, "12", (0, 1), (-1, 2), (2, 1), (1, 1), (1, 2, 2),
    (True, 2), (1, 2.0), (1, "2"), {1, 2},
])
def test_invalid_capture_sizes_are_rejected_even_on_an_eager_miss(sizes):
    with pytest.raises(ValueError, match="capture_sizes.*positive, sorted, unique integers"):
        choose_graph_mode("prefill", 64, sizes)


@pytest.mark.parametrize("tokens", [0, -1, True, 1.0, None, "1"])
def test_invalid_token_counts_are_rejected(tokens):
    with pytest.raises(ValueError, match="num_tokens"):
        choose_graph_mode("decode", tokens)


@pytest.mark.parametrize("phase", ["", "Prefill", "verify", None, []])
def test_unknown_phases_are_rejected(phase):
    with pytest.raises(ValueError, match="phase"):
        choose_graph_mode(phase, 1)


def reject_native_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torch", "vllm"), f"Unexpected native import: {name}"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_module_import_and_dispatch_need_no_torch_or_vllm(monkeypatch):
    reject_native_imports(monkeypatch)
    namespace = runpy.run_path(replay.__file__)
    assert namespace["choose_graph_mode"]("decode", 1) == "full"
    assert namespace["choose_graph_mode"]("prefill", 64) == "none"


def test_none_defers_forward_and_returns_each_output_without_native_imports(monkeypatch):
    reject_native_imports(monkeypatch)
    calls = []

    def forward(*, runtime_mode, layer=None):
        calls.append((runtime_mode, layer))
        return object()

    run = capture_native_replay(object(), object(), object(), forward, "none")
    assert calls == []
    first, second = run(), run()
    assert first is not second
    assert calls == [("none", None), ("none", None)]


@pytest.mark.parametrize("mode", ["", "FULL", None, 0])
def test_unknown_graph_mode_fails_before_native_imports(monkeypatch, mode):
    reject_native_imports(monkeypatch)
    with pytest.raises(ValueError, match="graph_mode"):
        capture_native_replay(object(), object(), object(), lambda **_: None, mode)


class NativeBoundary:
    """Stand-ins record imported APIs and buffer references without CUDA calls."""

    def __init__(self):
        self.calls = []
        self.native, self.device = object(), object()
        self.global_pool, self.capture_pool_value = object(), object()
        self.default_stream, self.capture_stream = object(), object()
        self.stream = self.default_stream
        self.mode = SimpleNamespace(PIECEWISE=object())
        self.positions, self.hidden = object(), SimpleNamespace(value=1)
        self.graphs, self.wrappers, self.forwards, self.layer_inputs = [], [], [], []
        self.active_graph = None
        self.torch = SimpleNamespace(
            device=self.make_device,
            cuda=SimpleNamespace(CUDAGraph=self.new_graph, graph=self.cuda_graph),
        )
        self.offloader = SimpleNamespace(
            sync_prev_onload=lambda: self.calls.append("offloader:sync"),
            join_after_forward=lambda: self.calls.append("offloader:join"),
        )
        self.layer = self.layer_forward

    def make_device(self, name):
        assert name == "cuda"
        self.calls.append("device:cuda")
        return self.device

    def new_graph(self):
        self.calls.append("graph:new")
        graph = SimpleNamespace(action=None)

        def run():
            self.calls.append("graph:replay")
            graph.action()

        graph.replay = run
        self.graphs.append(graph)
        return graph

    @contextmanager
    def graph_capture(self, *, device):
        assert device is self.device
        previous_stream = self.stream
        self.stream = self.capture_stream
        self.calls.append("graph_capture:enter")
        try:
            yield
        finally:
            self.stream = previous_stream
            self.calls.append("graph_capture:exit")

    def get_pool(self):
        self.calls.append("pool:get")
        return self.global_pool

    @contextmanager
    def capture_pool(self, pool, native):
        assert pool is self.global_pool and native is self.native
        self.calls.append("pool:enter")
        try:
            yield self.capture_pool_value
        finally:
            self.calls.append("pool:exit")

    def current_stream(self):
        self.calls.append("stream:get")
        return self.stream

    @contextmanager
    def cuda_graph(self, graph, pool, *, stream):
        assert graph is self.graphs[-1]
        assert pool is self.capture_pool_value and stream is self.stream
        self.calls.append("cuda_graph:enter")
        self.active_graph = graph
        try:
            yield
        finally:
            self.active_graph = None
            self.calls.append("cuda_graph:exit")

    def get_offloader(self):
        self.calls.append("offloader:get")
        return self.offloader

    def layer_forward(self, positions, hidden):
        self.calls.append("layer")
        self.layer_inputs.append((positions, hidden))
        output = SimpleNamespace(value=hidden.value)
        if self.active_graph is not None:
            self.active_graph.action = lambda: setattr(output, "value", hidden.value)
        return output

    def forward(self, *, runtime_mode, layer=None):
        self.calls.append(f"forward:{runtime_mode}")
        self.forwards.append((runtime_mode, layer))
        return (self.layer if layer is None else layer)(self.positions, self.hidden)

    def breakable_wrapper(self, layer, native, *, runtime_mode):
        assert layer is self.layer and native is self.native
        assert runtime_mode is self.mode.PIECEWISE
        self.calls.append("breakable:new")
        boundary = self

        class Wrapper:
            def __init__(self):
                self.inputs = None
                self.output = None

            def __call__(self, positions, hidden):
                if self.inputs is None:
                    assert boundary.stream is boundary.capture_stream, (
                        "PIECEWISE capture requires the native non-default stream"
                    )
                    boundary.calls.append("breakable:capture")
                    self.inputs = (positions, hidden)
                    self.output = layer(positions, hidden)
                else:
                    assert boundary.stream is boundary.default_stream, (
                        "PIECEWISE replay must remain outside the capture stream context"
                    )
                    assert positions is self.inputs[0] and hidden is self.inputs[1]
                    boundary.calls.append("breakable:replay")
                    self.output.value = hidden.value
                return self.output

        wrapped = Wrapper()
        self.wrappers.append(wrapped)
        return wrapped


@pytest.fixture
def boundary(monkeypatch):
    native = NativeBoundary()
    modules = {}
    exports = {
        "vllm.config": {"CUDAGraphMode": native.mode},
        "vllm.compilation.breakable_cudagraph": {"BreakableCUDAGraphWrapper": native.breakable_wrapper},
        "vllm.compilation.cudagraph_pool": {"capture_pool": native.capture_pool},
        "vllm.distributed.parallel_state": {"graph_capture": native.graph_capture},
        "vllm.model_executor.offloader.base": {"get_offloader": native.get_offloader},
        "vllm.platforms": {"current_platform": SimpleNamespace(get_global_graph_pool=native.get_pool)},
        "vllm.utils.torch_utils": {"current_stream": native.current_stream},
    }
    for name, attributes in exports.items():
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            full_name = ".".join(parts[:end])
            if full_name not in modules:
                module = modules[full_name] = ModuleType(full_name)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, full_name, module)
                if end > 1:
                    setattr(modules[".".join(parts[:end - 1])], parts[end - 1], module)
        modules[name].__dict__.update(attributes)
    return native


def test_full_uses_native_capture_order_runtime_none_and_output_reference(boundary):
    b = boundary
    run = capture_native_replay(b.torch, b.native, b.layer, b.forward, "full")
    assert b.calls == [
        "device:cuda", "graph_capture:enter", "graph:new", "offloader:get", "offloader:sync",
        "pool:get", "pool:enter", "stream:get", "cuda_graph:enter", "forward:none", "layer",
        "offloader:get", "offloader:join", "cuda_graph:exit", "pool:exit", "graph_capture:exit",
    ]
    assert b.forwards == [("none", None)]
    assert b.layer_inputs == [(b.positions, b.hidden)]
    b.calls.clear()
    b.hidden.value = 2
    first = run()
    assert first.value == 2
    b.hidden.value = 3
    assert run() is first
    assert first.value == 3
    assert b.forwards == [("none", None)]
    assert b.calls == ["offloader:get", "offloader:sync", "graph:replay"] * 2
    assert not b.wrappers


def test_piecewise_uses_only_native_wrapper_with_stable_inputs_and_runtime(boundary):
    b = boundary
    torch = SimpleNamespace(device=b.make_device, cuda=object())
    run = capture_native_replay(torch, b.native, b.layer, b.forward, "piecewise")
    assert b.calls == [
        "breakable:new", "device:cuda", "graph_capture:enter", "forward:piecewise",
        "breakable:capture", "layer", "graph_capture:exit",
    ]
    assert b.stream is b.default_stream
    wrapped = b.wrappers[0]
    assert wrapped.inputs[0] is b.positions and wrapped.inputs[1] is b.hidden
    b.calls.clear()
    b.hidden.value = 2
    first = run()
    assert first.value == 2
    b.hidden.value = 3
    assert run() is first
    assert first.value == 3
    assert b.forwards == [("piecewise", wrapped)] * 3
    assert b.calls == ["forward:piecewise", "breakable:replay"] * 2
    assert b.stream is b.default_stream
    assert not b.graphs


@pytest.mark.parametrize("mode", ["full", "piecewise"])
def test_each_capture_has_independent_graph_storage_for_distinct_scope_events(boundary, mode):
    b = boundary
    runs = [capture_native_replay(b.torch, b.native, b.layer, b.forward, mode) for _ in range(3)]
    captures = b.graphs if mode == "full" else b.wrappers
    assert len(captures) == len({id(capture) for capture in captures}) == 3
    assert len({id(run()) for run in runs}) == 3


@pytest.mark.parametrize("mode", ["full", "piecewise"])
def test_capture_failure_propagates_without_replay_or_retry(boundary, mode):
    b = boundary
    calls = []

    def failing(*, runtime_mode, layer=None):
        calls.append((runtime_mode, layer))
        assert b.stream is b.capture_stream, "Native capture must use the non-default stream"
        raise RuntimeError("native forward failure")

    with pytest.raises(RuntimeError, match="native forward failure"):
        capture_native_replay(b.torch, b.native, b.layer, failing, mode)
    assert len(calls) == 1
    assert "graph:replay" not in b.calls
    assert "breakable:replay" not in b.calls
    assert b.stream is b.default_stream
    if mode == "full":
        assert b.calls[-3:] == ["cuda_graph:exit", "pool:exit", "graph_capture:exit"]
    else:
        assert b.calls == ["breakable:new", "device:cuda", "graph_capture:enter", "graph_capture:exit"]

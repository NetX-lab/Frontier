"""Exercise replay orchestration with CPU tensors and a simulated graph runtime."""

from contextlib import contextmanager, nullcontext
import sys
from types import SimpleNamespace

import pytest

from frontier.profiling.experimental.sglang import graph_replay as replay


@pytest.mark.parametrize(
    "name,builder_name,metadata",
    [
        ("shared_expert_activation", "make_dense_primitive", ()),
        ("gdn_core_decode", "make_gdn_core_primitive", ({},)),
        ("attn_rope", "make_attention_primitive", ({}, {})),
        ("moe_routing_topk", "make_moe_routing_primitive", ({},)),
    ],
)
def test_delegated_builders_complete_capture_replay_and_trace(
    monkeypatch, name, builder_name, metadata
):
    torch = pytest.importorskip("torch")
    active_graphs = []
    invocations = []

    class Graph:
        def __init__(self):
            self.calls = []

        def replay(self):
            for fn in self.calls:
                fn()

    @contextmanager
    def capture(graph, **kwargs):
        active_graphs.append(graph)
        yield
        active_graphs.pop()

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 3.0

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(torch.profiler, "record_function", lambda *args: nullcontext())

    def builder(*args, **kwargs):
        output = torch.zeros(1)
        stateful = bool(metadata)

        def run():
            invocations.append(name)
            output.add_(1) if stateful else output.fill_(1)
            return output

        def fn():
            if active_graphs:
                active_graphs[-1].calls.append(run)
            return run()

        reference = torch.ones(1)
        if stateful:
            return fn, reference, (output,), "test_backend", *metadata
        return fn, reference, "test_backend"

    monkeypatch.setattr(replay, builder_name, builder)
    monkeypatch.setattr(replay, "dense_primitive_spec", lambda *args: {})
    monkeypatch.setattr(replay, "gdn_core_spec", lambda *args: {})
    group = SimpleNamespace(
        world_size=1, cpu_group=None,
        graph_capture=lambda: nullcontext(SimpleNamespace(stream=None)),
    )
    row, trace_replay = replay.profile_graph(
        name, 8, 3, 5, SimpleNamespace(embedding_dim=16), group, 0,
        logical_size=4, physical_context_lens=(32,) * 8, trace=True,
    )
    trace_replay()
    trace_replay()

    assert row["backend"] == "test_backend"
    assert row["samples_ms"] == [1.0] * 5
    assert row["correctness_checked"] is True
    assert len(invocations) > 3 * 5


@pytest.mark.parametrize("count", [-1, 0, 1])
def test_direct_graph_api_rejects_invalid_count_before_runtime_import(monkeypatch, count):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="Invalid invocations"):
        replay.profile_graph("shared_expert_activation", 8, count, 5, None, None, 0)


def test_routed_graph_api_rejects_invalid_count_before_runtime_import(monkeypatch):
    routed = __import__(
        "frontier.profiling.experimental.sglang.routed_moe_replay",
        fromlist=["profile_routed_graph"],
    )
    monkeypatch.setitem(sys.modules, "torch", None)
    query = {"component": "moe_sorting", "physical_size": 8, "physical_expert_counts": ()}
    with pytest.raises(ValueError, match="Invalid invocations"):
        routed.profile_routed_graph(query, 1, 5, None, None)
    query["physical_size"] = 8.0
    with pytest.raises(ValueError, match="Invalid sizes"):
        routed.profile_routed_graph(query, 2, 5, None, None)

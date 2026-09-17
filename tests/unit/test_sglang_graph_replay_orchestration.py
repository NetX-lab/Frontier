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
        ("moe_sorting", "make_moe_sorting_primitive", ({}, {})),
        ("moe_experts_quant_gemm_combine", "make_moe_experts_primitive", ({}, {})),
    ],
)
@pytest.mark.parametrize("trace", [False, True])
def test_delegated_builders_complete_capture_replay_and_trace(
    monkeypatch, name, builder_name, metadata, trace
):
    _exercise_replay(monkeypatch, name, builder_name, metadata, trace=trace)


def _exercise_replay(
    monkeypatch, name, builder_name, metadata, *, fault=None, trace=True,
    keyword_call=False,
):
    torch = pytest.importorskip("torch")
    active_graphs = []
    invocations = []
    events = []
    graphs = []
    buffers = []
    buffer_ids = {}
    replaying = False

    class Graph:
        def __init__(self):
            self.calls = []
            self.index = len(graphs)
            graphs.append(self)

        def replay(self):
            nonlocal replaying
            events.append(("replay_start", self.index))
            replaying = True
            for fn in self.calls:
                fn()
            replaying = False
            events.append(("replay_end", self.index))

    @contextmanager
    def capture(graph, **kwargs):
        events.append(("capture_start", graph.index))
        active_graphs.append(graph)
        yield
        active_graphs.pop()
        events.append(("capture_end", graph.index))

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            events.append(("record",))

        def synchronize(self):
            events.append(("event_synchronize",))

        def elapsed_time(self, other):
            return 3.0

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append(("synchronize",)))
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: events.append(("barrier",)))
    monkeypatch.setattr(torch.profiler, "record_function", lambda *args: nullcontext())

    original_copy = torch.Tensor.copy_
    original_check = torch.testing.assert_close

    def copy(tensor, source, *args, **kwargs):
        if id(tensor) in buffer_ids:
            events.append(("reset", buffer_ids[id(tensor)], source.tolist()))
        return original_copy(tensor, source, *args, **kwargs)

    def check(actual, expected, *args, **kwargs):
        events.append(("check", actual.tolist(), expected.tolist()))
        return original_check(actual, expected, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    monkeypatch.setattr(torch.testing, "assert_close", check)

    def builder(*args, **kwargs):
        output = torch.zeros(1)
        stateful = bool(metadata)
        index = len(buffers)
        buffer_ids[id(output)] = index
        buffers.append(output)

        def run():
            invocations.append(name)
            events.append(("call", index, output.tolist()))
            output.add_(1) if stateful else output.fill_(1)
            if fault == "eager" or (fault == "replay" and replaying):
                output.add_(1)
            return output

        def fn():
            if active_graphs:
                active_graphs[-1].calls.append(run)
            return run()

        reference = torch.ones(1)
        if stateful:
            return fn, reference, (output,), "test_backend", *metadata
        return fn, reference, "test_backend"

    routed_primitive = name in {"moe_sorting", "moe_experts_quant_gemm_combine"}
    if routed_primitive:
        from frontier.profiling.experimental.sglang import moe

        monkeypatch.setattr(moe, builder_name, builder)
    else:
        monkeypatch.setattr(replay, builder_name, builder)
    monkeypatch.setattr(replay, "dense_primitive_spec", lambda *args: {})
    group = SimpleNamespace(
        world_size=1, cpu_group=None, rank_in_group=0,
        graph_capture=lambda: nullcontext(SimpleNamespace(stream=None)),
    )
    if routed_primitive:
        from frontier.profiling.experimental.sglang.routed_moe_replay import profile_routed_graph

        row, trace_replay = profile_routed_graph(
            {"component": name, "physical_size": 8, "physical_expert_counts": (8, 0)},
            3, 5, SimpleNamespace(embedding_dim=16), group, trace=trace,
        )
        assert row["physical_expert_counts"] == [8, 0]
        assert row["moe_routed_spec"] == metadata[0]
    elif keyword_call:
        row, trace_replay = replay.profile_graph(
            name=name, size=8, count=3, repetitions=5,
            model=SimpleNamespace(embedding_dim=16), group=group, rank=0,
            logical_size=4, physical_context_lens=(32,) * 8, trace=trace,
        )
    else:
        row, trace_replay = replay.profile_graph(
            name, 8, 3, 5, SimpleNamespace(embedding_dim=16), group, 0,
            logical_size=4, physical_context_lens=(32,) * 8, trace=trace,
        )
    if trace:
        trace_replay()
        trace_replay()
    else:
        with pytest.raises(RuntimeError, match="without a trace probe"):
            trace_replay()

    assert row["backend"] == "test_backend"
    assert row["samples_ms"] == [1.0] * 5
    assert row["correctness_checked"] is True
    assert row["measurement_type"] == "HIP_GRAPH_REPLAY"
    assert row["kernel_trace_kind"] == ("representative_graph" if trace else None)
    assert row["kernel_trace_invocations"] == (2 if trace else 0)
    assert len(invocations) == (44 if trace else 38)
    if name == "gdn_core_decode":
        assert row["gdn_core_spec"] == metadata[0]
    if metadata:
        assert all(event[2] == [0.0] for event in events if event[0] == "call")
        assert sum(event[0] == "reset" for event in events) == len(invocations)
    return {"row": row, "events": events, "outputs": [value.tolist() for value in buffers]}


def test_gdn_replay_preserves_builder_metadata(monkeypatch):
    _exercise_replay(
        monkeypatch, "gdn_core_decode", "make_gdn_core_primitive",
        ({"conv_state_shape": (4, 3), "recurrent_state_shape": (2, 2, 2)},),
    )


def test_graph_api_accepts_explicit_named_arguments(monkeypatch):
    _exercise_replay(
        monkeypatch, "gdn_core_decode", "make_gdn_core_primitive", ({},),
        keyword_call=True,
    )


@pytest.mark.parametrize("fault", ["eager", "replay"])
def test_routed_replay_rejects_incorrect_outputs(monkeypatch, fault):
    with pytest.raises(AssertionError):
        _exercise_replay(
            monkeypatch, "moe_experts_quant_gemm_combine",
            "make_moe_experts_primitive", ({}, {}), fault=fault,
        )


def test_replay_requires_exact_integer_metadata():
    torch = pytest.importorskip("torch")
    expected = torch.tensor([96, 64], dtype=torch.int32)
    actual = torch.tensor([97, 64], dtype=torch.int32)
    call = replay._make_replay_call(
        lambda: actual, expected, (), "sorting_fixture", row_metadata={},
    )

    with pytest.raises(AssertionError):
        call.check(actual)


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

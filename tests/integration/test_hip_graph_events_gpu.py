"""Explicit opt-in GPU test; never reserve GPUs during ordinary CPU tests."""

import os

import pytest


@pytest.mark.skipif(os.environ.get("FRONTIER_RUN_HIP_GRAPH_TEST") != "1",
                    reason="Set FRONTIER_RUN_HIP_GRAPH_TEST=1 on an idle ROCm GPU")
def test_hip_graph_event_nodes_preserve_results_and_replay_timestamps():
    torch = pytest.importorskip("torch")
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from frontier.validation.hip_graph_events import HipGraphEvents

    api = HipGraphEvents()
    stream = torch.cuda.Stream()
    tensor = torch.ones(1024, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            tensor.add_(1)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        start, end = api.create_event(), api.create_event()
        start.record(stream)
        for _ in range(100):
            tensor.add_(1)
        end.record(stream)
    tensor.zero_()
    outer_start, outer_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for repetition in range(3):
        outer_start.record()
        graph.replay()
        outer_end.record()
        torch.cuda.synchronize()
        assert bool((tensor == 100 * (repetition + 1)).all())
        assert 0 < start.elapsed_time(end) <= outer_start.elapsed_time(outer_end)
    del graph
    api.close_after_graphs_destroyed()


@pytest.mark.skipif(os.environ.get("FRONTIER_RUN_HIP_GRAPH_TEST") != "1",
                    reason="Set FRONTIER_RUN_HIP_GRAPH_TEST=1 on an idle ROCm GPU")
def test_routing_buffers_follow_gpu_graph_replay_without_modifying_outputs():
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from frontier.validation.sglang_routing import RoutingGraphObserver

    class Qwen3_5LinearDecoderLayer:
        def __init__(self):
            self.mlp = SimpleNamespace(num_experts=4, num_fused_shared_experts=0,
                shared_expert=object(), topk=SimpleNamespace(forward=self.route,
                    topk_config=SimpleNamespace(top_k=2, num_fused_shared_experts=0)))

        def route(self, scores):
            values, ids = torch.topk(scores, 2, dim=-1)
            return SimpleNamespace(topk_ids=ids, topk_weights=torch.softmax(values, dim=-1))

        def forward(self, scores, forward_batch=None):
            return self.mlp.topk.forward(scores)

    layer = Qwen3_5LinearDecoderLayer()
    model = SimpleNamespace(named_modules=lambda: [("layers.0", layer)])
    batch = SimpleNamespace(batch_size=3, forward_mode=SimpleNamespace(is_decode=lambda: True))
    scores = torch.tensor([[4., 3., 2., 1.], [1., 2., 4., 3.], [1., 2., 3., 4.]], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            layer.forward(scores)
    stream.synchronize()
    observer = RoutingGraphObserver(model)
    graph = torch.cuda.CUDAGraph()
    with observer, torch.cuda.graph(graph, stream=stream):
        output = layer.forward(scores, forward_batch=batch)
    observer.validate_captures()
    graph.replay()
    torch.cuda.synchronize()
    expected = layer.route(scores)
    assert torch.equal(output.topk_ids, expected.topk_ids)
    assert torch.equal(output.topk_weights, expected.topk_weights)
    saved = observer.snapshot_graph(physical_size=3, logical_size=2, batch_id="b", rank=0)
    scores.neg_()
    graph.replay()
    torch.cuda.synchronize()
    later = observer.collect_graph(physical_size=3, logical_size=2, batch_id="c", rank=0)[0]
    earlier = observer.summarize_snapshot(saved)[0]
    assert earlier.logical_assignment_sha256 != later.logical_assignment_sha256
    assert earlier.logical_expert_counts == (1, 1, 1, 1)
    assert earlier.padding_expert_counts == (0, 0, 1, 1)

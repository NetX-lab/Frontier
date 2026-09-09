from types import SimpleNamespace

import numpy as np
import pytest

from frontier.validation.sglang_routing import RoutingGraphObserver
from tests.unit.test_sglang_operator_events import Model
from tests.unit.test_sglang_graph_events import API, batch


def setup():
    model, api = Model(), API()
    mlp = model.layer.mlp
    mlp.num_experts, mlp.num_fused_shared_experts = 4, 0
    mlp.topk.topk_config = SimpleNamespace(top_k=2, num_fused_shared_experts=0)
    result = SimpleNamespace(topk_ids=np.array([[0, 1], [1, 2], [0, 0]], dtype=np.int32),
                             topk_weights=np.array([[.5, .5], [.4, .6], [0., 0.]], dtype=np.float32))
    mlp.topk.forward = lambda value: result
    model.layer.forward = lambda value, forward_batch=None: mlp.topk(value)
    copies = []
    def copy(buffers):
        copies.append(True)
        return np.stack([b[0] for b in buffers]), np.stack([b[1] for b in buffers])
    observer = RoutingGraphObserver(model, api=api, current_stream=lambda: 0, copy_to_cpu=copy)
    return model, api, observer, result, copies


def test_capture_retains_original_outputs_and_copies_only_after_replay():
    model, api, observer, result, copies = setup()
    original = model.layer.forward
    with observer:
        assert model.layer(1) is result
        assert not observer.captures
        api.capture_id = 1
        assert model.layer(1, forward_batch=batch(3)) is result
        assert observer.captures[1]["outputs"][0][0] is result.topk_ids
        assert not copies
    assert model.layer.forward is original
    observer.validate_captures()
    rows = observer.collect_graph(physical_size=3, logical_size=2, batch_id="b", rank=0)
    assert len(copies) == 1 and rows[0].logical_expert_counts == (1, 2, 1, 0)
    result.topk_ids[0, :] = [2, 3]  # emulate a subsequent graph replay updating the same buffer
    assert observer.collect_graph(physical_size=3, logical_size=2, batch_id="c", rank=0)[0].logical_expert_counts == (0, 1, 2, 1)


def test_missing_shapes_variants_duplicate_calls_and_partial_layers_fail():
    model, api, observer, result, _ = setup()
    with pytest.raises(ValueError, match="complete routing"):
        observer.validate_captures()
    with observer:
        api.capture_id = 1
        model.layer(1, forward_batch=batch(3))
        with pytest.raises(ValueError, match="exactly once"):
            model.layer(1, forward_batch=batch(3))
        api.capture_id = 2
        with pytest.raises(ValueError, match="variant"):
            model.layer(1, forward_batch=batch(3))
    observer.captures.pop(2)
    with pytest.raises(ValueError, match="physical size"):
        observer.collect_graph(physical_size=4, logical_size=2, batch_id="b", rank=0)
    observer.captures[1]["outputs"].clear()
    with pytest.raises(ValueError, match="missing selected"):
        observer.validate_captures()


def test_unsupported_fused_shared_experts_are_rejected_before_mutation():
    model, api, _, _, _ = setup()
    original = model.layer.forward
    model.layer.mlp.num_fused_shared_experts = 1
    with pytest.raises(ValueError, match="separate shared"):
        RoutingGraphObserver(model, api=api, current_stream=lambda: 0)
    assert model.layer.forward is original


def test_cpu_histograms_can_wait_until_after_the_repetition():
    model, api, observer, result, _ = setup()
    with observer:
        api.capture_id = 1
        model.layer(1, forward_batch=batch(3))
    snapshot = observer.snapshot_graph(physical_size=3, logical_size=2, batch_id="b", rank=0)
    result.topk_ids[0, :] = [2, 3]
    rows = observer.summarize_snapshot(snapshot)
    assert rows[0].logical_expert_counts == (1, 2, 1, 0)  # immutable CPU copy of earlier replay

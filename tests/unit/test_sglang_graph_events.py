from types import SimpleNamespace

import pytest

from frontier.validation.sglang_graph_events import (
    GraphDecoderEventObserver, GraphOperatorEventObserver,
)
from tests.unit.test_sglang_operator_events import Event, Model


class API:
    capture_id = None
    def capture_info(self, stream):
        return None if self.capture_id is None else (self.capture_id,)
    def create_event(self):
        return Event()


def setup():
    model, api = Model(), API()
    original = model.layer.forward
    model.layer.forward = lambda value, forward_batch=None: original(value)
    observer = GraphOperatorEventObserver(model, [0], api=api, current_stream=lambda: 0)
    return model, api, observer


def batch(size):
    return SimpleNamespace(batch_size=size, forward_mode=SimpleNamespace(is_decode=lambda: True))


def test_only_capture_is_instrumented_and_handles_survive_repeated_collection():
    model, api, observer = setup()
    expected = model.layer(1)
    with observer:
        assert model.layer(1) == expected
        assert observer.captures == {}
        for capture_id, size in ((10, 16), (11, 24)):
            api.capture_id = capture_id
            assert model.layer(1, forward_batch=batch(size)) == expected
    observer.validate_captures()
    first = observer.collect_graph(24)
    assert len(first) == 4  # coarse disjoint scopes; untimed ownership markers are omitted
    assert first == observer.collect_graph(24)
    assert {r["physical_size"] for r in first} == {24}
    assert {r["measurement_type"] for r in first} == {"HIP_GRAPH_EVENT"}
    assert observer.collect_graph(16)[0]["capture_id"] == 10
    with pytest.raises(ValueError, match="No instrumented"):
        observer.collect_graph(20)  # logical batch must resolve its physical graph first


def test_variants_missing_scopes_and_non_decode_fail():
    model, api, observer = setup()
    with pytest.raises(ValueError, match="No full"):
        observer.validate_captures()
    with observer:
        api.capture_id = 1
        with pytest.raises(ValueError, match="decode"):
            model.layer(1)
        model.layer(1, forward_batch=batch(16))
        api.capture_id = 2
        with pytest.raises(ValueError, match="variants"):
            model.layer(1, forward_batch=batch(16))
    observer.captures[1]["samples"].pop()
    with pytest.raises(ValueError, match="every selected"):
        observer.validate_captures()


def test_nested_timers_are_rejected_and_leaf_timers_are_admitted():
    for components, accepted in ((["gdn", "gdn_attn"], False),
                                 (["gdn_attn", "gdn_norm", "moe_experts"], True)):
        model, api, _ = setup()
        observer = GraphOperatorEventObserver(model, [0], api=api, current_stream=lambda: 0,
                                               components=components)
        with observer:
            api.capture_id = 1
            model.layer(1, forward_batch=batch(16))
        if accepted:
            observer.validate_captures()
            assert {r["component"] for r in observer.collect_graph(16)} == set(components)
        else:
            with pytest.raises(ValueError, match="Nested"):
                observer.validate_captures()


def test_whole_decoder_uses_one_event_pair_and_restores_layer():
    model, api, _ = setup()
    original = model.layer.forward
    expected = model.layer(1)
    observer = GraphDecoderEventObserver(model, api=api, current_stream=lambda: 0)
    with observer:
        assert model.layer(1, forward_batch=batch(16)) == expected
        api.capture_id = 10
        assert model.layer(1, forward_batch=batch(16)) == expected
    observer.validate_captures()
    row = observer.collect_graph(16)
    assert row == {
        "component": "decoder",
        "first_layer_id": 0,
        "last_layer_id": 0,
        "measurement_type": "HIP_GRAPH_EVENT",
        "capture_id": 10,
        "physical_size": 16,
        "inclusive_gpu_ms": 1.0,
    }
    assert model.layer.forward is original
    assert observer.collect_graph(16) == row


def test_whole_decoder_rejects_missing_and_duplicate_graph_variants():
    model, api, _ = setup()
    observer = GraphDecoderEventObserver(model, api=api, current_stream=lambda: 0)
    with pytest.raises(ValueError, match="No full decoder"):
        observer.validate_captures()
    with observer:
        api.capture_id = 1
        model.layer(1, forward_batch=batch(16))
        with pytest.raises(ValueError, match="more than once"):
            model.layer(1, forward_batch=batch(16))
        api.capture_id = 2
        with pytest.raises(ValueError, match="variants"):
            model.layer(1, forward_batch=batch(16))

from types import SimpleNamespace

import pytest

from frontier.validation.sglang_operator_events import OperatorEventObserver


class Leaf:
    def forward(self, value):
        return value + 1

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class GDN(Leaf):
    def __init__(self):
        self.attn, self.norm, self.out_proj = Leaf(), Leaf(), Leaf()

    def _forward_input_proj(self, value):
        return value + 1

    def forward(self, value):
        return self.out_proj(self.norm(self.attn(self._forward_input_proj(value))))


class MLP(Leaf):
    def __init__(self):
        self.shared_expert, self.gate, self.topk, self.experts = Leaf(), Leaf(), Leaf(), Leaf()

    def forward(self, value):
        return self.shared_expert(value) + self.experts(self.topk(self.gate(value)))


class Qwen3_5LinearDecoderLayer(Leaf):
    def __init__(self):
        self.input_layernorm, self.post_attention_layernorm = Leaf(), Leaf()
        self.layer_communicator = SimpleNamespace(prepare_mlp=self.post_attention_layernorm)
        self.linear_attn, self.mlp = GDN(), MLP()

    def forward(self, value):
        value = self.linear_attn(self.input_layernorm(value))
        return self.mlp(self.layer_communicator.prepare_mlp(value))


class Model:
    def __init__(self):
        self.layer = Qwen3_5LinearDecoderLayer()

    def named_modules(self):
        return [("model.layers.0", self.layer)]


class Event:
    clock = 0

    def record(self, stream):
        self.time = Event.clock
        Event.clock += 1

    def elapsed_time(self, other):
        return other.time - self.time


def observer(model, ids=None):
    return OperatorEventObserver(model, [0] if ids is None else ids,
                                 event_factory=Event, current_stream=lambda: 0)


def test_events_preserve_outputs_and_parent_tree_and_restore_methods():
    model = Model()
    expected = model.layer(7)
    assert "forward" not in vars(model.layer)
    with observer(model) as scopes:
        assert model.layer(7) == expected
        rows = scopes.collect()
        assert scopes.collect() == []
        assert rows[0]["component"] == "decoder_layer"
        assert rows[0]["parent_id"] is None
        assert all(r["inclusive_gpu_ms"] > 0 for r in rows)
        gdn = next(r for r in rows if r["component"] == "gdn")
        proj = next(r for r in rows if r["component"] == "gdn_input_projections")
        assert proj["parent_id"] == gdn["sample_id"]
        assert gdn["inclusive_gpu_ms"] > proj["inclusive_gpu_ms"]
        assert model.layer(7) == expected
        assert scopes.collect()[0]["sample_id"] == 0
    assert "forward" not in vars(model.layer)
    assert model.layer(7) == expected


def test_exceptions_restore_instance_overrides_and_model_methods():
    model = Model()
    def fail(value):
        raise RuntimeError("operator failed")
    model.layer.mlp.forward = fail
    with pytest.raises(RuntimeError, match="operator failed"):
        with observer(model):
            model.layer(7)
    assert "forward" not in vars(model.layer)
    assert model.layer.mlp.forward is fail


@pytest.mark.parametrize("ids", [[], [1], [0, 0], [-1]])
def test_invalid_layers_never_install_partial_wrappers(ids):
    model = Model()
    with pytest.raises(ValueError, match="layer"):
        observer(model, ids)
    assert "forward" not in vars(model.layer)


def test_reject_shared_expert_fusion_before_installing_any_wrappers():
    model = Model()
    model.layer.mlp.shared_expert = None
    with pytest.raises(ValueError, match="separate shared"):
        observer(model)
    assert "forward" not in vars(model.layer)


def test_shared_modules_use_active_layer_and_ignore_unselected_calls():
    first, second = Qwen3_5LinearDecoderLayer(), Qwen3_5LinearDecoderLayer()
    shared = Leaf()
    first.input_layernorm = second.input_layernorm = shared
    model = SimpleNamespace(named_modules=lambda: [("layers.0", first), ("layers.1", second)])
    with observer(model, [0, 1]) as scopes:
        first(0)
        shared(0)  # a call outside either selected decoder must not be counted
        second(0)
        rows = scopes.collect()
    norms = [r for r in rows if r["component"] == "input_layernorm"]
    assert [r["layer_id"] for r in norms] == [0, 1]
    assert all(r["parent_id"] is not None for r in norms)
    assert "forward" not in vars(shared)


def test_missing_scope_calls_fail_coverage_check():
    model = Model()
    model.layer.mlp.forward = lambda value: value
    with observer(model) as scopes:
        model.layer(0)
        with pytest.raises(ValueError, match="exactly once"):
            scopes.collect()

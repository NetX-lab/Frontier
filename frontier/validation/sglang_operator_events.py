"""Opt-in eager operator GPU-event measurements on the loaded SGLang model.

Scopes are nested/inclusive, not additive. Install only for separate diagnostic
passes, after ordinary benchmark repetitions and graph capture. No weights,
tensors, model outputs or persistent SGLang source files are modified.
"""

from __future__ import annotations

from functools import wraps
import math


def qwen_decoder_layers(model, layer_ids=None):
    """Resolve selected runtime layers once, shared by timing/routing observers."""
    layers = {}
    for name, module in model.named_modules():
        if type(module).__name__ in {"Qwen3_5LinearDecoderLayer", "Qwen3_5AttentionDecoderLayer"}:
            index = int(name.rsplit(".", 1)[-1])
            if index in layers:
                raise ValueError(f"Duplicate decoder layer {index}")
            layers[index] = module
    layer_ids = sorted(layers) if layer_ids is None else list(layer_ids)
    if (not layer_ids or len(set(layer_ids)) != len(layer_ids) or set(layer_ids) - set(layers)
            or any(type(i) is not int or i < 0 for i in layer_ids)):
        raise ValueError("Requested unique operator-event layers must exist in the Qwen3.5 model")
    return {i: layers[i] for i in layer_ids}


class OperatorEventObserver:
    def __init__(self, model, layer_ids: list[int], *, event_factory=None, current_stream=None):
        if event_factory is None:
            import torch
            event_factory = lambda: torch.cuda.Event(enable_timing=True)
            current_stream = torch.cuda.current_stream
        self.event_factory = event_factory
        self.current_stream = current_stream
        self._restores = []
        self._samples = []
        self._stack = []
        self._plans = []
        for layer_id, layer in qwen_decoder_layers(model, layer_ids).items():
            self._plan(layer, "forward", layer_id, "decoder_layer")
            for name in ("input_layernorm", "post_attention_layernorm"):
                self._plan(getattr(layer, name), "forward", layer_id, name)
            if hasattr(layer, "linear_attn"):
                self._plan(layer.linear_attn, "forward", layer_id, "gdn")
                self._plan(layer.linear_attn, "_forward_input_proj", layer_id, "gdn_input_projections")
                for name in ("attn", "norm", "out_proj"):
                    self._plan(getattr(layer.linear_attn, name), "forward", layer_id, f"gdn_{name}")
            else:
                self._plan(layer, "self_attention", layer_id, "full_attention")
                for name in ("qkv_proj", "rotary_emb", "attn", "o_proj"):
                    self._plan(getattr(layer, name), "forward", layer_id, f"attention_{name}")
            self._plan(layer.layer_communicator, "prepare_mlp", layer_id, "attention_reduce_and_mlp_norm")
            self._plan(layer.mlp, "forward", layer_id, "mlp_including_tp_reduce")
            if layer.mlp.shared_expert is None:
                raise ValueError("Operator-event contract requires a separate shared expert")
            for name in ("shared_expert", "gate", "topk", "experts"):
                self._plan(getattr(layer.mlp, name), "forward", layer_id, f"moe_{name}")

    def _plan(self, owner, attribute, layer, component):
        original = getattr(owner, attribute)
        if not callable(original):
            raise ValueError(f"Missing callable {component}.{attribute}")
        self._plans.append((owner, attribute, layer, component, original))

    def __enter__(self):
        # Rotary modules can be cached/shared across decoder layers. Wrap an
        # object once and resolve ownership from the active decoder scope,
        # otherwise calls from unselected layers would be mislabeled.
        grouped = {}
        for owner, attribute, layer, component, original in self._plans:
            key = (id(owner), attribute)
            if key not in grouped:
                grouped[key] = (owner, attribute, original, {})
            grouped[key][3][layer] = component
        for owner, attribute, original, identities in grouped.values():
            self._restores.append((owner, attribute, attribute in vars(owner), original))
            setattr(owner, attribute, self._wrap(original, identities))
        return self

    def _wrap(self, original, identities):
        @wraps(original)
        def measured(*args, **kwargs):
            if "decoder_layer" in identities.values():
                if len(identities) != 1 or self._stack:
                    raise ValueError("Decoder scopes must be distinct and non-nested")
                layer = next(iter(identities))
            else:
                if not self._stack:
                    return original(*args, **kwargs)
                layer = self._samples[self._stack[0]]["layer_id"]
                if layer not in identities:
                    return original(*args, **kwargs)
            component = identities[layer]
            stream = self.current_stream()
            start, end = self.event_factory(), self.event_factory()
            sample_id = len(self._samples)
            sample = {"sample_id": sample_id, "parent_id": self._stack[-1] if self._stack else None,
                      "layer_id": layer, "component": component, "start": start, "end": end}
            self._samples.append(sample)
            self._stack.append(sample_id)
            start.record(stream)
            try:
                return original(*args, **kwargs)
            finally:
                end.record(stream)
                self._stack.pop()
                if self.current_stream() != stream:
                    raise ValueError("Operator scope changed its current stream")
        return measured

    def collect(self) -> list[dict]:
        """Call only after the normal end-of-step device synchronization."""
        if self._stack:
            raise ValueError("Cannot collect inside an unfinished scope")
        if self._samples:
            expected = {(layer, component) for _, _, layer, component, _ in self._plans}
            observed = [(sample["layer_id"], sample["component"]) for sample in self._samples]
            if len(observed) != len(expected) or set(observed) != expected:
                raise ValueError("Operator scopes must execute exactly once per selected layer")
        result = []
        for sample in self._samples:
            duration = float(sample["start"].elapsed_time(sample["end"]))
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Invalid operator GPU-event timing")
            result.append({k: v for k, v in sample.items() if k not in {"start", "end"}} |
                          {"inclusive_gpu_ms": duration, "measurement_type": "CUDA_EVENT"})
        self._samples.clear()
        return result

    def __exit__(self, *exc):
        for owner, attribute, was_local, original in reversed(self._restores):
            if was_local:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        self._restores.clear()

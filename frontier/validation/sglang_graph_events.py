"""Capture-time scope instrumentation for SGLang full decode HIP graphs."""

from __future__ import annotations

from functools import wraps
import math

from frontier.validation.sglang_operator_events import OperatorEventObserver, qwen_decoder_layers


DEFAULT_COMPONENTS = ("input_layernorm", "gdn", "full_attention",
                      "attention_reduce_and_mlp_norm", "mlp_including_tp_reduce")


class _UntimedBoundary:
    def record(self, stream):
        pass
    def elapsed_time(self, other):
        return 0.0


class GraphOperatorEventObserver(OperatorEventObserver):
    def __init__(self, model, layer_ids, *, components=DEFAULT_COMPONENTS, api=None, current_stream=None):
        if api is None:
            import torch
            from frontier.validation.hip_graph_events import HipGraphEvents
            if not torch.version.hip:
                raise ValueError("HIP graph events require ROCm PyTorch")
            api, current_stream = HipGraphEvents(), torch.cuda.current_stream
        self.api = api
        components = tuple(components)
        self.components = set(components)
        self._current_component = None
        self.captures = {}
        self.by_size = {}
        self._validated = False
        super().__init__(model, layer_ids, event_factory=self._event, current_stream=current_stream)
        available = {component for _, _, _, component, _ in self._plans}
        # The default spans both mixer kinds; a model selecting only one kind
        # need not expose the other. Explicit unknown names fail admission.
        missing = self.components - available
        if (not self.components or len(self.components) != len(components)
                or (missing and components != DEFAULT_COMPONENTS)):
            raise ValueError(f"Unknown/empty graph timing components: {sorted(missing)}")

    def _event(self):
        return (self.api.create_event() if self._current_component in self.components
                else _UntimedBoundary())

    def _wrap(self, original, identities):
        measured = super()._wrap(original, identities)

        @wraps(original)
        def capture_only(*args, **kwargs):
            info = self.api.capture_info(self.current_stream())
            if info is None:
                return original(*args, **kwargs)  # no instrumentation in eager warmup
            capture_id = info[0]
            self._validated = False
            if "decoder_layer" in identities.values():
                batch = kwargs.get("forward_batch", args[3] if len(args) > 3 else None)
                if batch is None or not batch.forward_mode.is_decode():
                    raise ValueError("Graph operator events require non-speculative decode")
                size = int(batch.batch_size)
                if capture_id not in self.captures:
                    self.captures[capture_id] = {"physical_size": size, "samples": []}
                capture = self.captures[capture_id]
                if capture["physical_size"] != size:
                    raise ValueError("A captured graph cannot change physical batch size")
                self._samples = capture["samples"]
                if size in self.by_size and self.by_size[size] != capture_id:
                    raise ValueError("Multiple graph variants for one physical size are unsupported")
                self.by_size[size] = capture_id
                layer = next(iter(identities))
            elif self._stack:
                layer = self._samples[self._stack[0]]["layer_id"]
            else:
                return original(*args, **kwargs)
            self._current_component = identities.get(layer)
            return measured(*args, **kwargs)
        return capture_only

    def collect_graph(self, physical_size):
        if not self._validated:
            self.validate_captures()
        if physical_size not in self.by_size:
            raise ValueError(f"No instrumented graph for physical batch {physical_size}")
        capture_id = self.by_size[physical_size]
        # Parent collection validates coverage and clears its working list.
        # Keep the captured handles alive and reusable across graph replays.
        self._samples = list(self.captures[capture_id]["samples"])
        rows = super().collect()
        return [{**row, "parent_id": None, "measurement_type": "HIP_GRAPH_EVENT",
                 "capture_id": capture_id, "physical_size": physical_size}
                for row in rows if row["component"] in self.components]

    def validate_captures(self):
        if not self.captures:
            raise ValueError("No full decode graphs were instrumented")
        expected = {(layer, component) for _, _, layer, component, _ in self._plans}
        for capture in self.captures.values():
            actual = [(row["layer_id"], row["component"]) for row in capture["samples"]]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("Each graph must capture every selected scope exactly once")
            by_id = {row["sample_id"]: row for row in capture["samples"]}
            for row in capture["samples"]:
                if row["component"] not in self.components:
                    continue
                parent = row["parent_id"]
                while parent is not None:
                    ancestor = by_id[parent]
                    if ancestor["component"] in self.components:
                        raise ValueError("Nested graph timers bias parent spans; select disjoint scopes")
                    parent = ancestor["parent_id"]
        self._validated = True


class GraphDecoderEventObserver:
    """Place only two event nodes around the complete decoder-layer sequence."""

    def __init__(self, model, *, api=None, current_stream=None):
        if api is None:
            import torch
            from frontier.validation.hip_graph_events import HipGraphEvents
            if not torch.version.hip:
                raise ValueError("HIP graph events require ROCm PyTorch")
            api, current_stream = HipGraphEvents(), torch.cuda.current_stream
        layers = qwen_decoder_layers(model)
        self.api, self.current_stream = api, current_stream
        self.first_layer_id, self.last_layer_id = min(layers), max(layers)
        self.first, self.last = layers[self.first_layer_id], layers[self.last_layer_id]
        self.captures, self.by_size, self._restores = {}, {}, []
        self._validated = False

    @staticmethod
    def _batch(args, kwargs):
        batch = kwargs.get("forward_batch", args[3] if len(args) > 3 else None)
        if batch is None or not batch.forward_mode.is_decode():
            raise ValueError("Decoder graph events require non-speculative decode")
        return batch

    def _capture(self, args, kwargs):
        info = self.api.capture_info(self.current_stream())
        if info is None:
            return None
        batch = self._batch(args, kwargs)
        capture_id, physical_size = info[0], int(batch.batch_size)
        if physical_size in self.by_size and self.by_size[physical_size] != capture_id:
            raise ValueError("Multiple graph variants for one physical size are unsupported")
        self.by_size[physical_size] = capture_id
        capture = self.captures.setdefault(capture_id, {
            "physical_size": physical_size, "start": None, "end": None})
        if capture["physical_size"] != physical_size:
            raise ValueError("A captured graph cannot change physical batch size")
        self._validated = False
        return capture_id, capture

    def _wrap_first(self, original):
        @wraps(original)
        def first(*args, **kwargs):
            resolved = self._capture(args, kwargs)
            if resolved is None:
                return original(*args, **kwargs)
            _, capture = resolved
            if capture["start"] is not None:
                raise ValueError("Decoder graph start was captured more than once")
            capture["start"] = self.api.create_event()
            capture["start"].record(self.current_stream())
            return original(*args, **kwargs)
        return first

    def _wrap_last(self, original):
        @wraps(original)
        def last(*args, **kwargs):
            resolved = self._capture(args, kwargs)
            if resolved is None:
                return original(*args, **kwargs)
            _, capture = resolved
            if capture["start"] is None or capture["end"] is not None:
                raise ValueError("Decoder graph end has no unique matching start")
            result = original(*args, **kwargs)
            capture["end"] = self.api.create_event()
            capture["end"].record(self.current_stream())
            return result
        return last

    def _wrap_single(self, original):
        @wraps(original)
        def single(*args, **kwargs):
            resolved = self._capture(args, kwargs)
            if resolved is None:
                return original(*args, **kwargs)
            _, capture = resolved
            if capture["start"] is not None or capture["end"] is not None:
                raise ValueError("Decoder graph was captured more than once")
            capture["start"] = self.api.create_event()
            capture["start"].record(self.current_stream())
            result = original(*args, **kwargs)
            capture["end"] = self.api.create_event()
            capture["end"].record(self.current_stream())
            return result
        return single

    def __enter__(self):
        owners = (self.first,) if self.first is self.last else (self.first, self.last)
        for owner in owners:
            original = owner.forward
            self._restores.append((owner, "forward" in vars(owner), original))
        if self.first is self.last:
            self.first.forward = self._wrap_single(self.first.forward)
        else:
            self.first.forward = self._wrap_first(self.first.forward)
            self.last.forward = self._wrap_last(self.last.forward)
        return self

    def validate_captures(self):
        if not self.captures:
            raise ValueError("No full decoder graphs were instrumented")
        if any(capture["start"] is None or capture["end"] is None
               for capture in self.captures.values()):
            raise ValueError("Every decoder graph requires exactly one start and end")
        self._validated = True

    def collect_graph(self, physical_size):
        if not self._validated:
            self.validate_captures()
        if physical_size not in self.by_size:
            raise ValueError(f"No instrumented decoder graph for physical batch {physical_size}")
        capture_id = self.by_size[physical_size]
        capture = self.captures[capture_id]
        duration = float(capture["start"].elapsed_time(capture["end"]))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid decoder graph event duration")
        return {
            "component": "decoder",
            "first_layer_id": self.first_layer_id,
            "last_layer_id": self.last_layer_id,
            "measurement_type": "HIP_GRAPH_EVENT",
            "capture_id": capture_id,
            "physical_size": physical_size,
            "inclusive_gpu_ms": duration,
        }

    def __exit__(self, *exc):
        for owner, was_local, original in reversed(self._restores):
            if was_local:
                owner.forward = original
            else:
                delattr(owner, "forward")
        self._restores.clear()

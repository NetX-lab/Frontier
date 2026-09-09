"""Retain graph-produced top-k buffers and read them only after measured forward.

No GPU operation is added during capture. Retaining outputs can alter graph
pool allocation, so routing passes remain separate, quality-checked diagnostics.
"""

from functools import wraps

from frontier.validation.routing import snapshot_routing
from frontier.validation.sglang_operator_events import qwen_decoder_layers


class RoutingGraphObserver:
    def __init__(self, model, layer_ids=None, *, api=None, current_stream=None, copy_to_cpu=None):
        if api is None:
            import torch
            from frontier.validation.hip_graph_events import HipGraphEvents
            if not torch.version.hip:
                raise ValueError("Routing graph observation requires ROCm")
            api, current_stream = HipGraphEvents(), torch.cuda.current_stream
        self.api, self.current_stream = api, current_stream
        self.copy_to_cpu = copy_to_cpu or self._copy_to_cpu
        self.layers = qwen_decoder_layers(model, layer_ids)
        self.captures, self.by_size, self._restores = {}, {}, []
        self._active = None
        self._validated = False
        topk_owners = set()
        for layer in self.layers.values():
            if (layer.mlp.shared_expert is None or layer.mlp.num_fused_shared_experts != 0
                    or getattr(layer.mlp, "alt_stream", None) is not None
                    or layer.mlp.topk.topk_config.num_fused_shared_experts != 0):
                raise ValueError("Routing contract requires separate shared experts and a single stream")
            if not callable(layer.forward) or not callable(layer.mlp.topk.forward):
                raise ValueError("Routing layers require callable forward/top-k boundaries")
            if id(layer.mlp.topk) in topk_owners:
                raise ValueError("Routing top-k modules must have distinct layer ownership")
            topk_owners.add(id(layer.mlp.topk))

    @staticmethod
    def _copy_to_cpu(buffers):
        import torch
        # Stacking/copies occur AFTER outer GPU and wall timings are finalized.
        # No hooks, clones, counters or reductions are inserted in the graph.
        ids = torch.stack([pair[0] for pair in buffers]).detach().cpu().numpy()
        weights = torch.stack([pair[1] for pair in buffers]).detach().cpu().numpy()
        return ids, weights

    def _install(self, owner, attribute, wrapper):
        original = getattr(owner, attribute)
        self._restores.append((owner, attribute, attribute in vars(owner), original))
        setattr(owner, attribute, wrapper(original))

    def __enter__(self):
        for layer_id, layer in self.layers.items():
            self._install(layer, "forward", lambda original, layer_id=layer_id: self._root(original, layer_id))
            self._install(layer.mlp.topk, "forward", lambda original, layer_id=layer_id: self._topk(original, layer_id))
        return self

    def _root(self, original, layer_id):
        @wraps(original)
        def capture(*args, **kwargs):
            info = self.api.capture_info(self.current_stream())
            if info is None:
                return original(*args, **kwargs)
            batch = kwargs.get("forward_batch", args[3] if len(args) > 3 else None)
            if self._active is not None or batch is None or not batch.forward_mode.is_decode():
                raise ValueError("Routing graph requires distinct non-speculative decode layer scopes")
            capture_id, physical = info[0], int(batch.batch_size)
            state = self.captures.setdefault(capture_id, {"physical_size": physical, "outputs": {}})
            if (state["physical_size"] != physical
                    or self.by_size.setdefault(physical, capture_id) != capture_id):
                raise ValueError("Routing graph physical size/variant mismatch")
            self._validated = False
            self._active = capture_id, layer_id, self.current_stream()
            try:
                return original(*args, **kwargs)
            finally:
                self._active = None
        return capture

    def _topk(self, original, layer_id):
        @wraps(original)
        def capture(*args, **kwargs):
            output = original(*args, **kwargs)
            if self._active is None:
                return output
            capture_id, active_layer, stream = self._active
            if active_layer != layer_id or self.current_stream() != stream:
                raise ValueError("Routing module ownership or stream mismatch")
            state = self.captures[capture_id]
            if layer_id in state["outputs"]:
                raise ValueError("Each layer must emit top-k exactly once per graph")
            ids, weights = getattr(output, "topk_ids", None), getattr(output, "topk_weights", None)
            shape = (state["physical_size"], self.layers[layer_id].mlp.topk.topk_config.top_k)
            if ids is None or weights is None or tuple(ids.shape) != shape or tuple(weights.shape) != shape:
                raise ValueError("Requires standard top-k IDs/weights with the physical graph shape")
            # Retain original references only; never modify the runtime outputs.
            state["outputs"][layer_id] = (ids, weights)
            return output
        return capture

    def validate_captures(self):
        if self._active is not None or not self.captures:
            raise ValueError("No complete routing graphs")
        for state in self.captures.values():
            if set(state["outputs"]) != set(self.layers):
                raise ValueError("Routing graph is missing selected layers")
        self._validated = True

    def snapshot_graph(self, *, physical_size, logical_size, batch_id, rank):
        if not self._validated:
            self.validate_captures()
        if physical_size not in self.by_size:
            raise ValueError("No routing graph for physical size")
        capture_id = self.by_size[physical_size]
        state = self.captures[capture_id]
        ids, weights = self.copy_to_cpu([state["outputs"][i] for i in self.layers])
        return dict(ids=ids, weights=weights, batch_id=batch_id, rank=rank,
                    capture_id=capture_id, logical_size=logical_size)

    def summarize_snapshot(self, snapshot):
        result = []
        for index, (layer_id, layer) in enumerate(self.layers.items()):
            result.append(snapshot_routing(snapshot["ids"][index], snapshot["weights"][index],
                batch_id=snapshot["batch_id"], rank=snapshot["rank"],
                layer_id=layer_id, capture_id=snapshot["capture_id"], logical_size=snapshot["logical_size"],
                num_experts=layer.mlp.num_experts, top_k=layer.mlp.topk.topk_config.top_k))
        return result

    def collect_graph(self, **kwargs):
        return self.summarize_snapshot(self.snapshot_graph(**kwargs))

    def __exit__(self, *exc):
        for owner, attribute, was_local, original in reversed(self._restores):
            if was_local:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        self._restores.clear()

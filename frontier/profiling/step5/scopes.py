"""Observe native call boundaries without replacing any attention computation."""

from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch


class NativeScopes:
    """Outermost native calls own fused work and its CUDA-event duration once."""

    def __init__(self, torch):
        self.torch = torch
        self.enabled = False
        self.active = None
        self.events = {}
        self.graph_capture = False
        self.record_events = True

    @contextmanager
    def scope(self, name):
        if not self.enabled or self.active is not None:
            yield
            return
        self.active = name
        start = self.torch.cuda.Event(enable_timing=True, external=self.graph_capture) if self.record_events else None
        end = self.torch.cuda.Event(enable_timing=True, external=self.graph_capture) if self.record_events else None
        try:
            with self.torch.profiler.record_function(name):
                if start is not None:
                    start.record()
                yield
                if end is not None:
                    end.record()
            if start is not None:
                self.events.setdefault(name, []).append((start, end))
        finally:
            self.active = None

    @contextmanager
    def observe(self, bindings, *, record_events=True):
        with ExitStack() as stack:
            for owner, attr, name in bindings:
                original = getattr(owner, attr)

                @wraps(original)
                def observed(*args, _original=original, _name=name, **kwargs):
                    with self.scope(_name):
                        return _original(*args, **kwargs)

                stack.enter_context(patch.object(owner, attr, observed))
            self.enabled = True
            self.record_events = record_events
            try:
                yield
            finally:
                self.enabled = False
                self.record_events = True

    def times(self):
        return {name: sum(start.elapsed_time(end) for start, end in events)
                for name, events in self.events.items()}


def trace_signature(path):
    """Keep copies separately; unlike the legacy tracer, do not discard them."""
    import json

    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
    copies = [e for e in events if "memcpy" in e.get("cat", "").lower()]
    return {"kernel_names": json.dumps(sorted(set(kernels))), "kernel_count": len(kernels),
            "memory_copy_count": len(copies),
            "memory_copy_time_ms": sum(e.get("dur", 0) for e in copies) / 1000}

"""Immutable workload and evidence for the selected Step TP8/DP2 model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math


@dataclass(frozen=True)
class CollectiveTensor:
    name: str
    dtype: str
    element_size: int
    columns: int
    row_stride: int

    def __post_init__(self):
        if not self.name or not self.dtype or any(type(v) is not int or v <= 0
                for v in (self.element_size, self.columns, self.row_stride)):
            raise ValueError("Collective tensor requires positive integer geometry")
        if self.row_stride != self.columns:
            raise ValueError("Selected Step collectives require contiguous row layout")


@dataclass(frozen=True)
class CollectivePlan:
    phase: str
    sizes: tuple[int, int]
    tensors: tuple[CollectiveTensor, ...]
    groups: tuple[tuple[int, ...], ...]
    graph_mode: str
    context_json: str
    buffer_json: str

    def __post_init__(self):
        if self.phase not in ("dispatch", "combine", "shared_tp", "routed_tp"):
            raise ValueError("Unsupported Step collective phase")
        if len(self.sizes) != 2 or any(type(n) is not int or n <= 0 for n in self.sizes):
            raise ValueError("Step DP2 requires two positive physical row counts")
        expected = (tuple((t, 8 + t) for t in range(8)) if self.phase in
                    ("dispatch", "combine") else (tuple(range(8)), tuple(range(8, 16))))
        if self.groups != expected:
            raise ValueError("Step collective groups must match native TP8/DP2 rank order")
        if not self.tensors or len({t.name for t in self.tensors}) != len(self.tensors):
            raise ValueError("Collective requires distinct ordered tensors")
        if self.phase != "dispatch" and len(self.tensors) != 1:
            raise ValueError("Step combine/TP reduction requires one output tensor")
        if self.graph_mode not in ("none", "full"):
            raise ValueError("Step plan requires actual eager/full graph modality")
        if self.graph_mode == "full" and self.sizes[0] != self.sizes[1]:
            raise ValueError("Captured Step DP input must have uniform physical sizes")
        for value in (self.context_json, self.buffer_json):
            if not isinstance(json.loads(value), dict) or not json.loads(value):
                raise ValueError("Collective context and buffer receipts must be nonempty objects")

    @property
    def stages(self):
        if self.phase in ("shared_tp", "routed_tp"):
            return (("AllReduce", 0, None),)
        uniform = self.sizes[0] == self.sizes[1]
        primitive = ("AllGather" if self.phase == "dispatch" else "ReduceScatter") if uniform else (
            "Broadcast" if self.phase == "dispatch" else "Reduce")
        return tuple((primitive, j, root) for j in range(len(self.tensors))
                     for root in ((None,) if uniform else (0, 1)))

    def to_dict(self):
        return {**asdict(self), "context": json.loads(self.context_json),
                "buffers": json.loads(self.buffer_json), "stages": self.stages,
                "native_grouping": "outer_tensor_group/nested_root_group"}

    def cache_key(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def endpoint_bytes(self, rank):
        own = self.sizes[rank // 8] * sum(t.columns * t.element_size for t in self.tensors)
        total = sum(self.sizes) * sum(t.columns * t.element_size for t in self.tensors)
        return (own, total) if self.phase == "dispatch" else (
            (total, own) if self.phase == "combine" else (own, own))


@dataclass(frozen=True)
class CollectivePlanCost:
    duration_ms: float
    network_ms: float
    local_rank_ms: tuple[float, ...]
    remainder_ms: float
    kernel_counts: tuple[int, ...]
    count_receipt_json: str
    plan_key: str
    model_identity: str

    def __post_init__(self):
        values = (self.duration_ms, self.network_ms, self.remainder_ms, *self.local_rank_ms)
        if len(self.local_rank_ms) != 16 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Collective model requires finite nonnegative per-rank service")
        if len(self.kernel_counts) != 16 or any(type(n) is not int or n < 1 for n in self.kernel_counts):
            raise ValueError("Actual paired native per-rank kernel counts are required")
        if not math.isclose(self.duration_ms, self.network_ms + max(self.local_rank_ms)
                            + self.remainder_ms, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Step collective composition must be network+max(local)+remainder")
        if not json.loads(self.count_receipt_json) or not self.plan_key or not self.model_identity:
            raise ValueError("Collective cost requires actual count/plan/model identities")

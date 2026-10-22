"""Immutable workload and evidence for the selected Step TP8/DP2 model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from frontier.profiling.collectives.vllm_forward_collectives import DTYPE_BYTES


def collective_runtime_identity(runtime):
    """Frozen policy excludes enabled state changed by native lazy initialization."""
    return {k:v for k,v in runtime.items() if k != "enabled_communicators"}


def collective_model_context(context):
    """Fit one frozen policy across its size-dependent native implementations."""
    policies=context["execution_policy"]
    if not policies:
        raise ValueError("Collective model requires actual execution policy receipts")
    return {**{k:v for k,v in context.items() if k != "execution_policy"},
        "runtime_policy":{rank:collective_runtime_identity(policy["runtime"]) for rank,policy in policies.items()}}


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
        if DTYPE_BYTES.get(self.dtype) != self.element_size:
            raise ValueError("Collective dtype and actual element size disagree")


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
        object.__setattr__(self,"sizes",tuple(self.sizes))
        object.__setattr__(self,"tensors",tuple(self.tensors))
        object.__setattr__(self,"groups",tuple(tuple(group) for group in self.groups))
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
        if type(rank) is not int or not 0 <= rank < 16:
            raise ValueError("Step endpoint rank must be in the admitted sixteen-rank world")
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
        object.__setattr__(self,"local_rank_ms",tuple(self.local_rank_ms))
        object.__setattr__(self,"kernel_counts",tuple(self.kernel_counts))
        values = (self.duration_ms, self.network_ms, self.remainder_ms, *self.local_rank_ms)
        if len(self.local_rank_ms) != 16 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Collective model requires finite nonnegative per-rank service")
        if len(self.kernel_counts) != 16 or any(type(n) is not int or n < 1 for n in self.kernel_counts):
            raise ValueError("Actual paired native per-rank kernel counts are required")
        if not math.isclose(self.duration_ms, self.network_ms + max(self.local_rank_ms)
                            + self.remainder_ms, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Step collective composition must be network+max(local)+remainder")
        receipt = json.loads(self.count_receipt_json)
        if (not receipt or not self.plan_key or not self.model_identity or
                receipt.get("kernel_counts") != list(self.kernel_counts) or
                not receipt.get("per_rank_events") or not receipt.get("source_receipt") or
                receipt.get("evidence_kind") not in ("native_measurement", "cpu_fixture")):
            raise ValueError("Collective cost requires actual count/plan/model identities")
        if receipt["evidence_kind"] == "native_measurement":
            calls = receipt["profiled_calls"]
            events = receipt["per_rank_events"]
            if type(calls) is not int or calls < 1 or len(events) != 16 or any(
                    len(e) != n * calls for e,n in zip(events,self.kernel_counts)):
                raise ValueError("Native counts must pair every device's actual events and replay multiplicity")

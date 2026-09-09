"""Observed top-k selections; logical and graph-padding assignments stay separate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RoutingRecord:
    batch_id: str
    rank: int
    layer_id: int
    capture_id: int
    logical_size: int
    physical_size: int
    num_experts: int
    top_k: int
    logical_expert_counts: tuple[int, ...]
    padding_expert_counts: tuple[int, ...]
    logical_assignment_sha256: str
    logical_expert_set_sha256: str
    padding_assignment_sha256: str
    logical_weights_sha256: str
    padding_weights_sha256: str
    logical_positive_weight_slots: int
    padding_positive_weight_slots: int
    invalid_padding_slots: int
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported routing schema")
        if not isinstance(self.batch_id, str) or not self.batch_id:
            raise ValueError("Routing batch_id must be nonempty")
        for name in ("rank", "layer_id", "capture_id", "logical_size", "physical_size",
                     "num_experts", "top_k", "logical_positive_weight_slots",
                     "padding_positive_weight_slots", "invalid_padding_slots"):
            value = getattr(self, name)
            minimum = 1 if name in {"capture_id", "logical_size", "physical_size", "num_experts", "top_k"} else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"Invalid integer routing field {name}")
        if self.physical_size < self.logical_size or self.top_k > self.num_experts:
            raise ValueError("Invalid logical/physical routing shape or top-k")
        for name in ("logical_expert_counts", "padding_expert_counts"):
            counts = getattr(self, name)
            if len(counts) != self.num_experts or any(type(v) is not int or v < 0 for v in counts):
                raise ValueError("Expert counts require one nonnegative integer per expert")
        if (sum(self.logical_expert_counts) != self.logical_size * self.top_k
                or max(self.logical_expert_counts) > self.logical_size):
            raise ValueError("Logical counts disagree with distinct top-k selections")
        if (sum(self.padding_expert_counts) + self.invalid_padding_slots
                != (self.physical_size - self.logical_size) * self.top_k):
            raise ValueError("Padding counts disagree with graph lanes")
        if (self.logical_positive_weight_slots > sum(self.logical_expert_counts)
                or self.padding_positive_weight_slots > sum(self.padding_expert_counts)):
            raise ValueError("Positive weights exceed selected expert slots")
        for name in ("logical_assignment_sha256", "logical_expert_set_sha256", "padding_assignment_sha256",
                     "logical_weights_sha256", "padding_weights_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("Invalid routing digest")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        fields = dict(value)
        for name in ("logical_expert_counts", "padding_expert_counts"):
            fields[name] = tuple(fields[name])
        return cls(**fields)

    def to_lane_workload(self, *, include_padding: bool):
        """Use Frontier's canonical EP=1 descriptor, never invent EP placement."""
        from frontier.moe_ep_workload import split_global_expert_tokens_into_lanes
        counts = [a + (b if include_padding else 0)
                  for a, b in zip(self.logical_expert_counts, self.padding_expert_counts)]
        return split_global_expert_tokens_into_lanes(dict(enumerate(counts)),
            total_expert_num=self.num_experts, moe_expert_parallel_size=1,
            router_topk=self.top_k)[0]

    def native_load_features(self, *, model_config, include_padding: bool):
        """Canonical predictor features only; no latency fit or CSV admission."""
        from frontier.moe_load_imbalance import MoELoadImbalanceInput
        if (model_config.num_experts != self.num_experts
                or model_config.num_experts_per_tok != self.top_k):
            raise ValueError("Routing vector does not match model experts/top-k")
        lane = self.to_lane_workload(include_padding=include_padding)
        return MoELoadImbalanceInput(
            num_tokens=self.physical_size if include_padding else self.logical_size,
            num_experts_per_device=lane.local_expert_width,
            hidden_dim=model_config.embedding_dim, expert_hidden_dim=model_config.mlp_hidden_dim,
            router_topk=self.top_k, expert_token_counts=list(lane.local_token_counts),
            load_distribution="runtime").to_features_dict()


def snapshot_routing(ids, weights, *, batch_id, rank, layer_id, capture_id,
                     logical_size, num_experts, top_k):
    """Validate a CPU snapshot of the final top-k output passed to the experts.

    Zero-weight valid IDs remain counted: weighting alone does not establish
    whether sorting/GEMM skips them. Padding may contain sentinel -1 with zero
    weight, but live tokens must select distinct valid experts.
    """
    ids, weights = np.asarray(ids), np.asarray(weights)
    if (ids.ndim != 2 or ids.dtype.kind not in "iu" or weights.dtype.kind != "f"
            or weights.shape != ids.shape or ids.shape[1] != top_k
            or not 0 < logical_size <= ids.shape[0] or not 0 < top_k <= num_experts):
        raise ValueError("Invalid top-k tensor shape, dtype or logical size")
    if (not np.isfinite(weights).all() or (weights < 0).any()
            or (ids >= num_experts).any() or (ids < -1).any()
            or (ids[:logical_size] < 0).any() or (weights[ids == -1] != 0).any()):
        raise ValueError("Invalid expert IDs or routing weights")
    logical_sorted = np.sort(ids[:logical_size], axis=1)
    if (np.diff(logical_sorted, axis=1) == 0).any():
        raise ValueError("Live tokens must select distinct experts")
    def digest(value, dtype):
        return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes(order="C")).hexdigest()
    def counts(value):
        return tuple(int(v) for v in np.bincount(value[value >= 0].astype(np.int64), minlength=num_experts))
    return RoutingRecord(batch_id=batch_id, rank=rank, layer_id=layer_id, capture_id=capture_id,
        logical_size=logical_size, physical_size=ids.shape[0], num_experts=num_experts, top_k=top_k,
        logical_expert_counts=counts(ids[:logical_size]), padding_expert_counts=counts(ids[logical_size:]),
        logical_assignment_sha256=digest(ids[:logical_size], "<i4"),
        logical_expert_set_sha256=digest(logical_sorted, "<i4"),
        padding_assignment_sha256=digest(ids[logical_size:], "<i4"),
        logical_weights_sha256=digest(weights[:logical_size], "<f4"),
        padding_weights_sha256=digest(weights[logical_size:], "<f4"),
        logical_positive_weight_slots=int((weights[:logical_size] > 0).sum()),
        padding_positive_weight_slots=int((weights[logical_size:] > 0).sum()),
        invalid_padding_slots=int((ids[logical_size:] == -1).sum()))


def read_routing(path: str | Path):
    seen = set()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = RoutingRecord.from_dict(json.loads(line))
                key = record.batch_id, record.rank, record.layer_id
                if key in seen:
                    raise ValueError("Duplicate routing batch/rank/layer")
                seen.add(key)
                yield record
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not seen:
        raise ValueError(f"Empty routing ledger: {path}")

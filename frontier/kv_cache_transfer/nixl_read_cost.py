"""Exact, input-bound NIXL read costs for the existing measured PD consumer."""

from dataclasses import dataclass
import json
from math import isfinite
from numbers import Real
from pathlib import Path

from frontier.attention.memory import (
    AttentionKVCacheCapacity,
    AttentionKVCacheComponent,
    AttentionKVCacheGroup,
)
from frontier.entities.kv_cache_transfer_info import NixlTransferCoverage
from frontier.model_architectures import get_model_architecture_profile


@dataclass(frozen=True)
class _Read:
    coverage: NixlTransferCoverage
    transfer_bytes_per_rank: int
    time_ms: float


class NixlReadCost:
    """Load one fixed hardware/layout profile without extrapolating missing rows.

    Input validity establishes a consumable cost contract, not independent native
    qualification. The canonical gate owns the latter evidence requirement.
    """

    def __init__(self, architecture_profile: str, records: dict[int, _Read]):
        self._architecture_profile = architecture_profile
        self._records = records

    @classmethod
    def from_file(cls, path: str | Path | None) -> "NixlReadCost":
        if not path:
            raise ValueError("Missing measured NIXL profile")
        with Path(path).open() as stream:
            profile = json.load(stream)
        try:
            if profile["schema"] != "nixl_read_cost.v1":
                raise ValueError("Unsupported measured NIXL profile schema")
            for field in ("model_architecture_profile", "profile_identity", "source_device", "target_device"):
                if not isinstance(profile[field], str) or not profile[field].strip():
                    raise ValueError(f"Measured NIXL profile requires {field}")
            layout = dict(profile["cache_capacity"])
            layout["components"] = tuple(AttentionKVCacheComponent(**row) for row in layout["components"])
            layout["groups"] = tuple(AttentionKVCacheGroup(**row) for row in layout["groups"])
            capacity = AttentionKVCacheCapacity(**layout)
            records = {}
            for row in profile["measurements"]:
                coverage = NixlTransferCoverage(
                    num_prefill_tokens=row["num_prefill_tokens"],
                    num_computed_tokens=row["num_computed_tokens"],
                    cache_capacity=capacity,
                    source_attention_tp=profile["source_attention_tp"],
                    target_attention_tp=profile["target_attention_tp"],
                    source_device=profile["source_device"],
                    target_device=profile["target_device"],
                    profile_identity=profile["profile_identity"],
                    duration_scope=profile["duration_scope"],
                )
                size = row["transfer_bytes_per_rank"]
                duration = row["time_ms"]
                if type(size) is not int or size <= 0:
                    raise ValueError("Measured NIXL payload bytes must be a positive integer")
                if isinstance(duration, bool) or not isinstance(duration, Real) or not isfinite(duration) or duration <= 0:
                    raise ValueError("Measured NIXL READ cost must be positive and finite")
                if coverage.num_prefill_tokens in records:
                    raise ValueError("Duplicate measured NIXL prompt coverage")
                records[coverage.num_prefill_tokens] = _Read(coverage, size, float(duration))
            if not records:
                raise ValueError("Missing measured NIXL coverage rows")
        except (KeyError, TypeError) as error:
            raise ValueError(f"Incomplete measured NIXL profile: {error}") from error
        return cls(profile["model_architecture_profile"], records)

    def get_coverage(
        self, *, request, source_replica_config, target_replica_config,
        cache_capacity, target_cost_provider,
    ) -> NixlTransferCoverage:
        row = self._records.get(request.num_prefill_tokens)
        if row is None:
            raise ValueError("Missing measured NIXL coverage for prompt length")
        for config in (source_replica_config, target_replica_config):
            if get_model_architecture_profile(config.model_config).profile_id != self._architecture_profile:
                raise ValueError("Measured NIXL model architecture mismatch")
        actual = (
            cache_capacity, source_replica_config.device, target_replica_config.device,
            source_replica_config.attn_tensor_parallel_size,
            target_replica_config.attn_tensor_parallel_size,
        )
        coverage = row.coverage
        expected = (
            coverage.cache_capacity, coverage.source_device, coverage.target_device,
            coverage.source_attention_tp, coverage.target_attention_tp,
        )
        if actual != expected:
            raise ValueError("Measured NIXL cache layout, device or TP mismatch")
        return coverage

    def get_time_ms(self, *, coverage, payload) -> float:
        row = self._records.get(coverage.num_prefill_tokens)
        if row is None or row.coverage != coverage:
            raise ValueError("Measured NIXL coverage does not belong to this profile")
        if payload.transfer_bytes_per_rank != row.transfer_bytes_per_rank:
            raise ValueError("Measured NIXL payload does not match the reserved prefix")
        return row.time_ms

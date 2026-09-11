"""Versioned, tensor-free snapshots of actual model execution batches."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BatchRecord:
    """One rank of one iteration; context lengths exclude scheduled query tokens.

    Model/topology/runtime identity belongs to the run's manifest. Timings are
    observations only, never features passed to the execution-time predictor.
    """

    batch_id: str
    rank: int
    phase: str
    request_ids: tuple[str, ...]
    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    prefill_mask: tuple[bool, ...]
    prompt_lens: tuple[int, ...] | None = None
    graph_mode: str = "NONE"
    capture_size: int = 0
    repetition: int = 0
    step: int = 0
    profiled: bool = False
    forward_gpu_ms: float | None = None
    step_wall_ms: float | None = None
    schema_version: int = 1
    decode_input_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(f"Unsupported batch schema version: {self.schema_version}")
        if not isinstance(self.batch_id, str) or not self.batch_id:
            raise ValueError("batch_id must be a nonempty string")
        for key in ("rank", "capture_size", "repetition", "step"):
            value = getattr(self, key)
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        size = len(self.request_ids)
        if not size or len(set(self.request_ids)) != size:
            raise ValueError("request_ids must be nonempty and unique within a batch")
        if any(not isinstance(value, str) or not value for value in self.request_ids):
            raise ValueError("request_ids must contain nonempty strings")
        for key in ("query_lens", "context_lens", "prefill_mask"):
            if len(getattr(self, key)) != size:
                raise ValueError(f"{key} must have one entry per request")
        if any(type(v) is not int or v <= 0 for v in self.query_lens):
            raise ValueError("query_lens must contain positive integers")
        if any(type(v) is not int or v < 0 for v in self.context_lens):
            raise ValueError("context_lens must contain nonnegative integers")
        if any(type(v) is not bool for v in self.prefill_mask):
            raise ValueError("prefill_mask must contain booleans")
        if self.prompt_lens is not None:
            if len(self.prompt_lens) != size or any(type(v) is not int or v <= 0 for v in self.prompt_lens):
                raise ValueError("prompt_lens must have one positive integer per request")
            for prompt, context, query, prefill in zip(
                self.prompt_lens, self.context_lens, self.query_lens, self.prefill_mask
            ):
                if (prefill and context + query > prompt) or (not prefill and context < prompt):
                    raise ValueError("prompt_lens disagree with query/context progress")
        expected_phase = (
            "prefill" if all(self.prefill_mask)
            else "mixed" if any(self.prefill_mask) else "decode"
        )
        if self.phase != expected_phase:
            raise ValueError(f"phase disagrees with prefill_mask: expected {expected_phase}")
        if any(q != 1 or c < 1 for q, c, p in zip(
            self.query_lens, self.context_lens, self.prefill_mask
        ) if not p):
            raise ValueError("Non-speculative decode requires query=1 and positive context")
        if self.graph_mode not in {"NONE", "FULL"}:
            raise ValueError("Batch replay currently supports NONE/FULL graph modes")
        if self.graph_mode == "FULL":
            if self.phase != "decode" or self.capture_size < size:
                raise ValueError("FULL graphs require pure decode and capture_size >= batch size")
        elif self.capture_size:
            raise ValueError("Eager batches must have capture_size=0")
        if type(self.profiled) is not bool:
            raise ValueError("profiled must be boolean")
        if self.decode_input_sha256 is not None:
            digest = self.decode_input_sha256
            if (self.phase != "decode" or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("decode_input_sha256 requires decode and a lowercase SHA-256 digest")
        for key in ("forward_gpu_ms", "step_wall_ms"):
            value = getattr(self, key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive when present")

    @classmethod
    def from_dict(cls, value: dict) -> "BatchRecord":
        fields = dict(value)
        for key in ("request_ids", "query_lens", "context_lens", "prefill_mask"):
            fields[key] = tuple(fields[key])
        if fields.get("prompt_lens") is not None:
            fields["prompt_lens"] = tuple(fields["prompt_lens"])
        return cls(**fields)

    def to_dict(self) -> dict:
        return asdict(self)

    def shape_key(self) -> tuple:
        return (self.phase, self.query_lens, self.context_lens,
                self.prefill_mask, self.graph_mode, self.capture_size, self.prompt_lens)

    def to_frontier_batch(self, *, is_moe: bool):
        """Hydrate an isolated prediction snapshot, with no scheduler side effects."""
        from frontier.entities.batch import Batch, DecodeCudaGraphMetadata
        from frontier.entities.request import Request

        requests = []
        for index, (query, context, prefill) in enumerate(zip(
            self.query_lens, self.context_lens, self.prefill_mask
        )):
            prompt = self.prompt_lens[index] if self.prompt_lens else (
                context + query if prefill else context
            )
            request = Request(
                arrived_at=0.0,
                num_prefill_tokens=prompt,
                num_decode_tokens=max(2, context - prompt + 2),
                num_processed_tokens=context,
            )
            # Snapshot hydration: no prefill-completion event should be invented.
            request._is_prefill_complete = not prefill
            requests.append(request)
        batch = Batch(0, requests, list(self.query_lens), is_moe=is_moe)
        decode_size = sum(not value for value in self.prefill_mask)
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            config_mode="full_decode_only", runtime_mode=self.graph_mode,
            capture_hit=self.graph_mode == "FULL", is_mixed_batch=self.phase == "mixed",
            original_total_tokens=sum(self.query_lens),
            padded_total_tokens=self.capture_size or sum(self.query_lens),
            original_decode_batch_size=decode_size,
            padded_decode_batch_size=self.capture_size or decode_size,
        )
        return batch


def read_batches(path: str | Path) -> list[BatchRecord]:
    records = []
    seen = set()
    with open(path, encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = BatchRecord.from_dict(json.loads(line))
                key = (record.batch_id, record.rank)
                if key in seen:
                    raise ValueError(f"Duplicate batch/rank: {key}")
                seen.add(key)
                records.append(record)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not records:
        raise ValueError(f"Empty batch ledger: {path}")
    return records


def validate_rank_cohorts(records: Iterable[BatchRecord], *, tensor_parallel_size: int) -> dict:
    """Require complete, shape-identical TP cohorts before aggregating timings."""
    if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be a positive integer")
    cohorts: dict[str, list[BatchRecord]] = {}
    for record in records:
        cohorts.setdefault(record.batch_id, []).append(record)
    for batch_id, cohort in cohorts.items():
        if sorted(r.rank for r in cohort) != list(range(tensor_parallel_size)):
            raise ValueError(f"Incomplete or duplicate TP ranks for {batch_id}")
        if any(r.shape_key() != cohort[0].shape_key() or
               r.request_ids != cohort[0].request_ids or
               r.decode_input_sha256 != cohort[0].decode_input_sha256 or
               (r.profiled, r.repetition, r.step) !=
               (cohort[0].profiled, cohort[0].repetition, cohort[0].step)
               for r in cohort):
            raise ValueError(f"TP ranks disagree on batch shape/identity for {batch_id}")
    return cohorts

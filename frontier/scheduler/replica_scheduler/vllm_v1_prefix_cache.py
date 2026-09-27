"""Prefix-cache admission and its identity ledger for the vLLM V1 scheduler.

Admission decides how many cached blocks a request may reuse; the ledger emits
the identity events that let a run be checked for block-reuse correctness.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from frontier.entities.batch import Request
from frontier.kv_cache.base_kv_cache_manager import KVCacheAllocationResult
from frontier.kv_cache.kv_cache_block import KVCacheBlock, KVCacheBlockBinding
from frontier.scheduler.replica_scheduler.vllm_v1_decision_log import (
    _log_frontier_vllm_v1_schedule_decision,
)
from frontier.types import ClusterType


@dataclass(frozen=True)
class PrefixCacheAdmission:
    raw_hit_blocks: tuple[KVCacheBlock, ...]
    effective_hit_blocks: tuple[KVCacheBlock, ...]
    raw_hit_bindings: tuple[KVCacheBlockBinding, ...]
    effective_hit_bindings: tuple[KVCacheBlockBinding, ...]
    raw_cached_tokens: int
    effective_cached_tokens: int
    num_new_tokens: int
    full_hit_backoff_applied: bool


def _serialize_prefix_cache_binding(
    binding: KVCacheBlockBinding,
) -> Dict[str, Any]:
    return {
        "block_hash": binding.block_hash,
        "block_id": int(binding.block_id),
        "creator_request_id": str(binding.creator_request_id),
        "binding_epoch": int(binding.binding_epoch),
    }


class PrefixCacheLedger:
    """Prefix-cache admission and identity-event emission."""

    def _is_prefix_caching_enabled(self) -> bool:
        return getattr(self, "_kv_cache_manager", None) is not None

    def _sync_prefix_cache_allocation_state(
        self, request: Optional[Request] = None
    ) -> None:
        if not self._is_prefix_caching_enabled():
            return
        assert self._kv_cache_manager is not None
        self._num_allocated_blocks = int(self._kv_cache_manager.num_used_blocks)
        if request is not None:
            num_blocks = int(self._kv_cache_manager.get_num_blocks_for_request(request))
            if num_blocks > 0:
                self._allocation_map[request.id] = num_blocks
            else:
                self._allocation_map.pop(request.id, None)

    def _prepare_prefix_cache_admission(
        self, request: Request
    ) -> PrefixCacheAdmission:
        if not self._is_prefix_caching_enabled():
            return PrefixCacheAdmission(
                raw_hit_blocks=(),
                effective_hit_blocks=(),
                raw_hit_bindings=(),
                effective_hit_bindings=(),
                raw_cached_tokens=0,
                effective_cached_tokens=0,
                num_new_tokens=self._get_request_next_num_tokens(request),
                full_hit_backoff_applied=False,
            )
        if request.block_hash_ids is None:
            raise ValueError(
                "block_hash_ids are required when enable_prefix_caching=True"
            )
        assert self._kv_cache_manager is not None
        computed_blocks, num_computed_tokens = self._kv_cache_manager.get_computed_blocks(
            request
        )
        raw_hit_blocks = tuple(computed_blocks)
        raw_hit_bindings: list[KVCacheBlockBinding] = []
        query_hashes = list(request.block_hash_ids)
        for query_index, block in enumerate(raw_hit_blocks):
            binding = block.binding
            if binding is None:
                raise ValueError(
                    f"Prefix cache hit block {block.block_id} has no binding identity."
                )
            if binding.block_hash != query_hashes[query_index]:
                raise ValueError(
                    "Prefix cache hit binding disagrees with ordered query hash: "
                    f"query_index={query_index}, "
                    f"query_hash={query_hashes[query_index]!r}, "
                    f"binding_hash={binding.block_hash!r}"
                )
            raw_hit_bindings.append(binding)
        raw_cached_tokens = int(num_computed_tokens)
        schedule_target_tokens = int(
            request.num_processed_tokens
            if request.is_recomputing
            else request.num_prefill_tokens
        )
        num_new_tokens = schedule_target_tokens - int(num_computed_tokens)
        full_hit_backoff_applied = False
        if num_new_tokens == 0 and computed_blocks:
            num_computed_tokens -= int(self._config.block_size)
            num_new_tokens = int(self._config.block_size)
            computed_blocks = list(computed_blocks[:-1])
            self._kv_cache_manager.prefix_cache_stats.hits -= 1
            full_hit_backoff_applied = True
        return PrefixCacheAdmission(
            raw_hit_blocks=raw_hit_blocks,
            effective_hit_blocks=tuple(computed_blocks),
            raw_hit_bindings=tuple(raw_hit_bindings),
            effective_hit_bindings=tuple(
                raw_hit_bindings[: len(computed_blocks)]
            ),
            raw_cached_tokens=raw_cached_tokens,
            effective_cached_tokens=int(num_computed_tokens),
            num_new_tokens=int(num_new_tokens),
            full_hit_backoff_applied=full_hit_backoff_applied,
        )

    def _prefix_cache_identity_event_base(
        self,
        *,
        event: str,
        request: Request,
    ) -> Dict[str, Any]:
        event_seq = int(self._prefix_cache_identity_event_seq)
        self._prefix_cache_identity_event_seq = event_seq + 1
        cluster_name = (
            self._cluster_type.name
            if self._cluster_type is not None
            else ClusterType.MONOLITHIC.name
        )
        replica_local_id = self._replica_local_id
        if replica_local_id is not None and (
            type(replica_local_id) is not int or replica_local_id < 0
        ):
            raise ValueError(
                "Prefix cache identity replica_local_id must be None or an "
                f"exact non-negative int, got {replica_local_id!r}"
            )
        return {
            "event": event,
            "prefix_identity_schema_version": 1,
            "source": "frontier",
            "scheduler": "vllm_v1",
            "cluster_type": cluster_name,
            "replica_id": int(self._replica_id),
            "replica_local_id": replica_local_id,
            "iteration_id": int(self._active_schedule_iteration_id),
            "identity_event_seq": event_seq,
            "request_id": str(request.id),
            "prefix_cache_block_size": int(self._config.block_size),
            "simulation_time": float(self._current_schedule_time),
            "simulation_time_semantics": "frontier_event_time_seconds",
        }

    def _serialize_prefix_cache_hit_bindings(
        self,
        *,
        request: Request,
        bindings: Sequence[KVCacheBlockBinding],
    ) -> List[Dict[str, Any]]:
        query_hashes = list(request.block_hash_ids or [])
        if len(bindings) > len(query_hashes):
            raise ValueError(
                "Prefix cache hit count exceeds the ordered query hash count."
            )
        rows: List[Dict[str, Any]] = []
        for query_index, binding in enumerate(bindings):
            if binding.block_hash != query_hashes[query_index]:
                raise ValueError(
                    "Prefix cache hit binding disagrees with ordered query hash: "
                    f"query_index={query_index}, "
                    f"query_hash={query_hashes[query_index]!r}, "
                    f"binding_hash={binding.block_hash!r}"
                )
            rows.append(
                {
                    "query_index": query_index,
                    **_serialize_prefix_cache_binding(binding),
                }
            )
        return rows

    def _emit_prefix_cache_identity_events(
        self,
        *,
        request: Request,
        num_new_tokens: int,
        allocation: KVCacheAllocationResult,
        admission: Optional[PrefixCacheAdmission],
    ) -> None:
        if admission is not None:
            admission_payload = self._prefix_cache_identity_event_base(
                event="prefix_cache_admission",
                request=request,
            )
            admission_payload.update(
                {
                    "query_hashes": list(request.block_hash_ids or []),
                    "raw_hit_blocks": self._serialize_prefix_cache_hit_bindings(
                        request=request,
                        bindings=admission.raw_hit_bindings,
                    ),
                    "admitted_hit_blocks": self._serialize_prefix_cache_hit_bindings(
                        request=request,
                        bindings=admission.effective_hit_bindings,
                    ),
                    "raw_cached_tokens": int(admission.raw_cached_tokens),
                    "admitted_cached_tokens": int(
                        admission.effective_cached_tokens
                    ),
                    "num_new_tokens": int(num_new_tokens),
                    "full_hit_backoff_applied": bool(
                        admission.full_hit_backoff_applied
                    ),
                }
            )
            _log_frontier_vllm_v1_schedule_decision(admission_payload)

        allocation_payload = self._prefix_cache_identity_event_base(
            event="prefix_cache_allocation",
            request=request,
        )
        reused_blocks: List[Dict[str, Any]] = []
        for block in allocation.reused_blocks:
            binding = block.binding
            if binding is None:
                raise ValueError(
                    f"Reused Prefix cache block {block.block_id} has no binding identity."
                )
            reused_blocks.append(_serialize_prefix_cache_binding(binding))
        allocation_payload.update(
            {
                "num_new_tokens": int(num_new_tokens),
                "reused_blocks": reused_blocks,
                "new_block_ids": [
                    int(block.block_id) for block in allocation.new_blocks
                ],
                "evicted_bindings": [
                    _serialize_prefix_cache_binding(binding)
                    for binding in allocation.evicted_bindings
                ],
                "new_bindings": [
                    _serialize_prefix_cache_binding(binding)
                    for binding in allocation.new_bindings
                ],
            }
        )
        _log_frontier_vllm_v1_schedule_decision(allocation_payload)

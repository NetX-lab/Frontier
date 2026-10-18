from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Optional

from frontier.attention.memory import AttentionKVCacheCapacity
from frontier.entities.request import Request
from frontier.kv_cache.kv_cache_block import KVCacheBlock, KVCacheBlockBinding
from frontier.kv_cache.kv_cache_block_pool import BlockPool


@dataclass
class PrefixCacheStats:
    reset: bool = False
    requests: int = 0
    queries: int = 0
    hits: int = 0


@dataclass(frozen=True)
class KVCacheAllocationResult:
    reused_blocks: tuple[KVCacheBlock, ...]
    new_blocks: tuple[KVCacheBlock, ...]
    evicted_bindings: tuple[KVCacheBlockBinding, ...]
    new_bindings: tuple[KVCacheBlockBinding, ...]
    group_new_block_ids: tuple[tuple[str, tuple[int, ...]], ...] = ()
    zero_required_block_ids: tuple[int, ...] = ()


def _require_positive_int(value: int, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{field} must be a positive integer, got {value!r}"
        )
    return value


def _require_nonnegative_int(value: int, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(
            f"{field} must be a non-negative integer, got {value!r}"
        )
    return value


class KVCacheManager:
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        enable_caching: bool,
        caching_hash_algo: str,
        num_preallocate_tokens: int,
        *,
        cache_capacity: AttentionKVCacheCapacity | None = None,
    ) -> None:
        self.block_size = int(block_size)
        self.num_gpu_blocks = int(num_gpu_blocks)
        self.enable_caching = bool(enable_caching)
        self.caching_hash_algo = str(caching_hash_algo)
        self.num_preallocate_tokens = int(num_preallocate_tokens)
        self.num_preallocate_blocks = (
            ceil(self.num_preallocate_tokens / self.block_size)
            if self.num_preallocate_tokens > 0
            else 0
        )
        self.cache_capacity = cache_capacity
        if cache_capacity is not None:
            if not cache_capacity.is_grouped:
                raise ValueError("cache_capacity must declare a grouped layout")
            if cache_capacity.block_size != self.block_size:
                raise ValueError("grouped cache scheduler block size disagrees with layout")
            if self.enable_caching or self.num_preallocate_tokens:
                raise ValueError("grouped allocation does not support prefix caching or preallocation")
        self.block_pool = BlockPool(
            self.num_gpu_blocks, self.enable_caching,
            reserved_blocks=cache_capacity.reserved_blocks if cache_capacity is not None else 0,
        )
        self.req_to_blocks: defaultdict[int, list[KVCacheBlock]] = defaultdict(list)
        self._group_tables: dict[int, dict[str, list[KVCacheBlock | None]]] = {}
        self._settled_tokens: dict[int, int] = {}
        self.num_cached_blocks: dict[int, int] = {}
        self.prefix_cache_stats = PrefixCacheStats()

    @property
    def usage(self) -> float:
        return self.block_pool.get_usage()

    @property
    def num_used_blocks(self) -> int:
        return self.block_pool.get_num_used_blocks()

    def get_num_blocks_for_request(self, request: Request) -> int:
        if self.cache_capacity is not None:
            return sum(
                block is not None
                for table in self._group_tables.get(request.id, {}).values() for block in table
            )
        return len(self.req_to_blocks.get(request.id, []))

    def has_grouped_allocation(self, request_id: int) -> bool:
        """Return whether the group tables own this request."""
        return request_id in self._group_tables

    def get_group_block_ids(self, request_id: int) -> dict[str, tuple[int | None, ...]]:
        """Return logical positions, retaining null slots for released pages."""
        return {
            group_id: tuple(None if block is None else block.block_id for block in table)
            for group_id, table in self._group_tables.get(request_id, {}).items()
        }

    def reclaim_skipped_blocks(self, request: Request, *, settled_num_computed_tokens: int) -> None:
        """Retire whole window pages using completed KV progress only."""
        if self.cache_capacity is None:
            raise ValueError("explicit window reclamation requires grouped allocation")
        settled = _require_nonnegative_int(
            settled_num_computed_tokens, field="settled_num_computed_tokens"
        )
        if settled < self._settled_tokens.get(request.id, 0):
            raise ValueError("settled grouped-cache progress cannot move backwards")
        tables = self._group_tables.get(request.id, {})
        for group in self.cache_capacity.groups:
            table = tables.get(group.group_id, [])
            skipped = self._num_skipped_group_blocks(group, settled)
            released = [block for block in table[:skipped] if block is not None]
            self.block_pool.free_blocks(reversed(released))
            table[:min(skipped, len(table))] = [None] * min(skipped, len(table))
        self._settled_tokens[request.id] = settled

    @staticmethod
    def _num_skipped_group_blocks(group, settled: int) -> int:
        if group.retention == "full_history":
            return 0
        return max(0, settled - group.window_tokens + 1 - group.extra_retained_tokens) // group.logical_block_size

    def _group_allocation_plan(self, request: Request, num_tokens: int, scheduler_num_computed_tokens):
        _require_positive_int(num_tokens, field="num_tokens")
        frontier = self._resolve_scheduler_num_computed_tokens(request, scheduler_num_computed_tokens)
        settled = self._settled_tokens.get(request.id, 0)
        if frontier < settled:
            raise ValueError("allocation frontier cannot precede settled grouped-cache progress")
        tables = self._group_tables.get(request.id, {})
        plan = []
        for group in self.cache_capacity.groups:
            current = len(tables.get(group.group_id, []))
            skipped = self._num_skipped_group_blocks(group, settled)
            required = ceil((frontier + num_tokens) / group.logical_block_size)
            prefix = max(current, skipped)
            plan.append((group.group_id, max(0, skipped - current), max(0, required - prefix)))
        return plan

    def get_num_blocks_to_allocate(self, request: Request, num_tokens: int, *, scheduler_num_computed_tokens=None) -> int:
        """Query new physical demand without changing ownership or free pages."""
        if self.cache_capacity is None:
            return self._get_num_new_blocks_required(
                request, num_tokens, scheduler_num_computed_tokens=scheduler_num_computed_tokens
            )
        return sum(item[2] for item in self._group_allocation_plan(request, num_tokens, scheduler_num_computed_tokens))

    def get_num_admission_blocks(self, num_tokens: int, *, max_model_len: int, max_in_flight_tokens: int) -> int:
        """Return recycling-aware full-request demand, separate from step growth."""
        if self.cache_capacity is None:
            raise ValueError("group admission demand requires grouped allocation")
        _require_positive_int(num_tokens, field="num_tokens")
        _require_positive_int(max_model_len, field="max_model_len")
        _require_positive_int(max_in_flight_tokens, field="max_in_flight_tokens")
        total = 0
        for group in self.cache_capacity.groups:
            required = ceil(min(num_tokens, max_model_len) / group.logical_block_size)
            if group.retention == "sliding_window":
                cap = min(group.window_tokens - 1 + group.extra_retained_tokens + max_in_flight_tokens, max_model_len)
                required = min(required, ceil(cap / group.logical_block_size) + 1)
            total += required
        return total

    def make_prefix_cache_stats(self) -> PrefixCacheStats:
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def _resolve_scheduler_num_computed_tokens(
        self,
        request: Request,
        scheduler_num_computed_tokens: Optional[int],
    ) -> int:
        """Resolve the scheduler frontier before this allocation."""
        if scheduler_num_computed_tokens is None:
            return int(request.num_processed_tokens)
        return _require_nonnegative_int(
            scheduler_num_computed_tokens,
            field="scheduler_num_computed_tokens",
        )

    def get_computed_blocks(self, request: Request) -> tuple[list[KVCacheBlock], int]:
        if not self.enable_caching:
            return [], 0

        block_hashes = request.prompt_block_hash_ids(self.block_size)
        computed_blocks: list[KVCacheBlock] = []
        for block_hash in block_hashes:
            cached_block = self.block_pool.get_cached_block(block_hash)
            if cached_block is None:
                break
            computed_blocks.append(cached_block)

        self.prefix_cache_stats.requests += 1
        self.prefix_cache_stats.queries += len(block_hashes)
        self.prefix_cache_stats.hits += len(computed_blocks)
        return computed_blocks, len(computed_blocks) * self.block_size

    def _get_num_new_blocks_required(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        *,
        scheduler_num_computed_tokens: Optional[int] = None,
    ) -> int:
        _require_positive_int(num_tokens, field="num_tokens")

        computed_blocks = new_computed_blocks or []
        num_computed_tokens = self._resolve_scheduler_num_computed_tokens(
            request,
            scheduler_num_computed_tokens,
        ) + len(computed_blocks) * self.block_size
        num_required_blocks = ceil((num_computed_tokens + num_tokens) / self.block_size)
        current_blocks = self.req_to_blocks[request.id]
        return num_required_blocks - len(current_blocks) - len(computed_blocks)

    def can_allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        *,
        scheduler_num_computed_tokens: Optional[int] = None,
    ) -> bool:
        if self.cache_capacity is not None:
            if new_computed_blocks:
                raise ValueError("grouped allocation does not support computed prefix blocks")
            return self.get_num_blocks_to_allocate(
                request, num_tokens, scheduler_num_computed_tokens=scheduler_num_computed_tokens
            ) <= self.block_pool.get_num_free_blocks()
        num_new_blocks = self._get_num_new_blocks_required(
            request,
            num_tokens,
            new_computed_blocks,
            scheduler_num_computed_tokens=scheduler_num_computed_tokens,
        )
        evictable_computed_blocks = sum(
            1 for block in (new_computed_blocks or []) if block.ref_cnt == 0
        )
        return num_new_blocks <= (
            self.block_pool.get_num_free_blocks() - evictable_computed_blocks
        )

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        *,
        scheduler_num_computed_tokens: Optional[int] = None,
    ) -> Optional[KVCacheAllocationResult]:
        if self.cache_capacity is not None:
            if new_computed_blocks:
                raise ValueError("grouped allocation does not support computed prefix blocks")
            return self._allocate_group_slots(request, num_tokens, scheduler_num_computed_tokens)
        computed_blocks = list(new_computed_blocks or [])
        if not self.can_allocate_slots(
            request,
            num_tokens,
            computed_blocks,
            scheduler_num_computed_tokens=scheduler_num_computed_tokens,
        ):
            return None

        if self.enable_caching:
            self.block_pool.touch(computed_blocks)
        elif computed_blocks:
            raise ValueError(
                "Computed prefix blocks are not allowed when prefix caching is disabled."
            )

        num_new_blocks = self._get_num_new_blocks_required(
            request,
            num_tokens,
            computed_blocks,
            scheduler_num_computed_tokens=scheduler_num_computed_tokens,
        )
        request_blocks = self.req_to_blocks[request.id]
        request_blocks.extend(computed_blocks)
        evicted_bindings: tuple[KVCacheBlockBinding, ...] = ()
        if num_new_blocks > 0:
            num_new_blocks = min(
                num_new_blocks + self.num_preallocate_blocks,
                self.block_pool.get_num_free_blocks(),
            )
            pool_allocation = self.block_pool.get_new_blocks(num_new_blocks)
            new_blocks = list(pool_allocation.blocks)
            evicted_bindings = pool_allocation.evicted_bindings
            request_blocks.extend(new_blocks)
        else:
            new_blocks = []

        if not self.enable_caching:
            return KVCacheAllocationResult(
                reused_blocks=tuple(computed_blocks),
                new_blocks=tuple(new_blocks),
                evicted_bindings=evicted_bindings,
                new_bindings=(),
            )

        num_computed_tokens = self._resolve_scheduler_num_computed_tokens(
            request,
            scheduler_num_computed_tokens,
        ) + len(computed_blocks) * self.block_size
        num_full_blocks = (num_computed_tokens + num_tokens) // self.block_size
        num_cached_blocks = self.num_cached_blocks.get(request.id, len(computed_blocks))
        new_bindings = self.block_pool.cache_full_blocks(
            blocks=request_blocks,
            block_hashes=request.prompt_block_hash_ids(self.block_size),
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            creator_request_id=request.id,
        )
        self.num_cached_blocks[request.id] = num_full_blocks
        return KVCacheAllocationResult(
            reused_blocks=tuple(computed_blocks),
            new_blocks=tuple(new_blocks),
            evicted_bindings=evicted_bindings,
            new_bindings=new_bindings,
        )

    def free(self, request: Request) -> None:
        self.free_request_id(request.id)

    def free_request_id(self, request_id: int) -> None:
        """Release ownership even after the request has left scheduler queues."""
        if self.cache_capacity is not None:
            tables = self._group_tables.pop(request_id, {})
            self.block_pool.free_blocks(
                block for table in tables.values() for block in reversed(table) if block is not None
            )
            self._settled_tokens.pop(request_id, None)
            return
        request_blocks = self.req_to_blocks.pop(request_id, [])
        if request_blocks:
            self.block_pool.free_blocks(reversed(request_blocks))
        self.num_cached_blocks.pop(request_id, None)

    def _allocate_group_slots(self, request: Request, num_tokens: int, frontier):
        plan = self._group_allocation_plan(request, num_tokens, frontier)
        total = sum(item[2] for item in plan)
        if total > self.block_pool.get_num_free_blocks():
            return None
        allocation = self.block_pool.get_new_blocks(total)
        try:
            tables = {key: list(value) for key, value in self._group_tables.get(request.id, {}).items()}
            assigned = []
            offset = 0
            for group_id, skipped, count in plan:
                blocks = allocation.blocks[offset:offset + count]
                tables.setdefault(group_id, []).extend([None] * skipped + list(blocks))
                assigned.append((group_id, tuple(block.block_id for block in blocks)))
                offset += count
            result = KVCacheAllocationResult(
                reused_blocks=(), new_blocks=allocation.blocks,
                evicted_bindings=allocation.evicted_bindings, new_bindings=(),
                group_new_block_ids=tuple(assigned),
                zero_required_block_ids=(
                    tuple(block.block_id for block in allocation.blocks)
                    if self.cache_capacity.zero_new_blocks else ()
                ),
            )
        except Exception:
            self.block_pool.free_blocks(reversed(allocation.blocks))
            raise
        self._group_tables[request.id] = tables
        return result

    def free_block_hashes(self, request: Request) -> None:
        self.num_cached_blocks.pop(request.id, None)

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Optional

from frontier.entities.request import Request
from frontier.kv_cache.kv_cache_block import KVCacheBlock
from frontier.kv_cache.kv_cache_block_pool import BlockPool


@dataclass
class PrefixCacheStats:
    reset: bool = False
    requests: int = 0
    queries: int = 0
    hits: int = 0


class KVCacheManager:
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        enable_caching: bool,
        caching_hash_algo: str,
        num_preallocate_tokens: int,
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
        self.block_pool = BlockPool(self.num_gpu_blocks, self.enable_caching)
        self.req_to_blocks: defaultdict[int, list[KVCacheBlock]] = defaultdict(list)
        self.num_cached_blocks: dict[int, int] = {}
        self.prefix_cache_stats = PrefixCacheStats()

    @property
    def usage(self) -> float:
        return self.block_pool.get_usage()

    @property
    def num_used_blocks(self) -> int:
        return self.block_pool.get_num_used_blocks()

    def get_num_blocks_for_request(self, request: Request) -> int:
        return len(self.req_to_blocks.get(request.id, []))

    def make_prefix_cache_stats(self) -> PrefixCacheStats:
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def _get_request_block_hashes(self, request: Request) -> list[int]:
        return list(request.block_hash_ids or [])

    def get_computed_blocks(self, request: Request) -> tuple[list[KVCacheBlock], int]:
        if not self.enable_caching:
            return [], 0

        block_hashes = self._get_request_block_hashes(request)
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
    ) -> int:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")

        computed_blocks = new_computed_blocks or []
        num_computed_tokens = int(request.num_processed_tokens) + len(
            computed_blocks
        ) * self.block_size
        num_required_blocks = ceil((num_computed_tokens + num_tokens) / self.block_size)
        current_blocks = self.req_to_blocks[request.id]
        return num_required_blocks - len(current_blocks) - len(computed_blocks)

    def can_allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
    ) -> bool:
        num_new_blocks = self._get_num_new_blocks_required(
            request,
            num_tokens,
            new_computed_blocks,
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
    ) -> Optional[list[KVCacheBlock]]:
        computed_blocks = list(new_computed_blocks or [])
        if not self.can_allocate_slots(request, num_tokens, computed_blocks):
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
        )
        request_blocks = self.req_to_blocks[request.id]
        request_blocks.extend(computed_blocks)
        if num_new_blocks > 0:
            num_new_blocks = min(
                num_new_blocks + self.num_preallocate_blocks,
                self.block_pool.get_num_free_blocks(),
            )
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            request_blocks.extend(new_blocks)
        else:
            new_blocks = []

        if not self.enable_caching:
            return new_blocks

        num_computed_tokens = int(request.num_processed_tokens) + len(
            computed_blocks
        ) * self.block_size
        num_full_blocks = (num_computed_tokens + num_tokens) // self.block_size
        num_cached_blocks = self.num_cached_blocks.get(request.id, len(computed_blocks))
        self.block_pool.cache_full_blocks(
            blocks=request_blocks,
            block_hashes=self._get_request_block_hashes(request),
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
        )
        self.num_cached_blocks[request.id] = num_full_blocks
        return new_blocks

    def free(self, request: Request) -> None:
        request_blocks = self.req_to_blocks.pop(request.id, [])
        if request_blocks:
            self.block_pool.free_blocks(reversed(request_blocks))
        self.num_cached_blocks.pop(request.id, None)

    def free_block_hashes(self, request: Request) -> None:
        self.num_cached_blocks.pop(request.id, None)

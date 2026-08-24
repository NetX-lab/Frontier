from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Hashable, Optional

from frontier.kv_cache.kv_cache_block import KVCacheBlock, KVCacheBlockBinding
from frontier.kv_cache.kv_cache_block_queue import FreeKVCacheBlockQueue


@dataclass(frozen=True)
class BlockPoolAllocation:
    blocks: tuple[KVCacheBlock, ...]
    evicted_bindings: tuple[KVCacheBlockBinding, ...]


def _require_nonnegative_int(value: int, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(
            f"{field} must be a non-negative integer, got {value!r}"
        )
    return value


class BlockPool:
    def __init__(self, num_gpu_blocks: int, enable_caching: bool):
        self.num_gpu_blocks = int(num_gpu_blocks)
        self.enable_caching = bool(enable_caching)
        self.blocks = [KVCacheBlock(idx) for idx in range(self.num_gpu_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block: dict[Hashable, dict[int, KVCacheBlock]] = (
            defaultdict(dict)
        )
        self.eviction_count = 0

    def get_cached_block(self, block_hash: Hashable) -> Optional[KVCacheBlock]:
        cached = self.cached_block_hash_to_block.get(block_hash)
        if not cached:
            return None
        first_block_id = sorted(cached.keys())[0]
        return cached[first_block_id]

    def get_num_free_blocks(self) -> int:
        return int(self.free_block_queue.num_free_blocks)

    def get_num_used_blocks(self) -> int:
        return int(self.num_gpu_blocks - self.get_num_free_blocks())

    def get_usage(self) -> float:
        if self.num_gpu_blocks <= 0:
            return 0.0
        return self.get_num_used_blocks() / self.num_gpu_blocks

    def _maybe_evict_cached_block(
        self, block: KVCacheBlock
    ) -> Optional[KVCacheBlockBinding]:
        block_hash = block.block_hash
        if block_hash is None:
            return None
        cached_blocks = self.cached_block_hash_to_block.get(block_hash)
        if not cached_blocks or block.block_id not in cached_blocks:
            raise ValueError(
                "KV cache block binding is absent from the cache index: "
                f"block_id={block.block_id}, block_hash={block_hash!r}"
            )
        binding = block.binding
        if binding is None:
            raise ValueError(
                f"Cached KV block {block.block_id} has no binding identity."
            )
        del cached_blocks[block.block_id]
        if not cached_blocks:
            del self.cached_block_hash_to_block[block_hash]
        block.reset_binding()
        self.eviction_count += 1
        return binding

    def get_new_blocks(self, num_blocks: int) -> BlockPoolAllocation:
        _require_nonnegative_int(num_blocks, field="num_blocks")
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot allocate {num_blocks} KV blocks from pool with {self.get_num_free_blocks()} free blocks."
            )
        new_blocks: list[KVCacheBlock] = []
        evicted_bindings: list[KVCacheBlockBinding] = []
        while len(new_blocks) < num_blocks:
            block = self.free_block_queue.popleft()
            if self.enable_caching:
                evicted_binding = self._maybe_evict_cached_block(block)
                if evicted_binding is not None:
                    evicted_bindings.append(evicted_binding)
            block.incr_ref()
            new_blocks.append(block)
        return BlockPoolAllocation(
            blocks=tuple(new_blocks),
            evicted_bindings=tuple(evicted_bindings),
        )

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.incr_ref()

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        for block in ordered_blocks:
            block.decr_ref()
            if block.ref_cnt == 0:
                self.free_block_queue.append(block)

    def cache_full_blocks(
        self,
        *,
        blocks: list[KVCacheBlock],
        block_hashes: list[Hashable],
        num_cached_blocks: int,
        num_full_blocks: int,
        creator_request_id: int,
    ) -> tuple[KVCacheBlockBinding, ...]:
        if num_cached_blocks >= num_full_blocks:
            return ()
        upper_bound = min(num_full_blocks, len(block_hashes))
        new_bindings: list[KVCacheBlockBinding] = []
        for block_index in range(num_cached_blocks, upper_bound):
            block = blocks[block_index]
            if block.block_hash is None:
                binding = block.bind(
                    block_hashes[block_index],
                    creator_request_id=creator_request_id,
                )
                self.cached_block_hash_to_block[binding.block_hash][
                    binding.block_id
                ] = block
                new_bindings.append(binding)
        return tuple(new_bindings)

from __future__ import annotations

from typing import Optional

from frontier.kv_cache.kv_cache_block import KVCacheBlock


class FreeKVCacheBlockQueue:
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)
        self.free_list_head: Optional[KVCacheBlock] = blocks[0] if blocks else None
        self.free_list_tail: Optional[KVCacheBlock] = blocks[-1] if blocks else None
        for index, block in enumerate(blocks):
            if index > 0:
                block.prev_free_block = blocks[index - 1]
            if index < len(blocks) - 1:
                block.next_free_block = blocks[index + 1]

    def popleft(self) -> KVCacheBlock:
        if self.free_list_head is None:
            raise ValueError("No free KV cache blocks available.")
        block = self.free_list_head
        self.remove(block)
        return block

    def remove(self, block: KVCacheBlock) -> None:
        if block.prev_free_block is not None:
            block.prev_free_block.next_free_block = block.next_free_block
        if block.next_free_block is not None:
            block.next_free_block.prev_free_block = block.prev_free_block
        if block == self.free_list_head:
            self.free_list_head = block.next_free_block
        if block == self.free_list_tail:
            self.free_list_tail = block.prev_free_block
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        if self.free_list_tail is not None:
            self.free_list_tail.next_free_block = block
            block.prev_free_block = self.free_list_tail
            self.free_list_tail = block
        else:
            self.free_list_head = block
            self.free_list_tail = block
            block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks += 1

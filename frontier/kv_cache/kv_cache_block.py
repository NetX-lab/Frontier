from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional


@dataclass
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: Optional[Hashable] = None
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    def incr_ref(self) -> None:
        self.ref_cnt += 1

    def decr_ref(self) -> None:
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[Hashable]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, value: Hashable) -> None:
        if self._block_hash is not None:
            raise ValueError("KV cache block hash is already assigned.")
        self._block_hash = value

    def reset_hash(self) -> None:
        self._block_hash = None

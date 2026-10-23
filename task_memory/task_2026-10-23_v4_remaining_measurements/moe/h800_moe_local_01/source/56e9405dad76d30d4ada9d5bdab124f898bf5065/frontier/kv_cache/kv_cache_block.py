from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional


@dataclass(frozen=True)
class KVCacheBlockBinding:
    block_hash: Hashable
    block_id: int
    creator_request_id: int
    binding_epoch: int


@dataclass
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: Optional[Hashable] = None
    _creator_request_id: Optional[int] = None
    binding_epoch: int = 0
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    def incr_ref(self) -> None:
        self.ref_cnt += 1

    def decr_ref(self) -> None:
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[Hashable]:
        return self._block_hash

    @property
    def creator_request_id(self) -> Optional[int]:
        return self._creator_request_id

    @property
    def binding(self) -> Optional[KVCacheBlockBinding]:
        if self._block_hash is None:
            if self._creator_request_id is not None:
                raise ValueError(
                    "Unbound KV cache block still carries creator identity."
                )
            return None
        if self._creator_request_id is None or self.binding_epoch <= 0:
            raise ValueError("Bound KV cache block has incomplete identity.")
        return KVCacheBlockBinding(
            block_hash=self._block_hash,
            block_id=int(self.block_id),
            creator_request_id=int(self._creator_request_id),
            binding_epoch=int(self.binding_epoch),
        )

    def bind(self, block_hash: Hashable, creator_request_id: int) -> KVCacheBlockBinding:
        if self._block_hash is not None:
            raise ValueError("KV cache block hash is already assigned.")
        self.binding_epoch += 1
        self._block_hash = block_hash
        self._creator_request_id = int(creator_request_id)
        binding = self.binding
        if binding is None:
            raise ValueError("KV cache block binding was not created.")
        return binding

    def reset_binding(self) -> None:
        self._block_hash = None
        self._creator_request_id = None

from __future__ import annotations

from dataclasses import dataclass

from frontier.attention.ops import AttentionFamilySpec


@dataclass(frozen=True)
class AttentionRuntimeKVLayout:
    """Runtime KV-cache layout for one attention family on one worker."""

    family_id: str
    kv_factor: int
    runtime_num_kv_heads_per_worker: int
    runtime_head_size: int
    bytes_per_element: int = 2

    def __post_init__(self) -> None:
        if self.kv_factor <= 0:
            raise ValueError(f"kv_factor must be positive, got={self.kv_factor!r}")
        if self.runtime_num_kv_heads_per_worker <= 0:
            raise ValueError(
                "runtime_num_kv_heads_per_worker must be positive, "
                f"got={self.runtime_num_kv_heads_per_worker!r}"
            )
        if self.runtime_head_size <= 0:
            raise ValueError(
                f"runtime_head_size must be positive, got={self.runtime_head_size!r}"
            )
        if self.bytes_per_element <= 0:
            raise ValueError(
                f"bytes_per_element must be positive, got={self.bytes_per_element!r}"
            )

    @property
    def elements_per_token_per_worker(self) -> int:
        return (
            self.kv_factor
            * self.runtime_num_kv_heads_per_worker
            * self.runtime_head_size
        )

    def page_bytes(self, block_size: int) -> int:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got={block_size!r}")
        return (
            self.bytes_per_element
            * int(block_size)
            * self.elements_per_token_per_worker
        )


def get_attention_runtime_kv_layout(
    family: AttentionFamilySpec,
    *,
    runtime_num_kv_heads_per_worker: int,
    runtime_head_size: int,
    bytes_per_element: int = 2,
) -> AttentionRuntimeKVLayout:
    family.require_enabled_for_execution()

    if family.kv_factor is None:
        raise NotImplementedError(
            f"Attention family {family.family_id!r} does not declare a runtime "
            "kv_factor; cannot size its KV-cache layout."
        )

    return AttentionRuntimeKVLayout(
        family_id=family.family_id,
        kv_factor=family.kv_factor,
        runtime_num_kv_heads_per_worker=int(runtime_num_kv_heads_per_worker),
        runtime_head_size=int(runtime_head_size),
        bytes_per_element=int(bytes_per_element),
    )

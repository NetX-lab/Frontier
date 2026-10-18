from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frontier.attention.ops import AttentionFamilySpec


@dataclass(frozen=True)
class AttentionKVCacheComponent:
    """One resident cache component with equal-size pages for its owners."""

    component_id: str
    owner_layer_ids: tuple[int, ...]
    page_size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id:
            raise ValueError("component_id must be a non-empty string")
        if type(self.page_size_bytes) is not int or self.page_size_bytes <= 0:
            raise ValueError("page_size_bytes must be a positive int")
        owners = tuple(self.owner_layer_ids)
        if not owners or any(type(owner) is not int or owner < 0 for owner in owners):
            raise ValueError("owner_layer_ids must contain non-negative ints")
        if len(set(owners)) != len(owners):
            raise ValueError("duplicate owner in cache component")
        object.__setattr__(self, "owner_layer_ids", owners)


@dataclass(frozen=True)
class AttentionKVCacheCapacity:
    """Resident bytes for one common logical block across cache components."""

    family_id: str
    block_size: int
    num_layers: int
    components: tuple[AttentionKVCacheComponent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValueError("family_id must be a non-empty string")
        for name in ("block_size", "num_layers"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        components = tuple(self.components)
        if not components:
            raise ValueError("components must be non-empty")
        owners = set()
        for component in components:
            if not isinstance(component, AttentionKVCacheComponent):
                raise ValueError("components must contain cache components")
            for owner in component.owner_layer_ids:
                if owner >= self.num_layers:
                    raise ValueError("cache owner is outside the backbone layer range")
                identity = (owner, component.component_id)
                if identity in owners:
                    raise ValueError("duplicate cache component for a layer owner")
                owners.add(identity)
        object.__setattr__(self, "components", components)

    @property
    def resident_bytes_per_common_block(self) -> int:
        return sum(
            component.page_size_bytes * len(component.owner_layer_ids)
            for component in self.components
        )


def get_attention_kv_cache_capacity(
    family: AttentionFamilySpec, config: Any, *, block_size: int
) -> AttentionKVCacheCapacity:
    """Resolve static resident capacity independently of execution admission."""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive int")
    if family.kv_cache_capacity_resolver is None:
        raise NotImplementedError(
            f"Attention family {family.family_id!r} has no component capacity resolver"
        )
    capacity = family.kv_cache_capacity_resolver(config, block_size)
    if capacity.family_id != family.family_id or capacity.block_size != block_size:
        raise ValueError(
            "cache capacity does not match the requested family/block size"
        )
    if capacity.num_layers != config.num_layers:
        raise ValueError("cache capacity does not match the model backbone layer count")
    return capacity


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

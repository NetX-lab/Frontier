from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frontier.attention.ops import AttentionFamilySpec


@dataclass(frozen=True)
class AttentionKVCacheSelection:
    """Explicit runtime inputs used to select one resident cache layout."""

    device: str
    gpu_platform: str
    block_size: int
    attention_backend: str | None
    kv_cache_dtype: str | None
    indexer_kv_cache_dtype: str | None
    decode_context_parallel_size: int = 1
    attention_tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        for name in ("device", "gpu_platform"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("block_size", "decode_context_parallel_size", "attention_tensor_parallel_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")

    @classmethod
    def from_replica(
        cls, replica_config: Any, *, block_size: int
    ) -> AttentionKVCacheSelection:
        device = replica_config.device_config
        return cls(
            device=str(device.get_type()),
            gpu_platform=device.gpu_platform,
            block_size=block_size,
            attention_backend=replica_config.attention_backend,
            kv_cache_dtype=replica_config.kv_cache_dtype,
            indexer_kv_cache_dtype=replica_config.indexer_kv_cache_dtype,
            # Frontier has no context-parallel execution domain. Its current
            # parallel contract is DCP1, independently of attention TP/DP.
            decode_context_parallel_size=1,
            attention_tensor_parallel_size=replica_config.attn_tensor_parallel_size,
        )


@dataclass(frozen=True)
class AttentionKVCacheComponent:
    """One resident cache component with equal-size pages for its owners."""

    component_id: str
    owner_layer_ids: tuple[int, ...]
    page_size_bytes: int
    stored_rows: int | None = None
    compression_ratio: int = 1
    precision: str | None = None
    content_size_bytes: int | None = None

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
        for name in ("stored_rows", "content_size_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive int when supplied")
        if type(self.compression_ratio) is not int or self.compression_ratio <= 0:
            raise ValueError("compression_ratio must be a positive int")
        if self.precision is not None and (
            not isinstance(self.precision, str) or not self.precision
        ):
            raise ValueError("precision must be non-empty when supplied")
        if (
            self.content_size_bytes is not None
            and self.content_size_bytes > self.page_size_bytes
        ):
            raise ValueError("content_size_bytes cannot exceed page_size_bytes")
        object.__setattr__(self, "owner_layer_ids", owners)


@dataclass(frozen=True)
class AttentionKVCacheGroup:
    """Independent logical block table sharing one physical backing pool."""

    group_id: str
    logical_block_size: int
    component_owners: tuple[tuple[str, int], ...]
    retention: str = "full_history"
    window_tokens: int | None = None
    extra_retained_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a non-empty string")
        if type(self.logical_block_size) is not int or self.logical_block_size <= 0:
            raise ValueError("logical_block_size must be a positive int")
        owners = tuple(tuple(owner) for owner in self.component_owners)
        if not owners or any(
            len(owner) != 2 or not isinstance(owner[0], str) or not owner[0]
            or type(owner[1]) is not int or owner[1] < 0
            for owner in owners
        ):
            raise ValueError("component_owners must contain component/layer pairs")
        if len(set(owners)) != len(owners):
            raise ValueError("duplicate component owner in cache group")
        if self.retention not in {"full_history", "sliding_window"}:
            raise ValueError("unknown cache group retention")
        if self.retention == "sliding_window":
            if type(self.window_tokens) is not int or self.window_tokens <= 0:
                raise ValueError("sliding_window group requires positive window_tokens")
        elif self.window_tokens is not None or self.extra_retained_tokens != 0:
            raise ValueError("full_history group cannot declare window retention")
        if type(self.extra_retained_tokens) is not int or self.extra_retained_tokens < 0:
            raise ValueError("extra_retained_tokens must be a non-negative int")
        object.__setattr__(self, "component_owners", owners)


@dataclass(frozen=True)
class AttentionKVCacheCapacity:
    """Resident pages, optionally packed into independent allocation groups."""

    family_id: str
    block_size: int
    num_layers: int
    components: tuple[AttentionKVCacheComponent, ...]
    groups: tuple[AttentionKVCacheGroup, ...] = ()
    reserved_blocks: int = 0
    zero_new_blocks: bool = False
    fixed_bytes_per_request_slot: int = 0

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
        groups = tuple(self.groups)
        group_ids = set()
        grouped_owners = set()
        by_owner = {
            (component.component_id, owner): component
            for component in components for owner in component.owner_layer_ids
        }
        for group in groups:
            if not isinstance(group, AttentionKVCacheGroup):
                raise ValueError("groups must contain cache groups")
            if group.group_id in group_ids:
                raise ValueError("duplicate cache group ID")
            group_ids.add(group.group_id)
            if self.block_size % group.logical_block_size:
                raise ValueError("scheduler block_size must be divisible by every group")
            for component_id, owner in group.component_owners:
                identity = (owner, component_id)
                if identity not in owners:
                    raise ValueError("cache group contains an unknown component owner")
                if identity in grouped_owners:
                    raise ValueError("cache component owner occurs in multiple groups")
                component = by_owner[(component_id, owner)]
                if (
                    component.stored_rows is None or component.precision is None
                    or component.content_size_bytes is None
                ):
                    raise ValueError(
                        "grouped components require row, precision and content metadata"
                    )
                if (
                    component.stored_rows * component.compression_ratio
                    != group.logical_block_size
                ):
                    raise ValueError(
                        "component stored rows do not match its logical group block"
                    )
                grouped_owners.add(identity)
        if groups and grouped_owners != owners:
            raise ValueError("cache groups must partition every component owner")
        if type(self.reserved_blocks) is not int or self.reserved_blocks < 0:
            raise ValueError("reserved_blocks must be a non-negative int")
        if type(self.fixed_bytes_per_request_slot) is not int or self.fixed_bytes_per_request_slot < 0:
            raise ValueError("fixed_bytes_per_request_slot must be a non-negative int")
        if type(self.zero_new_blocks) is not bool:
            raise ValueError("zero_new_blocks must be a bool")
        if not groups and (self.reserved_blocks or self.zero_new_blocks):
            raise ValueError("pool reservations and zeroing require grouped capacity")
        object.__setattr__(self, "groups", groups)

    @property
    def is_grouped(self) -> bool:
        return bool(self.groups)

    def get_group_size_bytes(self, group: AttentionKVCacheGroup) -> int:
        components = {
            (component.component_id, owner): component
            for component in self.components for owner in component.owner_layer_ids
        }
        return sum(
            components[identity].page_size_bytes for identity in group.component_owners
        )

    @property
    def physical_block_size_bytes(self) -> int:
        if self.groups:
            return max(self.get_group_size_bytes(group) for group in self.groups)
        return self.resident_bytes_per_common_block

    @property
    def resident_bytes_per_common_block(self) -> int:
        if self.groups:
            raise ValueError(
                "Grouped cache has no additive common block; use physical_block_size_bytes"
            )
        return sum(
            component.page_size_bytes * len(component.owner_layer_ids)
            for component in self.components
        )


def get_attention_kv_cache_capacity(
    family: AttentionFamilySpec, config: Any, *, block_size: int,
    selection: AttentionKVCacheSelection | None = None,
) -> AttentionKVCacheCapacity:
    """Resolve static resident capacity independently of execution admission."""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive int")
    if family.kv_cache_capacity_resolver is None:
        raise NotImplementedError(
            f"Attention family {family.family_id!r} has no component capacity resolver"
        )
    if selection is not None and selection.block_size != block_size:
        raise ValueError("cache selection does not match scheduler block_size")
    capacity = family.kv_cache_capacity_resolver(config, block_size, selection)
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

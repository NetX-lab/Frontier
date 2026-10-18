"""Step resident page arrays and source transfer views, without an allocator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Any

from frontier.attention.memory import (
    AttentionKVCacheCapacity,
    AttentionKVCacheComponent,
    AttentionKVCacheGroup,
    AttentionKVCacheSelection,
    AttentionRuntimeKVLayout,
)
from frontier.attention.step5 import (
    resolve_step5_attention_geometry,
    resolve_step5_model_precision,
    validate_step5_config,
)
from frontier.config.precision_type import PrecisionType


@dataclass(frozen=True)
class Step5KVCacheDescription:
    """Per-rank backbone arrays for independently supplied SWA and sparse pages.

    Counts are per owner layer and per TP rank. Padding scratch is counted in
    max_num_seqs slots, independently of the resident page pools. Unknown
    workspaces, MTP state and speculative request journals are excluded. This
    is not a complete native memory measurement or a common-slot capacity
    denominator. A source transfer view does not establish actual wire bytes,
    state integrity, readiness or lifetime qualification.
    """

    num_layers: int
    tensor_parallel_size: int
    components: tuple[AttentionKVCacheComponent, ...]
    resident_page_counts: tuple[int, ...]
    # FP8 mean bytes for one sparse-layer page on one TP rank.
    summary_transfer_page_size_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "num_layers", "tensor_parallel_size", "summary_transfer_page_size_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if not isinstance(self.components, (tuple, list)) or not isinstance(
            self.resident_page_counts, (tuple, list)
        ):
            raise ValueError("components and resident_page_counts must be sequences")
        components = tuple(self.components)
        counts = tuple(self.resident_page_counts)
        if not components or len(components) != len(counts):
            raise ValueError("resident_page_counts must contain one count per component")
        components_by_id = {}
        for component, pages in zip(components, counts):
            if not isinstance(component, AttentionKVCacheComponent):
                raise ValueError("components must contain cache components")
            if type(pages) is not int or pages < 0:
                raise ValueError("resident_page_counts must contain non-negative ints")
            if any(owner >= self.num_layers for owner in component.owner_layer_ids):
                raise ValueError("cache owner is outside the backbone layer range")
            if component.component_id in components_by_id:
                raise ValueError("duplicate Step cache component")
            components_by_id[component.component_id] = (component, pages)
        if set(components_by_id) != {
            "swa_raw_kv", "sparse_raw_kv", "region_summary", "open_region_state",
            "padding_summary_scratch",
        }:
            raise ValueError("Step cache description requires exactly its five page components")
        sliding, _ = components_by_id["swa_raw_kv"]
        sparse, sparse_pages = components_by_id["sparse_raw_kv"]
        sliding_owners = set(sliding.owner_layer_ids)
        sparse_owners = set(sparse.owner_layer_ids)
        if sliding_owners & sparse_owners or len(sliding_owners | sparse_owners) != self.num_layers:
            raise ValueError("raw KV owners must partition the backbone layers")
        for name in ("region_summary", "open_region_state", "padding_summary_scratch"):
            component, pages = components_by_id[name]
            if set(component.owner_layer_ids) != sparse_owners:
                raise ValueError("summary and tail owners must match sparse raw KV owners")
            if name != "padding_summary_scratch" and pages != sparse_pages:
                raise ValueError("summary and tail page counts must match sparse raw KV pages")
        summary, _ = components_by_id["region_summary"]
        tail, _ = components_by_id["open_region_state"]
        scratch, scratch_slots = components_by_id["padding_summary_scratch"]
        if scratch_slots <= 0:
            raise ValueError("padding summary scratch must have positive max_num_seqs slots")
        if self.summary_transfer_page_size_bytes != summary.page_size_bytes:
            raise ValueError("summary transfer page size must match the FP8 mean component")
        if scratch.page_size_bytes != summary.page_size_bytes + tail.page_size_bytes:
            raise ValueError("padding summary scratch must contain one mean page and tail state")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "resident_page_counts", counts)

    @property
    def resident_bytes_per_rank(self) -> int:
        return sum(
            component.page_size_bytes * len(component.owner_layer_ids) * pages
            for component, pages in zip(self.components, self.resident_page_counts)
        )

    @property
    def aggregate_resident_bytes(self) -> int:
        """Sum the described arrays across TP ranks, including replicated KV."""
        return self.tensor_parallel_size * self.resident_bytes_per_rank

    def summary_transfer_view_bytes(self, transferred_sparse_pages: int) -> int:
        """Sum full-page FP8 mean views over all sparse layers and TP ranks.

        transferred_sparse_pages is the same per-layer, per-rank page count for
        all sparse owners. Tail state and padding scratch are outside this
        view. Page eligibility, state integrity, readiness and lifetime remain
        unqualified; its size does not establish actual wire bytes or qualify PD.
        """
        if type(transferred_sparse_pages) is not int or transferred_sparse_pages < 0:
            raise ValueError("transferred_sparse_pages must be a non-negative int")
        summary_index = next(
            index for index, component in enumerate(self.components)
            if component.component_id == "region_summary"
        )
        summary = self.components[summary_index]
        if transferred_sparse_pages > self.resident_page_counts[summary_index]:
            raise ValueError("summary transfer cannot exceed the allocated sparse page pool")
        return (
            transferred_sparse_pages * self.summary_transfer_page_size_bytes
            * len(summary.owner_layer_ids) * self.tensor_parallel_size
        )


def describe_step5_kv_cache(
    config: Any, *, tensor_parallel_size: int, block_size: int,
    sparse_pages: int, sliding_pages: int, max_num_seqs: int,
    cache_dtype: str,
) -> Step5KVCacheDescription:
    """Describe pinned native page arrays with MTP and speculative journals off.

    cache_dtype is a resolved BF16/E4M3 cache format, or auto with BF16 model
    metadata. Weight FP8 precision does not determine the raw KV cache dtype.
    """
    validate_step5_config(config)
    for name, value, allow_zero in (
        ("block_size", block_size, False),
        ("sparse_pages", sparse_pages, True),
        ("sliding_pages", sliding_pages, True),
        ("max_num_seqs", max_num_seqs, False),
    ):
        if type(value) is not int or value < (0 if allow_zero else 1):
            raise ValueError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} int")
    region_tokens = config.sparse_config["region_block_size"]
    if block_size % region_tokens:
        raise ValueError("Step block_size must align to the configured region tokens")
    if not isinstance(cache_dtype, str):
        raise ValueError("Step cache_dtype must name a supported native cache format")
    dtype = cache_dtype.strip().lower().removeprefix("torch.")
    if dtype in ("fp8", "fp8_e4m3"):
        cache_precision = PrecisionType.FP8
    else:
        cache_precision = (
            resolve_step5_model_precision(config) if dtype == "auto"
            else PrecisionType.from_torch_dtype(cache_dtype)
        )
        if cache_precision is not PrecisionType.BF16:
            raise ValueError("Step cache_dtype requires bfloat16 or E4M3 FP8")
    sparse_ids = tuple(i for i, value in enumerate(config.layer_types) if value == "full_attention")
    sliding_ids = tuple(i for i, value in enumerate(config.layer_types) if value == "sliding_attention")
    components: list[AttentionKVCacheComponent] = []
    counts: list[int] = []
    for layer_ids, name, pages in (
        (sliding_ids, "swa_raw_kv", sliding_pages),
        (sparse_ids, "sparse_raw_kv", sparse_pages),
    ):
        page_sizes = set()
        for layer_id in layer_ids:
            geometry = resolve_step5_attention_geometry(
                config, layer_id, tensor_parallel_size=tensor_parallel_size,
            )
            layout = AttentionRuntimeKVLayout(
                family_id="step5_attention", kv_factor=2,
                runtime_num_kv_heads_per_worker=geometry.local_kv_heads,
                runtime_head_size=geometry.head_dim,
                bytes_per_element=int(cache_precision.bytes_per_element),
            )
            page_sizes.add(layout.page_bytes(block_size))
        if len(page_sizes) != 1:
            raise NotImplementedError("Step component descriptions require equal page size within each attention variant")
        components.append(AttentionKVCacheComponent(name, layer_ids, page_sizes.pop()))
        counts.append(pages)
    proxy_dim = config.sparse_config["proxy_dim"]
    # sparse_summary_cache.py:25-43, 214-219: E4M3 mean bytes have no scale array.
    summary_page_bytes = (
        block_size // region_tokens * proxy_dim * int(PrecisionType.FP8.bytes_per_element)
    )
    # Per page: FP32 numerator/denominator[proxy_dim], maximum[1], int64 last_slot.
    tail_page_bytes = (2 * proxy_dim + 1) * int(PrecisionType.FP32.bytes_per_element) + 8
    for name, page_bytes, pages in (
        ("region_summary", summary_page_bytes, sparse_pages),
        ("open_region_state", tail_page_bytes, sparse_pages),
        ("padding_summary_scratch", summary_page_bytes + tail_page_bytes, max_num_seqs),
    ):
        components.append(AttentionKVCacheComponent(name, sparse_ids, page_bytes))
        counts.append(pages)
    return Step5KVCacheDescription(
        config.num_layers, tensor_parallel_size, tuple(components), tuple(counts), summary_page_bytes,
    )


def resolve_step5_kv_cache_capacity(
    config: Any, block_size: int, selection: AttentionKVCacheSelection | None,
) -> AttentionKVCacheCapacity:
    """Bind equal raw Hopper pages and one sparse sidecar to the existing pool.

    Native Step selects STEP5_FLASH_ATTN for SWA and STEP5_DSA for sparse
    layers. Both inherit the 16-token kernel granularity at head size 192.
    Equal unpadded raw pages retain the primary span during page unification;
    their final span and scheduler LCM are therefore equal in this scope.
    This source-derived binding is not native allocation or capacity evidence.
    """
    if not isinstance(selection, AttentionKVCacheSelection):
        raise NotImplementedError("Step cache requires explicit runtime cache selection")
    if selection.gpu_platform != "cuda" or selection.device not in ("h800", "h200"):
        raise NotImplementedError("Step structural cache binding requires H800 or H200 Hopper")
    if selection.attention_backend != "STEP5_DSA":
        raise NotImplementedError("Step cache requires the native STEP5_DSA/STEP5_FLASH_ATTN pair")
    if selection.decode_context_parallel_size != 1:
        raise NotImplementedError("Step cache binding requires decode context parallel size 1")
    if selection.indexer_kv_cache_dtype is not None:
        raise NotImplementedError("Step summary E4M3 storage has no configurable indexer cache dtype")
    if block_size % 16:
        raise NotImplementedError("Step Hopper kernel pages require a multiple of 16 tokens")
    if selection.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        raise NotImplementedError("Step cache requires explicit native BF16 or E4M3 cache format")
    description = describe_step5_kv_cache(
        config, tensor_parallel_size=selection.attention_tensor_parallel_size,
        block_size=block_size, sparse_pages=1, sliding_pages=1, max_num_seqs=1,
        cache_dtype=selection.kv_cache_dtype,
    )
    swa, sparse, summary, tail, scratch = description.components
    if swa.page_size_bytes != sparse.page_size_bytes:
        raise NotImplementedError("Step one-pool binding requires equal unpadded raw pages")
    # With this fixed head geometry neither FA4 head512 nor head256 exceptions apply.
    for layer_id in range(config.num_layers):
        geometry = resolve_step5_attention_geometry(
            config, layer_id, tensor_parallel_size=selection.attention_tensor_parallel_size,
        )
        if geometry.head_dim != 192:
            raise NotImplementedError("Step Hopper page binding requires head dimension 192")
    cache_precision = "fp8" if selection.kv_cache_dtype.startswith("fp8") else "bf16"
    components = (
        replace(swa, stored_rows=block_size, precision=cache_precision,
                content_size_bytes=swa.page_size_bytes),
        replace(sparse, stored_rows=block_size, precision=cache_precision,
                content_size_bytes=sparse.page_size_bytes),
        replace(summary, stored_rows=block_size // config.sparse_config["region_block_size"],
                compression_ratio=config.sparse_config["region_block_size"], precision="fp8",
                content_size_bytes=summary.page_size_bytes),
        replace(tail, stored_rows=1, compression_ratio=block_size, precision="fp32/int64",
                content_size_bytes=tail.page_size_bytes),
    )
    group_size = min(len(swa.owner_layer_ids), len(sparse.owner_layer_ids))
    if len(swa.owner_layer_ids) % group_size or len(sparse.owner_layer_ids) != group_size:
        raise NotImplementedError("Step one-pool binding requires complete equal-size raw groups")
    swa_group_count = ceil(len(swa.owner_layer_ids) / group_size)
    groups = tuple(
        AttentionKVCacheGroup(
            group_id=f"swa_{index}", logical_block_size=block_size,
            component_owners=tuple((swa.component_id, layer) for layer in swa.owner_layer_ids[index::swa_group_count]),
            retention="sliding_window", window_tokens=config.sliding_window,
        )
        for index in range(swa_group_count)
    ) + (AttentionKVCacheGroup(
        group_id="sparse", logical_block_size=block_size,
        component_owners=tuple(
            (component.component_id, layer)
            for component in components[1:] for layer in sparse.owner_layer_ids
        ),
    ),)
    # All raw group views begin at byte zero of the same backing allocation.
    # The sparse group's sidepages grow with N; scratch grows only with max_num_seqs.
    return AttentionKVCacheCapacity(
        family_id="step5_attention", block_size=block_size, num_layers=config.num_layers,
        components=components, groups=groups, reserved_blocks=1,
        fixed_bytes_per_request_slot=len(scratch.owner_layer_ids) * scratch.page_size_bytes,
    )

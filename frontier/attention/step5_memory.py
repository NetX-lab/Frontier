"""Step resident page arrays and source transfer views, without an allocator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frontier.attention.memory import AttentionKVCacheComponent, AttentionRuntimeKVLayout
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

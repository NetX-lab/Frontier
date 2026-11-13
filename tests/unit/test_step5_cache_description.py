"""Known Step page-array formulas, independent of native execution qualification."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from frontier.attention.memory import AttentionKVCacheComponent
from frontier.attention.step5_memory import (
    Step5KVCacheDescription,
    describe_step5_kv_cache,
)
from frontier.config.model_config import BaseModelConfig
from frontier.config.precision_type import PrecisionType
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    return BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")


@pytest.fixture
def allocation():
    return dict(
        tensor_parallel_size=8, block_size=64, sparse_pages=19,
        sliding_pages=3, max_num_seqs=7, cache_dtype="bfloat16",
    )


@pytest.fixture
def description(model, allocation):
    return describe_step5_kv_cache(model, **allocation)


def test_known_bf16_page_arrays_and_owner_counts(description):
    swa, sparse, mean, tail, scratch = description.components
    sparse_layers = tuple(range(3, 92, 4))
    sliding_layers = tuple(layer for layer in range(92) if layer % 4 != 3)
    assert description.num_layers == 92
    assert description.tensor_parallel_size == 8
    assert swa == AttentionKVCacheComponent("swa_raw_kv", sliding_layers, 49152)
    assert sparse == AttentionKVCacheComponent("sparse_raw_kv", sparse_layers, 49152)
    # One E4M3 mean per eight-token region; no per-region scale tensor.
    assert mean == AttentionKVCacheComponent("region_summary", sparse_layers, 2048)
    # Two 256-wide FP32 arrays, one FP32 maximum and one int64 last slot.
    assert tail == AttentionKVCacheComponent("open_region_state", sparse_layers, 2060)
    assert scratch == AttentionKVCacheComponent("padding_summary_scratch", sparse_layers, 4108)
    assert description.resident_page_counts == (3, 19, 19, 19, 7)
    expected_raw = (69 * 3 + 23 * 19) * (2 * 64 * 1 * 192 * 2)
    expected_side = 23 * (19 + 7) * (64 // 8 * 256 + (2 * 256 + 1) * 4 + 8)
    assert description.resident_bytes_per_rank == expected_raw + expected_side == 34110472
    assert description.aggregate_resident_bytes == 8 * (expected_raw + expected_side)


@pytest.mark.parametrize("block_size", [8, 16, 64, 128])
@pytest.mark.parametrize("cache_dtype, element_bytes", [("bfloat16", 2), ("fp8", 1), ("fp8_e4m3", 1)])
def test_page_formulas_keep_raw_tokens_separate_from_region_means(
    model, allocation, block_size, cache_dtype, element_bytes,
):
    allocation.update(block_size=block_size, cache_dtype=cache_dtype)
    description = describe_step5_kv_cache(model, **allocation)
    swa, sparse, mean, tail, scratch = description.components
    raw_page = 2 * block_size * 1 * 192 * element_bytes
    assert swa.page_size_bytes == sparse.page_size_bytes == raw_page
    assert sparse.page_size_bytes != raw_page // 8
    assert mean.page_size_bytes == block_size // 8 * 256
    assert tail.page_size_bytes == (256 + 256 + 1) * 4 + 8
    assert scratch.page_size_bytes == block_size // 8 * 256 + 2060


@pytest.mark.parametrize("tp_size, local_kv_heads, replication", [(1, 4, 1), (2, 2, 1), (4, 1, 1), (8, 1, 2), (16, 1, 4)])
def test_tp_resident_bytes_include_kv_and_summary_replication(
    model, allocation, tp_size, local_kv_heads, replication,
):
    allocation["tensor_parallel_size"] = tp_size
    description = describe_step5_kv_cache(model, **allocation)
    raw_page = 2 * 64 * local_kv_heads * 192 * 2
    logical_raw = (69 * 3 + 23 * 19) * 2 * 64 * 4 * 192 * 2
    side_per_rank = 23 * (19 + 7) * (8 * 256 + 513 * 4 + 8)
    assert description.components[0].page_size_bytes == raw_page
    assert description.components[1].page_size_bytes == raw_page
    assert description.aggregate_resident_bytes == logical_raw * replication + tp_size * side_per_rank


def test_tp8_raw_kv_has_one_head_per_rank_and_two_copies(model, allocation):
    tp8 = describe_step5_kv_cache(model, **allocation)
    allocation["tensor_parallel_size"] = 4
    tp4 = describe_step5_kv_cache(model, **allocation)
    assert tp8.components[:2] == tp4.components[:2]
    assert tp8.components[1].page_size_bytes == 2 * 64 * 192 * 2
    raw_per_rank = (69 * 3 + 23 * 19) * 49152
    assert raw_per_rank * 8 == 2 * (raw_per_rank * 4)


@pytest.mark.parametrize("sliding_pages, sparse_pages", [(0, 0), (0, 19), (3, 0), (1, 7), (20, 2)])
def test_independent_sparse_and_sliding_page_counts(model, allocation, sliding_pages, sparse_pages):
    allocation.update(sliding_pages=sliding_pages, sparse_pages=sparse_pages)
    description = describe_step5_kv_cache(model, **allocation)
    assert description.resident_page_counts == (sliding_pages, sparse_pages, sparse_pages, sparse_pages, 7)
    expected = 69 * sliding_pages * 49152 + 23 * sparse_pages * 49152
    expected += 23 * (sparse_pages + 7) * 4108
    assert description.resident_bytes_per_rank == expected
    assert description.summary_transfer_view_bytes(sparse_pages) == sparse_pages * 2048 * 23 * 8


def test_swa_pages_do_not_follow_window_visibility(model, allocation):
    small_window = describe_step5_kv_cache(model, **allocation)
    model.sliding_window = 1024
    large_window = describe_step5_kv_cache(model, **allocation)
    assert small_window == large_window
    assert small_window.resident_page_counts[0] == 3


def test_max_num_seqs_changes_only_known_summary_scratch(model, allocation):
    first = describe_step5_kv_cache(model, **allocation)
    allocation["max_num_seqs"] = 11
    second = describe_step5_kv_cache(model, **allocation)
    assert first.components == second.components
    assert first.resident_page_counts[:-1] == second.resident_page_counts[:-1]
    assert second.resident_bytes_per_rank - first.resident_bytes_per_rank == 23 * (11 - 7) * 4108
    assert first.summary_transfer_view_bytes(19) == second.summary_transfer_view_bytes(19)


@pytest.mark.parametrize("cache_dtype", ["auto", "bf16", "torch.bfloat16", " BF16 "])
def test_bf16_cache_aliases_use_resolved_bf16_metadata(model, allocation, description, cache_dtype):
    allocation["cache_dtype"] = cache_dtype
    assert describe_step5_kv_cache(model, **allocation) == description


def test_auto_cache_uses_existing_runtime_precision_getter(model, allocation, description, monkeypatch):
    monkeypatch.delattr(model, "torch_dtype")
    monkeypatch.setattr(model, "get_default_precision", lambda: PrecisionType.BF16)
    allocation["cache_dtype"] = "auto"
    assert describe_step5_kv_cache(model, **allocation) == description


@pytest.mark.parametrize("dtype", ["bfloat16", "torch.bfloat16"])
def test_auto_cache_uses_profiling_dtype_without_torch(
    model, allocation, description, monkeypatch, dtype,
):
    monkeypatch.setattr(ProfilingModelConfig, "_import_torch", staticmethod(lambda: None))
    profiling = ProfilingModelConfig.from_model_name("stepfun-ai/Step-5-Preview")
    assert not hasattr(profiling, "torch_dtype")
    assert not hasattr(profiling, "get_default_precision")
    profiling._dtype = dtype
    allocation["cache_dtype"] = "auto"
    assert describe_step5_kv_cache(profiling, **allocation) == description


@pytest.mark.parametrize("cache_dtype", ["float16", "fp16", "half", "float32", "fp32", "torch.float16", "fp8_e5m2", "float8_e4m3fn", "uint8", "int8", "", None, True])
def test_rejects_cache_formats_outside_native_supported_path(model, allocation, cache_dtype):
    allocation["cache_dtype"] = cache_dtype
    with pytest.raises(ValueError):
        describe_step5_kv_cache(model, **allocation)


@pytest.mark.parametrize("name, value", [
    ("block_size", 0), ("block_size", -8), ("block_size", True),
    ("block_size", 16.0), ("block_size", 15),
    ("sparse_pages", -1), ("sparse_pages", True), ("sparse_pages", 1.0),
    ("sliding_pages", -1), ("sliding_pages", False), ("sliding_pages", 1.0),
    ("max_num_seqs", 0), ("max_num_seqs", -1), ("max_num_seqs", True),
    ("max_num_seqs", 2.0),
    ("tensor_parallel_size", 0), ("tensor_parallel_size", -1),
    ("tensor_parallel_size", True), ("tensor_parallel_size", 8.0),
    ("tensor_parallel_size", 3),
])
def test_invalid_allocation_dimensions_fail_before_description(model, allocation, name, value):
    allocation[name] = value
    with pytest.raises(ValueError, match=name):
        describe_step5_kv_cache(model, **allocation)


def test_tp32_rejects_unsupported_local_q_heads_per_kv_group(model, allocation):
    allocation["tensor_parallel_size"] = 32
    with pytest.raises(ValueError, match="local Q heads per KV group"):
        describe_step5_kv_cache(model, **allocation)


def test_invalid_kv_replication_is_rejected_by_public_geometry(model, allocation):
    model.num_kv_heads = 3
    with pytest.raises(ValueError, match="KV groups|local KV"):
        describe_step5_kv_cache(model, **allocation)


@pytest.mark.parametrize("layer_type", ["full_attention", "sliding_attention"])
def test_description_requires_both_native_cache_variants(model, allocation, layer_type):
    model.layer_types = (layer_type,) * 92
    with pytest.raises(ValueError, match="both SWA and sparse"):
        describe_step5_kv_cache(model, **allocation)


@pytest.mark.parametrize("transferred_pages", [0, 1, 3, 19])
def test_summary_transfer_counts_every_layer_and_tp_rank_owner(description, transferred_pages):
    assert description.summary_transfer_page_size_bytes == 64 // 8 * 256
    assert description.summary_transfer_view_bytes(transferred_pages) == transferred_pages * 2048 * 23 * 8


@pytest.mark.parametrize("transferred_pages", [-1, True, 1.0, None, 20])
def test_summary_transfer_rejects_invalid_counts_and_sparse_pool_overflow(description, transferred_pages):
    with pytest.raises(ValueError, match="transferred_sparse_pages|sparse page pool"):
        description.summary_transfer_view_bytes(transferred_pages)


def test_description_does_not_require_components_in_a_fixed_tuple_order(description):
    order = (1, 0, 4, 3, 2)
    reordered = replace(
        description,
        components=tuple(description.components[index] for index in order),
        resident_page_counts=tuple(description.resident_page_counts[index] for index in order),
    )
    assert reordered.resident_bytes_per_rank == description.resident_bytes_per_rank
    assert reordered.summary_transfer_view_bytes(19) == 19 * 2048 * 23 * 8


@pytest.mark.parametrize("counts", [(), (3, 19, 19, 19), (3, 19, 19, 19, 7, 1), None])
def test_description_constructor_rejects_missing_or_extra_counts(description, counts):
    with pytest.raises(ValueError, match="resident_page_counts"):
        replace(description, resident_page_counts=counts)


@pytest.mark.parametrize("counts", [(-1, 19, 19, 19, 7), (True, 19, 19, 19, 7), (3, 19.0, 19, 19, 7), (3, 19, 19, 19, -1), (3, 19, 19, 19, 0)])
def test_description_constructor_rejects_invalid_page_or_scratch_counts(description, counts):
    with pytest.raises(ValueError, match="resident_page_counts|max_num_seqs"):
        replace(description, resident_page_counts=counts)


@pytest.mark.parametrize("counts", [(3, 19, 18, 19, 7), (3, 19, 19, 18, 7)])
def test_description_constructor_rejects_side_pages_unrelated_to_sparse_pool(description, counts):
    with pytest.raises(ValueError, match="page counts must match sparse"):
        replace(description, resident_page_counts=counts)


@pytest.mark.parametrize("name, value", [
    ("num_layers", 0), ("num_layers", True), ("num_layers", 92.0),
    ("tensor_parallel_size", 0), ("tensor_parallel_size", True),
    ("summary_transfer_page_size_bytes", 0),
    ("summary_transfer_page_size_bytes", 2048.0),
])
def test_description_constructor_rejects_invalid_owner_dimensions(description, name, value):
    with pytest.raises(ValueError, match=name):
        replace(description, **{name: value})


def test_description_constructor_rejects_duplicate_and_missing_components(description):
    with pytest.raises(ValueError, match="duplicate Step cache component"):
        replace(description, components=(*description.components[:-1], description.components[0]))
    with pytest.raises(ValueError, match="five page components"):
        replace(description, components=description.components[:-1], resident_page_counts=description.resident_page_counts[:-1])
    with pytest.raises(ValueError, match="cache components"):
        replace(description, components=(*description.components[:-1], object()))
    with pytest.raises(ValueError, match="resident_page_counts"):
        replace(description, components=(), resident_page_counts=())


def test_description_constructor_rejects_nonbackbone_and_conflicting_owners(description):
    swa = description.components[0]
    with pytest.raises(ValueError, match="backbone layer range"):
        replace(description, components=(replace(swa, owner_layer_ids=(92,)), *description.components[1:]))
    with pytest.raises(ValueError, match="partition the backbone"):
        replace(description, components=(replace(swa, owner_layer_ids=(0, 3)), *description.components[1:]))
    mean = replace(description.components[2], owner_layer_ids=(0,))
    with pytest.raises(ValueError, match="owners must match sparse"):
        replace(description, components=(*description.components[:2], mean, *description.components[3:]))


def test_description_constructor_rejects_wrong_summary_or_scratch_extent(description):
    with pytest.raises(ValueError, match="transfer page size"):
        replace(description, summary_transfer_page_size_bytes=1)
    scratch = replace(description.components[-1], page_size_bytes=2048)
    with pytest.raises(ValueError, match="one mean page and tail state"):
        replace(description, components=(*description.components[:-1], scratch))


def test_description_freezes_input_sequences(description):
    components = list(description.components)
    counts = list(description.resident_page_counts)
    copied = replace(description, components=components, resident_page_counts=counts)
    components.clear()
    counts[0] = 100
    assert copied == description
    with pytest.raises(FrozenInstanceError):
        copied.tensor_parallel_size = 4


def test_backbone_arrays_exclude_unknown_workspaces_and_mtp_state(model, allocation, description):
    assert model.num_nextn_predict_layers == 3
    model.mtp_layer_types = ("sliding_attention",) * 3
    assert describe_step5_kv_cache(model, **allocation) == description
    assert {component.component_id for component in description.components} == {
        "swa_raw_kv", "sparse_raw_kv", "region_summary", "open_region_state", "padding_summary_scratch",
    }
    assert all(owner < 92 for component in description.components for owner in component.owner_layer_ids)
    assert not hasattr(description, "resident_bytes_per_common_block")
    assert not hasattr(description, "wire_bytes")
    assert not hasattr(description, "pd_supported")
    assert "excluded" in Step5KVCacheDescription.__doc__
    assert "qualification" in Step5KVCacheDescription.__doc__


def test_summary_view_excludes_tail_scratch_and_raw_kv(model, allocation, description):
    allocation.update(cache_dtype="fp8_e4m3", max_num_seqs=11, sliding_pages=20)
    fp8 = describe_step5_kv_cache(model, **allocation)
    assert fp8.resident_bytes_per_rank != description.resident_bytes_per_rank
    assert fp8.summary_transfer_view_bytes(19) == description.summary_transfer_view_bytes(19)
    assert fp8.summary_transfer_view_bytes(19) == 19 * 8 * 256 * 23 * 8
    assert "actual wire bytes" in Step5KVCacheDescription.summary_transfer_view_bytes.__doc__
    assert "qualify PD" in Step5KVCacheDescription.summary_transfer_view_bytes.__doc__

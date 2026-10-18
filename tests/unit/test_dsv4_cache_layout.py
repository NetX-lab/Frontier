"""Source-defined V4 cache storage; no native allocation or execution claims."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
from frontier.attention.memory import (
    AttentionKVCacheSelection,
    get_attention_kv_cache_capacity,
)
from frontier.attention.ops import AttentionFamilySpec
from frontier.config import (
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.config.utils import dataclass_to_dict
from frontier.entities.replica import Replica
from frontier.errors import FrontierMemoryOOMError
from frontier.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType


@pytest.fixture
def replica_config(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    return ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h800",
        network_device="h800_dgx", attn_tensor_parallel_size=8, attn_dp=2,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
        attention_backend="FLASHMLA_SPARSE_DSV4", kv_cache_dtype="fp8",
        indexer_kv_cache_dtype="fp8",
    )


def _capacity(replica_config, *, model=None, selection=None):
    return get_attention_kv_cache_capacity(
        DEEPSEEK_V4_ATTENTION_FAMILY, model or replica_config.model_config,
        block_size=256,
        selection=selection or AttentionKVCacheSelection.from_replica(
            replica_config, block_size=256,
        ),
    )


def _planner(replica_config):
    return MemoryPlanner(
        replica_config,
        Replica(replica_config, SyntheticRequestGeneratorConfig(), ClusterType.MONOLITHIC),
        ClusterType.MONOLITHIC,
    )


def _predictor(replica_config, tmp_path):
    return RandomForrestExecutionTimePredictor(
        RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True),
        replica_config, VllmV1SchedulerConfig(block_size=256),
        MetricsConfig(cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "outputs")),
    )


def test_original_pages_and_packed_groups_match_independent_arithmetic(replica_config):
    capacity = _capacity(replica_config)
    csa = tuple(range(2, 43, 2))
    hca = tuple(range(3, 43, 2))
    # Separate integer oracle: source rows times content width rounded to 576B.
    expected = {
        "main_csa": (csa, 64, 4, 64 * (448 + 128 + 8), "fp8_ds_mla"),
        "main_hca": (hca, 2, 128, 2 * (448 + 128 + 8), "fp8_ds_mla"),
        "indexer_csa": (csa, 64, 4, 64 * (128 + 4), "fp8_fp32_scale"),
        "swa": (tuple(range(43)), 64, 1, 64 * (448 + 128 + 8), "fp8_ds_mla"),
        "main_csa_state": (csa, 4, 1, 4 * 2048 * 4, "float32"),
        "indexer_csa_state": (csa, 4, 1, 4 * 512 * 4, "float32"),
        "main_hca_state": (hca, 8, 1, 8 * 1024 * 4, "float32"),
    }
    assert capacity.num_layers == 43
    assert set(expected) == {component.component_id for component in capacity.components}
    for component in capacity.components:
        owners, rows, ratio, content, precision = expected[component.component_id]
        assert component.owner_layer_ids == owners
        assert component.stored_rows == rows
        assert component.compression_ratio == ratio
        assert component.content_size_bytes == content
        assert component.page_size_bytes == (content + 575) // 576 * 576
        assert component.precision == precision
    assert [capacity.get_group_size_bytes(group) for group in capacity.groups] == [
        1002240, 823680, 786240, 870912, 656640,
    ]
    assert [group.logical_block_size for group in capacity.groups] == [256, 64, 64, 4, 8]
    assert [group.window_tokens for group in capacity.groups] == [None, 128, 128, 8, 128]
    assert all(group.extra_retained_tokens == 0 for group in capacity.groups)
    assert capacity.groups[1].component_owners == tuple(("swa", i) for i in range(0, 43, 2))
    assert capacity.groups[2].component_owners == tuple(("swa", i) for i in range(1, 43, 2))
    assert capacity.physical_block_size_bytes == 1002240
    assert capacity.reserved_blocks == 1
    assert capacity.zero_new_blocks is True
    all_pairs = [(component.component_id, owner) for component in capacity.components for owner in component.owner_layer_ids]
    grouped_pairs = [pair for group in capacity.groups for pair in group.component_owners]
    assert len(grouped_pairs) == len(set(grouped_pairs)) == len(all_pairs)
    assert set(grouped_pairs) == set(all_pairs)
    assert max(owner for _, owner in grouped_pairs) == 42
    with pytest.raises(ValueError, match="no additive common block"):
        _ = capacity.resident_bytes_per_common_block
    with pytest.raises(FrozenInstanceError):
        capacity.groups[0].logical_block_size = 1


def test_cache_storage_follows_configured_index_width_and_window(replica_config):
    model = replace(replica_config.model_config, index_head_dim=256, sliding_window=256)
    capacity = _capacity(replica_config, model=model)
    indexer = next(component for component in capacity.components if component.component_id == "indexer_csa")
    state = next(component for component in capacity.components if component.component_id == "indexer_csa_state")
    assert indexer.content_size_bytes == 64 * (256 + 8)
    assert indexer.page_size_bytes == 17280
    assert state.content_size_bytes == 4 * 1024 * 4
    assert state.page_size_bytes == 16704
    assert [group.window_tokens for group in capacity.groups[1:3]] == [256, 256]
    assert capacity.physical_block_size_bytes == 1183680


@pytest.mark.parametrize("field,value", [
    ("attention_backend", None), ("attention_backend", "FLASHMLA_SPARSE"),
    ("kv_cache_dtype", None), ("kv_cache_dtype", "bfloat16"),
    ("indexer_kv_cache_dtype", None), ("indexer_kv_cache_dtype", "mxfp4"),
])
def test_missing_or_unsupported_selectors_reject_at_both_capacity_callers(
    replica_config, tmp_path, monkeypatch, field, value,
):
    setattr(replica_config, field, value)
    # Isolate the capacity consumer behind V4's intentionally closed execution
    # gate. This unit-only bypass does not establish runtime admission.
    monkeypatch.setattr(AttentionFamilySpec, "require_enabled_for_execution", lambda *args, **kwargs: None)
    with pytest.raises(NotImplementedError, match="DeepSeek-V4"):
        _planner(replica_config)._get_component_kv_cache_capacity(256)
    with pytest.raises(NotImplementedError, match="DeepSeek-V4"):
        _predictor(replica_config, tmp_path)
    assert not (tmp_path / "cache").exists()


def test_predictor_and_planner_use_the_same_resolved_selection(replica_config, tmp_path, monkeypatch):
    import frontier.execution_time_predictor.base_execution_time_predictor as predictor_module
    import frontier.scheduler.utils.memory_planner as planner_module

    calls = []

    def collect(*args, **kwargs):
        capacity = get_attention_kv_cache_capacity(*args, **kwargs)
        calls.append((kwargs["selection"], capacity))
        return capacity

    monkeypatch.setattr(predictor_module, "get_attention_kv_cache_capacity", collect)
    monkeypatch.setattr(planner_module, "get_attention_kv_cache_capacity", collect)
    _planner(replica_config)._get_component_kv_cache_capacity(256)
    with pytest.raises(NotImplementedError, match="compressed cache, compressor"):
        _predictor(replica_config, tmp_path)
    assert len(calls) == 1  # Real execution guard runs before capacity inputs.
    monkeypatch.setattr(AttentionFamilySpec, "require_enabled_for_execution", lambda *args, **kwargs: None)

    def stop_after_capacity(_self):
        raise RuntimeError("unit boundary: capacity resolved; execution not tested")

    monkeypatch.setattr(predictor_module.BaseExecutionTimePredictor, "_validate_dummy_mode_config", stop_after_capacity)
    with pytest.raises(RuntimeError, match="unit boundary"):
        _predictor(replica_config, tmp_path)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert DEEPSEEK_V4_ATTENTION_FAMILY.execution_enabled is False
    assert DEEPSEEK_V4_ATTENTION_FAMILY.structural_execution_enabled is False
    assert not (tmp_path / "cache").exists()


def test_alias_normalizes_once_without_changing_capacity(replica_config):
    fp8 = _capacity(replica_config)
    replica_config.kv_cache_dtype = "fp8_ds_mla"
    assert _capacity(replica_config) == fp8


@pytest.mark.parametrize("changes", [
    {"device": "h200"}, {"gpu_platform": "rocm"},
    {"block_size": 64}, {"decode_context_parallel_size": 2},
])
def test_selection_rejects_unrepresented_runtime_tuple(replica_config, changes):
    selection = replace(AttentionKVCacheSelection.from_replica(replica_config, block_size=256), **changes)
    with pytest.raises((NotImplementedError, ValueError)):
        _capacity(replica_config, selection=selection)


@pytest.mark.parametrize("changes", [
    {"index_head_dim": 96}, {"sliding_window": True},
    {"qk_rope_head_dim": 32}, {"torch_dtype": "float16"},
    {"compress_ratios": (0,) * 42},
])
def test_unsupported_model_geometry_fails_closed(replica_config, changes):
    with pytest.raises((NotImplementedError, ValueError)):
        _capacity(replica_config, model=replace(replica_config.model_config, **changes))


def test_mutated_group_metadata_cannot_lose_or_duplicate_an_owner(replica_config):
    capacity = _capacity(replica_config)
    groups = capacity.groups
    with pytest.raises(ValueError, match="partition every component owner"):
        replace(capacity, groups=groups[:-1])
    with pytest.raises(ValueError, match="multiple groups"):
        replace(capacity, groups=(*groups, replace(groups[0], group_id="duplicate")))
    with pytest.raises(ValueError, match="unknown component owner"):
        replace(capacity, groups=(replace(groups[0], component_owners=(("main_csa", 43),)), *groups[1:]))
    with pytest.raises(ValueError, match="stored rows"):
        replace(capacity, groups=(replace(groups[0], logical_block_size=128), *groups[1:]))


def test_planner_counts_total_slab_including_null_reservation(replica_config, monkeypatch):
    planner = _planner(replica_config)
    monkeypatch.setattr(planner, "_get_parameter_memory_per_device", lambda: 0)
    monkeypatch.setattr(planner._replica._device_config, "total_memory_gb", 1)
    for total_blocks in (2, 17, 100):
        assert planner.get_num_blocks(
            block_size=256, gpu_memory_utilization=1,
            non_kv_cache_overhead_bytes=1024**3 - total_blocks * 1002240,
        ) == total_blocks
    with pytest.raises(FrontierMemoryOOMError) as raised:
        planner.get_num_blocks(
            block_size=256, gpu_memory_utilization=1,
            non_kv_cache_overhead_bytes=1024**3 - 1002240,
        )
    assert raised.value.details["reserved_blocks"] == 1
    assert raised.value.details["physical_block_size_bytes"] == 1002240
    with pytest.raises(NotImplementedError, match="no cache block size"):
        planner.get_max_batch_size()


def test_absent_selectors_do_not_change_legacy_config_serialization():
    config = ReplicaConfig(model_name="step-moe-noquant")
    serialized = dataclass_to_dict(config)
    for name in ("attention_backend", "kv_cache_dtype", "indexer_kv_cache_dtype"):
        assert getattr(config, name) is None
        assert name not in serialized

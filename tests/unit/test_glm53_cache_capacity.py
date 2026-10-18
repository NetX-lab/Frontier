"""Source-backed resident GLM cache capacity, separate from execution support."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention.families import (
    DEEPSEEK_V4_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
)
from frontier.attention.gdn.config import LayerAttentionSpec
from frontier.attention.memory import (
    AttentionKVCacheCapacity,
    AttentionKVCacheComponent,
    get_attention_kv_cache_capacity,
    get_attention_runtime_kv_layout,
)
from frontier.config import ReplicaConfig, SyntheticRequestGeneratorConfig
from frontier.config.model_config import BaseModelConfig
from frontier.entities.replica import Replica
from frontier.errors import FrontierMemoryOOMError
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    return BaseModelConfig.create_from_name("zai-org/GLM-5.3")


def _capacity(model, block_size=64):
    return get_attention_kv_cache_capacity(
        model.get_attention_family(), model, block_size=block_size
    )


def _planner(model, pp_size=1):
    replica_config = ReplicaConfig(
        model_name="zai-org/GLM-5.3",
        num_pipeline_stages=pp_size,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=16,
    )
    assert replica_config.model_config == model
    replica = Replica(
        replica_config,
        SyntheticRequestGeneratorConfig(),
        ClusterType.MONOLITHIC,
    )
    planner = MemoryPlanner(replica_config, replica, ClusterType.MONOLITHIC)
    return planner


def test_glm_capacity_accounts_for_all_main_and_only_full_index_owners(model):
    capacity = _capacity(model)
    main, indexer = capacity.components
    assert capacity.family_id == "dsa_attention"
    assert capacity.block_size == 64
    assert capacity.num_layers == 78
    assert main.component_id == "main_mla"
    assert main.owner_layer_ids == tuple(range(78))
    assert main.page_size_bytes == 73728
    assert indexer.component_id == "indexer_k_and_scale"
    assert indexer.owner_layer_ids == (0, 1, 2, *range(6, 78, 4))
    assert indexer.page_size_bytes == 8448
    assert sum(
        len(component.owner_layer_ids) for component in capacity.components
    ) == 99
    assert capacity.resident_bytes_per_common_block == 5928192
    shared_layers = {
        spec.global_layer_id
        for spec in model.get_layer_attention_specs()
        if spec.variant_id == "shared_index"
    }
    assert len(shared_layers) == 57
    assert shared_layers.isdisjoint(indexer.owner_layer_ids)


def test_glm_capacity_derives_layer_count_and_index_schedule_from_configuration(model):
    small_model = replace(model, num_layers=8)
    capacity = _capacity(small_model)
    assert capacity.components[0].owner_layer_ids == tuple(range(8))
    assert capacity.components[1].owner_layer_ids == (0, 1, 2, 6)
    assert capacity.resident_bytes_per_common_block == 8 * 73728 + 4 * 8448


def test_glm_planner_uses_common_block_budget_once(model, monkeypatch):
    planner = _planner(model)
    parameter_bytes = 64 * 1024**2
    available_bytes = 100 * 5928192
    monkeypatch.setattr(
        planner, "_get_parameter_memory_per_device", lambda: parameter_bytes
    )
    overhead = 1024**3 - parameter_bytes - available_bytes
    assert overhead > 0
    monkeypatch.setattr(planner._replica._device_config, "total_memory_gb", 1)
    assert planner._replica.total_memory_gb == 1
    assert planner.get_num_blocks(
        block_size=64,
        gpu_memory_utilization=1,
        non_kv_cache_overhead_bytes=overhead,
    ) == 100
    assert available_bytes // (78 * 73728) == 103


def test_glm_component_capacity_is_independent_of_frozen_execution(model):
    assert _capacity(model).resident_bytes_per_common_block == 5928192
    assert DSA_ATTENTION_FAMILY.execution_enabled is False
    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        DSA_ATTENTION_FAMILY.require_enabled_for_execution()
    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        get_attention_runtime_kv_layout(
            DSA_ATTENTION_FAMILY,
            runtime_num_kv_heads_per_worker=1,
            runtime_head_size=576,
        )


@pytest.mark.parametrize("available_bytes", [0, 5928191])
def test_glm_planner_rejects_less_than_one_common_block_with_component_details(
    model, monkeypatch, available_bytes
):
    planner = _planner(model)
    monkeypatch.setattr(planner, "_get_parameter_memory_per_device", lambda: 0)
    monkeypatch.setattr(planner._replica._device_config, "total_memory_gb", 1)
    with pytest.raises(FrontierMemoryOOMError) as raised:
        planner.get_num_blocks(
            block_size=64,
            gpu_memory_utilization=1,
            non_kv_cache_overhead_bytes=1024**3 - available_bytes,
        )
    assert raised.value.reason == "insufficient_kv_cache_budget"
    details = raised.value.details
    assert details["resident_bytes_per_common_block"] == 5928192
    assert details["num_cache_components"] == 2
    assert details["num_cache_component_owners"] == 99
    assert "num_full_attention_layers" not in details
    assert "page_size_bytes" not in details


def test_component_planner_rejects_pp_and_blockless_legacy_request_estimator(model):
    with pytest.raises(NotImplementedError, match="num_pipeline_stages=1"):
        _planner(model, pp_size=2).get_num_blocks(block_size=64)
    planner = _planner(model)
    with pytest.raises(NotImplementedError, match="no cache block size"):
        planner._get_kv_cache_memory_per_device_per_request()
    with pytest.raises(NotImplementedError, match="no cache block size"):
        planner.get_max_batch_size()
    with pytest.raises(NotImplementedError, match="no cache block size"):
        planner.get_max_request_slots()


def test_no_compressed_deepseek_capacity_is_guessed(model):
    with pytest.raises(NotImplementedError, match="no component capacity resolver"):
        get_attention_kv_cache_capacity(
            DEEPSEEK_V4_ATTENTION_FAMILY, model, block_size=64
        )


@pytest.mark.parametrize("block_size", [0, -1, True, 64.0])
def test_capacity_rejects_invalid_block_sizes(model, block_size):
    with pytest.raises(ValueError, match="block_size"):
        _capacity(model, block_size)


def test_glm_capacity_rejects_unrepresented_kernel_geometry(model):
    with pytest.raises(NotImplementedError, match="CUDA kernel block size 64"):
        _capacity(model, 16)
    with pytest.raises(NotImplementedError, match="128-element"):
        _capacity(replace(model, index_head_dim=96))
    with pytest.raises(ValueError, match="positive int index_head_dim"):
        _capacity(replace(model, index_head_dim=True))


@pytest.mark.parametrize(
    "specs",
    [
        (LayerAttentionSpec(1, "dsa_attention", "full_index", 1),),
        (LayerAttentionSpec(0, "dense_attention", "full_index", 0),),
        (LayerAttentionSpec(0, "dsa_attention", "dsa", 0),),
        (LayerAttentionSpec(0, "dsa_attention", "shared_index", 0),),
        (
            LayerAttentionSpec(0, "dsa_attention", "full_index", 0),
            LayerAttentionSpec(1, "dsa_attention", "shared_index", 1),
        ),
    ],
)
def test_glm_capacity_rejects_incomplete_or_invalid_producer_bindings(specs):
    config = SimpleNamespace(
        num_layers=len(specs), kv_lora_rank=512, qk_rope_head_dim=64,
        index_head_dim=128, torch_dtype="bfloat16",
        get_layer_attention_specs=lambda: specs,
    )
    with pytest.raises(ValueError):
        get_attention_kv_cache_capacity(DSA_ATTENTION_FAMILY, config, block_size=64)


@pytest.mark.parametrize("page_bytes", [0, -1, True, 1.0])
def test_cache_component_rejects_nonpositive_or_noninteger_pages(page_bytes):
    with pytest.raises(ValueError, match="page_size_bytes"):
        AttentionKVCacheComponent("main", (0,), page_bytes)


@pytest.mark.parametrize("owners", [(), (-1,), (True,), (0, 0)])
def test_cache_component_rejects_invalid_owners(owners):
    with pytest.raises(ValueError, match="owner"):
        AttentionKVCacheComponent("main", owners, 1)


def test_capacity_rejects_duplicate_components_and_nonbackbone_owners():
    component = AttentionKVCacheComponent("main", (0,), 1)
    with pytest.raises(ValueError, match="duplicate cache component"):
        AttentionKVCacheCapacity("test", 64, 1, (component, component))
    with pytest.raises(ValueError, match="backbone layer range"):
        AttentionKVCacheCapacity(
            "test", 64, 1, (AttentionKVCacheComponent("main", (1,), 1),)
        )


def test_capacity_records_are_immutable_and_preserve_distinct_components():
    component = AttentionKVCacheComponent("main", [0], 1)
    capacity = AttentionKVCacheCapacity(
        "test", 64, 1, [component, AttentionKVCacheComponent("indexer", (0,), 2)]
    )
    assert component.owner_layer_ids == (0,)
    assert isinstance(capacity.components, tuple)
    assert capacity.resident_bytes_per_common_block == 3
    with pytest.raises(FrozenInstanceError):
        capacity.block_size = 16

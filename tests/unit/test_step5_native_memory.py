"""CPU reconciliation against retained native Step-5 H800 observations.

Sources (not dependencies of these tests):
task_memory/task_2026-10-15_step5_preview_h800/
  native_request_physical_audit_20261020_04/cloud_selected/node_0/
  device_preparation.json resident_parameters records: TP16 exact
  registered tensor storage 41,909,084,704 B, allocator load 41,993,472,000 B.
native_simulation_scope_20261020/disposition01.md: 1,036,282 KV tokens.
/mnt/codesign-exp/ycfeng/step5-preview/native_pd_h800_r1w_02/:
  each role's two server.log files: 40.74 GiB load, 41.11 GiB consumed,
  1.18 GiB activation peak, 979,459 KV tokens per DP lane (not per role).
Locked image 69308454c, vLLM 0.1.dev22912+g8224bb82f:
  step_vl.py:147-419; rotary_embedding/__init__.py:74-85;
  v1/core/kv_cache_utils.py get_max_concurrency_for_kv_cache_config;
  v1/kv_cache_interface.py:1016-1059; config/vllm.py:578-600.

Rounded profiling logs bound reconstruction error. These tests do not declare
an E2E accuracy tolerance or qualify native runtime fit/predictive performance.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from frontier.config import ReplicaConfig
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter


GIB = 1024**3


def replica(tp, dp, moe_tp, ep):
    return ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview", device="h800",
        attn_tensor_parallel_size=tp, attn_dp=dp,
        moe_tensor_parallel_size=moe_tp, moe_expert_parallel_size=ep,
        attention_backend="STEP5_DSA", kv_cache_dtype="auto",
    )


@pytest.mark.parametrize(("tp", "dp", "moe_tp", "ep", "resident"), [
    (16, 1, 16, 1, 41_909_084_704),
    (8, 2, 1, 16, 43_585_517_952),
    (8, 4, 1, 32, 24_687_967_104),
])
def test_native_topology_resident_storage(tp, dp, moe_tp, ep, resident):
    counter = ParamCounter(replica(tp, dp, moe_tp, ep))
    assert counter.get_parameter_memory_per_device_bytes() == resident
    if tp == 16:
        # Exact native tensor census, not the rounded 39.11 GiB load log.
        assert counter._get_step5_rotary_memory_bytes() == 603_979_776
        assert counter._get_step5_vision_totals() == (328_172_464, 705_146_720)
        assert 41_993_472_000 - resident == 84_387_296


def test_loaded_wrapper_does_not_enable_multimodal_execution_or_duplicate_rope():
    configured = replica(16, 1, 16, 1)
    loaded = ParamCounter(configured)
    configured.model_config = replace(configured.model_config, vision_config=None)
    text_only = ParamCounter(configured)
    assert loaded.get_parameter_memory_per_device_bytes() - text_only.get_parameter_memory_per_device_bytes() == 705_146_720
    assert loaded.get_num_parameters_per_device() - text_only.get_num_parameters_per_device() == 328_172_464
    assert loaded.get_num_attention_parameters_per_device() == text_only.get_num_attention_parameters_per_device()
    assert loaded._get_step5_rotary_memory_bytes() == text_only._get_step5_rotary_memory_bytes()


@pytest.mark.parametrize(("tp", "dp", "moe_tp", "ep", "consumed", "peak", "native_tokens", "native_blocks"), [
    (16, 1, 16, 1, 39.49, 1.03, 1_036_282, 101_579),
    (8, 2, 1, 16, 41.11, 1.18, 979_459, 96_009),
])
def test_native_profiled_capacity_and_rounding_bounds(
    tp, dp, moe_tp, ep, consumed, peak, native_tokens, native_blocks,
):
    configured = replica(tp, dp, moe_tp, ep)
    planner = MemoryPlanner(
        configured, SimpleNamespace(total_memory_gb=80, memory_margin_fraction=0.1),
        ClusterType.MONOLITHIC, max_num_seqs=1,
    )
    capacity = planner._get_component_kv_cache_capacity(16)
    assert capacity.physical_block_size_bytes == 341_780
    assert capacity.fixed_bytes_per_request_slot == 59_156
    assert [len(group.component_owners) for group in capacity.groups] == [23, 23, 23, 69]
    # Native async scheduling has two in-flight batches, each <=512 tokens.
    swa_blocks = -(-(512 - 1 + 2 * 512) // 16) + 1
    sparse_blocks = 8192 // 16
    required_blocks = 3 * swa_blocks + sparse_blocks
    assert (swa_blocks, sparse_blocks, required_blocks) == (97, 512, 803)
    assert int(native_blocks / required_blocks * 8192) == native_tokens
    # The profile input is built from logged measurements, not fitted tokens:
    # nominal-vs-visible H800 budget + native consumed/peak - counted tensors.
    # Graph's estimate is already included in peak; do not subtract it twice.
    overhead = round((80 - 79.19) * 0.92 * GIB + (consumed + peak) * GIB
                     - planner._get_parameter_memory_per_device())
    blocks = planner.get_num_blocks(block_size=16, gpu_memory_utilization=0.92,
                                    non_kv_cache_overhead_bytes=overhead)
    # Each two-decimal GiB log contributes <=0.005 GiB. TP16 also has the
    # observed 32.33-32.35 GiB rank spread, then native takes the minimum.
    rounding = (0.005 * (0.92 + 2) + (0.02 if tp == 16 else 0)) * GIB
    assert abs(blocks - native_blocks) * capacity.physical_block_size_bytes <= rounding
    assert abs(int(blocks / required_blocks * 8192) - native_tokens) <= (
        rounding / capacity.physical_block_size_bytes / required_blocks * 8192 + 1
    )


def test_native_sparse_side_storage_and_swa_window_transfer():
    configured = replica(8, 2, 1, 16)
    planner = MemoryPlanner(configured, SimpleNamespace(total_memory_gb=80),
                            ClusterType.PREFILL, max_num_seqs=1)
    capacity = planner._get_component_kv_cache_capacity(16)
    assert {component.component_id: component.page_size_bytes for component in capacity.components} == {
        "swa_raw_kv": 12_288, "sparse_raw_kv": 12_288,
        "region_summary": 512, "open_region_state": 2060,
    }
    assert [group.transfer_page_count(8192 // 16) for group in capacity.groups] == [33, 33, 33, 512]
    # Summary means remain E4M3 and tail accumulators FP32/int64 despite BF16 KV.
    assert capacity.physical_block_size_bytes == 23 * (12_288 + 512 + 2060)

"""Regression tests for PD-AF 2-microbatch A<->F overlap simulation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.entities.batch import AFDStageMetadata
from frontier.entities.m2n_transfer_info import M2NTransferInfo
from frontier.metrics.af_overlap_tracker import (
    AFOverlapTracker,
    intersect_intervals,
    intervals_duration_ms,
    merge_intervals,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.scheduler.utils.layer_admission import release_batch_admission
from frontier.scheduler.utils.pdaf_transfer import prepare_dp_padding
from frontier.types import ClusterType


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "architecture" / "pd-af-disagg"


def test_af_overlap_tracker_interval_math_and_serial_vs_ping_pong() -> None:
    assert merge_intervals([(0.2, 0.4), (0.1, 0.25), (0.5, 0.6), (0.6, 0.6)]) == [
        (0.1, 0.4),
        (0.5, 0.6),
    ]
    assert intersect_intervals([(0.0, 0.5), (0.6, 1.0)], [(0.2, 0.7)]) == [
        (0.2, 0.5),
        (0.6, 0.7),
    ]
    assert intervals_duration_ms([(0.0, 0.015), (0.02, 0.025)]) == pytest.approx(20.0)

    # Serial AF=1: disjoint ATTN -> M2N A->F -> FFN -> M2N F->A.
    serial = AFOverlapTracker()
    serial.record_stage_start(
        cluster_type=ClusterType.DECODE_ATTN,
        replica_id=1,
        replica_local_id=None,
        stage_id=0,
        batch_id=10,
        start_time=0.000,
    )
    serial.record_stage_end(
        cluster_type=ClusterType.DECODE_ATTN,
        replica_id=1,
        replica_local_id=None,
        stage_id=0,
        batch_id=10,
        end_time=0.010,
    )
    serial.record_m2n_transfer(
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        start_time=0.010,
        end_time=0.011,
    )
    serial.record_stage_start(
        cluster_type=ClusterType.DECODE_FFN,
        replica_id=2,
        replica_local_id=None,
        stage_id=0,
        batch_id=11,
        start_time=0.011,
    )
    serial.record_stage_end(
        cluster_type=ClusterType.DECODE_FFN,
        replica_id=2,
        replica_local_id=None,
        stage_id=0,
        batch_id=11,
        end_time=0.019,
    )
    serial.record_m2n_transfer(
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        start_time=0.019,
        end_time=0.020,
    )
    serial_stats = serial.summarize(af_pipeline_num_micro_batch=1)
    assert serial_stats["af_pipeline_num_micro_batch"] == 1
    assert serial_stats["decode_attn_active_time_ms"] == pytest.approx(10.0)
    assert serial_stats["decode_ffn_active_time_ms"] == pytest.approx(8.0)
    assert serial_stats["m2n_active_time_ms"] == pytest.approx(2.0)
    assert serial_stats["attn_ffn_compute_overlap_time_ms"] == pytest.approx(0.0)
    assert serial_stats["ffn_hidden_ratio"] == pytest.approx(0.0)
    assert serial_stats["af_offload_hidden_ratio"] == pytest.approx(0.0)
    assert serial_stats["pipeline_overlap_savings_ms"] == pytest.approx(0.0)

    # Ping-pong AF=2: mb1 ATTN [10, 20] hides mb0 A->F [10, 11], FFN [11, 18], F->A [18, 19],
    # plus MoE EP fallback_start_time path for an EP rank interval [11, 18].
    overlapped = AFOverlapTracker()
    overlapped.record_stage_start(
        cluster_type=ClusterType.DECODE_ATTN,
        replica_id=1,
        replica_local_id=None,
        stage_id=0,
        batch_id=20,
        start_time=0.000,
    )
    overlapped.record_stage_end(
        cluster_type=ClusterType.DECODE_ATTN,
        replica_id=1,
        replica_local_id=None,
        stage_id=0,
        batch_id=20,
        end_time=0.010,
    )
    overlapped.record_stage_start(
        cluster_type=ClusterType.DECODE_ATTN,
        replica_id=1,
        replica_local_id=None,
        stage_id=0,
        batch_id=21,
        start_time=0.010,
    )
    overlapped.record_stage_end(
        cluster_type=ClusterType.DECODE_ATTN,
        replica_id=1,
        replica_local_id=None,
        stage_id=0,
        batch_id=21,
        end_time=0.020,
    )
    overlapped.record_m2n_transfer(
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        start_time=0.010,
        end_time=0.011,
    )
    overlapped.record_stage_end(
        cluster_type=ClusterType.DECODE_FFN,
        replica_id=2,
        replica_local_id=0,
        stage_id=0,
        batch_id=22,
        end_time=0.018,
        fallback_start_time=0.011,
    )
    overlapped.record_m2n_transfer(
        source_cluster_type=ClusterType.DECODE_FFN,
        target_cluster_type=ClusterType.DECODE_ATTN,
        start_time=0.018,
        end_time=0.019,
    )
    # Concurrent opposite-direction transfer during [18, 19] tests bidirectional M2N tracking.
    overlapped.record_m2n_transfer(
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        start_time=0.0185,
        end_time=0.0195,
    )
    stats = overlapped.summarize(af_pipeline_num_micro_batch=2)
    assert stats["af_pipeline_num_micro_batch"] == 2
    assert stats["decode_attn_active_time_ms"] == pytest.approx(20.0)
    assert stats["decode_ffn_active_time_ms"] == pytest.approx(7.0)
    assert stats["attn_ffn_compute_overlap_time_ms"] == pytest.approx(7.0)
    assert stats["ffn_hidden_ratio"] == pytest.approx(1.0)
    assert stats["af_offload_hidden_ratio"] == pytest.approx(1.0)
    assert stats["m2n_bidirectional_overlap_time_ms"] == pytest.approx(0.5)
    assert stats["decode_phase_wall_time_ms"] == pytest.approx(20.0)
    assert stats["pipeline_overlap_savings_ms"] == pytest.approx(9.5)


def test_release_batch_admission_uses_ticket_stage_id_over_afd_stage_idx() -> None:
    context = StageExecutionContext(replica_id=2, stage_id=0, ep_size=1)
    ticket = context.enqueue_full_stage(operation_id=101)
    assert context.try_acquire(ticket) is True

    looked_up: list[tuple[int, int]] = []

    class _SchedulerStub:
        def get_stage_execution_context(self, replica_id: int, stage_id: int):
            looked_up.append((replica_id, stage_id))
            if (replica_id, stage_id) != (2, 0):
                raise ValueError(
                    f"Unknown Replica/stage admission context: replica_id={replica_id}, stage_id={stage_id}"
                )
            return context

    batch = SimpleNamespace(afd_stage_idx=1, _stage_admission_ticket=ticket)
    release_batch_admission(_SchedulerStub(), batch)

    assert looked_up == [(2, 0)]
    assert context.is_idle is True
    assert not hasattr(batch, "_stage_admission_ticket")


def test_prepare_dp_padding_preserves_physical_microbatch_tokens_and_ffn_cudagraph() -> None:
    meta_lane0 = AFDStageMetadata.from_batch_params(
        num_reqs=8,
        num_tokens_per_req=[1] * 8,
        num_stages=2,
        use_cuda_graph=True,
        cudagraph_capture_sizes=[4, 8, 16],
        ffn_use_cuda_graph=True,
        ffn_cudagraph_capture_sizes=[8, 16],
    )
    meta_lane1 = AFDStageMetadata.from_batch_params(
        num_reqs=12,
        num_tokens_per_req=[1] * 12,
        num_stages=2,
        use_cuda_graph=True,
        cudagraph_capture_sizes=[4, 8, 16],
        ffn_use_cuda_graph=True,
        ffn_cudagraph_capture_sizes=[8, 16],
    )
    # Physical microbatch 0 from each replica carries 4 and 6 requests respectively,
    # while original_stage_token_lens already encodes the per-stage split [4, 4] and [6, 6].
    batch_lane0 = SimpleNamespace(
        is_idle=False,
        requests=[object()] * 4,
        num_tokens=[1] * 4,
        afd_stage_metadata=meta_lane0,
        afd_stage_represents_all_stages=False,
    )
    batch_lane1 = SimpleNamespace(
        is_idle=False,
        requests=[object()] * 6,
        num_tokens=[1] * 6,
        afd_stage_metadata=meta_lane1,
        afd_stage_represents_all_stages=False,
    )
    transfer_lane0 = M2NTransferInfo(
        batch=batch_lane0,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=0,
        source_replica_local_id=None,
        activation_size_bytes=4096,
        transfer_time_ms=0.05,
        transfer_start_time=0.0,
    )
    transfer_lane1 = M2NTransferInfo(
        batch=batch_lane1,
        source_cluster_type=ClusterType.DECODE_ATTN,
        target_cluster_type=ClusterType.DECODE_FFN,
        source_replica_id=1,
        source_replica_local_id=None,
        activation_size_bytes=6144,
        transfer_time_ms=0.05,
        transfer_start_time=0.0,
    )

    padding_plan, dp_summary = prepare_dp_padding(
        [
            (batch_lane0, transfer_lane0),
            (batch_lane1, transfer_lane1),
        ]
    )

    assert dp_summary == (2, [6, 6])
    for _batch, padded_meta in padding_plan:
        assert padded_meta.padded_stage_token_lens == [6, 6]
        assert padded_meta.ffn_use_cuda_graph is True
        assert padded_meta.ffn_cudagraph_capture_sizes == [8, 16]
        assert padded_meta.ffn_compute_stage_token_lens == [8, 8]


def _run_pdaf_example(
    tmp_path: Path,
    relative: str,
    run_id: str,
    env_overrides: dict[str, str],
    extra_args: list[str] | None = None,
) -> dict:
    output_dir = tmp_path / run_id
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "METRICS_OUTPUT_DIR": str(output_dir),
            "RUN_ID": run_id,
            "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1",
        }
    )
    env.update(env_overrides)
    cmd = ["bash", str(EXAMPLES_ROOT / relative)]
    if extra_args:
        cmd.extend(["--", *extra_args])
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    metric_files = list(output_dir.rglob("system_metrics.json"))
    assert len(metric_files) == 1, f"Expected 1 system_metrics.json under {output_dir}"
    return json.loads(metric_files[0].read_text(encoding="utf-8"))


def test_pdaf_dense_af2_e2e_overlaps_attn_and_ffn_while_af1_stays_serial(
    tmp_path: Path,
) -> None:
    common_env = {
        "NUM_REQUESTS": "4",
        "PREFILL_TOKENS": "16",
        "DECODE_TOKENS": "4",
        "DECODE_ATTN_MICRO_BATCH_SIZE": "4",
    }
    common_args = [
        "--cluster_config_decode_attn_request_allocation_threshold",
        "4",
    ]
    af1 = _run_pdaf_example(
        tmp_path,
        "offline/dense_model_basic.sh",
        "dense_af1",
        {
            **common_env,
            "DECODE_ATTN_AF_MICRO_BATCH": "1",
            "DECODE_FFN_AF_MICRO_BATCH": "1",
        },
        extra_args=common_args,
    )
    af2 = _run_pdaf_example(
        tmp_path,
        "offline/dense_model_basic.sh",
        "dense_af2",
        {
            **common_env,
            "DECODE_ATTN_AF_MICRO_BATCH": "2",
            "DECODE_FFN_AF_MICRO_BATCH": "2",
        },
        extra_args=common_args,
    )

    assert af1["simulation_metadata"]["completed_requests"] == 4
    assert af2["simulation_metadata"]["completed_requests"] == 4

    ovl1 = af1["af_pipeline_overlap_statistics"]
    ovl2 = af2["af_pipeline_overlap_statistics"]
    assert ovl1["af_pipeline_num_micro_batch"] == 1
    assert ovl1["attn_ffn_compute_overlap_time_ms"] == pytest.approx(0.0)
    assert ovl1["ffn_hidden_ratio"] == pytest.approx(0.0)
    assert ovl1["pipeline_overlap_savings_ms"] == pytest.approx(0.0)

    assert ovl2["af_pipeline_num_micro_batch"] == 2
    assert ovl2["attn_ffn_compute_overlap_time_ms"] > 0.0
    assert ovl2["ffn_hidden_ratio"] > 0.90
    assert ovl2["af_offload_hidden_ratio"] > 0.90
    assert ovl2["pipeline_overlap_savings_ms"] > 0.0
    assert ovl2["decode_phase_wall_time_ms"] < ovl2["serial_equivalent_time_ms"]

    # Edge cases: single-request graceful fallback and odd-batch CUDA Graph run.
    single = _run_pdaf_example(
        tmp_path,
        "offline/dense_model_basic.sh",
        "dense_af2_n1",
        {
            "NUM_REQUESTS": "1",
            "PREFILL_TOKENS": "16",
            "DECODE_TOKENS": "3",
            "DECODE_ATTN_MICRO_BATCH_SIZE": "4",
            "DECODE_ATTN_AF_MICRO_BATCH": "2",
            "DECODE_FFN_AF_MICRO_BATCH": "2",
        },
    )
    assert single["simulation_metadata"]["completed_requests"] == 1
    assert single["af_pipeline_overlap_statistics"]["ffn_hidden_ratio"] == pytest.approx(0.0)

    odd_cg = _run_pdaf_example(
        tmp_path,
        "offline/dense_cuda_graph.sh",
        "dense_cg_af2_n5",
        {
            "NUM_REQUESTS": "5",
            "PREFILL_TOKENS": "16",
            "DECODE_TOKENS": "3",
            "DECODE_ATTN_MICRO_BATCH_SIZE": "5",
            "DECODE_ATTN_AF_MICRO_BATCH": "2",
            "DECODE_FFN_AF_MICRO_BATCH": "2",
        },
        extra_args=[
            "--cluster_config_decode_attn_request_allocation_threshold",
            "5",
        ],
    )
    assert odd_cg["simulation_metadata"]["completed_requests"] == 5
    assert odd_cg["af_pipeline_overlap_statistics"]["ffn_hidden_ratio"] > 0.90


def test_pdaf_moe_af2_e2e_ep2_multi_attn_replica_and_online(tmp_path: Path) -> None:
    # Single DECODE_ATTN -> DECODE_FFN with EP=2 and AF=2.
    ep2 = _run_pdaf_example(
        tmp_path,
        "offline/moe_model_ep.sh",
        "moe_ep2_af2",
        {
            "NUM_REQUESTS": "4",
            "PREFILL_TOKENS": "16",
            "DECODE_TOKENS": "3",
            "DECODE_ATTN_MICRO_BATCH_SIZE": "4",
            "DECODE_ATTN_AF_MICRO_BATCH": "2",
            "DECODE_FFN_AF_MICRO_BATCH": "2",
        },
        extra_args=[
            "--cluster_config_decode_attn_request_allocation_threshold",
            "4",
        ],
    )
    assert ep2["simulation_metadata"]["completed_requests"] == 4
    assert ep2["af_pipeline_overlap_statistics"]["ffn_hidden_ratio"] > 0.90

    # Multi-replica (2 DECODE_ATTN -> 1 DECODE_FFN) with EP=2 and AF=2 exercises DP padding.
    multi_attn = _run_pdaf_example(
        tmp_path,
        "offline/moe_model_ep.sh",
        "moe_2attn_ep2_af2",
        {
            "DECODE_ATTN_REPLICAS": "2",
            "NUM_REQUESTS": "8",
            "PREFILL_TOKENS": "16",
            "DECODE_TOKENS": "3",
            "DECODE_ATTN_MICRO_BATCH_SIZE": "4",
            "DECODE_ATTN_AF_MICRO_BATCH": "2",
            "DECODE_FFN_AF_MICRO_BATCH": "2",
        },
        extra_args=[
            "--cluster_config_decode_attn_request_allocation_threshold",
            "4",
        ],
    )
    assert multi_attn["simulation_metadata"]["completed_requests"] == 8
    assert multi_attn["af_pipeline_overlap_statistics"]["ffn_hidden_ratio"] > 0.90

    # Online mode MoE AF=2 smoke test.
    online = _run_pdaf_example(
        tmp_path,
        "online/moe_model_basic_online.sh",
        "moe_online_af2",
        {
            "NUM_REQUESTS": "4",
            "PREFILL_TOKENS": "16",
            "DECODE_TOKENS": "3",
            "QPS": "100",
            "DECODE_ATTN_MICRO_BATCH_SIZE": "4",
            "DECODE_ATTN_AF_MICRO_BATCH": "2",
            "DECODE_FFN_AF_MICRO_BATCH": "2",
        },
    )
    assert online["simulation_metadata"]["completed_requests"] == 4
    assert online["af_pipeline_overlap_statistics"]["attn_ffn_compute_overlap_time_ms"] > 0.0

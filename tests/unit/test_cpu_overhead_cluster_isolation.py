"""Regression tests for cluster-scoped CPU overhead isolation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.config import global_vars
from frontier.config.execution_time_predictor_config import (
    LinearRegressionExecutionTimePredictorConfig,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.types import ClusterType, MeasurementType


class _ModelConfig:
    def get_name(self) -> str:
        return "Qwen3-4B"


def _write_cpu_csv(
    path: Path,
    prepare_by_tp: dict[int, float],
    output_by_tp: dict[int, float],
    measurement_type: MeasurementType = MeasurementType.KERNEL_ONLY,
) -> None:
    rows = [
        {
            "model_name": "Qwen3-4B",
            "batch_size": batch_size,
            "tensor_parallel_degree": tp,
            "num_prefill_tokens": 0,
            "num_decode_tokens": batch_size,
            "scheduling_mode": "sync",
            "schedule_mean": 0.0,
            "schedule_median": 0.0,
            "prepare_inputs_e2e_mean": prepare_by_tp[tp],
            "prepare_inputs_e2e_median": prepare_by_tp[tp],
            "sampler_e2e_mean": 0.0,
            "sampler_e2e_median": 0.0,
            "process_model_outputs_mean": output_by_tp[tp],
            "process_model_outputs_median": output_by_tp[tp],
            "ray_comm_time_mean": 0.0,
            "profiling_precision": "FP8",
            "measurement_type": measurement_type.value,
        }
        for tp in prepare_by_tp
        for batch_size in (1, 8)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _predictor_config(tmp_path: Path) -> LinearRegressionExecutionTimePredictorConfig:
    return LinearRegressionExecutionTimePredictorConfig(
        skip_cpu_overhead_modeling=False,
        cpu_overhead_input_file=str(
            tmp_path / "cpu" / "{NETWORK_DEVICE}" / "{MODEL}" / "cpu_overheads.csv"
        ),
        cpu_overhead_kernel_only_input_file=str(
            tmp_path / "cpu" / "{NETWORK_DEVICE}" / "{MODEL}" / "cpu_overheads_kernel_only.csv"
        ),
    )


def _pdaf_manager(tmp_path: Path) -> ExecutionTimePredictionModelManager:
    global_vars.set_global_vars("offline", "pd-af-disaggregation")
    pred_cfg = _predictor_config(tmp_path)
    model_cfg = _ModelConfig()
    manager = ExecutionTimePredictionModelManager({}, SimpleNamespace(cache_dir=str(tmp_path / "cache")))
    manager._all_dummy_mode = False
    manager._cluster_configs = {
        ClusterType.DECODE_ATTN: SimpleNamespace(
            replica_config=SimpleNamespace(
                device="h200", network_device="h200_dgx", model_name="Qwen3-4B",
                model_config=model_cfg, attn_tensor_parallel_size=8,
                moe_tensor_parallel_size=1, cluster_prefix="decode_attn",
            ),
            execution_time_predictor_config=pred_cfg,
        ),
        ClusterType.DECODE_FFN: SimpleNamespace(
            replica_config=SimpleNamespace(
                device="hpp2", network_device="hpp2_16rank", model_name="Qwen3-4B",
                model_config=model_cfg, attn_tensor_parallel_size=1,
                moe_tensor_parallel_size=16, cluster_prefix="decode_ffn",
            ),
            execution_time_predictor_config=pred_cfg,
        ),
    }
    return manager


def _train_cpu_for_clusters(manager, clusters, measurement_type) -> set[str]:
    manager._set_active_measurement_type(measurement_type)
    sigs: set[str] = set()
    for cluster_type in clusters:
        cfg = manager._cluster_configs[cluster_type]
        manager._train_cpu_overhead_models_for_cluster(
            cluster_type, cfg.replica_config, cfg.execution_time_predictor_config, sigs
        )
    return sigs


def test_decode_ffn_cpu_overhead_does_not_leak_to_decode_attn(tmp_path: Path) -> None:
    _write_cpu_csv(
        tmp_path / "cpu" / "hpp2_16rank" / "Qwen3-4B" / "cpu_overheads_kernel_only.csv",
        {16: 0.0187}, {16: 0.0018},
    )
    manager = _pdaf_manager(tmp_path)
    _train_cpu_for_clusters(manager, (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN), MeasurementType.KERNEL_ONLY)

    attn = manager.get_models_for_cluster(ClusterType.DECODE_ATTN)["kernel_only"]
    ffn = manager.get_models_for_cluster(ClusterType.DECODE_FFN)["kernel_only"]
    assert "prepare_inputs_e2e" not in attn
    assert "process_model_outputs" not in attn
    assert ffn["prepare_inputs_e2e"]._frontier_exact_lookup[(8.0, 0.0, 8.0)] == pytest.approx(0.0187)


def test_heterogeneous_clusters_keep_distinct_cpu_overhead_models(tmp_path: Path) -> None:
    _write_cpu_csv(
        tmp_path / "cpu" / "h200_dgx" / "Qwen3-4B" / "cpu_overheads_kernel_only.csv",
        {8: 0.005}, {8: 0.004},
    )
    _write_cpu_csv(
        tmp_path / "cpu" / "hpp2_16rank" / "Qwen3-4B" / "cpu_overheads_kernel_only.csv",
        {16: 0.025}, {16: 0.002},
    )
    manager = _pdaf_manager(tmp_path)
    _train_cpu_for_clusters(manager, (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN), MeasurementType.KERNEL_ONLY)

    attn = manager.get_models_for_cluster(ClusterType.DECODE_ATTN)["kernel_only"]
    ffn = manager.get_models_for_cluster(ClusterType.DECODE_FFN)["kernel_only"]
    assert attn["prepare_inputs_e2e"]._frontier_exact_lookup[(8.0, 0.0, 8.0)] == pytest.approx(0.005)
    assert ffn["prepare_inputs_e2e"]._frontier_exact_lookup[(8.0, 0.0, 8.0)] == pytest.approx(0.025)
    with pytest.raises(ValueError, match="multiple cluster scopes"):
        manager.get_models()


def test_homogeneous_clusters_share_and_ffn_signature_uses_moe_tp(tmp_path: Path) -> None:
    _write_cpu_csv(
        tmp_path / "cpu" / "h200_dgx" / "Qwen3-4B" / "cpu_overheads.csv",
        {8: 0.009}, {8: 0.003}, MeasurementType.CUDA_EVENT,
    )
    _write_cpu_csv(
        tmp_path / "cpu" / "hpp2_16rank" / "Qwen3-4B" / "cpu_overheads_kernel_only.csv",
        {8: 0.001, 16: 0.019}, {8: 0.001, 16: 0.002},
    )
    pred_cfg = _predictor_config(tmp_path)
    model_cfg = _ModelConfig()
    h200_replica = SimpleNamespace(
        device="h200", network_device="h200_dgx", model_name="Qwen3-4B",
        model_config=model_cfg, attn_tensor_parallel_size=8, moe_tensor_parallel_size=8,
        cluster_prefix="prefill",
    )
    manager = ExecutionTimePredictionModelManager({}, SimpleNamespace(cache_dir=str(tmp_path / "cache")))
    manager._all_dummy_mode = False
    manager._cluster_configs = {
        ClusterType.PREFILL: SimpleNamespace(replica_config=h200_replica, execution_time_predictor_config=pred_cfg),
        ClusterType.DECODE_ATTN: SimpleNamespace(replica_config=h200_replica, execution_time_predictor_config=pred_cfg),
    }
    _train_cpu_for_clusters(manager, (ClusterType.PREFILL, ClusterType.DECODE_ATTN), MeasurementType.CUDA_EVENT)
    assert (
        manager.get_models_for_cluster(ClusterType.PREFILL)["eager"]["prepare_inputs_e2e"]
        is manager.get_models_for_cluster(ClusterType.DECODE_ATTN)["eager"]["prepare_inputs_e2e"]
    )

    manager._set_active_measurement_type(MeasurementType.KERNEL_ONLY)
    sigs: set[str] = set()
    ffn_tp8 = SimpleNamespace(
        device="hpp2", network_device="hpp2_16rank", model_name="Qwen3-4B",
        model_config=model_cfg, attn_tensor_parallel_size=1, moe_tensor_parallel_size=8,
        cluster_prefix="decode_ffn",
    )
    ffn_tp16 = SimpleNamespace(
        device="hpp2", network_device="hpp2_16rank", model_name="Qwen3-4B",
        model_config=model_cfg, attn_tensor_parallel_size=1, moe_tensor_parallel_size=16,
        cluster_prefix="decode_ffn",
    )
    m8 = manager._train_cpu_overhead_models_for_cluster(ClusterType.DECODE_FFN, ffn_tp8, pred_cfg, sigs)
    m16 = manager._train_cpu_overhead_models_for_cluster(ClusterType.DECODE_FFN, ffn_tp16, pred_cfg, sigs)
    assert m8["prepare_inputs_e2e"]._frontier_exact_lookup[(8.0, 0.0, 8.0)] == pytest.approx(0.001)
    assert m16["prepare_inputs_e2e"]._frontier_exact_lookup[(8.0, 0.0, 8.0)] == pytest.approx(0.019)

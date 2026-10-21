"""Generate explicitly synthetic CPU inputs and fit the real shared cache/bundles.

This is a single-request integration fixture, never native profiling or admission.
Run with the repository and tests/unit on PYTHONPATH; no GPU or network is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pytest import MonkeyPatch

from frontier.config import global_vars
from frontier.config.config import RandomForrestExecutionTimePredictorConfig
from frontier.config.model_config import BaseModelConfig
from frontier.attention.step5 import Step5AttentionWorkload
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.profiling.step5.rows import rows_to_dataframe
from frontier.types import CCBackendType, ClusterType, MeasurementType
import test_step5_measured_ffn as ffn
from test_step5_measured_training import _row, _scheduler


def generate(output: Path, *, device: str, gpus_per_role: int, nixl_read_profile: str | None = None):
    if gpus_per_role <= 0 or gpus_per_role % 8:
        raise ValueError("Fixture GPUs per role must be divisible by attention TP8")
    output.mkdir(parents=True, exist_ok=False)
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    inputs = {}
    # Explicit fixture topology and reached single-request cases, not native evidence.
    ep, capacity, block = gpus_per_role, 8192, 16
    cases = [(6144, 0, "prefill", "none"), (16, 6128, "prefill", "none"),
             (1, 6144, "decode", "full"), (1, 7167, "decode", "full")]
    for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT):
        rows = []
        for query, context, phase, graph in cases:
            if (graph == "none") != (measurement is MeasurementType.KERNEL_ONLY):
                continue
            for layer in range(model.num_layers):
                row = _row(model, layer, query=query, context=context, phase=phase, graph_mode=graph)
                # The serving fixture uses allocated capacity, not live-prefix allocation.
                workload = Step5AttentionWorkload.from_lengths(
                    model, phase=phase, query_lengths=(query,), sequence_lengths=(context + query,),
                    physical_query_rows=query, block_size_tokens=block,
                    block_table_columns=capacity // block, max_model_len_tokens=capacity,
                    graph_mode=graph,
                )
                row.update(device=device, block_table_columns=capacity // block,
                           score_regions=workload.score_regions)
                rows.append(row)
        path = output / f"fixture_attention_{measurement.value.lower()}.csv"
        rows_to_dataframe(rows, measurement_type=measurement).to_csv(path, index=False)
        inputs[f"attention_{measurement.value}"] = str(path)
    compute = {}
    for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT):
        directory = output / f"fixture_compute_{measurement.value.lower()}"
        directory.mkdir()
        with MonkeyPatch.context() as patch:
            metadata = ffn._metadata
            patch.setattr(ffn, "_metadata", lambda config: {
                **metadata(config), "device": device, "measurement_type": measurement.value,
            })
            compute[measurement] = ffn._tables(
                directory, model, token_counts=(1, 8, 16, 32, 6144), expert_parallel_size=ep,
                include_stage_boundaries=True,
            )
    gap = output / "fixture_kernel_gap.csv"
    pd.DataFrame({"execution_mode": ["eager", "cuda_graph"],
                  "num_tensor_parallel_workers": [8, 8],
                  "kernel_gap_us": [1., 2.]}).to_csv(gap, index=False)
    options = dict(
        enable_dummy_mode=False, skip_cpu_overhead_modeling=True,
        linear_op_input_file=str(compute[MeasurementType.CUDA_EVENT][0]),
        moe_input_file=str(compute[MeasurementType.CUDA_EVENT][1]),
        linear_op_kernel_only_input_file=str(compute[MeasurementType.KERNEL_ONLY][0]),
        moe_kernel_only_input_file=str(compute[MeasurementType.KERNEL_ONLY][1]),
        atten_input_file=inputs["attention_CUDA_EVENT"],
        atten_kernel_only_input_file=inputs["attention_KERNEL_ONLY"],
        kernel_gap_input_file=str(gap), num_estimators=[2], max_depth=[3],
        min_samples_split=[2], k_fold_cv_splits=2, num_training_job_threads=1,
    )
    config = RandomForrestExecutionTimePredictorConfig(**options)
    config.nixl_read_profile = nixl_read_profile
    replica = ffn._replica(model)
    replica.device, replica.moe_expert_parallel_size = device, ep
    replica.attn_dp, replica.num_pipeline_stages = gpus_per_role // 8, 1
    replica.network_device = "cpu_fixture_network"
    replica.speculative_decoding_config = SimpleNamespace(enabled=False)
    scheduler = _scheduler()
    scheduler.get_type = lambda: "cpu_fixture_scheduler"
    backend = SimpleNamespace(get_type=lambda: CCBackendType.COLLECTIVE_SIM,
                              get_name=lambda: "collective_sim")
    cluster = SimpleNamespace(replica_config=replica, replica_scheduler_config=scheduler,
                              execution_time_predictor_config=config, cc_backend_config=backend)
    with MonkeyPatch.context() as patch:
        patch.setattr(global_vars, "get_sys_arch", lambda: "pd-disaggregation")
        patch.setattr(global_vars, "get_decode_cuda_graph_mode", lambda: "full")
        manager = ExecutionTimePredictionModelManager(
            {ClusterType.PREFILL: cluster, ClusterType.DECODE: cluster},
            SimpleNamespace(cache_dir=str(output / "cache")),
        )
    manifest = {
        "schema": "step5_measured_compute_cpu_fixture.v1", "evidence_kind": "cpu_fixture",
        "native_admitted": False, "device": device, "attention_tp": 8, "attention_dp": gpus_per_role // 8,
        "moe_tp": 1, "moe_ep": ep, "global_layer_ids": list(range(model.num_layers)),
        "max_model_len": capacity, "block_size": block, "batch_width": 1,
        "cases": cases, "predictor_config": {**options, "nixl_read_profile": nixl_read_profile},
        "cache_dir": str(output / "cache"),
        "bundles": {str(role.value): receipts for role, receipts in manager._step5_model_bundles.items()},
        "limitations": ["Synthetic timings/counts; no native admission or accuracy claim",
                        "CPU and communication isolation are explicit",
                        "Unlisted physical shapes remain coverage errors"],
    }
    (output / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("h800", "h200"))
    parser.add_argument("--gpus-per-role", required=True, type=int, choices=(16, 32))
    parser.add_argument("--nixl-read-profile")
    args = parser.parse_args()
    result = generate(args.output_dir.resolve(), device=args.device, gpus_per_role=args.gpus_per_role,
                      nixl_read_profile=args.nixl_read_profile)
    print(json.dumps({"manifest": str(args.output_dir.resolve() / "fixture_manifest.json"),
                      "native_admitted": False, "roles": list(result["bundles"])}))

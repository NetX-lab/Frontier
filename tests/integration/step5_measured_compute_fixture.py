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
from frontier.attention.step5 import Step5AttentionWorkload, describe_step5_layer
from frontier.attention.step5_inputs import MRv2DummyDescriptor
from frontier.execution_time_predictor.step5 import STEP5_SCOPE_NAMES, STEP5_OPERATOR_NAMES, step5_measurement_type
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.profiling.step5.inputs import Step5ProfileInput
from frontier.profiling.step5.rows import build_row, rows_to_dataframe
from frontier.moe_ep_workload import (
    build_contiguous_expert_ownership, generate_moe_routing_ratios, materialize_layer_ep_workload,
)
from frontier.moe_load_imbalance import MoELoadImbalanceInput
from frontier.types import CCBackendType, ClusterType, MeasurementType
from tests.unit import test_step5_measured_ffn as ffn
from tests.unit.test_step5_measured_training import _scheduler


CAPTURE_SIZES = (1, 2, 4, *range(8, 256, 8), 256)
MAX_MODEL_LEN = 7168
LOCAL_COMPUTE_WIDTHS = (1, 16, 6144)
AGGREGATE_COMPUTE_WIDTHS = (2, 32, 6145)


def attention_row(model, cell, *, physical_query_rows=None, device="h800"):
    """Build labelled CPU costs while preserving real queries and physical padding."""
    workload = Step5AttentionWorkload.from_lengths(
        model, phase=cell.phase, query_lengths=cell.query_lens, sequence_lengths=cell.sequence_lens,
        physical_query_rows=physical_query_rows or cell.num_tokens, block_size_tokens=16,
        block_table_columns=448, max_model_len_tokens=MAX_MODEL_LEN, graph_mode=cell.graph_mode,
    )
    metadata = describe_step5_layer(model, cell.global_layer_id, attention_tp_size=8,
                                   moe_tp_size=1, expert_parallel_size=1)
    scopes = metadata.attention_scopes(workload)
    active = {name for name, _ in scopes}
    stats = {}
    for name in (*STEP5_SCOPE_NAMES, *STEP5_OPERATOR_NAMES):
        value = .01 + workload.physical_query_rows / 10000 if name in active or name == "step5_layer_e2e" else 0.
        stats[name] = dict(min=value, max=value, mean=value, median=value, std=0., count=10)
        if name == "step5_layer_e2e":
            stats[name]["kernel_count"] = 17 if cell.dummy_descriptor else 23
    measurement = step5_measurement_type(cell.graph_mode)
    row = build_row(
        model, cell, tp=8, block_size=16, max_model_len=MAX_MODEL_LEN,
        runtime_signature="cpu_fixture_unadmitted", device=device, time_stats=stats,
        measurement_type=measurement.value,
        evidence=dict(attention_backend="cpu_fixture_unadmitted", evidence_kind="cpu_fixture",
                      timing_basis="cuda_event_native_runtime_latency" if measurement is MeasurementType.CUDA_EVENT
                      else "correlated_kernel_plus_memcpy_duration_sum", active_operators=json.dumps(sorted(active))),
    )
    contracts = json.loads(row["typed_operator_contracts"])
    row.update(typed_operator_contracts=json.dumps({name: contracts[name] for name in STEP5_SCOPE_NAMES}),
               padded_tokens=workload.padded_tokens, fusion=json.dumps(scopes),
               block_table_columns=448, score_regions=workload.score_regions)
    return row


def reached_rows(model, *, device):
    """Exact1023 context rows reuse one layer/shape contract, never serving credit."""
    for layer in range(model.num_layers):
        for query, context, phase, graph in ((6144, 0, "prefill", "none"),
                                              (16, 6128, "prefill", "piecewise")):
            yield attention_row(model, Step5ProfileInput(layer, (query,), (context,), phase, graph,
                                                       block_table_columns=448), device=device)
        # The final output does not execute a subsequent forward at context7167.
        template = attention_row(model, Step5ProfileInput(layer, (1,), (6144,), "decode", "full",
                                                        block_table_columns=448), device=device)
        # Uniform FULL scopes/score allocation depend on physical capture/capacity,
        # not the live context. Every row still retains its own actual context.
        for context in range(6144, 7167):
            yield {**template, "context_lens": json.dumps((context,))}
        for descriptor in (
            MRv2DummyDescriptor(1, 1, "none", None, "CPU fixture of MRv2 InputBatch"),
            MRv2DummyDescriptor(16, None, "piecewise", None, "CPU fixture of MRv2 InputBatch"),
            MRv2DummyDescriptor(1, 1, "full", None, "CPU fixture of MRv2 InputBatch", 1),
        ):
            yield attention_row(model, Step5ProfileInput.from_mrv2_dummy(layer, descriptor, block_table_columns=448),
                                device=device)


def compute_tables(directory, model, *, device, expert_parallel_size, measurement):
    """Cover local operators and exact balanced physical EP loads, including dummy work."""
    with MonkeyPatch.context() as patch:
        metadata = ffn._metadata
        patch.setattr(ffn, "_metadata", lambda config: {
            **metadata(config), "device": device, "measurement_type": measurement.value,
        })
        paths = ffn._tables(
            directory, model, token_counts=tuple(sorted(set(LOCAL_COMPUTE_WIDTHS + AGGREGATE_COMPUTE_WIDTHS))),
            expert_parallel_size=expert_parallel_size, include_stage_boundaries=True,
        )
    templates = pd.read_csv(paths[1], float_precision="round_trip")
    rows = []
    for _, template in templates[templates.num_tokens.isin(AGGREGATE_COMPUTE_WIDTHS)].iterrows():
        layer = int(template.moe_layer_ids.split("-")[0])
        tokens = int(template.num_tokens)
        workload = materialize_layer_ep_workload(
            routing_ratios=generate_moe_routing_ratios(
                total_expert_num=model.num_experts, distribution_type="balanced", seed=42, layer_id=layer),
            target_replica_id=0, global_layer_id=layer, routing_token_count=tokens,
            router_topk=model.num_experts_per_tok, total_expert_num=model.num_experts,
            moe_expert_parallel_size=expert_parallel_size,
            expert_to_ep=build_contiguous_expert_ownership(model.num_experts, expert_parallel_size),
        )
        for ep_id in workload.participant_ep_ids:
            lane = workload.lane(ep_id)
            if lane.routed_token_count == 0:
                continue
            counts = list(lane.local_token_counts)
            features = MoELoadImbalanceInput(
                num_tokens=tokens, num_experts_per_device=lane.local_expert_width,
                hidden_dim=model.embedding_dim, expert_hidden_dim=model.mlp_hidden_dim,
                router_topk=lane.router_topk, expert_token_counts=counts,
            ).to_features_dict()
            rows.append({**template.to_dict(), **features, "expert_token_counts": json.dumps(counts)})
    pd.DataFrame(rows).drop_duplicates().to_csv(paths[1], index=False)
    return paths


def refresh_compute(output: Path):
    """Repair only synthetic compute inputs; retain attention output and the shared cache."""
    path = output / "fixture_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["evidence_kind"] != "cpu_fixture" or manifest["native_admitted"]:
        raise ValueError("Only unadmitted CPU fixture compute inputs may be refreshed")
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT):
        directory = output / f"fixture_compute_{measurement.value.lower()}"
        for name in ("linear_op.csv", "moe.csv"):
            source = directory / name
            backup = directory / f"{name}.before_mrv2_compute"
            with backup.open("xb") as stream:
                stream.write(source.read_bytes())
        compute_tables(directory, model, device=manifest["device"],
                       expert_parallel_size=manifest["moe_ep"], measurement=measurement)
    manifest["compute_coverage"] = {
        "local_widths": LOCAL_COMPUTE_WIDTHS, "aggregate_widths": AGGREGATE_COMPUTE_WIDTHS,
        "routing": "existing balanced Hamilton materializer, exact nonempty EP lanes",
    }
    manifest["bundles"] = {}
    manifest["limitations"].append("Compute changed; actual CLI must rebuild shared compute bundles")
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def generate(output: Path, *, device: str, gpus_per_role: int, nixl_read_profile: str | None = None):
    if gpus_per_role <= 0 or gpus_per_role % 8:
        raise ValueError("Fixture GPUs per role must be divisible by attention TP8")
    output.mkdir(parents=True, exist_ok=False)
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    inputs = {}
    # Explicit fixture topology and reached single-request cases, not native evidence.
    ep, capacity, block = gpus_per_role, MAX_MODEL_LEN, 16
    cases = [(6144, 0, "prefill", "none"), (16, 6128, "prefill", "piecewise"),
             (1, "6144..7166 inclusive", "decode", "full")]
    rows_by_measurement = {measurement: [] for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT)}
    for row in reached_rows(model, device=device):
        rows_by_measurement[step5_measurement_type(row["graph_mode"])].append(row)
    for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT):
        path = output / f"fixture_attention_{measurement.value.lower()}.csv"
        rows_to_dataframe(rows_by_measurement.pop(measurement), measurement_type=measurement).to_csv(path, index=False)
        inputs[f"attention_{measurement.value}"] = str(path)
    compute = {}
    for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT):
        directory = output / f"fixture_compute_{measurement.value.lower()}"
        directory.mkdir()
        compute[measurement] = compute_tables(directory, model, device=device,
                                             expert_parallel_size=ep, measurement=measurement)
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
    replica.kv_cache_dtype = "auto"
    replica.network_device = "cpu_fixture_network"
    replica.speculative_decoding_config = SimpleNamespace(enabled=False)
    scheduler = _scheduler()
    scheduler.max_model_len = capacity
    scheduler.max_tokens_in_batch, scheduler.batch_size_cap = 8192, 128
    scheduler.long_prefill_token_threshold, scheduler.enable_chunked_prefill = 0, False
    scheduler.get_type = lambda: "cpu_fixture_scheduler"
    backend = SimpleNamespace(get_type=lambda: CCBackendType.COLLECTIVE_SIM,
                              get_name=lambda: "collective_sim")
    cluster = SimpleNamespace(replica_config=replica, replica_scheduler_config=scheduler,
                              execution_time_predictor_config=config, cc_backend_config=backend)
    with MonkeyPatch.context() as patch:
        patch.setattr(global_vars, "get_sys_arch", lambda: "pd-disaggregation")
        patch.setattr(global_vars, "get_decode_cuda_graph_mode", lambda: "full_and_piecewise")
        manager = ExecutionTimePredictionModelManager(
            {ClusterType.PREFILL: cluster, ClusterType.DECODE: cluster},
            SimpleNamespace(cache_dir=str(output / "cache")),
        )
    manifest = {
        "schema": "step5_measured_compute_cpu_fixture.v1", "evidence_kind": "cpu_fixture",
        "native_admitted": False, "device": device, "attention_tp": 8, "attention_dp": gpus_per_role // 8,
        "moe_tp": 1, "moe_ep": ep, "global_layer_ids": list(range(model.num_layers)),
        "max_model_len": capacity, "block_size": block, "batch_width": 1,
        "decode_cuda_graph_mode": "full_and_piecewise", "cudagraph_capture_sizes": CAPTURE_SIZES,
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
    parser.add_argument("--device", choices=("h800", "h200"))
    parser.add_argument("--gpus-per-role", type=int, choices=(16,))
    parser.add_argument("--refresh-compute", action="store_true")
    parser.add_argument("--nixl-read-profile")
    args = parser.parse_args()
    if args.refresh_compute:
        result = refresh_compute(args.output_dir.resolve())
    else:
        if args.device is None or args.gpus_per_role is None:
            parser.error("Generation requires --device and --gpus-per-role")
        result = generate(args.output_dir.resolve(), device=args.device, gpus_per_role=args.gpus_per_role,
                          nixl_read_profile=args.nixl_read_profile)
    print(json.dumps({"manifest": str(args.output_dir.resolve() / "fixture_manifest.json"),
                      "native_admitted": False, "roles": list(result["bundles"])}))

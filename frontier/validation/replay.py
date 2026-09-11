"""Replay captured batches through Frontier's native execution-time predictor.

Pass normal frontier.main arguments after --. Observations are used only after
prediction; missing profiles and dummy mode cannot yield numerical parity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

from frontier.validation.batch_record import read_batches, validate_rank_cohorts


def replay_batches(records, predictor, *, model_config, tensor_parallel_size: int) -> list[dict]:
    from frontier.types import ClusterType

    if getattr(predictor, "_enable_dummy_mode", False):
        raise ValueError("Dummy predictor output is not numerical correlation evidence")
    cohorts = validate_rank_cohorts(records, tensor_parallel_size=tensor_parallel_size)
    rows = []
    for batch_id, cohort in cohorts.items():
        record = cohort[0]
        if record.profiled:
            continue
        if any(r.forward_gpu_ms is None for r in cohort):
            raise ValueError(f"Missing forward GPU timing for {batch_id}")
        batch = record.to_frontier_batch(is_moe=model_config.is_moe)
        execution = predictor.predict_stage_execution_time(
            batch, stage_id=0, cluster_type=ClusterType.MONOLITHIC,
            num_layers=model_config.num_layers, layer_id=0,
        )
        predicted = float(execution.model_time_ms)
        if not math.isfinite(predicted) or predicted <= 0:
            raise ValueError(f"Invalid prediction for {batch_id}: {predicted}")
        observed = max(r.forward_gpu_ms for r in cohort)
        rows.append({
            "batch_id": batch_id, "phase": record.phase,
            "batch_size": len(record.request_ids),
            "query_tokens": sum(record.query_lens),
            "max_context": max(record.context_lens),
            "graph_mode": record.graph_mode, "capture_size": record.capture_size,
            "repetition": record.repetition, "step": record.step,
            "observed_forward_gpu_ms": observed,
            "predicted_model_ms": predicted,
            "signed_error_pct": 100 * (predicted / observed - 1),
            "absolute_error_pct": 100 * abs(predicted / observed - 1),
        })
    if not rows:
        raise ValueError("No unprofiled timed batches remain for validation")
    return rows


def summarize_errors(rows: list[dict]) -> dict:
    def summary(values):
        errors = sorted(row["absolute_error_pct"] for row in values)
        return {"batches": len(values), "mape_pct": statistics.mean(errors),
                "median_absolute_error_pct": statistics.median(errors),
                "p90_absolute_error_pct": errors[math.ceil(0.9 * len(errors)) - 1],
                "max_absolute_error_pct": max(errors)}
    if not rows:
        raise ValueError("Cannot summarize empty predictions")
    return {"all": summary(rows), "by_phase": {
        phase: summary([row for row in rows if row["phase"] == phase])
        for phase in sorted({row["phase"] for row in rows})
    }, "scope": "locked static batches; no serving scheduler/TTFT/throughput validation",
        "boundary_note": "Native model_time covers decoder blocks. Captured forward GPU time "
                         "also includes embedding/final norm/logits work. Profile and account for "
                         "those boundaries before interpreting this diagnostic as full-forward parity."}


def profile_file_audit(config, records) -> dict:
    """Report missing inputs before predictor initialization/training starts."""
    replica = config.cluster_config.replica_config
    predictor = config.cluster_config.execution_time_predictor_config
    if predictor.enable_dummy_mode:
        raise ValueError("Numerical replay requires dummy mode disabled")
    attributes = []
    for kernel_only in (False, True):
        active = any((r.graph_mode == "FULL") == kernel_only for r in records if not r.profiled)
        if not active:
            continue
        attributes += (["linear_op_kernel_only_input_file", "atten_kernel_only_input_file"]
                       if kernel_only else ["linear_op_input_file", "atten_input_file"])
        if replica.model_config.is_moe:
            attributes.append("moe_kernel_only_input_file" if kernel_only else "moe_input_file")
        if replica.model_config.get_num_gdn_layers():
            attributes.append("gdn_kernel_only_input_file" if kernel_only else "gdn_input_file")
    files = []
    for attribute in attributes:
        value = getattr(predictor, attribute)
        path = Path(value.replace("{DEVICE}", replica.device)
                    .replace("{MODEL}", replica.model_config.get_name())
                    .replace("{NETWORK_DEVICE}", replica.network_device))
        files.append({"input": attribute, "path": str(path), "exists": path.is_file()})
    return {"complete": all(row["exists"] for row in files), "compute_profiles": files,
            "note": "Existence only; predictor subsequently enforces schema/runtime/TP contracts. "
                    "Communication and CPU profiles are validated by their native backends."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args, remaining = parser.parse_known_args()
    if not remaining or remaining[0] != "--":
        parser.error("Pass normal Frontier configuration arguments after --")
    records = read_batches(args.ledger)
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Capture manifest is not complete")
    from frontier.config import SimulationConfig
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining[1:]]
        config = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = original_argv
    replica = config.cluster_config.replica_config
    topology = manifest["topology"]
    if (config.sys_arch != "co-location" or config.cluster_config.num_replicas != 1 or
            replica.num_pipeline_stages != 1 or replica.moe_expert_parallel_size != 1 or
            topology != {"tp": replica.attn_tensor_parallel_size, "ep": 1, "pp": 1, "nodes": 1} or
            replica.moe_tensor_parallel_size != replica.attn_tensor_parallel_size):
        raise ValueError("Replay configuration must match the captured one-node TP-only topology")
    if Path(manifest["model_path"]).name != Path(replica.model_name).name:
        raise ValueError("Replay model name does not match captured checkpoint")
    if replica.speculative_decoding_config.enabled:
        raise ValueError("Captured non-speculative batches cannot use a speculative predictor")
    if config.cluster_config.replica_scheduler_config.block_size != manifest["server_args"]["page_size"]:
        raise ValueError("Replay block size does not match captured page size")
    for device in manifest.get("devices", []):
        if replica.device.lower() not in device["device_name"].lower().replace(" ", ""):
            raise ValueError("Replay device does not match captured GPU identity")
    validate_rank_cohorts(records, tensor_parallel_size=topology["tp"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    audit = profile_file_audit(config, records)
    (output / "profile-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return
    if not audit["complete"]:
        raise ValueError(f"Missing required compute profiles; see {output / 'profile-audit.json'}")
    from frontier.entities.cluster import Cluster
    from frontier.execution_time_predictor import ExecutionTimePredictorRegistry
    from frontier.types import ClusterType
    cluster = Cluster(config.cluster_config, config.metrics_config, config.request_generator_config)
    predictor_config = config.cluster_config.execution_time_predictor_config
    predictor = ExecutionTimePredictorRegistry.get(
        predictor_config.get_type(), predictor_config, replica,
        config.cluster_config.replica_scheduler_config, config.metrics_config,
        cluster_config=config.cluster_config, cluster_type=ClusterType.MONOLITHIC,
        cc_backend=cluster.cc_backend,
    )
    rows = replay_batches(records, predictor, model_config=replica.model_config,
                          tensor_parallel_size=topology["tp"])
    with (output / "correlation.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_errors(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

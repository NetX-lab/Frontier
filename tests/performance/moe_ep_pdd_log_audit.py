"""Direct audit for one non-dummy PDD MoE EP runtime trace."""

from __future__ import annotations

import argparse
import ast
import collections
import json
import math
import re
from pathlib import Path
from typing import Any


_WORKLOAD_RE = re.compile(
    r"\[EP-WORKLOAD\]\[(?P<cluster>[^\]]+)\]\s+"
    r"batch_id=(?P<batch>\d+),\s+layer_id=(?P<layer>\d+),\s+"
    r"ep_id=(?P<ep>\d+),\s+moe_ep_size=(?P<ep_size>\d+),\s+"
    r"per_expert_tokens=(?P<tokens>\{.*?\}),\s+"
    r"lane_compute_ms=(?P<lane>[0-9.eE+-]+),\s+"
    r"routed_compute_ms=(?P<routed>[0-9.eE+-]+),\s+"
    r"lane_comm_ms=(?P<comm>[0-9.eE+-]+),\s+"
    r"replica_id=(?P<replica>\d+),\s+stage_id=(?P<stage>\d+),\s+"
    r"request_ids=(?P<request_ids>\[[^\]]*\]),\s+"
    r"request_runtime_epochs=(?P<runtime_epochs>\[[^\]]*\]),\s+"
    r"iteration_ids=(?P<iteration_ids>\[[^\]]*\]),\s+"
    r"schedule_epoch=(?P<schedule_epoch>\d+),\s+"
    r"afd_stage_idx=(?P<afd_stage_idx>-?\d+),\s+"
    r"operation_id=(?P<operation_id>\d+),\s+"
    r"operation_kind=(?P<operation_kind>[A-Za-z0-9_.-]+)"
)

_BARRIER_RE = re.compile(
    r"\[EP-BARRIER\]\[(?P<cluster>[^\]]+)\]\s+"
    r"batch_id=(?P<batch>\d+),\s+layer_id=(?P<layer>\d+),\s+"
    r"phase=(?P<phase>[A-Za-z0-9_.-]+),\s+"
    r"expected_ep_ids=(?P<expected>\[[^\]]*\]),\s+"
    r"arrived_ep_ids=(?P<arrived>\[[^\]]*\]),\s+"
    r"max_lane_time_ms=(?P<max_lane>[0-9.eE+-]+),\s+"
    r"barrier_time_ms=(?P<barrier>[0-9.eE+-]+),\s+"
    r"barrier_start_time_s=(?P<start>[0-9.eE+-]+),\s+"
    r"barrier_end_time_s=(?P<end>[0-9.eE+-]+),\s+"
    r"replica_id=(?P<replica>\d+),\s+stage_id=(?P<stage>\d+),\s+"
    r"request_ids=(?P<request_ids>\[[^\]]*\]),\s+"
    r"request_runtime_epochs=(?P<runtime_epochs>\[[^\]]*\]),\s+"
    r"iteration_ids=(?P<iteration_ids>\[[^\]]*\]),\s+"
    r"schedule_epoch=(?P<schedule_epoch>\d+),\s+"
    r"afd_stage_idx=(?P<afd_stage_idx>-?\d+),\s+"
    r"operation_id=(?P<operation_id>\d+),\s+"
    r"operation_kind=(?P<operation_kind>[A-Za-z0-9_.-]+)"
)


def _identity(groups: dict[str, str]) -> tuple[Any, ...]:
    return (
        int(groups["replica"]),
        int(groups["stage"]),
        tuple(ast.literal_eval(groups["request_ids"])),
        tuple(ast.literal_eval(groups["runtime_epochs"])),
        tuple(ast.literal_eval(groups["iteration_ids"])),
        int(groups["schedule_epoch"]),
        int(groups["afd_stage_idx"]),
        int(groups["operation_id"]),
        groups["operation_kind"],
    )


def _wave_key(groups: dict[str, str]) -> tuple[Any, ...]:
    return (
        groups["cluster"].upper(),
        int(groups["batch"]),
        int(groups["layer"]),
        _identity(groups),
    )


def _parse_log(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    workloads: list[dict[str, Any]] = []
    barriers: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        workload_match = _WORKLOAD_RE.search(line)
        if workload_match:
            groups = workload_match.groupdict()
            workloads.append(
                {
                    "line": line_number,
                    "cluster": groups["cluster"].upper(),
                    "batch": int(groups["batch"]),
                    "layer": int(groups["layer"]),
                    "ep": int(groups["ep"]),
                    "ep_size": int(groups["ep_size"]),
                    "tokens": {
                        int(key): int(value)
                        for key, value in ast.literal_eval(groups["tokens"]).items()
                    },
                    "lane_ms": float(groups["lane"]),
                    "routed_ms": float(groups["routed"]),
                    "comm_ms": float(groups["comm"]),
                    "wave": _wave_key(groups),
                }
            )

        barrier_match = _BARRIER_RE.search(line)
        if barrier_match:
            groups = barrier_match.groupdict()
            barriers.append(
                {
                    "line": line_number,
                    "cluster": groups["cluster"].upper(),
                    "batch": int(groups["batch"]),
                    "layer": int(groups["layer"]),
                    "phase": groups["phase"],
                    "expected": list(ast.literal_eval(groups["expected"])),
                    "arrived": list(ast.literal_eval(groups["arrived"])),
                    "max_lane_ms": float(groups["max_lane"]),
                    "barrier_ms": float(groups["barrier"]),
                    "start_s": float(groups["start"]),
                    "end_s": float(groups["end"]),
                    "wave": _wave_key(groups),
                }
            )

        marker = "[ROUTING-SNAPSHOT]"
        if marker in line:
            payload = json.loads(line.partition(marker)[2].strip())
            snapshots.append(payload)
    return workloads, barriers, snapshots


def _load_config(path: Path) -> dict[str, int | bool | str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    cluster = config["cluster_config"]
    prefill = cluster["prefill_replica_config"]
    decode = cluster["decode_replica_config"]
    model = prefill["model_config"]
    request = config["request_generator_config"]["length_generator_config"]
    return {
        "num_layers": int(model["num_layers"]),
        "total_experts": int(prefill["total_expert_num"]),
        "router_topk": int(prefill["router_topk"]),
        "prefill_ep_size": int(prefill["moe_expert_parallel_size"]),
        "decode_ep_size": int(decode["moe_expert_parallel_size"]),
        "prefill_tokens": int(request["prefill_tokens"]),
        "decode_tokens": int(request["decode_tokens"]),
        "num_requests": int(config["request_generator_config"]["num_requests"]),
        "dummy_mode": bool(cluster["execution_time_predictor_config"]["enable_dummy_mode"]),
        "parallel_clusters": bool(config["enable_parallel_clusters"]),
        "chunked_prefill": bool(cluster["replica_scheduler_config"]["enable_chunked_prefill"]),
        "cuda_graph_mode": str(config["decode_cuda_graph_mode"]),
    }


def _check(
    workloads: list[dict[str, Any]],
    barriers: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    config: dict[str, int | bool | str],
) -> dict[str, Any]:
    errors: list[str] = []
    num_layers = int(config["num_layers"])
    total_experts = int(config["total_experts"])
    router_topk = int(config["router_topk"])
    expected_ep_sizes = {
        "PREFILL": int(config["prefill_ep_size"]),
        "DECODE": int(config["decode_ep_size"]),
    }
    expected_batches = {"PREFILL": 1, "DECODE": int(config["decode_tokens"])}
    expected_tokens = {
        "PREFILL": int(config["prefill_tokens"]),
        "DECODE": 1,
    }

    workloads_by_wave: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    barriers_by_wave: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in workloads:
        workloads_by_wave[record["wave"]].append(record)
    for record in barriers:
        barriers_by_wave[record["wave"]].append(record)

    expected_wave_count = sum(expected_batches.values()) * num_layers
    if len(workloads) != expected_wave_count * int(config["prefill_ep_size"]):
        errors.append(
            f"workload count mismatch expected={expected_wave_count * int(config['prefill_ep_size'])} "
            f"actual={len(workloads)}"
        )
    if len(barriers) != expected_wave_count * 2:
        errors.append(
            f"barrier count mismatch expected={expected_wave_count * 2} actual={len(barriers)}"
        )
    if len(workloads_by_wave) != expected_wave_count:
        errors.append(
            f"wave count mismatch expected={expected_wave_count} actual={len(workloads_by_wave)}"
        )

    zero_lanes = 0
    zero_lane_errors: list[dict[str, Any]] = []
    ownership_errors: list[dict[str, Any]] = []
    token_errors: list[dict[str, Any]] = []
    assignment_total = 0
    for wave, records in workloads_by_wave.items():
        cluster, batch, layer, _identity_value = wave
        ep_size = expected_ep_sizes.get(cluster)
        if ep_size is None:
            errors.append(f"unexpected workload cluster={cluster}")
            continue
        expected_ids = list(range(ep_size))
        actual_ids = sorted(record["ep"] for record in records)
        if actual_ids != expected_ids:
            errors.append(
                f"participant mismatch cluster={cluster} batch={batch} layer={layer} "
                f"expected={expected_ids} actual={actual_ids}"
            )
        local_experts, remainder = divmod(total_experts, ep_size)
        if remainder:
            errors.append("total_experts is not divisible by ep_size")
        expected_assignment_count = expected_tokens[cluster] * router_topk
        observed_assignment_count = sum(
            sum(record["tokens"].values()) for record in records
        )
        assignment_total += observed_assignment_count
        if observed_assignment_count != expected_assignment_count:
            token_errors.append(
                {
                    "cluster": cluster,
                    "batch": batch,
                    "layer": layer,
                    "expected": expected_assignment_count,
                    "actual": observed_assignment_count,
                }
            )
        all_experts: list[int] = []
        for record in records:
            owner_experts = set(record["tokens"])
            expected_owner_experts = set(
                range(record["ep"] * local_experts, (record["ep"] + 1) * local_experts)
            )
            if owner_experts != expected_owner_experts:
                ownership_errors.append(
                    {
                        "cluster": cluster,
                        "batch": batch,
                        "layer": layer,
                        "ep": record["ep"],
                        "expected": sorted(expected_owner_experts),
                        "actual": sorted(owner_experts),
                    }
                )
            all_experts.extend(owner_experts)
            routed = sum(record["tokens"].values())
            if routed == 0:
                zero_lanes += 1
                if abs(record["routed_ms"]) > 1e-12:
                    zero_lane_errors.append(
                        {
                            "cluster": cluster,
                            "batch": batch,
                            "layer": layer,
                            "ep": record["ep"],
                            "routed_compute_ms": record["routed_ms"],
                        }
                    )
            if record["lane_ms"] + 1e-12 < record["routed_ms"]:
                errors.append(
                    f"lane time is below routed compute cluster={cluster} "
                    f"batch={batch} layer={layer} ep={record['ep']}"
                )
        if sorted(all_experts) != list(range(total_experts)):
            errors.append(
                f"expert domain mismatch cluster={cluster} batch={batch} layer={layer}"
            )
        if layer < 0 or layer >= num_layers:
            errors.append(f"layer out of range cluster={cluster} layer={layer}")

    barrier_equation_errors = 0
    participant_errors = 0
    ordering_errors = 0
    for wave, records in barriers_by_wave.items():
        cluster, batch, layer, _identity_value = wave
        ep_size = expected_ep_sizes[cluster]
        expected_ids = list(range(ep_size))
        phases = {record["phase"]: record for record in records}
        if set(phases) != {"dispatch", "combine"}:
            errors.append(f"barrier phase set mismatch cluster={cluster} batch={batch} layer={layer}")
        for record in records:
            if record["expected"] != expected_ids or record["arrived"] != expected_ids:
                participant_errors += 1
            expected_end = record["start_s"] + record["barrier_ms"] * 1e-3
            if abs(record["end_s"] - expected_end) > 1e-9:
                barrier_equation_errors += 1
        if "dispatch" in phases and "combine" in phases:
            if abs(phases["combine"]["start_s"] - phases["dispatch"]["end_s"]) > 1e-9:
                errors.append(
                    f"dispatch/combine timestamp gap cluster={cluster} batch={batch} layer={layer}"
                )
        workload_lines = [record["line"] for record in workloads_by_wave.get(wave, [])]
        if workload_lines and records:
            dispatch_line = phases.get("dispatch", {}).get("line", -1)
            combine_line = phases.get("combine", {}).get("line", -1)
            if max(workload_lines) >= dispatch_line or dispatch_line >= combine_line:
                ordering_errors += 1

    sequence_errors = 0
    for cluster in expected_ep_sizes:
        batch_ids = sorted(
            {wave[1] for wave in workloads_by_wave if wave[0] == cluster}
        )
        if len(batch_ids) != expected_batches[cluster]:
            sequence_errors += 1
        for batch in batch_ids:
            previous_combine_line = -1
            for layer in range(num_layers):
                waves = [
                    wave
                    for wave in workloads_by_wave
                    if wave[0] == cluster and wave[1] == batch and wave[2] == layer
                ]
                if len(waves) != 1:
                    sequence_errors += 1
                    continue
                wave = waves[0]
                first_workload_line = min(record["line"] for record in workloads_by_wave[wave])
                combine_lines = [
                    record["line"]
                    for record in barriers_by_wave[wave]
                    if record["phase"] == "combine"
                ]
                if not combine_lines or first_workload_line <= previous_combine_line:
                    sequence_errors += 1
                previous_combine_line = max(combine_lines, default=previous_combine_line)

    snapshot_ratio_error = 0.0
    snapshot_domains: set[int] = set()
    snapshot_replicas: dict[str, int] = {}
    for snapshot in snapshots:
        cluster = str(snapshot.get("cluster", "")).upper()
        details = snapshot.get("routing_details", {})
        snapshot_replicas[cluster] = len(details)
        for layers in details.values():
            for experts in layers.values():
                snapshot_domains.add(len(experts))
                snapshot_ratio_error = max(
                    snapshot_ratio_error,
                    abs(sum(float(value) for value in experts.values()) - 1.0),
                )

    errors.extend(
        [
            f"token conservation errors={len(token_errors)}"
            if token_errors
            else "",
            f"ownership errors={len(ownership_errors)}"
            if ownership_errors
            else "",
            f"zero-routed compute errors={len(zero_lane_errors)}"
            if zero_lane_errors
            else "",
            f"participant errors={participant_errors}"
            if participant_errors
            else "",
            f"barrier equation errors={barrier_equation_errors}"
            if barrier_equation_errors
            else "",
            f"workload/barrier order errors={ordering_errors}"
            if ordering_errors
            else "",
            f"layer sequence errors={sequence_errors}"
            if sequence_errors
            else "",
        ]
    )
    errors = [error for error in errors if error]
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "workload_records": len(workloads),
        "barrier_records": len(barriers),
        "wave_count": len(workloads_by_wave),
        "assignment_total": assignment_total,
        "expected_assignment_total": (
            int(config["prefill_tokens"]) * router_topk * num_layers
            + int(config["decode_tokens"]) * router_topk * num_layers
        ),
        "zero_routed_lane_records": zero_lanes,
        "zero_routed_compute_errors": len(zero_lane_errors),
        "ownership_errors": len(ownership_errors),
        "token_conservation_errors": len(token_errors),
        "participant_errors": participant_errors,
        "barrier_equation_errors": barrier_equation_errors,
        "workload_barrier_order_errors": ordering_errors,
        "layer_sequence_errors": sequence_errors,
        "snapshot_records": len(snapshots),
        "snapshot_replicas": snapshot_replicas,
        "snapshot_expert_domains": sorted(snapshot_domains),
        "snapshot_max_ratio_sum_abs_error": snapshot_ratio_error,
        "config": config,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-ledger", type=Path)
    args = parser.parse_args()

    workloads, barriers, snapshots = _parse_log(args.log)
    result = _check(workloads, barriers, snapshots, _load_config(args.config))
    if args.stage_ledger:
        rows = [
            json.loads(line)
            for line in args.stage_ledger.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result["stage_ledger_rows"] = len(rows)
        result["stage_ledger_execution_scopes"] = collections.Counter(
            str(row.get("execution_scope")) for row in rows
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

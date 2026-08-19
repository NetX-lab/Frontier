"""Contract tests for the real-data MoE EP matrix harness."""

from __future__ import annotations

import json
import re
import shlex
import threading
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import tests.e2e.moe_ep_non_dummy_matrix as matrix_module
import frontier.scheduler.cluster_scheduler.base_cluster_scheduler as cluster_scheduler_module
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import BaseClusterScheduler
from frontier.types import ClusterType
from tests.e2e.moe_ep_non_dummy_matrix import (
    _find_metrics_dir,
    _merge_result_rows,
    _parse_ep_conservation_records,
    _parse_ep_barrier_records,
    _parse_ep_workload_records,
    _validate_ep_barrier_time_equations,
    _validate_persisted_case_metadata,
    _validate_result_ledger_provenance,
    build_matrix,
    build_optimization_comparison,
    build_optimization_matrix,
    build_shell_command,
    calculate_case_cards,
    check_case_log,
    main,
    preflight_cases,
    run_cases,
    validate_case_parallel_semantics,
    validate_optimization_case,
    validate_optimization_pairs,
    validate_profile_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_minimal_profile(
    path: Path,
    *,
    measurement_type: str,
    tp_sizes: tuple[int, ...],
    ep_size: int | None = None,
    routing_runtime_path: str | None = None,
    include_mtp_columns: bool = False,
    include_mtp_same_tp_columns: bool = False,
) -> None:
    fields = [
        "profiling_precision",
        "model_arch",
        "model_architecture_profile",
        "quant_signature",
        "measurement_type",
        "num_tensor_parallel_workers",
    ]
    if ep_size is not None:
        fields.extend(
            [
                "expert_parallel_size",
                "routing_runtime_path",
                "gating_runtime_context",
                "time_stats.moe_gating_linear.median",
                "time_stats.moe_gating_routing_topk.median",
                "time_stats.moe_shuffling.median",
                "time_stats.moe_grouped_gemm.median",
            ]
        )
    if include_mtp_columns:
        fields.extend(
            [
                "time_stats.mtp_fusion_proj.median",
                "time_stats.lm_head_linear.median",
            ]
        )
    if include_mtp_same_tp_columns:
        fields.extend(
            [
                "time_stats.emb.median",
                "time_stats.input_layernorm.median",
                "time_stats.post_attention_layernorm.median",
            ]
        )

    rows = []
    for tp_size in tp_sizes:
        values = [
            "BF16",
            "generic",
            "generic",
            "none",
            measurement_type,
            str(tp_size),
        ]
        if ep_size is not None:
            values.extend(
                [
                    str(ep_size),
                    str(routing_runtime_path),
                    "standalone_legacy",
                    "1.0",
                    "2.0",
                    "3.0",
                    "4.0",
                ]
            )
        if include_mtp_columns:
            values.extend(["1.0", "2.0"])
        if include_mtp_same_tp_columns:
            values.extend(["", "", ""])
        rows.append(",".join(values))
    path.write_text(
        ",".join(fields) + "\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _write_minimal_metrics(metrics_dir: Path) -> None:
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}),
        encoding="utf-8",
    )


def _write_stage_batch_ledger(
    metrics_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "frontier_stage_batch_ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prefill_stage_ledger_row(
    *,
    batch_id: int,
    stage_id: int,
    request_ids: list[str],
    request_num_tokens: list[int],
    request_num_prefill_tokens: list[int] | None = None,
    request_runtime_epochs: list[int] | None = None,
    cluster_type: str = "MONOLITHIC",
    decode_component_ms: float = 0.0,
    iteration_ids: list[int] | None = None,
    schedule_epoch: int = 0,
    afd_stage_idx: int = -1,
    operation_id: int | None = None,
    operation_kind: str = "ep_ffn",
) -> dict[str, object]:
    if iteration_ids is None:
        iteration_ids = [0] * len(request_ids)
    if operation_id is None:
        operation_id = batch_id
    row = {
        "batch_id": batch_id,
        "stage_id": stage_id,
        "cluster_type": cluster_type,
        "replica_id": 0,
        "execution_scope": "FULL_STAGE_WORLD",
        "replica_local_id": None,
        "request_ids": request_ids,
        "request_num_tokens": request_num_tokens,
        "iteration_ids": iteration_ids,
        "schedule_epoch": schedule_epoch,
        "afd_stage_idx": afd_stage_idx,
        "operation_id": operation_id,
        "operation_kind": operation_kind,
        "execution_time": {
            "component_ledger_ms": {
                "attention_prefill_execution_time": 1.0,
                "attention_decode_execution_time": decode_component_ms,
                "attn_mla_prefill_time": 0.0,
            }
        },
    }
    row["request_num_prefill_tokens"] = (
        request_num_tokens
        if request_num_prefill_tokens is None
        else request_num_prefill_tokens
    )
    row["request_runtime_epochs"] = (
        [0] * len(request_ids)
        if request_runtime_epochs is None
        else request_runtime_epochs
    )
    return row


def _write_request_metrics(
    metrics_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    headers = [
        "Request Id",
        "request_inter_arrival_delay",
        "request_prefix_cache_hit_blocks",
        "request_spec_total_iterations",
        "request_spec_committed_tokens",
        "request_num_prefill_tokens",
        "request_prefill_preemption_count",
    ]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row.get(header, "")) for header in headers))
    (metrics_dir / "request_metrics.csv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_success_log(
    log_path: Path,
    *,
    extra_lines: list[str] | None = None,
) -> None:
    lines = [
        "Dummy Mode: false",
        "Simulation completed successfully.",
        "[OP-TRACE][MONOLITHIC][ATTENTION] batch_id=1, layer_id=0, num_tokens=1",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dense_optimization_case(**changes: object):
    case = next(
        case
        for case in build_optimization_matrix(REPO_ROOT)
        if case.architecture == "co-location"
        and case.model_kind == "dense"
        and case.optimization_stratum == "ordinary"
    )
    return replace(
        case,
        num_layers=1,
        moe_layer_ids=(),
        **changes,
    )


def _strict_ep_log(
    *,
    workload_lines: list[str],
    conservation_lines: list[str],
    barrier_lines: list[str],
) -> str:
    return "\n".join(
        [
            "Dummy Mode: false",
            "Simulation completed successfully.",
            "[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] layer_id=0",
            "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] layer_id=0",
            *[_add_test_ep_identity(line) for line in workload_lines],
            *[_add_test_ep_identity(line) for line in conservation_lines],
            *[_add_test_ep_identity(line) for line in barrier_lines],
        ]
    )


def _strict_ep_log_from_events(
    event_lines: list[str],
    *,
    layer_ids: tuple[int, ...] = (0,),
    include_pdaf_participant_map: bool = False,
    add_identity: bool = True,
) -> str:
    lines = [
        "Dummy Mode: false",
        "Simulation completed successfully.",
    ]
    for layer_id in layer_ids:
        lines.extend(
            [
                f"[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] layer_id={layer_id}",
                f"[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] layer_id={layer_id}",
            ]
        )
    if include_pdaf_participant_map:
        lines.append("per_expert_tokens extracted:")
    if add_identity:
        lines.extend(_add_test_ep_identity(line) for line in event_lines)
    else:
        lines.extend(event_lines)
    return "\n".join(lines)


def _test_ep_identity_suffix(batch_id: int, layer_id: int) -> str:
    del layer_id
    operation_id = batch_id
    return (
        ", replica_id=0, stage_id=0, request_ids=[0], "
        "request_runtime_epochs=[0], iteration_ids=[0], "
        f"schedule_epoch=0, afd_stage_idx=-1, operation_id={operation_id}, "
        "operation_kind=ep_ffn"
    )


def _add_test_ep_identity(line: str) -> str:
    if not line.startswith("[EP-"):
        return line
    if "replica_id=" in line:
        return line
    batch_match = re.search(r"batch_id=(\d+)", line)
    layer_match = re.search(r"layer_id=(\d+)", line)
    if batch_match is None or layer_match is None:
        return line
    return line + _test_ep_identity_suffix(
        int(batch_match.group(1)),
        int(layer_match.group(1)),
    )


def _ep_wave_lines(
    *,
    cluster: str,
    ep_size: int,
    batch_id: int,
    layer_id: int,
    include_identity: bool = True,
) -> dict[str, list[str] | str]:
    ep_ids = list(range(ep_size))
    workloads = [
        f"[EP-WORKLOAD][{cluster}] batch_id={batch_id}, layer_id={layer_id}, "
        f"ep_id={ep_id}, moe_ep_size={ep_size}, "
        f"per_expert_tokens={{{ep_id}: 1}}, "
        "lane_compute_ms=1.0, lane_comm_ms=0.0"
        for ep_id in ep_ids
    ]
    per_ep_tokens = "{" + ", ".join(f"{ep_id}: 1" for ep_id in ep_ids) + "}"
    conservation = (
        f"[EP-CONSERVATION][{cluster}] batch_id={batch_id}, "
        f"layer_id={layer_id}, routing_token_count={ep_size}, router_topk=1, "
        f"total_routed_assignments={ep_size}, "
        f"per_ep_routed_tokens={per_ep_tokens}"
    )
    dispatch = (
        f"[EP-BARRIER][{cluster}] batch_id={batch_id}, layer_id={layer_id}, "
        f"phase=dispatch, expected_ep_ids={ep_ids}, arrived_ep_ids={ep_ids}, "
        "max_lane_time_ms=1.0, barrier_time_ms=1.0, "
        "barrier_end_time_s=0.001"
    )
    combine = (
        f"[EP-BARRIER][{cluster}] batch_id={batch_id}, layer_id={layer_id}, "
        f"phase=combine, expected_ep_ids={ep_ids}, arrived_ep_ids={ep_ids}, "
        "max_lane_time_ms=1.0, barrier_time_ms=1.0, "
        "barrier_end_time_s=0.002"
    )
    if include_identity:
        suffix = _test_ep_identity_suffix(batch_id, layer_id)
        workloads = [line + suffix for line in workloads]
        conservation += suffix
        dispatch += suffix
        combine += suffix
    return {
        "workloads": workloads,
        "conservation": conservation,
        "dispatch": dispatch,
        "combine": combine,
    }


def _single_layer_ep2_case(*, zero_routed: bool = False):
    return next(
        replace(case, num_layers=1, moe_layer_ids=(0,))
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location"
        and case.model_kind == "moe"
        and case.ep_size == 2
        and (not zero_routed or case.workload_kind == "zero-routed")
    )


def _zero_routed_checker_case():
    """Return a small case whose ownership and Hamilton counts are explicit."""

    return replace(
        _single_layer_ep2_case(zero_routed=True),
        total_experts=4,
        router_topk=2,
        routing_distribution="balanced",
    )


def _independent_ep_wave_fixture(
    tmp_path: Path,
    *,
    batch_ids: tuple[int, ...] = (10,),
) -> tuple[object, Path, Path]:
    """Create a strict one-layer EP fixture backed by an independent ledger."""

    case = replace(
        _single_layer_ep2_case(),
        total_experts=2,
        router_topk=1,
        routing_distribution="balanced",
        num_requests=1,
        pipeline_stages=1,
        prefill_tokens=2,
    )
    log_path = tmp_path / "ep_wave.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=batch_id,
                stage_id=0,
                request_ids=["0"],
                request_num_tokens=[2],
                request_num_prefill_tokens=[2],
                request_runtime_epochs=[0],
            )
            for batch_id in batch_ids
        ],
    )
    routing_input = matrix_module._build_routing_input_ledger(
        case,
        source_provenance=matrix_module._source_provenance(REPO_ROOT),
    )
    routing_input["expected_routing_token_totals"] = [
        {
            "cluster": "MONOLITHIC",
            "replica_id": 0,
            "layer_id": 0,
            "routing_token_count": 2,
        }
    ]
    (metrics_dir / "frontier_routing_input_ledger.json").write_text(
        json.dumps(routing_input),
        encoding="utf-8",
    )
    event_lines: list[str] = []
    for batch_id in batch_ids:
        wave = _ep_wave_lines(
            cluster="MONOLITHIC",
            ep_size=2,
            batch_id=batch_id,
            layer_id=0,
        )
        event_lines.extend(
            [
                str(wave["conservation"]),
                *list(wave["workloads"]),
                str(wave["dispatch"]),
                str(wave["combine"]),
            ]
        )
    log_path.write_text(
        _strict_ep_log_from_events(event_lines),
        encoding="utf-8",
    )
    return case, log_path, metrics_dir


def _routing_snapshot_line(case: object, *, mutate_first_ratio: bool = False) -> str:
    snapshot = matrix_module._expected_routing_details_snapshot(case)
    details: dict[str, dict[str, dict[str, float]]] = {"0": {}}
    for entry in snapshot:
        ratios = list(entry["ratios"])
        if mutate_first_ratio and int(entry["layer_id"]) == 0:
            ratios[0] += 0.01
            ratios[1] -= 0.01
        details["0"][str(entry["layer_id"])] = {
            str(expert_id): float(ratio)
            for expert_id, ratio in enumerate(ratios)
        }
    return (
        "[ROUTING-SNAPSHOT] "
        + json.dumps(
            {
                "schema_version": 1,
                "cluster": "MONOLITHIC",
                "routing_details": details,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def test_strict_checker_binds_runtime_routing_snapshot_to_independent_sidecar(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    log_path.write_text(
        log_path.read_text(encoding="utf-8")
        + "\n"
        + _routing_snapshot_line(case)
        + "\n",
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
        require_routing_details_oracle=True,
    )

    assert result["status"] == "PASS", result["errors"]
    assert result["routing_snapshot_records"] == 1


def test_strict_checker_rejects_mutated_runtime_routing_snapshot(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    log_path.write_text(
        log_path.read_text(encoding="utf-8")
        + "\n"
        + _routing_snapshot_line(case, mutate_first_ratio=True)
        + "\n",
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
        require_routing_details_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "runtime routing snapshot ratio mismatch" in result["errors"]


def test_strict_checker_rejects_missing_runtime_routing_snapshot(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
        require_routing_details_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "missing runtime routing snapshot" in result["errors"]


def test_matrix_has_required_cross_architecture_coverage() -> None:
    cases = build_matrix(REPO_ROOT)

    assert len(cases) >= 100
    assert Counter(case.architecture for case in cases) == Counter(
        {
            "co-location": 50,
            "pd-disaggregation": 50,
            "pd-af-disaggregation": 10,
        }
    )
    assert {case.model_kind for case in cases} == {"dense", "moe", "mixed"}
    assert {case.routing_distribution for case in cases if case.model_kind != "dense"} >= {
        "balanced",
        "random",
        "skewed",
        "zipf",
    }
    assert all(
        case.model_name == "step-moe-noquant-small"
        and case.routing_distribution == "random"
        for case in cases
        if case.model_kind == "mixed"
    )
    assert all(
        case.model_name == "qwen3-a3b-30b-moe"
        and case.device == "a800"
        for case in cases
        if case.model_kind == "moe" and case.routing_distribution != "random"
    )
    assert {case.ep_size for case in cases if case.model_kind != "dense"} >= {1, 2, 4}
    assert {case.workload_kind for case in cases} >= {
        "prefill-heavy",
        "decode-heavy",
        "mixed",
        "zero-routed",
    }


def test_optimization_matrix_has_exact_required_marginals() -> None:
    cases = build_optimization_matrix(REPO_ROOT)

    assert len(cases) == 200
    assert Counter(case.architecture for case in cases) == Counter(
        {
            "co-location": 91,
            "pd-disaggregation": 91,
            "pd-af-disaggregation": 18,
        }
    )
    assert Counter(case.simulation_mode for case in cases) == Counter(
        {"offline": 100, "online": 100}
    )
    assert Counter(case.total_cards for case in cases) == Counter({8: 100, 32: 100})
    assert Counter(case.model_kind for case in cases) == Counter(
        {"dense": 67, "moe": 67, "mixed": 66}
    )
    assert Counter(case.enable_chunked_prefill for case in cases) == Counter(
        {False: 100, True: 100}
    )
    assert Counter(
        case.routing_distribution for case in cases if case.is_moe
    ) == Counter(
        {
            "random": 34,
            "balanced": 33,
            "skewed": 33,
            "zipf": 33,
        }
    )


def test_optimization_matrix_effective_configs_are_unique() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    effective_keys = []
    for case in cases:
        payload = asdict(case)
        for ignored_field in (
            "case_id",
            "baseline_case_id",
            "seed",
            "optimization_stratum",
            "pair_id",
            "comparison_group_id",
            "pair_role",
        ):
            payload.pop(ignored_field, None)
        effective_keys.append(json.dumps(payload, sort_keys=True))

    assert len(set(effective_keys)) == 200


def test_optimization_matrix_rejects_unsupported_combinations() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    for case in cases:
        validate_optimization_case(case)

    pdaf_case = next(
        case for case in cases if case.architecture == "pd-af-disaggregation"
    )
    mtp_case = next(
        case for case in cases if case.optimization_stratum == "mtp"
    )
    prefix_case = next(
        case for case in cases if case.optimization_stratum == "prefix"
    )
    shared_case = next(
        case for case in cases if case.architecture == "co-location"
    )

    with pytest.raises(ValueError, match="PD-AF.*Prefix"):
        validate_optimization_case(
            replace(pdaf_case, enable_prefix_caching=True)
        )
    with pytest.raises(ValueError, match="MTP.*CUDA Graph"):
        validate_optimization_case(
            replace(mtp_case, decode_cuda_graph_mode="full_decode_only")
        )
    with pytest.raises(ValueError, match="Prefix.*MTP"):
        validate_optimization_case(
            replace(prefix_case, enable_mtp=True)
        )
    with pytest.raises(ValueError, match="global CUDA Graph"):
        validate_optimization_case(replace(shared_case, use_cuda_graph=True))
    with pytest.raises(ValueError, match="PD-AF.*decode CUDA Graph"):
        validate_optimization_case(
            replace(pdaf_case, decode_cuda_graph_mode="piecewise")
        )
    with pytest.raises(ValueError, match="MTP.*random routing"):
        validate_optimization_case(
            replace(mtp_case, routing_distribution="balanced")
        )
    with pytest.raises(ValueError, match="PD-AF.*pipeline stages"):
        validate_optimization_case(
            replace(pdaf_case, pipeline_stages=2)
        )


def test_optimization_matrix_recomputes_exact_topology_and_pp_marginals() -> None:
    cases = build_optimization_matrix(REPO_ROOT)

    assert Counter(case.pipeline_stages for case in cases) == Counter({1: 140, 2: 60})
    assert all(
        case.pipeline_stages == 1
        for case in cases
        if case.architecture == "pd-af-disaggregation"
    )
    assert all(calculate_case_cards(case) == case.total_cards for case in cases)
    assert all(case.total_cards in {8, 32} for case in cases)
    assert all(
        case.total_cards == 32
        for case in cases
        if case.architecture == "pd-af-disaggregation"
        and case.model_kind == "mixed"
    )

    pdd_relations = Counter(
        "equal"
        if case.prefill_replicas == case.decode_replicas
        else (
            "prefill_gt"
            if case.prefill_replicas > case.decode_replicas
            else "prefill_lt"
        )
        for case in cases
        if case.architecture == "pd-disaggregation"
    )
    assert set(pdd_relations) == {"equal", "prefill_gt", "prefill_lt"}

    pdaf_relations = Counter(
        "equal"
        if case.decode_attn_replicas == case.decode_ffn_replicas
        else (
            "attn_gt"
            if case.decode_attn_replicas > case.decode_ffn_replicas
            else "attn_lt"
        )
        for case in cases
        if case.architecture == "pd-af-disaggregation"
    )
    assert pdaf_relations == Counter({"equal": 6, "attn_gt": 6, "attn_lt": 6})


def test_zero_routed_cases_are_mathematically_guaranteed() -> None:
    zero_routed_cases = [
        case
        for case in build_optimization_matrix(REPO_ROOT)
        if case.expects_zero_routed_lane
    ]

    assert zero_routed_cases
    assert all(
        case.prefill_tokens * case.router_topk < case.ep_size
        for case in zero_routed_cases
    )


def test_optimization_cases_respect_production_prefill_request_contract() -> None:
    cases = build_optimization_matrix(REPO_ROOT)

    assert cases
    assert all(case.prefill_tokens > 1 for case in cases)
    with pytest.raises(ValueError, match="prefill_tokens must be >1"):
        validate_optimization_case(replace(cases[0], prefill_tokens=1))


def test_online_optimization_cases_can_emit_inter_arrival_evidence() -> None:
    online_cases = [
        case
        for case in build_optimization_matrix(REPO_ROOT)
        if case.simulation_mode == "online"
    ]

    assert online_cases
    assert all(case.num_requests > 1 for case in online_cases)
    with pytest.raises(ValueError, match="online.*at least two requests"):
        validate_optimization_case(replace(online_cases[0], num_requests=1))


def test_optimization_case_rejects_unprovable_zero_routed_workload() -> None:
    zero_routed_case = next(
        case
        for case in build_optimization_matrix(REPO_ROOT)
        if case.expects_zero_routed_lane
    )

    with pytest.raises(ValueError, match="cannot guarantee a zero-routed EP lane"):
        validate_optimization_case(
            replace(
                zero_routed_case,
                prefill_tokens=zero_routed_case.ep_size
                // zero_routed_case.router_topk,
            )
        )


def test_optimization_pairs_change_only_the_declared_fields() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    validate_optimization_pairs(cases)

    prefix_pair_id = next(
        case.pair_id
        for case in cases
        if case.optimization_stratum == "prefix"
    )
    corrupted = [
        (
            replace(case, routing_distribution="random")
            if case.pair_id == prefix_pair_id and case.pair_role == "enabled"
            else case
        )
        for case in cases
    ]
    with pytest.raises(ValueError, match="undeclared fields.*routing_distribution"):
        validate_optimization_pairs(corrupted)


def test_optimization_pair_manifest_has_exact_expected_set() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    specs = matrix_module._expected_optimization_pair_specs(cases)

    assert len(specs) == 122
    assert Counter(spec["optimization"] for spec in specs) == Counter(
        {
            "cuda_graph": 74,
            "chunked_prefill": 12,
            "prefix_cache": 22,
            "mtp": 14,
        }
    )
    assert len(
        {
            (
                spec["comparison_id"],
                spec["control"].case_id,
                spec["enabled"].case_id,
            )
            for spec in specs
        }
    ) == 122


def test_optimization_pair_validation_rejects_split_factorial_group() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    target = next(
        case
        for case in cases
        if case.comparison_group_id == "co_location_ordinary_graph_00"
        and case.case_id.endswith("piecewise_chunk_on")
    )
    corrupted = [
        replace(
            case,
            comparison_group_id="unexpected_singleton",
            baseline_case_id=case.case_id,
            pair_role="standalone",
        )
        if case.case_id == target.case_id
        else case
        for case in cases
    ]

    with pytest.raises(
        ValueError,
        match="undeclared singleton|pair set mismatch",
    ):
        validate_optimization_pairs(corrupted)


def test_ordinary_optimization_groups_cover_exact_graph_chunk_factorials() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    groups: dict[str, list[object]] = {}
    for case in cases:
        if (
            case.optimization_stratum == "ordinary"
            and case.comparison_group_id is not None
        ):
            groups.setdefault(case.comparison_group_id, []).append(case)

    factorial_groups = [
        group
        for group in groups.values()
        if {
            (case.decode_cuda_graph_mode, case.enable_chunked_prefill)
            for case in group
        }
        == {
            ("none", False),
            ("none", True),
            ("full_decode_only", False),
            ("full_decode_only", True),
            ("piecewise", False),
            ("piecewise", True),
        }
    ]

    assert len(factorial_groups) >= 4
    for group in factorial_groups:
        control = next(
            case
            for case in group
            if case.decode_cuda_graph_mode == "none"
            and not case.enable_chunked_prefill
        )
        for candidate in group:
            differences = {
                field_name
                for field_name in asdict(control)
                if field_name
                not in {
                    "case_id",
                    "baseline_case_id",
                    "pair_id",
                    "comparison_group_id",
                    "pair_role",
                    "decode_cuda_graph_mode",
                    "enable_chunked_prefill",
                }
                and getattr(control, field_name) != getattr(candidate, field_name)
            }
            assert not differences


def _optimization_result_row(case, *, metric_offset: float) -> dict[str, object]:
    check: dict[str, object] = {
        "status": "PASS",
        "errors": "",
        "ttft_mean_ms": 10.0 + metric_offset,
        "tpot_mean_ms": 2.0 + metric_offset,
        "e2e_mean_ms": 20.0 + metric_offset,
        "layer_ids": list(range(case.num_layers)),
        "moe_trace_count": len(case.moe_layer_ids) if case.is_moe else 0,
        "ep_workload_records": 1 if case.is_moe else 0,
        "ep_barrier_records": 2 if case.is_moe else 0,
        "ep_conservation_records": 1 if case.is_moe else 0,
        "chunked_prefill_request_token_totals": {
            str(request_id): case.prefill_tokens
            for request_id in range(case.num_requests)
        },
    }
    if case.decode_cuda_graph_mode != "none" or case.use_cuda_graph:
        check["cuda_graph_capture_count"] = 2
        check["cuda_graph_capture_roles"] = (
            ["DECODE_ATTN", "DECODE_FFN"]
            if case.architecture == "pd-af-disaggregation"
            else [
                "MONOLITHIC"
                if case.architecture == "co-location"
                else "DECODE"
            ]
        )
    if case.enable_chunked_prefill:
        check["chunked_prefill_split_count"] = 2
    if case.enable_prefix_caching:
        check["prefix_cache_hit_blocks"] = 4
    if case.enable_mtp:
        check["spec_decode_iterations"] = 3
        check["spec_decode_committed_tokens"] = 5
    return {
        "case_id": case.case_id,
        "case": json.loads(json.dumps(asdict(case), sort_keys=True)),
        "status": "PASS",
        "check": check,
    }


def test_optimization_comparison_expands_factorial_axes_and_reports_metrics() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    factorial_group_id = next(
        case.comparison_group_id
        for case in cases
        if case.optimization_stratum == "ordinary"
        and case.comparison_group_id is not None
        and case.decode_cuda_graph_mode == "piecewise"
        and case.enable_chunked_prefill
        and sum(
            candidate.comparison_group_id == case.comparison_group_id
            for candidate in cases
        )
        == 6
    )
    group = [
        case for case in cases if case.comparison_group_id == factorial_group_id
    ]
    result_rows = [
        _optimization_result_row(case, metric_offset=float(index))
        for index, case in enumerate(group)
    ]

    report = build_optimization_comparison(group, result_rows)

    assert report["status"] == "PASS"
    assert report["pair_count"] == 7
    assert report["failed_pair_count"] == 0
    assert report["latency_oracle"] == "report_only"
    assert Counter(pair["optimization"] for pair in report["pairs"]) == Counter(
        {"cuda_graph": 4, "chunked_prefill": 3}
    )
    for pair in report["pairs"]:
        assert pair["status"] == "PASS"
        assert pair["changed_fields"] == [pair["target_field"]]
        assert pair["workflow"]["control"]["layer_ids"] == pair["workflow"]["enabled"][
            "layer_ids"
        ]
        assert pair["workflow"]["control"]["ep_barrier_records"] >= 0
        assert pair["workflow"]["enabled"]["ep_barrier_records"] >= 0
        for metric in ("ttft_mean_ms", "tpot_mean_ms", "e2e_mean_ms"):
            values = pair["metrics"][metric]
            assert values["control_ms"] is not None
            assert values["enabled_ms"] is not None
            assert values["delta_ms"] == pytest.approx(
                values["enabled_ms"] - values["control_ms"]
            )


def test_optimization_comparison_rejects_incomplete_campaign_when_required() -> None:
    all_cases = build_optimization_matrix(REPO_ROOT)
    reduced_cases = all_cases[:-1]
    result_rows = [
        _optimization_result_row(case, metric_offset=float(index))
        for index, case in enumerate(reduced_cases)
    ]

    with pytest.raises(
        ValueError,
        match=r"requires the complete matrix: expected=200 actual=199",
    ):
        build_optimization_comparison(
            reduced_cases,
            result_rows,
            require_complete_matrix=True,
        )


def test_optimization_comparison_rejects_layer_workflow_mismatch() -> None:
    all_cases = build_optimization_matrix(REPO_ROOT)
    pair_id = next(
        case.pair_id
        for case in all_cases
        if case.optimization_stratum == "prefix"
    )
    cases = [case for case in all_cases if case.pair_id == pair_id]
    result_rows = [
        _optimization_result_row(case, metric_offset=float(index))
        for index, case in enumerate(cases)
    ]
    enabled_row = next(
        row for row in result_rows if row["case_id"].endswith("_enabled")
    )
    enabled_row["check"]["layer_ids"] = [0]

    report = build_optimization_comparison(cases, result_rows)

    assert report["status"] == "FAIL"
    assert report["failed_pair_count"] == 1
    assert "layer identity differs" in report["pairs"][0]["errors"]


def test_chunked_prefill_pair_rejects_request_token_conservation_mismatch() -> None:
    all_cases = build_optimization_matrix(REPO_ROOT)
    group_id = next(
        case.comparison_group_id
        for case in all_cases
        if case.optimization_stratum == "ordinary"
        and case.comparison_group_id is not None
        and case.enable_chunked_prefill
    )
    cases = [
        case
        for case in all_cases
        if case.comparison_group_id == group_id
        and case.decode_cuda_graph_mode == "none"
    ]
    assert len(cases) == 2
    result_rows = [
        _optimization_result_row(case, metric_offset=float(index))
        for index, case in enumerate(cases)
    ]
    enabled_row = next(
        row for row in result_rows if row["case"]["enable_chunked_prefill"]
    )
    enabled_row["check"]["chunked_prefill_request_token_totals"]["0"] += 1

    report = build_optimization_comparison(cases, result_rows)

    assert report["status"] == "FAIL"
    assert report["failed_pair_count"] == 1
    assert "Chunked Prefill request token conservation differs" in report["pairs"][
        0
    ]["errors"]


def test_optimization_comparison_fails_only_the_pair_missing_activation() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    pair_id = next(
        case.pair_id
        for case in cases
        if case.optimization_stratum == "prefix"
    )
    pair_cases = [case for case in cases if case.pair_id == pair_id]
    result_rows = [
        _optimization_result_row(case, metric_offset=float(index))
        for index, case in enumerate(pair_cases)
    ]
    enabled_row = next(
        row
        for row in result_rows
        if row["case_id"].endswith("_enabled")
    )
    enabled_row["check"].pop("prefix_cache_hit_blocks")

    report = build_optimization_comparison(pair_cases, result_rows)

    assert report["status"] == "FAIL"
    assert report["pair_count"] == 1
    assert report["failed_pair_count"] == 1
    assert report["pairs"][0]["status"] == "FAIL"
    assert "Prefix Cache activation" in report["pairs"][0]["errors"]
    assert report["pairs"][0]["metrics"]["ttft_mean_ms"]["delta_ms"] is not None


def test_optimization_comparison_does_not_use_missing_latency_as_verdict() -> None:
    all_cases = build_optimization_matrix(REPO_ROOT)
    pair_id = next(
        case.pair_id
        for case in all_cases
        if case.optimization_stratum == "prefix"
    )
    cases = [case for case in all_cases if case.pair_id == pair_id]
    result_rows = [
        _optimization_result_row(case, metric_offset=float(index))
        for index, case in enumerate(cases)
    ]
    for row in result_rows:
        row["check"]["tpot_mean_ms"] = None

    report = build_optimization_comparison(cases, result_rows)

    assert report["status"] == "PASS"
    assert report["failed_pair_count"] == 0
    assert report["pairs"][0]["metrics"]["tpot_mean_ms"] == {
        "control_ms": None,
        "enabled_ms": None,
        "delta_ms": None,
        "relative_delta_percent": None,
    }


def test_optimization_comparison_rejects_result_case_metadata_mismatch() -> None:
    all_cases = build_optimization_matrix(REPO_ROOT)
    pair_id = next(
        case.pair_id
        for case in all_cases
        if case.optimization_stratum == "prefix"
    )
    cases = [case for case in all_cases if case.pair_id == pair_id]
    result_rows = [
        _optimization_result_row(case, metric_offset=float(index))
        for index, case in enumerate(cases)
    ]
    result_rows[0]["case"] = {
        **result_rows[0]["case"],
        "seed": int(result_rows[0]["case"]["seed"]) + 1,
    }

    with pytest.raises(ValueError, match="result-row case metadata mismatch"):
        build_optimization_comparison(cases, result_rows)


def test_optimization_comparison_rejects_persisted_case_metadata_mismatch(
    tmp_path: Path,
) -> None:
    case = next(
        case
        for case in build_optimization_matrix(REPO_ROOT)
        if case.optimization_stratum == "prefix"
    )
    case_dir = tmp_path / case.case_id
    case_dir.mkdir()
    log_path = case_dir / f"{case.case_id}.log"
    log_path.write_text("", encoding="utf-8")
    (case_dir / "case_metadata.json").write_text(
        json.dumps(
            {
                "case": {
                    **asdict(case),
                    "seed": case.seed + 1,
                }
            }
        ),
        encoding="utf-8",
    )
    result_row = {
        **_optimization_result_row(case, metric_offset=0.0),
        "log_path": str(log_path),
    }

    with pytest.raises(ValueError, match="persisted case metadata mismatch"):
        _validate_persisted_case_metadata([case], [result_row])


def test_build_optimization_matrix_validates_pair_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated: list[list[object]] = []

    def record_validation(cases) -> None:
        validated.append(cases)

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.validate_optimization_pairs",
        record_validation,
    )

    cases = build_optimization_matrix(REPO_ROOT)

    assert validated == [cases]
    assert validated[0] is cases


def test_matrix_uses_frontier_vllm_parallel_semantics() -> None:
    for case in build_matrix(REPO_ROOT):
        validate_case_parallel_semantics(case)


def test_matrix_enforces_dense_topology_and_card_limit() -> None:
    cases = build_matrix(REPO_ROOT)

    assert all(case.total_cards <= 32 for case in cases)
    assert all(case.total_cards > 0 for case in cases)
    assert all(case.prefill_tokens > 1 for case in cases)
    assert all(case.ep_size == 1 for case in cases if case.model_kind == "dense")
    assert all(
        case.moe_tensor_parallel_size == (4 if case.model_kind == "mixed" else 1)
        for case in cases
    )
    assert all(case.pipeline_stages == 1 for case in cases)


def test_mixed_matrix_shapes_stay_within_step_profile_tp_domain() -> None:
    mixed_cases = [case for case in build_matrix(REPO_ROOT) if case.model_kind == "mixed"]

    assert mixed_cases
    assert all(case.moe_tensor_parallel_size == 4 for case in mixed_cases)
    assert all(case.ep_size <= 2 for case in mixed_cases)
    assert all(case.attn_tensor_parallel_size <= 8 for case in mixed_cases)


def test_non_dummy_command_has_no_dummy_switch() -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "moe"
    )

    command, env = build_shell_command(case, REPO_ROOT, Path("/data/ycfeng/tmp/matrix"))

    assert "--random_forrest_execution_time_predictor_config_enable_dummy_mode" not in command
    assert "--replica_config_device h800" in command
    assert env["ENABLE_DUMMY_MODE"] == "false"
    assert env["DECODE_CUDA_GRAPH_MODE"] == "none"


def test_optimization_wrappers_log_dummy_mode_state() -> None:
    output_root = Path("/data/ycfeng/tmp/optimization-matrix")
    scripts = {
        Path(shlex.split(build_shell_command(case, REPO_ROOT, output_root)[0])[1])
        for case in build_optimization_matrix(REPO_ROOT)
        if case.architecture == "co-location"
        and case.optimization_stratum in {"prefix", "mtp"}
    }

    missing = [
        str(script.relative_to(REPO_ROOT))
        for script in sorted(scripts)
        if "Dummy Mode: $ENABLE_DUMMY_MODE"
        not in script.read_text(encoding="utf-8")
    ]

    assert missing == []


def test_non_dummy_command_uses_output_scoped_predictor_cache(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "moe"
    )
    output_root = tmp_path / "matrix-output"

    command, _ = build_shell_command(case, REPO_ROOT, output_root)

    expected_cache = (output_root / "_predictor_cache").resolve()
    assert "--metrics_config_cache_dir" in command
    assert str(expected_cache) in command


def test_optimization_command_selects_mode_and_feature_recipes() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    selected = {
        "online_ordinary": next(
            case
            for case in cases
            if case.optimization_stratum == "ordinary"
            and case.simulation_mode == "online"
            and case.architecture == "co-location"
            and case.model_kind == "dense"
        ),
        "offline_prefix": next(
            case
            for case in cases
            if case.optimization_stratum == "prefix"
            and case.simulation_mode == "offline"
            and case.architecture == "co-location"
        ),
        "online_prefix": next(
            case
            for case in cases
            if case.optimization_stratum == "prefix"
            and case.simulation_mode == "online"
            and case.architecture == "pd-disaggregation"
        ),
        "offline_mtp": next(
            case
            for case in cases
            if case.optimization_stratum == "mtp"
            and case.simulation_mode == "offline"
            and case.architecture == "co-location"
        ),
        "online_mtp": next(
            case
            for case in cases
            if case.optimization_stratum == "mtp"
            and case.simulation_mode == "online"
            and case.architecture == "pd-disaggregation"
        ),
        "online_pdaf": next(
            case
            for case in cases
            if case.architecture == "pd-af-disaggregation"
            and case.simulation_mode == "online"
            and case.model_kind == "moe"
        ),
    }
    expected_suffixes = {
        "online_ordinary": "co-location/online/dense_model_basic_online.sh",
        "offline_prefix": "co-location/offline/moe_prefix_caching.sh",
        "online_prefix": "pdd/online/moe_prefix_caching_online.sh",
        "offline_mtp": "co-location/offline/moe_spec_dec.sh",
        "online_mtp": "pdd/online/moe_spec_dec_online.sh",
        "online_pdaf": "pd-af-disagg/online/moe_model_ep_online.sh",
    }

    for label, case in selected.items():
        command, _ = build_shell_command(
            case,
            REPO_ROOT,
            Path("/data/ycfeng/tmp/optimization-matrix"),
        )
        script = Path(shlex.split(command)[1])
        assert script.as_posix().endswith(expected_suffixes[label])


def test_optimization_command_materializes_declared_controls() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    prefix_control = next(
        case
        for case in cases
        if case.optimization_stratum == "prefix"
        and case.pair_role == "control"
    )
    prefix_enabled = next(
        case
        for case in cases
        if case.pair_id == prefix_control.pair_id
        and case.pair_role == "enabled"
    )
    mtp_control = next(
        case
        for case in cases
        if case.optimization_stratum == "mtp"
        and case.pair_role == "control"
    )
    mtp_enabled = next(
        case
        for case in cases
        if case.pair_id == mtp_control.pair_id
        and case.pair_role == "enabled"
    )
    pdaf_graph = next(
        case
        for case in cases
        if case.architecture == "pd-af-disaggregation" and case.use_cuda_graph
    )
    decode_graph = next(
        case
        for case in cases
        if case.architecture == "co-location"
        and case.decode_cuda_graph_mode == "piecewise"
    )
    chunked_prefill = next(case for case in cases if case.enable_chunked_prefill)
    output_root = Path("/data/ycfeng/tmp/optimization-matrix")

    prefix_control_command, prefix_control_env = build_shell_command(
        prefix_control, REPO_ROOT, output_root
    )
    prefix_enabled_command, prefix_enabled_env = build_shell_command(
        prefix_enabled, REPO_ROOT, output_root
    )
    mtp_control_command, mtp_control_env = build_shell_command(
        mtp_control, REPO_ROOT, output_root
    )
    mtp_enabled_command, mtp_enabled_env = build_shell_command(
        mtp_enabled, REPO_ROOT, output_root
    )
    _, pdaf_graph_env = build_shell_command(pdaf_graph, REPO_ROOT, output_root)
    _, decode_graph_env = build_shell_command(decode_graph, REPO_ROOT, output_root)
    _, chunked_prefill_env = build_shell_command(
        chunked_prefill, REPO_ROOT, output_root
    )

    assert prefix_control_env["TRACE_FILE"] == str(
        REPO_ROOT / "examples/fixtures/prefix_cache_shared_session_trace.csv"
    )
    assert prefix_enabled_env["TRACE_FILE"] == prefix_control_env["TRACE_FILE"]
    assert "--no-vllm_v1_scheduler_config_enable_prefix_caching" in (
        prefix_control_command
    )
    assert "--vllm_v1_scheduler_config_enable_prefix_caching" in (
        prefix_enabled_command
    )
    assert mtp_control_env["SPEC_METHOD"] == "qwen3_next_mtp"
    assert mtp_enabled_env["SPEC_METHOD"] == mtp_control_env["SPEC_METHOD"]
    assert int(mtp_enabled_env["MTP_N_PREDICT"]) > 0
    assert int(mtp_enabled_env["MTP_NUM_LAYERS"]) > 0
    assert "--no-speculative_decoding_config_enabled" in mtp_control_command
    assert "--speculative_decoding_config_enabled" in mtp_enabled_command
    assert prefix_control_env["MAX_TOKENS_IN_BATCH"] == "32"
    assert prefix_enabled_env["MAX_TOKENS_IN_BATCH"] == "32"
    assert int(chunked_prefill_env["MAX_TOKENS_IN_BATCH"]) == min(
        64,
        chunked_prefill.prefill_tokens // 2,
    )
    assert chunked_prefill_env["LONG_PREFILL_TOKEN_THRESHOLD"] == "16"
    assert pdaf_graph_env["ENABLE_CUDA_GRAPH"] == "true"
    assert decode_graph_env["DECODE_CUDA_GRAPH_MODE"] == "piecewise"
    assert chunked_prefill_env["ENABLE_CHUNKED_PREFILL"] == "true"

    chunked_disabled = next(
        case
        for case in cases
        if not case.enable_chunked_prefill
        and case.optimization_stratum == "ordinary"
    )
    _, chunked_disabled_env = build_shell_command(
        chunked_disabled, REPO_ROOT, output_root
    )
    assert chunked_disabled_env["LONG_PREFILL_TOKEN_THRESHOLD"] == "0"


def test_optimization_command_batch_budget_is_schedulable_and_pair_stable() -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    output_root = Path("/data/ycfeng/tmp/optimization-matrix")
    budgets: dict[str, int] = {}

    for case in cases:
        _, env = build_shell_command(case, REPO_ROOT, output_root)
        budget = int(env["MAX_TOKENS_IN_BATCH"])
        budgets[case.case_id] = budget
        if case.enable_chunked_prefill:
            assert 0 < budget < case.prefill_tokens, case.case_id
        elif case.optimization_stratum != "prefix":
            assert budget >= case.prefill_tokens, case.case_id

    groups: dict[tuple[str, bool], list[object]] = {}
    for case in cases:
        group_id = case.pair_id or case.comparison_group_id
        if group_id is not None:
            group_key = (group_id, case.enable_chunked_prefill)
            groups.setdefault(group_key, []).append(case)

    for (group_id, chunked_prefill_enabled), group in groups.items():
        if len(group) > 1:
            assert len({budgets[case.case_id] for case in group}) == 1, (
                group_id,
                chunked_prefill_enabled,
            )


def test_chunked_prefill_command_budget_forces_multiple_prefill_chunks() -> None:
    for case in build_optimization_matrix(REPO_ROOT):
        if not case.enable_chunked_prefill:
            continue
        _, env = build_shell_command(
            case,
            REPO_ROOT,
            Path("/data/ycfeng/tmp/optimization-matrix"),
        )
        budget = int(env["MAX_TOKENS_IN_BATCH"])
        assert 0 < budget < case.prefill_tokens, case.case_id


def test_online_activation_requires_a_positive_inter_arrival_delay(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(simulation_mode="online")
    log_path = tmp_path / "online.log"
    metrics_dir = tmp_path / "metrics"
    _write_success_log(log_path)
    _write_minimal_metrics(metrics_dir)
    _write_request_metrics(
        metrics_dir,
        [
            {
                "Request Id": 0,
                "request_inter_arrival_delay": 0.0,
            }
        ],
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "FAIL"
    assert "online activation" in result["errors"]


def test_online_activation_accepts_a_positive_inter_arrival_delay(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(simulation_mode="online")
    log_path = tmp_path / "online.log"
    metrics_dir = tmp_path / "metrics"
    _write_success_log(log_path)
    _write_minimal_metrics(metrics_dir)
    _write_request_metrics(
        metrics_dir,
        [
            {
                "Request Id": 0,
                "request_inter_arrival_delay": 0.25,
            }
        ],
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "PASS"


def test_prefix_activation_requires_positive_and_consistent_hit_totals(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(
        optimization_stratum="prefix",
        enable_prefix_caching=True,
        request_source="prefix-trace",
    )
    log_path = tmp_path / "prefix.log"
    metrics_dir = tmp_path / "metrics"
    _write_success_log(log_path)
    _write_request_metrics(
        metrics_dir,
        [
            {
                "Request Id": 0,
                "request_prefix_cache_hit_blocks": 0,
            }
        ],
    )
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps(
            {
                "ttft_statistics": {"mean": 1.0},
                "prefix_cache_statistics": {
                    "total_query_blocks": 1,
                    "total_hit_blocks": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    zero_result = check_case_log(case, log_path, metrics_dir)

    assert zero_result["status"] == "FAIL"
    assert "Prefix Cache activation" in zero_result["errors"]

    _write_request_metrics(
        metrics_dir,
        [
            {
                "Request Id": 0,
                "request_prefix_cache_hit_blocks": 1,
            }
        ],
    )
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps(
            {
                "ttft_statistics": {"mean": 1.0},
                "prefix_cache_statistics": {
                    "total_query_blocks": 2,
                    "total_hit_blocks": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    positive_result = check_case_log(case, log_path, metrics_dir)

    assert positive_result["status"] == "PASS"
    assert positive_result["prefix_cache_hit_blocks"] == 1


def test_mtp_activation_requires_positive_and_consistent_counters(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(
        optimization_stratum="mtp",
        enable_mtp=True,
    )
    log_path = tmp_path / "mtp.log"
    metrics_dir = tmp_path / "metrics"
    _write_success_log(log_path)
    _write_request_metrics(
        metrics_dir,
        [
            {
                "Request Id": 0,
                "request_spec_total_iterations": 0,
                "request_spec_committed_tokens": 0,
            }
        ],
    )
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps(
            {
                "ttft_statistics": {"mean": 1.0},
                "spec_decode_statistics": {
                    "total_iterations": 0,
                    "total_committed_tokens": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    zero_result = check_case_log(case, log_path, metrics_dir)

    assert zero_result["status"] == "FAIL"
    assert "MTP activation" in zero_result["errors"]

    _write_request_metrics(
        metrics_dir,
        [
            {
                "Request Id": 0,
                "request_spec_total_iterations": 2,
                "request_spec_committed_tokens": 3,
            }
        ],
    )
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps(
            {
                "ttft_statistics": {"mean": 1.0},
                "spec_decode_statistics": {
                    "total_iterations": 2,
                    "total_committed_tokens": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    positive_result = check_case_log(case, log_path, metrics_dir)

    assert positive_result["status"] == "PASS"
    assert positive_result["spec_decode_iterations"] == 2
    assert positive_result["spec_decode_committed_tokens"] == 3


def test_chunked_prefill_rejects_single_full_prefill_batch(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(
        enable_chunked_prefill=True,
        prefill_tokens=32,
        num_requests=1,
        pipeline_stages=2,
    )
    log_path = tmp_path / "chunked.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=0,
                stage_id=stage_id,
                request_ids=["7"],
                request_num_tokens=[32],
            )
            for stage_id in (0, 1)
        ],
    )
    _write_success_log(log_path)

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "FAIL"
    assert result["chunked_prefill_split_count"] == 0
    assert "multiple positive prefill chunks" in result["errors"]


def test_chunked_prefill_rejects_pipeline_stage_payload_mismatch(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(
        enable_chunked_prefill=True,
        prefill_tokens=32,
        num_requests=1,
        pipeline_stages=2,
    )
    log_path = tmp_path / "chunked.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=0,
                stage_id=0,
                request_ids=["7"],
                request_num_tokens=[16],
            ),
            _prefill_stage_ledger_row(
                batch_id=0,
                stage_id=1,
                request_ids=["7"],
                request_num_tokens=[8],
            ),
        ],
    )
    _write_success_log(log_path)

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "FAIL"
    assert "pipeline-stage payload mismatch" in result["errors"]


def test_chunked_prefill_rejects_non_conserving_stage_ledger(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(
        enable_chunked_prefill=True,
        prefill_tokens=32,
        num_requests=1,
        pipeline_stages=2,
    )
    log_path = tmp_path / "chunked.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=batch_id,
                stage_id=stage_id,
                request_ids=["7"],
                request_num_tokens=[token_count],
            )
            for batch_id, token_count in ((0, 16), (1, 8))
            for stage_id in (0, 1)
        ],
    )
    _write_success_log(log_path)

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "FAIL"
    assert "prefill-token conservation" in result["errors"]


def test_chunked_prefill_accepts_token_conserving_stage_ledger_without_debug_logs(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(
        enable_chunked_prefill=True,
        prefill_tokens=32,
        num_requests=1,
        pipeline_stages=2,
    )
    log_path = tmp_path / "chunked.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=batch_id,
                stage_id=stage_id,
                request_ids=["7"],
                request_num_tokens=[16],
            )
            for batch_id in (0, 1)
            for stage_id in (0, 1)
        ],
    )
    _write_success_log(log_path)

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "PASS"
    assert result["chunked_prefill_split_count"] == 1


def test_chunked_prefill_accepts_explicit_preemption_recompute_epochs(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(
        enable_chunked_prefill=True,
        prefill_tokens=32,
        num_requests=1,
        pipeline_stages=1,
    )
    log_path = tmp_path / "chunked.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_request_metrics(
        metrics_dir,
        [
            {
                "Request Id": 7,
                "request_num_prefill_tokens": 32,
                "request_prefill_preemption_count": 1,
            }
        ],
    )
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=0,
                stage_id=0,
                request_ids=["7"],
                request_num_tokens=[16],
                request_runtime_epochs=[0],
            ),
            _prefill_stage_ledger_row(
                batch_id=1,
                stage_id=0,
                request_ids=["7"],
                request_num_tokens=[16],
                request_runtime_epochs=[1],
            ),
            _prefill_stage_ledger_row(
                batch_id=2,
                stage_id=0,
                request_ids=["7"],
                request_num_tokens=[16],
                request_runtime_epochs=[1],
            ),
        ],
    )
    _write_success_log(log_path)

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "PASS"
    assert result["chunked_prefill_request_token_totals"] == {"7": 48}
    assert result["chunked_prefill_request_recompute_token_totals"] == {"7": 16}
    assert result["chunked_prefill_prefill_preemption_count"] == 1


def test_chunked_prefill_counts_only_request_prefill_tokens_in_mixed_batch(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(
        enable_chunked_prefill=True,
        prefill_tokens=32,
        num_requests=2,
        pipeline_stages=1,
    )
    log_path = tmp_path / "chunked.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=0,
                stage_id=0,
                request_ids=["0"],
                request_num_tokens=[16],
                request_num_prefill_tokens=[16],
            ),
            _prefill_stage_ledger_row(
                batch_id=1,
                stage_id=0,
                request_ids=["0"],
                request_num_tokens=[16],
                request_num_prefill_tokens=[16],
            ),
            _prefill_stage_ledger_row(
                batch_id=2,
                stage_id=0,
                request_ids=["1", "0"],
                request_num_tokens=[16, 1],
                request_num_prefill_tokens=[16, 0],
                decode_component_ms=1.0,
            ),
            _prefill_stage_ledger_row(
                batch_id=3,
                stage_id=0,
                request_ids=["0", "1"],
                request_num_tokens=[1, 16],
                request_num_prefill_tokens=[0, 16],
                decode_component_ms=1.0,
            ),
        ],
    )
    _write_success_log(log_path)

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "PASS"
    assert result["chunked_prefill_request_token_totals"] == {
        "0": 32,
        "1": 32,
    }
    assert result["chunked_prefill_split_count"] == 2


def test_chunked_prefill_ignores_speculative_verify_prefill_kernel_rows(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(
        enable_chunked_prefill=False,
        prefill_tokens=32,
        num_requests=1,
        pipeline_stages=1,
    )
    log_path = tmp_path / "verify.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=0,
                stage_id=0,
                request_ids=["7"],
                request_num_tokens=[32],
                request_num_prefill_tokens=[32],
            ),
            _prefill_stage_ledger_row(
                batch_id=1,
                stage_id=0,
                request_ids=["7"],
                request_num_tokens=[2],
                request_num_prefill_tokens=[0],
                decode_component_ms=1.0,
            ),
        ],
    )
    _write_success_log(log_path)

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "PASS"
    assert result["chunked_prefill_prefill_row_count"] == 1
    assert result["chunked_prefill_split_count"] == 0


@pytest.mark.parametrize(
    "architecture",
    ("pd-disaggregation", "pd-af-disaggregation"),
)
def test_chunked_prefill_reads_prefill_role_for_disaggregated_architectures(
    tmp_path: Path,
    architecture: str,
) -> None:
    case = _dense_optimization_case(
        architecture=architecture,
        enable_chunked_prefill=True,
        prefill_tokens=32,
        num_requests=1,
        pipeline_stages=2,
    )
    log_path = tmp_path / "chunked.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=batch_id,
                stage_id=stage_id,
                request_ids=["7"],
                request_num_tokens=[16],
                cluster_type="PREFILL",
            )
            for batch_id in (0, 1)
            for stage_id in (0, 1)
        ],
    )
    _write_success_log(log_path)

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "PASS"
    assert result["chunked_prefill_split_count"] == 1


def test_chunked_prefill_control_split_fails_paired_comparison(
    tmp_path: Path,
) -> None:
    group_id = next(
        case.comparison_group_id
        for case in build_optimization_matrix(REPO_ROOT)
        if case.architecture == "co-location"
        and case.model_kind == "dense"
        and case.optimization_stratum == "ordinary"
        and case.decode_cuda_graph_mode == "none"
        and case.enable_chunked_prefill
        and case.comparison_group_id is not None
    )
    pair_cases = [
        replace(
            case,
            prefill_tokens=32,
            num_requests=1,
            pipeline_stages=2,
        )
        for case in build_optimization_matrix(REPO_ROOT)
        if case.comparison_group_id == group_id
        and case.decode_cuda_graph_mode == "none"
    ]
    result_rows = []
    for case in pair_cases:
        log_path = tmp_path / f"{case.case_id}.log"
        metrics_dir = tmp_path / case.case_id
        _write_minimal_metrics(metrics_dir)
        _write_stage_batch_ledger(
            metrics_dir,
            [
                _prefill_stage_ledger_row(
                    batch_id=batch_id,
                    stage_id=stage_id,
                    request_ids=["7"],
                    request_num_tokens=[16],
                )
                for batch_id in (0, 1)
                for stage_id in (0, 1)
            ],
        )
        _write_success_log(log_path)
        check = check_case_log(case, log_path, metrics_dir)
        row = _optimization_result_row(case, metric_offset=0.0)
        row["status"] = check["status"]
        row["check"] = check
        result_rows.append(row)

    control_check = next(
        row["check"]
        for row in result_rows
        if not next(
            case
            for case in pair_cases
            if case.case_id == row["case_id"]
        ).enable_chunked_prefill
    )
    report = build_optimization_comparison(pair_cases, result_rows)

    assert control_check["chunked_prefill_split_count"] == 1
    assert report["status"] == "FAIL"
    assert "Chunked Prefill control unexpectedly split" in report["pairs"][0]["errors"]


@pytest.mark.parametrize(
    ("case_changes", "runtime_mode"),
    (
        (
            {"decode_cuda_graph_mode": "piecewise"},
            "PIECEWISE",
        ),
        (
            {"use_cuda_graph": True, "optimization_stratum": "pd-af-cube"},
            "GLOBAL",
        ),
    ),
)
def test_cuda_graph_activation_requires_matching_production_runtime_records(
    tmp_path: Path,
    case_changes: dict[str, object],
    runtime_mode: str,
) -> None:
    if runtime_mode == "GLOBAL":
        case = next(
            case
            for case in build_optimization_matrix(REPO_ROOT)
            if case.architecture == "pd-af-disaggregation"
            and case.use_cuda_graph
        )
    else:
        case = _dense_optimization_case(**case_changes)
    log_path = tmp_path / "cuda_graph.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_success_log(log_path)

    missing_result = check_case_log(case, log_path, metrics_dir)

    assert missing_result["status"] == "FAIL"
    assert "CUDA Graph activation" in missing_result["errors"]

    expected_roles = (
        ("DECODE_ATTN", "DECODE_FFN")
        if runtime_mode == "GLOBAL"
        else (
            ("MONOLITHIC",)
            if case.architecture == "co-location"
            else ("DECODE",)
        )
    )
    activation_lines = []
    for role in expected_roles:
        activation_lines.append(
            "[CUDA-GRAPH-ACTIVATION] "
            + json.dumps(
                {
                    "batch_id": 7,
                    "cluster_role": role,
                    "config_mode": (
                        "global"
                        if runtime_mode == "GLOBAL"
                        else case.decode_cuda_graph_mode
                    ),
                    "runtime_mode": runtime_mode,
                    "capture_hit": True,
                    "capture_sizes": [8],
                    "original_tokens": [5],
                    "padded_tokens": [8],
                    "measurement_family": "kernel_only",
                },
                sort_keys=True,
            )
        )
    _write_success_log(log_path, extra_lines=activation_lines)

    positive_result = check_case_log(case, log_path, metrics_dir)

    assert positive_result["status"] == "PASS"
    assert positive_result["cuda_graph_capture_count"] == len(expected_roles)
    assert positive_result["cuda_graph_capture_roles"] == list(expected_roles)

    eager_line = activation_lines[0].replace(
        '"measurement_family": "kernel_only"',
        '"measurement_family": "eager"',
    )
    _write_success_log(log_path, extra_lines=[eager_line, *activation_lines[1:]])

    wrong_family_result = check_case_log(case, log_path, metrics_dir)

    assert wrong_family_result["status"] == "FAIL"
    assert "measurement_family" in wrong_family_result["errors"]


def test_cuda_graph_control_rejects_unexpected_activation_record(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(decode_cuda_graph_mode="none")
    log_path = tmp_path / "cuda_graph_control.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_success_log(
        log_path,
        extra_lines=[
            "[CUDA-GRAPH-ACTIVATION] "
            + json.dumps(
                {
                    "batch_id": 7,
                    "cluster_role": "MONOLITHIC",
                    "config_mode": "full_decode_only",
                    "runtime_mode": "FULL",
                    "capture_hit": True,
                    "capture_sizes": [8],
                    "original_tokens": [5],
                    "padded_tokens": [8],
                    "measurement_family": "kernel_only",
                },
                sort_keys=True,
            )
        ],
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "FAIL"
    assert "CUDA Graph control unexpectedly activated" in result["errors"]


def test_optimization_cli_writes_independent_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task_dir = tmp_path / "task"

    exit_code = main(
        [
            "--mode",
            "generate",
            "--matrix-kind",
            "optimization",
            "--repo-root",
            str(REPO_ROOT),
            "--task-dir",
            str(task_dir),
        ]
    )

    optimization_manifest = (
        task_dir / "moe_ep_non_dummy_optimization_matrix_manifest.jsonl"
    )
    regression_manifest = task_dir / "moe_ep_non_dummy_matrix_manifest.jsonl"
    rows = [
        json.loads(line)
        for line in optimization_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert exit_code == 0
    assert len(rows) == 200
    assert not regression_manifest.exists()
    assert str(optimization_manifest) in capsys.readouterr().out


def test_optimization_compare_cli_writes_json_csv_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    task_dir = tmp_path / "task"
    output_root = tmp_path / "outputs"
    results_path = task_dir / "results.jsonl"
    source_provenance = matrix_module._source_provenance(REPO_ROOT)
    pair_manifest_path = (
        task_dir / "moe_ep_non_dummy_optimization_expected_pairs.jsonl"
    )
    task_dir.mkdir(parents=True)
    matrix_module.write_manifest(
        task_dir / "moe_ep_non_dummy_optimization_matrix_manifest.jsonl",
        cases,
    )
    matrix_module.write_optimization_pair_manifest(pair_manifest_path, cases)
    result_rows = []
    for index, case in enumerate(cases):
        row = _optimization_result_row(case, metric_offset=float(index))
        case_dir = output_root / case.case_id
        case_dir.mkdir(parents=True)
        (case_dir / "case_metadata.json").write_text(
            json.dumps(
                {
                    "case": asdict(case),
                    "source_provenance": source_provenance,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        row.update(
            {
                "repo_root": str(REPO_ROOT.resolve()),
                "output_root": str(output_root.resolve()),
                "results_path": str(results_path.resolve()),
                "log_path": str(case_dir / f"{case.case_id}.log"),
                "metrics_path": str(case_dir / "metrics"),
                "source_provenance": source_provenance,
            }
        )
        result_rows.append(row)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.build_optimization_matrix",
        lambda _repo_root: cases,
    )

    exit_code = main(
        [
            "--mode",
            "compare",
            "--matrix-kind",
            "optimization",
            "--repo-root",
            str(REPO_ROOT),
            "--task-dir",
            str(task_dir),
            "--output-root",
            str(output_root),
            "--results-path",
            str(results_path),
        ]
    )

    json_path = task_dir / "moe_ep_non_dummy_optimization_comparison.json"
    csv_path = task_dir / "moe_ep_non_dummy_optimization_comparison.csv"
    markdown_path = task_dir / "moe_ep_non_dummy_optimization_comparison.md"
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["case_count"] == 200
    assert report["pair_count"] == 122
    assert "workflow" in report["pairs"][0]
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "control_ttft_mean_ms" in csv_text
    assert "control_workflow" in csv_text
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "Latency values are report-only" in markdown_text
    assert "Workflow control/enabled" in markdown_text
    output = capsys.readouterr().out
    assert f"comparison_json={json_path}" in output
    assert f"comparison_markdown={markdown_path}" in output


def test_optimization_compare_rejects_mutated_persisted_pair_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    task_dir = tmp_path / "task"
    output_root = tmp_path / "outputs"
    results_path = task_dir / "results.jsonl"
    source_provenance = matrix_module._source_provenance(REPO_ROOT)
    pair_manifest_path = (
        task_dir / "moe_ep_non_dummy_optimization_expected_pairs.jsonl"
    )
    task_dir.mkdir(parents=True)
    matrix_module.write_manifest(
        task_dir / "moe_ep_non_dummy_optimization_matrix_manifest.jsonl",
        cases,
    )
    matrix_module.write_optimization_pair_manifest(pair_manifest_path, cases)
    pair_rows = [
        json.loads(line)
        for line in pair_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pair_rows[0]["control_case_id"] = "nonexistent-control"
    pair_manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pair_rows),
        encoding="utf-8",
    )

    result_rows = []
    for index, case in enumerate(cases):
        case_dir = output_root / case.case_id
        case_dir.mkdir(parents=True)
        (case_dir / "case_metadata.json").write_text(
            json.dumps(
                {
                    "case": asdict(case),
                    "source_provenance": source_provenance,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        row = _optimization_result_row(case, metric_offset=float(index))
        row.update(
            {
                "repo_root": str(REPO_ROOT.resolve()),
                "output_root": str(output_root.resolve()),
                "results_path": str(results_path.resolve()),
                "log_path": str(case_dir / f"{case.case_id}.log"),
                "metrics_path": str(case_dir / "metrics"),
                "source_provenance": source_provenance,
            }
        )
        result_rows.append(row)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.build_optimization_matrix",
        lambda _repo_root: cases,
    )

    with pytest.raises(ValueError, match="persisted optimization pair manifest"):
        main(
            [
                "--mode",
                "compare",
                "--matrix-kind",
                "optimization",
                "--repo-root",
                str(REPO_ROOT),
                "--task-dir",
                str(task_dir),
                "--output-root",
                str(output_root),
                "--results-path",
                str(results_path),
                "--pair-manifest-path",
                str(pair_manifest_path),
            ]
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("delete", "persisted optimization pair manifest mismatch"),
        ("optimization", "persisted optimization pair manifest mismatch"),
        ("target_field", "persisted optimization pair manifest mismatch"),
        ("enabled", "persisted optimization pair manifest mismatch"),
    ],
)
def test_persisted_pair_manifest_requires_exact_rows(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    cases = build_optimization_matrix(REPO_ROOT)
    path = tmp_path / "expected_pairs.jsonl"
    matrix_module.write_optimization_pair_manifest(path, cases)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if mutation == "delete":
        rows.pop()
    elif mutation == "optimization":
        rows[0]["optimization"] = "mtp"
    elif mutation == "target_field":
        rows[0]["target_field"] = "enable_mtp"
    elif mutation == "enabled":
        rows[0]["enabled_case_id"] = "missing-enabled-case"
    else:
        raise AssertionError(f"unsupported test mutation={mutation!r}")
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        matrix_module._validate_persisted_optimization_pair_manifest(path, cases)


def test_preflight_classifies_each_case_without_launching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = build_optimization_matrix(REPO_ROOT)[:2]
    ready_case, blocked_case = cases

    def fake_validate_profile_inputs(case, root):
        if case.case_id == blocked_case.case_id:
            raise ValueError("missing required routing profile")
        return [root / "data/profiling/compute/ready.csv"]

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.validate_profile_inputs",
        fake_validate_profile_inputs,
    )

    rows = preflight_cases(
        cases,
        REPO_ROOT,
        tmp_path / "output",
    )

    assert [row["case_id"] for row in rows] == [
        ready_case.case_id,
        blocked_case.case_id,
    ]
    assert rows[0]["status"] == "READY"
    assert rows[0]["required_profile_paths"] == [
        str(REPO_ROOT / "data/profiling/compute/ready.csv")
    ]
    assert rows[0]["command"].startswith("bash ")
    assert rows[1]["status"] == "BLOCKED"
    assert rows[1]["required_profile_paths"] == []
    assert rows[1]["blockers"] == [
        {
            "stage": "profile",
            "type": "ValueError",
            "message": "missing required routing profile",
        }
    ]


def test_preflight_cli_writes_independent_ledger_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = build_optimization_matrix(REPO_ROOT)[:2]
    preflight_path = tmp_path / "optimization_preflight.jsonl"
    task_dir = tmp_path / "task"
    expected_rows = [
        {
            "case_id": cases[0].case_id,
            "status": "READY",
            "blockers": [],
            "preflight_only": True,
        },
        {
            "case_id": cases[1].case_id,
            "status": "BLOCKED",
            "blockers": [
                {
                    "stage": "profile",
                    "type": "ValueError",
                    "message": "missing required routing profile",
                }
            ],
            "preflight_only": True,
        },
    ]

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.build_optimization_matrix",
        lambda _repo_root: cases,
    )

    def fake_preflight_cases(
        selected_cases,
        repo_root,
        output_root,
        *,
        matrix_kind,
    ):
        assert list(selected_cases) == cases
        assert repo_root == REPO_ROOT
        assert output_root == Path(
            "/data/ycfeng/tmp/frontier_non_dummy_optimization_matrix"
        )
        assert matrix_kind == "optimization"
        return expected_rows

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.preflight_cases",
        fake_preflight_cases,
    )

    def fail_if_run(*_args, **_kwargs):
        raise AssertionError("preflight must not launch simulator cases")

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.run_cases",
        fail_if_run,
    )

    exit_code = main(
        [
            "--mode",
            "preflight",
            "--matrix-kind",
            "optimization",
            "--repo-root",
            str(REPO_ROOT),
            "--task-dir",
            str(task_dir),
            "--preflight-path",
            str(preflight_path),
        ]
    )

    rows = [
        json.loads(line)
        for line in preflight_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = capsys.readouterr().out
    assert exit_code == 1
    assert rows == expected_rows
    assert f"preflight={preflight_path}" in output
    assert "ready=1 blocked=1" in output
    assert not (
        task_dir / "moe_ep_non_dummy_optimization_matrix_results.jsonl"
    ).exists()


def test_optimization_run_preflights_full_matrix_before_partial_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = build_optimization_matrix(REPO_ROOT)[:3]
    task_dir = tmp_path / "task"
    output_root = tmp_path / "outputs"
    inspected_case_ids: list[str] = []

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.build_optimization_matrix",
        lambda _repo_root: cases,
    )

    def fake_preflight_cases(
        selected_cases,
        repo_root,
        selected_output_root,
        *,
        matrix_kind,
    ):
        inspected_case_ids.extend(case.case_id for case in selected_cases)
        assert repo_root == REPO_ROOT
        assert selected_output_root == output_root
        assert matrix_kind == "optimization"
        return [
            {
                "case_id": case.case_id,
                "status": "BLOCKED" if index == 2 else "READY",
                "blockers": (
                    [
                        {
                            "stage": "profile",
                            "type": "ValueError",
                            "message": "missing profile",
                        }
                    ]
                    if index == 2
                    else []
                ),
                "preflight_only": True,
            }
            for index, case in enumerate(selected_cases)
        ]

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.preflight_cases",
        fake_preflight_cases,
    )

    def fail_if_run(*_args, **_kwargs):
        raise AssertionError("blocked optimization matrix must not launch")

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.run_cases",
        fail_if_run,
    )

    exit_code = main(
        [
            "--mode",
            "run",
            "--matrix-kind",
            "optimization",
            "--repo-root",
            str(REPO_ROOT),
            "--task-dir",
            str(task_dir),
            "--output-root",
            str(output_root),
            "--start",
            "0",
            "--limit",
            "1",
        ]
    )

    assert exit_code == 1
    assert inspected_case_ids == [case.case_id for case in cases]
    assert "ready=2 blocked=1" in capsys.readouterr().out
    assert not (
        task_dir / "moe_ep_non_dummy_optimization_matrix_results.jsonl"
    ).exists()


def test_optimization_run_forwards_bounded_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = build_optimization_matrix(REPO_ROOT)[:2]
    captured_parallelism: list[int] = []

    monkeypatch.setattr(
        matrix_module,
        "build_optimization_matrix",
        lambda _repo_root: cases,
    )
    monkeypatch.setattr(
        matrix_module,
        "preflight_cases",
        lambda selected_cases, *_args, **_kwargs: [
            {
                "case_id": case.case_id,
                "status": "READY",
                "blockers": [],
                "preflight_only": True,
            }
            for case in selected_cases
        ],
    )

    def fake_run_cases(*_args, max_parallel_cases, **_kwargs):
        captured_parallelism.append(max_parallel_cases)
        return [{"status": "PASS"} for _case in cases]

    monkeypatch.setattr(matrix_module, "run_cases", fake_run_cases)

    exit_code = main(
        [
            "--mode",
            "run",
            "--matrix-kind",
            "optimization",
            "--repo-root",
            str(REPO_ROOT),
            "--task-dir",
            str(tmp_path / "task"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--max-parallel-cases",
            "2",
        ]
    )

    assert exit_code == 0
    assert captured_parallelism == [2]


def test_run_cases_preflights_every_selected_case_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = build_optimization_matrix(REPO_ROOT)[:2]
    validated_case_ids: list[str] = []

    def fake_validate_profile_inputs(case, _repo_root):
        validated_case_ids.append(case.case_id)
        if case.case_id == cases[1].case_id:
            raise ValueError("second case missing profile")
        return []

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.validate_profile_inputs",
        fake_validate_profile_inputs,
    )

    def fail_if_launched(*_args, **_kwargs):
        raise AssertionError("no simulator may launch before full preflight")

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.subprocess.run",
        fail_if_launched,
    )

    with pytest.raises(ValueError, match="second case missing profile"):
        run_cases(
            cases,
            REPO_ROOT,
            tmp_path / "outputs",
            tmp_path / "results.jsonl",
        )

    assert validated_case_ids == [case.case_id for case in cases]


def test_run_cases_bounds_parallelism_and_serializes_ledger_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = build_optimization_matrix(REPO_ROOT)[:4]
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    write_thread_ids: list[int] = []
    real_write_jsonl = matrix_module._write_jsonl

    monkeypatch.setattr(
        matrix_module,
        "validate_profile_inputs",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        matrix_module,
        "build_shell_command",
        lambda case, *_args, **_kwargs: (case.case_id, {}),
    )
    monkeypatch.setattr(
        matrix_module,
        "_find_metrics_dir",
        lambda output_root, case, **_kwargs: output_root / case.case_id / "metrics",
    )
    monkeypatch.setattr(
        matrix_module,
        "check_case_log",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )

    def fake_run(_command, *, stdout, **_kwargs):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with active_lock:
            active -= 1
        return type("Completed", (), {"returncode": 0})()

    def recording_write_jsonl(path, rows):
        write_thread_ids.append(threading.get_ident())
        return real_write_jsonl(path, rows)

    monkeypatch.setattr(matrix_module.subprocess, "run", fake_run)
    monkeypatch.setattr(matrix_module, "_write_jsonl", recording_write_jsonl)

    results_path = tmp_path / "results.jsonl"
    results = run_cases(
        cases,
        REPO_ROOT,
        tmp_path / "outputs",
        results_path,
        max_parallel_cases=2,
    )

    assert [result["case_id"] for result in results] == [
        case.case_id for case in cases
    ]
    assert [row["case_id"] for row in _load_result_rows_for_test(results_path)] == [
        case.case_id for case in cases
    ]
    assert max_active == 2
    assert write_thread_ids
    assert set(write_thread_ids) == {threading.get_ident()}


def _load_result_rows_for_test(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_run_cases_uses_filesystem_freshness_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = replace(
        next(
            case
            for case in build_matrix(REPO_ROOT)
            if case.architecture == "co-location" and case.model_kind == "dense"
        ),
        num_layers=1,
    )
    output_root = tmp_path / "outputs"
    results_path = tmp_path / "results.jsonl"
    real_time_ns = __import__("time").time_ns

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.time.time_ns",
        lambda: real_time_ns() + 1_000_000_000,
    )
    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.validate_profile_inputs",
        lambda *_args, **_kwargs: [],
    )

    def fake_run(*_args, stdout, **_kwargs):
        metrics_dir = output_root / case.case_id / "metrics"
        _write_minimal_metrics(metrics_dir)
        stdout.write(
            "Dummy Mode: false\n"
            "Simulation completed successfully.\n"
            "[OP-TRACE][MONOLITHIC][ATTENTION] "
            "batch_id=1, layer_id=0, num_tokens=1\n"
        )
        stdout.flush()
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(
        "tests.e2e.moe_ep_non_dummy_matrix.subprocess.run",
        fake_run,
    )

    results = run_cases(
        [case],
        REPO_ROOT,
        output_root,
        results_path,
    )

    assert results[0]["status"] == "PASS"
    assert results[0]["metrics_path"] == str(
        (output_root / case.case_id / "metrics").resolve()
    )


def test_profile_validation_is_fail_fast(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.model_kind == "moe"
    )
    with pytest.raises(FileNotFoundError, match="moe.csv"):
        validate_profile_inputs(case, tmp_path)


def test_pdaf_profile_validation_requires_kernel_only_family(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "pd-af-disaggregation" and case.model_kind == "moe"
    )
    model_dir = tmp_path / case.device / case.model_name
    model_dir.mkdir(parents=True)
    common_header = (
        "profiling_precision,model_arch,model_architecture_profile,"
        "quant_signature,measurement_type\n"
    )
    common_row = "BF16,generic,generic,none,CUDA_EVENT\n"
    (model_dir / "attention.csv").write_text(
        common_header + common_row,
        encoding="utf-8",
    )
    (model_dir / "linear_op.csv").write_text(
        common_header + common_row,
        encoding="utf-8",
    )
    (model_dir / "moe.csv").write_text(
        common_header.removesuffix("\n")
        + ",routing_runtime_path\n"
        + common_row.removesuffix("\n")
        + ",uniform_topk\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="linear_op_kernel_only.csv"):
        validate_profile_inputs(case, tmp_path)


def test_pdaf_moe_cases_use_kernel_only_capable_profile() -> None:
    cases = [
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "pd-af-disaggregation" and case.model_kind == "moe"
    ]

    assert {case.ep_size for case in cases} == {1, 2, 4}
    assert all(case.model_name == "Phi-tiny-MoE-instruct" for case in cases)
    assert all(case.routing_distribution == "random" for case in cases)


def test_profile_validation_rejects_missing_architecture_metadata(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "moe"
    )
    model_dir = tmp_path / case.device / case.model_name
    model_dir.mkdir(parents=True)
    (model_dir / "attention.csv").write_text(
        "profiling_precision,model_arch,quant_signature,measurement_type\n"
        "BF16,generic,none,CUDA_EVENT\n",
        encoding="utf-8",
    )
    (model_dir / "linear_op.csv").write_text(
        "profiling_precision,model_arch,quant_signature,measurement_type\n"
        "BF16,generic,none,CUDA_EVENT\n",
        encoding="utf-8",
    )
    (model_dir / "moe.csv").write_text(
        "profiling_precision,model_arch,quant_signature,measurement_type,"
        "model_architecture_profile,routing_runtime_path\n"
        "BF16,generic,none,CUDA_EVENT,generic,standard_fused_topk\n"
        "BF16,generic,none,CUDA_EVENT,generic,uniform_topk\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_architecture_profile"):
        validate_profile_inputs(case, tmp_path)


def test_profile_validation_requires_exact_measurement_family(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_optimization_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "dense"
    )
    model_dir = tmp_path / case.device / case.model_name
    model_dir.mkdir(parents=True)
    _write_minimal_profile(
        model_dir / "attention.csv",
        measurement_type="KERNEL_ONLY",
        tp_sizes=(case.attn_tensor_parallel_size,),
    )
    _write_minimal_profile(
        model_dir / "linear_op.csv",
        measurement_type="KERNEL_ONLY",
        tp_sizes=(case.attn_tensor_parallel_size,),
    )

    with pytest.raises(ValueError, match="measurement_type.*CUDA_EVENT"):
        validate_profile_inputs(case, tmp_path)


def test_profile_validation_requires_exact_moe_tp_ep_tuple(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_optimization_matrix(REPO_ROOT)
        if case.architecture == "co-location"
        and case.model_kind == "moe"
        and case.routing_distribution == "random"
        and case.optimization_stratum == "ordinary"
    )
    model_dir = tmp_path / case.device / case.model_name
    model_dir.mkdir(parents=True)
    _write_minimal_profile(
        model_dir / "attention.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(case.attn_tensor_parallel_size,),
    )
    _write_minimal_profile(
        model_dir / "linear_op.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(
            case.attn_tensor_parallel_size,
            case.moe_tensor_parallel_size,
        ),
    )
    _write_minimal_profile(
        model_dir / "moe.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(case.moe_tensor_parallel_size,),
        ep_size=case.ep_size + 1,
        routing_runtime_path="uniform_topk",
    )

    with pytest.raises(ValueError, match=r"TP/EP.*\(.*\)"):
        validate_profile_inputs(case, tmp_path)


def test_pdd_profile_validation_uses_runtime_op_level_tp_policy(
    tmp_path: Path,
) -> None:
    case = next(
        case
        for case in build_optimization_matrix(REPO_ROOT)
        if case.case_id == "pd_disaggregation_ordinary_graph_11_none"
    )
    model_dir = tmp_path / case.device / case.model_name
    model_dir.mkdir(parents=True)
    _write_minimal_profile(
        model_dir / "attention.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(case.attn_tensor_parallel_size,),
    )
    _write_minimal_profile(
        model_dir / "linear_op.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(
            case.attn_tensor_parallel_size,
            case.moe_tensor_parallel_size,
        ),
    )
    (model_dir / "moe.csv").write_text(
        "profiling_precision,model_arch,model_architecture_profile,"
        "quant_signature,measurement_type,num_tensor_parallel_workers,"
        "expert_parallel_size,routing_runtime_path,"
        "gating_runtime_context,"
        "time_stats.moe_gating_linear.median,"
        "time_stats.moe_gating_routing_topk.median,"
        "time_stats.moe_shuffling.median,"
        "time_stats.moe_grouped_gemm.median\n"
        "BF16,generic,generic,none,CUDA_EVENT,1,2,uniform_topk,"
        "standalone_legacy,"
        "1.0,2.0,3.0,4.0\n"
        "BF16,generic,generic,none,CUDA_EVENT,4,2,standard_fused_topk,"
        "standalone_legacy,"
        "1.0,2.0,3.0,4.0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "moe_gating_routing_topk.*TP=1.*"
            "routing_runtime_path=standard_fused_topk"
        ),
    ):
        validate_profile_inputs(case, tmp_path)


def test_profile_validation_requires_target_embedded_mtp_columns(
    tmp_path: Path,
) -> None:
    case = next(
        case
        for case in build_optimization_matrix(REPO_ROOT)
        if case.enable_mtp and case.architecture == "co-location"
    )
    model_dir = tmp_path / case.device / case.model_name
    model_dir.mkdir(parents=True)
    _write_minimal_profile(
        model_dir / "attention.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(case.attn_tensor_parallel_size,),
    )
    _write_minimal_profile(
        model_dir / "linear_op.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(
            case.attn_tensor_parallel_size,
            case.moe_tensor_parallel_size,
        ),
    )
    _write_minimal_profile(
        model_dir / "moe.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(case.moe_tensor_parallel_size,),
        ep_size=case.ep_size,
        routing_runtime_path="uniform_topk",
    )

    with pytest.raises(ValueError, match="mtp_fusion_proj.*lm_head_linear"):
        validate_profile_inputs(case, tmp_path)


def test_profile_validation_rejects_empty_target_embedded_mtp_same_tp_columns(
    tmp_path: Path,
) -> None:
    case = next(
        case
        for case in build_optimization_matrix(REPO_ROOT)
        if case.enable_mtp and case.architecture == "co-location"
    )
    model_dir = tmp_path / case.device / case.model_name
    model_dir.mkdir(parents=True)
    _write_minimal_profile(
        model_dir / "attention.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(case.attn_tensor_parallel_size,),
    )
    _write_minimal_profile(
        model_dir / "linear_op.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(
            case.attn_tensor_parallel_size,
            case.moe_tensor_parallel_size,
        ),
        include_mtp_columns=True,
        include_mtp_same_tp_columns=True,
    )
    _write_minimal_profile(
        model_dir / "moe.csv",
        measurement_type="CUDA_EVENT",
        tp_sizes=(case.moe_tensor_parallel_size,),
        ep_size=case.ep_size,
        routing_runtime_path="uniform_topk",
    )

    with pytest.raises(ValueError, match="same-TP.*emb"):
        validate_profile_inputs(case, tmp_path)


def test_log_checker_requires_layer_trace_and_finite_metrics(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "pd-af-disaggregation" and case.model_kind == "moe"
    )
    log_path = tmp_path / "case.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps(
            {
                "ttft_statistics": {"mean": 1.25},
                "request_e2e_time_statistics": {"mean": 2.5},
            }
        ),
        encoding="utf-8",
    )
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][DECODE_FFN][MOE][moe_shuffling] batch_id=1, layer_id=0, predicted_time_ms=0.1",
                "[OP-TRACE][DECODE_FFN][MOE][moe_grouped_gemm] batch_id=1, layer_id=0, predicted_time_ms=0.2",
                "[OP-TRACE][DECODE_FFN][MOE][TOTAL] batch_id=1, layer_id=0, total_moe_time_ms=1.0",
                "[EP-WORKLOAD][DECODE_FFN] batch_id=1, layer_id=0, ep_id=0, moe_ep_size=1, per_expert_tokens={0: 1, 1: 0}, lane_compute_ms=0.2, lane_comm_ms=0.0",
                "[EP-BARRIER][DECODE_FFN] batch_id=1, layer_id=0, phase=combine, expected_ep_ids=[0], arrived_ep_ids=[0], max_lane_time_ms=0.2, barrier_time_ms=0.2, barrier_end_time_s=0.001",
                "[EP-CONSERVATION][DECODE_FFN] batch_id=1, layer_id=0, routing_token_count=1, router_topk=1, total_routed_assignments=1, per_ep_routed_tokens={0: 1}",
                "[DECODE_FFN] per_expert_tokens extracted: {0: 1, 1: 0}",
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "PASS"
    assert result["layer_ids"] == [0]
    assert result["ep_workload_records"] == 1
    assert result["ep_barrier_records"] == 1
    assert result["numeric_metric_count"] == 2


def test_shared_moe_checker_requires_ep_workload_trace(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "moe"
    )
    log_path = tmp_path / "shared_moe.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "Dummy Mode: false\nSimulation completed successfully.\n"
        "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] batch_id=1, layer_id=0, predicted_time_ms=0.2\n",
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "FAIL"
    assert "EP workload" in result["errors"]


def test_shared_moe_checker_requires_dispatch_and_combine_barriers(
    tmp_path: Path,
) -> None:
    case = replace(
        next(
            case
            for case in build_matrix(REPO_ROOT)
            if case.architecture == "co-location"
            and case.model_kind == "moe"
            and case.ep_size == 1
        ),
        num_layers=1,
        moe_layer_ids=(0,),
    )
    log_path = tmp_path / "shared_barrier.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] batch_id=1, layer_id=0, predicted_time_ms=0.1",
                "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] batch_id=1, layer_id=0, predicted_time_ms=0.2",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=1, layer_id=0, ep_id=0, moe_ep_size=1, per_expert_tokens={0: 1}, lane_compute_ms=0.2, lane_comm_ms=0.0",
                "[EP-CONSERVATION][MONOLITHIC] batch_id=1, layer_id=0, routing_token_count=1, router_topk=1, total_routed_assignments=1, per_ep_routed_tokens={0: 1}",
                "[EP-BARRIER][MONOLITHIC] batch_id=1, layer_id=0, phase=combine, expected_ep_ids=[0], arrived_ep_ids=[0], max_lane_time_ms=0.2, barrier_time_ms=0.2, barrier_end_time_s=0.001",
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert (
        "missing EP barrier evidence for workload waves "
        "phase=dispatch waves=[('MONOLITHIC', 1, 0)]"
    ) in result["errors"]


def test_ep_workload_parser_accepts_logger_prefix() -> None:
    records = _parse_ep_workload_records(
        "INFO 12:00:00 scheduler.py:1] "
        "[EP-WORKLOAD][DECODE] batch_id=7, layer_id=3, ep_id=1, "
        "moe_ep_size=2, per_expert_tokens={0: 0, 1: 4}, "
        "lane_compute_ms=1.25, lane_comm_ms=0.5"
    )

    assert records == [
        {
            "cluster": "DECODE",
            "batch_id": 7,
            "layer_id": 3,
            "ep_id": 1,
            "moe_ep_size": 2,
            "per_expert_tokens": {0: 0, 1: 4},
            "lane_compute_ms": 1.25,
            "lane_comm_ms": 0.5,
        }
    ]


def test_ep_barrier_parser_accepts_logger_prefix() -> None:
    records = _parse_ep_barrier_records(
        "INFO 12:00:00 scheduler.py:1] "
        "[EP-BARRIER][PREFILL] batch_id=7, layer_id=3, phase=combine, "
        "expected_ep_ids=[0, 1], arrived_ep_ids=[0, 1], "
        "max_lane_time_ms=4.0, barrier_time_ms=4.0, barrier_end_time_s=0.008"
    )

    assert records == [
        {
            "cluster": "PREFILL",
            "batch_id": 7,
            "layer_id": 3,
            "phase": "combine",
            "expected_ep_ids": [0, 1],
            "arrived_ep_ids": [0, 1],
            "max_lane_time_ms": 4.0,
            "barrier_time_ms": 4.0,
            "barrier_end_time_s": 0.008,
        }
    ]


def test_ep_barrier_log_precision_preserves_timestamp_equation(monkeypatch) -> None:
    messages: list[str] = []

    def capture_info(message: str, *args: object) -> None:
        messages.append(message % args)

    monkeypatch.setattr(cluster_scheduler_module.logger, "info", capture_info)
    start_time_s = 0.0001956
    barrier_time_ms = 0.147713
    BaseClusterScheduler._log_ep_barrier_trace(
        cluster_type=ClusterType.PREFILL,
        batch_id=7,
        layer_id=3,
        phase="dispatch",
        expected_ep_ids=(0, 1),
        arrived_ep_ids=(0, 1),
        max_lane_time_ms=0.146522,
        barrier_time_ms=barrier_time_ms,
        barrier_start_time_s=start_time_s,
        barrier_end_time_s=start_time_s + barrier_time_ms * 1e-3,
        trace_identity={
            "replica_id": 0,
            "stage_id": 0,
            "request_ids": (11,),
            "request_runtime_epochs": (0,),
            "iteration_ids": (0,),
            "schedule_epoch": 1,
            "afd_stage_idx": -1,
            "operation_id": 7,
            "operation_kind": "ep_ffn",
        },
    )

    assert len(messages) == 1
    records = _parse_ep_barrier_records(messages[0], require_start_time=True)
    assert len(records) == 1
    assert _validate_ep_barrier_time_equations(records) == []


def test_ep_conservation_parser_accepts_logger_prefix() -> None:
    records = _parse_ep_conservation_records(
        "INFO 12:00:00 scheduler.py:1] "
        "[EP-CONSERVATION][DECODE_FFN] batch_id=7, layer_id=3, "
        "routing_token_count=2, router_topk=2, total_routed_assignments=4, "
        "per_ep_routed_tokens={0: 1, 1: 3}"
    )

    assert records == [
        {
            "cluster": "DECODE_FFN",
            "batch_id": 7,
            "layer_id": 3,
            "routing_token_count": 2,
            "router_topk": 2,
            "total_routed_assignments": 4,
            "per_ep_routed_tokens": {0: 1, 1: 3},
        }
    ]


def test_strict_checker_does_not_merge_ep_ids_from_different_waves(tmp_path: Path) -> None:
    case = next(
        replace(case, num_layers=1, moe_layer_ids=(0,))
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location"
        and case.model_kind == "moe"
        and case.ep_size == 2
    )
    log_path = tmp_path / "split_wave.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] layer_id=0",
                "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] layer_id=0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, ep_id=0, "
                "moe_ep_size=2, per_expert_tokens={0: 1}, lane_compute_ms=1.0, lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=11, layer_id=0, ep_id=1, "
                "moe_ep_size=2, per_expert_tokens={1: 1}, lane_compute_ms=1.0, lane_comm_ms=0.0",
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "participants are incomplete" in result["errors"]


def test_strict_shared_checker_requires_layer_barrier_evidence(tmp_path: Path) -> None:
    case = next(
        replace(case, num_layers=1, moe_layer_ids=(0,))
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location"
        and case.model_kind == "moe"
        and case.ep_size == 2
    )
    log_path = tmp_path / "missing_barrier.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] layer_id=0",
                "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] layer_id=0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, ep_id=0, "
                "moe_ep_size=2, per_expert_tokens={0: 1}, lane_compute_ms=1.0, lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, ep_id=1, "
                "moe_ep_size=2, per_expert_tokens={1: 1}, lane_compute_ms=2.0, lane_comm_ms=0.0",
                "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, routing_token_count=1, router_topk=2, total_routed_assignments=2, per_ep_routed_tokens={0: 1, 1: 1}",
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "missing EP barrier evidence" in result["errors"]


@pytest.mark.parametrize(
    "duplicate_kind",
    ("workload", "conservation", "barrier"),
)
def test_strict_checker_rejects_duplicate_wave_records(
    tmp_path: Path,
    duplicate_kind: str,
) -> None:
    case = _single_layer_ep2_case()
    log_path = tmp_path / f"duplicate_{duplicate_kind}.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    workloads = [
        "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, ep_id=0, "
        "moe_ep_size=2, per_expert_tokens={0: 1}, "
        "lane_compute_ms=1.0, lane_comm_ms=0.0",
        "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, ep_id=1, "
        "moe_ep_size=2, per_expert_tokens={1: 1}, "
        "lane_compute_ms=2.0, lane_comm_ms=0.0",
    ]
    conservation = [
        "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, "
        "routing_token_count=1, router_topk=2, "
        "total_routed_assignments=2, per_ep_routed_tokens={0: 1, 1: 1}",
    ]
    barriers = [
        "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
        "phase=dispatch, expected_ep_ids=[0, 1], arrived_ep_ids=[0, 1], "
        "max_lane_time_ms=2.0, barrier_time_ms=2.0, "
        "barrier_end_time_s=0.002",
        "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
        "phase=combine, expected_ep_ids=[0, 1], arrived_ep_ids=[0, 1], "
        "max_lane_time_ms=2.0, barrier_time_ms=2.0, "
        "barrier_end_time_s=0.004",
    ]
    if duplicate_kind == "workload":
        workloads.append(workloads[-1])
    elif duplicate_kind == "conservation":
        conservation.append(conservation[-1])
    else:
        barriers.append(barriers[-1])
    log_path.write_text(
        _strict_ep_log(
            workload_lines=workloads,
            conservation_lines=conservation,
            barrier_lines=barriers,
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "duplicate EP" in result["errors"]


def test_strict_checker_requires_barriers_for_the_same_workload_wave(
    tmp_path: Path,
) -> None:
    case = _single_layer_ep2_case()
    log_path = tmp_path / "wrong_barrier_wave.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log(
            workload_lines=[
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=0, moe_ep_size=2, per_expert_tokens={0: 1}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=1, moe_ep_size=2, per_expert_tokens={1: 1}, "
                "lane_compute_ms=2.0, lane_comm_ms=0.0",
            ],
            conservation_lines=[
                "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, "
                "routing_token_count=1, router_topk=2, "
                "total_routed_assignments=2, "
                "per_ep_routed_tokens={0: 1, 1: 1}",
            ],
            barrier_lines=[
                "[EP-BARRIER][MONOLITHIC] batch_id=11, layer_id=0, "
                "phase=dispatch, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=2.0, "
                "barrier_time_ms=2.0, barrier_end_time_s=0.002",
                "[EP-BARRIER][MONOLITHIC] batch_id=11, layer_id=0, "
                "phase=combine, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=2.0, "
                "barrier_time_ms=2.0, barrier_end_time_s=0.004",
            ],
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "EP barrier wave identity mismatch" in result["errors"]


def test_zero_routed_case_requires_a_lane_with_zero_total_workload(
    tmp_path: Path,
) -> None:
    case = _single_layer_ep2_case(zero_routed=True)
    log_path = tmp_path / "no_zero_total_lane.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log(
            workload_lines=[
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=0, moe_ep_size=2, per_expert_tokens={0: 0, 1: 1}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=1, moe_ep_size=2, per_expert_tokens={2: 1, 3: 0}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
            ],
            conservation_lines=[
                "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, "
                "routing_token_count=1, router_topk=2, "
                "total_routed_assignments=2, "
                "per_ep_routed_tokens={0: 1, 1: 1}",
            ],
            barrier_lines=[
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=dispatch, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.001",
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=combine, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.002",
            ],
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "zero-total EP lane" in result["errors"]


def test_zero_routed_case_requires_zero_local_compute_for_zero_lane(
    tmp_path: Path,
) -> None:
    case = _zero_routed_checker_case()
    log_path = tmp_path / "zero_lane_nonzero_compute.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log(
            workload_lines=[
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=0, moe_ep_size=2, per_expert_tokens={0: 0, 1: 0}, "
                "lane_compute_ms=1.0, routed_compute_ms=1.0, "
                "lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=1, moe_ep_size=2, per_expert_tokens={2: 1, 3: 1}, "
                "lane_compute_ms=1.0, routed_compute_ms=1.0, "
                "lane_comm_ms=0.0",
            ],
            conservation_lines=[
                "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, "
                "routing_token_count=1, router_topk=2, "
                "total_routed_assignments=2, "
                "per_ep_routed_tokens={0: 0, 1: 2}",
            ],
            barrier_lines=[
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=dispatch, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.001",
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=combine, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.002",
            ],
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "zero-routed EP lane has non-zero local compute" in result["errors"]


def test_zero_routed_case_requires_independent_routed_compute_evidence(
    tmp_path: Path,
) -> None:
    case = _zero_routed_checker_case()
    log_path = tmp_path / "zero_lane_missing_routed_compute.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log(
            workload_lines=[
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=0, moe_ep_size=2, per_expert_tokens={0: 0, 1: 0}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=1, moe_ep_size=2, per_expert_tokens={2: 1, 3: 1}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
            ],
            conservation_lines=[
                "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, "
                "routing_token_count=1, router_topk=2, "
                "total_routed_assignments=2, "
                "per_ep_routed_tokens={0: 0, 1: 2}",
            ],
            barrier_lines=[
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=dispatch, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.001",
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=combine, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.002",
            ],
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "lacks independent routed compute evidence" in result["errors"]


@pytest.mark.parametrize(
    ("architecture", "present_role", "missing_role", "ep_size_field"),
    (
        (
            "pd-disaggregation",
            "PREFILL",
            "DECODE",
            "prefill_moe_expert_parallel_size",
        ),
        (
            "pd-af-disaggregation",
            "PREFILL",
            "DECODE_FFN",
            "prefill_moe_expert_parallel_size",
        ),
    ),
)
def test_strict_checker_requires_each_moe_layer_in_every_execution_role(
    tmp_path: Path,
    architecture: str,
    present_role: str,
    missing_role: str,
    ep_size_field: str,
) -> None:
    case = next(
        replace(candidate, num_layers=1, moe_layer_ids=(0,))
        for candidate in build_matrix(REPO_ROOT)
        if candidate.architecture == architecture
        and candidate.model_kind == "moe"
    )
    wave = _ep_wave_lines(
        cluster=present_role,
        ep_size=int(getattr(case, ep_size_field)),
        batch_id=10,
        layer_id=0,
    )
    log_path = tmp_path / f"missing_{missing_role.lower()}.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log_from_events(
            [
                str(wave["conservation"]),
                *list(wave["workloads"]),
                str(wave["dispatch"]),
                str(wave["combine"]),
            ],
            include_pdaf_participant_map=architecture == "pd-af-disaggregation",
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert (
        "EP workload has no complete participant wave "
        f"cluster={missing_role} layer=0"
    ) in result["errors"]


def test_strict_checker_requires_workload_before_dispatch_barrier(
    tmp_path: Path,
) -> None:
    case = _single_layer_ep2_case()
    wave = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=10,
        layer_id=0,
    )
    log_path = tmp_path / "dispatch_before_workload.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log_from_events(
            [
                str(wave["conservation"]),
                str(wave["dispatch"]),
                *list(wave["workloads"]),
                str(wave["combine"]),
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "EP wave event order is invalid" in result["errors"]


def test_strict_checker_rejects_next_layer_start_before_previous_combine(
    tmp_path: Path,
) -> None:
    case = replace(_single_layer_ep2_case(), num_layers=2, moe_layer_ids=(0, 1))
    layer_0 = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=10,
        layer_id=0,
    )
    layer_1 = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=10,
        layer_id=1,
    )
    log_path = tmp_path / "next_layer_before_combine.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log_from_events(
            [
                str(layer_0["conservation"]),
                *list(layer_0["workloads"]),
                str(layer_0["dispatch"]),
                str(layer_1["conservation"]),
                *list(layer_1["workloads"]),
                str(layer_0["combine"]),
                str(layer_1["dispatch"]),
                str(layer_1["combine"]),
            ],
            layer_ids=(0, 1),
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "next MoE layer started before prior combine" in result["errors"]


def test_log_checker_rejects_traceback(tmp_path: Path) -> None:
    case = next(iter(build_matrix(REPO_ROOT)))
    log_path = tmp_path / "case.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text("{}", encoding="utf-8")
    log_path.write_text(
        "Dummy Mode: false\nTraceback (most recent call last):\n",
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "FAIL"
    assert "Traceback" in result["errors"]


def test_strict_dense_checker_requires_complete_layer_coverage(tmp_path: Path) -> None:
    case = replace(
        next(
            case
            for case in build_matrix(REPO_ROOT)
            if case.architecture == "co-location" and case.model_kind == "dense"
        ),
        num_layers=32,
        moe_layer_ids=(),
    )
    log_path = tmp_path / "dense.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "Dummy Mode: false\nSimulation completed successfully.\n"
        "[OP-TRACE][MONOLITHIC][ATTENTION] batch_id=0, layer_id=0, num_tokens=1\n",
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "layer ids are not contiguous" in result["errors"]


def test_strict_checker_does_not_count_unrelated_layer_ids_as_execution(
    tmp_path: Path,
) -> None:
    case = replace(
        next(
            case
            for case in build_matrix(REPO_ROOT)
            if case.architecture == "co-location" and case.model_kind == "dense"
        ),
        num_layers=2,
        moe_layer_ids=(),
    )
    log_path = tmp_path / "forged_layer.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][MONOLITHIC][ATTENTION] batch_id=0, layer_id=0, num_tokens=1",
                "[UNRELATED-METRIC] layer_id=1, value=1",
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_operation_layer_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "operation trace layer ids are not contiguous" in result["errors"]


def test_chunked_prefill_control_ignores_legacy_stage_ledger_schema(
    tmp_path: Path,
) -> None:
    case = _dense_optimization_case(
        enable_chunked_prefill=False,
        prefill_tokens=32,
        num_requests=1,
        pipeline_stages=1,
    )
    log_path = tmp_path / "legacy_control.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            {
                "batch_id": 0,
                "stage_id": 0,
                "cluster_type": "MONOLITHIC",
                "replica_id": 0,
                "execution_scope": "FULL_STAGE_WORLD",
                "replica_local_id": None,
                "request_ids": ["7"],
                "request_num_tokens": [32],
                "execution_time": {
                    "component_ledger_ms": {
                        "attention_prefill_execution_time": 1.0,
                    }
                },
            }
        ],
    )
    _write_success_log(log_path)

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "PASS"
    assert "chunked_prefill_stage_ledger_present" not in result


def test_strict_dense_checker_rejects_moe_protocol_records(
    tmp_path: Path,
) -> None:
    case = replace(
        next(
            case
            for case in build_matrix(REPO_ROOT)
            if case.architecture == "co-location" and case.model_kind == "dense"
        ),
        num_layers=1,
        moe_layer_ids=(),
    )
    wave = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=1,
        batch_id=1,
        layer_id=0,
    )
    log_path = tmp_path / "dense_with_ep.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][MONOLITHIC][ATTENTION] batch_id=1, layer_id=0, num_tokens=1",
                str(wave["conservation"]),
                *list(wave["workloads"]),
                str(wave["dispatch"]),
                str(wave["combine"]),
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "dense case emitted EP protocol evidence" in result["errors"]


def test_strict_mixed_checker_requires_dense_layer_coverage(
    tmp_path: Path,
) -> None:
    case = replace(
        next(
            case
            for case in build_matrix(REPO_ROOT)
            if case.architecture == "co-location"
            and case.model_kind == "mixed"
            and case.ep_size == 2
        ),
        num_layers=2,
        moe_layer_ids=(1,),
    )
    wave = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=1,
        layer_id=1,
    )
    log_path = tmp_path / "mixed_missing_dense.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log_from_events(
            [
                str(wave["conservation"]),
                *list(wave["workloads"]),
                str(wave["dispatch"]),
                str(wave["combine"]),
            ],
            layer_ids=(1,),
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "layer ids are not contiguous" in result["errors"]


def test_strict_mixed_checker_rejects_dense_layer_ep_wave(
    tmp_path: Path,
) -> None:
    case = replace(
        next(
            case
            for case in build_matrix(REPO_ROOT)
            if case.architecture == "co-location"
            and case.model_kind == "mixed"
            and case.ep_size == 2
        ),
        num_layers=2,
        moe_layer_ids=(1,),
    )
    dense_wave = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=1,
        layer_id=0,
    )
    moe_wave = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=1,
        layer_id=1,
    )
    log_path = tmp_path / "mixed_dense_as_moe.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][MONOLITHIC][ATTENTION] batch_id=1, layer_id=0, num_tokens=1",
                str(dense_wave["conservation"]),
                *list(dense_wave["workloads"]),
                str(dense_wave["dispatch"]),
                str(dense_wave["combine"]),
                "[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] layer_id=1",
                "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] layer_id=1",
                str(moe_wave["conservation"]),
                *list(moe_wave["workloads"]),
                str(moe_wave["dispatch"]),
                str(moe_wave["combine"]),
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "dense layer emitted EP protocol evidence" in result["errors"]


def test_strict_mixed_checker_requires_dense_operation_trace(
    tmp_path: Path,
) -> None:
    case = replace(
        next(
            case
            for case in build_matrix(REPO_ROOT)
            if case.architecture == "co-location"
            and case.model_kind == "mixed"
            and case.ep_size == 2
        ),
        num_layers=2,
        moe_layer_ids=(1,),
    )
    moe_wave = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=1,
        layer_id=1,
    )
    log_path = tmp_path / "mixed_missing_dense_op_trace.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[UNRELATED-METRIC] layer_id=0, value=1",
                "[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] layer_id=1",
                "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] layer_id=1",
                str(moe_wave["conservation"]),
                *list(moe_wave["workloads"]),
                str(moe_wave["dispatch"]),
                str(moe_wave["combine"]),
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_operation_layer_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "operation trace layer ids are not contiguous" in result["errors"]


def test_strict_checker_rejects_case_routing_contract_mismatch(
    tmp_path: Path,
) -> None:
    case = replace(
        _single_layer_ep2_case(),
        total_experts=2,
        router_topk=1,
        routing_distribution="balanced",
        num_requests=1,
    )
    log_path = tmp_path / "routing_contract_mismatch.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log(
            workload_lines=[
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=0, moe_ep_size=2, per_expert_tokens={0: 1}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=1, moe_ep_size=2, per_expert_tokens={999: 1}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
            ],
            conservation_lines=[
                "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, "
                "routing_token_count=1, router_topk=2, "
                "total_routed_assignments=2, "
                "per_ep_routed_tokens={0: 1, 1: 1}",
            ],
            barrier_lines=[
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=dispatch, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.001",
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=combine, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.002",
            ],
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "router_topk disagrees with case" in result["errors"]
    assert "expert ID is outside case.total_experts" in result["errors"]


def test_strict_checker_rejects_missing_request_wave(
    tmp_path: Path,
) -> None:
    case = replace(
        _single_layer_ep2_case(),
        total_experts=2,
        router_topk=1,
        routing_distribution="balanced",
        num_requests=2,
    )
    wave = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=10,
        layer_id=0,
    )
    log_path = tmp_path / "missing_request_wave.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log_from_events(
            [
                str(wave["conservation"]),
                *list(wave["workloads"]),
                str(wave["dispatch"]),
                str(wave["combine"]),
            ],
            add_identity=False,
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "EP wave cardinality is below case.num_requests" in result["errors"]


def test_strict_checker_uses_independent_request_token_oracle(
    tmp_path: Path,
) -> None:
    case = replace(
        _single_layer_ep2_case(),
        total_experts=2,
        router_topk=1,
        routing_distribution="balanced",
        num_requests=1,
        pipeline_stages=1,
    )
    log_path = tmp_path / "independent_token_oracle.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=10,
                stage_id=0,
                request_ids=["0"],
                request_num_tokens=[1],
                request_runtime_epochs=[0],
            )
        ],
    )
    routing_input = matrix_module._build_routing_input_ledger(
        case,
        source_provenance=matrix_module._source_provenance(REPO_ROOT),
    )
    routing_input["expected_routing_token_totals"] = [
        {
            "cluster": "MONOLITHIC",
            "replica_id": 0,
            "layer_id": 0,
            "routing_token_count": 1,
        }
    ]
    (metrics_dir / "frontier_routing_input_ledger.json").write_text(
        json.dumps(routing_input),
        encoding="utf-8",
    )
    log_path.write_text(
        _strict_ep_log(
            workload_lines=[
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=0, moe_ep_size=2, per_expert_tokens={0: 1}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=1, moe_ep_size=2, per_expert_tokens={1: 1}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
            ],
            conservation_lines=[
                "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, "
                "routing_token_count=2, router_topk=1, "
                "total_routed_assignments=2, "
                "per_ep_routed_tokens={0: 1, 1: 1}",
            ],
            barrier_lines=[
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=dispatch, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.001",
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=combine, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.002",
            ],
        ),
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert (
        "routing_token_count disagrees with independent request/stage ledger"
        in result["errors"]
    )


def test_strict_checker_rejects_self_consistent_but_wrong_runtime_token_ledger(
    tmp_path: Path,
) -> None:
    """The independent token oracle must not be derived from runtime output."""

    case = replace(
        _single_layer_ep2_case(),
        total_experts=2,
        router_topk=1,
        routing_distribution="balanced",
        num_requests=1,
        pipeline_stages=1,
        prefill_tokens=2,
        decode_tokens=0,
    )
    log_path = tmp_path / "self_consistent_wrong_token_ledger.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=10,
                stage_id=0,
                request_ids=["0"],
                request_num_tokens=[2],
                request_num_prefill_tokens=[2],
                request_runtime_epochs=[0],
            )
        ],
    )
    routing_input = matrix_module._build_routing_input_ledger(
        case,
        source_provenance=matrix_module._source_provenance(REPO_ROOT),
    )
    routing_input["expected_routing_token_totals"] = [
        {
            "cluster": "MONOLITHIC",
            "replica_id": 0,
            "layer_id": 0,
            "routing_token_count": 1,
        }
    ]
    (metrics_dir / "frontier_routing_input_ledger.json").write_text(
        json.dumps(routing_input),
        encoding="utf-8",
    )
    log_path.write_text(
        _strict_ep_log(
            workload_lines=[
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=0, moe_ep_size=2, per_expert_tokens={0: 1}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
                "ep_id=1, moe_ep_size=2, per_expert_tokens={1: 1}, "
                "lane_compute_ms=1.0, lane_comm_ms=0.0",
            ],
            conservation_lines=[
                "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, "
                "routing_token_count=2, router_topk=1, "
                "total_routed_assignments=2, "
                "per_ep_routed_tokens={0: 1, 1: 1}",
            ],
            barrier_lines=[
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=dispatch, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.001",
                "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
                "phase=combine, expected_ep_ids=[0, 1], "
                "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
                "barrier_time_ms=1.0, barrier_end_time_s=0.002",
            ],
        ),
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert (
        "routing_token_count disagrees with independent routing input ledger"
        in result["errors"]
    )


def test_strict_checker_rejects_mutated_routing_details_snapshot(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    sidecar_path = metrics_dir / "frontier_routing_input_ledger.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["routing_details_snapshot"][0]["ratios"][0] += 0.01
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "routing_details_snapshot mismatch" in result["errors"]


def test_strict_checker_accepts_one_batched_wave_for_all_requests(
    tmp_path: Path,
) -> None:
    case = replace(
        _single_layer_ep2_case(),
        total_experts=2,
        router_topk=1,
        routing_distribution="balanced",
        num_requests=3,
        pipeline_stages=1,
        prefill_tokens=1,
    )
    log_path = tmp_path / "batched_request_wave.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=10,
                stage_id=0,
                request_ids=["0", "1", "2"],
                request_num_tokens=[1, 1, 1],
                request_num_prefill_tokens=[case.prefill_tokens] * 3,
                request_runtime_epochs=[0, 0, 0],
            )
        ],
    )
    routing_input = matrix_module._build_routing_input_ledger(
        case,
        source_provenance=matrix_module._source_provenance(REPO_ROOT),
    )
    routing_input["expected_routing_token_totals"] = [
        {
            "cluster": "MONOLITHIC",
            "replica_id": 0,
            "layer_id": 0,
            "routing_token_count": 3,
        }
    ]
    (metrics_dir / "frontier_routing_input_ledger.json").write_text(
        json.dumps(routing_input),
        encoding="utf-8",
    )
    log_text = _strict_ep_log(
        workload_lines=[
            "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
            "ep_id=0, moe_ep_size=2, per_expert_tokens={0: 2}, "
            "lane_compute_ms=1.0, lane_comm_ms=0.0",
            "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, "
            "ep_id=1, moe_ep_size=2, per_expert_tokens={1: 1}, "
            "lane_compute_ms=1.0, lane_comm_ms=0.0",
        ],
        conservation_lines=[
            "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, "
            "routing_token_count=3, router_topk=1, "
            "total_routed_assignments=3, "
            "per_ep_routed_tokens={0: 2, 1: 1}",
        ],
        barrier_lines=[
            "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
            "phase=dispatch, expected_ep_ids=[0, 1], "
            "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
            "barrier_time_ms=1.0, barrier_end_time_s=0.001",
            "[EP-BARRIER][MONOLITHIC] batch_id=10, layer_id=0, "
            "phase=combine, expected_ep_ids=[0, 1], "
            "arrived_ep_ids=[0, 1], max_lane_time_ms=1.0, "
            "barrier_time_ms=1.0, barrier_end_time_s=0.002",
        ],
    )
    log_text = (
        log_text.replace("request_ids=[0]", "request_ids=[0, 1, 2]")
        .replace(
            "request_runtime_epochs=[0]",
            "request_runtime_epochs=[0, 0, 0]",
        )
        .replace("iteration_ids=[0]", "iteration_ids=[0, 0, 0]")
    )
    log_path.write_text(log_text, encoding="utf-8")

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "PASS", result["errors"]


def test_strict_checker_rejects_missing_structured_wave_identity(
    tmp_path: Path,
) -> None:
    case = _single_layer_ep2_case()
    wave = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=10,
        layer_id=0,
        include_identity=False,
    )
    log_path = tmp_path / "missing_structured_identity.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log_from_events(
            [
                str(wave["conservation"]),
                *list(wave["workloads"]),
                str(wave["dispatch"]),
                str(wave["combine"]),
            ],
            add_identity=False,
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "structured EP wave identity is missing" in result["errors"]


def test_strict_checker_rejects_mismatched_structured_wave_identity(
    tmp_path: Path,
) -> None:
    case = _single_layer_ep2_case()
    wave = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=10,
        layer_id=0,
    )
    mismatched_conservation = str(wave["conservation"]).replace(
        "replica_id=0",
        "replica_id=1",
        1,
    )
    log_path = tmp_path / "mismatched_structured_identity.log"
    metrics_dir = tmp_path / "metrics"
    _write_minimal_metrics(metrics_dir)
    log_path.write_text(
        _strict_ep_log_from_events(
            [
                mismatched_conservation,
                *list(wave["workloads"]),
                str(wave["dispatch"]),
                str(wave["combine"]),
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "structured EP wave identity mismatch" in result["errors"]


def test_strict_checker_rejects_missing_iteration_identity(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    log_path.write_text(
        log_path.read_text(encoding="utf-8").replace(
            "iteration_ids=[0]",
            "iteration_ids=[]",
        ),
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "iteration_ids must be a non-empty list" in result["errors"]


def test_strict_checker_rejects_duplicate_iteration_identity(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    log_path.write_text(
        log_path.read_text(encoding="utf-8").replace(
            "iteration_ids=[0]",
            "iteration_ids=[0, 0]",
        ),
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert (
        "request/epoch/iteration lists must have equal lengths"
        in result["errors"]
    )


@pytest.mark.parametrize(
    ("field", "replacement", "identity_field"),
    [
        ("iteration_ids=[0]", "iteration_ids=[7]", "iteration_ids"),
        ("schedule_epoch=0", "schedule_epoch=9", "schedule_epoch"),
        ("afd_stage_idx=-1", "afd_stage_idx=4", "afd_stage_idx"),
        ("operation_id=10", "operation_id=999999", "operation_id"),
    ],
)
def test_strict_checker_rejects_common_mode_identity_mutation(
    tmp_path: Path,
    field: str,
    replacement: str,
    identity_field: str,
) -> None:
    """Runtime identity must match the independent stage-ledger identity."""

    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    original = log_path.read_text(encoding="utf-8")
    assert field in original
    log_path.write_text(
        original.replace(field, replacement),
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL", result["errors"]
    assert f"fields=['{identity_field}']" in result["errors"]


def test_strict_checker_rejects_operation_kind_mismatch(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    ledger_path = metrics_dir / "frontier_stage_batch_ledger.jsonl"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
    ledger["operation_kind"] = "attention"
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL", result["errors"]
    assert "fields=['operation_kind']" in result["errors"]


def _add_barrier_start_times(log_text: str) -> str:
    return (
        log_text.replace(
            "barrier_end_time_s=0.001",
            "barrier_start_time_s=0.000, barrier_end_time_s=0.001",
        )
        .replace(
            "barrier_end_time_s=0.002",
            "barrier_start_time_s=0.001, barrier_end_time_s=0.002",
        )
    )


def test_strict_checker_accepts_barrier_des_timestamp_equations(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    log_path.write_text(
        _add_barrier_start_times(log_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
        require_barrier_time_oracle=True,
    )

    assert result["status"] == "PASS", result["errors"]


def test_strict_checker_rejects_missing_barrier_des_start_time(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
        require_barrier_time_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "barrier_start_time_s" in result["errors"]


def test_strict_checker_rejects_mutated_barrier_des_end_time(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    log_text = _add_barrier_start_times(log_path.read_text(encoding="utf-8"))
    log_path.write_text(
        log_text.replace(
            "barrier_start_time_s=0.001, barrier_end_time_s=0.002",
            "barrier_start_time_s=0.001, barrier_end_time_s=99.000",
        ),
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
        require_barrier_time_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "DES timestamp equation mismatch" in result["errors"]


def test_strict_checker_rejects_cross_replica_wave_identity(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    log_path.write_text(
        log_path.read_text(encoding="utf-8").replace(
            "replica_id=0",
            "replica_id=1",
            1,
        ),
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "EP expected wave manifest missing" in result["errors"]
    assert "EP expected wave manifest extra" in result["errors"]


def test_strict_checker_rejects_missing_expected_wave(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(
        tmp_path,
        batch_ids=(10, 11),
    )
    lines = log_path.read_text(encoding="utf-8").splitlines()
    log_path.write_text(
        "\n".join(line for line in lines if "batch_id=11" not in line) + "\n",
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "EP expected wave manifest missing" in result["errors"]


def test_strict_checker_rejects_extra_expected_wave(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    wave = _ep_wave_lines(
        cluster="MONOLITHIC",
        ep_size=2,
        batch_id=11,
        layer_id=0,
    )
    log_path.write_text(
        log_path.read_text(encoding="utf-8")
        + "\n"
        + "\n".join(
            [
                str(wave["conservation"]),
                *list(wave["workloads"]),
                str(wave["dispatch"]),
                str(wave["combine"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert "EP expected wave manifest extra" in result["errors"]


def test_strict_checker_rejects_duplicate_expected_wave(
    tmp_path: Path,
) -> None:
    case, log_path, metrics_dir = _independent_ep_wave_fixture(tmp_path)
    duplicate = log_path.read_text(encoding="utf-8")
    log_path.write_text(duplicate + duplicate, encoding="utf-8")

    result = check_case_log(
        case,
        log_path,
        metrics_dir,
        strict_layers=True,
        require_independent_token_oracle=True,
    )

    assert result["status"] == "FAIL"
    assert (
        "EP expected wave manifest duplicate" in result["errors"]
        or "duplicate EP conservation records" in result["errors"]
    )


def test_expected_wave_manifest_partitions_moe_layers_by_pipeline_stage(
    tmp_path: Path,
) -> None:
    case = replace(
        _single_layer_ep2_case(),
        num_layers=4,
        moe_layer_ids=(0, 3),
        pipeline_stages=2,
        prefill_tokens=2,
    )
    metrics_dir = tmp_path / "metrics"
    _write_stage_batch_ledger(
        metrics_dir,
        [
            _prefill_stage_ledger_row(
                batch_id=10,
                stage_id=0,
                request_ids=["0"],
                request_num_tokens=[2],
                request_num_prefill_tokens=[2],
                request_runtime_epochs=[0],
            ),
            _prefill_stage_ledger_row(
                batch_id=10,
                stage_id=1,
                request_ids=["0"],
                request_num_tokens=[2],
                request_num_prefill_tokens=[2],
                request_runtime_epochs=[0],
            ),
        ],
    )

    manifest = matrix_module._read_ep_expected_wave_manifest(case, metrics_dir)

    assert {
        (key[2], key[4])
        for key in manifest
    } == {(0, 0), (3, 1)}


def test_online_checker_accepts_online_success_marker(tmp_path: Path) -> None:
    case = replace(
        next(
            case
            for case in build_optimization_matrix(REPO_ROOT)
            if case.architecture == "co-location"
            and case.model_kind == "dense"
            and case.simulation_mode == "online"
            and case.optimization_stratum == "ordinary"
        ),
        num_layers=1,
        moe_layer_ids=(),
    )
    log_path = tmp_path / "online.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    _write_request_metrics(
        metrics_dir,
        [{"Request Id": "request-0", "request_inter_arrival_delay": "1.0"}],
    )
    log_path.write_text(
        "Dummy Mode: false\n"
        "Online simulation completed successfully.\n"
        "[OP-TRACE][MONOLITHIC][ATTENTION] "
        "batch_id=0, layer_id=0, num_tokens=1\n",
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "PASS"
    assert result["online_positive_inter_arrival_count"] == 1


def test_log_checker_rejects_unknown_success_marker(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "dense"
    )
    log_path = tmp_path / "unknown-marker.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "Dummy Mode: false\n"
        "Simulation finished successfully.\n"
        "[OP-TRACE][MONOLITHIC][ATTENTION] "
        "batch_id=0, layer_id=0, num_tokens=1\n",
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "FAIL"
    assert "missing success marker" in result["errors"]


def test_result_ledger_merges_partial_runs_without_erasing_prior_cases() -> None:
    existing = [
        {"case_id": "case-a", "status": "PASS", "attempt": 1},
        {"case_id": "case-b", "status": "FAIL", "attempt": 1},
    ]
    rerun = [{"case_id": "case-b", "status": "PASS", "attempt": 2}]

    merged = _merge_result_rows(
        existing,
        rerun,
        expected_case_ids=("case-a", "case-b", "case-c"),
    )

    assert [row["case_id"] for row in merged] == ["case-a", "case-b"]
    assert merged[0]["attempt"] == 1
    assert merged[1]["status"] == "PASS"
    assert merged[1]["attempt"] == 2


def test_result_ledger_rejects_rows_without_canonical_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance"):
        _validate_result_ledger_provenance(
            [{"case_id": "case-a", "status": "PASS"}],
            repo_root=tmp_path / "repo",
            output_root=tmp_path / "output",
            results_path=tmp_path / "results.jsonl",
        )


def test_result_ledger_rejects_rows_from_another_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    results_path = tmp_path / "results.jsonl"
    with pytest.raises(ValueError, match="output_root"):
        _validate_result_ledger_provenance(
            [
                {
                    "case_id": "case-a",
                    "status": "PASS",
                    "repo_root": str(tmp_path / "repo"),
                    "output_root": str(tmp_path / "old-output"),
                    "results_path": str(results_path),
                    "log_path": str(output_root / "case-a" / "case-a.log"),
                    "metrics_path": str(output_root / "case-a" / "metrics"),
                }
            ],
            repo_root=tmp_path / "repo",
            output_root=output_root,
            results_path=results_path,
        )


def test_result_ledger_rejects_external_log_and_metrics_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    case_root = output_root / "case-a"
    results_path = tmp_path / "results.jsonl"
    with pytest.raises(ValueError, match="log_path"):
        _validate_result_ledger_provenance(
            [
                {
                    "case_id": "case-a",
                    "status": "PASS",
                    "repo_root": str(tmp_path / "repo"),
                    "output_root": str(output_root),
                    "results_path": str(results_path),
                    "log_path": str(tmp_path / "old.log"),
                    "metrics_path": str(case_root / "metrics"),
                }
            ],
            repo_root=tmp_path / "repo",
            output_root=output_root,
            results_path=results_path,
        )


def test_result_ledger_rejects_missing_source_provenance_with_valid_paths(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    case_root = output_root / "case-a"
    case_root.mkdir(parents=True)
    results_path = tmp_path / "results.jsonl"
    with pytest.raises(ValueError, match="source_provenance"):
        _validate_result_ledger_provenance(
            [
                {
                    "case_id": "case-a",
                    "status": "PASS",
                    "repo_root": str(tmp_path / "repo"),
                    "output_root": str(output_root),
                    "results_path": str(results_path),
                    "log_path": str(case_root / "case-a.log"),
                    "metrics_path": str(case_root / "metrics"),
                }
            ],
            repo_root=tmp_path / "repo",
            output_root=output_root,
            results_path=results_path,
        )


def test_find_metrics_dir_rejects_stale_metrics(tmp_path: Path) -> None:
    case = next(iter(build_matrix(REPO_ROOT)))
    case_root = tmp_path / case.case_id / "metrics" / "run"
    case_root.mkdir(parents=True)
    metrics_path = case_root / "system_metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="fresh"):
        _find_metrics_dir(tmp_path, case, started_at_ns=metrics_path.stat().st_mtime_ns + 1)

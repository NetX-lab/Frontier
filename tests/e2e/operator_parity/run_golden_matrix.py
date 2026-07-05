#!/usr/bin/env python3
"""Generate and run the operator-parity golden equivalence matrix."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from frontier.config import ReplicaConfig
from frontier.config.device_sku_config import BaseDeviceSKUConfig
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter
from tests.e2e.attention_equivalence.profile_manifest import (
    file_manifest,
    files_manifest,
    write_json_report,
)
from tests.e2e.attention_equivalence.run_equivalence_matrix import run_matrix
from tests.e2e.operator_parity.profile_prerequisite_audit import (
    GOLDEN_CONFIG_FILENAMES,
    REQUIRED_BASE_PROFILE_FILES,
    REQUIRED_MOE_PROFILE_FILES,
    build_requirements,
)
from frontier.utils.output_paths import build_metrics_run_output_dir

SIMULATION_MODES: tuple[str, ...] = ("offline_batch", "online_serving")
SYSTEM_ARCHITECTURES: tuple[str, ...] = ("co-location", "pd-disaggregation")
MEASUREMENT_TYPES: tuple[str, ...] = ("CUDA_EVENT", "KERNEL_ONLY")

REQUEST_METRICS_CSV = "request_metrics.csv"
OP_TRACES_JSONL = "op_traces.jsonl"
OP_TRACES_CSV = "op_traces.csv"
DEFAULT_SIMULATION_ARTIFACTS: tuple[str, ...] = (REQUEST_METRICS_CSV, OP_TRACES_CSV)

TRACE_CSV_COLUMNS: tuple[str, ...] = (
    "trace_index",
    "type",
    "name",
    "ts_start",
    "duration_ms",
    "cluster",
    "replica_id",
    "batch_id",
    "request_id",
    "layer_id",
    "target_cluster",
    "meta_json",
)

_TRACE_NUMERIC_TOLERANCE = {
    "absolute": 1e-12,
    "relative_pct": 1e-7,
}

_REQUEST_METRICS_NUMERIC_TOLERANCE = {
    "absolute": 1e-12,
    "relative_pct": 1e-7,
}

@dataclass(frozen=True)
class GoldenParallelism:
    attn_tensor_parallel_size: int = 1
    attn_data_parallel_size: int = 1
    moe_tensor_parallel_size: int = 1
    moe_expert_parallel_size: int = 1
    num_pipeline_stages: int = 1

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


DEFAULT_GOLDEN_PARALLELISM = GoldenParallelism()


@dataclass(frozen=True)
class GoldenWorkloadProfile:
    num_requests: int
    prefill_tokens: int
    decode_tokens: int
    qps: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


WORKLOAD_PROFILES: dict[str, GoldenWorkloadProfile] = {
    # Historical golden workload. Keeping this default preserves the archived
    # B0.4 48-case reference output layout.
    "short_single": GoldenWorkloadProfile(
        num_requests=1,
        prefill_tokens=8,
        decode_tokens=2,
        qps=1.0,
    ),
    # Supplemental profile used by final-gate overlays to cover a distinct
    # workload shape without invalidating archived default references.
    "multi_medium": GoldenWorkloadProfile(
        num_requests=2,
        prefill_tokens=16,
        decode_tokens=4,
        qps=2.0,
    ),
    # Profile-supported supplemental profile for final numeric parity gates.
    # It varies token shape while avoiding true mixed batches that require
    # attn_decode_in_mixed profiling columns not present in the H800 golden CSVs.
    "long_single": GoldenWorkloadProfile(
        num_requests=1,
        prefill_tokens=16,
        decode_tokens=4,
        qps=1.0,
    ),
}
DEFAULT_WORKLOAD_PROFILES: tuple[str, ...] = ("short_single",)

GOLDEN_MATRIX_EXPECTED_CASES = (
    len(GOLDEN_CONFIG_FILENAMES)
    * len(SIMULATION_MODES)
    * len(SYSTEM_ARCHITECTURES)
    * len(MEASUREMENT_TYPES)
    * len(DEFAULT_WORKLOAD_PROFILES)
)

PINNED_GOLDEN_PARALLELISM_BY_MODEL: dict[str, GoldenParallelism] = {
    # The 1/1/1 layout produces a 159.39 GiB parameter shard for this full
    # step3_text MoE model. TP=4 for both attention and MoE preserves Frontier's
    # shared-domain invariant while staying inside the H800 requested memory
    # budget with real profiled TP=4 rows.
    "step-moe-noquant-small": GoldenParallelism(
        attn_tensor_parallel_size=4,
        attn_data_parallel_size=1,
        moe_tensor_parallel_size=4,
        moe_expert_parallel_size=1,
        num_pipeline_stages=1,
    ),
}


def selected_parallelism_for_model(model_name: str) -> GoldenParallelism:
    return PINNED_GOLDEN_PARALLELISM_BY_MODEL.get(
        model_name,
        DEFAULT_GOLDEN_PARALLELISM,
    )


def _parallelism_source(model_name: str) -> str:
    if model_name in PINNED_GOLDEN_PARALLELISM_BY_MODEL:
        return "pinned"
    return "default"


def requested_memory_bytes_for_device(
    *,
    device: str,
    memory_margin_fraction: float = 0.1,
) -> int:
    device_config = BaseDeviceSKUConfig.create_from_type_string(device)
    return int(device_config.total_memory_gb * 1024**3 * (1 - memory_margin_fraction))


def parameter_memory_per_device_bytes(
    *,
    model_name: str,
    device: str,
    parallelism: GoldenParallelism,
    cluster_type: ClusterType = ClusterType.MONOLITHIC,
) -> int:
    replica_config = ReplicaConfig(
        model_name=model_name,
        device=device,
        attn_tensor_parallel_size=parallelism.attn_tensor_parallel_size,
        attn_data_parallel_size=parallelism.attn_data_parallel_size,
        moe_tensor_parallel_size=parallelism.moe_tensor_parallel_size,
        moe_expert_parallel_size=parallelism.moe_expert_parallel_size,
        num_pipeline_stages=parallelism.num_pipeline_stages,
    )
    return 2 * ParamCounter(
        replica_config=replica_config,
        cluster_type=cluster_type,
    ).get_num_parameters_per_device()


def _assert_pinned_parallelism_memory_feasible(model_name: str, parallelism: GoldenParallelism) -> None:
    if _parallelism_source(model_name) != "pinned":
        return
    requested_memory = requested_memory_bytes_for_device(device="h800")
    parameter_memory = parameter_memory_per_device_bytes(
        model_name=model_name,
        device="h800",
        parallelism=parallelism,
    )
    if parameter_memory >= requested_memory:
        raise ValueError(
            "pinned golden parallelism is not H800 memory-feasible: "
            f"model_name={model_name}, parallelism={parallelism.to_dict()}, "
            f"parameter_memory_per_device_bytes={parameter_memory}, "
            f"requested_memory_bytes={requested_memory}"
        )


def _read_int_column_values(path: Path, column: str) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or ()):
            raise ValueError(f"required profiling coverage column missing: {path}:{column}")
        values: set[int] = set()
        for row in reader:
            raw_value = str(row.get(column, "")).strip()
            if not raw_value:
                continue
            values.add(int(raw_value))
    return values


def _require_profile_value(path: Path, column: str, value: int) -> None:
    values = _read_int_column_values(path, column)
    if value not in values:
        raise ValueError(
            "missing required profiling coverage: "
            f"path={path}, {column}={value}, observed={sorted(values)}"
        )


def _assert_profile_coverage(
    *,
    profile_root: Path,
    model_name: str,
    profile_files: Sequence[str],
    parallelism: GoldenParallelism,
) -> None:
    for filename in profile_files:
        path = profile_root / model_name / filename
        if filename.startswith("attention"):
            _require_profile_value(
                path,
                "num_tensor_parallel_workers",
                parallelism.attn_tensor_parallel_size,
            )
            continue
        if filename.startswith("linear_op"):
            _require_profile_value(
                path,
                "num_tensor_parallel_workers",
                parallelism.attn_tensor_parallel_size,
            )
            _require_profile_value(
                path,
                "num_tensor_parallel_workers",
                parallelism.moe_tensor_parallel_size,
            )
            continue
        if filename.startswith("moe"):
            _require_profile_value(
                path,
                "num_tensor_parallel_workers",
                parallelism.moe_tensor_parallel_size,
            )
            _require_profile_value(
                path,
                "expert_parallel_size",
                parallelism.moe_expert_parallel_size,
            )
            continue


def _measurement_profile_files(required_files: Sequence[str], measurement_type: str) -> tuple[str, ...]:
    if measurement_type == "CUDA_EVENT":
        return tuple(filename for filename in required_files if not filename.endswith("_kernel_only.csv"))
    if measurement_type == "KERNEL_ONLY":
        return tuple(required_files)
    raise ValueError(f"Unsupported measurement_type: {measurement_type!r}")


def _case_name(
    *,
    model_name: str,
    simulation_mode: str,
    sys_arch: str,
    measurement_type: str,
    workload_profile: str,
) -> str:
    arch_slug = sys_arch.replace("-", "_")
    base_name = f"{model_name}__{simulation_mode}__{arch_slug}__{measurement_type.lower()}"
    if workload_profile == DEFAULT_WORKLOAD_PROFILES[0]:
        return base_name
    return f"{base_name}__workload_{workload_profile}"


def _resolve_workload_profile(name: str) -> GoldenWorkloadProfile:
    try:
        return WORKLOAD_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported golden workload profile: {name!r}; "
            f"supported={sorted(WORKLOAD_PROFILES)}"
        ) from exc


def _case_profile_manifest(
    *,
    profile_root: Path,
    model_name: str,
    profile_files: Sequence[str],
) -> dict[str, dict[str, Any]]:
    return files_manifest(
        {
            filename: profile_root / model_name / filename
            for filename in profile_files
        }
    )


def _case_input_manifest(
    *,
    case_manifest: Mapping[str, Any],
    profile_root: Path,
    profile_files: Sequence[str],
) -> dict[str, Any]:
    model_name = str(case_manifest["model_name"])
    return {
        "case": dict(case_manifest),
        "profile_root": str(profile_root),
        "profile_files": _case_profile_manifest(
            profile_root=profile_root,
            model_name=model_name,
            profile_files=profile_files,
        ),
    }


def _case_output_dir(root: Path, *, model_name: str, simulation_mode: str, case_name: str) -> str:
    return build_metrics_run_output_dir(
        output_root=str(root),
        model_type=model_name,
        workload_type=simulation_mode,
        run_id=case_name,
    )


def build_golden_cases(
    *,
    config_root: Path,
    profile_root: Path,
    reference_root: Path,
    candidate_root: Path,
    config_filenames: Sequence[str] = GOLDEN_CONFIG_FILENAMES,
    simulation_artifacts: Sequence[str] = DEFAULT_SIMULATION_ARTIFACTS,
    workload_profiles: Sequence[str] = DEFAULT_WORKLOAD_PROFILES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the declared golden matrix cases without running comparisons."""

    workload_profile_names = tuple(workload_profiles)
    if not workload_profile_names:
        raise ValueError("at least one workload profile is required")
    for workload_profile_name in workload_profile_names:
        _resolve_workload_profile(workload_profile_name)

    requirements = build_requirements(
        config_root=config_root,
        config_filenames=tuple(config_filenames),
    )
    cases: list[dict[str, Any]] = []
    for requirement in requirements:
        parallelism = selected_parallelism_for_model(requirement.model_name)
        _assert_pinned_parallelism_memory_feasible(requirement.model_name, parallelism)
        for workload_profile_name in workload_profile_names:
            workload_profile = _resolve_workload_profile(workload_profile_name)
            for simulation_mode in SIMULATION_MODES:
                for sys_arch in SYSTEM_ARCHITECTURES:
                    for measurement_type in MEASUREMENT_TYPES:
                        profile_files = _measurement_profile_files(
                            requirement.required_files,
                            measurement_type,
                        )
                        _assert_profile_coverage(
                            profile_root=profile_root,
                            model_name=requirement.model_name,
                            profile_files=profile_files,
                            parallelism=parallelism,
                        )
                        case_manifest = {
                            "model_name": requirement.model_name,
                            "config_filename": requirement.config_filename,
                            "config_path": requirement.config_path,
                            "sys_arch": sys_arch,
                            "simulation_mode": simulation_mode,
                            "measurement_type": measurement_type,
                            "dummy_mode": False,
                            "parallelism": parallelism.to_dict(),
                            "parallelism_source": _parallelism_source(requirement.model_name),
                            "workload_profile": workload_profile_name,
                            "workload_shape": workload_profile.to_dict(),
                        }
                        input_manifest = _case_input_manifest(
                            case_manifest=case_manifest,
                            profile_root=profile_root,
                            profile_files=profile_files,
                        )
                        name = _case_name(
                            model_name=requirement.model_name,
                            simulation_mode=simulation_mode,
                            sys_arch=sys_arch,
                            measurement_type=measurement_type,
                            workload_profile=workload_profile_name,
                        )
                        cases.append(
                            {
                                "name": name,
                                "case_manifest": case_manifest,
                                "reference_input_manifest": input_manifest,
                                "candidate_input_manifest": input_manifest,
                                "reference_dir": _case_output_dir(
                                    reference_root,
                                    model_name=requirement.model_name,
                                    simulation_mode=simulation_mode,
                                    case_name=name,
                                ),
                                "candidate_dir": _case_output_dir(
                                    candidate_root,
                                    model_name=requirement.model_name,
                                    simulation_mode=simulation_mode,
                                    case_name=name,
                                ),
                                "simulation_artifacts": list(simulation_artifacts),
                                "simulation_key_columns": {
                                    REQUEST_METRICS_CSV: ["Request Id"],
                                    OP_TRACES_CSV: ["trace_index"],
                                },
                                "simulation_tolerance_allowlist": {
                                    REQUEST_METRICS_CSV: {
                                        "*": dict(_REQUEST_METRICS_NUMERIC_TOLERANCE),
                                    },
                                    OP_TRACES_CSV: {
                                        "ts_start": dict(_TRACE_NUMERIC_TOLERANCE),
                                        "duration_ms": dict(_TRACE_NUMERIC_TOLERANCE),
                                    },
                                },
                            }
                        )

    expected_effective = (
        len(tuple(config_filenames))
        * len(SIMULATION_MODES)
        * len(SYSTEM_ARCHITECTURES)
        * len(MEASUREMENT_TYPES)
        * len(workload_profile_names)
    )
    manifest = {
        "expected_original": GOLDEN_MATRIX_EXPECTED_CASES,
        "expected_effective": expected_effective,
        "actual": len(cases),
        "models": [requirement.model_name for requirement in requirements],
        "simulation_modes": list(SIMULATION_MODES),
        "system_architectures": list(SYSTEM_ARCHITECTURES),
        "measurement_types": list(MEASUREMENT_TYPES),
        "workload_profiles": {
            name: _resolve_workload_profile(name).to_dict()
            for name in workload_profile_names
        },
        "dummy_mode": False,
        "parallelism_by_model": {
            requirement.model_name: {
                "source": _parallelism_source(requirement.model_name),
                **selected_parallelism_for_model(requirement.model_name).to_dict(),
            }
            for requirement in requirements
        },
    }
    if len(cases) != expected_effective:
        raise ValueError(
            f"golden matrix case count mismatch: expected_effective={expected_effective}, "
            f"actual={len(cases)}"
        )
    return cases, manifest


def _format_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def normalize_op_traces_jsonl_to_csv(jsonl_path: str | Path, csv_path: str | Path) -> dict[str, Any]:
    """Convert Frontier's ragged op_traces.jsonl into a deterministic CSV artifact."""

    source = Path(jsonl_path)
    target = Path(csv_path)
    if not source.is_file():
        raise FileNotFoundError(f"required file missing: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    event_count = 0
    with source.open(encoding="utf-8") as input_handle, target.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=TRACE_CSV_COLUMNS)
        writer.writeheader()
        for line_number, line in enumerate(input_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            event = json.loads(stripped)
            if "meta" in event and set(event) == {"meta"}:
                continue
            meta = event.get("meta")
            if meta is not None and not isinstance(meta, dict):
                raise ValueError(
                    f"op trace event meta must be a JSON object when present: "
                    f"{source}:{line_number}"
                )
            row = {
                "trace_index": str(event_count),
                "type": _format_csv_value(event.get("type")),
                "name": _format_csv_value(event.get("name")),
                "ts_start": _format_csv_value(event.get("ts_start")),
                "duration_ms": _format_csv_value(event.get("duration_ms")),
                "cluster": _format_csv_value(event.get("cluster")),
                "replica_id": _format_csv_value(event.get("replica_id")),
                "batch_id": _format_csv_value(event.get("batch_id")),
                "request_id": _format_csv_value(event.get("request_id")),
                "layer_id": _format_csv_value(event.get("layer_id")),
                "target_cluster": _format_csv_value(event.get("target_cluster")),
                "meta_json": json.dumps(meta, sort_keys=True) if meta is not None else "",
            }
            writer.writerow(row)
            event_count += 1

    return {
        "source": file_manifest(source),
        "csv": file_manifest(target),
        "event_count": event_count,
    }


def _normalize_declared_op_traces(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for case in cases:
        if OP_TRACES_CSV not in tuple(case.get("simulation_artifacts", ())) :
            continue
        for side in ("reference", "candidate"):
            case_dir = Path(str(case[f"{side}_dir"]))
            reports[f"{case['name']}:{side}"] = normalize_op_traces_jsonl_to_csv(
                case_dir / OP_TRACES_JSONL,
                case_dir / OP_TRACES_CSV,
            )
    return reports


def _write_matrix_yaml(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"cases": list(cases)}, sort_keys=False),
        encoding="utf-8",
    )


def run_golden_matrix(
    *,
    config_root: Path,
    profile_root: Path,
    reference_root: Path,
    candidate_root: Path,
    output_matrix_yaml: Path,
    config_filenames: Sequence[str] = GOLDEN_CONFIG_FILENAMES,
    simulation_artifacts: Sequence[str] = DEFAULT_SIMULATION_ARTIFACTS,
    workload_profiles: Sequence[str] = DEFAULT_WORKLOAD_PROFILES,
) -> dict[str, Any]:
    """Generate the matrix, normalize trace artifacts, and invoke the existing runner."""

    cases, case_manifest = build_golden_cases(
        config_root=config_root,
        profile_root=profile_root,
        reference_root=reference_root,
        candidate_root=candidate_root,
        config_filenames=config_filenames,
        simulation_artifacts=simulation_artifacts,
        workload_profiles=workload_profiles,
    )
    trace_normalization = _normalize_declared_op_traces(cases)
    _write_matrix_yaml(output_matrix_yaml, cases)
    matrix_report = run_matrix(cases)
    return {
        "case_manifest": case_manifest,
        "matrix_yaml": str(output_matrix_yaml),
        "trace_normalization": trace_normalization,
        "matrix_report": matrix_report,
    }


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("comma-separated list must contain at least one value")
    return values


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=Path("data/config/models"))
    parser.add_argument("--profile-root", type=Path, default=Path("data/profiling/compute/h800"))
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-matrix-yaml", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--simulation-artifacts",
        default=",".join(DEFAULT_SIMULATION_ARTIFACTS),
        help="Comma-separated artifacts to compare inside each simulation output directory.",
    )
    parser.add_argument(
        "--workload-profiles",
        default=",".join(DEFAULT_WORKLOAD_PROFILES),
        help=(
            "Comma-separated golden workload profiles. The default preserves "
            "the archived 48-case reference layout; add long_single for "
            "profile-supported supplemental workload-shape coverage."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_golden_matrix(
        config_root=args.config_root,
        profile_root=args.profile_root,
        reference_root=args.reference_root,
        candidate_root=args.candidate_root,
        output_matrix_yaml=args.output_matrix_yaml,
        simulation_artifacts=_parse_csv_list(args.simulation_artifacts),
        workload_profiles=_parse_csv_list(args.workload_profiles),
    )
    write_json_report(args.output_json, report)
    return int(report["matrix_report"]["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

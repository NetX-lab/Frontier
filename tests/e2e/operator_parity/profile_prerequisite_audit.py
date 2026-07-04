"""Audit H800 profiling prerequisites for the operator parity golden matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

from frontier.model_architectures import get_model_architecture_profile
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    MOE_GATING_RUNTIME_CONTEXT_COLUMN,
)

GOLDEN_CONFIG_FILENAMES: tuple[str, ...] = (
    "llama2_7b_dense_example.json",
    "Phi-tiny-MoE-instruct.json",
    "Step2Mini-tiny.json",
    "step-moe-noquant-small.json",
    "Qwen3-30B-A3B-tiny.json",
    "qwen3-next-80b-a3b-instruct-reduced-l2.json",
)

REQUIRED_BASE_PROFILE_FILES: tuple[str, ...] = (
    "attention.csv",
    "attention_kernel_only.csv",
    "linear_op.csv",
    "linear_op_kernel_only.csv",
)

REQUIRED_MOE_PROFILE_FILES: tuple[str, ...] = (
    "moe.csv",
    "moe_kernel_only.csv",
)

_INVALID_TIME_STATS_VALUES = {"", "nan", "none", "null"}

TRUE_MIXED_ATTENTION_PROFILE_FILES: tuple[str, ...] = (
    "attention.csv",
    "attention_kernel_only.csv",
)

TRUE_MIXED_ATTENTION_FEATURE_COLUMNS: tuple[str, ...] = (
    "batch_composition_ratio",
    "decode_avg_kv_cache_size",
    "decode_batch_size",
    "num_prefill_seqs",
    "total_batch_size",
    "total_prefill_tokens",
    "total_tokens",
)

TRUE_MIXED_ATTENTION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "batch_composition_ratio",
    "decode_avg_kv_cache_size",
    "decode_batch_size",
    "is_true_mixed_batch",
    "num_prefill_seqs",
    "time_stats.attn_decode.median",
    "total_batch_size",
    "total_prefill_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class ProfileRequirement:
    config_filename: str
    config_path: str
    model_name: str
    expected_model_architecture_profile: str
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class ProfileFileAudit:
    path: str
    exists: bool
    row_count: int
    time_stats_column_count: int
    time_stats_valid_count: int
    time_stats_nan_count: int
    time_stats_empty_row_count: int
    time_stats_empty_column_count: int
    true_mixed_row_count: int
    true_mixed_attn_decode_valid_count: int
    true_mixed_required_numeric_valid_row_count: int
    true_mixed_required_numeric_invalid_cell_count: int
    semantic_coverage_errors: tuple[str, ...]


@dataclass(frozen=True)
class ProfileRequirementAudit:
    config_filename: str
    model_name: str
    status: str
    missing_files: tuple[str, ...]
    invalid_files: tuple[str, ...]
    files: dict[str, ProfileFileAudit]


def _is_moe_config(config: dict[str, object]) -> bool:
    num_experts = config.get("num_experts")
    if num_experts is None:
        return False
    if not isinstance(num_experts, (int, str)):
        raise ValueError(f"Invalid num_experts={num_experts!r} in model config.")
    try:
        return int(num_experts) > 0
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid num_experts={num_experts!r} in model config.") from exc


def build_requirements(
    *,
    config_root: Path,
    config_filenames: Sequence[str] = GOLDEN_CONFIG_FILENAMES,
) -> tuple[ProfileRequirement, ...]:
    requirements: list[ProfileRequirement] = []
    for config_filename in config_filenames:
        config_path = config_root / config_filename
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing golden model config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        required_files = list(REQUIRED_BASE_PROFILE_FILES)
        if _is_moe_config(config):
            required_files.extend(REQUIRED_MOE_PROFILE_FILES)
        requirements.append(
            ProfileRequirement(
                config_filename=config_filename,
                config_path=str(config_path),
                model_name=config_path.stem,
                expected_model_architecture_profile=get_model_architecture_profile(
                    SimpleNamespace(**config)
                ).profile_id,
                required_files=tuple(required_files),
            )
        )
    return tuple(requirements)


def _truthy_csv_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _is_valid_time_stats_value(value: object) -> bool:
    return str(value).strip().lower() not in _INVALID_TIME_STATS_VALUES


def _is_valid_numeric_csv_value(value: object) -> bool:
    if not _is_valid_time_stats_value(value):
        return False
    try:
        return math.isfinite(float(str(value).strip()))
    except (TypeError, ValueError):
        return False


def _audit_csv(
    path: Path,
    *,
    expected_model_architecture_profile: str,
    require_true_mixed_attention: bool,
) -> ProfileFileAudit:
    if not path.is_file():
        return ProfileFileAudit(
            path=str(path),
            exists=False,
            row_count=0,
            time_stats_column_count=0,
            time_stats_valid_count=0,
            time_stats_nan_count=0,
            time_stats_empty_row_count=0,
            time_stats_empty_column_count=0,
            true_mixed_row_count=0,
            true_mixed_attn_decode_valid_count=0,
            true_mixed_required_numeric_valid_row_count=0,
            true_mixed_required_numeric_invalid_cell_count=0,
            semantic_coverage_errors=(),
        )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        time_stats_columns = [
            column for column in fieldnames if column.startswith("time_stats.")
        ]
        valid_counts_by_column = {column: 0 for column in time_stats_columns}
        row_count = 0
        nan_count = 0
        valid_count = 0
        empty_row_count = 0
        true_mixed_row_count = 0
        true_mixed_attn_decode_valid_count = 0
        true_mixed_required_numeric_valid_row_count = 0
        true_mixed_required_numeric_invalid_cell_count = 0
        invalid_true_mixed_numeric_columns: dict[str, int] = {}
        gating_runtime_contexts: set[str] = set()
        model_architecture_profiles: set[str] = set()
        for row in reader:
            row_count += 1
            if MOE_GATING_RUNTIME_CONTEXT_COLUMN in row:
                gating_runtime_contexts.add(
                    str(row.get(MOE_GATING_RUNTIME_CONTEXT_COLUMN, "")).strip()
                )
            if "model_architecture_profile" in row:
                raw_profile = str(row.get("model_architecture_profile", "")).strip()
                if raw_profile:
                    model_architecture_profiles.add(raw_profile)
            is_true_mixed_row = _truthy_csv_value(row.get("is_true_mixed_batch", ""))
            if is_true_mixed_row:
                true_mixed_row_count += 1
                if _is_valid_numeric_csv_value(
                    row.get("time_stats.attn_decode.median", "")
                ):
                    true_mixed_attn_decode_valid_count += 1
                required_numeric_row_valid = True
                for column in TRUE_MIXED_ATTENTION_FEATURE_COLUMNS:
                    if _is_valid_numeric_csv_value(row.get(column, "")):
                        continue
                    required_numeric_row_valid = False
                    true_mixed_required_numeric_invalid_cell_count += 1
                    invalid_true_mixed_numeric_columns[column] = (
                        invalid_true_mixed_numeric_columns.get(column, 0) + 1
                    )
                if required_numeric_row_valid:
                    true_mixed_required_numeric_valid_row_count += 1
            row_valid_count = 0
            for column in time_stats_columns:
                if not _is_valid_time_stats_value(row.get(column, "")):
                    nan_count += 1
                    continue
                valid_counts_by_column[column] += 1
                valid_count += 1
                row_valid_count += 1
            if time_stats_columns and row_valid_count == 0:
                empty_row_count += 1

    semantic_coverage_error_list: list[str] = []
    if "model_architecture_profile" not in fieldnames:
        semantic_coverage_error_list.append(
            "model_architecture_profile column is missing"
        )
    elif not model_architecture_profiles:
        semantic_coverage_error_list.append(
            "model_architecture_profile column is empty"
        )
    elif model_architecture_profiles != {expected_model_architecture_profile}:
        semantic_coverage_error_list.append(
            "model_architecture_profile mismatch: "
            f"expected {expected_model_architecture_profile}, "
            f"observed {','.join(sorted(model_architecture_profiles))}"
        )
    if path.name in REQUIRED_MOE_PROFILE_FILES:
        required_context = DEFAULT_MOE_GATING_RUNTIME_CONTEXT
        if required_context not in gating_runtime_contexts:
            semantic_coverage_error_list.append(
                f"{MOE_GATING_RUNTIME_CONTEXT_COLUMN}={required_context}"
            )
    if require_true_mixed_attention and path.name in TRUE_MIXED_ATTENTION_PROFILE_FILES:
        missing_true_mixed_columns = [
            column
            for column in TRUE_MIXED_ATTENTION_REQUIRED_COLUMNS
            if column not in fieldnames
        ]
        if missing_true_mixed_columns:
            semantic_coverage_error_list.append(
                "missing true-mixed attention columns: "
                f"{', '.join(missing_true_mixed_columns)}"
            )
        elif true_mixed_row_count == 0:
            semantic_coverage_error_list.append("true-mixed attention rows are missing")
        elif true_mixed_attn_decode_valid_count != true_mixed_row_count:
            semantic_coverage_error_list.append(
                "true-mixed attention rows with invalid "
                "time_stats.attn_decode.median: "
                f"{true_mixed_row_count - true_mixed_attn_decode_valid_count}/"
                f"{true_mixed_row_count}"
            )
        if not missing_true_mixed_columns and invalid_true_mixed_numeric_columns:
            column_counts = ", ".join(
                f"{column}={invalid_true_mixed_numeric_columns[column]}"
                for column in sorted(invalid_true_mixed_numeric_columns)
            )
            semantic_coverage_error_list.append(
                "true-mixed attention rows have invalid numeric columns: "
                f"{column_counts}"
            )
    semantic_coverage_errors = tuple(semantic_coverage_error_list)

    return ProfileFileAudit(
        path=str(path),
        exists=True,
        row_count=row_count,
        time_stats_column_count=len(time_stats_columns),
        time_stats_valid_count=valid_count,
        time_stats_nan_count=nan_count,
        time_stats_empty_row_count=empty_row_count,
        time_stats_empty_column_count=sum(
            1 for count in valid_counts_by_column.values() if count == 0
        ),
        true_mixed_row_count=true_mixed_row_count,
        true_mixed_attn_decode_valid_count=true_mixed_attn_decode_valid_count,
        true_mixed_required_numeric_valid_row_count=(
            true_mixed_required_numeric_valid_row_count
        ),
        true_mixed_required_numeric_invalid_cell_count=(
            true_mixed_required_numeric_invalid_cell_count
        ),
        semantic_coverage_errors=semantic_coverage_errors,
    )


def audit_requirements(
    *,
    profile_root: Path,
    requirements: Iterable[ProfileRequirement],
    require_true_mixed_attention: bool = False,
) -> tuple[ProfileRequirementAudit, ...]:
    audits: list[ProfileRequirementAudit] = []
    for requirement in requirements:
        file_audits = {
            filename: _audit_csv(
                profile_root / requirement.model_name / filename,
                expected_model_architecture_profile=(
                    requirement.expected_model_architecture_profile
                ),
                require_true_mixed_attention=require_true_mixed_attention,
            )
            for filename in requirement.required_files
        }
        missing_files = tuple(
            filename for filename, audit in file_audits.items() if not audit.exists
        )
        invalid_files = tuple(
            filename
            for filename, audit in file_audits.items()
            if audit.exists
            and (
                audit.row_count == 0
                or audit.time_stats_column_count == 0
                or audit.time_stats_valid_count == 0
                or audit.time_stats_empty_row_count > 0
                or audit.time_stats_empty_column_count > 0
                or len(audit.semantic_coverage_errors) > 0
            )
        )
        if missing_files:
            status = "missing"
        elif invalid_files:
            status = "invalid"
        else:
            status = "present"
        audits.append(
            ProfileRequirementAudit(
                config_filename=requirement.config_filename,
                model_name=requirement.model_name,
                status=status,
                missing_files=missing_files,
                invalid_files=invalid_files,
                files=file_audits,
            )
        )
    return tuple(audits)


def audits_to_dict(audits: Sequence[ProfileRequirementAudit]) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    for audit in audits:
        status_counts[audit.status] = status_counts.get(audit.status, 0) + 1
    return {
        "expected_models": len(audits),
        "status_counts": status_counts,
        "models": [asdict(audit) for audit in audits],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit H800 profiling CSV prerequisites for operator parity.",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("data/config/models"),
        help="Directory containing golden model JSON configs.",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=Path("data/profiling/compute/h800"),
        help="Canonical H800 profiling root to audit.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a JSON audit report.",
    )
    parser.add_argument(
        "--require-true-mixed-attention",
        action="store_true",
        help=(
            "Require CUDA-event and kernel-only attention profile files to include "
            "valid true mixed prefill+decode rows for attn_decode_in_mixed training."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    requirements = build_requirements(config_root=args.config_root)
    audits = audit_requirements(
        profile_root=args.profile_root,
        requirements=requirements,
        require_true_mixed_attention=args.require_true_mixed_attention,
    )
    report = audits_to_dict(audits)
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8")
    return 0 if report["status_counts"] == {"present": len(audits)} else 1


if __name__ == "__main__":
    raise SystemExit(main())

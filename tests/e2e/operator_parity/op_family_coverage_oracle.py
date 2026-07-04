#!/usr/bin/env python3
"""Validate operator-family coverage in golden-matrix op traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from frontier.model_architectures import get_model_architecture_profile
from frontier.operators.binding import build_operator_manifest
from frontier.operators.families import iter_operator_families
from frontier.operators.spec import ResourceClass
from tests.e2e.attention_equivalence.profile_manifest import write_json_report
from tests.e2e.operator_parity.run_golden_matrix import (
    DEFAULT_WORKLOAD_PROFILES,
    GOLDEN_CONFIG_FILENAMES,
    OP_TRACES_CSV,
    build_golden_cases,
)

FAMILY_ATTENTION = "ATTENTION"
FAMILY_FFN = "FFN"
FAMILY_MEMORY = "MEMORY"
FAMILY_MOE = "MOE"
FAMILY_SHARE_EXPERT = "SHARE_EXPERT"
FAMILY_COMM = "COMM"


def _read_model_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"model config missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"model config must be a JSON object: {path}")
    return dict(payload)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _positive_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(value)


@dataclass
class _ManifestModelConfig:
    """Minimal config adapter for manifest-driven coverage checks."""

    model_type: str | None
    model_architecture_profile: str | None
    model_arch: str
    num_q_heads: int
    num_kv_heads: int
    embedding_dim: int
    head_dim: int | None
    is_moe: bool
    num_experts: int
    share_expert_dim: int | None
    use_mla: bool
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    qk_head_dim: int | None = None
    v_head_dim: int | None = None

    def get_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.num_q_heads <= 0:
            raise ValueError("num_q_heads must be positive")
        return self.embedding_dim // self.num_q_heads

    def get_model_architecture_profile(self):
        return get_model_architecture_profile(self)

    def supports_share_expert(self) -> bool:
        return self.get_model_architecture_profile().supports_share_expert(self)


def _config_value(config: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return None


def _manifest_model_config(config: Mapping[str, Any]) -> _ManifestModelConfig:
    num_experts = _positive_int(config.get("num_experts"))
    share_expert_dim = _optional_int(
        _config_value(config, "share_expert_dim", "shared_expert_intermediate_size")
    )
    return _ManifestModelConfig(
        model_type=str(config.get("model_type", "") or "").lower() or None,
        model_architecture_profile=(
            str(config.get("model_architecture_profile", "") or "").lower() or None
        ),
        model_arch=str(config.get("model_arch", "") or "generic").lower(),
        num_q_heads=_positive_int(
            _config_value(config, "num_q_heads", "num_attention_heads")
        ),
        num_kv_heads=_positive_int(
            _config_value(config, "num_kv_heads", "num_key_value_heads")
        ),
        embedding_dim=_positive_int(_config_value(config, "embedding_dim", "hidden_size")),
        head_dim=_optional_int(config.get("head_dim")),
        is_moe=bool(config.get("is_moe", False)) or num_experts > 1,
        num_experts=num_experts,
        share_expert_dim=share_expert_dim,
        use_mla=bool(config.get("use_mla", False)),
        kv_lora_rank=_optional_int(config.get("kv_lora_rank")),
        qk_nope_head_dim=_optional_int(config.get("qk_nope_head_dim")),
        qk_rope_head_dim=_optional_int(config.get("qk_rope_head_dim")),
        qk_head_dim=_optional_int(config.get("qk_head_dim")),
        v_head_dim=_optional_int(config.get("v_head_dim")),
    )


def _coverage_family_label(family_id: str) -> str:
    normalized = family_id.lower()
    if normalized.endswith("_attention"):
        return FAMILY_ATTENTION
    return normalized.upper()


def expected_families_for_model(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Derive expected co-location coverage families from the operator manifest."""

    manifest = build_operator_manifest(_manifest_model_config(config))
    families = {
        _coverage_family_label(binding.family_id)
        for binding in manifest.family_bindings
    }
    return tuple(sorted(families))


def _add_operator_family_mapping(
    mapping: dict[str, str],
    op_name: str,
    family_label: str,
) -> None:
    if not op_name:
        return
    existing = mapping.get(op_name)
    if existing is not None and existing != family_label:
        raise ValueError(
            f"operator {op_name!r} maps to both {existing!r} and {family_label!r}"
        )
    mapping[op_name] = family_label


def _operator_family_by_name_for_model(
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Build observed trace-op mapping from manifest/registry contracts."""

    model_config = _manifest_model_config(config)
    manifest = build_operator_manifest(model_config)
    mapping: dict[str, str] = {}

    for binding in manifest.family_bindings:
        family_label = _coverage_family_label(binding.family_id)
        for operator in binding.family.e2e_trace_ops():
            _add_operator_family_mapping(mapping, operator.name, family_label)
            _add_operator_family_mapping(mapping, operator.profiling_name(), family_label)

    architecture_profile = model_config.get_model_architecture_profile()
    for op_name in (
        *architecture_profile.linear_attention.sharded_ops,
        *architecture_profile.linear_attention.replicated_ops,
    ):
        _add_operator_family_mapping(mapping, op_name, FAMILY_ATTENTION)

    # Include disabled COMM trace families here on purpose.  Request-level
    # transfer events such as kv_cache_transfer are not batch ExecutionTime
    # targets, but the semantic oracle must still recognize them as COMM
    # coverage instead of treating PDD traces as unknown or exempt.
    for family in iter_operator_families():
        if family.resource_class is not ResourceClass.COMM:
            continue
        family_label = FAMILY_COMM
        for operator in family.operators:
            _add_operator_family_mapping(mapping, operator.name, family_label)
            _add_operator_family_mapping(mapping, operator.profiling_name(), family_label)

    return mapping


def _parse_duration_ms(row: Mapping[str, str], *, trace_path: Path, row_index: int) -> float:
    raw_value = row.get("duration_ms", "")
    if raw_value == "":
        raise ValueError(f"missing duration_ms in {trace_path} row {row_index}")
    return float(raw_value)


def _read_trace_rows(trace_path: Path) -> list[dict[str, str]]:
    if not trace_path.is_file():
        raise FileNotFoundError(f"required op trace CSV missing: {trace_path}")
    with trace_path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _case_output_dir(case: Mapping[str, Any], side: str) -> Path:
    key = f"{side}_dir"
    if key not in case:
        raise ValueError(f"case {case.get('name')!r} is missing {key}")
    return Path(str(case[key]))


def _analyze_case(case: Mapping[str, Any], *, side: str) -> dict[str, Any]:
    case_manifest = case.get("case_manifest")
    if not isinstance(case_manifest, Mapping):
        raise ValueError(f"case {case.get('name')!r} is missing case_manifest")

    sys_arch = str(case_manifest.get("sys_arch", ""))
    result: dict[str, Any] = {
        "case_name": str(case.get("name")),
        "side": side,
        "sys_arch": sys_arch,
    }
    if sys_arch not in {"co-location", "pd-disaggregation"}:
        result.update(
            {
                "status": "EXEMPT",
                "exemption_reason": (
                    "coverage oracle supports co-location and "
                    "pd-disaggregation cases only"
                ),
            }
        )
        return result

    config = _read_model_config(str(case_manifest.get("config_path", "")))
    expected_families = expected_families_for_model(config)
    if sys_arch == "pd-disaggregation":
        expected_families = tuple(sorted({*expected_families, FAMILY_COMM}))
    op_family_by_name = _operator_family_by_name_for_model(config)
    trace_path = _case_output_dir(case, side) / OP_TRACES_CSV
    rows = _read_trace_rows(trace_path)

    observed_rows_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in sorted(set(op_family_by_name.values()))
    }
    zero_only_candidates: dict[str, int] = {family: 0 for family in expected_families}
    unknown_observed_ops: list[str] = []
    unknown_nonzero_observed_ops: list[str] = []
    for row_index, row in enumerate(rows, start=2):
        op_name = str(row.get("name", ""))
        duration_ms = _parse_duration_ms(row, trace_path=trace_path, row_index=row_index)
        family = op_family_by_name.get(op_name)
        if family is None:
            if op_name:
                unknown_observed_ops.append(op_name)
                if duration_ms > 0.0:
                    unknown_nonzero_observed_ops.append(op_name)
            continue
        if family in zero_only_candidates:
            zero_only_candidates[family] += 1
        if duration_ms > 0.0:
            observed_rows_by_family.setdefault(family, []).append(
                {
                    "trace_index": row.get("trace_index", ""),
                    "name": op_name,
                    "duration_ms": duration_ms,
                }
            )

    observed_nonzero_families = sorted(
        family for family, family_rows in observed_rows_by_family.items() if family_rows
    )
    missing_families = [
        family for family in expected_families if family not in observed_nonzero_families
    ]
    zero_only_families = [
        family
        for family in missing_families
        if zero_only_candidates.get(family, 0) > 0
    ]
    status = "PASS" if not missing_families and not unknown_nonzero_observed_ops else "FAIL"
    result.update(
        {
            "status": status,
            "trace_path": str(trace_path),
            "trace_row_count": len(rows),
            "expected_families": list(expected_families),
            "observed_nonzero_families": observed_nonzero_families,
            "observed_registry_ops_by_family": {
                family: observed_rows_by_family.get(family, [])
                for family in expected_families
            },
            "missing_families": missing_families,
            "zero_only_families": zero_only_families,
            "unknown_observed_ops": sorted(set(unknown_observed_ops)),
            "unknown_nonzero_observed_ops": sorted(set(unknown_nonzero_observed_ops)),
        }
    )
    return result


def run_coverage_oracle(cases: Sequence[Mapping[str, Any]], *, side: str) -> dict[str, Any]:
    if side not in {"reference", "candidate"}:
        raise ValueError(f"side must be 'reference' or 'candidate', got {side!r}")
    case_reports = [_analyze_case(case, side=side) for case in cases]
    checked = [case for case in case_reports if case["status"] != "EXEMPT"]
    failed = [case for case in checked if case["status"] != "PASS"]
    if not checked:
        status = "FAIL_NO_CHECKED_CASES"
    else:
        status = "PASS" if not failed else "FAIL"
    return {
        "status": status,
        "side": side,
        "case_count": len(case_reports),
        "checked_case_count": len(checked),
        "exempt_case_count": len(case_reports) - len(checked),
        "failed_case_count": len(failed),
        "cases": case_reports,
    }


def run_coverage_oracle_from_roots(
    *,
    config_root: Path,
    profile_root: Path,
    reference_root: Path,
    candidate_root: Path,
    side: str,
    workload_profiles: Sequence[str] = DEFAULT_WORKLOAD_PROFILES,
) -> dict[str, Any]:
    cases, case_manifest = build_golden_cases(
        config_root=config_root,
        profile_root=profile_root,
        reference_root=reference_root,
        candidate_root=candidate_root,
        config_filenames=GOLDEN_CONFIG_FILENAMES,
        workload_profiles=workload_profiles,
    )
    report = run_coverage_oracle(cases, side=side)
    report["case_manifest"] = case_manifest
    return report


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
    parser.add_argument("--side", choices=("reference", "candidate"), default="reference")
    parser.add_argument(
        "--workload-profiles",
        default=",".join(DEFAULT_WORKLOAD_PROFILES),
        help=(
            "Comma-separated golden workload profiles for coverage analysis. "
            "The default preserves the archived 48-case reference layout; use "
            "long_single for supplemental workload-shape coverage."
        ),
    )
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_coverage_oracle_from_roots(
        config_root=args.config_root,
        profile_root=args.profile_root,
        reference_root=args.reference_root,
        candidate_root=args.candidate_root,
        side=args.side,
        workload_profiles=_parse_csv_list(args.workload_profiles),
    )
    write_json_report(args.json_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

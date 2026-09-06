"""Parity harness schema and comparison utilities for pd-af-disaggregation.

Compares simulation outputs between the main implementation and the reference
branch (integration/afd-only-merge) to verify numerical and structural parity.
"""

import ast
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


class ParityInputError(ValueError):
    """Raised when a parity input artifact is missing, ambiguous, or malformed."""


class ParityLayer(Enum):
    """Parity validation layer."""
    L1_DUMMY = "l1_dummy"
    L2_TRAINED = "l2_trained"


class ComparisonResult(Enum):
    """Result of a single field comparison."""
    EXACT_MATCH = "exact_match"
    WITHIN_TOLERANCE = "within_tolerance"
    MISMATCH = "mismatch"
    MISSING_FIELD = "missing_field"
    INVALID_VALUE = "invalid_value"


@dataclass
class ToleranceSpec:
    """Tolerance specification for numerical comparisons."""
    abs_tol: float = 1e-9
    rel_tol: float = 0.0
    description: str = ""


@dataclass
class FieldComparison:
    """Result of comparing a single field."""
    field_name: str
    main_value: object
    ref_value: object
    result: ComparisonResult
    abs_delta: Optional[float] = None
    rel_delta: Optional[float] = None
    tolerance_applied: Optional[ToleranceSpec] = None


@dataclass
class RequestComparison:
    """Per-request comparison result."""
    request_id: int
    fields: List[FieldComparison] = field(default_factory=list)
    passed: bool = True
    first_divergence_field: Optional[str] = None
    canonical_main_ttft_ms: Optional[float] = None
    canonical_main_ttft_provenance: Optional[str] = None
    main_cross_branch_ttft_ms: Optional[float] = None
    reference_cross_branch_ttft_ms: Optional[float] = None
    cross_branch_ttft_abs_delta_ms: Optional[float] = None
    cross_branch_ttft_rel_delta: Optional[float] = None
    cross_branch_ttft_passed: Optional[bool] = None
    main_cross_branch_ttft_provenance: Optional[str] = None
    reference_cross_branch_ttft_provenance: Optional[str] = None


@dataclass
class ParityCaseConfig:
    """Configuration for a single parity test case."""
    case_id: str
    model: str
    mode: str
    scale_gpu: int
    prefill_tokens: int
    decode_tokens: int
    num_requests: int
    cuda_graph: bool = False
    layer: ParityLayer = ParityLayer.L2_TRAINED
    qps: Optional[float] = None
    description: str = ""


@dataclass
class ParityReport:
    """Full parity comparison report."""
    case_config: ParityCaseConfig
    main_output_dir: str
    ref_output_dir: str
    request_comparisons: List[RequestComparison] = field(default_factory=list)
    overall_pass: bool = True
    total_fields_compared: int = 0
    total_mismatches: int = 0
    first_divergence_event_index: Optional[int] = None
    event_comparison: Optional["EventComparisonReport"] = None
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class NormalizedEventRecord:
    """One canonical COMPLETE event from a cluster-role log."""

    event_class: str
    cluster_role: str
    event_time: float
    fields: Mapping[str, object]
    source_line: int


@dataclass
class EventMismatch:
    """One failing global DES-time group."""

    category: str
    event_time: Optional[float]
    main_records: List[NormalizedEventRecord] = field(default_factory=list)
    ref_records: List[NormalizedEventRecord] = field(default_factory=list)
    mismatched_fields: List[str] = field(default_factory=list)


@dataclass
class RoleEventSummary:
    """Diagnostic event counts for one cluster role."""

    role: str
    main_count: int
    ref_count: int
    passed: bool


@dataclass
class EventComparisonReport:
    """Global merged event-choreography comparison result."""

    passed: bool
    total_events_main: int
    total_events_ref: int
    total_mismatches: int
    first_divergence_event_index: Optional[int]
    per_role: Dict[str, RoleEventSummary]
    mismatches: List[EventMismatch] = field(default_factory=list)


# Discrete fields: must be exact match
DISCRETE_FIELDS = [
    "request_num_prefill_tokens",
    "request_num_decode_tokens",
    "request_num_tokens",
    "request_num_restarts",
    "request_thinking_round_count",
]

# Floating-point metric fields with tolerance
METRIC_TOLERANCES: Dict[str, ToleranceSpec] = {
    "request_e2e_time": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6, description="End-to-end latency"),
    "request_execution_time": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6),
    "prefill_e2e_time": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6),
    "decode_e2e_time": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6),
    "cross_branch_first_token_ttft_ms": ToleranceSpec(
        abs_tol=1e-6,
        rel_tol=1e-6,
        description="Arrival to first completed decode token",
    ),
    "tpot": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6, description="Time per output token"),
    "transfer_kv_cache": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6),
    "transfer_m2n_total": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6),
    "transfer_m2n_attn_to_ffn": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6),
    "transfer_m2n_ffn_to_attn": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6),
    "cluster_prefill_computation": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6),
    "cluster_decode_attn_computation": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6),
    "cluster_decode_ffn_computation": ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6),
}

CANONICAL_MAIN_TTFT_FIELD = "canonical_main_ttft_ms"
CROSS_BRANCH_TTFT_FIELD = "cross_branch_first_token_ttft_ms"
REFERENCE_FIRST_REAL_DECODE_LIFECYCLE_FILENAME = (
    "reference_first_real_decode_lifecycle.json"
)
REFERENCE_FIRST_REAL_DECODE_SCHEMA_VERSION = (
    "frontier.pdaf.reference-first-real-decode/v1"
)
REFERENCE_REPO_ROOT = Path(
    "/data/ycfeng/stepfun-performance-optimization/Frontier/"
    "worktrees/ref-afd-readonly"
)
REFERENCE_GIT_HEAD = "dcb1cc8ee160a9c3c5412293d93b64042960aa4d"
REFERENCE_REQUEST_SOURCE_SHA256 = (
    "4cff6da775a1b04ba4c252ccc679a3f2919ed5bfc98f1c039dff1519b9bc42b0"
)
REFERENCE_CLUSTER_SCHEDULER_SOURCE_SHA256 = (
    "5a28d18a7cfdcfc04b2848a9861973c652947005b356fa9b837b90356329fb6d"
)
REFERENCE_GLOBAL_BATCH_END_EVENT_SOURCE_SHA256 = (
    "5366bd739c9765ef57b06448ce719d013795273535fd623aa17ed064279021b0"
)
REFERENCE_TTFT_PROVENANCE = (
    "reference_first_real_decode_lifecycle.json#"
    f"{REFERENCE_FIRST_REAL_DECODE_SCHEMA_VERSION}@{REFERENCE_GIT_HEAD}"
)
_REFERENCE_LIFECYCLE_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "producer",
        "request_count",
        "request_ids_sha256",
        "requests",
    }
)
_REFERENCE_LIFECYCLE_PRODUCER_KEYS = frozenset(
    {
        "branch_kind",
        "reference_repo_root",
        "reference_git_head",
        "python_executable",
        "argv_sha256",
        "observer_source_sha256",
        "bootstrap_source_sha256",
        "request_source_sha256",
        "cluster_scheduler_source_sha256",
        "global_batch_end_event_source_sha256",
        "candidate_hook",
        "transition_hook",
        "transition_contract",
        "timestamp_contract",
    }
)
_REFERENCE_LIFECYCLE_REQUEST_KEYS = frozenset(
    {
        "request_id",
        "cluster_type",
        "arrived_at_s",
        "prefill_completed_at_s",
        "raw_decode_execution_completed_at_s",
        "resolved_global_end_time_s",
        "processed_decode_tokens_before",
        "processed_decode_tokens_after",
    }
)
CANONICAL_MAIN_TTFT_PROVENANCE = "metrics_ground_truth.jsonl"
MAIN_CROSS_BRANCH_TTFT_PROVENANCE = (
    "metrics_ground_truth.jsonl#request_completion."
    "first_decode_token_completed_at"
)
_HARD_FAILURE_RESULTS = {
    ComparisonResult.MISMATCH,
    ComparisonResult.MISSING_FIELD,
    ComparisonResult.INVALID_VALUE,
}

_EVENT_ROLES = ("PREFILL", "DECODE_ATTN", "DECODE_FFN")
_EVENT_LINE_RE = re.compile(
    r"^\[[^\]]+\] (START|COMPLETE|ERROR) ([A-Za-z0-9_]+) "
    r"\| ID: ([^|]+?)(?: \| (.*))?$"
)
_EVENT_EXCLUDED_FIELDS = {
    "Duration",
    "event_id",
    "event_type",
    "time",
    "cluster",
    "cluster_type",
    "cluster_time",
    "batch_id",
    "batch_global_id",
    "batch_stage_id",
    "source_event_id",
    "batch_ids",
    "batch_stage_start_time",
    "batch_stage_end_time",
    "num_layers_per_stage",
}
_EVENT_INT_FIELDS = {
    "request_id",
    "source_replica_id",
    "replica_id",
    "dp_id",
    "ep_id",
    "stage_id",
    "layer_id",
    "afd_stage_idx",
    "activation_size_bytes",
    "kv_cache_size_bytes",
    "num_tokens",
    "batch_size",
    "new_events_generated",
    "decode_step",
}
_EVENT_FLOAT_MS_FIELDS = {"transfer_time_ms", "scheduling_interval_ms"}
_EVENT_FLOAT_SECOND_FIELDS = {
    "stage_execution_time",
    "batch_stage_execution_time",
    "transfer_start_time",
    "transfer_end_time",
    "combine_end_time",
}
_EVENT_BOOL_FIELDS = {"is_attn_to_ffn", "is_last_stage"}
_EVENT_LIST_FIELDS = {
    "request_ids",
    "request_decode_steps",
    "request_layer_ids",
    "cluster_set",
    "replica_dp_set",
    "replica_ep_set",
    "request_mapping",
}
_EVENT_STRING_FIELDS = {
    "target_cluster",
    "source_cluster_type",
    "target_cluster_type",
    "sync_stage",
    "pipeline_stage",
}
_EVENT_COMMON_REQUIRED = {"target_cluster", "new_events_generated"}
_EVENT_COMMON_BATCH_FIELDS = {
    "request_ids",
    "request_decode_steps",
    "request_layer_ids",
    "decode_step",
    "layer_id",
    "replica_id",
    "dp_id",
    "batch_size",
    "num_tokens",
}


def _event_schema(
    required: Sequence[str], optional: Sequence[str] = ()
) -> tuple[frozenset[str], frozenset[str]]:
    required_fields = frozenset((*_EVENT_COMMON_REQUIRED, *required))
    return required_fields, frozenset((*required_fields, *optional))


_EVENT_SCHEMAS = {
    "RequestArrivalEvent": _event_schema(["request_id"]),
    "ThinkingRoundRequeueEvent": _event_schema(
        ["request_id", "replica_id", "dp_id"]
    ),
    "GlobalScheduleEvent": _event_schema(["cluster_set", "request_mapping"]),
    "ClusterScheduleEvent": _event_schema(
        ["replica_dp_set", "replica_ep_set", "request_mapping"]
    ),
    "ReplicaScheduleEvent": _event_schema(
        ["replica_id", "dp_id"], ["request_ids"]
    ),
    "ReplicaStageScheduleEvent": _event_schema(
        ["replica_id", "stage_id", "dp_id", "is_last_stage"]
    ),
    "BatchStageArrivalEvent": _event_schema(
        ["replica_id", "stage_id", "dp_id", "request_ids"],
        list(_EVENT_COMMON_BATCH_FIELDS),
    ),
    "BatchStageEndEvent": _event_schema(
        [
            "replica_id",
            "stage_id",
            "dp_id",
            "is_last_stage",
            "request_ids",
            "layer_id",
            "num_tokens",
            "batch_stage_execution_time",
        ],
        list(_EVENT_COMMON_BATCH_FIELDS),
    ),
    "BatchEndEvent": _event_schema(
        ["replica_id", "dp_id", "request_ids"],
        list(_EVENT_COMMON_BATCH_FIELDS),
    ),
    "ClusterBatchEndEvent": _event_schema(
        ["replica_id", "dp_id", "request_ids"],
        list(_EVENT_COMMON_BATCH_FIELDS),
    ),
    "GlobalBatchEndEvent": _event_schema(
        ["replica_id", "dp_id", "request_ids"],
        list(_EVENT_COMMON_BATCH_FIELDS),
    ),
    "PrefillSyncEvent": _event_schema(
        [
            "replica_id",
            "stage_id",
            "dp_id",
            "sync_stage",
            "layer_id",
            "stage_execution_time",
        ],
        ["request_ids"],
    ),
    "DecodeSyncEvent": _event_schema(
        [
            "replica_id",
            "stage_id",
            "dp_id",
            "sync_stage",
            "layer_id",
            "stage_execution_time",
        ],
        ["request_ids"],
    ),
    "PrefillSyncCollectiveEvent": _event_schema(
        ["replica_id", "stage_id", "sync_stage", "layer_id"]
    ),
    "DecodeSyncCollectiveEvent": _event_schema(
        ["replica_id", "stage_id", "sync_stage", "layer_id"]
    ),
    "EPAllToAllDispatchReadyEvent": _event_schema(
        ["replica_id", "stage_id", "ep_id"]
    ),
    "EPAllToAllDispatchCollectiveEvent": _event_schema(
        ["replica_id", "stage_id"]
    ),
    "EPAllToAllCombineReadyEvent": _event_schema(
        ["replica_id", "stage_id", "ep_id"]
    ),
    "EPAllToAllCombineCollectiveEvent": _event_schema(
        ["replica_id", "stage_id"],
        ["combine_end_time"],
    ),
    "M2NTransferStartEvent": _event_schema(
        [
            "source_cluster_type",
            "target_cluster_type",
            "source_replica_id",
            "layer_id",
            "afd_stage_idx",
            "activation_size_bytes",
            "request_ids",
            "transfer_time_ms",
        ],
        [
            "is_attn_to_ffn",
            "num_tokens",
            "batch_size",
            "replica_id",
            "dp_id",
            "request_decode_steps",
            "request_layer_ids",
        ],
    ),
    "M2NTransferEndEvent": _event_schema(
        [
            "source_cluster_type",
            "target_cluster_type",
            "source_replica_id",
            "layer_id",
            "is_attn_to_ffn",
            "activation_size_bytes",
            "request_ids",
            "pipeline_stage",
            "transfer_time_ms",
            "transfer_start_time",
            "transfer_end_time",
        ],
        [
            "num_tokens",
            "batch_size",
            "replica_id",
            "dp_id",
            "request_decode_steps",
            "request_layer_ids",
        ],
    ),
    "KVCacheTransferStartEvent": _event_schema(
        [
            "source_cluster_type",
            "target_cluster_type",
            "source_replica_id",
            "kv_cache_size_bytes",
            "request_ids",
            "transfer_time_ms",
        ],
        ["num_tokens", "batch_size", "request_decode_steps", "request_layer_ids"],
    ),
    "KVCacheTransferEndEvent": _event_schema(
        [
            "source_cluster_type",
            "target_cluster_type",
            "source_replica_id",
            "kv_cache_size_bytes",
            "request_ids",
            "transfer_time_ms",
            "transfer_start_time",
            "transfer_end_time",
        ],
        ["num_tokens", "batch_size", "request_decode_steps", "request_layer_ids"],
    ),
}


def load_event_records(output_dir: str) -> List[NormalizedEventRecord]:
    """Load and validate the three PD-AF cluster-role event logs."""
    role_paths = _discover_event_logs(output_dir)
    records: List[NormalizedEventRecord] = []
    for role in _EVENT_ROLES:
        records.extend(_parse_event_log(role_paths[role], role))
    return records


def compare_event_logs(
    main_output_dir: str, ref_output_dir: str
) -> EventComparisonReport:
    """Compare one globally merged sequential DES choreography per tree."""
    main_records = load_event_records(main_output_dir)
    ref_records = load_event_records(ref_output_dir)
    main_groups = _group_event_records(main_records)
    ref_groups = _group_event_records(ref_records)
    per_role = {
        role: RoleEventSummary(
            role=role,
            main_count=sum(record.cluster_role == role for record in main_records),
            ref_count=sum(record.cluster_role == role for record in ref_records),
            passed=True,
        )
        for role in _EVENT_ROLES
    }
    mismatches: List[EventMismatch] = []
    first_divergence: Optional[int] = None
    base_index = 0

    for event_time in sorted(set(main_groups) | set(ref_groups)):
        main_group = main_groups.get(event_time, [])
        ref_group = ref_groups.get(event_time, [])
        mismatch = _compare_event_time_group(event_time, main_group, ref_group)
        if mismatch is not None:
            mismatches.append(mismatch)
            if first_divergence is None:
                first_divergence = base_index
            affected_roles = {
                record.cluster_role
                for record in (*mismatch.main_records, *mismatch.ref_records)
            }
            for role in affected_roles:
                per_role[role].passed = False
        base_index += len(main_group) + len(ref_group)

    return EventComparisonReport(
        passed=not mismatches,
        total_events_main=len(main_records),
        total_events_ref=len(ref_records),
        total_mismatches=len(mismatches),
        first_divergence_event_index=first_divergence,
        per_role=per_role,
        mismatches=mismatches,
    )


def _discover_event_logs(output_dir: str) -> Dict[str, Path]:
    root = Path(output_dir)
    if not root.is_dir():
        raise ParityInputError(f"Event output directory does not exist: {output_dir}")
    result: Dict[str, Path] = {}
    for role in _EVENT_ROLES:
        prefix = role.lower()
        direct = sorted(
            path
            for path in root.glob(f"{prefix}*.log")
            if path.is_file() and (path.stem == prefix or path.stem.startswith(prefix + "_"))
        )
        if len(direct) > 1:
            raise ParityInputError(
                f"Ambiguous event logs for role {role}: "
                + ", ".join(str(path) for path in direct)
            )
        if direct:
            result[role] = direct[0]
            continue
        recursive = sorted(
            path
            for path in root.rglob(f"{prefix}*.log")
            if path.is_file() and (path.stem == prefix or path.stem.startswith(prefix + "_"))
        )
        if not recursive:
            raise ParityInputError(f"Missing event log for role {role} under {output_dir}")
        if len(recursive) > 1:
            raise ParityInputError(
                f"Ambiguous event logs for role {role}: "
                + ", ".join(str(path) for path in recursive)
            )
        result[role] = recursive[0]
    return result


def _parse_event_log(path: Path, expected_role: str) -> List[NormalizedEventRecord]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    header_roles = [
        line.split(":", 1)[1].strip()
        for line in raw_lines
        if line.startswith("Cluster Type:")
    ]
    if len(header_roles) != 1 or _normalize_cluster_name(header_roles[0]) != expected_role:
        raise ParityInputError(
            f"Event log header cluster type mismatch for {path}: "
            f"expected={expected_role}, found={header_roles}"
        )

    pair_counts: Dict[tuple[str, str], Dict[str, int]] = {}
    records: List[NormalizedEventRecord] = []
    last_event_time: Optional[float] = None
    in_summary = False
    for line_number, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "=== EVENT PROCESSING SUMMARY ===":
            in_summary = True
            continue
        if in_summary:
            if _EVENT_LINE_RE.match(stripped):
                raise ParityInputError(
                    f"Event record appears after summary at {path}:{line_number}"
                )
            continue
        if _is_event_log_header_line(stripped):
            continue
        if stripped.startswith("[") and " BATCH " in stripped:
            raise ParityInputError(
                f"Unsupported BATCH record at {path}:{line_number}"
            )
        match = _EVENT_LINE_RE.match(stripped)
        if match is None:
            raise ParityInputError(
                f"Unrecognized event log line at {path}:{line_number}: {stripped!r}"
            )
        phase, event_class, raw_id, raw_details = match.groups()
        if event_class not in _EVENT_SCHEMAS:
            raise ParityInputError(
                f"Unknown event class {event_class!r} at {path}:{line_number}"
            )
        details = _parse_event_details(raw_details or "", path, line_number)
        pair_key = (event_class, raw_id.strip())
        counts = pair_counts.setdefault(pair_key, {"START": 0, "COMPLETE": 0})
        if phase == "ERROR":
            raise ParityInputError(
                f"ERROR record present at {path}:{line_number}: {event_class}"
            )
        counts[phase] += 1
        if phase != "COMPLETE":
            continue
        event_time = _parse_event_float(
            details.get("event_time"), "event_time", path, line_number
        )
        if last_event_time is not None and event_time < last_event_time:
            raise ParityInputError(
                f"event_time ordering violation at {path}:{line_number}: "
                f"{event_time} < {last_event_time}"
            )
        last_event_time = event_time
        records.append(
            _normalize_event_record(
                event_class, expected_role, event_time, details, line_number, path
            )
        )

    for (event_class, raw_id), counts in pair_counts.items():
        if counts != {"START": 1, "COMPLETE": 1}:
            raise ParityInputError(
                "START/COMPLETE pair integrity failure in "
                f"{path}: class={event_class}, id={raw_id}, counts={counts}"
            )
    if not records:
        raise ParityInputError(f"Event log contains no canonical event records: {path}")
    return records


def _is_event_log_header_line(line: str) -> bool:
    return (
        line == "=== VIDUR CLUSTER EVENT LOG ==="
        or line.startswith("Cluster Type:")
        or line.startswith("Start Time:")
        or line.startswith("Log Level:")
        or line.startswith("Log File:")
        or set(line) <= {"="}
    )


def _parse_event_details(raw: str, path: Path, line_number: int) -> Dict[str, str]:
    details: Dict[str, str] = {}
    if not raw:
        return details
    for segment in raw.split(" | "):
        if ": " not in segment:
            raise ParityInputError(
                f"Malformed event detail at {path}:{line_number}: {segment!r}"
            )
        key, value = segment.split(": ", 1)
        if not key or key in details:
            raise ParityInputError(
                f"Duplicate detail key {key!r} at {path}:{line_number}"
            )
        details[key] = value
    return details


def _normalize_event_record(
    event_class: str,
    cluster_role: str,
    event_time: float,
    raw_fields: Mapping[str, str],
    source_line: int,
    path: Path,
) -> NormalizedEventRecord:
    required_fields, allowed_fields = _EVENT_SCHEMAS[event_class]
    fields: Dict[str, object] = {}
    for field_name, raw_value in raw_fields.items():
        if field_name in _EVENT_EXCLUDED_FIELDS or field_name == "event_time":
            continue
        if field_name not in allowed_fields:
            raise ParityInputError(
                f"Unknown field {field_name!r} for {event_class} at "
                f"{path}:{source_line}"
            )
        fields[field_name] = _parse_event_field(
            field_name, raw_value, path, source_line
        )

    missing = sorted(required_fields - fields.keys())
    if missing:
        raise ParityInputError(
            f"Required field missing for {event_class} at {path}:{source_line}: {missing}"
        )
    _normalize_request_positions(fields, event_class, path, source_line)
    if event_class.startswith("M2NTransfer"):
        fields.pop("replica_id", None)
        fields.pop("dp_id", None)
    return NormalizedEventRecord(
        event_class=event_class,
        cluster_role=cluster_role,
        event_time=event_time,
        fields=fields,
        source_line=source_line,
    )


def _parse_event_field(
    field_name: str, raw_value: str, path: Path, line_number: int
) -> object:
    if field_name in _EVENT_INT_FIELDS:
        if raw_value == "unknown" and field_name in {"num_tokens", "batch_size"}:
            return raw_value
        try:
            number = int(raw_value)
        except ValueError as exc:
            raise ParityInputError(
                f"Expected integer field {field_name!r} at {path}:{line_number}: "
                f"{raw_value!r}"
            ) from exc
        return number
    if field_name in _EVENT_FLOAT_MS_FIELDS | _EVENT_FLOAT_SECOND_FIELDS:
        return _parse_event_float(raw_value, field_name, path, line_number)
    if field_name in _EVENT_BOOL_FIELDS:
        if raw_value not in {"True", "False"}:
            if raw_value == "None" and field_name == "is_last_stage":
                return None
            raise ParityInputError(
                f"Expected bool field {field_name!r} at {path}:{line_number}: "
                f"{raw_value!r}"
            )
        return raw_value == "True"
    if field_name in _EVENT_LIST_FIELDS:
        value = _safe_parse_event_literal(raw_value, field_name, path, line_number)
        if not isinstance(value, (list, tuple)):
            raise ParityInputError(
                f"Expected list literal for {field_name!r} at {path}:{line_number}"
            )
        normalized = _normalize_event_literal(value)
        if field_name in {
            "cluster_set",
            "replica_dp_set",
            "replica_ep_set",
            "request_mapping",
        }:
            return tuple(sorted(normalized, key=repr))
        return normalized
    if field_name in _EVENT_STRING_FIELDS:
        if field_name in {
            "target_cluster",
            "source_cluster_type",
            "target_cluster_type",
        }:
            return _normalize_cluster_name(raw_value)
        if field_name == "pipeline_stage":
            try:
                return int(raw_value)
            except ValueError:
                return raw_value
        return raw_value.lower()
    raise ParityInputError(
        f"No parser declared for event field {field_name!r} at {path}:{line_number}"
    )


def _parse_event_float(
    raw_value: object, field_name: str, path: Path, line_number: int
) -> float:
    try:
        value = float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ParityInputError(
            f"Expected numeric field {field_name!r} at {path}:{line_number}: "
            f"{raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise ParityInputError(
            f"Expected finite field {field_name!r} at {path}:{line_number}: "
            f"{raw_value!r}"
        )
    return value


def _safe_parse_event_literal(
    raw_value: str, field_name: str, path: Path, line_number: int
) -> object:
    try:
        value = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise ParityInputError(
            f"Invalid literal for {field_name!r} at {path}:{line_number}: "
            f"{raw_value!r}"
        ) from exc
    _validate_event_literal(value, field_name, path, line_number)
    return value


def _validate_event_literal(
    value: object, field_name: str, path: Path, line_number: int, depth: int = 0
) -> None:
    if depth > 10:
        raise ParityInputError(
            f"Literal nesting exceeds 10 for {field_name!r} at {path}:{line_number}"
        )
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ParityInputError(
                f"Non-finite literal for {field_name!r} at {path}:{line_number}"
            )
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_event_literal(item, field_name, path, line_number, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_event_literal(key, field_name, path, line_number, depth + 1)
            _validate_event_literal(item, field_name, path, line_number, depth + 1)
        return
    raise ParityInputError(
        f"Disallowed literal type for {field_name!r} at {path}:{line_number}: "
        f"{type(value).__name__}"
    )


def _normalize_event_literal(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    _normalize_event_literal(key),
                    _normalize_event_literal(item),
                )
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_event_literal(item) for item in value)
    if isinstance(value, str) and value.startswith("ClusterType."):
        return _normalize_cluster_name(value)
    return value


def _normalize_cluster_name(raw_value: str) -> str:
    value = raw_value.strip().strip("'\"")
    if "." in value:
        value = value.rsplit(".", 1)[1]
    return value.upper().replace("-", "_")


def _normalize_request_positions(
    fields: Dict[str, object], event_class: str, path: Path, line_number: int
) -> None:
    if "request_ids" not in fields:
        return
    request_ids = fields["request_ids"]
    if not isinstance(request_ids, tuple) or not all(
        isinstance(request_id, int) and not isinstance(request_id, bool)
        for request_id in request_ids
    ):
        raise ParityInputError(
            f"request_ids must be a list of integers for {event_class} at "
            f"{path}:{line_number}"
        )
    if len(set(request_ids)) != len(request_ids):
        raise ParityInputError(
            f"Duplicate request ID for {event_class} at {path}:{line_number}"
        )
    order = sorted(range(len(request_ids)), key=request_ids.__getitem__)
    fields["request_ids"] = tuple(request_ids[index] for index in order)
    for positional_name in ("request_decode_steps", "request_layer_ids"):
        if positional_name not in fields:
            continue
        values = fields[positional_name]
        if not isinstance(values, tuple) or len(values) != len(request_ids):
            raise ParityInputError(
                f"Positional array length mismatch for {positional_name!r} in "
                f"{event_class} at {path}:{line_number}"
            )
        fields[positional_name] = tuple(values[index] for index in order)


def _group_event_records(
    records: Sequence[NormalizedEventRecord],
) -> Dict[float, List[NormalizedEventRecord]]:
    groups: Dict[float, List[NormalizedEventRecord]] = {}
    for record in records:
        groups.setdefault(record.event_time, []).append(record)
    return groups


def _compare_event_time_group(
    event_time: float,
    main_group: Sequence[NormalizedEventRecord],
    ref_group: Sequence[NormalizedEventRecord],
) -> Optional[EventMismatch]:
    if not main_group or not ref_group:
        return EventMismatch(
            category="TIME_GROUP_MISMATCH",
            event_time=event_time,
            main_records=list(main_group),
            ref_records=list(ref_group),
        )
    if len(main_group) != len(ref_group):
        return EventMismatch(
            category="MULTIPLICITY_MISMATCH",
            event_time=event_time,
            main_records=list(main_group),
            ref_records=list(ref_group),
        )

    full_graph = [
        [
            ref_index
            for ref_index, ref_record in enumerate(ref_group)
            if _event_records_compatible(main_record, ref_record)
        ]
        for main_record in main_group
    ]
    if _has_perfect_event_matching(full_graph, len(ref_group)):
        return None

    category = "SEMANTIC_MISMATCH"
    mismatched_fields: List[str] = []
    if _is_role_shift(main_group, ref_group):
        category = "ROLE_SHIFT_MISMATCH"
    else:
        presence_fields = _find_presence_mismatches(main_group, ref_group)
        if presence_fields:
            category = "FIELD_PRESENCE_MISMATCH"
            mismatched_fields = presence_fields
        elif _has_identity_perfect_matching(main_group, ref_group):
            category = "FIELD_VALUE_MISMATCH"
            mismatched_fields = _find_value_mismatches(main_group, ref_group)
    return EventMismatch(
        category=category,
        event_time=event_time,
        main_records=list(main_group),
        ref_records=list(ref_group),
        mismatched_fields=mismatched_fields,
    )


def _event_records_compatible(
    main_record: NormalizedEventRecord, ref_record: NormalizedEventRecord
) -> bool:
    if not _event_identity_compatible(main_record, ref_record):
        return False
    return not _event_value_mismatches(main_record, ref_record)


def _event_identity_compatible(
    main_record: NormalizedEventRecord, ref_record: NormalizedEventRecord
) -> bool:
    if (
        main_record.event_class != ref_record.event_class
        or main_record.cluster_role != ref_record.cluster_role
        or main_record.fields.keys() != ref_record.fields.keys()
    ):
        return False
    for field_name in main_record.fields:
        if field_name in _EVENT_FLOAT_MS_FIELDS | _EVENT_FLOAT_SECOND_FIELDS:
            continue
        if main_record.fields[field_name] != ref_record.fields[field_name]:
            return False
    return True


def _event_value_mismatches(
    main_record: NormalizedEventRecord, ref_record: NormalizedEventRecord
) -> List[str]:
    mismatches: List[str] = []
    for field_name in main_record.fields.keys() & ref_record.fields.keys():
        main_value = main_record.fields[field_name]
        ref_value = ref_record.fields[field_name]
        if field_name in _EVENT_FLOAT_MS_FIELDS:
            if abs(float(main_value) - float(ref_value)) > 1e-9:
                mismatches.append(field_name)
        elif field_name in _EVENT_FLOAT_SECOND_FIELDS:
            if abs(float(main_value) - float(ref_value)) > 1e-12:
                mismatches.append(field_name)
        elif main_value != ref_value:
            mismatches.append(field_name)
    return sorted(mismatches)


def _has_perfect_event_matching(graph: Sequence[Sequence[int]], ref_count: int) -> bool:
    matched_main_by_ref = [-1] * ref_count

    def augment(main_index: int, seen: set[int]) -> bool:
        for ref_index in graph[main_index]:
            if ref_index in seen:
                continue
            seen.add(ref_index)
            prior_main = matched_main_by_ref[ref_index]
            if prior_main == -1 or augment(prior_main, seen):
                matched_main_by_ref[ref_index] = main_index
                return True
        return False

    return all(augment(main_index, set()) for main_index in range(len(graph)))


def _has_identity_perfect_matching(
    main_group: Sequence[NormalizedEventRecord],
    ref_group: Sequence[NormalizedEventRecord],
) -> bool:
    graph = [
        [
            ref_index
            for ref_index, ref_record in enumerate(ref_group)
            if _event_identity_compatible(main_record, ref_record)
        ]
        for main_record in main_group
    ]
    return _has_perfect_event_matching(graph, len(ref_group))


def _find_presence_mismatches(
    main_group: Sequence[NormalizedEventRecord],
    ref_group: Sequence[NormalizedEventRecord],
) -> List[str]:
    fields: set[str] = set()
    for main_record in main_group:
        for ref_record in ref_group:
            if (
                main_record.event_class == ref_record.event_class
                and main_record.cluster_role == ref_record.cluster_role
            ):
                fields.update(main_record.fields.keys() ^ ref_record.fields.keys())
    return sorted(fields)


def _find_value_mismatches(
    main_group: Sequence[NormalizedEventRecord],
    ref_group: Sequence[NormalizedEventRecord],
) -> List[str]:
    fields: set[str] = set()
    for main_record in main_group:
        for ref_record in ref_group:
            if _event_identity_compatible(main_record, ref_record):
                fields.update(_event_value_mismatches(main_record, ref_record))
    return sorted(fields)


def _is_role_shift(
    main_group: Sequence[NormalizedEventRecord],
    ref_group: Sequence[NormalizedEventRecord],
) -> bool:
    graph = [
        [
            ref_index
            for ref_index, ref_record in enumerate(ref_group)
            if _event_records_compatible_ignoring_role(main_record, ref_record)
        ]
        for main_record in main_group
    ]
    return _has_perfect_event_matching(graph, len(ref_group))


def _event_records_compatible_ignoring_role(
    main_record: NormalizedEventRecord, ref_record: NormalizedEventRecord
) -> bool:
    if main_record.event_class != ref_record.event_class:
        return False
    excluded_fields = (
        {"target_cluster"}
        if main_record.cluster_role != ref_record.cluster_role
        else set()
    )
    main_fields = {
        key: value
        for key, value in main_record.fields.items()
        if key not in excluded_fields
    }
    ref_fields = {
        key: value
        for key, value in ref_record.fields.items()
        if key not in excluded_fields
    }
    if main_fields.keys() != ref_fields.keys():
        return False
    for field_name, main_value in main_fields.items():
        ref_value = ref_fields[field_name]
        if field_name in _EVENT_FLOAT_MS_FIELDS:
            if abs(float(main_value) - float(ref_value)) > 1e-9:
                return False
        elif field_name in _EVENT_FLOAT_SECOND_FIELDS:
            if abs(float(main_value) - float(ref_value)) > 1e-12:
                return False
        elif main_value != ref_value:
            return False
    return True


def load_request_metrics(output_dir: str) -> Dict[int, Dict[str, object]]:
    """Load request_metrics.csv into a dict keyed by Request Id."""
    metrics_path = _find_request_metrics_csv(output_dir)
    if not metrics_path:
        raise ParityInputError(f"No request_metrics.csv found under {output_dir}")
    result: Dict[int, Dict[str, object]] = {}
    with open(metrics_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ParityInputError(f"request_metrics.csv has no header: {metrics_path}")
        duplicate_headers = sorted(
            {name for name in fieldnames if fieldnames.count(name) > 1}
        )
        if duplicate_headers:
            raise ParityInputError(
                "request_metrics.csv contains duplicate header names: "
                f"{duplicate_headers}"
            )
        if "Request Id" not in fieldnames:
            raise ParityInputError(
                f"request_metrics.csv is missing required 'Request Id' header: {metrics_path}"
            )

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ParityInputError(
                    "request_metrics.csv contains extra columns at "
                    f"line {line_number}: {row[None]!r}"
                )
            raw_request_id = row.get("Request Id")
            try:
                req_id = int(raw_request_id)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ParityInputError(
                    "request_metrics.csv contains an invalid Request Id at "
                    f"line {line_number}: {raw_request_id!r}"
                ) from exc
            if req_id in result:
                raise ParityInputError(
                    f"request_metrics.csv contains duplicate Request Id {req_id}"
                )
            result[req_id] = {k: _parse_value(v) for k, v in row.items()}
    if not result:
        raise ParityInputError(f"request_metrics.csv contains no request rows: {metrics_path}")
    return result


def load_main_request_metrics(output_dir: str) -> Dict[int, Dict[str, object]]:
    """Load main-branch metrics and derive both approved TTFT definitions."""
    metrics = load_request_metrics(output_dir)
    ground_truth_path = _find_unique_output_file(
        output_dir, "metrics_ground_truth.jsonl"
    )
    if ground_truth_path is None:
        raise ParityInputError(
            f"No metrics_ground_truth.jsonl found under {output_dir}"
        )
    timestamps = _load_completion_timestamps(ground_truth_path)
    if set(metrics) != set(timestamps):
        raise ParityInputError(
            "request_metrics.csv and metrics_ground_truth.jsonl request sets differ: "
            f"metrics={sorted(metrics)}, ground_truth={sorted(timestamps)}"
        )

    ttft_tolerance = ToleranceSpec(abs_tol=1e-6, rel_tol=1e-6)
    for request_id, row in metrics.items():
        arrived_at, prefill_completed_at, first_decode_token_completed_at = timestamps[
            request_id
        ]
        canonical_ttft_ms = (prefill_completed_at - arrived_at) * 1000.0
        cross_branch_ttft_ms = (
            first_decode_token_completed_at - arrived_at
        ) * 1000.0
        raw_ttft_ms = _require_finite_numeric(
            row.get("ttft"), f"request {request_id} field 'ttft'"
        )
        canonical_comparison = _compare_numeric_values(
            raw_ttft_ms, canonical_ttft_ms, ttft_tolerance
        )
        if canonical_comparison[0] not in {
            ComparisonResult.EXACT_MATCH,
            ComparisonResult.WITHIN_TOLERANCE,
        }:
            raise ParityInputError(
                "main request_metrics.csv canonical ttft does not match "
                "metrics_ground_truth.jsonl: "
                f"request_id={request_id}, csv_ttft_ms={raw_ttft_ms}, "
                f"derived_ttft_ms={canonical_ttft_ms}"
            )
        row[CANONICAL_MAIN_TTFT_FIELD] = canonical_ttft_ms
        row[CROSS_BRANCH_TTFT_FIELD] = cross_branch_ttft_ms
        row["canonical_main_ttft_provenance"] = CANONICAL_MAIN_TTFT_PROVENANCE
        row["cross_branch_ttft_provenance"] = (
            MAIN_CROSS_BRANCH_TTFT_PROVENANCE
        )
    return metrics


def load_reference_request_metrics(output_dir: str) -> Dict[int, Dict[str, object]]:
    """Load Reference metrics and derive TTFT from direct lifecycle evidence."""
    metrics = load_request_metrics(output_dir)
    lifecycle_path = _find_unique_output_file(
        output_dir,
        REFERENCE_FIRST_REAL_DECODE_LIFECYCLE_FILENAME,
    )
    if lifecycle_path is None:
        raise ParityInputError(
            f"No {REFERENCE_FIRST_REAL_DECODE_LIFECYCLE_FILENAME} found under "
            f"{output_dir}"
        )
    lifecycle = _load_reference_first_real_decode_lifecycle(lifecycle_path)
    if set(metrics) != set(lifecycle):
        raise ParityInputError(
            "request_metrics.csv and Reference lifecycle request sets differ: "
            f"metrics={sorted(metrics)}, lifecycle={sorted(lifecycle)}"
        )
    for request_id, row in metrics.items():
        raw_ttft_ms = _require_finite_numeric(
            row.get("ttft"), f"reference request {request_id} field 'ttft'"
        )
        del raw_ttft_ms
        arrived_at_s, execution_completed_at_s = lifecycle[request_id]
        derived_ttft_ms = (
            execution_completed_at_s - arrived_at_s
        ) * 1000.0
        if not math.isfinite(derived_ttft_ms):
            raise ParityInputError(
                "Reference lifecycle derived first-real-decode TTFT must be "
                f"finite: request_id={request_id}, value={derived_ttft_ms!r}"
            )
        row[CROSS_BRANCH_TTFT_FIELD] = derived_ttft_ms
        row["cross_branch_ttft_provenance"] = REFERENCE_TTFT_PROVENANCE
    return metrics


def _load_reference_first_real_decode_lifecycle(
    path: str,
) -> Dict[int, tuple[float, float]]:
    try:
        raw_bytes = Path(path).read_bytes()
        text = raw_bytes.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ParityInputError(
            f"Invalid {REFERENCE_FIRST_REAL_DECODE_LIFECYCLE_FILENAME}: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ParityInputError(
            f"{REFERENCE_FIRST_REAL_DECODE_LIFECYCLE_FILENAME} must contain an object"
        )
    _require_exact_keys(
        payload,
        _REFERENCE_LIFECYCLE_TOP_LEVEL_KEYS,
        "Reference lifecycle top-level object",
    )
    if payload["schema_version"] != REFERENCE_FIRST_REAL_DECODE_SCHEMA_VERSION:
        raise ParityInputError(
            "Reference lifecycle schema_version mismatch: "
            f"{payload['schema_version']!r}"
        )
    _validate_reference_lifecycle_producer(payload["producer"])

    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ParityInputError(
            f"{REFERENCE_FIRST_REAL_DECODE_LIFECYCLE_FILENAME} must contain requests"
        )

    result: Dict[int, tuple[float, float]] = {}
    for record in requests:
        if not isinstance(record, dict):
            raise ParityInputError("Reference lifecycle request must be an object")
        _require_exact_keys(
            record,
            _REFERENCE_LIFECYCLE_REQUEST_KEYS,
            "Reference lifecycle request",
        )
        request_id = record.get("request_id")
        if (
            isinstance(request_id, bool)
            or not isinstance(request_id, int)
            or request_id < 0
        ):
            raise ParityInputError(
                "Reference lifecycle request_id must be a non-negative integer: "
                f"{request_id!r}"
            )
        if request_id in result:
            raise ParityInputError(
                f"Reference lifecycle contains duplicate request_id {request_id}"
            )
        arrived_at_s = _require_json_number(
            record.get("arrived_at_s"),
            f"Reference lifecycle request {request_id} arrived_at_s",
        )
        execution_completed_at_s = _require_json_number(
            record.get("raw_decode_execution_completed_at_s"),
            f"Reference lifecycle request {request_id} "
            "raw_decode_execution_completed_at_s",
        )
        prefill_completed_at_s = _require_json_number(
            record.get("prefill_completed_at_s"),
            f"Reference lifecycle request {request_id} prefill_completed_at_s",
        )
        resolved_global_end_time_s = _require_json_number(
            record.get("resolved_global_end_time_s"),
            f"Reference lifecycle request {request_id} resolved_global_end_time_s",
        )
        for field_name, value in (
            ("arrived_at_s", arrived_at_s),
            ("prefill_completed_at_s", prefill_completed_at_s),
            ("raw_decode_execution_completed_at_s", execution_completed_at_s),
            ("resolved_global_end_time_s", resolved_global_end_time_s),
        ):
            if value < 0:
                raise ParityInputError(
                    f"Reference lifecycle request {request_id} {field_name} "
                    "must be non-negative"
                )
        if not (
            arrived_at_s
            <= prefill_completed_at_s
            <= execution_completed_at_s
            <= resolved_global_end_time_s
        ):
            raise ParityInputError(
                "Reference lifecycle timestamp order must satisfy "
                "arrived_at_s <= prefill_completed_at_s <= "
                "raw_decode_execution_completed_at_s <= "
                f"resolved_global_end_time_s for request {request_id}"
            )
        if record["cluster_type"] != "DECODE_ATTN":
            raise ParityInputError(
                f"Reference lifecycle request {request_id} cluster_type must be "
                "DECODE_ATTN"
            )
        _require_exact_json_integer(
            record["processed_decode_tokens_before"],
            0,
            "Reference lifecycle processed_decode_tokens_before",
        )
        _require_exact_json_integer(
            record["processed_decode_tokens_after"],
            1,
            "Reference lifecycle processed_decode_tokens_after",
        )
        result[request_id] = (arrived_at_s, execution_completed_at_s)

    request_count = payload["request_count"]
    if (
        isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count != len(result)
    ):
        raise ParityInputError(
            "Reference lifecycle request_count does not match requests: "
            f"declared={request_count!r}, actual={len(result)}"
        )
    expected_id_hash = hashlib.sha256(
        json.dumps(
            sorted(result),
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if payload["request_ids_sha256"] != expected_id_hash:
        raise ParityInputError(
            "Reference lifecycle request_ids_sha256 mismatch: "
            f"expected={expected_id_hash}, "
            f"actual={payload['request_ids_sha256']!r}"
        )
    canonical_bytes = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if raw_bytes != canonical_bytes:
        raise ParityInputError(
            f"{REFERENCE_FIRST_REAL_DECODE_LIFECYCLE_FILENAME} must use "
            "canonical JSON bytes with exactly one final newline"
        )
    return result


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ParityInputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ParityInputError(
            f"{context} keys differ: missing={missing}, unexpected={unexpected}"
        )


def _validate_reference_lifecycle_producer(value: object) -> None:
    if not isinstance(value, dict):
        raise ParityInputError("Reference lifecycle producer must be an object")
    _require_exact_keys(
        value,
        _REFERENCE_LIFECYCLE_PRODUCER_KEYS,
        "Reference lifecycle producer",
    )
    expected_values = {
        "branch_kind": "reference",
        "reference_repo_root": str(REFERENCE_REPO_ROOT),
        "reference_git_head": REFERENCE_GIT_HEAD,
        "request_source_sha256": REFERENCE_REQUEST_SOURCE_SHA256,
        "cluster_scheduler_source_sha256": (
            REFERENCE_CLUSTER_SCHEDULER_SOURCE_SHA256
        ),
        "global_batch_end_event_source_sha256": (
            REFERENCE_GLOBAL_BATCH_END_EVENT_SOURCE_SHA256
        ),
        "candidate_hook": (
            "BaseClusterScheduler."
            "resolve_decode_attn_boundary_first_mixed_global_end_time"
        ),
        "transition_hook": "GlobalBatchEndEvent.handle_event",
        "transition_contract": "num_processed_decode_tokens:0->1",
        "timestamp_contract": "resolver_input_time_before_observation_delay",
        "observer_source_sha256": _sha256_file(
            Path(__file__).with_name("reference_lifecycle_observer.py")
        ),
        "bootstrap_source_sha256": _sha256_file(
            Path(__file__).with_name("reference_observer_bootstrap.py")
        ),
    }
    for field_name, expected in expected_values.items():
        if value[field_name] != expected:
            raise ParityInputError(
                f"Reference lifecycle producer {field_name} mismatch: "
                f"expected={expected!r}, actual={value[field_name]!r}"
            )
    python_executable = value["python_executable"]
    if (
        not isinstance(python_executable, str)
        or not python_executable
        or not Path(python_executable).is_absolute()
    ):
        raise ParityInputError(
            "Reference lifecycle producer python_executable must be absolute"
        )
    for field_name in (
        "argv_sha256",
        "observer_source_sha256",
        "bootstrap_source_sha256",
    ):
        field_value = value[field_name]
        if (
            not isinstance(field_value, str)
            or re.fullmatch(r"[0-9a-f]{64}", field_value) is None
        ):
            raise ParityInputError(
                f"Reference lifecycle producer {field_name} must be lowercase SHA-256"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ParityInputError(
            f"Cannot read Reference lifecycle control source: {path}"
        ) from exc
    return digest.hexdigest()


def _require_json_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityInputError(f"{context} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ParityInputError(f"{context} must be finite")
    return result


def _require_exact_json_integer(
    value: object,
    expected: int,
    context: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != expected
    ):
        raise ParityInputError(
            f"{context} must be integer {expected}: {value!r}"
        )


def compare_requests(
    main_metrics: Dict[int, Dict[str, object]],
    ref_metrics: Dict[int, Dict[str, object]],
    layer: ParityLayer = ParityLayer.L2_TRAINED,
    expected_request_ids: Optional[Sequence[int]] = None,
) -> List[RequestComparison]:
    """Compare request metrics between main and ref runs."""
    comparisons = []
    expected_ids = set(expected_request_ids) if expected_request_ids is not None else None
    all_ids = set(main_metrics) | set(ref_metrics)
    if expected_ids is not None:
        all_ids |= expected_ids

    for req_id in sorted(all_ids):
        rc = RequestComparison(request_id=req_id)

        if expected_ids is not None and req_id not in expected_ids:
            rc.passed = False
            rc.first_divergence_field = "UNEXPECTED_REQUEST_ID"
            comparisons.append(rc)
            continue

        if req_id not in main_metrics and req_id not in ref_metrics:
            rc.passed = False
            rc.first_divergence_field = "REQUEST_MISSING_IN_BOTH"
            comparisons.append(rc)
            continue
        if req_id not in main_metrics:
            rc.passed = False
            rc.first_divergence_field = "REQUEST_MISSING_IN_MAIN"
            comparisons.append(rc)
            continue
        if req_id not in ref_metrics:
            rc.passed = False
            rc.first_divergence_field = "REQUEST_MISSING_IN_REF"
            comparisons.append(rc)
            continue

        main_row = main_metrics[req_id]
        ref_row = ref_metrics[req_id]
        canonical_ttft = main_row.get(CANONICAL_MAIN_TTFT_FIELD)
        if canonical_ttft is not None:
            try:
                rc.canonical_main_ttft_ms = _require_finite_numeric(
                    canonical_ttft,
                    f"request {req_id} field '{CANONICAL_MAIN_TTFT_FIELD}'",
                )
            except ParityInputError:
                rc.passed = False
                rc.first_divergence_field = CANONICAL_MAIN_TTFT_FIELD
            provenance = main_row.get("canonical_main_ttft_provenance")
            if provenance is not None:
                rc.canonical_main_ttft_provenance = str(provenance)

        for field_name in DISCRETE_FIELDS:
            fc = _compare_field_exact(field_name, main_row, ref_row)
            rc.fields.append(fc)
            if (
                fc.result in _HARD_FAILURE_RESULTS
                and rc.first_divergence_field is None
            ):
                rc.first_divergence_field = field_name
                rc.passed = False

        if layer is ParityLayer.L2_TRAINED:
            for field_name, tol in METRIC_TOLERANCES.items():
                fc = _compare_field_numeric(field_name, main_row, ref_row, tol)
                rc.fields.append(fc)
                if field_name == CROSS_BRANCH_TTFT_FIELD:
                    if isinstance(fc.main_value, (int, float)) and not isinstance(
                        fc.main_value, bool
                    ):
                        rc.main_cross_branch_ttft_ms = float(fc.main_value)
                    if isinstance(fc.ref_value, (int, float)) and not isinstance(
                        fc.ref_value, bool
                    ):
                        rc.reference_cross_branch_ttft_ms = float(fc.ref_value)
                    rc.cross_branch_ttft_abs_delta_ms = fc.abs_delta
                    rc.cross_branch_ttft_rel_delta = fc.rel_delta
                    rc.cross_branch_ttft_passed = (
                        fc.result not in _HARD_FAILURE_RESULTS
                    )
                    main_provenance = main_row.get(
                        "cross_branch_ttft_provenance"
                    )
                    reference_provenance = ref_row.get(
                        "cross_branch_ttft_provenance"
                    )
                    if main_provenance is not None:
                        rc.main_cross_branch_ttft_provenance = str(
                            main_provenance
                        )
                    if reference_provenance is not None:
                        rc.reference_cross_branch_ttft_provenance = str(
                            reference_provenance
                        )
                if (
                    fc.result in _HARD_FAILURE_RESULTS
                    and rc.first_divergence_field is None
                ):
                    rc.first_divergence_field = field_name
                    rc.passed = False

        comparisons.append(rc)

    return comparisons


def generate_report(
    case_config: ParityCaseConfig,
    main_output_dir: str,
    ref_output_dir: str,
) -> ParityReport:
    """Run a full parity comparison and generate a report."""
    event_comparison = compare_event_logs(main_output_dir, ref_output_dir)
    if case_config.layer is ParityLayer.L2_TRAINED:
        main_metrics = load_main_request_metrics(main_output_dir)
        ref_metrics = load_reference_request_metrics(ref_output_dir)
    else:
        main_metrics = load_request_metrics(main_output_dir)
        ref_metrics = load_request_metrics(ref_output_dir)

    comparisons = compare_requests(
        main_metrics,
        ref_metrics,
        case_config.layer,
        expected_request_ids=range(case_config.num_requests),
    )

    report = ParityReport(
        case_config=case_config,
        main_output_dir=main_output_dir,
        ref_output_dir=ref_output_dir,
        request_comparisons=comparisons,
        first_divergence_event_index=(
            event_comparison.first_divergence_event_index
        ),
        event_comparison=event_comparison,
    )

    for rc in comparisons:
        report.total_fields_compared += len(rc.fields)
        hard_field_failures = [
            field for field in rc.fields if field.result in _HARD_FAILURE_RESULTS
        ]
        report.total_mismatches += len(hard_field_failures)
        if not rc.passed and not hard_field_failures:
            report.total_mismatches += 1
        if not rc.passed:
            report.overall_pass = False

    report.total_mismatches += event_comparison.total_mismatches
    if not event_comparison.passed:
        report.overall_pass = False

    return report


def report_to_markdown(report: ParityReport) -> str:
    """Render a parity report as markdown."""
    lines = [
        f"# Parity Report: {report.case_config.case_id}",
        f"",
        f"**Model**: {report.case_config.model}",
        f"**Mode**: {report.case_config.mode}",
        f"**Scale**: {report.case_config.scale_gpu} GPU",
        f"**Layer**: {report.case_config.layer.value}",
        f"**Result**: {'PASS' if report.overall_pass else 'FAIL'}",
        f"**Main output**: `{report.main_output_dir}`",
        f"**Reference output**: `{report.ref_output_dir}`",
        f"",
        f"## Summary",
        f"- Fields compared: {report.total_fields_compared}",
        f"- Mismatches: {report.total_mismatches}",
        f"- Requests: {len(report.request_comparisons)}",
        f"",
    ]

    if report.event_comparison is not None:
        event_comparison = report.event_comparison
        first_divergence = event_comparison.first_divergence_event_index
        lines.extend(
            [
                "## Event Comparison",
                "",
                f"- Result: {'PASS' if event_comparison.passed else 'FAIL'}",
                f"- Total events (main): {event_comparison.total_events_main}",
                f"- Total events (reference): {event_comparison.total_events_ref}",
                f"- Event time-group mismatches: {event_comparison.total_mismatches}",
                "- First divergence event index: "
                + (str(first_divergence) if first_divergence is not None else "not present"),
                "",
                "| Role | Main events | Reference events | Result |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for role in _EVENT_ROLES:
            summary = event_comparison.per_role[role]
            lines.append(
                f"| {role} | {summary.main_count} | {summary.ref_count} | "
                f"{'PASS' if summary.passed else 'FAIL'} |"
            )
        lines.append("")

        for mismatch_index, mismatch in enumerate(
            event_comparison.mismatches, start=1
        ):
            lines.extend(
                [
                    f"### Event divergence {mismatch_index}",
                    "",
                    f"- Event time: {mismatch.event_time}",
                    f"- Category: `{mismatch.category}`",
                    "- Mismatched fields: "
                    + (
                        ", ".join(mismatch.mismatched_fields)
                        if mismatch.mismatched_fields
                        else "not present"
                    ),
                    "",
                    "**Main normalized record**",
                    "",
                    "```json",
                    _event_records_to_json(mismatch.main_records),
                    "```",
                    "",
                    "**Reference normalized record**",
                    "",
                    "```json",
                    _event_records_to_json(mismatch.ref_records),
                    "```",
                    "",
                ]
            )

    canonical_ttft_rows = [
        comparison
        for comparison in report.request_comparisons
        if comparison.canonical_main_ttft_ms is not None
    ]
    if canonical_ttft_rows:
        lines.extend(
            [
                "## Main Product TTFT",
                "",
                "| Request | canonical_main_ttft_ms | Provenance |",
                "| ---: | ---: | --- |",
            ]
        )
        for comparison in canonical_ttft_rows:
            lines.append(
                f"| {comparison.request_id} | "
                f"{comparison.canonical_main_ttft_ms} | "
                f"{comparison.canonical_main_ttft_provenance or 'not present'} |"
            )
        lines.append("")

    cross_branch_ttft_rows = [
        comparison
        for comparison in report.request_comparisons
        if comparison.main_cross_branch_ttft_ms is not None
        and comparison.reference_cross_branch_ttft_ms is not None
    ]
    if cross_branch_ttft_rows:
        lines.extend(
            [
                "## Cross-branch First Real Decode TTFT",
                "",
                "| Request | Main actual (ms) | Reference actual (ms) | "
                "Abs delta (ms) | Relative delta | Result | "
                "Main provenance | Reference provenance |",
                "| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for comparison in cross_branch_ttft_rows:
            lines.append(
                f"| {comparison.request_id} | "
                f"{comparison.main_cross_branch_ttft_ms} | "
                f"{comparison.reference_cross_branch_ttft_ms} | "
                f"{comparison.cross_branch_ttft_abs_delta_ms} | "
                f"{comparison.cross_branch_ttft_rel_delta} | "
                f"{'PASS' if comparison.cross_branch_ttft_passed else 'FAIL'} | "
                f"{comparison.main_cross_branch_ttft_provenance or 'not present'} | "
                f"{comparison.reference_cross_branch_ttft_provenance or 'not present'} |"
            )
        lines.append("")

    if not report.overall_pass:
        lines.append("## Failures")
        lines.append("")
        for rc in report.request_comparisons:
            if not rc.passed:
                lines.append(f"### Request {rc.request_id}")
                lines.append(f"First divergence: `{rc.first_divergence_field}`")
                lines.append("")
                for fc in rc.fields:
                    if fc.result in _HARD_FAILURE_RESULTS:
                        lines.append(
                            f"- **{fc.field_name}**: main={fc.main_value}, "
                            f"ref={fc.ref_value}, result={fc.result.value}, "
                            f"abs_delta={fc.abs_delta}, rel_delta={fc.rel_delta}"
                        )
                lines.append("")

    if report.notes:
        lines.append("## Notes")
        for note in report.notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


def _event_records_to_json(records: Sequence[NormalizedEventRecord]) -> str:
    """Render normalized event evidence in a stable, readable form."""
    payload = [
        {
            "event_class": record.event_class,
            "cluster_role": record.cluster_role,
            "event_time": record.event_time,
            "fields": dict(record.fields),
            "source_line": record.source_line,
        }
        for record in records
    ]
    return json.dumps(payload, indent=2, sort_keys=True)


def _find_request_metrics_csv(output_dir: str) -> Optional[str]:
    """Resolve an explicit or unique request_metrics.csv path."""
    path = _find_unique_output_file(output_dir, "request_metrics.csv")
    return str(path) if path is not None else None


def _find_unique_output_file(output_dir: str, filename: str) -> Optional[Path]:
    """Resolve a root-level output file or require one unique recursive match."""
    root = Path(output_dir)
    if root.is_file():
        if root.name != filename:
            raise ParityInputError(
                f"Expected {filename}, received explicit file {root}"
            )
        return root
    explicit_path = root / filename
    if explicit_path.is_file():
        return explicit_path
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        return None
    if len(matches) > 1:
        raise ParityInputError(
            f"Ambiguous {filename} under {output_dir}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _parse_value(s: str) -> object:
    """Parse a CSV value to int, float, or string."""
    if not s or s == "":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _compare_field_exact(
    field_name: str,
    main_row: Dict[str, object],
    ref_row: Dict[str, object],
) -> FieldComparison:
    """Compare a discrete field for exact equality."""
    main_val = main_row.get(field_name)
    ref_val = ref_row.get(field_name)
    if main_val is None or ref_val is None:
        result = ComparisonResult.MISSING_FIELD
    else:
        try:
            main_number = _require_integral_numeric(
                main_val, f"field '{field_name}' on main"
            )
            ref_number = _require_integral_numeric(
                ref_val, f"field '{field_name}' on reference"
            )
        except ParityInputError:
            result = ComparisonResult.INVALID_VALUE
        else:
            result = (
                ComparisonResult.EXACT_MATCH
                if main_number == ref_number
                else ComparisonResult.MISMATCH
            )
    return FieldComparison(
        field_name=field_name,
        main_value=main_val,
        ref_value=ref_val,
        result=result,
    )


def _compare_field_numeric(
    field_name: str,
    main_row: Dict[str, object],
    ref_row: Dict[str, object],
    tol: ToleranceSpec,
) -> FieldComparison:
    """Compare a numeric field with tolerance."""
    main_val = main_row.get(field_name)
    ref_val = ref_row.get(field_name)

    if main_val is None or ref_val is None:
        return FieldComparison(
            field_name=field_name, main_value=main_val, ref_value=ref_val,
            result=ComparisonResult.MISSING_FIELD, tolerance_applied=tol,
        )

    try:
        m = _require_finite_numeric(main_val, f"field '{field_name}' on main")
        r = _require_finite_numeric(ref_val, f"field '{field_name}' on reference")
    except ParityInputError:
        return FieldComparison(
            field_name=field_name, main_value=main_val, ref_value=ref_val,
            result=ComparisonResult.INVALID_VALUE, tolerance_applied=tol,
        )

    result, abs_delta, rel_delta = _compare_numeric_values(m, r, tol)

    return FieldComparison(
        field_name=field_name,
        main_value=m,
        ref_value=r,
        result=result,
        abs_delta=abs_delta,
        rel_delta=rel_delta,
        tolerance_applied=tol,
    )


def _require_finite_numeric(value: object, context: str) -> float:
    """Return a finite float or raise an explicit parity input error."""
    if isinstance(value, bool):
        raise ParityInputError(f"{context} must be numeric, received boolean")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ParityInputError(f"{context} must be numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ParityInputError(f"{context} must be finite: {value!r}")
    return number


def _require_integral_numeric(value: object, context: str) -> int:
    """Return an integer-valued number or raise an explicit input error."""
    number = _require_finite_numeric(value, context)
    if not number.is_integer():
        raise ParityInputError(f"{context} must be integer-valued: {value!r}")
    return int(number)


def _compare_numeric_values(
    main_value: float,
    ref_value: float,
    tolerance: ToleranceSpec,
) -> tuple[ComparisonResult, float, float]:
    """Compare two already-validated finite values."""
    abs_delta = abs(main_value - ref_value)
    rel_delta = (
        abs_delta / max(abs(ref_value), 1e-15)
        if ref_value != 0
        else (0.0 if main_value == 0 else float("inf"))
    )
    if abs_delta == 0:
        result = ComparisonResult.EXACT_MATCH
    elif abs_delta <= tolerance.abs_tol:
        result = ComparisonResult.WITHIN_TOLERANCE
    elif tolerance.rel_tol > 0 and rel_delta <= tolerance.rel_tol:
        result = ComparisonResult.WITHIN_TOLERANCE
    else:
        result = ComparisonResult.MISMATCH
    return result, abs_delta, rel_delta


def _load_completion_timestamps(
    path: Path,
) -> Dict[int, tuple[float, float, float]]:
    """Load one valid request-completion timestamp record per request."""
    records: Dict[int, tuple[float, float, float]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise ParityInputError(
                    f"Malformed JSONL blank line at {path}:{line_number}"
                )
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ParityInputError(
                    f"Malformed JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, Mapping):
                raise ParityInputError(
                    f"Malformed JSON record at {path}:{line_number}: expected object"
                )
            if record.get("event_type") != "request_completion":
                continue
            request_id = _require_integral_numeric(
                record.get("request_id"),
                f"metrics ground truth request_id at line {line_number}",
            )
            if request_id in records:
                raise ParityInputError(
                    f"Duplicate request completion record for request {request_id}"
                )
            arrived_at = _require_finite_numeric(
                record.get("arrived_at"),
                f"request {request_id} arrived_at",
            )
            prefill_completed_at = _require_finite_numeric(
                record.get("prefill_completed_at"),
                f"request {request_id} prefill_completed_at",
            )
            first_decode_token_completed_at = _require_finite_numeric(
                record.get("first_decode_token_completed_at"),
                f"request {request_id} first_decode_token_completed_at",
            )
            if not (
                arrived_at <= prefill_completed_at <= first_decode_token_completed_at
            ):
                raise ParityInputError(
                    "Invalid TTFT timestamp order: "
                    f"request_id={request_id}, arrived_at={arrived_at}, "
                    f"prefill_completed_at={prefill_completed_at}, "
                    "first_decode_token_completed_at="
                    f"{first_decode_token_completed_at}"
                )
            records[request_id] = (
                arrived_at,
                prefill_completed_at,
                first_decode_token_completed_at,
            )
    if not records:
        raise ParityInputError(
            f"metrics_ground_truth.jsonl contains no request completion records: {path}"
        )
    return records

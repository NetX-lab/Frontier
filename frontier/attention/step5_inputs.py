"""Step workload kinds and the source-bound MRv2 dummy input descriptor."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real


@dataclass(frozen=True)
class Step5WorkloadKind:
    code: int
    physical_scope: str
    has_attention_metadata: bool


STEP5_WORKLOAD_KINDS = {
    "attention_workload": Step5WorkloadKind(0, "native_attention_layer_without_tp_allreduce", True),
    "mrv2_dp_dummy": Step5WorkloadKind(1, "mrv2_dp_dummy_attention_layer_without_tp_allreduce", True),
    "isolated_no_metadata": Step5WorkloadKind(2, "isolated_no_metadata_attention_layer_without_tp_allreduce", False),
}
MRV2_DUMMY_ROW_IDENTITY = {
    "native_runner": "MRv2", "dummy_is_padding": True, "dummy_has_prefill": False,
    "dummy_owned_requests": 0, "dummy_local_num_tokens": 1,
    "slot_mapping_kind": "PAD_SLOT_ID",
    "cache_policy": "live_pages_unchanged_scratch_updates_allowed",
    "valid_token_semantics": "MRv2_InputBatch_physical_rows_not_owned_serving_tokens",
}


def step5_workload_kind(label):
    try:
        return STEP5_WORKLOAD_KINDS[label]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported or missing Step native_workload={label!r}") from exc


def step5_input_kind(*, has_attention_metadata, dummy_descriptor=None):
    return ("mrv2_dp_dummy" if dummy_descriptor is not None else
            "attention_workload" if has_attention_metadata else "isolated_no_metadata")


def step5_workload_label(code):
    for label, kind in STEP5_WORKLOAD_KINDS.items():
        if kind.code == code:
            return label
    raise ValueError(f"Unsupported Step native_workload_code={code!r}")


@dataclass(frozen=True)
class MRv2DummyDescriptor:
    """Explicit post-DP-sync descriptor; never inferred from a local dummy token."""

    num_tokens: int
    num_reqs: int | None
    graph_mode: str
    max_query_len: int | None
    evidence: str
    uniform_token_count: int | None = None

    def __post_init__(self):
        if type(self.num_tokens) is not int or self.num_tokens <= 0:
            raise ValueError("MRv2 dummy descriptor requires positive synchronized num_tokens")
        if self.num_reqs is not None and (
                type(self.num_reqs) is not int or not 0 < self.num_reqs <= self.num_tokens):
            raise ValueError("MRv2 dummy descriptor num_reqs must fit num_tokens")
        if self.graph_mode not in ("none", "piecewise", "full"):
            raise ValueError("MRv2 dummy graph mode must come from the synchronized descriptor")
        if self.max_query_len is not None and (
                type(self.max_query_len) is not int or self.max_query_len <= 0):
            raise ValueError("MRv2 descriptor max_query_len must be positive or None")
        if self.uniform_token_count is not None and (
                type(self.uniform_token_count) is not int or self.uniform_token_count <= 0):
            raise ValueError("MRv2 uniform_token_count must be positive or None")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("MRv2 dummy descriptor requires an explicit source evidence reference")
        if self.max_query_len is not None and max(self.query_lens) > self.max_query_len:
            raise ValueError("MRv2 dummy token split exceeds the descriptor max_query_len")
        if self.graph_mode == "full" and (
                self.max_query_len not in (None, 1) or max(self.query_lens) != 1
                or self.max_query_len is None and self.uniform_token_count != 1
                or self.uniform_token_count not in (None, 1)):
            raise ValueError("This no-MTP producer supports FULL dummy descriptors only for single-token decode")

    @property
    def query_lens(self):
        # execute_dummy_batch schedules one local request without MTP. MRv2
        # uses that count only when PIECEWISE supplies no request dimension.
        requests = self.num_reqs or 1
        base, extra = divmod(self.num_tokens, requests)
        return tuple(base + (index >= requests - extra) for index in range(requests))


def _missing(value):
    return value is None or isinstance(value, Real) and math.isnan(value)


def validate_step5_input_row(row, workload):
    """Validate producer kind and provenance without admitting native evidence."""
    label = row.get("native_workload")
    kind = step5_workload_kind(label)
    if row.get("physical_scope") != kind.physical_scope:
        raise ValueError(f"Step physical_scope conflicts with native_workload={label!r}")
    if workload.has_attention_metadata != kind.has_attention_metadata:
        raise ValueError(f"Step has_attention_metadata conflicts with native_workload={label!r}")
    if label != "mrv2_dp_dummy":
        if any(not _missing(row.get(name)) for name in (*MRV2_DUMMY_ROW_IDENTITY, "dummy_descriptor")):
            raise ValueError(f"Step native_workload={label!r} cannot carry MRv2 dummy provenance")
        return kind
    for name, expected in MRV2_DUMMY_ROW_IDENTITY.items():
        value = row.get(name)
        if isinstance(expected, bool):
            # Mixed-kind CSV boolean columns contain blanks and may load as text.
            matches = (isinstance(value, (bool, str)) and value in (expected, str(expected)))
        elif isinstance(expected, int):
            matches = isinstance(value, Real) and not isinstance(value, bool) and value == expected
        else:
            matches = isinstance(value, str) and value == expected
        if not matches:
            raise ValueError(f"Step MRv2 dummy provenance requires {name}={expected!r}")
    try:
        descriptor = row.get("dummy_descriptor")
        values = json.loads(descriptor) if isinstance(descriptor, str) else descriptor
        descriptor = MRv2DummyDescriptor(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Step MRv2 dummy_descriptor: {exc}") from exc
    phase = "decode" if max(descriptor.query_lens) == 1 else "prefill"
    if (workload.query_lengths != descriptor.query_lens
            or workload.sequence_lengths != descriptor.query_lens
            or workload.physical_query_rows != descriptor.num_tokens
            or workload.phase != phase or workload.graph_mode != descriptor.graph_mode):
        raise ValueError("Step MRv2 dummy row must preserve descriptor counts, split, phase, mode and zero history")
    return kind

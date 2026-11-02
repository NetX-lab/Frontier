"""Raw vLLM CPU overhead record normalization."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from frontier.profiling.cpu_overhead.schema import (
    DEFAULT_NUM_DECODE_TOKENS_AMPLIFICATION_FACTOR,
    DEFAULT_NUM_PREFILL_TOKENS,
    DEFAULT_SCHEDULING_MODE,
    VALID_SCHEDULING_MODES,
)
from frontier.types import MeasurementType

_COMMON_REQUIRED_FIELDS = (
    "model_name",
    "batch_size",
    "tensor_parallel_degree",
    "profiling_precision",
    "schedule_ms",
    "sampler_ms",
    "step_wall_time_ms",
)

_SARATHI_STYLE_REQUIRED_FIELDS = (
    *_COMMON_REQUIRED_FIELDS,
    "prepare_inputs_ms",
    "process_model_outputs_ms",
)

_VLLM_V1_NATIVE_REQUIRED_FIELDS = (
    *_COMMON_REQUIRED_FIELDS,
    "preprocess_ms",
    "postprocess_ms",
    "bookkeep_ms",
)

_RAY_COMM_TIME_NUMERIC_EPSILON_MS = 1e-9


def _validate_positive_number(raw: Mapping[str, Any], field_name: str) -> float:
    value = float(raw[field_name])
    if not np.isfinite(value):
        raise ValueError(f"Raw vLLM field '{field_name}' must be finite, got {value}.")
    if value < 0:
        raise ValueError(f"Raw vLLM field '{field_name}' must be >= 0, got {value}.")
    return value


def _normalize_common_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    model_name = str(raw["model_name"]).strip()
    if not model_name:
        raise ValueError("Raw vLLM field 'model_name' must be non-empty.")

    batch_size = int(raw["batch_size"])
    tensor_parallel_degree = int(raw["tensor_parallel_degree"])
    if batch_size <= 0:
        raise ValueError(f"Raw vLLM field 'batch_size' must be > 0, got {batch_size}.")
    if tensor_parallel_degree <= 0:
        raise ValueError(
            "Raw vLLM field 'tensor_parallel_degree' must be > 0, "
            f"got {tensor_parallel_degree}."
        )

    precision = str(raw["profiling_precision"]).strip().upper()
    if not precision:
        raise ValueError("Raw vLLM field 'profiling_precision' must be non-empty.")

    num_prefill_tokens = int(raw.get("num_prefill_tokens", DEFAULT_NUM_PREFILL_TOKENS))
    if num_prefill_tokens < 0:
        raise ValueError(
            "Raw vLLM field 'num_prefill_tokens' must be >= 0, "
            f"got {num_prefill_tokens}."
        )

    default_decode_tokens = (
        batch_size * DEFAULT_NUM_DECODE_TOKENS_AMPLIFICATION_FACTOR
    )
    num_decode_tokens = int(raw.get("num_decode_tokens", default_decode_tokens))
    if num_decode_tokens < 0:
        raise ValueError(
            "Raw vLLM field 'num_decode_tokens' must be >= 0, "
            f"got {num_decode_tokens}."
        )
    if num_prefill_tokens + num_decode_tokens <= 0:
        raise ValueError(
            "Raw vLLM fields 'num_prefill_tokens' + 'num_decode_tokens' must be > 0."
        )

    scheduling_mode = str(
        raw.get("scheduling_mode", DEFAULT_SCHEDULING_MODE)
    ).strip().lower()
    if scheduling_mode not in VALID_SCHEDULING_MODES:
        raise ValueError(
            f"Raw vLLM field 'scheduling_mode' must be one of {VALID_SCHEDULING_MODES}, "
            f"got {scheduling_mode!r}."
        )

    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "tensor_parallel_degree": tensor_parallel_degree,
        "profiling_precision": precision,
        "num_prefill_tokens": num_prefill_tokens,
        "num_decode_tokens": num_decode_tokens,
        "scheduling_mode": scheduling_mode,
    }


def _build_normalized_row(
    raw: Mapping[str, Any],
    *,
    prepare_inputs_ms: float,
    process_model_outputs_ms: float,
) -> dict[str, Any]:
    identity = _normalize_common_identity(raw)
    schedule_ms = _validate_positive_number(raw, "schedule_ms")
    sampler_ms = _validate_positive_number(raw, "sampler_ms")
    step_wall_time_ms = _validate_positive_number(raw, "step_wall_time_ms")

    recorded_cpu_time_ms = (
        schedule_ms + prepare_inputs_ms + sampler_ms + process_model_outputs_ms
    )
    ray_comm_time_ms = step_wall_time_ms - recorded_cpu_time_ms
    if ray_comm_time_ms < 0:
        if abs(ray_comm_time_ms) <= _RAY_COMM_TIME_NUMERIC_EPSILON_MS:
            ray_comm_time_ms = 0.0
        else:
            raise ValueError(
                "Derived ray_comm_time_mean is negative. "
                f"step_wall_time_ms={step_wall_time_ms}, "
                f"recorded_cpu_time_ms={recorded_cpu_time_ms}, "
                f"ray_comm_time_mean={ray_comm_time_ms}."
            )

    return {
        "model_name": identity["model_name"],
        "batch_size": identity["batch_size"],
        "tensor_parallel_degree": identity["tensor_parallel_degree"],
        "num_prefill_tokens": identity["num_prefill_tokens"],
        "num_decode_tokens": identity["num_decode_tokens"],
        "scheduling_mode": identity["scheduling_mode"],
        "schedule_mean": schedule_ms,
        "schedule_median": schedule_ms,
        "sampler_e2e_mean": sampler_ms,
        "sampler_e2e_median": sampler_ms,
        "prepare_inputs_e2e_mean": prepare_inputs_ms,
        "prepare_inputs_e2e_median": prepare_inputs_ms,
        "process_model_outputs_mean": process_model_outputs_ms,
        "process_model_outputs_median": process_model_outputs_ms,
        "ray_comm_time_mean": ray_comm_time_ms,
        "profiling_precision": identity["profiling_precision"],
        "measurement_type": MeasurementType.CUDA_EVENT.value,
        "cpu_overhead_source": "vllm_replay",
    }


def _normalize_vllm_sarathi_style_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing_fields = [
        field for field in _SARATHI_STYLE_REQUIRED_FIELDS if field not in raw
    ]
    if missing_fields:
        raise ValueError(
            "Raw vLLM CPU overhead record (Sarathi-style) is missing required fields: "
            f"{missing_fields}."
        )

    prepare_inputs_ms = _validate_positive_number(raw, "prepare_inputs_ms")
    process_model_outputs_ms = _validate_positive_number(raw, "process_model_outputs_ms")
    return _build_normalized_row(
        raw,
        prepare_inputs_ms=prepare_inputs_ms,
        process_model_outputs_ms=process_model_outputs_ms,
    )


def normalize_vllm_v1_native_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Map vLLM v1 native phase names to Frontier CPU overhead schema."""
    missing_fields = [field for field in _VLLM_V1_NATIVE_REQUIRED_FIELDS if field not in raw]
    if missing_fields:
        raise ValueError(
            "Raw vLLM CPU overhead record (vLLM-v1-native) is missing required fields: "
            f"{missing_fields}."
        )

    preprocess_ms = _validate_positive_number(raw, "preprocess_ms")
    postprocess_ms = _validate_positive_number(raw, "postprocess_ms")
    bookkeep_ms = _validate_positive_number(raw, "bookkeep_ms")
    process_model_outputs_ms = postprocess_ms + bookkeep_ms

    return _build_normalized_row(
        raw,
        prepare_inputs_ms=preprocess_ms,
        process_model_outputs_ms=process_model_outputs_ms,
    )


def normalize_vllm_cpu_overhead_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Auto-detect vLLM record format and normalize to Frontier schema."""
    if {"preprocess_ms", "postprocess_ms", "bookkeep_ms"}.issubset(raw.keys()):
        return normalize_vllm_v1_native_record(raw)

    if {"prepare_inputs_ms", "process_model_outputs_ms"}.issubset(raw.keys()):
        return _normalize_vllm_sarathi_style_record(raw)

    raise ValueError(
        "Unrecognized vLLM CPU overhead record format. Expected either "
        "(prepare_inputs_ms + process_model_outputs_ms) or "
        "(preprocess_ms + postprocess_ms + bookkeep_ms)."
    )

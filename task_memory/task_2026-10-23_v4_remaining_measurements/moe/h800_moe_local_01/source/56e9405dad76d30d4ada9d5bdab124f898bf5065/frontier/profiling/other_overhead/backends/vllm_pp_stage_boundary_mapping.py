"""Normalize raw vLLM PP stage-boundary trace records."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

_REQUIRED_STAGE_TRACE_FIELDS = (
    "model_name",
    "timestamp",
    "batch_id",
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "request_ids",
    "pp_rank",
    "pp_world_size",
    "is_first_rank",
    "is_last_rank",
    "activation_bytes_per_rank",
    "recv_start_ts",
    "recv_end_ts",
    "preprocess_start_ts",
    "preprocess_end_ts",
    "forward_start_ts",
    "forward_end_ts",
    "send_start_ts",
    "send_end_ts",
)

_MODEL_CONFIG_DIR = Path(__file__).resolve().parents[4] / "data" / "config" / "models"
_CANONICAL_MODEL_ALIASES = {
    "llama3.1-8b": "llama3.1-8b",
    "meta-llama/llama-3.1-8b": "llama3.1-8b",
    "meta-llama/llama-3.1-8b-instruct": "llama3.1-8b",
    "qwen3-a3b-30b-moe": "qwen3-a3b-30b-moe",
    "qwen/qwen3-a3b-30b-moe": "qwen3-a3b-30b-moe",
}
_MODEL_SIGNATURE_KEYS = (
    "architectures",
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "head_dim",
    "num_experts",
    "num_experts_per_tok",
    "max_position_embeddings",
    "hidden_act",
    "torch_dtype",
    "tie_word_embeddings",
    "quantization_config",
)


def _canonicalize_alias(raw_model_name: str) -> str | None:
    lowered = raw_model_name.strip().lower()
    if not lowered:
        return None
    return _CANONICAL_MODEL_ALIASES.get(lowered)


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalize_signature_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            (str(key), _normalize_signature_value(val))
            for key, val in sorted(value.items())
        )
    return value


def _build_model_signature(config_payload: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        _normalize_signature_value(config_payload.get(field_name))
        for field_name in _MODEL_SIGNATURE_KEYS
    )


@lru_cache(maxsize=1)
def _canonical_model_signatures() -> dict[tuple[Any, ...], str]:
    signatures: dict[tuple[Any, ...], str] = {}
    if not _MODEL_CONFIG_DIR.exists():
        return signatures

    for config_path in sorted(_MODEL_CONFIG_DIR.glob("*.json")):
        with open(config_path, "r", encoding="utf-8") as file:
            config_payload = json.load(file)
        signature = _build_model_signature(config_payload)
        canonical_model_name = config_path.stem
        if signature in signatures and signatures[signature] != canonical_model_name:
            raise ValueError(
                "Ambiguous canonical model signature in other_overhead mapping: "
                f"{config_path} conflicts with {signatures[signature]!r}."
            )
        signatures[signature] = canonical_model_name
    return signatures


def canonicalize_pp_stage_trace_model_name(raw_model_name: str) -> str:
    model_name = str(raw_model_name).strip()
    if not model_name:
        raise ValueError("Raw PP stage trace field 'model_name' must be non-empty.")

    direct_alias = _canonicalize_alias(model_name)
    if direct_alias is not None:
        return direct_alias

    candidate_path = Path(model_name)
    if candidate_path.exists():
        config_path = (
            candidate_path
            if candidate_path.is_file() and candidate_path.name == "config.json"
            else candidate_path / "config.json"
        )
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                config_payload = json.load(file)
            signature = _build_model_signature(config_payload)
            matched_model_name = _canonical_model_signatures().get(signature)
            if matched_model_name is not None:
                return matched_model_name

    basename = candidate_path.name
    if basename.endswith(".json"):
        basename = basename[:-5]
    basename_alias = _canonicalize_alias(basename)
    if basename_alias is not None:
        return basename_alias

    return model_name


def _validate_finite_number(raw: Mapping[str, Any], field_name: str) -> float:
    value = float(raw[field_name])
    if not math.isfinite(value):
        raise ValueError(
            f"Raw PP stage trace field '{field_name}' must be finite, got {value}."
        )
    return value


def _validate_non_negative_number(raw: Mapping[str, Any], field_name: str) -> float:
    value = _validate_finite_number(raw, field_name)
    if value < 0:
        raise ValueError(
            f"Raw PP stage trace field '{field_name}' must be >= 0, got {value}."
        )
    return value


def _validate_optional_timestamp(
    raw: Mapping[str, Any],
    field_name: str,
) -> float | None:
    value = raw[field_name]
    if value is None:
        return None
    return _validate_non_negative_number(raw, field_name)


def _duration_ms(
    start_ts: float | None,
    end_ts: float | None,
    *,
    field_prefix: str,
) -> float:
    if start_ts is None and end_ts is None:
        return 0.0
    if start_ts is None or end_ts is None:
        raise ValueError(
            f"Raw PP stage trace fields '{field_prefix}_start_ts' and "
            f"'{field_prefix}_end_ts' must both be present or both be None."
        )
    if end_ts < start_ts:
        raise ValueError(
            f"Raw PP stage trace field '{field_prefix}_end_ts' must be >= "
            f"'{field_prefix}_start_ts'."
        )
    return (end_ts - start_ts) * 1000.0


def normalize_vllm_pp_stage_trace_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing_fields = [field for field in _REQUIRED_STAGE_TRACE_FIELDS if field not in raw]
    if missing_fields:
        raise ValueError(
            "Raw PP stage trace record is missing required fields: "
            f"{missing_fields}."
        )

    model_name = canonicalize_pp_stage_trace_model_name(str(raw["model_name"]))

    request_ids_raw = raw["request_ids"]
    if not isinstance(request_ids_raw, (list, tuple)) or not request_ids_raw:
        raise ValueError(
            "Raw PP stage trace field 'request_ids' must be a non-empty list or tuple."
        )
    request_ids = tuple(str(req_id) for req_id in request_ids_raw)

    batch_id = int(raw["batch_id"])
    batch_size = int(raw["batch_size"])
    tensor_parallel_degree = int(raw["tensor_parallel_degree"])
    num_prefill_tokens = int(raw["num_prefill_tokens"])
    num_decode_tokens = int(raw["num_decode_tokens"])
    pp_rank = int(raw["pp_rank"])
    pp_world_size = int(raw["pp_world_size"])
    activation_bytes_per_rank = int(raw["activation_bytes_per_rank"])
    is_first_rank = bool(raw["is_first_rank"])
    is_last_rank = bool(raw["is_last_rank"])

    if batch_id < 0:
        raise ValueError(f"Raw PP stage trace field 'batch_id' must be >= 0, got {batch_id}.")
    if batch_size <= 0:
        raise ValueError(
            f"Raw PP stage trace field 'batch_size' must be > 0, got {batch_size}."
        )
    if tensor_parallel_degree <= 0:
        raise ValueError(
            "Raw PP stage trace field 'tensor_parallel_degree' must be > 0, "
            f"got {tensor_parallel_degree}."
        )
    if num_prefill_tokens < 0 or num_decode_tokens < 0:
        raise ValueError(
            "Raw PP stage trace fields 'num_prefill_tokens' and "
            "'num_decode_tokens' must be >= 0."
        )
    if pp_world_size <= 0:
        raise ValueError(
            f"Raw PP stage trace field 'pp_world_size' must be > 0, got {pp_world_size}."
        )
    if pp_rank < 0 or pp_rank >= pp_world_size:
        raise ValueError(
            "Raw PP stage trace field 'pp_rank' must be within "
            f"[0, {pp_world_size}), got {pp_rank}."
        )
    if activation_bytes_per_rank < 0:
        raise ValueError(
            "Raw PP stage trace field 'activation_bytes_per_rank' must be >= 0, "
            f"got {activation_bytes_per_rank}."
        )

    timestamp = _validate_non_negative_number(raw, "timestamp")
    preprocess_start_ts = _validate_non_negative_number(raw, "preprocess_start_ts")
    preprocess_end_ts = _validate_non_negative_number(raw, "preprocess_end_ts")
    forward_start_ts = _validate_non_negative_number(raw, "forward_start_ts")
    forward_end_ts = _validate_non_negative_number(raw, "forward_end_ts")
    recv_start_ts = _validate_optional_timestamp(raw, "recv_start_ts")
    recv_end_ts = _validate_optional_timestamp(raw, "recv_end_ts")
    send_start_ts = _validate_optional_timestamp(raw, "send_start_ts")
    send_end_ts = _validate_optional_timestamp(raw, "send_end_ts")

    if preprocess_end_ts < preprocess_start_ts:
        raise ValueError(
            "Raw PP stage trace field 'preprocess_end_ts' must be >= "
            "'preprocess_start_ts'."
        )
    if forward_start_ts < preprocess_end_ts:
        raise ValueError(
            "Raw PP stage trace field 'forward_start_ts' must be >= "
            "'preprocess_end_ts'."
        )
    if forward_end_ts < forward_start_ts:
        raise ValueError(
            "Raw PP stage trace field 'forward_end_ts' must be >= "
            "'forward_start_ts'."
        )

    recv_duration_ms = _duration_ms(
        recv_start_ts,
        recv_end_ts,
        field_prefix="recv",
    )
    send_duration_ms = _duration_ms(
        send_start_ts,
        send_end_ts,
        field_prefix="send",
    )
    preprocess_duration_ms = _duration_ms(
        preprocess_start_ts,
        preprocess_end_ts,
        field_prefix="preprocess",
    )

    return {
        "model_name": model_name,
        "timestamp": timestamp,
        "batch_id": batch_id,
        "batch_size": batch_size,
        "tensor_parallel_degree": tensor_parallel_degree,
        "num_prefill_tokens": num_prefill_tokens,
        "num_decode_tokens": num_decode_tokens,
        "request_ids": request_ids,
        "pp_rank": pp_rank,
        "pp_world_size": pp_world_size,
        "is_first_rank": is_first_rank,
        "is_last_rank": is_last_rank,
        "activation_bytes_per_rank": activation_bytes_per_rank,
        "recv_start_ts": recv_start_ts,
        "recv_end_ts": recv_end_ts,
        "preprocess_start_ts": preprocess_start_ts,
        "preprocess_end_ts": preprocess_end_ts,
        "forward_start_ts": forward_start_ts,
        "forward_end_ts": forward_end_ts,
        "send_start_ts": send_start_ts,
        "send_end_ts": send_end_ts,
        "recv_duration_ms": recv_duration_ms,
        "preprocess_duration_ms": preprocess_duration_ms,
        "send_duration_ms": send_duration_ms,
    }


def _pairing_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["model_name"],
        record["batch_id"],
        record["batch_size"],
        record["tensor_parallel_degree"],
        record["num_prefill_tokens"],
        record["num_decode_tokens"],
        tuple(record["request_ids"]),
        record["pp_world_size"],
    )


def _derive_phase_label(
    num_prefill_tokens: int,
    num_decode_tokens: int,
) -> str:
    if num_decode_tokens > 0 and num_prefill_tokens == 0:
        return "decode"
    if num_prefill_tokens > 0 and num_decode_tokens == 0:
        return "prefill"
    return "mixed"


def pair_adjacent_vllm_pp_stage_trace_records(
    records: list[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    records_by_rank_and_key: dict[tuple[tuple[Any, ...], int], dict[str, Any]] = {}
    for raw_record in records:
        normalized = (
            raw_record
            if "recv_duration_ms" in raw_record
            else normalize_vllm_pp_stage_trace_record(raw_record)
        )
        pair_key = (_pairing_key(normalized), int(normalized["pp_rank"]))
        if pair_key in records_by_rank_and_key:
            raise ValueError(
                "Duplicate PP stage trace record encountered for pairing key "
                f"{pair_key}."
            )
        records_by_rank_and_key[pair_key] = dict(normalized)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for (key, rank), producer in sorted(
        records_by_rank_and_key.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        consumer_key = (key, rank + 1)
        consumer = records_by_rank_and_key.get(consumer_key)
        if consumer is None:
            continue
        pairs.append((producer, consumer))
    return pairs


def build_pp_stage_boundary_overhead_row(
    producer: Mapping[str, Any],
    consumer: Mapping[str, Any],
    *,
    existing_pp_send_recv_wire_ms: float,
) -> dict[str, Any]:
    producer_rank = int(producer["pp_rank"])
    consumer_rank = int(consumer["pp_rank"])
    if consumer_rank != producer_rank + 1:
        raise ValueError(
            "PP stage boundary producer/consumer ranks must be consecutive, got "
            f"{producer_rank} and {consumer_rank}."
        )

    producer_key = _pairing_key(producer)
    consumer_key = _pairing_key(consumer)
    if producer_key != consumer_key:
        raise ValueError(
            "PP stage boundary producer/consumer records do not describe the same "
            "logical batch."
        )

    boundary_start_ts = producer.get("forward_end_ts")
    if boundary_start_ts is None:
        raise ValueError(
            "PP stage boundary producer record must contain 'forward_end_ts'."
        )
    boundary_end_ts = float(consumer["forward_start_ts"])
    if boundary_end_ts < float(boundary_start_ts):
        raise ValueError(
            "PP stage boundary 'forward_start_ts' must be >= producer "
            "'forward_end_ts'."
        )

    boundary_critical_path_ms = (
        float(boundary_end_ts) - float(boundary_start_ts)
    ) * 1000.0
    wire_ms = float(existing_pp_send_recv_wire_ms)
    if not math.isfinite(wire_ms) or wire_ms < 0:
        raise ValueError(
            "existing_pp_send_recv_wire_ms must be finite and >= 0, "
            f"got {existing_pp_send_recv_wire_ms}."
        )
    pp_stage_boundary_overhead_ms = boundary_critical_path_ms - wire_ms
    if pp_stage_boundary_overhead_ms < 0:
        raise ValueError(
            "Derived pp_stage_boundary_overhead_ms must be >= 0, got "
            f"{pp_stage_boundary_overhead_ms}."
        )

    return {
        "model_name": producer["model_name"],
        "batch_size": int(producer["batch_size"]),
        "tensor_parallel_degree": int(producer["tensor_parallel_degree"]),
        "num_prefill_tokens": int(producer["num_prefill_tokens"]),
        "num_decode_tokens": int(producer["num_decode_tokens"]),
        "producer_pp_rank": producer_rank,
        "consumer_pp_rank": consumer_rank,
        "pp_world_size": int(producer["pp_world_size"]),
        "activation_bytes_per_rank": int(consumer["activation_bytes_per_rank"]),
        "boundary_critical_path_ms": boundary_critical_path_ms,
        "producer_send_duration_ms": float(producer["send_duration_ms"]),
        "consumer_recv_duration_ms": float(consumer["recv_duration_ms"]),
        "consumer_preprocess_duration_ms": float(
            consumer["preprocess_duration_ms"]
        ),
        "existing_pp_send_recv_wire_ms": wire_ms,
        "pp_stage_boundary_overhead_ms": pp_stage_boundary_overhead_ms,
    }


def build_pp_receiver_head_row(
    producer: Mapping[str, Any],
    consumer: Mapping[str, Any],
    *,
    previous_consumer_forward_end_ts_same_stage: float | None,
) -> dict[str, Any]:
    producer_rank = int(producer["pp_rank"])
    consumer_rank = int(consumer["pp_rank"])
    if consumer_rank != producer_rank + 1:
        raise ValueError(
            "PP receiver-head producer/consumer ranks must be consecutive, got "
            f"{producer_rank} and {consumer_rank}."
        )

    producer_key = _pairing_key(producer)
    consumer_key = _pairing_key(consumer)
    if producer_key != consumer_key:
        raise ValueError(
            "PP receiver-head producer/consumer records do not describe the same "
            "logical batch."
        )

    consumer_recv_end_ts = consumer["recv_end_ts"]
    if consumer_recv_end_ts is None:
        raise ValueError(
            "PP receiver-head consumer record must contain 'recv_end_ts'."
        )
    # Receiver-head runtime starts from the consumer-visible receive completion.
    # Producer send completion is modeled separately by pp_producer_send_path.
    handoff_complete_ts = float(consumer_recv_end_ts)
    if previous_consumer_forward_end_ts_same_stage is None:
        consumer_ready_ts = handoff_complete_ts
    else:
        consumer_ready_ts = max(
            handoff_complete_ts,
            float(previous_consumer_forward_end_ts_same_stage),
        )

    forward_start_ts = float(consumer["forward_start_ts"])
    preprocess_start_ts = float(consumer["preprocess_start_ts"])
    preprocess_end_ts = float(consumer["preprocess_end_ts"])
    if consumer_ready_ts > forward_start_ts:
        raise ValueError(
            "Derived consumer_ready_ts must be <= consumer.forward_start_ts, got "
            f"consumer_ready_ts={consumer_ready_ts}, forward_start_ts={forward_start_ts}."
        )
    if consumer_ready_ts > preprocess_start_ts:
        raise ValueError(
            "Derived consumer_ready_ts must be <= consumer.preprocess_start_ts, got "
            f"consumer_ready_ts={consumer_ready_ts}, preprocess_start_ts={preprocess_start_ts}."
        )
    if preprocess_end_ts > forward_start_ts:
        raise ValueError(
            "consumer.preprocess_end_ts must be <= consumer.forward_start_ts, got "
            f"preprocess_end_ts={preprocess_end_ts}, forward_start_ts={forward_start_ts}."
        )

    handoff_complete_to_consumer_ready_ms = (
        consumer_ready_ts - handoff_complete_ts
    ) * 1000.0
    consumer_ready_to_preprocess_start_ms = (
        preprocess_start_ts - consumer_ready_ts
    ) * 1000.0
    consumer_pre_forward_gap_ms = (
        forward_start_ts - preprocess_end_ts
    ) * 1000.0
    pp_receiver_head_runtime_ms = (
        forward_start_ts - consumer_ready_ts
    ) * 1000.0

    if handoff_complete_to_consumer_ready_ms < 0:
        raise ValueError(
            "Derived handoff_complete_to_consumer_ready_ms must be >= 0."
        )
    if consumer_ready_to_preprocess_start_ms < 0:
        raise ValueError(
            "Derived consumer_ready_to_preprocess_start_ms must be >= 0."
        )
    if consumer_pre_forward_gap_ms < 0:
        raise ValueError("Derived consumer_pre_forward_gap_ms must be >= 0.")
    if pp_receiver_head_runtime_ms < 0:
        raise ValueError("Derived pp_receiver_head_runtime_ms must be >= 0.")

    phase_label = _derive_phase_label(
        int(consumer["num_prefill_tokens"]),
        int(consumer["num_decode_tokens"]),
    )

    return {
        "model_name": consumer["model_name"],
        "batch_size": int(consumer["batch_size"]),
        "tensor_parallel_degree": int(consumer["tensor_parallel_degree"]),
        "num_prefill_tokens": int(consumer["num_prefill_tokens"]),
        "num_decode_tokens": int(consumer["num_decode_tokens"]),
        "consumer_pp_rank": consumer_rank,
        "pp_world_size": int(consumer["pp_world_size"]),
        "activation_bytes_per_rank": int(consumer["activation_bytes_per_rank"]),
        "phase_label": phase_label,
        "producer_pp_rank": producer_rank,
        "pp_receiver_head_runtime_ms": pp_receiver_head_runtime_ms,
        "consumer_ready_to_preprocess_start_ms": consumer_ready_to_preprocess_start_ms,
        "consumer_preprocess_duration_ms": float(consumer["preprocess_duration_ms"]),
        "consumer_pre_forward_gap_ms": consumer_pre_forward_gap_ms,
        "handoff_complete_to_consumer_ready_ms": handoff_complete_to_consumer_ready_ms,
        "producer_send_duration_ms": float(producer["send_duration_ms"]),
        "consumer_recv_duration_ms": float(consumer["recv_duration_ms"]),
    }


def build_pp_producer_send_path_row(
    producer: Mapping[str, Any],
    *,
    existing_pp_send_recv_wire_ms: float,
) -> dict[str, Any]:
    producer_rank = int(producer["pp_rank"])
    pp_world_size = int(producer["pp_world_size"])
    if producer_rank >= pp_world_size - 1:
        raise ValueError(
            "PP producer send-path rows require a non-last producer rank, got "
            f"producer_pp_rank={producer_rank}, pp_world_size={pp_world_size}."
        )

    producer_send_duration_ms = float(producer["send_duration_ms"])
    wire_ms = float(existing_pp_send_recv_wire_ms)
    if not math.isfinite(wire_ms) or wire_ms < 0:
        raise ValueError(
            "existing_pp_send_recv_wire_ms must be finite and >= 0, "
            f"got {existing_pp_send_recv_wire_ms}."
        )

    pp_producer_send_path_runtime_ms = producer_send_duration_ms - wire_ms
    if pp_producer_send_path_runtime_ms < 0:
        raise ValueError(
            "Derived pp_producer_send_path_runtime_ms must be >= 0, got "
            f"{pp_producer_send_path_runtime_ms}."
        )

    return {
        "model_name": producer["model_name"],
        "batch_size": int(producer["batch_size"]),
        "tensor_parallel_degree": int(producer["tensor_parallel_degree"]),
        "num_prefill_tokens": int(producer["num_prefill_tokens"]),
        "num_decode_tokens": int(producer["num_decode_tokens"]),
        "producer_pp_rank": producer_rank,
        "consumer_pp_rank": producer_rank + 1,
        "pp_world_size": pp_world_size,
        "activation_bytes_per_rank": int(producer["activation_bytes_per_rank"]),
        "phase_label": _derive_phase_label(
            int(producer["num_prefill_tokens"]),
            int(producer["num_decode_tokens"]),
        ),
        "producer_send_duration_ms": producer_send_duration_ms,
        "existing_pp_send_recv_wire_ms": wire_ms,
        "pp_producer_send_path_runtime_ms": pp_producer_send_path_runtime_ms,
    }

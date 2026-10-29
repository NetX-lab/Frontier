"""Schema definitions for PP-specific overhead profiling CSVs."""

from __future__ import annotations

from typing import Final

PP_STAGE_BOUNDARY_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "producer_pp_rank",
    "consumer_pp_rank",
    "pp_world_size",
    "activation_bytes_per_rank",
)

PP_STAGE_BOUNDARY_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "producer_pp_rank",
    "consumer_pp_rank",
    "pp_world_size",
    "activation_bytes_per_rank",
    "boundary_critical_path_ms",
    "producer_send_duration_ms",
    "consumer_recv_duration_ms",
    "consumer_preprocess_duration_ms",
    "existing_pp_send_recv_wire_ms",
    "pp_stage_boundary_overhead_ms",
)

PP_STAGE_BOUNDARY_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    *PP_STAGE_BOUNDARY_IDENTITY_COLUMNS,
    "boundary_critical_path_ms",
    "producer_send_duration_ms",
    "consumer_recv_duration_ms",
    "consumer_preprocess_duration_ms",
    "existing_pp_send_recv_wire_ms",
    "pp_stage_boundary_overhead_ms",
)

PP_RECEIVER_HEAD_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "consumer_pp_rank",
    "pp_world_size",
    "activation_bytes_per_rank",
    "phase_label",
)

PP_RECEIVER_HEAD_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "producer_pp_rank",
    "profiling_precision",
    "other_overhead_source",
    "sample_count",
)

PP_RECEIVER_HEAD_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "pp_receiver_head_runtime_ms",
    "consumer_ready_to_preprocess_start_ms",
    "consumer_preprocess_duration_ms",
    "consumer_pre_forward_gap_ms",
)

PP_RECEIVER_HEAD_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "handoff_complete_to_consumer_ready_ms",
    "producer_send_duration_ms",
    "consumer_recv_duration_ms",
)

PP_RECEIVER_HEAD_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "consumer_pp_rank",
    "pp_world_size",
    "activation_bytes_per_rank",
    "producer_pp_rank",
    "sample_count",
    *PP_RECEIVER_HEAD_TARGET_COLUMNS,
    *PP_RECEIVER_HEAD_AUDIT_COLUMNS,
)

PP_RECEIVER_HEAD_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    *PP_RECEIVER_HEAD_IDENTITY_COLUMNS,
    *PP_RECEIVER_HEAD_METADATA_COLUMNS,
    *PP_RECEIVER_HEAD_TARGET_COLUMNS,
    *PP_RECEIVER_HEAD_AUDIT_COLUMNS,
)

PP_PRODUCER_SEND_PATH_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "producer_pp_rank",
    "pp_world_size",
    "activation_bytes_per_rank",
    "phase_label",
)

PP_PRODUCER_SEND_PATH_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "consumer_pp_rank",
    "profiling_precision",
    "other_overhead_source",
    "sample_count",
)

PP_PRODUCER_SEND_PATH_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "pp_producer_send_path_runtime_ms",
)

PP_PRODUCER_SEND_PATH_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "producer_send_duration_ms",
    "existing_pp_send_recv_wire_ms",
)

PP_PRODUCER_SEND_PATH_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "producer_pp_rank",
    "pp_world_size",
    "activation_bytes_per_rank",
    "consumer_pp_rank",
    "sample_count",
    *PP_PRODUCER_SEND_PATH_TARGET_COLUMNS,
    *PP_PRODUCER_SEND_PATH_AUDIT_COLUMNS,
)

PP_PRODUCER_SEND_PATH_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    *PP_PRODUCER_SEND_PATH_IDENTITY_COLUMNS,
    *PP_PRODUCER_SEND_PATH_METADATA_COLUMNS,
    *PP_PRODUCER_SEND_PATH_TARGET_COLUMNS,
    *PP_PRODUCER_SEND_PATH_AUDIT_COLUMNS,
)

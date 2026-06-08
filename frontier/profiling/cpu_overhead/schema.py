"""CPU overhead profiling CSV schema definitions."""

from __future__ import annotations

from typing import Final

# Schema-v2 defaults for legacy (schema-v1) compatibility.
DEFAULT_NUM_PREFILL_TOKENS: Final[int] = 256
DEFAULT_NUM_DECODE_TOKENS_AMPLIFICATION_FACTOR: Final[int] = 3
DEFAULT_SCHEDULING_MODE: Final[str] = "sync"
VALID_SCHEDULING_MODES: Final[tuple[str, ...]] = ("sync", "async")

# Identity fields for one profiling sample.
CPU_OVERHEAD_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "scheduling_mode",
)

# Numeric fields produced by profiling.
CPU_OVERHEAD_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "schedule_mean",
    "schedule_median",
    "sampler_e2e_mean",
    "sampler_e2e_median",
    "prepare_inputs_e2e_mean",
    "prepare_inputs_e2e_median",
    "process_model_outputs_mean",
    "process_model_outputs_median",
    "ray_comm_time_mean",
)

# Required fields for contract validation.
CPU_OVERHEAD_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "model_name",
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "scheduling_mode",
    "schedule_mean",
    "schedule_median",
    "sampler_e2e_mean",
    "sampler_e2e_median",
    "prepare_inputs_e2e_mean",
    "prepare_inputs_e2e_median",
    "process_model_outputs_mean",
    "process_model_outputs_median",
    "ray_comm_time_mean",
    "profiling_precision",
)

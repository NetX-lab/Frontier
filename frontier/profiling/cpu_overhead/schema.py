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

# Optional host time to launch an eager forward step's kernels. A table that
# carries it prices eager steps as the slower of the launch and device streams.
CPU_OVERHEAD_FORWARD_LAUNCH_COLUMNS: Final[tuple[str, ...]] = (
    "forward_launch_mean",
    "forward_launch_median",
)

# Optional device time after an eager forward's last kernel launch. It comes
# with forward_launch and extends a step whose launch outlasts its device work.
CPU_OVERHEAD_FORWARD_DRAIN_COLUMNS: Final[tuple[str, ...]] = (
    "forward_drain_mean",
    "forward_drain_median",
)

# The terms a CPU-overhead table prices, one model each. The eager-forward terms
# are optional columns.
CPU_OVERHEAD_TERMS: Final[tuple[str, ...]] = (
    "schedule",
    "sampler_e2e",
    "prepare_inputs_e2e",
    "process_model_outputs",
    "ray_comm_time",
    "forward_launch",
    "forward_drain",
)
CPU_OVERHEAD_OPTIONAL_TERMS: Final[tuple[str, ...]] = ("forward_launch", "forward_drain")

# A PP>1 CPU-probe table keys each row by the pipeline stage that runs the row's
# intervals; single-stage tables have no such column.
CPU_OVERHEAD_PIPELINE_STAGE_COLUMN: Final[str] = "pipeline_stage_id"

# A table probed with engine-idle buckets keys each row by the lower edge, in ms,
# of the bucket that holds the engine's idle time before the step.
CPU_OVERHEAD_ENGINE_IDLE_COLUMN: Final[str] = "engine_idle_ms"

# Step features of the CPU-overhead models; a stage-keyed or idle-keyed table
# adds its key columns.
CPU_OVERHEAD_STEP_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "batch_size",
    "num_prefill_tokens",
    "num_decode_tokens",
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

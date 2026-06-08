# Non-KV Cache Overhead Profiling

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-12 | Switched `memory_planner_profiled` default to measured param memory; added `use_analytical_param_memory` opt-in |
| 2026-02-12 | Clarified `overhead` vs `non_kv_cache_memory` and added mode-specific formulas for block calculation |
| 2026-02-12 | Added `Profile return vs Planner input` section to explain why planner consumes `overhead_bytes` |

## Overview

This module measures **non-KV cache GPU memory overhead** for LLM inference — all GPU memory consumed by a model *except* the KV cache. The measured overhead feeds into Frontier's `MemoryPlanner` to compute accurate KV cache block counts, aligning with vLLM's runtime memory planning.

**This module measures memory footprint, not latency.** A forward pass is executed solely to trigger memory allocations.

## Scale-Out Semantics

The runtime profiler is intentionally **single-rank**:

- Tensor parallelism is represented by one **representative rank** with `world_size=tp_size`.
- If `torch.distributed` is not initialized, `initialize_model_parallel(...)` simulates TP world size locally instead of requiring real multi-process launch.
- NCCL/non-torch memory is added analytically by `nccl_buffer_estimator.py`; it does **not** perform real NCCL communicator initialization.
- For pipeline parallelism (`pp > 1`), the profiler measures each pipeline stage slice separately and selects the **largest per-device stage footprint** as the planner input.

Implication:

- The profiling workflow can be extended to large logical deployments such as `pp * tp * dp = 128` **without requiring a physical 128-GPU cluster** for the measurement step.
- What still matters is that the measured rank is representative of the target hardware/runtime path (for example A800 + target TP/PP sharding semantics).

## Memory Breakdown Formula

```text
non_kv_cache_memory = weights_memory + torch_peak_increase + non_torch_increase
```

| Component | What It Captures | Measurement Method |
|-----------|------------------|--------------------|
| `weights_memory_bytes` | Model parameter weights | `param_counter` (theoretical) or `runtime_model_load` (actual delta) |
| `torch_peak_increase_bytes` | Activations, intermediate tensors, workspace buffers | `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` delta |
| `non_torch_increase_bytes` | NCCL buffers, custom allreduce, driver overhead | `torch.cuda.mem_get_info()` delta |

The `overhead_bytes` returned to callers equals `non_kv_cache_memory - weights_memory`.

In scheduler integration, `MemoryPlanner` receives `overhead_bytes` (not full
`non_kv_cache_memory`). The planner computes:

```text
available_kv = requested_memory - param_memory - overhead
```

Where:

- `param_memory` is `parameter_memory_per_device`, computed by `ParamCounter`
  and converted to bytes as `2 * num_parameters_per_device`.
- `overhead` is `non_kv_cache_overhead_bytes`, i.e.
  `non_kv_cache_memory - weights_memory`.

Therefore, planner-side `param_memory + overhead` approximates
`non_kv_cache_memory`.

When `num_blocks_mode=memory_planner_profiled` and runtime profiling is enabled,
Frontier now defaults to measured param memory semantics. Scheduler rewrites the
effective planner overhead so that planner-side subtraction matches measured
`weights_memory_bytes` from runtime profiling.

## File Structure

| File | Responsibility |
|------|----------------|
| `types.py` | `NonKVMemoryBreakdown` — immutable dataclass with validation |
| `memory_accounting.py` | `MemorySnapshot`, `MemoryProfilingResult`, `memory_profiling()` context manager |
| `gpu_idle_guard.py` | GPU idle state validation via nvidia-smi queries |
| `nccl_buffer_estimator.py` | Mechanism-based NCCL buffer overhead estimation |
| `runner.py` | Single-rank profiling orchestration (`run_single_rank_profile()`) |
| `runtime_estimator.py` | Main entry point: `estimate_non_kv_cache_profile()`, `_FullStructureGPTModel`, caching |

## Profiling Workflow

`estimate_non_kv_cache_profile()` in `runtime_estimator.py` orchestrates a 6-phase process:

1. **Setup** — Initialize model-parallel state, build `_FullStructureGPTModel` (mirrors vLLM's `LlamaForCausalLM` layer structure), load dummy weights to GPU.
2. **Weights Measurement** — Either use the input `weights_memory_bytes` directly (`param_counter` mode) or measure the actual model-load memory delta (`runtime_model_load` mode).
3. **NCCL Buffer Allocation** — Estimate NCCL overhead via `estimate_nccl_non_torch_bytes()`, then allocate an explicit non-torch buffer via `cudaMalloc` (class `_CudaNonTorchAllocation`) to simulate NCCL memory pressure.
4. **Profile Run** — Enter `memory_profiling()` context manager, allocate a torch peak padding buffer (96–112 MiB), execute `model(input_ids, positions)` forward pass, synchronize CUDA.
5. **Result Calculation** — Extract `NonKVMemoryBreakdown` from before/after memory snapshots, compute `overhead_bytes = non_kv_cache_memory - weights_memory`.
6. **Cleanup** — Free non-torch allocation, delete model and tensors, empty CUDA cache, destroy model-parallel state.

Results are cached (thread-safe) by a composite key of model config + parallelism + max_tokens + weights_memory + NCCL config.

## NCCL Buffer Estimation

`nccl_buffer_estimator.py` provides a mechanism-based formula (no GPU required):

```text
effective_channels = min(num_peers × channels_per_peer, max_channels)
nccl_channel_bytes = effective_channels × buffsize × 2 × num_communicators
nccl_comm_overhead = (base_overhead + per_peer_overhead × num_peers) × num_communicators
total = max(sum(components), nccl_min_pool_bytes)
```

Defaults are calibrated for A800 TP=8 (645 MiB empirical floor). Returns 0 for TP=1.

## Integration Path

```text
CLI Config (num_blocks_mode, enable_runtime_profiling, non_kv_cache_overhead_bytes)
  │
  ▼
BaseReplicaScheduler.__init__()          [base_replica_scheduler.py:87-186]
  │  if enable_runtime_non_kv_cache_overhead_profiling:
  ▼
estimate_non_kv_cache_profile()          [runtime_estimator.py:733-806]
  │  → loads model, allocates NCCL buffers, runs forward pass
  │  → returns RuntimeNonKVProfileResult with overhead_bytes
  ▼
config.non_kv_cache_overhead_bytes = effective_overhead_bytes
  │
  ▼
MemoryPlanner.get_num_blocks()           [memory_planner.py:100-146]
  │  available_kv = total_memory × gpu_utilization - param_memory - overhead
  │  num_blocks = available_kv / page_size / num_layers
  ▼
VllmV1EngineReplicaScheduler            [vllm_v1_engine_replica_scheduler.py]
  │  _can_allocate_request: available_blocks >= watermark?
  │  _schedule_waiting_requests: admission control based on num_blocks
```

## Profile return vs Planner input

`estimate_non_kv_cache_profile()` returns a rich structure
(`RuntimeNonKVProfileResult`) containing both full and decomposed values:

- `input_weights_memory_bytes`
- `measured_weights_memory_bytes`
- `non_kv_cache_memory_bytes`
- `overhead_bytes`
- `torch_peak_increase_bytes`
- `non_torch_increase_bytes`

Planner input is intentionally narrower: `MemoryPlanner.get_num_blocks()` takes
`non_kv_cache_overhead_bytes` (i.e. `overhead_bytes`) while obtaining
`param_memory` from `ParamCounter`.

Equivalent planner decomposition:

```text
available_kv = requested_memory - param_memory - overhead
```

Where:

```text
overhead = non_kv_cache_memory - weights_memory
```

Rationale for this split design:

1. Keep planner usable even without runtime profiling (`memory_planner` mode).
2. Allow manual calibration via `non_kv_cache_overhead_bytes`.
3. Preserve compatibility with explicit block mode (`num_blocks_mode=explicit`).

In `memory_planner_profiled` + runtime profiling, scheduler computes effective
planner overhead as:

```text
effective_overhead = overhead + measured_weights_memory_bytes - input_weights_memory_bytes
```

This makes planner-side subtraction equivalent to measured non-KV memory:

```text
param_memory(input) + effective_overhead = measured_weights_memory_bytes + overhead
```

If `use_analytical_param_memory=True`, scheduler skips this adjustment and keeps
analytical ParamCounter semantics.

## num_blocks_mode

| Mode | Overhead Usage | Block Calculation | Use Case |
| ---- | -------------- | ----------------- | -------- |
| `memory_planner` | Ignored (set to 0) | Based on param memory only | Quick estimation |
| `memory_planner_profiled` | Uses profiled overhead (auto-adjusted by default) | Measured param memory + overhead (default) | Accurate runtime alignment |
| `explicit` | Not used | Direct config value | Manual override |

Mode-specific formulas:

| Mode | Effective formula used by scheduler |
|------|-------------------------------------|
| `memory_planner` | `available_kv = requested_memory - param_memory - 0` |
| `memory_planner_profiled` (default) | `available_kv = requested_memory - analytical_param_memory - (overhead + measured_param_memory - analytical_param_memory)` |
| `memory_planner_profiled` + `use_analytical_param_memory=True` | `available_kv = requested_memory - analytical_param_memory - overhead` |
| `explicit` | Skip planner derivation; use configured `num_blocks` directly |

Notes:

- In profiled mode, runtime profiling returns `overhead_bytes`, not full
  `non_kv_cache_memory`.
- In profiled mode, default behavior uses measured param memory semantics.
- Set `use_analytical_param_memory=True` to keep analytical ParamCounter
  semantics.

`memory_planner_profiled` requires either `enable_runtime_non_kv_cache_overhead_profiling=True` (auto-profile) or a manually specified `non_kv_cache_overhead_bytes` value.

## Configuration Parameters

Key CLI flags (prefixed with `--vllm_v1_scheduler_config_`):

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `num_blocks_mode` | `memory_planner_profiled` | Block calculation strategy |
| `enable_runtime_non_kv_cache_overhead_profiling` | `False` | Enable runtime GPU profiling |
| `non_kv_cache_overhead_bytes` | `0` | Manual overhead value (bytes) |
| `gpu_memory_utilization` | `None` | GPU memory fraction for KV cache planning |
| `runtime_weights_memory_source` | `param_counter` | `param_counter` or `runtime_model_load` |
| `use_analytical_param_memory` | `False` | In profiled mode, force planner to use analytical ParamCounter param memory |

NCCL buffer estimation is configured via `NCCLBufferEstimationConfig` fields passed through the scheduler. The `NCCL_BUFFSIZE` environment variable can override the per-channel buffer size.

## Usage Example

```bash
# With runtime profiling (auto-measures overhead on GPU)
python -m frontier.main --simulation_mode offline \
    --sys_arch co-location \
    --vllm_v1_scheduler_config_num_blocks_mode memory_planner_profiled \
    --vllm_v1_scheduler_config_enable_runtime_non_kv_cache_overhead_profiling \
    --vllm_v1_scheduler_config_runtime_weights_memory_source runtime_model_load \
    --vllm_v1_scheduler_config_gpu_memory_utilization 0.7 \
    ...

# Keep analytical ParamCounter param memory in profiled mode
python -m frontier.main --simulation_mode offline \
    --sys_arch co-location \
    --vllm_v1_scheduler_config_num_blocks_mode memory_planner_profiled \
    --vllm_v1_scheduler_config_enable_runtime_non_kv_cache_overhead_profiling \
    --vllm_v1_scheduler_config_use_analytical_param_memory \
    --vllm_v1_scheduler_config_gpu_memory_utilization 0.7 \
    ...

# With pre-calibrated overhead (no GPU needed at simulation time)
python -m frontier.main --simulation_mode offline \
    --sys_arch co-location \
    --vllm_v1_scheduler_config_num_blocks_mode memory_planner_profiled \
    --vllm_v1_scheduler_config_non_kv_cache_overhead_bytes 1422393344 \
    --vllm_v1_scheduler_config_gpu_memory_utilization 0.7 \
    ...
```

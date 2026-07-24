# Mixed-Layer (MoE + Dense) DECODE_FFN Scheduler Specification

## Overview

Models like `step-moe-noquant` have 61 layers where layers 0-3 and 60 are **dense**
and layers 4-59 are **MoE**. The DECODE_FFN scheduler must route each layer through
the correct execution path within a single simulation run.

## Single Source of Truth

```python
model_config.is_moe_layer(layer_global_id) -> bool
```

All branching decisions below use this method. It delegates to `get_moe_layer_ids()`
which is derived from the `moe_layers_enum` config field.

## Methods That Must Branch on Layer Type

### 1. A2F Path (Attention-to-FFN transition)

| Aspect | MoE Layer | Dense Layer |
|--------|-----------|-------------|
| Lane barrier | EP lane barrier (wait for all EP ranks) | Streaming (no barrier) |
| Rationale | EP dispatch requires synchronized token routing | No parallelism beyond TP |

### 2. FFN Group Entity

| Aspect | MoE Layer | Dense Layer |
|--------|-----------|-------------|
| Entity type | `EPBatchGroup` | `DenseFFNBatchGroup` |
| Contains | Per-expert token distributions, routing metadata | Simple batch token count |

### 3. Pre-Dispatch Operations

| Aspect | MoE Layer | Dense Layer |
|--------|-----------|-------------|
| Share-expert | Compute if model has share_expert | Skip |
| Gating/routing | Router top-k selection | Skip |
| Token shuffling | All-to-all prep (EP > 1) | Skip |

### 4. EP Dispatch/Combine Events

| Aspect | MoE Layer (EP > 1) | Dense Layer |
|--------|---------------------|-------------|
| Dispatch event | Emitted (all-to-all send) | Skipped entirely |
| Combine event | Emitted (all-to-all recv) | Skipped entirely |
| Single-EP MoE | Also skipped (EP == 1) | N/A |

### 5. FFN Compute

| Aspect | MoE Layer | Dense Layer |
|--------|-----------|-------------|
| Computation | Grouped-GEMM over routed tokens per expert | Standard dense FFN (2x or 3x GEMM) |
| Token count | Post-routing (top_k * batch_tokens / EP) | batch_tokens (unchanged) |
| Predictor call | `include_moe=True`, passes moe_tokens_input | `include_moe=False` |

### 6. F2A Return (FFN-to-Attention transition)

| Aspect | MoE Layer | Dense Layer |
|--------|-----------|-------------|
| Return path | EP group reconstruction (combine tokens from EP ranks) | Direct return to attention |
| State | EPBatchGroup dissolved after combine | DenseFFNBatchGroup consumed directly |

## Waiting-Room Isolation

At dense-to-MoE and MoE-to-dense layer boundaries:

- **No EP state carry-forward**: Dense layers produce no EP routing metadata.
  A subsequent MoE layer must initialize fresh EP state (routing, dispatch plan).
- **No barrier leak**: MoE lane barriers must not block dense layers that follow.
  Dense layers operate independently of EP synchronization.
- **Implementation**: The waiting room (if used) must tag entries with layer type.
  When draining for a new layer, discard incompatible state rather than attempting
  conversion.

## Implementation Notes

- The predictor already handles per-layer switching at
  `sklearn_moe_execution_time_predictor.py:2619-2624` via `is_moe_layer()`.
- Entity classes `EPBatchGroup` and `DenseFFNBatchGroup` already exist in
  `frontier/entities/batch.py`.
- The reference implementation in `round_robin_cluster_scheduler.py` uses
  `_is_decode_ffn_moe_layer()` as the per-layer gate (lines 1932-1943, 2094-2366).

# Non-Ray Multi-GPU Support Changelog

**Date**: 2025-12-03
**Version**: 2.1

---

## Overview

This document records the implementation of non-Ray multi-GPU support for MoE profiling. Due to compatibility issues between Ray 2.52.1 and grpcio 1.67.1, we implemented an alternative multi-GPU solution using `torch.multiprocessing` and `ProcessPoolExecutor`.

---

## Modified Files

### 1. `frontier/profiling/moe/main.py`

**Lines Modified**: 1-100, 203-413, 416-497

#### New Imports (Lines 28-45)
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Conditionally import ray - only needed when not using --disable_ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
```

#### New Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `_worker_init(gpu_id)` | 52-56 | Initialize worker process with specific GPU binding |
| `_worker_profile_task(task_args)` | 59-99 | Worker function for multiprocessing profiling |
| `_get_available_gpus()` | 395-413 | Get list of available GPU IDs from environment |

#### Modified Functions

| Function | Key Changes |
|----------|-------------|
| `profile_model()` | Complete rewrite to support three modes: Ray, multi-GPU multiprocessing, single-GPU |
| `main()` | Added execution mode logging and explicit Ray initialization error handling |

### 2. `frontier/profiling/common/model_config.py`

**Lines Modified**: 93-123

#### New Method: `to_dict()` (Lines 93-123)

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert ModelConfig to a dictionary for serialization."""
    return {
        "name": self.name,
        "num_layers": self.num_layers,
        "hidden_dim": self.hidden_dim,
        # ... all other fields
    }
```

**Purpose**: Enable `ModelConfig` serialization across process boundaries for multiprocessing.

---

## Execution Modes

The MoE profiling module now supports three execution modes:

| Mode | Condition | Description |
|------|-----------|-------------|
| **Ray Mode** | `--disable_ray` not set | Uses Ray actors for distributed profiling (currently broken) |
| **Multi-GPU Multiprocessing** | `--disable_ray` + `--num_gpus > 1` | Uses `ProcessPoolExecutor` with `spawn` context |
| **Single-GPU Sequential** | `--disable_ray` + `--num_gpus = 1` | Original single-GPU behavior |

### `--num_gpus` Behavior

| Mode | `--num_gpus` Behavior |
|------|----------------------|
| Ray Mode | Controls number of Ray actors created |
| Multi-GPU Mode | Controls number of worker processes and GPU allocation |
| Single-GPU Mode | Ignored (always uses 1 GPU) |

---

## Performance Test Results

| Test Case | GPUs | Tasks | Time | Status |
|-----------|------|-------|------|--------|
| Basic (no load imbalance) | 1 | 35 | ~2.9s | ✅ Pass |
| Basic (no load imbalance) | 2 | 35 | ~48s | ✅ Pass |
| Load imbalance | 2 | 140 | ~12m13s | ✅ Pass |

**Note**: Multi-GPU mode has higher per-task overhead due to process creation for each batch. For small profiling tasks, single-GPU mode may be faster.

---

## Known Issues

### Ray Mode Incompatibility

**Status**: ⚠️ Not functional

**Root Cause**: grpcio 1.67.1 is incompatible with Ray 2.52.1's dashboard_agent.

**Symptoms**:
```
Raylet crashes with: UnknownError: UNKNOWN:ipv4:127.0.0.1:62979: Trying to connect an http1.x server
```

**Workaround**: Always use `--disable_ray` flag.

**Related Files**:
- Ray dashboard_agent: `/path/to/ray/_private/dashboard/dashboard_agent.py`
- grpcio issue: https://github.com/ray-project/ray/issues/...

---

## Usage Examples

### Single GPU (Recommended for Small Tasks)
```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --device a800 \
    --num_gpus 1 \
    --max_tokens 256 \
    --disable_ray \
    --output_dir data/profiling
```

### Multi-GPU (Recommended for Large Tasks)
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --device a800 \
    --num_gpus 4 \
    --max_tokens 1024 \
    --num_tensor_parallel_workers 1 2 4 \
    --expert_parallel_sizes 1 2 4 8 \
    --enable_load_imbalance \
    --disable_ray \
    --output_dir data/profiling
```

---

## Migration Notes

1. **Always add `--disable_ray`** to profiling commands
2. **Set `CUDA_VISIBLE_DEVICES`** before running multi-GPU profiling
3. **`--num_gpus`** should not exceed the number of visible GPUs
4. **Ray and OpenTelemetry** are no longer required dependencies for profiling


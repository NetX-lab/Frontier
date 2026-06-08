# Frontier Profiling Module: Migration Roadmap

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-06-06 | Replaced stale pre-migration validation paths with current release profiling example paths and removed references to non-shipped helper scripts. |

**Objective**: Remove all Sarathi dependencies from `frontier/profiling/`  
**Timeline**: 4-5 days (25-33 hours)  
**Approach**: Incremental migration with validation at each phase

---

## Visual Migration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CURRENT STATE                                │
│  frontier/profiling/ → depends on sarathi-serve-vidur (15 imports)  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Foundation (Day 1)                       │
│  ✓ Create common/constants.py (OperationMetrics)                    │
│  ✓ Create common/parallel_config.py (ParallelConfig)                │
│  ✓ Create common/utils.py (initialize_dummy_weights)                │
│  ✓ Update model_config.py to use local ParallelConfig               │
│  Dependencies Removed: 4/15 (27%)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 2: Model Layers (Day 1-2)                     │
│  ✓ Create common/layers/activation.py (SiluAndMul)                  │
│  ✓ Create common/layers/layernorm.py (RMSNorm)                      │
│  ✓ Create common/layers/rotary_embedding.py (get_rope)              │
│  ✓ Update linear_op_impl.py and moe_impl.py imports                       │
│  Dependencies Removed: 7/15 (47%)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                PHASE 3: Parallel Layers (Day 2-3)                    │
│  ✓ Create common/parallel_layers.py (TP layers)                     │
│  ✓ Add unit tests for TP=1, 2, 4, 8                                 │
│  ✓ Update linear_op_impl.py and moe_impl.py imports                       │
│  Dependencies Removed: 9/15 (60%)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 4: Attention Backends (Day 3-4)                   │
│  ✓ Create attention/sequence_metadata.py                            │
│  ✓ Create attention/backends/base.py                                │
│  ✓ Create attention/backends/no_op.py                               │
│  ✓ Create attention/backends/flashinfer.py                          │
│  ✓ Create attention/backends/__init__.py (factory)                  │
│  ✓ Update attention_wrapper.py and main.py                          │
│  Dependencies Removed: 14/15 (93%)                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 5: Validation & Cleanup (Day 4-5)                 │
│  ✓ Remove all Sarathi imports                                       │
│  ✓ Remove CudaTimer monkey patching                                 │
│  ✓ Run before/after profiling comparison                            │
│  ✓ Update documentation                                             │
│  Dependencies Removed: 15/15 (100%) ✅                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         TARGET STATE                                 │
│  frontier/profiling/ → fully independent, zero Sarathi dependencies │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Phase Breakdown

### 📅 Day 1: Foundation + Model Layers (7-8 hours)

#### Morning Session (3-4 hours): Phase 1 - Foundation

**Task 1.1**: Create `frontier/profiling/common/constants.py` (30 min)
```python
# NEW FILE
import enum

class OperationMetrics(enum.Enum):
    # Attention operations
    ATTN_PREFILL = "attn_prefill"
    ATTN_DECODE = "attn_decode"
    ATTN_PRE_PROJ = "attn_pre_proj"
    ATTN_POST_PROJ = "attn_post_proj"
    ATTN_ROPE = "attn_rope"
    ATTN_KV_CACHE_SAVE = "attn_kv_cache_save"
    
    # MLP operations
    MLP_UP_PROJ = "mlp_up_proj"
    MLP_DOWN_PROJ = "mlp_down_proj"
    MLP_ACTIVATION = "mlp_activation"
    
    # MoE operations
    MOE_GATING = "moe_gating"
    MOE_SHUFFLING = "moe_shuffling"
    MOE_GROUPED_GEMM = "moe_grouped_gemm"
    
    # Normalization
    INPUT_LAYERNORM = "input_layernorm"
    POST_ATTENTION_LAYERNORM = "post_attention_layernorm"
    
    # Other
    ADD = "add"
```

**Task 1.2**: Create `frontier/profiling/common/parallel_config.py` (30 min)
```python
# NEW FILE
from dataclasses import dataclass

@dataclass
class ParallelConfig:
    """Simple parallel configuration for profiling."""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size
```

**Task 1.3**: Create `frontier/profiling/common/utils.py` (15 min)
```python
# NEW FILE
import torch

def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """Initialize model weights with random values."""
    for param in model.parameters():
        param.data.uniform_(low, high)
```

**Task 1.4**: Update `frontier/profiling/common/model_config.py` (15 min)
```python
# MODIFY: Line 5
# OLD: from sarathi.config import ParallelConfig
# NEW: from frontier.profiling.common.parallel_config import ParallelConfig
```

**Task 1.5**: Update `frontier/profiling/utils/__init__.py` (15 min)
```python
# MODIFY: Line 8
# OLD: from sarathi.config import ParallelConfig
# NEW: from frontier.profiling.common.parallel_config import ParallelConfig
```

**Validation**: Run `python -m pytest tests/profiling/test_foundation.py` (if exists)

---

#### Afternoon Session (4 hours): Phase 2 - Model Layers

**Task 2.1**: Create `frontier/profiling/common/layers/activation.py` (1 hour)
- Copy `SiluAndMul` from Sarathi
- Update imports to use local `CudaTimer`
- Test with simple forward pass

**Task 2.2**: Create `frontier/profiling/common/layers/layernorm.py` (1 hour)
- Copy `RMSNorm` from Sarathi
- Update imports
- Test with simple forward pass

**Task 2.3**: Create `frontier/profiling/common/layers/rotary_embedding.py` (2 hours)
- Copy `get_rope()` factory and all RoPE variants
- Update imports
- Test with different RoPE types (standard, linear scaling, dynamic)

**Task 2.4**: Update MLP/MoE implementations (30 min)
```python
# MODIFY: frontier/profiling/linear_op/linear_op_impl.py
# OLD: from sarathi.model_executor.layers.activation import SiluAndMul
# NEW: from frontier.profiling.common.layers.activation import SiluAndMul

# OLD: from sarathi.model_executor.layers.layernorm import RMSNorm
# NEW: from frontier.profiling.common.layers.layernorm import RMSNorm

# OLD: from sarathi.model_executor.layers.rotary_embedding import get_rope
# NEW: from frontier.profiling.common.layers.rotary_embedding import get_rope
```

**Validation**: Run linear_op profiling on small model, compare with baseline

---

### 📅 Day 2-3: Parallel Layers (8-11 hours)

#### Phase 3: Tensor Parallel Layers

**Task 3.1**: Create `frontier/profiling/common/parallel_layers.py` (6-8 hours)

**Sub-task 3.1.1**: Migrate parallel state management (2 hours)
- Copy parallel state initialization from Sarathi
- Adapt for profiling use case (simpler than full serving)

**Sub-task 3.1.2**: Migrate `ColumnParallelLinear` (2 hours)
- Copy implementation from Sarathi
- Update imports
- Test with TP=1, 2, 4

**Sub-task 3.1.3**: Migrate `RowParallelLinear` (2 hours)
- Copy implementation from Sarathi
- Update imports
- Test with TP=1, 2, 4

**Sub-task 3.1.4**: Migrate `VocabParallelEmbedding` (1 hour)
- Copy implementation from Sarathi
- Update imports
- Test with TP=1, 2

**Task 3.2**: Add unit tests (2-3 hours)
```python
# NEW FILE: tests/profiling/test_parallel_layers.py
def test_column_parallel_linear_tp1():
    # Test TP=1 (no parallelism)
    pass

def test_column_parallel_linear_tp2():
    # Test TP=2 with All-Gather
    pass

def test_row_parallel_linear_tp2():
    # Test TP=2 with All-Reduce
    pass
```

**Task 3.3**: Update MLP/MoE implementations (30 min)
```python
# MODIFY: frontier/profiling/linear_op/linear_op_impl.py
# OLD: from sarathi.model_executor.parallel_utils.tensor_parallel.layers import (
# NEW: from frontier.profiling.common.parallel_layers import (
```

**Task 3.4**: Update MLP wrapper (15 min)
```python
# MODIFY: frontier/profiling/linear_op/linear_op_wrapper.py
# OLD: from sarathi.model_executor.weight_utils import initialize_dummy_weights
# NEW: from frontier.profiling.common.utils import initialize_dummy_weights
```

**Validation**: Run full linear_op and MoE profiling, compare with baseline

---

### 📅 Day 3-4: Attention Backends (6-8 hours)

#### Phase 4: Attention Backend System

**Task 4.1**: Create `frontier/profiling/attention/sequence_metadata.py` (1-2 hours)
```python
# NEW FILE
from typing import List, Optional

class SequenceMetadata:
    """Lightweight sequence metadata for attention profiling."""
    def __init__(
        self,
        block_table: Optional[List[int]] = None,
        prompt_chunk_len: int = 0,
        is_prompt: bool = True,
    ):
        self.block_table = block_table
        self.prompt_chunk_len = prompt_chunk_len
        self.is_prompt = is_prompt
        # Add other fields as needed for profiling
```

**Task 4.2**: Create `frontier/profiling/attention/backends/base.py` (1 hour)
- Copy `BaseAttentionWrapper` from Sarathi
- Update imports to use local `SequenceMetadata`, `OperationMetrics`, `CudaTimer`

**Task 4.3**: Create `frontier/profiling/attention/backends/no_op.py` (30 min)
- Copy `NoOpAttentionWrapper` from Sarathi
- Update imports

**Task 4.4**: Create `frontier/profiling/attention/backends/flashinfer.py` (2-3 hours)
- Copy `FlashinferAttentionWrapper` from Sarathi
- Update imports
- Test with FlashInfer library

**Task 4.5**: Create `frontier/profiling/attention/backends/__init__.py` (1 hour)
```python
# NEW FILE
from enum import Enum
from typing import Union

from frontier.profiling.attention.backends.base import BaseAttentionWrapper
from frontier.profiling.attention.backends.flashinfer import FlashinferAttentionWrapper
from frontier.profiling.attention.backends.no_op import NoOpAttentionWrapper

class AttentionBackend(Enum):
    FLASHINFER = "FLASHINFER"
    NO_OP = "NO_OP"

ATTENTION_BACKEND = AttentionBackend.NO_OP

def set_attention_backend(backend: Union[str, AttentionBackend]):
    if isinstance(backend, str):
        backend = backend.upper()
        if backend not in AttentionBackend.__members__:
            raise ValueError(f"Unsupported attention backend: {backend}")
        backend = AttentionBackend[backend]
    elif not isinstance(backend, AttentionBackend):
        raise ValueError(f"Unsupported attention backend: {backend}")
    
    global ATTENTION_BACKEND
    ATTENTION_BACKEND = backend

def get_attention_wrapper():
    if ATTENTION_BACKEND == AttentionBackend.FLASHINFER:
        return FlashinferAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.NO_OP:
        return NoOpAttentionWrapper.get_instance()
    
    raise ValueError(f"Unsupported attention backend: {ATTENTION_BACKEND}")
```

**Task 4.6**: Update attention profiling files (1 hour)
```python
# MODIFY: frontier/profiling/attention/attention_wrapper.py
# OLD: from sarathi.config import ParallelConfig
# NEW: from frontier.profiling.common.parallel_config import ParallelConfig

# OLD: from sarathi.model_executor.attention import (
# NEW: from frontier.profiling.attention.backends import (

# REMOVE: Monkey patching lines (5-11)

# MODIFY: frontier/profiling/attention/main.py
# OLD: from sarathi.config import ParallelConfig
# NEW: from frontier.profiling.common.parallel_config import ParallelConfig

# OLD: from sarathi.model_executor.attention import AttentionBackend
# NEW: from frontier.profiling.attention.backends import AttentionBackend

# MODIFY: frontier/profiling/attention/sequence_proxy.py
# Update to use local SequenceMetadata
```

**Validation**: Run full attention profiling with FlashInfer backend

---

### 📅 Day 4-5: Validation & Cleanup (4-6 hours)

#### Phase 5: Final Validation

**Task 5.1**: Remove all Sarathi imports (1 hour)
```bash
# Search for remaining Sarathi imports
grep -r "from sarathi\|import sarathi" frontier/profiling --include="*.py"

# Should only find cpu_overhead/benchmark_runner.py (deferred)
```

**Task 5.2**: Remove CudaTimer monkey patching (30 min)
- Remove monkey patch lines from all files
- Verify CudaTimer is imported directly

**Task 5.3**: Run before/after profiling comparison (2-3 hours)

**Ad-hoc comparison flow**:
```bash
#!/bin/bash

MODEL="meta-llama/Llama-2-7b-hf"
DEVICE="a100"

echo "=== Validating Attention Profiling ==="
bash frontier/profiling/example/test_profiling_attn.sh \
  --device $DEVICE --model $MODEL --attention-backend flashinfer

echo "=== Validating Linear Operator Profiling ==="
bash frontier/profiling/example/test_profiling_linear_op.sh \
  --device $DEVICE --model $MODEL

echo "=== Validating MoE Profiling ==="
bash frontier/profiling/example/test_profiling_moe.sh \
  --device $DEVICE --model qwen2_moe_example

echo "=== Compare Results ==="
echo "Compare data/profiling_before/ and data/profiling_after/ with your release validation tool."
```

**Task 5.4**: Update documentation (1-2 hours)
- Update `frontier/profiling/README.md`
- Update `frontier/profiling/example/README.md`
- Create `frontier/profiling/attention/backends/README.md` (backend extension guide)

---

## Validation Checklist

### ✅ Functional Validation

- [ ] Attention profiling runs without errors
- [ ] linear_op profiling runs without errors
- [ ] MoE profiling runs without errors
- [ ] FlashInfer backend works correctly
- [ ] NoOp backend works correctly
- [ ] All TP sizes work (1, 2, 4, 8)
- [ ] All models work (Llama, Qwen, Phi, etc.)

### ✅ Correctness Validation

- [ ] Attention profiling results match baseline (within 1%)
- [ ] linear_op profiling results match baseline (within 1%)
- [ ] MoE profiling results match baseline (within 1%)
- [ ] CSV output format unchanged
- [ ] All operations are profiled (no missing metrics)

### ✅ Code Quality Validation

- [ ] Zero imports from `sarathi` (except cpu_overhead, deferred)
- [ ] No monkey patching
- [ ] All unit tests pass
- [ ] No circular dependencies
- [ ] Clean directory structure

### ✅ Documentation Validation

- [ ] README updated with new structure
- [ ] Backend extension guide created
- [ ] Migration notes documented
- [ ] API changes documented (if any)

---

## Rollback Plan

If migration fails at any phase:

1. **Immediate Rollback**: Revert to previous commit
2. **Identify Issue**: Debug the specific component that failed
3. **Fix Forward**: Fix the issue and continue migration
4. **Alternative**: Keep Sarathi dependency for problematic component, defer migration

**Git Strategy**:
```bash
# Create feature branch
git checkout -b refactor/remove-sarathi-dependency

# Commit after each phase
git commit -m "Phase 1: Foundation - constants, parallel_config, utils"
git commit -m "Phase 2: Model Layers - activation, layernorm, rope"
git commit -m "Phase 3: Parallel Layers - TP layers, unit tests"
git commit -m "Phase 4: Attention Backends - backends, factory, integration"
git commit -m "Phase 5: Validation - cleanup, tests, docs"

# If rollback needed
git revert <commit-hash>
```

---

## Success Metrics

### Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Sarathi imports removed | 100% (14/15, excluding cpu_overhead) | `grep -r "from sarathi" frontier/profiling` |
| Profiling result accuracy | >99% match | CSV comparison script |
| Test coverage | >80% | `pytest --cov` |
| Migration time | <35 hours | Time tracking |

### Qualitative Metrics

- Code maintainability improved (no external dependency)
- Easier to add new backends
- Clearer separation of concerns
- Better documentation

---

## Post-Migration Tasks

### Immediate (Week 1)
- [ ] Monitor profiling results for anomalies
- [ ] Address any issues reported by users
- [ ] Update CI/CD to remove Sarathi installation

### Short-term (Month 1)
- [ ] Add new attention backend (e.g., xFormers)
- [ ] Improve unit test coverage
- [ ] Performance optimization

### Long-term (Quarter 1)
- [ ] Migrate cpu_overhead module (if needed)
- [ ] Consider upstreaming improvements to Sarathi (if applicable)
- [ ] Explore additional profiling features

---

**End of Roadmap**

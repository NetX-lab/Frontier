# Frontier Profiling Module: Sarathi Dependency Analysis Report

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-06-06 | Replaced stale pre-migration test script paths with current release profiling example paths. |

**Date**: 2025-10-25  
**Objective**: Remove all dependencies on `sarathi-serve-vidur` from `frontier/profiling/` module  
**Status**: Phase 1 - Analysis Complete

---

## Executive Summary

The `frontier/profiling/` module currently has **15 distinct import statements** from `sarathi-serve-vidur` across **9 Python files**. These dependencies can be categorized into three groups:

1. **Essential for Profiling** (Must migrate): 60% of dependencies
2. **Can be replaced with Frontier equivalents**: 30% of dependencies
3. **Can be removed/simplified**: 10% of dependencies

**Key Finding**: Most Sarathi dependencies are **lightweight abstractions** (enums, dataclasses, simple utilities) that can be easily migrated. The only complex dependency is the **tensor parallel layers** module (~461 lines), which is already well-encapsulated.

---

## 1. Complete Dependency Inventory

### 1.1 Dependency Matrix

| Sarathi Module | Used By | Category | Lines of Code | Migration Priority |
|----------------|---------|----------|---------------|-------------------|
| `sarathi.config.ParallelConfig` | 3 files | **Replace** | ~10 | HIGH |
| `sarathi.model_executor.attention.*` | 2 files | **Migrate** | ~374 | CRITICAL |
| `sarathi.model_executor.layers.activation` | 2 files | **Migrate** | ~61 | MEDIUM |
| `sarathi.model_executor.layers.layernorm` | 1 file | **Migrate** | ~40 | MEDIUM |
| `sarathi.model_executor.layers.rotary_embedding` | 1 file | **Migrate** | ~344 | MEDIUM |
| `sarathi.model_executor.parallel_utils.tensor_parallel.layers` | 2 files | **Migrate** | ~461 | HIGH |
| `sarathi.model_executor.weight_utils.initialize_dummy_weights` | 1 file | **Migrate** | ~15 | LOW |
| `sarathi.metrics.constants.OperationMetrics` | 1 file | **Migrate** | ~30 | MEDIUM |
| `sarathi.metrics.constants.CpuOperationMetrics` | 1 file | **Remove** | ~10 | LOW |
| `sarathi.metrics.cuda_timer.CudaTimer` | 3 files | **Already Replaced** | 0 | DONE ✅ |
| `sarathi.LLMEngine` | 1 file | **Remove** | N/A | LOW |
| `sarathi.SamplingParams` | 1 file | **Remove** | N/A | LOW |
| `sarathi.core.datatypes.sequence.SequenceMetadata` | 1 file | **Migrate** | ~75 | MEDIUM |

**Total Sarathi Code to Migrate**: ~1,420 lines (estimated)

---

## 2. Detailed Dependency Analysis

### 2.1 Category A: Essential for Profiling (Must Migrate)

#### A1. Attention Backend System ⭐ **CRITICAL**

**Files**:
- `sarathi/model_executor/attention/__init__.py` (35 lines)
- `sarathi/model_executor/attention/base_attention_wrapper.py` (69 lines)
- `sarathi/model_executor/attention/flashinfer_attention_wrapper.py` (259 lines)
- `sarathi/model_executor/attention/no_op_attention_wrapper.py` (46 lines)
- `sarathi/types.py` (AttentionBackend enum, ~5 lines)

**Used By**:
- `frontier/profiling/attention/main.py`
- `frontier/profiling/attention/attention_wrapper.py`

**Functionality**:
- `AttentionBackend` enum: Defines available backends (FLASHINFER, NO_OP)
- `set_attention_backend()`: Global backend setter
- `get_attention_wrapper()`: Factory function returning backend instance
- `BaseAttentionWrapper`: Abstract base class defining interface
- `FlashinferAttentionWrapper`: FlashInfer implementation
- `NoOpAttentionWrapper`: No-op implementation for testing

**Migration Strategy**: **MIGRATE FULLY**
- This is the core of attention profiling extensibility
- Must preserve Strategy + Factory + Singleton patterns
- All 5 files should be migrated to `frontier/profiling/attention/backends/`

**Dependencies**:
- `SequenceMetadata` (from `sarathi.core.datatypes.sequence`)
- `OperationMetrics` (from `sarathi.metrics.constants`)
- `CudaTimer` (already replaced ✅)
- `ModelConfig`, `ParallelConfig` (can be replaced)

---

#### A2. Tensor Parallel Layers

**File**: `sarathi/model_executor/parallel_utils/tensor_parallel/layers.py` (461 lines)

**Used By**:
- `frontier/profiling/linear_op/linear_op_impl.py`
- `frontier/profiling/moe/moe_impl.py`

**Functionality**:
- `ColumnParallelLinear`: Column-wise tensor parallel linear layer
- `RowParallelLinear`: Row-wise tensor parallel linear layer
- `VocabParallelEmbedding`: Vocabulary parallel embedding layer

**Migration Strategy**: **MIGRATE FULLY**
- These are essential for MLP/MoE profiling
- Implement distributed communication (All-Reduce, All-Gather)
- Migrate to `frontier/profiling/common/parallel_layers.py`

**Dependencies**:
- `torch.distributed` (standard PyTorch)
- `CudaTimer` (already replaced ✅)
- Parallel state management (needs migration)

---

#### A3. Model Layers (Activation, LayerNorm, RoPE)

**Files**:
- `sarathi/model_executor/layers/activation.py` (61 lines) - `SiluAndMul`
- `sarathi/model_executor/layers/layernorm.py` (40 lines) - `RMSNorm`
- `sarathi/model_executor/layers/rotary_embedding.py` (344 lines) - `get_rope()`

**Used By**:
- `frontier/profiling/linear_op/linear_op_impl.py`
- `frontier/profiling/moe/moe_impl.py`

**Functionality**:
- `SiluAndMul`: Fused SiLU activation with gating
- `RMSNorm`: RMS normalization layer
- `get_rope()`: Rotary position embedding factory

**Migration Strategy**: **MIGRATE FULLY**
- These are standard LLM components
- Migrate to `frontier/profiling/common/layers/`
- Keep implementation identical for consistency

---

#### A4. SequenceMetadata

**File**: `sarathi/core/datatypes/sequence.py` (SequenceMetadata class, ~75 lines)

**Used By**:
- `frontier/profiling/attention/sequence_proxy.py` (as proxy)

**Functionality**:
- Metadata for sequences in attention profiling
- Contains: `block_table`, `prompt_chunk_len`, `is_prompt`, etc.

**Migration Strategy**: **SIMPLIFY & MIGRATE**
- Current usage is minimal (only in `SequenceMetadataProxy`)
- Create lightweight version in `frontier/profiling/attention/sequence_metadata.py`
- Only include fields needed for profiling

---

#### A5. OperationMetrics Enum

**File**: `sarathi/metrics/constants.py` (OperationMetrics enum, ~30 lines)

**Used By**:
- `frontier/profiling/attention/base_attention_wrapper.py` (via Sarathi)

**Functionality**:
- Enum defining operation names for timing
- Values: `ATTN_PREFILL`, `ATTN_DECODE`, `MLP_UP_PROJ`, etc.

**Migration Strategy**: **MIGRATE**
- Simple enum, easy to migrate
- Move to `frontier/profiling/common/constants.py`

---

### 2.2 Category B: Can Be Replaced with Frontier Equivalents

#### B1. ParallelConfig ⭐ **HIGH PRIORITY**

**File**: `sarathi/config/config.py` (ParallelConfig dataclass, ~10 lines)

**Used By**:
- `frontier/profiling/attention/main.py`
- `frontier/profiling/attention/attention_wrapper.py`
- `frontier/profiling/common/model_config.py`
- `frontier/profiling/utils/__init__.py`

**Sarathi Definition**:
```python
@dataclass
class ParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 1
    
    def __post_init__(self):
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size
```

**Frontier Equivalent**: `frontier.config.config.ReplicaConfig`
- Has `num_pipeline_stages`, `attn_tensor_parallel_size`, `moe_tensor_parallel_size`
- More comprehensive than Sarathi's version

**Migration Strategy**: **CREATE LIGHTWEIGHT ADAPTER**
- Create `frontier/profiling/common/parallel_config.py`
- Simple dataclass with `tensor_parallel_size` and `pipeline_parallel_size`
- No dependency on Frontier's complex `ReplicaConfig`

**Rationale**: Profiling needs a simple, standalone config, not the full Frontier simulation config.

---

### 2.3 Category C: Can Be Removed/Simplified

#### C1. CudaTimer ✅ **ALREADY DONE**

**Status**: Already replaced with `frontier/profiling/common/cuda_timer.py`

**Current Implementation**:
```python
# Monkey patching in all profiling files
import sarathi.metrics.cuda_timer
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer
```

**Migration Strategy**: **REMOVE MONKEY PATCHING**
- After migrating attention/MLP/MoE layers, remove all monkey patches
- Use Frontier's `CudaTimer` directly

---

#### C2. CPU Overhead Profiling (LLMEngine, SamplingParams, CpuOperationMetrics)

**File**: `frontier/profiling/cpu_overhead/benchmark_runner.py`

**Used By**: CPU overhead benchmarking (separate from core profiling)

**Migration Strategy**: **DEFER OR REMOVE**
- This is a separate benchmarking tool, not core profiling
- Can be handled in a separate phase
- Low priority for initial migration

---

#### C3. Weight Initialization Utility

**File**: `sarathi/model_executor/weight_utils.py` (`initialize_dummy_weights`, ~15 lines)

**Used By**: `frontier/profiling/linear_op/linear_op_wrapper.py`

**Functionality**:
```python
def initialize_dummy_weights(model, low=-1e-3, high=1e-3):
    for param in model.parameters():
        param.data.uniform_(low, high)
```

**Migration Strategy**: **MIGRATE (TRIVIAL)**
- Simple utility function
- Move to `frontier/profiling/common/utils.py`

---

## 3. Proposed Directory Structure

```
frontier/profiling/
├── attention/
│   ├── backends/                          # NEW: Attention backend system
│   │   ├── __init__.py                   # Factory functions (set_attention_backend, get_attention_wrapper)
│   │   ├── base.py                       # BaseAttentionWrapper (abstract base class)
│   │   ├── flashinfer.py                 # FlashinferAttentionWrapper
│   │   ├── no_op.py                      # NoOpAttentionWrapper
│   │   └── README.md                     # Backend extension guide
│   ├── attention_input.py                # Existing
│   ├── attention_wrapper.py              # Modified: use local backends
│   ├── main.py                           # Modified: use local backends
│   ├── sequence_metadata.py              # NEW: Lightweight SequenceMetadata
│   └── sequence_proxy.py                 # Modified: use local SequenceMetadata
│
├── mlp/
│   ├── linear_op_impl.py                       # Modified: use local layers
│   ├── linear_op_wrapper.py                    # Modified: use local utils
│   └── main.py                           # Existing
│
├── moe/
│   ├── moe_impl.py                       # Modified: use local layers
│   ├── moe_wrapper.py                    # Existing
│   └── main.py                           # Existing
│
├── common/
│   ├── constants.py                      # NEW: OperationMetrics enum
│   ├── cuda_timer.py                     # Existing
│   ├── model_config.py                   # Modified: use local ParallelConfig
│   ├── parallel_config.py                # NEW: Simple ParallelConfig dataclass
│   ├── parallel_layers.py                # NEW: ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
│   ├── timer_stats_store.py             # Existing
│   ├── utils.py                          # NEW: initialize_dummy_weights, etc.
│   └── layers/                           # NEW: Model layer implementations
│       ├── __init__.py
│       ├── activation.py                 # SiluAndMul
│       ├── layernorm.py                  # RMSNorm
│       └── rotary_embedding.py           # get_rope()
│
├── cpu_overhead/                          # DEFER: Handle separately
│   └── benchmark_runner.py
│
├── utils/
│   ├── __init__.py                       # Modified: use local ParallelConfig
│   └── ...
│
└── README.md                              # Updated: Document new structure
```

---

## 4. Migration Complexity Assessment

### 4.1 Complexity Levels

| Component | Lines | Complexity | External Deps | Estimated Effort |
|-----------|-------|------------|---------------|------------------|
| Attention Backends | 374 | **Medium** | flashinfer, torch | 4-6 hours |
| Tensor Parallel Layers | 461 | **High** | torch.distributed | 6-8 hours |
| Model Layers (Activation, Norm, RoPE) | 445 | **Low-Medium** | torch | 3-4 hours |
| ParallelConfig | 10 | **Trivial** | None | 30 minutes |
| SequenceMetadata | 75 | **Low** | None | 1-2 hours |
| OperationMetrics | 30 | **Trivial** | None | 30 minutes |
| Weight Utils | 15 | **Trivial** | torch | 15 minutes |
| **TOTAL** | **~1,410** | **Medium** | - | **15-22 hours** |

### 4.2 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Profiling results change after migration | Medium | High | Validate with before/after comparison tests |
| Tensor parallel layers behavior differs | Low | High | Copy implementation exactly, add unit tests |
| FlashInfer backend breaks | Low | Critical | Extensive testing with real profiling workloads |
| Import path updates miss some files | Medium | Medium | Use automated search/replace, verify with grep |
| Circular dependencies in migrated code | Low | Medium | Careful dependency ordering during migration |

---

## 5. File-by-File Migration Plan

### Phase 1: Foundation (Low-Risk, High-Value)

**Step 1.1**: Create `frontier/profiling/common/constants.py`
- Migrate `OperationMetrics` enum from Sarathi
- Add `CpuOperationMetrics` if needed
- **Effort**: 30 minutes
- **Risk**: Minimal

**Step 1.2**: Create `frontier/profiling/common/parallel_config.py`
- Simple dataclass with `tensor_parallel_size`, `pipeline_parallel_size`
- Add `world_size` property
- **Effort**: 30 minutes
- **Risk**: Minimal

**Step 1.3**: Create `frontier/profiling/common/utils.py`
- Migrate `initialize_dummy_weights()`
- **Effort**: 15 minutes
- **Risk**: Minimal

**Step 1.4**: Update `frontier/profiling/common/model_config.py`
- Replace `from sarathi.config import ParallelConfig` with local version
- **Effort**: 15 minutes
- **Risk**: Minimal

---

### Phase 2: Model Layers (Medium-Risk, Medium-Value)

**Step 2.1**: Create `frontier/profiling/common/layers/activation.py`
- Migrate `SiluAndMul` class
- **Effort**: 1 hour
- **Risk**: Low (simple activation function)

**Step 2.2**: Create `frontier/profiling/common/layers/layernorm.py`
- Migrate `RMSNorm` class
- **Effort**: 1 hour
- **Risk**: Low (standard normalization)

**Step 2.3**: Create `frontier/profiling/common/layers/rotary_embedding.py`
- Migrate `get_rope()` factory and RoPE implementations
- **Effort**: 2-3 hours
- **Risk**: Medium (complex logic, multiple RoPE variants)

**Step 2.4**: Update MLP/MoE implementations
- Update imports in `linear_op_impl.py` and `moe_impl.py`
- **Effort**: 30 minutes
- **Risk**: Low

---

### Phase 3: Tensor Parallel Layers (High-Risk, High-Value)

**Step 3.1**: Create `frontier/profiling/common/parallel_layers.py`
- Migrate `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`
- Migrate parallel state management utilities
- **Effort**: 6-8 hours
- **Risk**: High (distributed communication, complex logic)

**Step 3.2**: Add unit tests for parallel layers
- Test TP=1, TP=2, TP=4 configurations
- Validate All-Reduce, All-Gather operations
- **Effort**: 2-3 hours
- **Risk**: Medium

**Step 3.3**: Update MLP/MoE implementations
- Update imports in `linear_op_impl.py` and `moe_impl.py`
- **Effort**: 30 minutes
- **Risk**: Medium

---

### Phase 4: Attention Backend System (Critical, High-Risk)

**Step 4.1**: Create `frontier/profiling/attention/sequence_metadata.py`
- Lightweight version of `SequenceMetadata`
- Only include fields used in profiling
- **Effort**: 1-2 hours
- **Risk**: Low

**Step 4.2**: Create `frontier/profiling/attention/backends/base.py`
- Migrate `BaseAttentionWrapper` abstract class
- Update imports to use local `SequenceMetadata`, `OperationMetrics`
- **Effort**: 1 hour
- **Risk**: Low

**Step 4.3**: Create `frontier/profiling/attention/backends/no_op.py`
- Migrate `NoOpAttentionWrapper`
- **Effort**: 30 minutes
- **Risk**: Minimal

**Step 4.4**: Create `frontier/profiling/attention/backends/flashinfer.py`
- Migrate `FlashinferAttentionWrapper`
- **Effort**: 2-3 hours
- **Risk**: High (complex FlashInfer integration)

**Step 4.5**: Create `frontier/profiling/attention/backends/__init__.py`
- Migrate `AttentionBackend` enum
- Migrate `set_attention_backend()`, `get_attention_wrapper()` factory
- **Effort**: 1 hour
- **Risk**: Low

**Step 4.6**: Update attention profiling files
- Update `attention_wrapper.py` to use local backends
- Update `main.py` to use local backends
- Update `sequence_proxy.py` to use local `SequenceMetadata`
- **Effort**: 1 hour
- **Risk**: Medium

---

### Phase 5: Validation & Cleanup

**Step 5.1**: Remove all Sarathi imports
- Search for `from sarathi` and `import sarathi` across profiling module
- Remove monkey patching of `CudaTimer`
- **Effort**: 1 hour
- **Risk**: Low

**Step 5.2**: Run profiling validation tests
- Profile same model/config before and after migration
- Compare CSV outputs (should be identical)
- **Effort**: 2-3 hours
- **Risk**: Medium

**Step 5.3**: Update documentation
- Update `README.md` in profiling module
- Document new directory structure
- Update backend extension guide
- **Effort**: 1-2 hours
- **Risk**: Minimal

---

## 6. Validation Strategy

### 6.1 Before/After Comparison Tests

**Test 1: Attention Profiling**
```bash
# Before migration
bash frontier/profiling/example/test_profiling_attn.sh \
  --device a100 --model meta-llama/Llama-2-7b-hf --attention-backend flashinfer

# After migration (should produce identical results)
bash frontier/profiling/example/test_profiling_attn.sh \
  --device a100 --model meta-llama/Llama-2-7b-hf --attention-backend flashinfer

# Compare outputs
diff data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention_before.csv \
     data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention_after.csv
```

**Test 2: Linear Op Profiling**
```bash
# Similar comparison for linear_op profiling
```

**Test 3: MoE Profiling**
```bash
# Similar comparison for MoE profiling
```

### 6.2 Unit Tests

**Test Suite 1: Parallel Layers**
- Test `ColumnParallelLinear` with TP=1, 2, 4
- Test `RowParallelLinear` with TP=1, 2, 4
- Validate All-Reduce, All-Gather correctness

**Test Suite 2: Attention Backends**
- Test `FlashinferAttentionWrapper` with various batch sizes
- Test `NoOpAttentionWrapper` returns correct shapes
- Test backend switching via factory

**Test Suite 3: Model Layers**
- Test `SiluAndMul` activation
- Test `RMSNorm` normalization
- Test RoPE embeddings

### 6.3 Integration Tests

**Test 1: End-to-End Attention Profiling**
- Run full attention profiling pipeline
- Verify CSV output format
- Check timing statistics are reasonable

**Test 2: End-to-End Linear Op Profiling**
- Run full linear_op profiling pipeline
- Verify all operations are profiled

**Test 3: End-to-End MoE Profiling**
- Run full MoE profiling pipeline
- Verify gating, shuffling, grouped GEMM

---

## 7. Answers to Key Questions

### Q1: Should we keep the same class/function names?

**Answer**: **YES, keep identical names**

**Rationale**:
- Minimizes code changes in profiling scripts
- Maintains consistency with Sarathi terminology (familiar to developers)
- Easier to compare implementations side-by-side
- Only change: module paths (e.g., `sarathi.model_executor.attention` → `frontier.profiling.attention.backends`)

**Exception**: Add `Frontier` prefix only if there's a naming conflict with existing Frontier code.

---

### Q2: Are there any Sarathi utilities (like CudaTimer) that should also be migrated or replaced?

**Answer**: **CudaTimer is already replaced ✅**

**Current Status**:
- `frontier/profiling/common/cuda_timer.py` already exists
- Currently using monkey patching to inject into Sarathi modules
- After migration, remove monkey patching and use directly

**Other Utilities**:
- `initialize_dummy_weights`: Migrate (trivial, 15 lines)
- Parallel state management: Migrate as part of tensor parallel layers
- No other critical utilities identified

---

### Q3: How should we handle future updates to Sarathi's attention implementations?

**Answer**: **Fork and maintain independently**

**Rationale**:
1. **Profiling needs are stable**: Attention profiling interface is unlikely to change significantly
2. **Frontier-specific optimizations**: We may want to add Frontier-specific features (e.g., new backends)
3. **Dependency isolation**: Avoid breaking changes from Sarathi updates

**Strategy**:
- Document the Sarathi version we forked from (commit hash)
- Monitor Sarathi releases for critical bug fixes or performance improvements
- Selectively backport important changes if needed
- Maintain our own test suite to catch regressions

**Future Backend Additions**:
- New backends (e.g., xFormers, vLLM PagedAttention) will be added to `frontier/profiling/attention/backends/`
- No dependency on Sarathi for new backends

---

### Q4: Should we maintain a compatibility shim for any existing code outside profiling that might depend on Sarathi?

**Answer**: **NO compatibility shim needed**

**Rationale**:
1. **Profiling is isolated**: No other Frontier modules depend on `frontier/profiling/`
2. **One-way dependency**: Profiling uses Frontier's `ModelConfig`, but Frontier doesn't use profiling
3. **Clean separation**: Profiling is a standalone tool, not a library

**Verification**:
```bash
# Check if any non-profiling code imports from profiling
grep -r "from frontier.profiling" frontier --exclude-dir=profiling
# Expected: No results (or only test files)
```

---

## 8. Risk Mitigation Strategies

### Risk 1: Profiling Results Change After Migration

**Mitigation**:
1. **Bit-exact validation**: Compare CSV outputs before/after migration
2. **Statistical validation**: If minor numerical differences exist, verify they're within acceptable tolerance (<0.1%)
3. **Regression test suite**: Create automated tests that run profiling and compare results

**Acceptance Criteria**:
- Timing statistics differ by <1% (acceptable due to system noise)
- Operation names and CSV structure identical
- No missing or extra operations

---

### Risk 2: Tensor Parallel Layers Behavior Differs

**Mitigation**:
1. **Copy implementation exactly**: Don't "improve" or refactor during migration
2. **Unit tests**: Test each TP size (1, 2, 4, 8) independently
3. **Gradient checking**: Verify forward/backward pass correctness (if applicable)

**Acceptance Criteria**:
- All-Reduce produces identical results to Sarathi version
- All-Gather produces identical results to Sarathi version
- Profiling times match within 1%

---

### Risk 3: FlashInfer Backend Breaks

**Mitigation**:
1. **Incremental testing**: Test with simple cases first (batch_size=1, small seq_len)
2. **Extensive profiling**: Run full profiling suite on multiple models
3. **Fail-fast validation**: If FlashInfer breaks, profiling should raise a clear backend error before collecting release data

**Acceptance Criteria**:
- FlashInfer profiling completes without errors
- Timing statistics are reasonable (not 0, not infinity)
- CSV output matches pre-migration format

---

## 9. Timeline Estimate

| Phase | Tasks | Estimated Time | Dependencies |
|-------|-------|----------------|--------------|
| **Phase 1: Foundation** | Constants, ParallelConfig, Utils | 2 hours | None |
| **Phase 2: Model Layers** | Activation, LayerNorm, RoPE | 5-6 hours | Phase 1 |
| **Phase 3: Parallel Layers** | TP layers, unit tests | 8-11 hours | Phase 1, 2 |
| **Phase 4: Attention Backends** | SequenceMetadata, backends, factory | 6-8 hours | Phase 1 |
| **Phase 5: Validation** | Tests, cleanup, docs | 4-6 hours | Phase 2, 3, 4 |
| **TOTAL** | - | **25-33 hours** | - |

**Recommended Approach**: **Incremental migration over 4-5 days**
- Day 1: Phase 1 + Phase 2 (foundation + model layers)
- Day 2: Phase 3 (tensor parallel layers)
- Day 3: Phase 4 (attention backends)
- Day 4: Phase 5 (validation + cleanup)
- Day 5: Buffer for unexpected issues

---

## 10. Success Criteria

### Must-Have (Blocking)
- ✅ Zero imports from `sarathi` in `frontier/profiling/`
- ✅ All existing profiling scripts run without modification
- ✅ Profiling results match pre-migration outputs (within 1% tolerance)
- ✅ All three profiling modules work: attention, MLP, MoE
- ✅ FlashInfer backend functional

### Should-Have (Important)
- ✅ Unit tests for all migrated components
- ✅ Documentation updated
- ✅ Clean directory structure
- ✅ No monkey patching

### Nice-to-Have (Optional)
- ✅ Performance improvements
- ✅ Additional backends (xFormers, vLLM)
- ✅ Improved error messages

---

## 11. Next Steps

**Awaiting Approval** for:
1. Proposed directory structure
2. Migration plan (phases 1-5)
3. Validation strategy

**After Approval**:
1. Create feature branch: `refactor/remove-sarathi-dependency`
2. Begin Phase 1 implementation
3. Incremental commits with validation at each phase
4. Final PR with comprehensive testing

---

## Appendix A: Import Dependency Graph

```
frontier/profiling/attention/main.py
├── sarathi.config.ParallelConfig → REPLACE with local
└── sarathi.model_executor.attention.AttentionBackend → MIGRATE

frontier/profiling/attention/attention_wrapper.py
├── sarathi.config.ParallelConfig → REPLACE with local
└── sarathi.model_executor.attention.* → MIGRATE
    ├── AttentionBackend
    ├── get_attention_wrapper
    └── set_attention_backend

frontier/profiling/common/model_config.py
└── sarathi.config.ParallelConfig → REPLACE with local

frontier/profiling/linear_op/linear_op_impl.py
├── sarathi.model_executor.layers.activation.SiluAndMul → MIGRATE
├── sarathi.model_executor.layers.layernorm.RMSNorm → MIGRATE
├── sarathi.model_executor.layers.rotary_embedding.get_rope → MIGRATE
└── sarathi.model_executor.parallel_utils.tensor_parallel.layers.* → MIGRATE
    ├── ColumnParallelLinear
    ├── RowParallelLinear
    └── VocabParallelEmbedding

frontier/profiling/linear_op/linear_op_wrapper.py
└── sarathi.model_executor.weight_utils.initialize_dummy_weights → MIGRATE

frontier/profiling/moe/moe_impl.py
├── sarathi.model_executor.layers.activation.SiluAndMul → MIGRATE
└── sarathi.model_executor.parallel_utils.tensor_parallel.layers.* → MIGRATE
    ├── ColumnParallelLinear
    └── RowParallelLinear

frontier/profiling/utils/__init__.py
└── sarathi.config.ParallelConfig → REPLACE with local

frontier/profiling/cpu_overhead/benchmark_runner.py (DEFER)
├── sarathi.LLMEngine → DEFER
├── sarathi.SamplingParams → DEFER
└── sarathi.metrics.constants.CpuOperationMetrics → DEFER
```

---

---

## Appendix B: File Migration Mapping Table

| Sarathi Source File | Frontier Target File | Lines | Status |
|---------------------|---------------------|-------|--------|
| `sarathi/types.py` (AttentionBackend) | `frontier/profiling/attention/backends/__init__.py` | 5 | To Migrate |
| `sarathi/model_executor/attention/__init__.py` | `frontier/profiling/attention/backends/__init__.py` | 35 | To Migrate |
| `sarathi/model_executor/attention/base_attention_wrapper.py` | `frontier/profiling/attention/backends/base.py` | 69 | To Migrate |
| `sarathi/model_executor/attention/flashinfer_attention_wrapper.py` | `frontier/profiling/attention/backends/flashinfer.py` | 259 | To Migrate |
| `sarathi/model_executor/attention/no_op_attention_wrapper.py` | `frontier/profiling/attention/backends/no_op.py` | 46 | To Migrate |
| `sarathi/core/datatypes/sequence.py` (SequenceMetadata) | `frontier/profiling/attention/sequence_metadata.py` | 75 | To Simplify & Migrate |
| `sarathi/metrics/constants.py` (OperationMetrics) | `frontier/profiling/common/constants.py` | 30 | To Migrate |
| `sarathi/config/config.py` (ParallelConfig) | `frontier/profiling/common/parallel_config.py` | 10 | To Create New |
| `sarathi/model_executor/layers/activation.py` | `frontier/profiling/common/layers/activation.py` | 61 | To Migrate |
| `sarathi/model_executor/layers/layernorm.py` | `frontier/profiling/common/layers/layernorm.py` | 40 | To Migrate |
| `sarathi/model_executor/layers/rotary_embedding.py` | `frontier/profiling/common/layers/rotary_embedding.py` | 344 | To Migrate |
| `sarathi/model_executor/parallel_utils/tensor_parallel/layers.py` | `frontier/profiling/common/parallel_layers.py` | 461 | To Migrate |
| `sarathi/model_executor/weight_utils.py` (initialize_dummy_weights) | `frontier/profiling/common/utils.py` | 15 | To Migrate |
| `sarathi/metrics/cuda_timer.py` | `frontier/profiling/common/cuda_timer.py` | N/A | Already Exists ✅ |

---

## Appendix C: Import Statement Changes

### Before Migration

```python
# frontier/profiling/attention/main.py
from sarathi.config import ParallelConfig
from sarathi.model_executor.attention import AttentionBackend

# frontier/profiling/attention/attention_wrapper.py
import sarathi.metrics.cuda_timer
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer  # Monkey patch
from sarathi.config import ParallelConfig
from sarathi.model_executor.attention import (
    AttentionBackend,
    get_attention_wrapper,
    set_attention_backend,
)

# frontier/profiling/linear_op/linear_op_impl.py
from sarathi.model_executor.layers.activation import SiluAndMul
from sarathi.model_executor.layers.layernorm import RMSNorm
from sarathi.model_executor.layers.rotary_embedding import get_rope
from sarathi.model_executor.parallel_utils.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

# frontier/profiling/linear_op/linear_op_wrapper.py
import sarathi.metrics.cuda_timer
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer  # Monkey patch
from sarathi.model_executor.weight_utils import initialize_dummy_weights
```

### After Migration

```python
# frontier/profiling/attention/main.py
from frontier.profiling.common.parallel_config import ParallelConfig
from frontier.profiling.attention.backends import AttentionBackend

# frontier/profiling/attention/attention_wrapper.py
from frontier.profiling.common.parallel_config import ParallelConfig
from frontier.profiling.attention.backends import (
    AttentionBackend,
    get_attention_wrapper,
    set_attention_backend,
)

# frontier/profiling/linear_op/linear_op_impl.py
from frontier.profiling.common.layers.activation import SiluAndMul
from frontier.profiling.common.layers.layernorm import RMSNorm
from frontier.profiling.common.layers.rotary_embedding import get_rope
from frontier.profiling.common.parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

# frontier/profiling/linear_op/linear_op_wrapper.py
from frontier.profiling.common.utils import initialize_dummy_weights
```

---

**End of Report**

# Linear Operations Profiling - Q&A Documentation

**Date**: December 8, 2025  
**Module**: `frontier/profiling/linear_op`  
**Purpose**: Verify profiling behavior and implementation details

---

## Question 1: Profiling Granularity - Single Layer vs. Multiple Layers

**Q**: When the profiling module measures each operation, does it measure:
- (A) n layers (total number of layers in the model)?
- (B) A single layer?
- (C) Single layer × num_layers?

**A**: **(B) Single layer**

### Evidence
- **File**: `frontier/profiling/linear_op/linear_op_impl.py`, lines 201-213
  - `GPTModel` creates **ONE `GPTBlock` instance**
  
- **File**: `frontier/profiling/linear_op/linear_op_impl.py`, lines 215-227
  - Forward loop repeats the same single `GPTBlock` `num_repeat_steps` times
  - Default: 20 iterations for RECORD_FUNCTION, 1 for other methods
  - Purpose: Averaging measurements, not profiling multiple layers

### Code Snippet
```python
class GPTModel(torch.nn.Module):
    def __init__(self, config: ModelConfig, world_size: int, num_repeat_steps: int = 1):
        self.block = GPTBlock(config, world_size=world_size)  # Single block

    def forward(self, input_ids, positions):
        for _ in range(self.num_repeat_steps):  # Repeat same block
            hidden_states = self.block(positions, hidden_states, residual)
        return hidden_states
```

---

## Question 2: CPU Time vs. GPU Time in Profiling Measurements

**Q**: Do measurements include:
- (A) CPU time only?
- (B) GPU time only?
- (C) Both CPU and GPU time?

**A**: **(B) GPU time only** (for RECORD_FUNCTION method)

### Evidence
- **File**: `frontier/profiling/utils/record_function_tracer.py`, lines 63-92
  - Only `cuda_time` is extracted and recorded
  - CPU activities captured but filtered out
  
- **Extraction Logic**: Only `cuda_runtime` and `cuda_driver` events are summed
  
### Code Snippet
```python
def get_operation_time_stats(self, debug=False):
    cuda_time = 0
    for child in children:
        # Only CUDA events are processed
        if not ("cat" in child and child["cat"] in ("cuda_runtime", "cuda_driver")):
            continue
        correlated_event = self.find_correlated_event(trace, child)
        if not correlated_event:
            continue
        cuda_time += correlated_event["dur"]  # Only CUDA kernel duration
    
    stats[name].append(cuda_time * 1e-3)  # Record GPU time in ms
```

### Alternative Methods
| Method | Timing Source |
|--------|---------------|
| RECORD_FUNCTION | GPU (CUDA kernel) time |
| CUDA_EVENT | GPU time (`torch.cuda.Event`) |
| KINETO | GPU (`cuda_time_total`) |
| PERF_COUNTER | Wall-clock time (CPU + GPU) with `torch.cuda.synchronize()` |

---

## Question 3: Fused Residual Addition in vLLM input_layernorm

**Q**: Does `input_layernorm` fuse residual addition?

**A**: **Yes, (A) Inside the function itself**

### Evidence
- **File**: `sota-infer-engine/vllm/vllm/model_executor/models/llama.py`, lines 319-327
  - Residual passed as parameter to `input_layernorm`
  
- **File**: `sota-infer-engine/vllm/vllm/model_executor/layers/layernorm.py`, lines 211-223
  - Calls `fused_add_rms_norm` when residual is provided
  - Both operations performed in single CUDA kernel

### Code Snippet
```python
# Caller (llama.py:325)
hidden_states, residual = self.input_layernorm(hidden_states, residual)

# Implementation (layernorm.py:211-223)
def forward_cuda(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
    add_residual = residual is not None
    if add_residual:
        return fused_add_rms_norm(x, residual, self.weight.data,
                                  self.variance_epsilon)
    else:
        return rms_norm(x, self.weight.data, self.variance_epsilon)
```

### Fused Kernel Call
```python
def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                      variance_epsilon: float):
    from vllm import _custom_ops as ops
    ops.fused_add_rms_norm(x, residual, weight, variance_epsilon)
    return x, residual
```

---

## Question 4: PyTorch Native API vs. vLLM Custom Operation

**Q**: Does `input_layernorm` call:
- (A) PyTorch native API?
- (B) vLLM custom operation?

**A**: **(B) vLLM custom CUDA kernel**

### Evidence Trace

**Step 1: Initialization** (`llama.py`, lines 308-309)
```python
self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

**Step 2: Class Definition** (`sota-infer-engine/vllm/vllm/model_executor/layers/layernorm.py`, lines 128-167)
```python
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Uses vLLM custom CUDA operations, not PyTorch native."""
```

**Step 3: CUDA Forward Method** (`layernorm.py`, lines 211-223)
```python
def forward_cuda(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
    if add_residual:
        return fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
    else:
        return rms_norm(x, self.weight.data, self.variance_epsilon)
```

**Step 4: Custom Kernel Invocation** (`layernorm.py`, lines 21-29)
```python
def rms_norm(x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float):
    from vllm import _custom_ops as ops
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, variance_epsilon)  # <-- Custom CUDA kernel
    return out
```

### Conclusion
- **NOT** `torch.nn.LayerNorm` or `F.layer_norm`
- **IS** `vllm._custom_ops.rms_norm` and `vllm._custom_ops.fused_add_rms_norm`
- **Benefits**: Fused operations, better performance, reduced memory bandwidth

---

## Summary Table

| Question | Answer | Key File & Lines |
|----------|--------|------------------|
| Q1: Granularity | Single layer | `linear_op_impl.py:201-227` |
| Q2: CPU vs GPU | GPU only | `record_function_tracer.py:63-92` |
| Q3: Fused Residual | Yes (inside) | `layernorm.py:211-223` |
| Q4: Native vs Custom | Custom kernel | `layernorm.py:21-29` |

---

## Key Takeaways

1. **Profiling measures single layer execution** repeated 20 times for averaging
2. **GPU time only** is captured (via CUDA kernel events)
3. **vLLM fuses residual addition** into LayerNorm kernel for efficiency
4. **Custom CUDA kernels** are used for better performance, not PyTorch native ops

---

## Related Documentation

- `ROPE_LLAMA3_FIX.md`: Llama 3.x RoPE scaling implementation details
- `README.md`: Linear operations profiling overview
- `PROFILING_NAN_FIX_REPORT.md`: NaN handling in profiling

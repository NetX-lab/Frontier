# Linear Op Profiling NaN 值问题修复报告

## 1. 问题描述

### 1.1 现象
在运行 `test_profiling_linear_op.sh` 进行 linear op profiling 后，执行模型训练时出现以下错误：

```
ValueError: Input y contains NaN.
```

### 1.2 影响范围
- **受影响的操作**: `mlp_up_proj`, `attn_post_proj`
- **NaN 值分布**:
  - `mlp_up_proj`: 在 `num_tokens` 248-512 范围内出现 29 个 NaN 值
  - `attn_post_proj`: 在 `num_tokens` 96-192 和 648-768 范围内出现 23 个 NaN 值

### 1.3 数据示例
```csv
num_tokens,time_stats
8,0.051869
16,0.051358
...
248,nan
256,nan
...
512,nan
```

---

## 2. 问题根因分析

### 2.1 分析过程

通过对 PyTorch Profiler 生成的 Chrome Trace JSON 文件进行深入分析，发现了问题的根本原因。

#### 2.1.1 正常工作的操作 (`mlp_down_proj`)

```json
// user_annotation 事件
{
  "name": "vidur_mlp_down_proj",
  "cat": "user_annotation",
  "ts": 5743218985307.825,
  "dur": 60.863
}

// 在时间范围内的 cudaLaunchKernel (cuda_runtime)
{
  "name": "cudaLaunchKernel",
  "cat": "cuda_runtime",
  "ts": 5743218985344.071,
  "args": {"correlation": 57863}
}

// 关联的 GPU kernel
{
  "name": "ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_tn",
  "cat": "kernel",
  "dur": 113.952,
  "args": {"correlation": 57863}
}
```

#### 2.1.2 出问题的操作 (`mlp_up_proj`)

```json
// user_annotation 事件
{
  "name": "vidur_mlp_up_proj",
  "cat": "user_annotation",
  "ts": 5743218985133.403,
  "dur": 68.354
}

// 在时间范围内的 cuLaunchKernel (cuda_driver，注意不是 cuda_runtime!)
{
  "name": "cuLaunchKernel",
  "cat": "cuda_driver",  // <-- 关键差异！
  "ts": 5743218985174.967,
  "args": {"correlation": 57819}
}

// 关联的 GPU kernel
{
  "name": "void cutlass::Kernel2<cutlass_80_tensorop_f16_s168...",
  "cat": "kernel",
  "dur": 248.127,
  "args": {"correlation": 57819}
}
```

### 2.2 根本原因

**PyTorch/CUDA 使用两种不同的 API 来启动 GPU kernels**：

| API 类型 | 函数名 | Profiler 类别 | 级别 |
|---------|--------|--------------|------|
| CUDA Runtime API | `cudaLaunchKernel` | `cuda_runtime` | 高级 API |
| CUDA Driver API | `cuLaunchKernel` | `cuda_driver` | 低级 API |

原始的 `record_function_tracer.py` 只检查了 `cuda_runtime` 类别的事件：

```python
# 原始代码 (有问题)
if not ("cat" in child and child["cat"] == "cuda_runtime"):
    continue
```

这导致使用 CUDA Driver API (`cuLaunchKernel`) 启动的 kernel **无法被正确关联**，其执行时间被计算为 0，最终在 CSV 中表现为 NaN 值。

### 2.3 为什么不同操作使用不同的 API？

PyTorch 内部使用的 CUDA kernel launch 方式可能取决于：
1. **Kernel 类型**: 某些 optimized kernels (如 CUTLASS) 可能直接使用 Driver API
2. **输入大小**: 不同的 `num_tokens` 可能触发不同的代码路径
3. **JIT 编译**: 某些 kernels 在运行时 JIT 编译，可能使用不同的 launch 机制

---

## 3. 解决方案

### 3.1 代码修改

**文件**: `frontier/profiling/utils/record_function_tracer.py`

**修改位置**: `get_operation_time_stats()` 方法，第 70-72 行

#### 修改前：
```python
for child in children:
    if not ("cat" in child and child["cat"] == "cuda_runtime"):
        continue
```

#### 修改后：
```python
for child in children:
    # Check for both cuda_runtime (cudaLaunchKernel) and cuda_driver (cuLaunchKernel)
    if not ("cat" in child and child["cat"] in ("cuda_runtime", "cuda_driver")):
        continue
```

### 3.2 修复原理

通过将检查条件从只接受 `cuda_runtime` 扩展为同时接受 `cuda_runtime` 和 `cuda_driver`，可以正确捕获所有 kernel launch 事件，无论它们使用哪种 CUDA API 启动。

---

## 4. 验证结果

### 4.1 修复前

```
Operations with cuda_time > 0:
  add: 20 samples
  attn_pre_proj: 20 samples
  mlp_act: 20 samples
  mlp_down_proj: 20 samples
  ...

Expected operations status:
  mlp_up_proj: ✗ MISSING
  attn_post_proj: ✗ MISSING (部分)
```

### 4.2 修复后

```
Operations with cuda_time > 0:
  add: 20 samples, mean=0.006ms
  attn_post_proj: 20 samples, mean=0.045ms
  attn_pre_proj: 20 samples, mean=0.067ms
  attn_rope: 20 samples, mean=0.060ms
  emb: 21 samples, mean=0.005ms
  input_layernorm: 20 samples, mean=0.044ms
  mlp_act: 20 samples, mean=0.030ms
  mlp_down_proj: 20 samples, mean=0.114ms
  mlp_up_proj: 20 samples, mean=0.249ms
  post_attention_layernorm: 20 samples, mean=0.043ms

Expected operations status:
  mlp_up_proj: ✓ CAPTURED
  mlp_down_proj: ✓ CAPTURED
  mlp_act: ✓ CAPTURED
  attn_pre_proj: ✓ CAPTURED
  attn_post_proj: ✓ CAPTURED
  attn_rope: ✓ CAPTURED
```

---

## 5. 相关修复历史

### 5.1 `--disable_ray` 标志修复

在同一次调试过程中，还修复了 `--disable_ray` 标志无法正常工作的问题。

**问题**: 设置 `--disable_ray` 后仍然尝试初始化 Ray，导致在没有 Ray 环境的机器上无法运行。

**解决方案**: 在 `frontier/profiling/linear_op/main.py` 中实现了基于 `ProcessPoolExecutor` 的非 Ray 模式：

```python
def _worker_init(gpu_id: int):
    """初始化 worker 进程，绑定到指定 GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

def _worker_profile_linear_op_task(args):
    """Worker 函数，在指定 GPU 上执行 profiling 任务"""
    # ... profiling 逻辑
```

### 5.2 多 GPU Worker 绑定修复

**问题**: 使用 `ProcessPoolExecutor` 时，所有 worker 进程都在同一个 GPU 上运行。

**解决方案**: 为每个 GPU 创建独立的 executor，并使用 `initializer` 参数确保 worker 正确绑定到指定 GPU：

```python
for gpu_id in gpu_ids:
    with ProcessPoolExecutor(
        max_workers=1,
        initializer=_worker_init,
        initargs=(gpu_id,)
    ) as executor:
        # 提交任务到这个 GPU
```

---

## 6. 建议

### 6.1 后续改进

1. **添加更多 CUDA 事件类型**: 如果未来发现其他 CUDA API 类型，应继续扩展检查条件。

2. **增加诊断日志**: 在遇到 `cuda_time == 0` 时，记录详细的诊断信息，便于快速定位问题。

3. **单元测试**: 为 `RecordFunctionTracer` 添加单元测试，验证其能正确处理不同类型的 kernel launch 事件。

### 6.2 调试技巧

如需调试类似问题，可使用以下方法分析 Chrome Trace JSON：

```python
import json

trace = json.load(open("profiler_trace_xxx.json"))["traceEvents"]

# 查找特定操作的事件
op_events = [e for e in trace if e.get("name", "").startswith("vidur_mlp_up_proj")]

# 分析时间范围内的子事件
for event in op_events:
    start, end = event["ts"], event["ts"] + event["dur"]
    children = [e for e in trace 
                if e.get("ts", 0) > start 
                and e.get("ts", 0) + e.get("dur", 0) < end]
    
    for child in children:
        print(f"  [{child.get('cat')}] {child.get('name')}")
```

---

## 7. 总结

| 项目 | 详情 |
|-----|------|
| **问题类型** | Profiler 数据收集不完整 |
| **根本原因** | 只检查 `cuda_runtime` 类别，遗漏了 `cuda_driver` 类别的 kernel launch 事件 |
| **影响** | 部分操作的 CUDA 时间为 0，导致训练数据包含 NaN |
| **修复文件** | `frontier/profiling/utils/record_function_tracer.py` |
| **修复行数** | 1 行代码修改 |
| **验证状态** | ✓ 已验证，所有操作均正确捕获 |

---

*文档生成日期: 2025-12-07*

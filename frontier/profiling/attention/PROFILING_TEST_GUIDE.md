# Mixed Batch Prefill Profiling 测试指南

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-13 | Added deterministic canonical/alias output, run-scoped sidecar checks, explicit block allocation provenance, and measured KV-grid start guidance |
| 2026-08-12 | Separated serving `max_model_len` from profiling `max_seq_len`/`profile_max_seq_len`; documented strict TP sizes and profile-context validation |
| 2026-06-06 | Replaced private checkout path with a repo-relative profiling example; refreshed legacy CSV naming note for release docs |
| 2026-01-16 | Documented `--batch_size_list` for explicit batch-size profiling control.          |
| 2025-12-06 | Added multi-GPU mode documentation; updated CSV naming from legacy MLP naming to `linear_op.csv`; added --disable_ray usage; updated --compute_dataset_path as optional |
| 2025-11-09 | Initial documentation for mixed batch prefill profiling                            |

---

## 🔧 环境要求

- **FlashInfer**: `flashinfer-python==0.3.0`

---

## 📋 测试命令和参数详解

本文档提供详细的测试命令、参数说明和测试场景，帮助验证 mixed batch prefill profiling 功能的正确性。

## Output/provenance assertions

一个成功的 attention run 应同时满足：

1. `attention.csv`（或 `attention_kernel_only.csv`）是 standard/mixed/true-mixed 的 deterministic union；
2. `attention_combined*.csv` 与 canonical 文件 bytes 完全相同，不能把它当作第二份 supplement；
3. `runs/<run_id>/` 下保留本次不可覆盖的 CSV，sidecar 的 `csv_sha256` 必须匹配该文件；
4. sidecar 中 `physical_max_num_blocks`、`selected_max_num_blocks`、`required_max_num_blocks` 是正整数，`requested_max_num_blocks` 只能是正整数或 JSON `null`；
5. requested/emitted structural tuple multiset 完全相等，不能有 missing、extra 或重复；
6. `TP` 只允许 `{1, 2, 4, 8}`（communication-only / dummy 例外）。

若 serving `max_model_len` 小于显式 `profile_max_seq_len`，仍可 profiling；后者必须有实际 GPU capacity。runtime materialization 的每个 tuple 仍须落在模型 sidecar/domain 声明范围内，不允许 nearest、clamp 或 generic predict-on-miss。

---

## 🚀 快速测试命令

### Training 命令示例

```bash
# Full training (with compute dataset)
python -m frontier.training.cli attention \
    --compute_dataset_path path/to/linear_op.csv \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./testing_training/models \
    --model_name "Qwen/Qwen2.5-7B" \
    --device h100 \
    --tensor_parallel_size 1 \
    --block_size 16 \
    --predictor_type random_forest

# Attention-only training (without compute dataset)
python -m frontier.training.cli attention \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./testing_training/models \
    --model_name "Qwen/Qwen2.5-7B" \
    --device h100 \
    --tensor_parallel_size 1
```

### 1. 最简单的测试（推荐入门）

```bash
cd /path/to/frontier

# 最简单的测试 (使用 --disable_ray 避免 Ray 兼容性问题)
python -m frontier.profiling.attention.main \
    --disable_ray \
    --enable_mixed_prefill \
    --mixed_mode random \
    --models "meta-llama/Llama-2-7b-hf" \
    --num_gpus 1 \
    --max_seq_len 4096
```

## 核心参数

### Mixed Batch 专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_mixed_prefill` | False | 启用混合 batch profiling |
| `--mixed_mode` | even | even: 相同长度<br>random: 随机长度<br>both: 两者都测 |
| `--max_mixed_batch_size` | 8 | 混合 batch 最大大小 |
| `--mixed_num_samples` | 3 | Random 模式每个配置的样本数 |
| `--max_pipeline_parallel_size` | 8 | Pipeline 并行度（影响内存计算）<br>**注意**: 模型层数需能被此值整除 |

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--models` | 多个 | 模型列表，如 "meta-llama/Llama-2-7b-hf" |
| `--num_gpus` | 8 | GPU 数量 |
| `--num_tensor_parallel_workers` | [1,2,4,8] | Tensor 并行度列表；profile-backed compute 只支持 `1/2/4/8`，其他值在 GPU 初始化前拒绝 |
| `--max_seq_len` | 4096 | 本次 profiling 生成的输入 grid 最大序列长度；必须 `<= profile_max_seq_len` |
| `--profile_max_seq_len` | `max_seq_len` | Profiling workload 的 context 上限，用于 input validity 和 KV block-table 安全检查；可以大于 serving 的 `max_model_len`，但不能超过实际 backend/显存可执行范围 |
| `--max_model_len` | 4096 | Serving context metadata；不是 profiling 的硬上限，可小于 `max_seq_len` |
| `--min_batch_size` | 1 | 最小 decode batch size |
| `--max_batch_size` | 128 | 最大 decode batch size |
| `--batch_size_list` | None | 显式指定 batch size 列表（覆盖 min/max） |
| `--output_dir` | data/profiling | Root output directory; the CLI writes `compute/<device>/<model_name>/` under this root |
| `--block_size` | 16 | KV cache block 大小 |
| `--attention_backend` | flashinfer | Attention 后端 |

### 模式控制

| 参数 | 说明 |
|------|------|
| `--profile_only_prefill` | 只 profile prefill |
| `--profile_only_decode` | 只 profile decode |
| `--disable_ray` | 禁用 Ray（调试用） |

## 测试场景

### 场景 1: 快速验证

**目的**: 验证基本功能

```bash
python -m frontier.profiling.attention.main \
    --enable_mixed_prefill \
    --mixed_mode random \
    --max_mixed_batch_size 4 \
    --mixed_num_samples 2 \
    --models "microsoft/phi-2" \
    --num_gpus 1 \
    --max_seq_len 2048 \
    --max_model_len 2048 \
    --profile_only_prefill
```

**预期时间**: 2-3 分钟  
**检查点**:
- ✅ 生成 `attention_mixed.csv`
- ✅ CSV 包含 `is_mixed_batch=True` 行
- ✅ 包含 `seq_lens`, `seq_len_cv` 等字段

### 场景 2: Even vs Random 对比

**目的**: 对比两种模式

```bash
python -m frontier.profiling.attention.main \
    --enable_mixed_prefill \
    --mixed_mode both \
    --max_mixed_batch_size 8 \
    --models "meta-llama/Llama-2-7b-hf" \
    --num_gpus 1 \
    --max_seq_len 4096 \
    --max_model_len 4096 \
    --num_tensor_parallel_workers 2 \
    --output_dir ./testing_profiling/attention/
```

**预期时间**: 10-15 分钟  
**分析**:
```python
import pandas as pd
df = pd.read_csv("attention_mixed.csv")
even = df[df['mode']=='even']['time_stats.attn_prefill.median'].mean()
random = df[df['mode']=='random']['time_stats.attn_prefill.median'].mean()
print(f"Overhead: {(random/even-1)*100:.1f}%")
```

### 场景 3: 大规模测试

**目的**: 全面测试

```bash
python -m frontier.profiling.attention.main \
    --enable_mixed_prefill \
    --mixed_mode random \
    --max_mixed_batch_size 64 \
    --mixed_num_samples 5 \
    --models "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" \
    --num_gpus 2 \
    --num_tensor_parallel_workers 1 2 \
    --max_seq_len 8192 \
    --max_model_len 8192
```

**预期时间**: 30-60 分钟

### 场景 4: 特殊模型（Qwen2.5-7B）

**问题**: 28 层模型，默认 `max_pipeline_parallel_size=8` 会报错

```bash
# ❌ 错误: 28 % 8 != 0
python -m frontier.profiling.attention.main \
    --enable_mixed_prefill \
    --models "Qwen/Qwen2.5-7B" \
    --num_gpus 1

# ✅ 正确: 使用 4 或 7
python -m frontier.profiling.attention.main \
    --enable_mixed_prefill \
    --mixed_mode random \
    --max_mixed_batch_size 8 \
    --max_pipeline_parallel_size 4 \
    --models "Qwen/Qwen2.5-7B" \
    --num_gpus 1 \
    --max_seq_len 6400 \
    --max_model_len 12800
```

## 结果验证

### 1. 检查输出文件

```bash
# 查看目录结构
ls -lh data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/

# 应包含:
# - attention.csv (标准)
# - attention_mixed.csv (混合)
# - attention_combined.csv (合并)
```

### 2. 验证 CSV 格式

```bash
# 查看列名
head -n 1 attention_mixed.csv | tr ',' '\n' | grep -E "(mixed|seq_len|mode)"

# 应包含:
# - is_mixed_batch
# - mode
# - seq_lens
# - total_tokens
# - min_seq_len, max_seq_len, avg_seq_len
# - seq_len_variance, seq_len_std, seq_len_cv
```

### 3. 验证数据正确性

```python
import pandas as pd
import numpy as np

df = pd.read_csv("attention_mixed.csv")

# 检查 is_mixed_batch 标记
assert df['is_mixed_batch'].all(), "All rows should have is_mixed_batch=True"

# 检查 mode
assert df['mode'].isin(['even', 'random']).all()

# 检查数值一致性
for _, row in df.iterrows():
    seq_lens = eval(row['seq_lens'])
    assert row['total_tokens'] == sum(seq_lens)
    assert row['min_seq_len'] == min(seq_lens)
    assert row['max_seq_len'] == max(seq_lens)
    assert abs(row['avg_seq_len'] - np.mean(seq_lens)) < 0.01
    assert abs(row['seq_len_cv'] - np.std(seq_lens)/np.mean(seq_lens)) < 0.01

print("✅ All checks passed!")
```

## Training 测试

### 基础训练

```bash
python -m frontier.training.cli attention \
    --compute_dataset_path path/to/linear_op.csv \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./cache/models \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --device a100 \
    --tensor_parallel_size 1
```

### 检查训练结果

```bash
# 1. 查看日志
# 应包含:
# "Training mixed-batch prefill model with XXX samples"
# "Mixed-batch prefill model trained successfully"

# 2. 检查模型文件
ls cache/models/meta-llama/Llama-2-7b-hf/a100/tp_1/attn_prefill_mixed.pkl

# 3. 验证模型可加载
python -c "
import pickle
with open('cache/models/.../attn_prefill_mixed.pkl', 'rb') as f:
    model = pickle.load(f)
print(f'Model type: {type(model)}')
print(f'Features: {model.n_features_in_}')
"
```

## 常见问题排查

### 问题 1: AssertionError: num_layers % max_pipeline_parallel_size == 0

**原因**: 模型层数不能被 `max_pipeline_parallel_size` 整除

**解决**:
```bash
# 查看模型层数
python -c "
from frontier.config.model_config import BaseModelConfig
config = BaseModelConfig.create_from_name('Qwen/Qwen2.5-7B')
print(f'Layers: {config.num_layers}')
"

# 使用合适的值（如 28 层用 4 或 7）
--max_pipeline_parallel_size 4
```

### 问题 2: AssertionError: prefill_chunk_size + kv_cache_size > profile_max_seq_len

**原因**: 生成的 prefill 输入组合超过了声明的 profiling context。`max_seq_len` 控制生成的输入 grid，`profile_max_seq_len` 控制 wrapper 的 validity 和 KV block-table 检查；`max_model_len` 只记录当前 serving context，不再自动限制 profiling 范围。

**解决**:
```bash
# 让生成 grid 不超过 profiling context；serving context 可以独立设置
--max_seq_len 6400 --profile_max_seq_len 12800 --max_model_len 4096
```

只有在模型 backend、GPU KV-cache capacity 和实际 kernel 都能执行该 profile workload 时，才应使用大于 serving context 的 profiling 范围。若超过可分配的 `max_num_blocks`，profiling 会明确失败，不会静默截断或降级。

### 问题 3: KeyError: 'time_stats.attn_prefill_mixed.median'

**原因**: CSV 中使用 `attn_prefill` 而不是 `attn_prefill_mixed`

**解决**: 已在 `attention_trainer.py` 中修复，无需操作

### 问题 4: 无 mixed 数据生成

**检查**:
```bash
# 1. 确认启用了 --enable_mixed_prefill
# 2. 检查日志中是否有 "Profiling mixed prefill"
# 3. 验证 attention_mixed.csv 是否为空
wc -l attention_mixed.csv  # 应 > 1
```

## 性能基准参考

基于 Llama-2-7B, A100, TP=1:

| Batch | Seq Lens | Mode | Time (ms) | 备注 |
|-------|----------|------|-----------|------|
| 1 | [512] | - | 0.5 | 单序列 |
| 4 | [512]*4 | even | 2.1 | Baseline |
| 4 | [256,512,768,1024] | random | 2.3 | +9.5% |
| 8 | [1024]*8 | even | 8.5 | Baseline |
| 8 | [512...2304] | random | 9.2 | +8.2% |

## 完整测试流程

```bash
# 1. Profiling
python -m frontier.profiling.attention.main \
    --enable_mixed_prefill \
    --mixed_mode both \
    --max_mixed_batch_size 8 \
    --mixed_num_samples 3 \
    --models "meta-llama/Llama-2-7b-hf" \
    --num_gpus 1 \
    --max_seq_len 4096 \
    --max_model_len 4096 \
    --output_dir ./test_profiling

# 2. 验证数据
python -c "
import pandas as pd
df = pd.read_csv('./test_profiling/attention/.../attention_combined.csv')
print(f'Total rows: {len(df)}')
print(f'Mixed rows: {df[\"is_mixed_batch\"].sum()}')
print(f'Modes: {df[df[\"is_mixed_batch\"]][\"mode\"].value_counts()}')
"

# 3. Training
python -m frontier.training.cli attention \
    --compute_dataset_path ./test_profiling/.../linear_op.csv \
    --layer_dataset_path ./test_profiling/.../attention_combined.csv \
    --output_dir ./test_cache \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --device a100 \
    --tensor_parallel_size 1

# 4. 检查模型
ls -lh ./test_cache/meta-llama/Llama-2-7b-hf/a100/tp_1/*.pkl
```

## 相关文档

- `README_MIXED_BATCH.md` - 功能详细说明
- `mixed_attention_input.py` - 数据结构定义
- `attention_trainer.py` - 训练实现

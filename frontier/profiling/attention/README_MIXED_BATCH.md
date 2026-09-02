# Mixed-Length Batch Prefilling Support

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-06-06 | Replaced private checkout path with a repo-relative profiling example; refreshed legacy CSV naming note for release docs |
| 2026-02-22 | Added true mixed (`--enable_true_mixed`) full CLI parameter set, output files, and mixed dataset fail-fast input contract notes |
| 2025-12-06 | Added multi-GPU mode documentation; updated CSV naming from legacy MLP naming to `linear_op.csv`; added --disable_ray usage; documented optional --compute_dataset_path |
| 2025-11-09 | Initial documentation for mixed-length batch prefilling support                    |

---

## 环境要求

- **FlashInfer**: `flashinfer-python==0.3.0`

---

## 概述

支持混合长度 batch 的 attention prefill profiling 和预测，更贴近真实 serving 场景。

## 核心特性

### Profiling
- 支持单个 batch 中多个不同长度的序列
- **Even 模式**: 所有序列相同长度（baseline）
- **Random 模式**: 序列长度随机分布（真实混合）
- **True Mixed 模式**: 同一 batch 内同时包含 prefill + decode（`--enable_true_mixed`）
- 收集长度分布统计（方差、变异系数等）
- 与原有功能完全兼容
- **Multi-GPU 支持**: 使用 `--disable_ray` 启用 multiprocessing 模式

### Prediction
- 专用 Random Forest 预测器 (`attn_prefill_mixed`)
- 11 个特征：batch_size, total_tokens, avg/min/max_seq_len, variance, CV 等
- 若缺少 mixed predictor，则明确报错并提示先收集 mixed profiling 数据

## 快速开始

### 1. Profiling

```bash
cd /path/to/frontier

# 基础测试 (使用 --disable_ray 避免 Ray 兼容性问题)
python -m frontier.profiling.attention.main \
    --disable_ray \
    --enable_mixed_prefill \
    --enable_true_mixed \
    --mixed_mode random \
    --max_mixed_batch_size 8 \
    --mixed_num_samples 3 \
    --true_mixed_prefill_batch_sizes 1 2 4 \
    --true_mixed_prefill_chunk_sizes 64 128 256 512 1024 \
    --true_mixed_decode_batch_sizes 1 2 4 8 \
    --true_mixed_decode_kv_cache_sizes 128 256 512 1024 2048 \
    --true_mixed_prefill_kv_cache_size 0 \
    --models "meta-llama/Llama-2-7b-hf" \
    --num_gpus 1 \
    --max_seq_len 4096 \
    --max_model_len 4096

# Multi-GPU profiling
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m frontier.profiling.attention.main \
    --disable_ray \
    --num_gpus 4 \
    --enable_mixed_prefill \
    --enable_true_mixed \
    --mixed_mode random \
    --models "meta-llama/Llama-2-7b-hf" \
    --max_seq_len 4096

# 输出文件:
# - attention.csv (标准)
# - attention_mixed.csv (混合)
# - attention_true_mixed.csv (prefill+decode 同批)
# - attention_combined.csv (合并)
```

### 2. Training

```bash
# Full training (with compute dataset - trains all 10 models)
python -m frontier.training.cli attention \
    --compute_dataset_path path/to/linear_op.csv \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./cache/models \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --device a100 \
    --tensor_parallel_size 1

# Attention-only training (without compute dataset - trains 4 layer models only)
python -m frontier.training.cli attention \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./cache/models \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --device a100 \
    --tensor_parallel_size 1
```

**Note**: `--compute_dataset_path` is now OPTIONAL. When not provided, compute-dependent models (attn_pre_proj, attn_post_proj, attn_rope, input_layernorm, post_attention_layernorm, add) are skipped.

### 3. Prediction

预测器自动集成到 `SklearnExecutionTimePredictor`：
- 检测到 `is_mixed_batch=True` 数据时自动训练
- 预测时优先使用 mixed predictor
- true mixed decode 使用 `attn_decode_in_mixed`
- 当检测到 mixed profiling 文件存在但输入数据不包含 mixed 列时，触发 fail-fast 阻断

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_mixed_prefill` | False | 启用混合 batch profiling |
| `--mixed_mode` | even | even/random/both |
| `--max_mixed_batch_size` | 8 | 混合 batch 最大大小 |
| `--mixed_num_samples` | 3 | Random 模式样本数 |
| `--enable_true_mixed` | False | 启用 true mixed profiling（prefill+decode 同批） |
| `--true_mixed_prefill_batch_sizes` | `1 2 4` | true mixed 中 prefill 序列数 |
| `--true_mixed_prefill_chunk_sizes` | `64 128 256 512 1024` | true mixed 中 prefill chunk size |
| `--true_mixed_decode_batch_sizes` | `1 2 4 8` | true mixed 中 decode 序列数 |
| `--true_mixed_decode_kv_cache_sizes` | `128 256 512 1024 2048` | true mixed 中 decode KV cache size |
| `--true_mixed_prefill_kv_cache_size` | 0 | true mixed 中 prefill 侧 KV cache size |
| `--max_pipeline_parallel_size` | 8 | Pipeline 并行度（影响内存计算） |

## CSV 字段

### 标准字段
```
time_stats.attn_prefill.{min,max,mean,median,std}
n_embd, n_q_head, n_kv_head
batch_size, prefill_chunk_size, kv_cache_size
is_prefill, is_mixed_batch
```

### 混合 Batch 特有字段
```
mode                # even/random
seq_lens            # [128, 256, 512, 1024]
total_tokens        # sum(seq_lens)
min_seq_len         # 128
max_seq_len         # 1024
avg_seq_len         # mean(seq_lens)
equal_seq_len       # sqrt(sum(s_i^2))
seq_len_variance    # var(seq_lens)
seq_len_std         # std(seq_lens)
seq_len_cv          # std/mean (变异系数)
```

### True Mixed Batch 特有字段
```
is_true_mixed_batch      # true mixed 行标记
num_prefill_seqs         # 同批 prefill 序列数
num_decode_seqs          # 同批 decode 序列数
total_prefill_tokens     # 同批 prefill token 总数
decode_avg_kv_cache_size # 同批 decode 平均 KV cache
batch_composition_ratio  # num_prefill_seqs / total_batch_size
```

## 架构设计

### 文件结构
```
frontier/profiling/attention/
├── mixed_attention_input.py    # MixedAttentionInput dataclass
├── attention_wrapper.py        # profile_mixed() 方法
├── main.py                     # CLI 入口
└── utils/__init__.py           # get_mixed_prefill_input_combinations()

frontier/training/
└── attention_trainer.py        # 集成 attn_prefill_mixed 训练

frontier/execution_time_predictor/
└── sklearn_execution_time_predictor.py  # 集成预测逻辑
```

### 数据流
```
Profiling → CSV (is_mixed_batch=True)
         ↓
Training → attn_prefill_mixed.pkl
         ↓
Prediction → 优先使用 mixed predictor
          ↓
      (true mixed decode) → attn_decode_in_mixed
```

## 特征工程

### 训练特征 (11个)
```python
[
    "batch_size",              # 批次大小
    "total_tokens",            # 总 token 数
    "avg_seq_len",             # 平均序列长度
    "min_seq_len",             # 最短序列
    "max_seq_len",             # 最长序列
    "total_tokens_squared",    # total_tokens^2
    "seq_len_variance",        # 方差
    "seq_len_cv",              # 变异系数
    "seq_len_range",           # max - min
    "batch_variance_interaction",  # batch_size * variance
    "batch_cv_interaction",    # batch_size * cv
]
```

### 目标变量
```python
"time_stats.attn_prefill.median"  # 注意：不是 attn_prefill_mixed
```

## 兼容性

### 向后兼容
- ✅ 无 `is_mixed_batch` 列 → 自动填充 False
- ✅ 无 mixed 数据 → 明确跳过 mixed predictor 训练，并在请求 mixed 预测时给出清晰错误
- ✅ 标准 prefill → 不受影响

### 预测兼容
```python
# 三层防御
if (batch_size > 1 
    and "attn_prefill_mixed" in models  # 模型存在
    and not has_chunked_prefill):       # 纯 prefill
    try:
        return mixed_predictor.predict(...)
    except:
        # Explicit error path
```

## 数据分析示例

```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv("attention_mixed.csv")

# 1. 变异系数 vs 性能
plt.scatter(df['seq_len_cv'], df['time_stats.attn_prefill.median'])
plt.xlabel('Coefficient of Variation')
plt.ylabel('Attention Time (ms)')
plt.savefig('cv_vs_time.png')

# 2. Even vs Random 对比
even_df = df[df['mode'] == 'even']
random_df = df[df['mode'] == 'random']
print(f"Even avg: {even_df['time_stats.attn_prefill.median'].mean():.3f} ms")
print(f"Random avg: {random_df['time_stats.attn_prefill.median'].mean():.3f} ms")

# 3. 特征重要性
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
X = df[['batch_size', 'total_tokens', 'seq_len_cv', ...]]
y = df['time_stats.attn_prefill.median']
model.fit(X, y)
print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))
```

## 常见问题

### Q1: 为什么目标列是 `attn_prefill` 而不是 `attn_prefill_mixed`?
**A**: Profiling 阶段使用相同的 attention kernel，统计键名统一为 `attn_prefill`。通过 `is_mixed_batch` 列区分数据类型。

### Q2: Even 模式有什么用？
**A**: Even 模式作为 baseline，用于对比真实混合（Random）的性能差异。

### Q3: 如何验证 mixed predictor 是否生效？
**A**: 查看日志中的 `Using mixed prefill predictor` 信息，或检查模型文件 `attn_prefill_mixed.pkl` 是否存在。

### Q4: `max_pipeline_parallel_size` 有什么影响？
**A**: 影响 `get_max_num_blocks()` 的内存计算。模型层数必须能被此值整除（如 Qwen2.5-7B 28层，用4或7）。

### Q5: Random 模式如何生成序列长度？
**A**: 在指定的长度范围内随机采样，每个配置生成 `--mixed_num_samples` 个样本（默认3个）。

## 测试验证

### 基础测试
```bash
# 1. Profiling 正确性
python -m frontier.profiling.attention.main \
    --enable_mixed_prefill \
    --mixed_mode both \
    --max_mixed_batch_size 4 \
    --models "microsoft/phi-2" \
    --num_gpus 1 \
    --max_seq_len 2048

# 检查输出
head -n 2 data/profiling/compute/a100/microsoft/phi-2/attention_mixed.csv
# 应包含: is_mixed_batch=True, mode, seq_lens, seq_len_cv 等字段

# 2. Training 正确性
python -m frontier.training.cli attention \
    --compute_dataset_path path/to/linear_op.csv \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./test_cache \
    --model_name "microsoft/phi-2" \
    --device a100

# 检查输出
ls ./test_cache/microsoft/phi-2/a100/tp_1/attn_prefill_mixed.pkl
# 应存在此文件

# 3. Prediction 正确性
# 在日志中查找:
# "Training mixed-batch prefill model with XXX samples"
# "Using mixed prefill predictor: batch_size=X, seq_lens=[...]"
```

## 性能基准

基于 Llama-2-7B, A100, TP=1:

| Batch Size | Seq Lens | Even Time | Random Time | Overhead |
|------------|----------|-----------|-------------|----------|
| 4 | [512]*4 | 2.1 ms | 2.1 ms | 0% |
| 4 | [256,512,768,1024] | - | 2.3 ms | +9.5% |
| 8 | [1024]*8 | 8.5 ms | 8.5 ms | 0% |
| 8 | [512,768,1024,1280,1536,1792,2048,2304] | - | 9.2 ms | +8.2% |

**结论**: 长度异质性引入约 8-10% 的性能开销。

## 相关文件

### 核心代码
- `frontier/profiling/attention/mixed_attention_input.py` (198 行)
- `frontier/profiling/attention/attention_wrapper.py` (291 行)
- `frontier/profiling/attention/main.py` (461 行)
- `frontier/training/attention_trainer.py` (549 行)
- `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`

### 文档
- `PROFILING_TEST_GUIDE.md` - 详细测试指南
- `README_MIXED_BATCH.md` - 本文档

## 版本历史

- **v1.0** (2025-11): 初始实现
  - Mixed batch profiling
  - attn_prefill_mixed 训练和预测
  - 集成到 AttentionTrainer
  - 完整文档和测试

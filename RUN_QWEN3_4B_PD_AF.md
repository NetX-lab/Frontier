# Qwen3-4B PD-AF 仿真运行手册

## 已安装内容

Profiling 数据（来自 `scale/qwen3-4b-pd-af.zip`）已安装到：

```text
data/profiling/compute/hpp2/Qwen3-4B/linear_op_kernel_only.csv
data/profiling/compute/h200/Qwen3-4B/{linear_op.csv,linear_op_kernel_only.csv,
                                      attention.csv,attention_kernel_only.csv}
data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only.csv
data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only_stream{1,2,3}.csv
data/profiling/network/a2f_h200_to_hpp2/m2n_transfer_lut.csv
data/profiling/network/hpp2_16rank/{all_reduce.csv,weight_stream_lut.csv}
data/profiling/network/h200_prefill_to_decode_attn/kv_cache_transfer_lut.csv
```

模型配置：

```text
data/config/models/Qwen3-4B.json
```

运行脚本：

```text
examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh
```

## 运行

在 Frontier 仓库根目录创建一次 uv 环境：

```bash
cd /home/yhu/Desktop/scale/Frontier
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e ".[test]"
```

然后运行：

```bash
cd /home/yhu/Desktop/scale/Frontier
export PYTHONPATH=$PWD
PYTHON_BIN=.venv/bin/python \
  bash examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh
```

也可以先激活 venv，再直接用 `bash`：

```bash
source .venv/bin/activate
bash examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh
```

常用覆盖参数：

```bash
PYTHON_BIN=.venv/bin/python \
NUM_REQUESTS=8 PREFILL_TOKENS=512 DECODE_TOKENS=64 QPS=1.0 \
RUN_ID=my_run bash examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh

# 0 = 36 层 FFN 全部常驻；1/2/3 = 流式对应层数
PYTHON_BIN=.venv/bin/python WEIGHT_STREAM_LAYERS=3 RUN_ID=stream3 \
bash examples/architecture/pd-af-disagg/offline/qwen3_4b_pd_af.sh
```

Metrics 输出到：

```text
outputs/qwen3-4b-pd-af/qwen3_4b/offline_batch/<RUN_ID>/
├── request_metrics.csv
├── system_metrics.json
└── op_precision_metadata.csv
```

## Baseline 结果（8 requests，prefill 512，decode 64，QPS 1.0）

```text
TTFT                 21.300 ms
TPOT                  1.661 ms
request E2E         125.926 ms
tokens/s          36,592.8  （仿真聚合值）
requests/s            63.53  （仿真聚合值）
DECODE_ATTN E2E      90.89 ms / 64 个 decode step
DECODE_FFN E2E        9.46 ms / 64 个 decode step
KV transfer           1.69 ms 总计（8 次传输）
M2N transfer          4.06 ms 总计（4536 次传输）
```

每步 FFN 计算相比 H200 attention 很小；对这个 4B dense 模型，AF 分离是
**attention 主导**。HPP2 FFN 集群仍有价值，但更适合作为容量/通信研究，
而不是延迟瓶颈。

全常驻 smoke（`1 request，prefill 64，decode 4`）：

```text
TTFT 8.736 ms, TPOT 1.395 ms, DECODE_ATTN E2E 4.053 ms, FFN E2E 0.064 ms
```

流式 3 层 smoke（同一 workload，33 层常驻 + 3 层流式）：

```text
TTFT 8.736 ms, TPOT 2.004 ms, DECODE_ATTN E2E 4.053 ms, FFN E2E 1.891 ms
```

流式 3 层跑 8-request baseline：

```text
TTFT              21.300 ms
TPOT               2.270 ms
request E2E      164.297 ms
DECODE_ATTN E2E   90.895 ms / 64 个 decode step
DECODE_FFN E2E    47.830 ms / 64 个 decode step
```

流式 3 层是一阶上界：权重流式时间被分摊到 36 层的 CPU overhead 表里，
没有模拟逐层流式加载与 FFN 计算的重叠。

## 已做的 Frontier 改动

1. **HPP2 device / node SKU**
   - `DeviceSKUType.HPP2`、`HPP2DeviceSKUConfig`（每 die 192 MiB APC DM，
     380 TFLOPS FP16）。
   - `NodeSKUType.HPP2_16RANK`，8 卡 = 16 rank。

2. **混合精度 profiling 元数据**
   - 新增可选 CSV 列 `profiling_precision_by_op`。
   - `ProfilingMetadata`、CSV 解析器和
     `QuantizationManager.register_profiling_metadata()` 现在支持按算子精度。
   - 元数据注册只覆盖文件中实际存在的算子，因此 FFN 表不会再把
     `emb`/`attn_rope`/norms 标成 FP8。
   - `op_precision_metadata.csv` 已验证：`mlp_up_proj`/`mlp_down_proj`/
     `attn_pre_proj`/`attn_post_proj` = FP8，`mlp_act`/norms/`attn_rope`/`emb`
     = BF16，全部 `data_source=profiling`（没有 2 倍 approximation）。

3. **DECODE_FFN 的 CPU overhead 与跨集群隔离**
   - CPU 表过滤与去重签名在 FFN 角色下改用 `moe_tensor_parallel_size`，
     因此 TP16 HPP2 表能被正确加载。
   - 共享模型管理器按 `(family, network_device|model|tp|csv)` 隔离 CPU
     overhead 模型，避免 `DECODE_FFN`（HPP2）的 PCIe host-bounce 与流式权重
     CPU overhead 泄露给同属 `kernel_only` family 的 `DECODE_ATTN`（H200）。
   - 运行脚本在 `WEIGHT_STREAM_LAYERS=1|2|3` 时保留 `{NETWORK_DEVICE}/{MODEL}`
     路径占位符，确保只有 `hpp2_16rank` 加载流式表。

4. **FP8 常驻权重内存核算**
   - `ParamCounter` 和 `system_metrics.model_weight_memory` 按配置的 MLP
     权重宽度计算（FP8 为 1 byte），不再固定按 2 bytes。
   - DECODE_FFN 现在显示 168.1 MB / 83.5% 的 192 MiB SRAM 预算。

## 从 LUT 拟合的 analytical 参数

| 路径 | 设置 |
|---|---|
| M2N direct | 366 Gbps，0 ms latency，关闭 p2p speedup |
| KV prefill→decode_attn（NVLink） | 3172.5 Gbps，0.0208 ms latency |
| HPP2 ring all-reduce | 2984 Gbps 节点内，0 latency |
| H200 clusters | 使用同一套 analytical CC 设置；未安装 H200 all-reduce 表 |

## 注意事项

- HPP2 FFN 表是 **SOL 理论值**，不是实测；H200 attention/linear 表是
  2026-11-26 实测。
- 全常驻 baseline 假设 36 层 FFN 都能放进每 die 192 MiB。按约 170 MiB
  可用算是临界情况；用 `WEIGHT_STREAM_LAYERS=3` 跑 33 层常驻 + 3 层流式，
  或者降低 batch / 提高 SRAM 预算。
- H200 CPU overhead 表没有安装；H200 host 项按 0 计并打 warning。
  HPP2 CPU overhead 已使用。
- `moe_grouped_gemm`/`moe_*` 的 warning 来自共享默认 MoE 路径；
  Qwen3-4B 是 dense，不会执行这些算子。
- Attention/MTP 常驻内存仍按 2 bytes 近似；只有 MLP 项跟随 FP8。
  H200 容量不受这个影响。
- `uv pip install -e ".[test]"` 会安装仿真器依赖：
  `numpy`、`pandas`、`scikit-learn`、`scipy`、`plotly`、`fasteners>=0.19`、
  `ddsketch`、`pyyaml` 和 `pytest`。
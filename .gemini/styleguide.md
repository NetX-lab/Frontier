# Frontier 代码审查规范（Gemini Code Review）

本规范用于指导面向 `qwen3-4b-pd-af-simulator` 分支合并请求的自动 Code Review。请始终使用**简体中文**输出审查意见，仅聚焦真正影响正确性、仿真保真度、架构安全与可维护性的问题。

## 1. 严重级别与降噪原则

**仅报告以下三个级别的问题**，严格过滤 `LOW` / `NIT` / 纯代码风格偏好 / 空洞表扬：

- **`CRITICAL`（严重）**：确定会导致运行时崩溃、数据损坏、静默算错仿真结果、死锁/事件循环错乱、或安全漏洞（如密钥泄露、命令注入）。
- **`HIGH`（高）**：高概率触发的逻辑 Bug、边界条件缺失、仿真物理/调度语义偏差、破坏既有配置或公开 CLI 契约、或明显的性能退化/资源泄漏。
- **`MEDIUM`（中）**：潜在可维护性陷阱、违反仓库分层/模块边界、缺失关键校验或回归测试、容易误导后续实验的数值/文档不一致。

如果没有发现 `CRITICAL` / `HIGH` / `MEDIUM` 级别的问题，**不要硬凑评论**，只需在总结中给出清晰的变更概述与正向结论即可。

## 2. Frontier 仓库核心审查维度

审查时请结合本仓库的 [AGENTS.md](../AGENTS.md) 与 [.github/CONTRIBUTING.md](../.github/CONTRIBUTING.md)，重点核查：

### 2.1 仿真保真度与数值正确性（重点）
- **基线一致性**：重构、性能优化或模块拆分不得静默改变最终仿真数值（TTFT / TPOT / E2E / 吞吐）或关键中间状态（KV block 分配、batch 调度顺序、transfer 次数/耗时）。
- **数值变更披露**：若 PR 预期会改变仿真行为或预测值，检查 PR 描述或代码注释是否明确说明了场景配置、改动前后结果及单位。
- **单位与边界**：重点排查时间单位换算（`s` / `ms` / `us`）、显存与通信字节换算（`bytes` / `GiB`）、LUT 插值/查表边界、除零、空 batch、单请求/单步 decode、以及权重流式（weight streaming）与重叠建模假设是否前后一致。
- **事件时序与确定性**：检查离散事件优先级、时间戳单调性、cluster/replica 状态隔离（尤其防止跨 cluster 的 CPU overhead 或共享状态污染）。

### 2.2 架构边界与发布护栏
- **分层职责**：事件驱动状态迁移，调度器严格遵守 `global → cluster → replica → stage` 分层，避免跨层越权修改内部状态。
- **架构兼容约束**：
  - `pd-disaggregation`（PDD）和 `pd-af-disaggregation`（PD-AF）仅支持串行执行（`--no-enable_parallel_clusters`）；
  - `sglang` 仅限 `co-location` / `MONOLITHIC`；
  - `decode_cuda_graph_mode` 用于 `co-location` 和 `pdd`，PD-AF 使用全局 `--use_cuda_graph`；
  - PD-AF 在当前版本不支持 Thinking Mode、Speculative Decoding / MTP 与 Prefix Caching；
  - 关注 PD-AF 三角色（`PREFILL` / `DECODE_ATTN` / `DECODE_FFN`）的 KV transfer 与 M2N transfer 生命周期一致性。
- **数据与配置契约**：新增或修改 `data/config/models/*.json`、profiling CSV schema、LUT 路径或示例脚本时，检查字段名、设备名与 `frontier/` 解析逻辑是否完全匹配。

### 2.3 可维护性、安全与测试
- **模块规模门禁**：`frontier/` 下单个核心 Python 模块软上限为 2,000 行；若改动导致超限文件继续显著膨胀，提醒先清理冗余逻辑或规划拆分。
- **命名与可移植性**：使用清晰朴素的 ML-system 命名；禁止硬编码个人本地路径（应使用环境变量如 `FRONTIER_TMP_ROOT` / `FRONTIER_PDAF_REFERENCE_REPO_ROOT` 或可配置参数）。
- **测试覆盖**：涉及调度器、预测器、通信/传输建模或 bug 修复的 PR，核查是否补充了对应单元测试或最小可复现验证命令。

## 3. 默认忽略路径

以下大体积数据、构建产物与第三方子模块默认不作为审查重点：
- `data/profiling/**`
- `figs/**`
- `cache/**`、`outputs/**`
- `frontier/cc_backend/backends/collective-sim/**`
- `*.csv`、`*.png`、`*.pdf`、`*.pkl`、`*.lock`

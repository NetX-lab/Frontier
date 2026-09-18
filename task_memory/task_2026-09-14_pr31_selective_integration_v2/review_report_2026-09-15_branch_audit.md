# Review Report: `feature-amd-sglang-gdn` 分支独立审阅（Plan 完成度 / 测试完备性 / main 矩阵对比 / 代码质量）

## Modification History

| Date | Summary of Changes |
| ---------- | ------------------------------------------------------------------------------------------------------ |
| 2026-09-15 | Initial independent audit of branch `feature-amd-sglang-gdn` (`7b6a3eb1`) against `main` (`0515589a`). |

## 0. 审阅范围与方法

- **审阅对象**：worktree 分支 `feature-amd-sglang-gdn`，HEAD `7b6a3eb1`，merge-base 与 `main` 为 `0515589a`。共 22 个 commit（`e8ac59ae` .. `7b6a3eb1`），`git diff main...HEAD --stat`：**123 files, +13073 / -714**；其中 `frontier/` 新增约 4977 行新模块代码，`tests/ docs/ examples/` 新增约 4093 行。
- **对照 Plan**：只读执行计划 `/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md`（2330 行），以及 task 目录内 `requirements.md / plan.md / progress.md / review.md / issues.md / summary.md / test_report_*.md`。
- **证据来源**：源码 diff、task 记录、`/data/ycfeng/tmp/pr31-*` 下的 fidelity 产物（`manifest.json / results.json`）、`tests/integration/run_scheduler_refactor_fidelity.py` 场景表。
- **性质**：本报告只做 review，不执行修复、不新增测试。所有 "建议" 均为待用户决策项。
- **严重度标签**：`P0` 合并前必须处理；`P1` 合并前应处理或明确记录为已接受风险；`P2` 建议后续清理；`INFO` 观察项。

### 0.1 总体结论（先给结果）

| 审阅维度 | 结论 |
| --- | --- |
| 1. Plan 规划的代码修改是否完成 | **大部分完成，但有 4 处与 Plan 明确约束相悖或未落地的偏差**（§1.2 D1–D4），其中 D1（`ExecutionTime` 内保留 multiplier / 双分派）和 D3（GDN state slot 生命周期未接入 scheduler）属于 Plan 文本明确要求的内容。 |
| 2. 关键模块是否完整测试验证 | **单元/焦点测试覆盖充分（新增 15 个测试文件，full unit 3276 passed / 19 baseline failed）；但关键运行时模块（`metrics_store` Stage 路径、MoE 多层递归 EP>1、dense predictor hybrid 分派、`CudaTimer -> DeviceTimer` NVIDIA 路径）缺乏对应的数值/回归证据**（§2.2）。 |
| 3. 是否与 `main` 做了充分的矩阵 E2E 对比 | **没有。** 记录中的三次 fidelity 运行，**没有任何一次以 `main`（`0515589a`）为 baseline**：Increment 5 的 baseline 是 feature 分支自己的前置 commit `57da3683`，且 manifest 记录 candidate SHA 与 baseline 相同；"final" 与 "inc14ab" 两次是 self-parity（baseline == candidate 同一 SHA、同一 worktree）。可用场景 58 个只跑了 5 个（均为 `_short`）。已观测到的 `sim_wallclock_s` 9.4x 退化被记录为 "observation"，未做根因分析或修复（§3）。 |
| 4. 代码质量 | **存在结构性问题**：`ExecutionTime` 双分派 + `StageExecutionTime.__getattr__` 动态代理（含 40+ 硬编码私有字段名单）、`metrics_store.py` 三条并行 Stage 路径、`getattr/hasattr` 新增 123 处、`"gated_delta_net"` 字面量分散、`_device_event_path` 三份拷贝、per-layer `deepcopy` 热路径、Singleton 私有字典穿透、`GatedDeltaNetStateSlotManager` 死代码、测试文件按 increment 编号命名等（§4）。 |

**建议的合并判定**：在 §5 的 P0 项（以 `main` 为 baseline 的矩阵 fidelity + 9.4x 慢化根因）完成前，**不建议合并到 `main`**。这一判定与 `requirements.md` 中 "该分支未经单独许可不得合并" 的既有决定一致。

---

## 1. Plan 完成度审阅

### 1.1 Increment 对照表

| Plan Increment | 主要产物（分支中） | 关键 commit | 状态 | 备注 |
| --- | --- | --- | --- | --- |
| 1. Linear attention metadata rename | `frontier/model_architectures.py` 等 | `e8ac59ae` | 完成 | 无 Co-authored-by trailer（summary 已说明）。 |
| 2. MI355X platform metadata & discovery | `frontier/profiling/common/accelerator.py`(232 行) | `0989f0ed` | 完成 | mocked `amd-smi`/`nvidia-smi` 测试。 |
| 3. Strict `DEVICE_EVENT` timing family | `profiling/common/device_timer.py`(129), `cuda_timer.py` 改为子类, `timer_stats_store.py`, `profiling/utils/__init__.py` | `49fcd9b8` | 完成，**但 NVIDIA `CUDA_EVENT` 路径被整体重写**（见 D4） | `CudaTimer` 原 110 行实现被删除，改为 `class CudaTimer(DeviceTimer)`；`summary.md`/`plan.md` 表述 "CUDA_EVENT unchanged" 仅在预期行为层面成立，实现已全部替换且无 GPU 验证。 |
| 4. Hybrid GDN semantic contracts + fixture | `frontier/attention/gdn/{config,features,memory,state,guards}.py`, `attention/model_binding.py` | `ddd9bbb3`, `57da3683` | 完成 | `state.py` 见 D3。 |
| 5. Real-layer timing & stage aggregation | `entities/stage_execution_time.py`(511), `entities/execution_time.py`, 两个 predictor, `metrics_store.py` | `b37cb38b` | **完成但偏离 Plan 约束**（D1、D2） | Plan 明确要求移除 `ExecutionTime` 内部 representative-layer multiplier、禁止永久双分派；实际引入 `_legacy_aggregate/_aggregation_factor`。 |
| 6. Hybrid GDN state memory | `scheduler/utils/memory_planner.py`, `utils/param_counter.py` | `0d1b87a4` | 完成 | `param_counter` 对 dense 模型的参数计算路径有改动，靠 `3_261_071_360` 既有 pin 保护。 |
| 7. Standard vLLM GDN profiler | `profiling/gdn/{main,inputs,vllm_wrapper}.py`(1020 行) | `0d4efa22` | 源码完成，**GPU 不可验证** | CPU 只覆盖 planning/schema。 |
| 8. GDN training & prediction | `training/gdn_trainer.py`(341), `execution_time_predictor/gdn_predictor.py`(261), `shared_prediction_model_manager.py` | `fd5f49b1` | 完成 | Review 2 修复了 `gdn_core_prefill/decode` 双 key 发射。 |
| 9. ROCm compat + `VLLM_ROCM` attention producer | `profiling/attention/backends/vllm_rocm_attention_wrapper.py`(329) | `294bc1b4` | 源码完成，GPU 不可验证 | Plan 验收 "grouped NVIDIA regression attempted if access exists" **未尝试且未记录为 SKIP**（D4）。 |
| 10. MoE MXFP4 profiling | `profiling/moe/moe_vllm_kernel.py`(+343/-27) | `3b20d702` | 源码完成，GPU 不可验证 | — |
| 11. Standalone RCCL collective profiler | `profiling/collectives/main.py`(+213/-54) | `a092f637` | 源码完成，GPU 不可验证 | `get_collectives_inputs` 的 `list(set(...))` 改为 `sorted(set(...))`，属良性确定性改动。 |
| 12/13. Experimental SGLang primitive replay | `profiling/experimental/sglang/*.py`(约 2050 行) | `f51231d4` | 源码完成，GPU 不可验证 | 隔离在 `experimental/`，输出 JSON 而非标准 CSV，边界清晰。 |
| 14A/B. Hybrid per-layer dispatch + CPU E2E | `sklearn_*_predictor.py`, `simulator.py`, `random_forrest_execution_time_predictor.py` | `45e66170`, `31b11762`, `2bdd96f2` | 完成 | Review 2 发现并修复 monolithic routing 使用 `range(cluster_num_replicas)` 与真实 `Replica.id` 不一致的问题。 |
| 14C. Docs | `docs/cli/README.md`, `docs/profiling/README.md`, `docs/profiling/ROCM_MI355X.md`, `docs/training/README.md`, `examples/profiling/README.md` | `7937e534`, `ffb9feea` | 完成 | 顶层 `README.md` 未改（符合 requirements）。 |

### 1.2 与 Plan 明确约束相悖 / 未落地的偏差

#### D1 (P1) `ExecutionTime` 保留了 multiplier 与双分派，违反 Increment 5 设计约束

- **Plan 原文要求**（Increment 5）：移除 `ExecutionTime` 内部 representative-layer 乘法；`StageExecutionTime` 是唯一的聚合点；不引入永久性的 "legacy 聚合 / 新路径" 双分派。
- **实际实现**：
  - `frontier/entities/execution_time.py:138` `self._legacy_aggregate = global_layer_id is None`
  - `frontier/entities/execution_time.py:545-548` `_aggregation_factor` 属性，在 `:541, :780, :794, :806, :1160, :1197, :1400` 等至少 8 处继续乘 `num_layers_per_pipeline_stage`
  - `frontier/entities/execution_time.py:1087` `num_layers` 属性按 `_legacy_aggregate` 分支
  - `frontier/entities/execution_time.py:1120` `as_single_layer()` 通过直接改写私有字段 `layer._legacy_aggregate = False` 切换模式
- **影响**：同一个类型的对象在 "有 `global_layer_id`" 和 "无 `global_layer_id`" 两种状态下返回不同倍数的时间值；这是 Review 1 中 32x all-reduce double-counting 事故的直接土壤（`review.md:51`）。任何新的调用点都需要知道自己拿到的是哪种对象。
- **建议**：将 `ExecutionTime` 收敛为严格单层语义（`num_layers == 1` 恒成立），legacy 多层构造入口统一改为返回 `StageExecutionTime.from_execution_time(...)`；删除 `_legacy_aggregate/_aggregation_factor`。若因调用面过大暂时无法完成，须在 `design.md` 记录为技术债并给出移除路径。

#### D2 (P0) 已观测到的 9.4x `sim_wallclock_s` 慢化未做根因分析与修复

- **Plan 原文要求**（Increment 5 verification）：per-layer 化后若出现 wall-time 退化，需在扩展实现前修复。
- **记录**：`test_report_2026-09-14_increment5.md:68`：`sim_wallclock_s` baseline 0.0092 s → candidate 0.0868 s（**9.42x**）；`issues.md:60` 记为 "observed cost ... not a blocker"。之后的 Increment 6–14 在此基础上继续叠加，再未测量。
- **代码层根因（审阅推断，未实测）**：
  1. `frontier/entities/stage_execution_time.py:226-243` `from_execution_time()` 对每一层调用 `source.as_single_layer()`，而 `execution_time.py:1118` `as_single_layer()` 内部是 `deepcopy(self)`。`ExecutionTime` 是含多个 component 对象和 operator map 的大对象；每个 batch stage 做 `num_layers_per_pipeline_stage` 次 deepcopy。
  2. `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:3436-3462` `predict_stage_execution_time()` 在 `num_layers > 1` 时**递归调用自身 num_layers 次**，每次重复执行参数校验、admission 检查、measurement-type 选择、CUDA graph 激活日志、OP-TRACE 日志等全部前置逻辑。
  3. `frontier/attention/model_binding.py:216, :240, :256` 直接调用未缓存的 `resolve_layer_attention_specs(config)`，而 `config/model_config.py:541` 已有 `_layer_attention_specs_cache`。热路径每次调用都重新构造 spec 对象。
- **建议**：(a) 用 `main` 为 baseline 重新测量 dense/MoE 各一组 `wallclock` 数据；(b) `from_execution_time` 改为共享不可变 component、只复制身份字段（或引入轻量 `LayerExecutionTime` 视图）；(c) MoE predictor 把 "一次前置检查 + N 次纯计算" 拆开，避免递归重入；(d) `model_binding.py` 改用 `config.get_layer_attention_specs()`。

#### D3 (P1) GDN state slot 生命周期只有孤立模块，未接入任何 scheduler

- **Plan 原文**（行 1825，Increment 14A E2E 验收）："GDN state slots persist across continuation/waiting and release on completion/cancellation. Feature rejection happens before state-dropping mutation."
- **实际**：`frontier/attention/gdn/state.py` 定义 `GatedDeltaNetStateSlotManager`（68 行），但 `rg GatedDeltaNetStateSlotManager frontier/` 在 `frontier/scheduler`、`frontier/events`、`frontier/entities` 中**零引用**；唯一使用者是 `tests/unit/test_gdn_state_lifecycle.py`。`review.md:63` 将其记为 "Simulator-only state lifecycle ... Lifecycle tests pass"，但这只是模块自测，不构成 E2E 验收。
- **当前运行时替代**：`scheduler/utils/memory_planner.py` 以 `state_per_request * max_num_seqs` 静态预留 GDN state 内存（新增 diff 行 116-124），`vllm_v1_engine_replica_scheduler.py:3052-3058` 在 preemption 前调用 `validate_gdn_runtime_support(... preemption_requires_state_drop=True)` 直接拒绝。静态预留 + 拒绝抢占在语义上可以自洽，但 Plan 要求的 "slot persist / release" 验收点没有被任何运行时代码或 E2E 覆盖。
- **建议**：二选一并记录决策：(a) 删除 `state.py` 与其测试，在 `design.md` 记录 "静态预留 + 禁止 state-drop preemption" 为本 release 的 GDN 状态模型；(b) 将 slot manager 接入 `VLLMv1EngineReplicaScheduler` 的 admit/finish/cancel 路径并补 E2E。当前状态属于死代码。

#### D4 (P1) NVIDIA `CUDA_EVENT` 路径被重写但无任何 GPU 验证，且未在最终验收矩阵中记录

- `frontier/profiling/common/cuda_timer.py` 从 110 行独立实现变为 15 行 `class CudaTimer(DeviceTimer)`；`DeviceTimer.__init__`（`device_timer.py:36-46`）改变了 `TimerStatsStore` 的获取方式：通过 `Singleton._instances.get(TimerStatsStore)` 穿透 metaclass 私有字典，未初始化时以 `DEVICE_EVENT` 新建 store。这与 `main` 上 `CudaTimer` 直接构造 `TimerStatsStore()` 的行为**不等价**（默认 method 不同）。
- Plan Increment 9 验收 "grouped NVIDIA regression attempted if access exists"；`test_report_2026-09-14_final_review2.md` 的验收矩阵只列出 `SKIP: AMD/MI355X hardware unavailable`，**没有 NVIDIA CUDA profiling 回归的 PASS 或 SKIP 条目**。用户规则允许通过 StepMind `RJobBackend` 提交 NVIDIA 作业，不属于 "hardware unavailable"。
- **建议**：至少运行一次 NVIDIA `examples/profiling/profile_linear_op.sh`（非 `--dry-run`）比较 `main` 与 candidate 的 `linear_op.csv` 分布；若确实无法执行，在验收矩阵显式记录 `SKIP: NVIDIA CUDA_EVENT regression not attempted` 及原因。

#### D5 (P2) Dense predictor 缺少 per-layer attention family 分派（潜在 fidelity 缺口）

- `sklearn_execution_time_predictor.py` 通过 `resolve_runtime_attention_family` 解析 family，但对 hybrid 模型仍按单一 family 施加到 stage 内所有层；目前 Qwen3.5 走 MoE predictor 所以未触发。若未来出现 dense-FFN + GDN 混合模型将得出错误时间。
- **建议**：在 `design.md`/`future.md` 记录该限制，并在 `validate_gdn_runtime_support` 中对 "hybrid GDN + dense FFN" 组合 fail-fast。

---

## 2. 关键模块测试验证完备性

### 2.1 已有覆盖（观察事实）

- 新增 15 个单元测试文件（`tests/unit/test_gdn_*.py`、`test_stage_execution_*.py`、`test_metrics_stage_execution_time.py`、`test_*_increment*.py` 等）。
- Full unit：`3276 passed, 19 failed, 25 skipped`，19 个失败与 `main` baseline 同名（`test_report_2026-09-14_final_review2.md:129`）→ **无 candidate-only 失败**。
- `compileall` 通过；Increment 14A/B 有合成 production-profile 构造的 CPU E2E（含真实 manager/predictor 训练与加载）。
- Review 2 期间发现并修复了 3 个真实 bug（GDN 双 key、profiling family-binding seam、monolithic routing identity），说明双审阅流程有效。

### 2.2 覆盖缺口（按风险排序）

| # | 模块 / 路径 | 缺口 | 严重度 | 说明 |
| --- | --- | --- | --- | --- |
| T1 | `metrics_store.py` `StageExecutionTime` 三条路径（`:691/705 _emit_stage_layer_traces`, `:3872/3904 _push_stage_layer_operation_metrics`, `:4127`） | 无 "同一 batch 用 `ExecutionTime` 与 `StageExecutionTime` 两种输入产生等价 metrics" 的等价性测试 | P1 | `:3914 return` 早退后 stage-owned 指标（`MLP_*` 等在 legacy 路径中按 `num_layers` 循环 push 的项）是否在 Stage 路径中被完整覆盖，目前只靠 `test_metrics_stage_execution_time.py` 的局部断言。 |
| T2 | `sklearn_moe_execution_time_predictor.py:3436` 多层递归 | 只在 `EP=1` 场景跑过数值一致性（MoE smoke） | P1 | Plan 强调 "MoE routing 和 EP workload 是 layer-specific"，但 `EP>1` 下 per-layer routing 是否与 `main` 的 representative-layer × N 一致（或有意不同）无证据。 |
| T3 | `ExecutionTime._legacy_aggregate` 双模式 | 无针对 "同一 payload 在 legacy 与 single-layer 模式下所有公开属性的比值恒为 N" 的表格化测试 | P1 | 32x 事故已证明该处易错。 |
| T4 | `CudaTimer -> DeviceTimer` | 无 NVIDIA GPU 回归（见 D4） | P1 | — |
| T5 | `param_counter.py` dense 参数计算改动 | 只有 `3_261_071_360` 单 pin（`tests/unit/test_param_counter_share_expert_memory_contract.py:28`） | P2 | 其他 dense/MoE 模型配置的 `num_parameters_per_device` 未与 `main` 对比。 |
| T6 | `simulator.py` 传入 `actual_replica_ids`、`random_forrest_execution_time_predictor.py` 行为改变 | 有 `7b6a3eb1` 焦点测试 | INFO | 覆盖到位。 |
| T7 | GPU 侧全部新增产物（约 3100 行：`profiling/gdn/*`, `vllm_rocm_attention_wrapper.py`, `moe_vllm_kernel.py` MXFP4, `collectives/main.py`, `experimental/sglang/*`） | 仅 import/schema/planning 级 CPU 测试 | INFO（已明确 SKIP） | 属已接受风险；但需注意这些代码合并后即成为 `main` 的维护负担。 |

---

## 3. 与 `main` 的矩阵 E2E 对比审阅

### 3.1 记录中的 fidelity 运行事实

| 产物目录 | baseline 路径 / SHA | candidate 路径 / SHA | 场景数 | 判定 |
| --- | --- | --- | --- | --- |
| `/data/ycfeng/tmp/pr31-inc5-fidelity-20260914-final-1789374105` | `/data/ycfeng/tmp/pr31-inc5-baseline` / `57da3683` | feature worktree / **`57da3683`** | 5（co-location & pd-af × dense/MoE `_short` + 1） | `57da3683` 是 **feature 分支上的 commit**（`git merge-base --is-ancestor 57da3683 main` 为假），不是 `main`；且 manifest 记录 candidate SHA 与 baseline 相同，说明 candidate 是在**未提交的 dirty worktree** 上运行，manifest 无法证明被测代码版本。 |
| `/data/ycfeng/tmp/pr31-final-fidelity-20260914` | feature worktree / `f51231d4` | feature worktree / `f51231d4` | 1 | **self-parity**，对回归检测无信息量。 |
| `/data/ycfeng/tmp/pr31-inc14ab-fidelity-20260914-run2` | feature worktree / `fd5f49b1` | feature worktree / `fd5f49b1` | 1 | **self-parity**。 |

**结论**：截至 HEAD `7b6a3eb1`，**不存在任何一份以 `main` (`0515589a`) 为 baseline 的 fidelity 证据**。`plan.md`/`summary.md` 中 "preserving existing dense/MoE simulator behavior" 的说法，其支撑仅为：(a) full unit 无 candidate-only 失败；(b) Increment 5 时相对 feature 分支前置 commit 的 5 个 `_short` 场景一致。

### 3.2 矩阵广度

- `tests/integration/run_scheduler_refactor_fidelity.py` 共 58 个场景（co-location / pdd / pd-af × offline / online × dense / MoE / thinking / spec_dec / prefix_caching 等），实际只用 5 个，且全部为 `_short` 变体。
- 未覆盖：`online` 模式、PDD（`pd-disaggregation`）、`spec_dec`、`prefix_caching`、`thinking_mode`、`EP>1`、`PP>1`（`num_layers_per_pipeline_stage` 变化是本次重构的核心变量）、非 dummy predictor 路径。
- 工作区规则（AGENTS.md "Development Gates"）要求 "至少 50 个具体场景设置" 的矩阵验证；当前 5 个。

### 3.3 已观测差异的分析与修复状态

| 差异 | 是否分析 | 是否修复 |
| --- | --- | --- |
| 32x `attention_all_reduce_time` double-counting（Increment 5 首轮） | 是 | 是（改为 first-layer raw value） |
| `sim_wallclock_s` 9.4x（0.0092 s → 0.0868 s） | **否**（记为 observation） | **否**，且后续 increment 未再测 |
| hybrid probe `query_len must be positive` | 是 | 是 |
| monolithic routing identity（`Replica.id` vs `range(n)`） | 是 | 是 |

### 3.4 建议的最小补充矩阵（供后续执行，本次不执行）

1. baseline = `main@0515589a` 干净 worktree；candidate = `7b6a3eb1` 干净 worktree（manifest SHA 必须不同）。
2. 场景：`run_scheduler_refactor_fidelity.py` 全部 58 个，或至少覆盖 {co-location, pdd, pd-af} × {offline, online} × {dense, MoE} × {PP=1, PP=2} × {EP=1, EP=2} 且含 spec_dec / prefix_caching 各一。
3. 指标：`request_metrics.csv` 全列 exact match（dummy 模式应逐 bit 相同）；`system_metrics.json` 的 `ttft/tpot/e2e/throughput`；`sim_wallclock_s` 与 `total_proc_s` 三次中位数比值。
4. 对任何非零差异逐项给出 root cause（预期改变 / bug），不可仅以 "observation" 记录。

---

## 4. 代码质量审阅

### 4.1 模块体量与集中度

| 文件 | 变化 | 当前总行数 | 评估 |
| --- | --- | --- | --- |
| `frontier/profiling/gdn/vllm_wrapper.py` | +639 新建 | 639 | 单一 GPU wrapper，体量偏大但职责单一；CPU 不可验证。 |
| `frontier/entities/stage_execution_time.py` | +511 新建 | 511 | 其中约 130 行是三张硬编码属性名单（`_PER_LAYER_PUBLIC_NAMES/_STAGE_ONLY_PUBLIC_NAMES/_PER_LAYER_PRIVATE_NAMES`）+ 15 个 `get_single_layer_*` 转发 + `__getattr__`。 |
| `frontier/metrics/metrics_store.py` | +324 / -7 | **5587** | 已远超 2000 行 gate；本次继续增加三条并行 Stage 路径，未做 AGENTS.md 要求的 cleanup-first / split analysis。 |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | +300 / -104 | 3719 | 同上，超 gate。 |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | +211 / -21 | 4760 | 同上，超 gate。 |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | +172 / -90 | 8397 | 同上，超 gate。 |
| `frontier/profiling/experimental/sglang/*` | +2050 新建（7 文件） | — | 隔离良好，但以 7 元 positional tuple 作为 orchestration contract（`graph_replay.py:205-240 _make_replay_call`）可读性差。 |

> AGENTS.md "Python Module Size and Naming" 要求对 >2000 行的关键模块 "first ... remove redundant ... code" 并 "perform a design analysis for a functional split"。本分支对 4 个已超限模块只做加法，未见相应分析记录。

### 4.2 具体问题清单

每项格式：**位置** → 问题 → 建议。

#### Q1 (P1) `StageExecutionTime.__getattr__` 动态代理 + 硬编码字段名单

- `frontier/entities/stage_execution_time.py:486-509`；名单 `:~20-131`。
- 问题：公开属性求和、私有属性取 first-layer、`_is_moe` 取集合、`_num_layers_per_pipeline_stage` 返回层数、其他 `_` 前缀转发给 owner——五种语义藏在一个 `__getattr__` 里；名单需与 `ExecutionTime` 的 40+ 私有字段手工同步，`ExecutionTime` 新增字段时静默漏掉。这也是 32x 事故的机制来源。
- 建议：删除 `__getattr__`，把真正被外部访问的属性显式声明为 property（可用一个小的 declarative table + `functools.partial` 生成），对私有字段访问直接在调用方改为 `stage.layer_execution_times[0].xxx`，让 "取哪一层" 在调用点可见。

#### Q2 (P1) `ExecutionTime` 双模式（见 D1）

- `frontier/entities/execution_time.py:138, 545-548, 1087, 1120` 及 8 处 `* self._aggregation_factor`。
- 建议：见 D1。

#### Q3 (P0) per-layer `deepcopy` 与 MoE 递归重入热路径（见 D2）

- `frontier/entities/execution_time.py:1118 deepcopy(self)`；`stage_execution_time.py:226-243`；`sklearn_moe_execution_time_predictor.py:3436-3462`。
- 建议：见 D2。

#### Q4 (P1) `metrics_store.py` 三条并行 Stage 路径 + 早退

- `_emit_stage_layer_traces` (`:1010`)、`_push_stage_layer_operation_metrics` (`:3690`)、`:4127` 分支；`:3914 return` 早退跳过 legacy 路径其余 push。
- 问题：与 legacy 路径逐 op 复制 `push(...)` 序列；两处 `if family_id != "gated_delta_net"` 字面量分支（`:1053, :1061, :3709, :3718, :4148`）。
- 建议：让 legacy 入口把 `ExecutionTime` 包装为单层 `StageExecutionTime` 后走同一条路径（一条路径、按层循环）；GDN 是否输出 `ATTN_PRE_PROJ/ROPE/POST_PROJ` 应由 attention family 描述表（`get_attention_family(...)` 已存在）声明，而非在 metrics 层写字面量。

#### Q5 (P1) `getattr/hasattr` 防御式访问激增

- `git diff main...HEAD -- frontier | rg -c '^\+.*(getattr|hasattr)\('` = **123 处新增**（删除 6 处）。
- 典型：`memory_planner.py` diff 行 37-39/104-105 `getattr(model_config, "get_num_gdn_layers", None)` + `callable(...)`；`sklearn_moe_execution_time_predictor.py:3438-3441` `callable(getattr(self._model_config, "is_moe_layer", None))`、`:2351 getattr(self, "_attention_query_cache", None)`（实例属性在 `__init__` 外懒建）；`vllm_v1_engine_replica_scheduler.py:3054-3056` `getattr(self, "_replica_config", None)` / `getattr(replica_config, "model_config", None)`。
- 问题：同一 "模型是否含 GDN 层" 判断在 ≥5 个文件各写一遍，而 `frontier/attention/gdn/` 已有 `model_has_gdn`/`get_num_gdn_layers` 帮助函数；部分 `getattr(..., None)` 是为了容忍测试 mock 而非真实运行时对象缺属性，属于 production code 迁就 test double。
- 建议：`BaseModelConfig` 上把 `get_num_gdn_layers()/is_gdn_layer()/get_gdn_config()` 定义为正式接口（非 GDN 模型返回 0/False/None），调用方直接调用；`_attention_query_cache` 在 `__init__` 初始化；对 scheduler 的 `_replica_config` 直接访问（它是构造时必填）。

#### Q6 (P2) 重复代码

- `_device_event_path` 三份完全相同的内嵌函数：`shared_prediction_model_manager.py:637`、`:4678`、`sklearn_execution_time_predictor.py:851`。→ 提为 `execution_time_predictor/utils` 模块级函数。
- `"gated_delta_net"` 字符串字面量分布于 `metrics_store.py`（5 处）、`attention/gdn/config.py`、`stage_execution_time.py`。→ 统一引用 `GDN_ATTENTION_FAMILY.family_id`（或等价常量）。
- `resolve_layer_attention_specs` 已有缓存入口却在 `model_binding.py` 三处直接调用未缓存版本。

#### Q7 (P2) `DeviceTimer` 穿透 Singleton 私有状态

- `frontier/profiling/common/device_timer.py:36` `Singleton._instances.get(TimerStatsStore)`。
- 问题：依赖 metaclass 实现细节；同时改变了未初始化时的默认 method（`DEVICE_EVENT`）。
- 建议：给 `TimerStatsStore` 增加显式 `current()`/`get_or_create(profile_method)` 类方法；`CudaTimer` 子类显式传 `profile_method=CUDA_EVENT` 保证与 `main` 默认一致。

#### Q8 (P2) 死代码 / 无用参数

- `frontier/attention/gdn/state.py` `GatedDeltaNetStateSlotManager` 运行时零引用（D3）。
- `frontier/attention/gdn/guards.py:24,47` `validate_gdn_runtime_support(..., waiting=...)`：`waiting=True` 仅提前 `return`，而 `waiting` 语义本应由 slot manager（D3）承载；当前是为未接线的生命周期预留的分支。
- `frontier/attention/model_binding.py:234-242` `_has_hybrid_attention_schedule` 吞掉 `ValueError` 返回 False，会把配置错误伪装成 "非 hybrid"。→ 只捕获明确的 "no schedule" 情形，其余上抛。
- `analytical_kv_cache_transfer_predictor.py:50-54` `_calculate_kv_cache_size_for_tokens` 每次 batch/request 计算都调用 `validate_gdn_runtime_support` 与 `bind_attention_family`（结果对同一 config 恒定）。→ 在 predictor 构造时校验/绑定一次并缓存。

#### Q9 (P2) 可读性 / 命名

- 测试文件以 plan increment 编号命名：`test_gdn_increment6_memory.py`, `test_gdn_profiler_cpu_increment7.py`, `test_gdn_training_predictor_increment8.py`, `test_vllm_rocm_attention_wrapper_increment9.py`, `test_moe_mxfp4_increment10.py`, `test_collectives_increment11.py`, `test_sglang_experimental_increment13.py`, `test_gdn_hybrid_e2e_increment14ab.py`。编号对脱离本 task 的读者无意义，且与 AGENTS.md "plain names that describe the component's ML-system responsibility" 冲突。→ 按组件命名（如 `test_gdn_memory_planner.py`, `test_vllm_rocm_attention_wrapper.py`）。
- `stage_execution_time.py` 15 个 `get_single_layer_*` 手写转发（`:444-484`）→ 与 Q1 一并用声明式表生成或删除。
- `graph_replay.py:205-240` 7 元 positional tuple contract → 改为 `NamedTuple`/`dataclass`（`ReplayCall`），成本极低。
- `StageExecutionTime.communication_operator_times`（`:331-345`）在 property 内把 first-layer 的通信 op 与 owner record 的 `pipeline_parallel_send_recv` 拼成一个新 `CommunicationOperatorTimes`，返回值既非 "第一层" 也非 "stage 总和"，是第三种混合语义，且与 `__getattr__` 对私有字段的 "first-layer" 规则不一致。→ 明确拆为 `first_layer_communication_operator_times` 与 `stage_communication_operator_times` 两个命名清晰的访问器。

#### Q10 (INFO) 硬编码 / 临时补丁

- 未发现魔法数字型 hard-coding 或 `# TODO/HACK` 式临时补丁；`profiling/utils/__init__.py:validate_profile_method_platform` 的 `{"cuda","rocm","cpu"}` 与 `accelerator.py` 的平台枚举是否同源需确认（若两处各自维护则应统一到一个 enum）。
- `experimental/sglang/graph_replay.py:236` `atol=0.03, rtol=0.03` 为研究性容差，位于 experimental 且有注释，可接受。

### 4.3 正面观察

- experimental SGLang 工具链严格隔离（独立目录、JSON 产物、`experimental: True` schema 校验、拒绝写标准 CSV），边界设计好。
- `config/utils.py::dataclass_to_dict` 过滑私有字段，避免缓存被序列化，是正确的小修。
- Review 2 阶段修复了 3 个真实缺陷，双审阅流程有效。
- Commit 拆分细、消息清晰、保留 `Co-authored-by`，满足 requirements 对贡献历史的要求。

---

## 5. 处理优先级建议（供决策，本次未执行）

| 优先级 | 项 | 对应条目 |
| --- | --- | --- |
| P0 | 以 `main@0515589a` 为 baseline、干净 worktree 双端、≥50 场景的 fidelity 矩阵，manifest 两端 SHA 必须不同；对每个非零差异给 root cause | §3.4, D2 |
| P0 | `sim_wallclock_s` 9.4x 根因确认与修复（deepcopy per layer、MoE 递归重入、未缓存 spec 解析） | D2, Q3 |
| P1 | 收敛 `ExecutionTime` 为单层语义，移除 `_legacy_aggregate/_aggregation_factor` 与 `StageExecutionTime.__getattr__` | D1, Q1, Q2 |
| P1 | `metrics_store.py` Stage/legacy 路径合并为一条；GDN op 可见性由 family 表声明 | Q4 |
| P1 | GDN state slot：删除死代码或接入 scheduler 并补 E2E，记录决策 | D3, Q8 |
| P1 | NVIDIA `CUDA_EVENT` profiling 回归（或显式 SKIP 记录）；`DeviceTimer` 默认 method 与 `main` 对齐 | D4, Q7 |
| P1 | `EP>1`、`PP>1` 的 MoE 多层数值对比；`ExecutionTime` 双模式比值表测试；metrics 等价性测试 | T1–T3 |
| P2 | `getattr/hasattr` 收敛为 `BaseModelConfig` 正式接口；`_device_event_path`、`"gated_delta_net"` 去重；`model_binding.py` 用缓存入口 | Q5, Q6 |
| P2 | 测试文件按组件重命名；`ReplayCall` NamedTuple；`waiting` 参数与 `ValueError` 吞噬清理；transfer predictor 校验前移 | Q8, Q9 |
| P2 | 对 4 个 >2000 行模块补 AGENTS.md 要求的 cleanup / split analysis 记录 | §4.1 |

---

## 6. 审阅结论

- **Plan 完成度**：实现面基本覆盖 Increment 1–14，且 GPU 不可验证部分已诚实标注 SKIP。但 Increment 5 的两条硬约束（去 multiplier / 无双分派；wall-time 退化先修）和 Increment 14A 的 state-slot 验收点未满足，Increment 9 的 NVIDIA 回归未尝试也未记录。
- **测试完备性**：单元层面充分；运行时关键路径（metrics Stage 路径、MoE 多层 EP>1、dense hybrid、CUDA timer）缺乏数值证据。
- **与 `main` 的矩阵对比**：**实质缺失**——三次 fidelity 均非 `main` baseline，两次为 self-parity，一次 candidate SHA 不可证明；覆盖 5/58 场景；已观测 9.4x 慢化未分析。这是当前最大的风险项。
- **代码质量**：功能正确性由测试兜底，但 `ExecutionTime/StageExecutionTime` 的双模式 + 动态代理设计、metrics 的并行路径、123 处防御式 `getattr` 使模块在合并后**比修改前更难维护**；四个超 2000 行模块继续增重且无拆分分析。

**判定**：`NOT READY TO MERGE`。完成 §5 P0 两项并处理（或明确记录接受）P1 项后可再次审阅。此判定不否定分支已完成的实质工作量与 GPU 侧源码交付。

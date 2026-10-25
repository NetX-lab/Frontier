# DSV4 主线推进与 spec/tickets 审查报告（2026-10-25）

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | 已落实 §7 的文档修复：requirements.md 新增 D-DSV4-H800-MAINLINE-CORRECTION；plan/progress/summary/future 与 tickets 05、07、09、10、17、19 改为指向 H4a 主线；子任务目录加“路线已被取代”说明。L-6 的代码位置更正为 `frontier/model_architectures.py:702`。ticket 18 改名、工单归档和 worktree 清理未执行（需单独批准）。 |
| 2026-10-25 | 首次创建：审查 DSV4（H800 32/64 卡）主线是否偏离、是否陷入循环，以及已完成 spec/tickets 的严重错误与遗漏。 |

## 0. 审查范围、方法与标注

- **对象：** worktree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`，branch `feat/glm53-dsv4-vllm028`。审查结束时 HEAD 为 `dc653845`（2026-10-25 12:32 +0800）。审查期间另一个 session 仍在这个 branch 上提交（12:07 → 12:32 共 4 次），所以本报告以 `dc653845` 的状态为准。
- **当前范围（用户 2026-10-25 指示）：** 只做 H800 上 DeepSeek-V4-Flash（DSV4）的模拟支持和精度矫正。
  - P0 = H800-32，P16+D16，每个角色 TP8/DP2/EP16/PP1。
  - P1 = H800-64，P32+D32，TP8/DP4/EP32。
  - GLM 已暂停（D-DSV4-ONLY-GLM-PAUSED）。H200 P2 不在当前范围内。
- **方法：**
  - 阅读主任务和子任务的 `requirements.md`、`spec.md`、`plan.md`、`progress.md`、`summary.md` 以及 tickets 00–19。
  - 用 `git log --numstat` 统计提交。
  - 阅读 DSV4 相关代码的拒绝路径（guard）。
  - 做一次 CPU 探测：`/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/review_20261025/v4_pd_guard_probe.py`。
- **没有做的事：** 没有运行 GPU，没有修改代码或已有记录，没有提交 commit。本报告是唯一的新文件。
- **标注：** [实测] 表示直接读文件或命令输出得到的结论；[推断] 表示在实测基础上推理得到、没有单独验证的结论。

## 1. 结论摘要

1. **主线已经明显偏离，而且正在循环。**
   - [实测] 10-23 原生 vLLM 参考（native06 R4）被接受之后，Frontier 模拟器代码只改了一次：`4a1b53e6`，93 行，10-23 11:53。10-24 起 `frontier/` 的改动为 0 行。
   - [实测] 10-24 23:21 到 10-25 12:32 的 13 小时里，18 次提交都在处理同一件事：在 60 秒 / 30 GiB 的限制内，算完 P0 那一个 prefill DP 组 8 个 rank 的 trace 指标（R5 从 24/32 推进到 32/32）。每次失败（超时、SIGABRT、GDB 观察被截断、SIGSEGV）都换来一个"新诊断分配"，R5 始终停在 24/32。
   - [实测] Frontier 目前仍然**无法运行 DSV4 的 PD（P/D 分离）模拟**。三处 guard 会直接抛出 `NotImplementedError`，所以从来没有算出过任何一个四指标误差。
   - 这违反了 D-MAINLINE-LOOP-CAP（`requirements.md:377`）和 D-MEASUREMENT-FIRST（`requirements.md:1066`）。
2. **spec/tickets 存在 4 个高严重度问题：**
   - V4 PD 执行路径没有实现，也没有活跃的负责 ticket；ticket 05 把它挂在已暂停的 GLM 前置项上。
   - 计划中没有"首次预测对比"里程碑，误差信号被放在全部资格认证之后。
   - 成本路线（逐 kernel 完整归属）与 P0 decode 的实际运行形态（CUDA Graph，每个 DP lane 约 10 个请求）不匹配，而且"完整"没有边界。
   - 没有 Frontier 侧的 P0 运行入口（driver/config）。
3. **下一步建议：**
   - 立即停止 R5 trace-audit 循环。
   - 把下一个里程碑改为"Frontier DSV4 P0 首次诊断性预测，与 native06 R4 对比四指标误差"。
   - 请 YC 一次性决定第 7 节列出的 3 个问题。

## 2. Worktree 与 task 目录梳理

### 2.1 Worktree

[实测] `git worktree list` 共 218 个，其中 75 个是 detached HEAD。按名称前缀统计：

| 前缀 | 数量 | 与当前任务的关系 |
| --- | --- | --- |
| `step5*` | 56 | 另一个任务（Step-5-Preview），只共享 68f891b 工作负载文件。不在本次范围内。 |
| `v4-*` | 35 | DSV4 子任务的分支，例如 `v4-grouped-nixl`、`v4-graph-measured`、`v4-balanced-*`、`v4-native-profile04-run`。通过的结果按内容合入主 branch，没有保留 git 祖先关系。 |
| `fidelity*` / `verify*` / `d29*` | 23 / 19 / 13 | 历史验证或其他任务，与当前 DSV4 主线没有直接关系。 |
| `dsv4-c07..c23-*-validation-2026101[89]` | 14 | 10-18/19 的历史单元验证（detached），现在没有作用。 |
| `glm*` / `glm53*` | 9 / 7 | GLM 已暂停。 |
| `glm53-dsv4-vllm028` | 1 | **唯一的交付 branch**，权威任务记录都在这里。 |

[实测] 主 worktree 根目录还有一些不属于本次审查的未跟踪文件：`.cortexkit/`、`.pi/`、`remaining_measurements/`，以及新的 `tests/analysis/v4_trace_audit_prefill_fault.{py,gdb}`（由并发 session 产生）。

### 2.2 Task 目录（21 个）

| 目录 | 角色 |
| --- | --- |
| `task_2026-10-15_glm53_dsv4_vllm028_h800/` | **主任务。** 权威的 requirements/spec/plan/tickets 00–19。 |
| `task_2026-10-23_v4_remaining_measurements/` | **当前执行子任务。** `plan.md:134` 指定它的 `progress.md` 为下一 session 的 canonical handoff。 |
| `task_2026-10-22_v4_balanced_routing/` | 已接受的 balanced prefill 数据（`prefill_results/accepted/PREFILL_WARM_6144_C0_BALANCED/`）。 |
| `task_2026-10-21_v4_grouped_nixl/`、`task_2026-10-21_v4_graph_measured/`、`task_2026-10-22_v4_isolated_decode/`、`task_2026-10-22_v4_h800_prefill/`、`task_2026-10-22_v4_native_*`、`task_2026-10-23_v4_*`、`task_2026-10-21_v4_attention_c26/`、`task_2026-10-21_v4_q1_warm_remeasure/` | 已完成的 DSV4 子步骤记录。 |
| `task_2026-10-21_glm_*`、`task_2026-10-22_glm_hardcoding_ttft/`、`task_2026-10-21_w128_dsa_native_config/` | GLM，已暂停。 |
| `task_2026-10-21_v4_h200_admission/`、`task_2026-10-21_glm_full_dummy_h200/` | H200，不在当前范围内。 |
| `task_2026-10-21_native_*`、`task_2026-10-21_p0_input_binding/` | 原生 vLLM 运行准备与 P0 输入绑定，已完成。 |

## 3. 主线目标与当前距离

主线目标是让 Frontier 能模拟 DSV4 P0，并使四个指标（mean client TTFT、mean TPOT、mean E2E、THPT）的误差都 ≤10%。之后再推进 P1。

### 3.1 原生 vLLM 侧已经完成

[实测] native06 R4 参考已被接受，数据见 `native06_acceptance_20261023/client_report.md:59-63`：

| Round | TTFT (ms) | TPOT (ms) | E2E (ms) | THPT (tok/s) | Window (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 403.31 | 19.031 | 19872 | 6273.3 | 146.25 |
| 2 | 393.22 | 19.017 | 19847 | 6273.1 | 146.26 |
| 3 | 399.95 | 18.981 | 19817 | 6269.8 | 146.34 |

[实测] 三轮之间的差距很小：TTFT 约 2.5%，TPOT 约 0.3%，E2E 约 0.3%，THPT 约 0.06%。作为对比参考，质量足够。

### 3.2 Frontier 侧仍然无法运行 DSV4 PD

[实测] 代码和 CPU 探测的结果：

| 位置 | 行为 |
| --- | --- |
| `frontier/attention/families.py:646-650` | DSV4 family 的 `execution_enabled=False`，原因写的是 "requires compressed cache, compressor, indexer, mHC and mixed-precision operator coverage"。 |
| `frontier/attention/deepseek_v4.py:39` | `afd_stage_metadata` 抛出 `NotImplementedError("V4 PD attention is unsupported")`。 |
| `frontier/attention/deepseek_v4.py:101-102` | 只要 `cluster_type` 不是 `MONOLITHIC`，就抛出 "V4 attention component requires co-location"。 |
| `frontier/execution_time_predictor/deepseek_v4.py:457-458` | 只要 `KV_CONNECTOR != "none"`，就抛出 "V4 predictor PD phase/graph equivalence remains unqualified"。 |
| `frontier/execution_time_predictor/deepseek_v4.py:640-648` | 读取实测数据时，graph mode 不是 NONE 就拒绝；isolated decode 必须正好 1 个请求、`num_tokens==[1]`。 |
| CPU 探测 | 输出 `execution_enabled= False`。PREFILL 和 DECODE 角色都触发 `NotImplementedError: V4 attention component requires co-location`，`serving_context(nixl)` 触发 "remains unqualified"。 |

[实测] ticket 19 第 66 行的状态是："native comparison, calibration and final measured acceptance have not started"。

[推断] 从 native06 被接受（10-23）到现在已经两天多，期间没有产生过任何 Frontier 预测值，因此精度矫正还没有开始。

## 4. 偏离与循环分析

### 4.1 提交统计

[实测] 用 `git log --since=<date> --numstat` 按路径统计（单位：行）：

| 时间窗口 | commits | `frontier/` | `tests/` | `task_memory/` | `data/` |
| --- | --- | --- | --- | --- | --- |
| 10-20 起 | 298 | +6820 / −909 | +19382 / −757 | +1436485 / −1269 | +38977 / −12013 |
| 10-23 起 | 112 | **+93 / −1** | +4177 / −160 | +1397213 / −736 | 0 |
| 10-24 起 | 59 | **0** | +1788 / −9 | +1045373 / −364 | 0 |

- [实测] 10-23 起 `frontier/` 只有一次提交：`4a1b53e6`（"Load externally admitted V4 isolated prefill layouts from private input bundles"）。
- [实测] 10-24 起新增的 `tests/` 代码主要是 `tests/analysis/v4_trace_audit*`，属于 trace 审计工具，不是模拟器能力。
- [推断] `task_memory/` 的行数包含 JSON/CSV 证据副本，但"记录与代码"约 15000:1 的比例说明，精力主要花在了证据整理上，而不是模拟能力上。

### 4.2 时间线

| 时间 | 事件 | 对主线的作用 |
| --- | --- | --- |
| 10-20 ~ 10-22 | DSV4 profiling producer、balanced routing、grouped NIXL 管道、D-Q10、isolated input bundle | **有效推进** |
| 10-23 | native06 R4 接受；balanced decode q1 数据接受；`4a1b53e6`；采纳 D-V4-KERNEL-ONLY-GPU-COST；提交 profile02（32×H800） | 参考到位，成本路线开始收紧 |
| 10-24 | profile02 只被接受为诊断数据；stream probe（1×H800）；graph callback probe（1×H800，原 3/3 次失败后重做）；profile03（32×H800）失败；profile04（32×H800） | 6 次 GPU 作业，都用于 trace 归属资格认证 |
| 10-24 23:21 → 10-25 12:32 | CPU trace-audit 循环，18 次提交（见 4.3） | **没有推进里程碑** |
| 10-25 02:12 | scope-lifetime probe（1×H800，`exp-1025-021228-212042`） | 只对 TLS/scope 生命周期有效 |

### 4.3 循环明细（目标始终是 R5 24/32 → 32/32）

[实测] 来自 `git log` 和子任务 `summary.md:7-24`：

| 时间 | commit | 内容 | 结果 |
| --- | --- | --- | --- |
| 10-24 23:21 | `84a66948` | 接受 lifetime CPU source，保留 P0 GC 诊断 | P0 rank 在 60 秒内跑不完 |
| 10-25 03:24 | `0d0fe648` | 优化 frozen trace metrics | — |
| 04:04 | `ffbfbc37` | P0 pilot | **FAILED_P0_PILOT_NO_FINAL_JSON**（60.33 s / 30 GiB） |
| 04:08 / 04:27 | `2fbd5af6` / `4ae81052` | sparse timing；记录被截断的诊断，分配 group correction | — |
| 04:42 / 04:44 | `8b4d6993` / `00840389` | group metrics；冻结 rank 分配 | — |
| 04:53 | `947da5ee` | group pilot | **FAILED**，再分配 segment 诊断 |
| 05:16 / 05:26 | `b3db1e0e` / `e46cafe3` | segment 接管；sparse segment timing | — |
| 05:37 | `b06dea5d` | segment 诊断 | **SIGABRT**（allocator），再分配只读 RCA |
| 05:48 | `47e7eaaf` | allocator inventory | 再分配 native stack 观察 |
| 12:07 / 12:13 | `0d72782b` / `8ecc3f93` | file-based native-stack control；GDB 观察 | **观察被截断**，再分配 interval correction |
| 12:25 | `781e902c` | interval metrics 单位验证 | — |
| 12:32 | `dc653845` | prefill control | **SIGSEGV（−11，20.37 s / 30 GiB）**，再分配 native-stop 观察 |

[实测] 子任务 `progress.md:123-142`（"Current session handoff — 2026-10-25" 一节）的 handoff 中，后续动作全部是 trace 审计：interval 修正、GDB、R5 24/32、R6/R7。主任务 `plan.md:134-136` 写的下一 session 路线是 `{… S4 observe/control -> S5 audit} ‖ {P0 metrics diagnosis -> …}`，同样都是诊断。

### 4.4 R5 32/32 是否阻塞主线

[推断] **不阻塞。** 理由如下：

- 缺少的 8 个 rank 是 P0 这一个 prefill DP 组（TP8）。另一个 prefill DP 组 P1 和两个 decode 组 D0/D1 的数据已经完整。P0 与 P1 的拓扑相同，路由都是 balanced_topk，处理的请求形状都是 6144 tokens，所以 P1 的数据可以代表 prefill 成本。
- 即使补齐 32/32，D-V4-KERNEL-ONLY 路线后面还有 R6/R7 和成本计算；而且只要第 3.2 节的 guard 不解除，Frontier 仍然无法运行 PD。补齐 R5 不会推进 H4（Frontier 完整执行）或 H6（对比与验收）。
- **这个推断的验证限制：** P0/P1 对称没有做过逐 kernel 的数值比较。

### 4.5 违反了哪些已有决定

- **D-MAINLINE-LOOP-CAP**（`requirements.md:377`）规定，一个诊断问题用掉超过 3 次 GPU 作业或约 6 小时而没有推进里程碑，就必须停下来：要么延后，要么上报 YC。
  - [实测] 围绕 "kernel-only 完整归属" 这一个问题，10-23 起有 6 次 GPU 作业（3×32 卡 + 3×1 卡），CPU 循环持续 13 小时。H3/H4/H6 都没有推进，也没有上报记录。
  - [实测] 这条决定的文字是 "toward trained GLM PD and calibration"。GLM 暂停后没有改写成 DSV4 的说法，执行者可能因此认为它不适用。
- **D-MEASUREMENT-FIRST**（`requirements.md:1066`）要求 "停止 CUPTI 扩展调查"。
  - [实测] 10-24 的 graph callback（resourceDescriptor 解引用）、stream probe 和 scope-lifetime/TLS probe 都属于 CUPTI 机制调查。
  - [实测] D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY（子任务 `requirements.md:148`，"批准，类似请求无需申请权限，皆授权"）只是授权"可以做"，并没有改变优先级；它的原文写明 "does not change native/C26/calibration predicates"。
- **执行规则 / Record keeping**（`requirements.md:379-380`）要求 "progress.md 按里程碑更新，不按 slice 更新"，并 "停止按 slice 复制快照"。
  - [实测] 子任务 `summary.md` 在 10-25 一天就有 18 条 Modification History，每条都对应一个 slice。
- **plan.md:305** 写明 `{H1 preparation, H2, H4 source preparation}` 可以并行。
  - [实测] H4 的 Frontier 源码准备 10-24 起是 0 行，这条并行许可没有被利用。

### 4.6 根因

1. **验收链完全串行，而且以"完整"优先。** [实测]
   - spec 规定 "missing coverage remains a typed failure" 且 "First-tail, compiled and physical-TP qualification are not waived"（`spec.md:158-161`）。
   - plan 的顺序是 H3 → H4 → H6（`plan.md:300-308`）。
   - D-V4-KERNEL-ONLY-GPU-COST（子任务 `requirements.md:71`）要求 serving 成本具备 "complete attribution plus explicit GPU-stream/dependency accounting"。
   - [推断] 结果是在第一个误差数值出现之前，所有工作都集中在上游的证据资格认证上，而且没有任何信号能说明"已经足够"。
2. **60 s / 30 GiB 限制被用在了不合适的场景。**
   - [实测] 这个限制来自 CLAUDE.md 的 "Limit memory-heavy inspections"，在子任务 `profile01/results/command.md:31` 被写成 `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s`，然后用到了一个 rank 约 120 万行 trace 的完整指标计算上。
   - [推断] 完整计算一个 rank 的 trace 属于离线数据处理，不是"检查"。正确做法是问一次 YC 能否放宽限制，而不是花 13 小时把计算压进 60 秒。
3. **流程开销过高。**
   - [实测] 每次修改都有单独的行数预算（例如 80 行、160 行），每次尝试都要"分配 → 冻结 → 双 review → acceptance JSON → 计数器（例如 controls 2/2 spent）"。一次失败通常产生 2–3 次记录类提交。10-24 起 `task_memory/` 增加约 105 万行。
4. **检查点不报告里程碑。** [实测] 10-23 起没有一次对 YC 的检查点是先报告 "H3/H4/H6 有没有推进" 的。D-MAINLINE-LOOP-CAP 要求 "Checkpoints report milestone movement first"。

## 5. Spec / tickets 审查发现

### 5.1 High

**H-1：V4 PD 执行路径没有实现，也没有有效的负责人。**

- 依据：
  - [实测] 第 3.2 节列出的 4 处 guard 和 CPU 探测结果。
  - [实测] ticket 05 第 314 行写着 "M4 (gap): the V4 structural path rejects PD … follows `09.full_frontier_path[GLM]`"，但 GLM 已暂停，这个前置条件永远无法满足。ticket 05 第 441 行 "Define and implement V4 structural PD on the existing transfer owners…" 仍然是 `[ ]`。
  - [实测] grouped NIXL 管道已经存在（`frontier/events/kv_cache_transfer_start_event.py:78-210`、`frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py:19-143`、`vllm_v1_kv_allocation.py`），但 `tests/unit/test_dsv4_structural_pd.py::test_grouped_transport_does_not_admit_v4_execution` 明确断言它不允许 V4 执行。
- 影响：H4 无法开始，也就不可能有任何预测。
- 建议：在 ticket 05（或 07）中把 V4 PD 设为 DSV4 自己的里程碑，删除对 GLM 的依赖，接到现有的 grouped NIXL 管道上。

**H-2：缺少"首次预测对比"里程碑，误差信号被放在最后。**

- 依据：
  - [实测] `spec.md:161`（Delivery order）要求先完成原生准入和 non-dummy 执行，再做对比。
  - [实测] `plan.md:300-303` 的顺序是 H3 → H4 → H6。
  - [实测] ticket 19 第 95 行 "Where a metric misses, localize the cause" 本身就需要先有一次对比。
  - [实测] ticket 19 第 85–97 行全部是 `[ ]`。
- 影响：精度矫正的反馈回路从来没有建立。所有资格认证工作都无法用误差判断"值不值得做"。
- 建议：新增 H4a "DIAGNOSTIC 首次对比"。它可以使用已接受的实测表加上 profile04 的 step 时间，结果明确标注为不计入 10%/30% 验收，只用来定位误差的主要来源。

**H-3：成本路线与 P0 decode 的实际运行形态不匹配，而且"完整"没有边界。**

- 依据：
  - [实测] 实测数据读取器（`deepseek_v4.py:640-648`）拒绝 graph mode，isolated decode 只支持单个请求。
  - [实测] P0 的 decode 实际运行在 PIECEWISE+FULL CUDA Graph 下。
  - [推断] 用 Little 定律估算：到达率 1 req/s，E2E 约 19.8 s，所以 decode 阶段约 19 个并发请求，分到 2 个 DP lane，每个 lane 约 10 个。
  - [实测] D-V4-KERNEL-ONLY-GPU-COST 要求完整归属和 stream 依赖核算。
  - [推断] 在 CUDA Graph replay 下，逐 kernel 时间之和不等于 step 时间。真正决定 TPOT（进而决定 E2E）的是"约 10 请求 × context 6143–7167"下的 graph step 时间。
- 影响：H3 不会自然收敛，H4/H6 会被一直推迟。
- 建议：请 YC 决定，graph decode 第一轮是否使用 profile04 D0/D1 实测的 graph replay step 时间（按 batch size 和 context 分桶）作为成本，并把逐 op KERNEL_ONLY 降为后续细化工作。

**H-4：没有 Frontier DSV4 P0 运行入口。**

- 依据：
  - [实测] 仓库中没有 DSV4 P0 的 Frontier driver 或 config。`canonical_fixed_6144_1024` 只被原生侧和 manifest 测试引用。
  - [实测] `data/profiling/compute/h800/` 下只有 `zai-org/GLM-5.3`，没有 `deepseek-ai/DeepSeek-V4-Flash`。
- 影响：即使 H-1 修好，也没有可以直接运行的对比命令。
- 建议：在 H-1 之后补一个 P0 config/driver，复用 `tests/comparison/e2e_gate/` 的 normalize/gate CLI（ticket 19 已经绑定 `gate: client`）。

### 5.2 Medium

- **M-1：V4 实测表不在规范位置。**
  - [实测] 已接受的表放在 `task_2026-10-22_v4_balanced_routing/prefill_results/accepted/.../attention.csv` 和 `/data/ycfeng/tmp/glm53_dsv4/w6_dsv4_profiles/`（attention、linear_op、moe 以及 kernel_only 变体）。
  - [推断] `/data/ycfeng/tmp` 是临时空间，有丢失风险，而且 Frontier 默认按 AGENTS.md 的 `data/profiling/compute/<device>/<model>/` 路径读取，现在读不到。
- **M-2：P0 内存 / KV 容量预算未完成**（ticket 17 第 58 行 `[ ]`）。
  - [推断] 第一轮可以用 AGENTS.md 中已有的 `explicit` 模式，直接使用原生实测的 grouped 54740 blocks。这属于实测绑定，不是任意缩放。
- **M-3：EP16 / sequence-parallel 下的 V4 payload 通信准入未完成。**
  - [实测] `data/profiling/network/h800_dgx/summary.csv` 只有 group size 4/8/32。group size 2–32 的曲线在 `glm_p0_ep16_20261021b/summary.csv`（1241 行），但目录名属于 GLM。`nixl_summary.csv` 有 186 行。
  - 需要确认 V4 payload 的大小范围是否在这些曲线覆盖之内。
- **M-4：P1 64 卡 basic 验证没有启动。**
  - [实测] ticket 10 第 93 行允许它在 P0 达到 30% 之前开始；H800 资源已获常设授权（D-V4-H800-RJOB-AUTHORITY）。
- **M-5：ticket 05 第 457 行与 balanced 路由决定可能冲突。**
  - [实测] 该行写 "Hash-layer routing … the gate load is never silently replaced by balanced"，而当前子任务规定 "Preserve balanced_topk"，原生参考本身也使用 balanced_topk。
  - 需要明确：hash 层在原生运行中的实际路由方式，以及这一行是否已经过时。
- **M-6：范围缩小没有写入记录。**
  - [实测] `plan.md:292-294` 仍然列出 P2 H200 32/64 的 formal/basic 用例，spec 的资源规则也仍然包含 H200。
  - 建议在 `requirements.md` 中记录用户 10-25 的决定："仅 H800 32/64 DSV4"。
- **M-7：D-MAINLINE-LOOP-CAP 需要改写成 DSV4 的版本。** 见第 4.5 节。

### 5.3 Low

- **L-1：ticket 10 状态过时。**
  - [实测] 第 21–30 行的 "Current execution status" 仍然链接到 10-22 的 checkpoint。
  - [实测] 第 81 行 "Collect and admit three clean native repetitions" 仍是 `[ ]`，但 native06 R4 已经完成。
- **L-2：ticket 18 文件名过时。** [实测] 文件名是 `18-native-ground-truth-h200.md`，内容标题已经是通用的 "Priority-case native validation"，H800 的证据也放在这里。
- **L-3：ticket 17 两组勾选状态看起来矛盾。** [实测] 第 53–57 行是带 "P0 only" 范围的 `[x]`；第 64–85 行是原始清单，覆盖同样的绑定项，却仍是 `[ ]`。第 44 行有解释，但读者容易误解。
- **L-4：ticket 文件过大。** [实测] ticket 05 有 87 KB，09 有 62 KB，01 有 49 KB，当前状态和历史快照混在一起。
- **L-5：worktree 难以追溯。** [实测] 有 218 个 worktree，其中 75 个 detached；通过的结果按内容合入，没有保留祖先关系。
- **L-6：警告文字过时。** [实测] `frontier/model_architectures.py:702` 仍然打印 "compressed attention, mHC and mixed precision are unsupported…"，和已经部分支持的现状不一致。
- **L-7：有误提交的风险。** [实测] 主 worktree 根目录的未跟踪目录（`.cortexkit/`、`.pi/`、`remaining_measurements/`）可能被误加入提交。

## 6. 已核实正确或有价值的工作

- [实测] native06 R4 三轮一致性好（第 3.1 节）。ticket 17 第 53–57 行的 P0 绑定项与 `client_report.md` 一致：128 个请求、6144/1024、16 个 warmup × 3 轮、窗口约 146.25 s、P16+D16/TP8/DP2/EP16。
- [实测] ticket 19 的 `canonical_metric_binding [x]`：复用了 `first_decode_token_completed_at` 和 `gate: client`，没有另写一套指标逻辑。
- [实测] grouped NIXL 管道和单元测试已经存在。balanced routing 决定、balanced prefill、decode q1 和 MoE isolated 数据都已被接受，可以直接用于首次对比。
- [实测] 缺少数据时一律报出明确的 typed error，没有静默回退。这一纪律对保真度是正确的，只是现在被用到了阻止任何对比的程度。

## 7. 下一步建议（按顺序）

1. **立即停止 R5 trace-audit 循环**（由并发 session 执行）。
   - 在 `future.md` 中把 R5 24/32 记录为"P0 prefill 成本使用对称的 P1 数据，P0 rank 补齐延后"。
   - 如果 YC 认为必须补齐 32/32，只申请一次决定：允许一次离线处理放宽限制（例如在 CPU 节点或 worker 上 ≤15 分钟 / ≤120 GiB），只尝试一次，失败即停。
2. **把下一个里程碑设为 H4a："Frontier DSV4 P0 首次诊断性预测"。** 运行 68f891b 的 128 个请求，与 native06 R4 对比，输出四个指标的误差，结果标注为 DIAGNOSTIC。工作项：
   - a. 在 PREFILL/DECODE 角色中启用 V4：通过显式 config 放开 `deepseek_v4.py:101` 的 co-location guard、`predictor deepseek_v4.py:457` 的 KV_CONNECTOR guard 和 `families.py:646` 的 family gate，并接到现有的 grouped NIXL 管道。
   - b. 成本输入：
     - prefill 使用已接受的 balanced prefill attention 表和 profile04 P1 的 step 时间；
     - decode 使用 profile04 D0/D1 的 graph replay step 时间（按 batch size 和 context 分桶）；
     - MoE/linear 使用 `w6_dsv4_profiles`；
     - 通信使用现有的 H800 曲线。
   - c. KV 容量：使用 `explicit` 模式，取原生实测的 54740 blocks。
   - d. P0 driver：复用 `tests/comparison/e2e_gate/` 的 normalize/gate CLI。
   - [推断] 预计涉及 5–8 个 `frontier/` 文件，超过子任务 "5 个文件 / 500 行" 的默认上限，也触发 CLAUDE.md 中"超过 5 个文件"的审批规则，需要 YC 批准。
3. **请 YC 一次性决定 3 个问题：**
   - (a) 是否允许在 H3 完整资格认证之前做 DIAGNOSTIC 对比，结果不计入验收；
   - (b) graph decode 第一轮是否接受 step 级实测成本，代替逐 op KERNEL_ONLY 的完整归属；
   - (c) 在 `requirements.md` 中记录"仅 H800 32/64 DSV4"，从当前计划中移除 H200 P2。
4. **修正 tickets：**
   - 05：删除 M4 对 GLM 的依赖，给 V4 PD 指定负责人；核对第 457 行与 balanced 决定是否冲突。
   - 19：新增 H4a 首次对比里程碑。
   - 10：更新状态链接，勾选原生三轮这一项。
   - 把 D-MAINLINE-LOOP-CAP 改写为针对 DSV4 的 H3/H4/H6。
5. **把 V4 实测表复制到规范路径** `data/profiling/compute/h800/deepseek-ai/DeepSeek-V4-Flash/`，赶在 `/data/ycfeng/tmp` 被清理之前完成。
6. **并行启动 P1 H800-64 basic 原生验证。** ticket 10 第 93 行允许这样做，H800 资源也已获常设授权。
7. **调整流程：**
   - 检查点先报告"哪个里程碑推进了"；
   - 停止按 slice 写 summary 的 Modification History；
   - 把 tickets 中的历史快照移到单独的归档文件。

## 8. 验证限制

- 没有运行 GPU，也没有重新分析 profile04 数据。P0/P1 prefill 对称（第 4.4 节）是根据拓扑和 balanced 路由做的推断，没有逐 kernel 比较。
- 每个 lane 约 10 个并发请求是 Little 定律的估算，不是从 batch log 读出的实测值。
- 没有逐行审查约 140 万行的 `task_memory/` 证据。重点审查了主任务记录、tickets 05/10/17/18/19、子任务记录和代码 guard。
- `task_memory/` 的行数统计包含 JSON/CSV 证据副本，不全是人工撰写的文档。
- 审查期间 HEAD 一直在前进，结论对应 `dc653845`（10-25 12:32 +0800）。

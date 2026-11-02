Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | 核对 D-P2-64-FORMAL-B 当前阶段；确认 S5 旧状态并修正当前入口，保留历史验收。 |

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 3｜TODO 1

范围：ticket 18/19 的 D-P2-64-FORMAL-B 当前阶段；不把阶段完成视为整张 ticket 完成。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 19：P2-32 晋级 receipt | PASS | `1a7950ed`，24/24 ≤30%，最大 5.607176%；07:38:49Z 签发。 |
| 19：此前 35 个 job 的 GPU-hours | PASS | `2bdf9fed`，19 个缺失时钟补齐，总计 158.271667 container GPUh。 |
| 18：Q5 harness、build、check-only | PASS | `4861e680` / `0ee7f6c2`，28 harness tests、25 tool tests、完整 build/control 已验收。 |
| 18：单次正式提交及身份核对 | PASS | `exp-1102-164536-160317`，08:45:36Z，64 H200 全部 preemptible。 |
| 18：正式 native execution | BLOCKED | 10:29:23Z 的 8 个 replica 全部未分配节点，调度器反复拒绝整个 co-scheduling group。 |
| 18/19：terminal、copy、coverage、R4、gate、warm | BLOCKED | 依赖正式 job 结束并产生业务输出；工具与旧数据 control 已就绪。 |
| 18：已接受的 harness 整合到最终分支 | TODO | Pi `p64-mainline-integration-1102c` 正在保留现有 profiling/shutdown 代码并验证五文件移植。 |
| 19：新 job GPU-hours 与 P2-64 case verdict | BLOCKED | 尚无容器起止时钟或新参考指标；不能按 64 GPUh 上限计费或宣告 PASS。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** Q5 已验收提交 `0ee7f6c2` 至 2026-11-02 10:35Z｜**当前前沿：** 18 的已提交 native job、harness 最终分支整合；19 当前记录核对。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 沿用 Q5 已通过 control 的工具；本轮只核对现有产物并整合已接受代码。 |
| D2 细节反复重试 | CLEAR | 正式 job 只有一次提交；没有因 Starting 另起 job，调度诊断检验了明确的启动原因。 |
| D3 支线问题修补 | CLEAR | 调度原因、当前状态和最终分支整合均属于 18/19 的现有验收路径。 |
| D4 可并行任务停滞 | CLEAR | 原 Pi 监控持续运行，独立诊断已完成，主分支整合与记录修正并行。 |
| D5 超范围镀金 | CLEAR | 只整合已接受的五文件改动；没有新模型、参数、表或运行场景。 |
| D6 分析空转 | CLEAR | 一次精确 job/replica/Event 调查确认 co-scheduling 阻塞，未知资源类型明确保留；调查到此结束。 |
| D7 路线违背 | CLEAR | 平台操作走 `/ds4-exec`；receipt 早于 build 和提交；保留原 launcher 和冻结输入。 |

**文档检查：** S1/S2/S3/S4/S6 在当前阶段未发现缺陷。S5：修正前 `plan.md` 当前路线、`progress.md` 当前 handoff、`summary.md` 当前状态和 ticket 19 当前 checkpoint/Status 仍写 direct Codex RJob、待用户选择 A/B、缺失 GPU 时钟或旧两处修正审批；与 D-P2-INPUT-QUALIFICATION、D-P2-64-FORMAL-B 和 `1a7950ed` / `2bdf9fed` / `0ee7f6c2` 矛盾。旧比较表保持历史含义。
**归因：** 无执行偏离；S5 是需要直接修正的文档缺陷。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续。

**文档更新（先执行）：**
1. `requirements.md`：记录本次原始续作请求，说明已存在的单次提交；不增加授权（可直接执行，已应用）。
2. `plan.md` / `progress.md` 当前入口：更新 receipt → Q5 → submitted → native → post-run 依赖，说明 co-scheduling 阻塞与独立整合工作（可直接执行，已应用）。
3. ticket 19 / `summary.md` 当前状态：移除过时的待决策与缺时钟断言；保留旧 reference 未获追溯晋级、新 job 用量待补和全 ticket OPEN（可直接执行，已应用）。

**需停止的偏离行为：无**

**接下去的路线：**
1. 18：沿用现有 Pi 监控已提交 job → 完成标志：native 业务输出及 terminal evidence；若真实失败，先做 exact-job RCA。
2. 并行：18 的五文件主分支整合与回归 → 完成标志：保留已接受行为和主分支现有逻辑，冻结 package/harness 不变，验证后提交。
3. 18/19：terminal receipt → hash-identical copy → pair coverage soft gate → R4 → all-request/warm readings → 完成标志：逐轮报告四指标与独立 10% 判断，保留任何未覆盖/失败。
4. 19：加入新 job 实际 container GPU-hours 并更新 case verdict；H800 先准备具体 package/用量再请求已有规则要求的批准。

## 4. 最值得关注的 3 点

1. 正式 job 已经存在；Starting 是整组未调度，不能重复提交，也不能计作运行失败或 64 GPUh 消耗。
2. 当前唯一平台阻塞有事件证据；具体缺哪种资源和启动时间未知，继续保持原 launcher。
3. P2-64 新参考的覆盖、R4 和逐轮四指标还没有结果；CPU package PASS 与旧参考的 warm PASS 都不能代替它们。

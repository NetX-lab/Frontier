Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | harness 最终分支整合已验证；记录并停止 D1 额外测试，修正 ticket 18 的 S5 旧路线。 |
| 2026-11-02 | 核对 D-P2-64-FORMAL-B 当前阶段；确认 S5 旧状态并修正当前入口，保留历史验收。 |

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 3｜TODO 0

范围：ticket 18/19 的 D-P2-64-FORMAL-B 当前阶段；不把阶段完成视为整张 ticket 完成。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 19：P2-32 晋级 receipt | PASS | `1a7950ed`，24/24 ≤30%，最大 5.607176%；07:38:49Z 签发。 |
| 19：此前 35 个 job 的 GPU-hours | PASS | `2bdf9fed`，19 个缺失时钟补齐，总计 158.271667 container GPUh。 |
| 18：Q5 harness、build、check-only | PASS | `4861e680` / `0ee7f6c2`，正式 package 已接受；整合后 frozen check-only 仍 PASS。 |
| 18：单次正式提交及身份核对 | PASS | `exp-1102-164536-160317`，08:45:36Z，64 H200 全部 preemptible。 |
| 18：正式 native execution | BLOCKED | 10:45:58Z 仍 Starting/8 pending；10:29:23Z 的 replica/events 证明整组未调度。 |
| 18/19：terminal、copy、coverage、R4、gate、warm | BLOCKED | 依赖正式 job 结束并产生业务输出；已有工具与 control。 |
| 18：已接受的 harness 整合到最终分支 | PASS | 六文件已验证：136 scoped PASS、两个 mock 修正后 2 PASS、native profile 36 PASS、frozen/hash/scope PASS；见 integration test report。 |
| 19：新 job GPU-hours 与 P2-64 case verdict | BLOCKED | 尚无容器起止时钟或新参考指标；不能按 64 GPUh 上限计费或宣告 PASS。 |

## 2. 偏离检测

**结论：** 已偏离：命中 D1；已停止额外测试并回到正式 job 的后续路径。
**检测窗口：** 前次 checkpoint 2026-11-02 10:35Z 至整合验证完成｜**当前前沿：** 18 的已提交 native job；19 的后续验收依赖其输出。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | HIT | Pi 在 warmup/receipt 和现有 profiling/shutdown 回归之外重复扩大 native/CUPTI suite；没有新的验收依赖。`integration_20261102/pi_verification.json` 保留 7 次测试调用；lead 已停止 executor。 |
| D2 细节反复重试 | CLEAR | 两项 mock 失败的原因已定位，一次修改后定向 PASS；没有多次改动猜测原因。额外 suite 重跑计入 D1。 |
| D3 支线问题修补 | CLEAR | 没有修改 CUPTI 或其他旁支代码；错误保留为未调查，未继续修复。 |
| D4 可并行任务停滞 | CLEAR | 原 GPU 监控与 harness 整合并行；整合已完成，post-run 确实依赖业务输出。 |
| D5 超范围镀金 | CLEAR | 交付只有已接受的五文件移植和必要的一处旧 mock 调整。 |
| D6 分析空转 | CLEAR | 没有重开调度原因调查；整合已从 TODO 变为 PASS。 |
| D7 路线违背 | CLEAR | 平台操作仍走 `/ds4-exec`，只有一次 GPU 提交；原 launcher、监控和 frozen inputs 均保留。 |

**文档检查：** S1/S2/S3/S4/S6 未发现新缺陷。S5：ticket 18 的 current checkpoint 仍写 direct Codex RJob 和旧 handoff，矛盾于 requirements D-P2-INPUT-QUALIFICATION；已直接修正。其他当前入口已同步为整合完成、native/post-run 未完成。
**归因：** D1 为执行偏离，明确的相关回归范围本可避免额外 suite；S5 是独立的旧文档缺陷。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续。

**文档更新（先执行）：**
1. `tickets/18-native-ground-truth-h200.md` 当前 checkpoint：改为 `/ds4-exec`、单次正式 job 和当前 handoff，保留历史 references（可直接执行，已应用）。
2. `plan.md` / `progress.md` / `summary.md`：将 harness 最终分支整合改为已验证，引用 test report，保留 native/post-run 和新 job 用量未完成（可直接执行，已应用）。

**需停止的偏离行为：**
- D1：停止重复扩大回归套件；只保留现有通过证据，完成整合提交。`p64-mainline-integration-1102c` 已停止，exit 130；GPU launcher/monitor 未停止。

**接下去的路线：**
1. 18：保持已有 launcher/监控，等待整组调度 → 完成标志：native 业务输出及 terminal evidence；真实失败先做 exact-job RCA。
2. 18/19：terminal receipt → hash-identical copy → pair coverage soft gate → R4 → all-request/warm readings → 完成标志：逐轮报告四指标与独立 10% 判断，保留任何未覆盖/失败。
3. 19：加入新 job 实际 container GPU-hours、更新 case verdict。H800 需先完成具体 package/用量方案，再请求批准；GLM/EP 修正保持暂停/延期。

**暂缓事项：** 额外 CUPTI suite 的 25 个错误原因 UNKNOWN，不声明 baseline/environment failure → 已补入 `future.md` 原 CUPTI 主题。

## 4. 最值得关注的 3 点

1. 当前六文件整合已有足够验收证据；额外测试停止，下一交付是提交，不再扩大检查。
2. 正式 job 的外部阻塞已有事件证据；保持原 launcher，既不重复提交，也不把排队时间计作 GPU-hours。
3. 新 P2-64 的 coverage、R4 和四指标尚无结果；ticket 18/19 仍 OPEN。

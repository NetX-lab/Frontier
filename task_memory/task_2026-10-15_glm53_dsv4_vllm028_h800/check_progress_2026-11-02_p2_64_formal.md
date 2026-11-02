Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | RCA 与独立 CPU 复现完成；保留时钟缺口，停止可选控制实验的反复扩展，下一步为具体恢复方案审批。 |
| 2026-11-02 | 以 Failed 终态和 74/74 文件复制校验替换排队状态；RCA、CPU 复现与用量核对继续。 |
| 2026-11-02 | harness 最终分支整合已验证；记录并停止 D1 额外测试，修正 ticket 18 的 S5 旧路线。 |
| 2026-11-02 | 核对 D-P2-64-FORMAL-B 当前阶段；确认 S5 旧状态并修正当前入口，保留历史验收。 |

## 1. 规划完成情况

**汇总：** PASS 8｜BLOCKED/FAILED/SKIPPED 3｜TODO 1

范围：ticket 18/19 的 D-P2-64-FORMAL-B 阶段；阶段完成不代表整张 ticket 完成。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 19：P2-32 晋级 receipt | PASS | `1a7950ed`，24/24 ≤30%，最大 5.607176%。 |
| 19：此前 35 个 job 的 GPU-hours | PASS | `2bdf9fed`，158.271667 exact container GPUh。 |
| 18：Q5 harness、build、check-only | PASS | `4861e680` / `0ee7f6c2`，已接受。 |
| 18：单次正式提交及身份核对 | PASS | `exp-1102-164536-160317`，64 H200 全部 preemptible。 |
| 18：正式 native execution 第一代 | FAILED | 12:50:51Z Failed；没有 warmup/formal 请求结果。 |
| 18：失败产物复制 | PASS | 0-GPU job Succeeded；lead 独立校验 74/74 文件与 inventory。 |
| 18：exact-job RCA 与 CPU 复现 | PASS | 校正 RCA；lead 复现 rank0 换端口、peers 留在旧端口；原因边界保留。 |
| 18/19：terminal receipt 与完整用量 | TODO | 终态已归档；8/8 starts、1/8 finishes，精确总量未知。 |
| 18/19：coverage、R4、gate、warm | BLOCKED | 第一代没有请求数据，须完整新参考。 |
| 18：harness 最终分支整合 | PASS | `e2289283`，此前定向测试保留，不重开。 |
| 18：具体恢复方案与成本 | PASS | `failure_01_20261102/recovery_proposal.md` 已可审；实现和第二代须 YC 决定。 |
| 19：新 P2-64 case verdict | BLOCKED | 不能从启动失败产物产生指标。 |

## 2. 偏离检测

**结论：** 已偏离：命中 D1；纠正已执行。
**检测窗口：** 前次 harness 整合 checkpoint 至本次失败调查收尾｜**当前前沿：** 18 的恢复方案决策；19 的缺失结束时钟。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | HIT | 核心碰撞已在 standalone、canonical_1 和 lead 对照复现，Pi 仍扩展可选 coord-store 断言并反复跑 aggregate suite；已停止该执行器，exit130。 |
| D2 细节反复重试 | CLEAR | 本窗口重复辅助试验计入 D1；无第二次 GPU 运行、timeout 调参或生产修复循环。 |
| D3 支线问题修补 | CLEAR | 未恢复 CUPTI/GLM/EP 支线；调查绑定 ticket18 的启动故障。 |
| D4 可并行任务停滞 | CLEAR | 平台 RCA、复制、CPU 复现并行；lead 独立复跑和整理证据。 |
| D5 超范围镀金 | CLEAR | 正式交付未增加接口或配置；native 修复只是待批准方案。 |
| D6 分析空转 | CLEAR | 形成可重复故障、明确证据边界与恢复选择；辅助反复计入 D1。 |
| D7 路线违背 | CLEAR | 平台经 `/ds4-exec`；未把 preemptible 当抢占，未改冻结源或提交第二代。 |

**文档检查：** S1/S2/S3/S4/S6 未发现新缺陷。S5：plan/progress/summary/tickets18/19 原当前段仍称 RCA/copy pending，已按失败报告同步；精确用量继续 OPEN。
**归因：** D1 执行偏离；既有 assignment 要求在复现及有限结论后停止，未授权继续打磨。S5 为成果落地后的状态滞后。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续。

**文档更新（先执行）：**
1. 当前 plan/progress/summary/tickets18/19：记录 RCA/copy/CPU 证据完成、结束时钟不足和新参考依赖（可直接执行，已应用）。
2. requirements：记录恢复提案待批准，保留 D-P2-64-FORMAL-B 三项变更范围；不补造用户决定（可直接执行，已应用）。
3. native 启动修复与第二代：仅在 YC 明确批准后写入决定并执行（需用户确认）。

**需停止的偏离行为：**
- D1 工具打磨：停止可选控制试验和全套辅助脚本的反复扩展；保存成功与失败，采用已完成的独立对照作机制证据（已执行）。

**接下去的路线：**
1. 18：提交已核验的失败证据、报告和恢复方案 → 完成标志：可审方案与范围/成本完整；之后向 YC 请求具体决定。
2. 18：获批后只修 native 启动协调，验证 CPU 通信/端口/生命周期并冻结新 generation → 完成标志：受影响检查和 check-only 通过后再提交正式运行。
3. 18/19：成功参考的 terminal/copy → coverage soft gate → R4 → all-request/warm 四指标验收；19 的缺失时钟可独立补齐，当前估算不能关闭精确计量。

**暂缓事项：** CUPTI 25 个错误、GLM、EP count 保持 `future.md` 状态；可选 coord-store 的 Gloo 失败保留在本次报告，作为获批修复必须验证的生命周期风险；不单开泛化调查。

## 4. 最值得关注的 3 点

1. 第一代在请求前失败；已复现端口分歧，但实际占用者未知，不能直接使用 infrastructure retry 授权。
2. 本次约 34.02 GPUh 是估算；七个 finish clocks 缺失，旧35-job精确账保持158.27，ticket19不关闭。
3. 恢复方案仅涉及 native 启动协调与新 generation；批准后沿原64 H200/3600s上限推进，原失败证据和已完成 replay 保留。

# Scope-transition self-check

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Apply the user-approved transition from closed R2 regression to D20 CPU implementation. |

## 1. 规划完成情况

**汇总：** PASS 1｜BLOCKED/FAILED/SKIPPED 1｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| R2 当前 P1 regression/disposition | PASS | cb09a27、636 tests、独立复审通过，分支保留于 b6ba7e5 |
| D20 A 接口与生命周期定位 | TODO | event/provider 已读，scheduler trace 正在并行完成 |
| D20 B,C measured PD 实现 | TODO | 已划定独占代码与明确拒绝边界 |
| D20 D integration tests | TODO | 使用用户指定的真实 event/scheduler seam |
| D20 E 实际 measured main | BLOCKED | 计算 owner 的 provider/bundle 集成尚待提供 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** R2 P1 follow-up review 通过至今｜**当前前沿：** D20 A/B/C CPU 实现

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | R2 当前回归闭环后停止；不扩展 staging |
| D2 细节反复重试 | CLEAR | P1 的两个新 findings 有独立 RED、修复和复审结论 |
| D3 支线问题修补 | CLEAR | 当前切换由用户明确要求 |
| D4 可并行任务停滞 | CLEAR | scheduler trace 与 event/provider 检查并行；三条实现流按文件隔离 |
| D5 超范围镀金 | CLEAR | 只做现有 PD owners，排除第二引擎和 observer |
| D6 分析空转 | CLEAR | 当前定位直接服务 measured-mode guard 和 first-D shape 实现 |
| D7 路线违背 | CLEAR | D20 已明确解除 CPU 实现等待，native 验收仍保留 |

**文档检查：** S1–S6 均未发现明显缺陷；本任务 plan 明确区分实现依赖与 native 验收依赖。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动

**需停止的偏离行为：** 无

**接下去的路线：**
1. A/B/C：在现有 transfer events 和 scheduler 中完成受支持 measured lifecycle → 完成标志：真实 owners 可执行且 missing coverage 明确拒绝。
2. 并行：PD CPU fixture 集成 ‖ computation owner 提供共享 profile/provider 接口。
3. D/E：提交 affected regression；provider 就绪后执行小型实际 main → 完成标志：命令、输出、token/ownership 平衡可核验。

## 4. 最值得关注的 3 点

1. R2 P1 已关闭；继续投入新的 R2 收集或工具扩展将不再推进当前主线。
2. 默认 analytical bandwidth 不能因删除 mode guard 就变成 measured NIXL 证据。
3. Step 首次 D 的 16..31-token tail 必须作为明确成本形状处理；CPU fixture 不能代替 T2/native 验收。

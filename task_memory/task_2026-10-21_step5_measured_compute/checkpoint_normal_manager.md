## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Self-check after repeated normal-manager test-assertion failures. |

## 1. 规划完成情况

**汇总：** PASS 1｜FAILED 1｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| CPU minimum | PASS | 1ca09be;真实拟合/cache/lookup/cost |
| Complete physical compute | TODO | routed replay12 PASS;public constructor并行验证 |
| Bundle/cache | FAILED | normal manager完成fit/pub/load,末尾private-bucket断言未通过 |
| Cost/proof integration | TODO | modality/count ownership已实现,待最终集成检查 |
| Verification/handoff | TODO | scope suite和最终commit未完成 |

## 2. 偏离检测

**结论：** 已偏离：命中 D2
**检测窗口：** 1ca09be 至今｜**当前前沿：** normal-manager acceptance、public constructor、NIXL薄接线

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 新测试均对应可执行验收 |
| D2 细节反复重试 | HIT | normal replay01/02/03反复修fixture/private registry假设,事前记录不足 |
| D3 支线问题修补 | CLEAR | round-trip parser和接口枚举确实阻断当前接入 |
| D4 可并行任务停滞 | CLEAR | public ctor独立子会话推进 |
| D5 超范围镀金 | CLEAR | NIXL薄接线由primary明确指定 |
| D6 分析空转 | CLEAR | 已交付真实FFN/runtime/bundle代码 |
| D7 路线违背 | CLEAR | 无新GPU/诊断任务或publisher |

**文档检查：** S5 requirements/progress旧文本仍将两个共享接口写作外部所有者;primary已明确授权本slice。S1–S4/S6未发现明显缺陷。
**归因：** D2 执行偏离,并非文档造成。

## 3. 纠正建议

**分支：** B：先同步已批准的ownership,不改变验收或范围。

**文档更新（先执行）：**
1. requirements.md/progress.md：记录rows/mapping ownership及NIXL薄接线,可直接执行。

**需停止的偏离行为：**
- D2：停止猜测private precision buckets；用实际published receipt registrations验证BF16/FP8/FP32,不改生产数据/阈值。

**接下去的路线：**
1. normal-manager当前fixture：公共receipt断言通过、输入mutation被拒绝。
2. 并行：public constructor acceptance ‖ lazy delegates和既有registry回归。
3. affected checks通过后提交完成slice并交接primary,不自行跑77-case matrix。

## 4. 最值得关注的 3 点

1. 实际fit/pub/load已执行；当前失败是验收读错private bucket,不是应当绕过的coverage边界。
2. routed replay12 PASS；缺失input_cast/final_residual仍明确报错。
3. 原native strict FAIL和四个10% gates完全不变。

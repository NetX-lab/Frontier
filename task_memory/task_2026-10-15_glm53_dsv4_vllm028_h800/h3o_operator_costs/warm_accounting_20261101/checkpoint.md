## 1. 规划完成情况

Modification History
| Date | Change |
| --- | --- |
| 2026-11-01 | 两个机制完成分类后的自身路线检查。 |

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| NIXL payload/lifecycle | PASS | payload_summary.json；bytes 缺陷已分类，timing 缺失已明确 |
| P-step segments/count/overlap | PASS | gap_verify.json；实际 129 kernels 对 registry1 的缺陷 |
| CPU validation/report | PASS | validation.json 7/7；report/summary 完成，core diff 为空 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** WARM-ACCOUNTING 请求至最终检查｜**当前前沿：** 返还 primary，等待修正决策

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | scoped AST 是最小 CPU 检查；不扩展到 predictor replay |
| D2 细节反复重试 | CLEAR | 一次 import 修正；未反复调整测量/模型 |
| D3 支线问题修补 | CLEAR | 两项都是 plan.md 定义的机制 |
| D4 可并行任务停滞 | CLEAR | 两项都推进，两个 native traces 并行读取 |
| D5 超范围镀金 | CLEAR | 只新增本目录证据，不实施修正 |
| D6 分析空转 | CLEAR | 两项已分类，有具体修正决策与缺失观测 |
| D7 路线违背 | CLEAR | 不调 offset/price，不删除 Frontier request0，不改 source |

**文档检查：** S1–S6 均未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 将 report、7/7 CPU 检查和准确观测需求返还 primary → 完成标志：primary 接收。
2. primary 决定两项现有路径修正；本 subagent 不继续实现或调价。

## 4. 最值得关注的 3 点

1. 两项 confirmed pricing/count 缺陷都很小，不能补掉 macro residual。
2. 逐 worker NIXL timestamps 缺失，不能据 max handle duration 调价。
3. Host-zero 是显式决定；没有当前 warm witness 就不提出宏观补价。

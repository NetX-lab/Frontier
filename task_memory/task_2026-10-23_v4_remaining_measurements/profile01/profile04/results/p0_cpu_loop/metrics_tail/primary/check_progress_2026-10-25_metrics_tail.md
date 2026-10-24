## 1. 规划完成情况

Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Timer milestone self-check; apply the separate receipt binding and proceed to the single allocated diagnosis. |

**汇总：** PASS3｜BLOCKED/FAILED/SKIPPED5｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Scope fresh gates、一次GPU、S5归档 | PASS | ffbfbc37；额度1/1，用途受限 |
| Metrics source和两次完整controls | PASS | 0d0fe648，原结果未重跑 |
| 原affected P0pilot | FAILED | exit124、缺final JSON，七个rank未启动 |
| 新timer和tiny证据审阅 | PASS | source_review.json；原receipt一行无效，独立fixture绑定保留 |
| 一次fresh timing diagnosis | TODO | allocation.json已分配，尚未执行 |
| 新correction/control/affected allocation | BLOCKED | 依赖新分段测量 |
| R5全32rank完整输出 | FAILED | 仍24/32，保留八个原timeout |
| R6完整review/synthesis | BLOCKED | 依赖完整R5 |
| R7限定archive | BLOCKED | 依赖R6 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次S5checkpoint至今｜**当前前沿：** 唯一fresh timing diagnosis

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1工具打磨 | CLEAR |98行timer是区分未测batch阶段的最小观测，证据已够，停止延伸 |
| D2细节反复重试 | CLEAR | 原pilot和tiny没有重跑；一次receipt失败按源码引用关系定位 |
| D3支线问题修补 | CLEAR | 证据绑定直接关闭timer执行前置，未改其他模块 |
| D4可并行任务停滞 | CLEAR | Scope归档与owner交付并行完成；R6/R7依赖完整R5 |
| D5超范围镀金 | CLEAR | 冻结adapter/native/package及所有判据不变 |
| D6分析空转 | CLEAR | source review已形成可执行的单次诊断决定 |
| D7路线违背 | CLEAR | 只审阅已有tiny，单次full诊断尚未用，GPU额度保持1/1 |

**文档检查：** S1–S6均未发现明显缺陷；当前记录同步timer交付和原receipt限制。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动。
**需停止的偏离行为：** 无。
**接下去的路线：**
1. Primary提交timer及证据绑定，执行一次60s/实际30GiB诊断；完成标志是关闭的stage/GC/receipt及分段测量或截断。
2. 依据结果记录单独correction/control/affected额度；完成标志是测量支持的具体范围和exact equality条件。
3. 完整R5→R6→R7；原timeout、部分index和UNKNOWN保持，未获完整输出不能推进验收。

## 4. 最值得关注的 3 点

1. 原tiny aggregate的一行无效，独立复制的fixture提供稳定顺序证据；两者均保留，未重跑。
2. batch_start之外的调用方构造成本也要看，sort和union必须分别记录。
3. 新诊断不关闭R5；Scope一次GPU也不证明native serving ownership或cost/reference。

## 1. 规划完成情况

Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | 单次segment诊断在metrics前SIGABRT；记录真实新依赖并只推进abort RCA。 |

**汇总：** PASS 8｜BLOCKED/FAILED/SKIPPED 6｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Scope gates/seal/singleGPU/limited acceptance | PASS | 原GPU1/1保持，不再运行 |
| First metrics source/controls | PASS | 0d0fe648，保持冻结 |
| First affected pilot | FAILED | 无final JSON，原失败保留 |
| First sparse timer/tiny | PASS | 原额度已用，保持冻结 |
| Construction/GC diagnosis | PASS | 已闭合诊断证据，不是R5 |
| Archive/group allocation | PASS | 4ae81052，原据保持 |
| Group source/unit/exact controls/reviews | PASS | 8b4d6993，不再执行 |
| Fresh group pilot | FAILED | 124/no final JSON |
| Seven conditional group ranks | BLOCKED | 完整pilot前置条件未满足 |
| New segment source/tiny/reviews | PASS | e46cafe3，32checks及两审零未关闭findings |
| Unique segment diagnostic | FAILED | -6/40.333549s，未到metrics，三项假设unmeasured |
| Read-only allocator-abort RCA | TODO | 新allocation明确持有阻塞项，不分配新run |
| R6/R7 retained24 preparation | PASS | b3db1e0e，无新audit |
| R5 full32 outputs | FAILED | 仍24/32 |
| R6 synthesis/R7 archive | BLOCKED | 真实依赖R5完整结果 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** e46cafe3至唯一segment诊断失败归档｜**当前前沿：** allocator-abort只读RCA。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | source gate完成即执行唯一诊断；没有额外tiny或harness扩展 |
| D2 细节反复重试 | CLEAR | 有事前的三项timing假设；新abort未检验它们，明确不推断、不重试 |
| D3 支线问题修补 | CLEAR | allocator abort实际阻塞timing测量，新RCA allocation持有该问题 |
| D4 可并行任务停滞 | CLEAR | runtime inventory sidecar与primary失败归档并行；R6准备已完成 |
| D5 超范围镀金 | CLEAR | 没有source/GC/loader/environment/native修改或新算法选择 |
| D6 分析空转 | CLEAR | compact logs生成有明确UNKNOWN和下一诊断前置条件的归档 |
| D7 路线违背 | CLEAR | 原单次诊断已用满；没有复跑或借用七rank额度 |

**文档检查：** S2新的allocator阻塞需归属、S5当前0/1状态需更新；已在requirements.md/plan.md/progress.md/summary.md及abort_rca_allocation.json应用。S1/S3/S4/S6没有支持的新缺陷。
**归因：** 无执行HIT；新运行故障带来的真实依赖已补齐。

## 3. 纠正建议

**分支：** B：补充实际abort依赖，再继续只读RCA。

**文档更新（先执行）：**
1. 当前task四份records及failure/report：单次诊断改为FAILED_SIGABRT/spent1/1，三项timing假设unmeasured（已执行，可直接执行）。
2. abort_rca_allocation.json：primary归档与sidecar单文件runtime/source inventory ownership，全部新run额度为0（已执行，可直接执行）。

**需停止的偏离行为：** 无。停止任何从零metrics观测直接选择算法修正的行为。

**接下去的路线：**
1. 归档本次失败并核对hash → 完成标志：closed日志/source/receipt一致，本地commit。
2. 只读runtime/native evidence调查 → 完成标志：可获得的core/symbol/runtime候选与UNKNOWN分别记录。
3. 只在证据支持后分配一个针对abort的新诊断 → 完成标志：具体假设、source/control/command/input/资源和单次额度先冻结；不能复跑旧脚本或放宽限制。

## 4. 最值得关注的 3 点

1. glibc检测到heap corruption，但原始写坏位置和native C stack未知；Pythonstack不能证明来源。
2. 当前诊断额度1/1已用，global/grouptiming全部unmeasured，不支持新的metrics算法选择。
3. R5仍24/32，完整R6/R7及native/cost/reference未验收；源文件PASS和GPU限定结论不变。

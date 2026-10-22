## 1. 规划完成情况

Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Core三处设计复审闭合；D30已获独立批准并实施，原D4/S1已纠正。 |
| 2026-10-22 | 来源设计闭合后，检查当前设计前沿与D30独立修复的依赖。 |

**汇总：** PASS 6｜BLOCKED/FAILED/SKIPPED 3｜TODO 2

本表仅覆盖当前D29设计与D30修复阶段；PASS表示该交付已记录完成。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| D29 source/producer trace | PASS | 原生调用、实际Batch与source/gathered来源已定位 |
| D29 native设计 | PASS | 19文件方案复审完成；实现仍待批准 |
| D29 core修订及复审 | PASS | 三处seam复审闭合；parent核算33文件/2966行 |
| D29 profile-source设计 | PASS | callable和具体emitter问题均已设计闭合 |
| D29 pricing完整设计 | TODO | native contract/reporting与既有模拟器扩展仍在修订 |
| D29联合范围与批准 | BLOCKED | core/pricing接口及完整预算未齐 |
| D30候选实现 | FAILED | 保留5个新增相邻测试失败及verdict接入缺口 |
| D30完整修复范围 | PASS | 7文件/750行方案已明确，0a0b9fe记录 |
| D30范围批准 | PASS | YC明确回答“批准，继续”，D30-SCOPE-7-750 |
| D30修复和集成验证 | TODO | 原owner正在修复；parent基线及77复用检查已完成 |
| 新generation/native及完整CLI验收 | BLOCKED | 新策略实现和合格输入尚未完成 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。前次D4/S1已纠正。
**检测窗口：** 前次本文件checkpoint至core三处设计闭合；D30旧停滞追溯0a0b9fe。**当前前沿：** D29有限pricing模型、D30修复及验证。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | emitter是来源准入缺少的具体接口；51行命令语法检查后停止扩大验证 |
| D2 细节反复重试 | CLEAR | 各轮依据预先记录的不同接口反例修订；callable和emitter已分别闭合 |
| D3 支线问题修补 | CLEAR | pricing与来源身份均属于完整新策略workflow的必要输入 |
| D4 可并行任务停滞 | CLEAR | 错误审批依赖已移除，D30获批并恢复实施；core复审和pricing同时推进 |
| D5 超范围镀金 | CLEAR | 未实施新CC查表或非必要PD-AF/MTP策略；收紧到真实接口 |
| D6 分析空转 | CLEAR | 当前前沿是设计交付；native/profile-source已改变状态，core/pricing仍有具体未决接口 |
| D7 路线违背 | CLEAR | measured CC只是未选提案，已显式保留既有collective_sim决定和CPU-gap后置顺序 |

**文档检查：** S1–S6均未发现新的活动缺陷。前次S1审批依赖、S5来源状态已更新；额外exact-NCCL前置条件已被正式plan纠正，pricing作者正据此完成有限模型。
**归因：** 当前无。历史D4由已纠正的S1导致。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需新增改动。

**需停止的偏离行为：** 无。已取消D30等待D29的审批条件。

**接下去的路线：**
1. D30修复与验证 → 完成标志：六个真实event用例及metadata consumer检查通过，无新增unit失败，独立审查和77严格回归通过后合并。
2. 并行D29有限pricing模型 → 完成标志：调度/protocol/local-work/overlap假设有限明确、待决项可回答，完整逐文件预算可核对；不把NCCL逐channel重建设为默认前置条件。
3. D29完整范围确认与实现 → 完成标志：两端位置策略、实际图重放及新来源输入验证，之后新包/native与完整CLI。

## 4. 最值得关注的 3 点

1. D30已按7文件/750行获批并实施；集成和77验收仍待结果。
2. native/core/profile-source设计已闭合；完整D29范围只等待有限pricing模型及跨owner汇总。
3. Step保留collective_sim；unit基线不是全绿，须比较新增失败。完整CLI和native验收仍开放。

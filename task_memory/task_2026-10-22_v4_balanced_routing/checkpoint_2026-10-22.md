## 1. 规划完成情况

**Modification History**

| Date | Change |
| --- | --- |
| 2026-10-22 | 更新routing范围检查；记录既成15/522越界、随后23/900批准及三个独占分工；主代理纠正D7归类。 |
| 2026-10-22 | 源码里程碑后的阶段检查；覆盖 native dispatch 与 balanced 两份 plan，共9项。 |

**汇总：** PASS3｜BLOCKED/FAILED/SKIPPED2｜TODO4

本次覆盖当前routing plan的可执行子项，不替代主任务20张票的验收。PASS只表示该子项完成。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| R1 source path/assignment analysis | PASS |6ca6777策略/原生注册入口/整数分配已审查 |
| R2a shared callers | TODO |186dfef；主代理218PASS，真实混合顺序2/7正确；复审和最终strict77待完成 |
| R2b native binding | TODO |ca1ca86为部分交付；实际来源/完整Frontier输入/新builder继续实施 |
| R2c prefill binding | TODO |adapter/构造后inventory/CSV/consumer/builder已分配独占owner |
| R3 full wiring review | TODO |共享复审进行中，native/prefill完整结果未到 |
| R4 fixed-strategy image CPU check | PASS |attempt03退出0，6144×6=36864，256专家各144；无CUDA初始化 |
| R5 fresh packages/pins | BLOCKED |等完整源码检查、各自package验证及native单写者freeze |
| R6 approved GPU runs | BLOCKED |等R5；native05保持UNVERIFIED |
| R7 aggregate scope/allocation | PASS |YC批准23/900；独占分配12/450、6/250、5/200 |

## 2. 偏离检测

**结论：** 已偏离：窗口内命中D2、D7；已停止对应行为并落实范围修订。
**检测窗口：** 上一routing阶段检查至186dfef主代理验证｜**当前前沿：** R2a复审/固定源码fidelity、R2b、R2c并行。

主线：匹配的routing执行身份→各自新包/实测→合格GPU成本与native reference→完整predictor/比较。完整任务的四指标误差仍未产出。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR |固定策略镜像检查服务R4；恢复异常后按已记录假设隔离native导入，PASS后停止。它不是未来package认证 |
| D2 细节反复重试 | HIT |caller报告保留三个fixture setup失败，事前假设记录不足；随后真实RED/修复GREEN收敛。见callers test report及progress |
| D3 支线问题修补 | CLEAR |工作均服务routing实际绑定；GLM和CPU-overhead未恢复 |
| D4 可并行任务停滞 | CLEAR |两片在共同授权上限耗尽时暂停；新批准后A/B/C已并行 |
| D5 超范围镀金 | CLEAR |增量用于既有身份/绑定验收；数值授权越界归D7，不重复归为新功能 |
| D6 分析空转 | CLEAR |调查产生实际覆盖RED、配置传播GREEN、完整输入及明确范围决定 |
| D7 路线违背 | HIT |root实计shared10/413+native5/109=15/522，已超过当时14/500；后获23/900批准不能抹去此前越界 |

**文档检查：**
- S1 CLEAR：prefill不等待native checker冻结；native收集不等待完整预测器或成本拟合。
- S2 HIT（已改）：原plan第2项未给两个typed-identity文件和prefill下游文件指定owner；现有独占路径表。
- S3 HIT（已改）：原计划的完整实现估算与14/500不相容；YC已明确修订23/900。
- S4 CLEAR：计数、身份拒绝、默认字节一致、实际来源和各自运行证据均可观察。
- S5 HIT（修正中）：旧binding设计仍写22/843，旧主记录只写进行中；主记录已补15/522、新批准和镜像结果，分支owner负责同步其旧估算。
- S6 HIT（已改）：原并行切分无累计额度预留；现按12/450、6/250、5/200分配，native owner独占共同integration test。

**归因：** D2为执行偏离；D7由root未落实总量检查造成，S2/S6增加了风险，但原批准要求总量检查，不能只归咎文档。原只读子代理把D7判CLEAR；主代理按实际越界证据修正。

## 3. 纠正建议

**分支：** B：文档存在范围/ownership缺陷，先更新文档再继续。

**文档更新（先执行）：**
1. requirements.md及main requirements：23/900额度修订（需用户确认，已收到原文批准并落盘）。
2. plan.md：完整文件/累计额度/共同测试唯一writer；保持package独立路径（可直接执行，已落实）。
3. progress、primary report及各分支设计：更新真实计数、镜像CPU结果、775字段输入及各阶段限制（可直接执行，主记录已更新）。

**需停止的偏离行为：**
- D2：停止无事前假设的fixture重试；每次针对具体失败保留原始错误和可证伪预期。
- D7：停止按分支各自预算代替总量；每次交付按统一基线核算，触及局部额度先由root在900内重新分配。

**接下去的路线：**
1. 并行R2a复审/最终strict77 ‖ R2b实际来源/输入/builder ‖ R2c注册/实际inventory/CSV/consumer。完成标志：真实调用链GREEN，默认不变，缺失/错配typed拒绝且计数在总量内。
2. R3→R5：联合复审、当前源码检查和各自新包验证；native checker所有writer停写后由root冻结。完成标志：全部CPU可计算项通过，输入/源码/外部依赖固定。
3. R6：分别执行已批1H800/900s、32H800/60min，核验GPU/business/rank/terminal。固定策略CPU检查不替代此步骤。

## 4. 最值得关注的 3 点

1. 原15/522是已发生的授权越界；现已明确批准23/900并独占分配，不再按各分支分别认定合规。
2. 共享调用链已到186dfef/主代理218PASS，完整Frontier输入的两个真实角色都解析为balanced_topk；最终fidelity/复审尚未完成。
3. native/prefill实际绑定和新包仍未完成；CPU镜像PASS不证明GPU后端或完整reference，CPU-overhead继续后置。

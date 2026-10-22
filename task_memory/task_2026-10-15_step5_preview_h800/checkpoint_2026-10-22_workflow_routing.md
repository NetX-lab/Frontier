## 1. 规划完成情况

Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | 关键源码步骤完成、第三项固定工作修正仍超时后，检查主线及D29顺序。 |

**汇总：** PASS 6｜BLOCKED/FAILED/SKIPPED 19｜TODO 6

PASS仅表示对应范围记录完成；完整票据状态没有被局部源码测试提升。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 01 基线复用 | PASS | 票据done |
| 02 原生锁定 | PASS | 票据done；新routing变体仍须独立决定 |
| 03 模型清单 | PASS | 票据done |
| 04 kernel/state范围 | PASS | 票据done |
| 05 本地物理测量 | TODO | 采集/完整验收未闭合 |
| 06 单角色simulation_scope | PASS | 批准范围done |
| 07 observer验证 | TODO | native关联/扰动仍待完成 |
| 08 PD state/G3 | FAILED | capture01不合格；capture03离线就绪但routing阻塞 |
| 09 四轨运行/校准资格 | BLOCKED | 各轨G4/ready/正式资格未闭合 |
| 10 全路径profiles | TODO | D22小探针完成，完整输入仍待采集/准入 |
| 11 routing统计 | BLOCKED | observer/原生条件未齐；D29策略核对在进行 |
| 12 TP/EP通信测量 | TODO | 队列/稳定性/准入未闭合 |
| 13 NIXL竞争测量 | BLOCKED | 前置原生及输入条件未齐 |
| 14 指标归一化 | BLOCKED | 代码完成，native映射资格未齐 |
| 15 混合模型路径 | TODO | representation/structural已完成，native reconciliation未完成 |
| 16 混合精度预测 | BLOCKED | 合格实测输入未齐 |
| 17 完整PD handoff | FAILED | actual06超时；D26/D27源码交付待整合 |
| 18 压力/取消 | BLOCKED | 原生拒绝/ownership证据未齐 |
| 19 graph/host | BLOCKED | workflow/GPU未闭合；D29后置CPU gap |
| 20 placement | BLOCKED | 结构部分完成，native-backed验收未齐 |
| 21 routing/communication预测 | BLOCKED | 独立合格输入未齐 |
| 22 完整bundle/replay | FAILED | actual06未完成1024；identity修正后短迭代仍RED |
| 23 case冻结 | TODO | workload已冻结，当前routing/source尚未绑定 |
| 24 旧代表场景 | SKIPPED | D12退役 |
| 25 旧重验/fit冻结 | SKIPPED | D12退役 |
| 26 旧H800矩阵 | SKIPPED | D12退役 |
| 27 旧H200矩阵 | SKIPPED | D12退役 |
| 28 四轨最终比较 | BLOCKED | 各轨ready及32→64条件未满足 |
| 29 三模型回归 | BLOCKED | 最终同HEAD合格证据未齐 |
| 30 支持声明 | BLOCKED | 完整任务尚未验收 |
| D25 完整32/64 workload | PASS | 两组432/432请求，指标已归档；无稳定态资格声明 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次D24恢复并行检查至今；循环项追溯actual05/06及三项固定工作修正。**当前前沿：** 11策略语义、08新包、17 P/传输、22最小执行能力。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | D27真实payload回归及D28既有分配器故障接缝直接验证当前修正；未扩展CPU gap项目 |
| D2 细节反复重试 | CLEAR | 每项修正前均有原因假设；identity42248次解析降为0，父端成本/ledger全等，仍明确RED；没有再盲跑2400s |
| D3 支线问题修补 | CLEAR | P/SWA/压力均获D26–D28批准；engine-cap和native尾部解释由用户明确请求 |
| D4 可并行任务停滞 | CLEAR | D24 capture02/03封包与D26、D27、identity源码并行推进，旧D4已纠正 |
| D5 超范围镀金 | CLEAR | 仅已批准P语义、live页和D-only参数；native balanced停在方案调查 |
| D6 分析空转 | CLEAR | 调查形成用户决定及源码交付；当前新增routing调查在解除真实提交阻塞 |
| D7 路线违背 | CLEAR | D29后未启动CPU gap采样/拟合；未以随机路由冒充balanced或提交capture03 |

**文档检查：**
- S1：plan.md:127–131未把routing绑定列入新native执行边；应显式加入。
- S2：11目前只有后置统计范围，没有当前11.routing_policy核对/方案的归属。
- S5：11:15/31/38仍要求原router不变并拒绝forced-balanced，与D29冲突；plan.md:118仍指向fixture02；08/17/22的最新状态停留在早期实施。
- S3/S4/S6：未发现独立的不可达、无界或不可验证切分问题。
**归因：** 无执行HIT；上述文档缺陷需同步。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续。

**文档更新（先执行）：**
1. plan.md当前路线：加入11.routing_policy→新generation→08；改为D26新fixture并保留fixture02只读（可直接执行）。
2. 11及spec中的原router约束：记录D29临时balanced覆盖、保留旧数据身份；策略核对可立即进行，统计准入仍保留其条件（可直接执行）。
3. 08/17/22/progress：记录capture03、D26/D27、父端identity仍RED；19记录CPU gap后置（可直接执行）。
4. 如果锁定native必须修改routing实现：先提交实际FP8/BF16/DP与reference语义方案（需用户确认）；不能把此调查当作修改授权。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 11策略核对：明确实际native路径、DP整数分配及P/D与独立reference一致性；完成标志为可执行且获批准的同义routing方案。
2. 并行17/22：审查整合D26/D27及已验证identity修正，生成新输入、完成必要回归；保持2400s及1024输出/READ/readiness/release验收条件。
3. 并行08：完成D28父端复验；routing完成后新包/live检查及H80032诊断，记录真实pressure/state/HTTP结果。
4. workflow完成后推进GPU/operator及op-gap；CPU overhead/gap最后开展。

**暂缓事项：** 新的大型CPU gap分析、采样和套件，按D29记录在requirements.md；当前不增加第四项Python优化或盲目完整CLI重试。

## 4. 最值得关注的 3 点

1. D24独立封包已恢复，D-only32000候选准备完成；现在真实阻塞是native balanced机制。
2. P6128/D16与SWA32页的源码交付已推进；它们仍需组合输入、回归和完整请求验收。
3. 第三项固定工作修正后，父端迭代7.983/8.116s仍超过2.012s预算；不会把它报成超时已解决或native CPU gap证据。

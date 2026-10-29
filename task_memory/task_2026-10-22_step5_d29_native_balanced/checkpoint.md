**Modification History**

| Date | Change |
| --- | --- |
| 2026-10-22 | 纳入 Frontier review：纠正初稿过宽范围及必需接口遗漏；三路设计并行。 |
| 2026-10-22 | 检查 D29/D30 当前设计与修正阶段；不重判历史票据或实现质量。 |

## 1. 规划完成情况

**汇总：** PASS 8｜BLOCKED/FAILED/SKIPPED 7｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 原生 FP8/BF16 路径调查 | PASS | native plan:31；proposal.md 已交付 |
| 既有策略与位置来源调查 | PASS | native plan:32；position policy 已获 YC 选择 |
| 原生源码接口与数学控制 | PASS | native plan:33；历史 CPU/source 证据已归档 |
| 首版提案与调查归档 | PASS | native plan:34–35；efcd7a0、651415a |
| native 设计及四项评审纠正 | PASS | c9e9274；同一 reviewer 四项 Resolved |
| native 文件预算父端核算 | PASS | 19 文件／1003–1582 行；本目录 test report |
| D30 完整修正提案 | PASS | 0a0b9fe；7 文件／≤750 行 |
| Frontier 初稿独立设计审查 | PASS | position_routing/review.md 记录六项 findings；完整范围未获接受 |
| Frontier count/schema/coverage 修正 | TODO | 修正 registry、lane cache、padded schema、empty exact coverage，fixture 必需 |
| source-local／physical collective 计费设计 | TODO | 原生 DP physical 传输与当前 EP16 assignment 计费不一致 |
| profiler 派生来源入口设计 | TODO | staging/submit 仍锁旧 image，须接同一 derived identity |
| D30 候选完整回归 | FAILED | cf1a53f 尚有 peer-driving 与 verdict 接入问题 |
| 合并数值范围确认 | BLOCKED | 等 Frontier 必需范围及用户批准 |
| D29 实现和源代码验证 | BLOCKED | 等合并范围批准 |
| D30 修正、集成与 strict77 | BLOCKED | 等七文件扩展批准及修正验证 |
| 派生 image 与新 package | BLOCKED | 等已审代码与完整 source identity |
| 新原生诊断及 workflow/GPU 判断 | BLOCKED | 等 image/package 与就绪检查 |
| 当前 native CPU overhead/gap | SKIPPED | YC 明确后置；不属于本阶段交付 |

## 2. 偏离检测

**结论：** 已偏离：命中 D5（Frontier 初稿设计范围过宽；未实施，已下达收窄纠正）。
**检测窗口：** 首版 native 设计 651415a 至 Frontier review 回执｜**当前前沿：** 三路并行补全 Frontier 必需范围，再一次合并确认；D30 待同批扩展批准。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 设计修改对应四项具体接入缺口；父端仅作预算/path/scope 检查 |
| D2 细节反复重试 | CLEAR | native 一轮四项纠正已闭合；D30 原因核验后停止未批准编辑 |
| D3 支线问题修补 | CLEAR | 新 findings 均阻塞新 policy 或 D30 记录消费 |
| D4 可并行任务停滞 | CLEAR | native、Frontier 和 D30 三线均有交付；后续代码受真实范围批准约束 |
| D5 超范围镀金 | HIT | Frontier review 证明新 policy 的 PD-AF/MTP/legacy fallback 扩展不是当前 Step P/D 验收必需；初稿却计入设计 |
| D6 分析空转 | CLEAR | 本阶段交付就是范围设计；native 设计和 D30 范围已完成 |
| D7 路线违背 | CLEAR | 未提前改新策略代码或启动 GPU/CPU-gap 工作；旧 generation 保留身份 |

**文档检查：** S5：父端 finish_sync/progress.md:10–11 状态过期，plan.md 缺少已确认的修正依赖（已纠正）。S5：Frontier design.md:587–601,670–679 将当前 requirements 要求的 fixture/完整计费列为可选；须修正。必要来源/计费任务已在 position_routing/plan.md 及 review.md 明确归属。S1/S2/S3/S4/S6 未发现其他明确缺陷。
**归因：** D5 执行偏离：按当前 Step P/D requirements 与 plan 的最小范围要求，本可不扩展这些新 policy 路径。纠正仍在设计阶段，没有未批准的源码实施。

## 3. 纠正建议

**分支：** B：先更新阶段文档，再继续完整范围确认。

**文档更新（先执行）：**
1. `../task_2026-10-22_step5_finish_sync/{plan,progress}.md`：记录候选未合并、六实例 fixture 缺口、verdict 消费修正、0a0b9fe 七文件提案及批准依赖（可直接执行；父端已应用）。
2. `../task_2026-10-22_step5_position_routing/{design,pricing_design,profile_source_design}.md`：删除可选验收替代、收窄新 policy 模式并补全必要接口；plan/review 已记录三个独立 owner（可直接执行；修订中）。
3. native/Frontier/D30 的合并代码范围：完成路径并集、必要性及预算后一次确认（需用户确认；尚未完成）。

**需停止的偏离行为：** D5：停止为当前验收不使用的 PD-AF/MTP/legacy fallback 扩展新 policy；保留旧策略，把工作用于必需来源、计费和 coverage 接口。

**接下去的路线：**
1. Frontier 设计审查与必需计费范围：收口 source-local router、physical DP collective、profile/fixture 接入 → 完成标志：每个必要改动有源码依据、文件及上限，无未估算 companion。
2. 合并确认：与已审 native19 文件及 D30 七文件去重 → 完成标志：用户批准完整数值范围。
3. 并行实现 native/Frontier/D30，约定共享文件单一编辑者 → 完成标志：各自受影响验证与源代码审查通过，再集成。
4. image/package 与批准的原生诊断 → 完成标志：新 source/policy identity、实际 graph/backend/READ/ownership/counter 等证据；不把静态检查当成 native PASS。

## 4. 最值得关注的 3 点

1. Native 四项设计 findings 已闭合，19 文件预算父端复算通过；实现和 GPU 结果仍未验证。
2. D30 五个函数／六个实例及 verdict 接入已有完整七文件提案，候选尚不可合并。
3. Frontier 初稿同时有过宽适配和必需接口遗漏；已要求收窄，并以三路设计补全后一次批准。

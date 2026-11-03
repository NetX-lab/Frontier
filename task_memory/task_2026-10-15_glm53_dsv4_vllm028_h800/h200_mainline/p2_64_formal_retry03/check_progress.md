Modification History

| Date | Change |
| --- | --- |
| 2026-11-03 | 第三代终态及第四候选包验收后的自查；记录本轮执行偏离并应用纠正。 |

## 1. 规划完成情况

**汇总：** PASS 1｜BLOCKED/FAILED/SKIPPED 7｜TODO 12

完整 ticket 状态沿用前次记录；本窗口推进的是18的第三代终态验收及19的参考依赖。第三代运行失败，没有正式指标。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 00 基线审计 | PASS | ticket00 recorded done。 |
| 01 运行环境与 native qualification | TODO | 已有 H200/source 证据，完整跨配置条件未关闭。 |
| 02 observer/reader | TODO | R5 在5bd68e9e验收；旧完整 observer 条件仍未全闭。 |
| 03 模型/cache/routing inventory | TODO | 已有结构与 native 局部证据，ticket未整体关闭。 |
| 04 GLM family | BLOCKED | D-DSV4-ONLY-GLM-PAUSED。 |
| 05 DSV4 family | TODO | H200 operator path 已交付；完整最终范围未关闭。 |
| 06 communication | TODO | P2所需实测输入已合格，完整硬件范围未关闭。 |
| 07 NIXL PD | TODO | 现有线长计价修复已验证，完整ticket未关闭。 |
| 08 workflow | TODO | P2现有路径已验证；保留其余未闭条件。 |
| 09 predictor bundle | TODO | P2所需输入已合格，完整交付未关闭。 |
| 10 H800 P0/P1 | BLOCKED | 当前路线要求独立资源方案；本轮未获新增H800资源。 |
| 11 H200 P2 | TODO | P2-32正式PASS；新P2-64参考不完整。 |
| 12 D-PD | BLOCKED | ticket仍记录未闭native生命周期/争用证据；非本轮前沿。 |
| 13 D-ASYNC | BLOCKED | ticket仍记录native证据依赖；非本轮前沿。 |
| 14 D-ROUTING | BLOCKED | ticket完整决策未关闭；本轮不扩大routing工作。 |
| 15 D-DP-PD | BLOCKED | 完整native决策未关闭；保留其证据边界。 |
| 16 GLM case | BLOCKED | GLM暂停。 |
| 17 DSV4 case | TODO | P2绑定已用；完整硬件范围未关闭。 |
| 18 native reference | TODO | 第三代因lead误停结束且0native输出；终态取证完成，第四候选包已验收但未获提交批准。 |
| 19 comparison/acceptance | TODO | P2-32PASS及晋级receipt已签；P2-64仍无新参考；旧用量缺时钟，第四提交需新决定。 |

## 2. 偏离检测

**结论：** 已偏离：命中 D2、D7。
**检测窗口：** 63573ca9 的第二代终态收尾至本次第三代终态和第四候选包验收｜**当前前沿：** 18第四提交的明确资源决定；19后续比较等待完整参考。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 既有R5/tests/replay未重跑；本次CPU检查仅验证新package和已发生的launcher生命周期事故。 |
| D2 细节反复重试 | HIT | p64-formal3-1103b五轮循环重复无权限的corePodAPI，并用head截断输出；未验证新假设，妨碍精确状态核验。 |
| D3 支线问题修补 | CLEAR | 未开启旧时钟深挖、GLM/EP-count或其他测试债。 |
| D4 可并行任务停滞 | CLEAR | terminal只读取证与第四候选的纯CPU构建并行；数值比较确需native输出。 |
| D5 超范围镀金 | CLEAR | 第四候选只有现有CASES身份条目，未改模块/配置机制/native/workload；仍未提交。 |
| D6 分析空转 | CLEAR |29项只读取证、独立terminal/cause验收与47文件candidate验收已落地，停止进一步原因探索。 |
| D7 路线违背 | HIT | lead调用ds4--stop导致launcher907414中断并发出平台stop，违背保持launcher存活的已知要求；明确归责lead。 |

**文档检查：** S1/S2/S3/S4/S6未发现本阶段新缺陷。S5：plan/progress/summary/tickets18/19仍有当前第三代待提交/运行表述，与01:26:58Z终态不符，已更新。第四资源决定明确存在，不以debug批准代替。
**归因：** D2执行偏离；D7执行偏离。按文档要求保持launcher存活即可避免本次终止，不能归因于平台抢占或文档授权。

## 3. 纠正建议

**分支：** B：spec/tickets存在状态缺陷，先更新文档再继续。

**文档更新（先执行）：**
1. plan/progress/summary/tickets18/19及两份generation报告：记录第三代lead误停、八replicas未调度、无native输出、第四候选已验收但未提交（可直接执行，已应用）。
2. requirements新增execution note及第四proposal：保留三次cap，第四提交须新决定；候选不授予资源权限（可直接执行，已应用）。

**需停止的偏离行为：**
- D2：停止重复被禁止的Pod查询；使用get_rjob_infos及精确replica读取，保留完整结果和失败回执。实际29项查询已通过。
- D7：禁止对拥有活跃StepMind launcher的Pi label调用ds4--stop。提交回合正常结束，独立label监控；辅助失败只返回证据。CPU验证了正常父进程结束与带marker定向signal的区别。

**接下去的路线：**
1. 提交已验收终态证据、候选包与记录；就reviewed manifest提出恰好一次第四GPU提交的资源决定。完成标志：明确用户答复，不能用等待时间或普通debug批准替代。
2. 仅在第四资源批准后提交一次并保活；否则候选保持未提交。完成标志：完整三repeat及terminal/copy证据，失败不自动重试。
3. 完整参考后的coverage→R5→all-request/warm，复用已完成Frontier replay。完成标志：每repeat四指标及限制明确，GPU计量如实保留不完整项。

**暂缓事项：** GLM、H800新资源、EP-count与旧collector-hash债继续留在既有future.md/报告；不扩大旧时钟查询。

## 4. 最值得关注的 3 点

1. 第三代中断是lead误操作；平台audit明确匹配local launcher stop，不能归为抢占。
2. 八replicas均未调度，无native参考；未启动证据支持0containerGPUh推断，genericexactnull和旧generation缺时钟仍保留。
3. 第四候选47seal/check-only已通过且未提交；新增64H200/3600s/64GPUh必须由YC作出新的资源决定。

## 1. 规划完成情况

**Modification History**

| Date | Change |
| --- | --- |
| 2026-11-01 | Collective逐点/逐pass重建完成，所有结果保留；等待唯一attention probe与三个CPU replay。 |
| 2026-11-01 | 单步重建probe的128案例、546文件和真实payload检查完成，1H200已提交；三个CPU replay继续。 |
| 2026-11-01 | CSA raw与独立G2h比较完成：不按planner拆模型，转为同geometry重新构建后的单步重复性验证。 |
| 2026-11-01 | A6 exact-feature归因里程碑自查：四scope都不能靠改进插值过门限；下一步检查既有测量和native输入。 |
| 2026-11-01 | 自查collectives03完成、F1b held-out失败和两次coverage终态；A6/spec/ticket当前状态已同步。 |
| 2026-11-01 | 自查A6恢复、singleton提交、03启动和F2双规模；F3保留真实拟合失败。 |
| 2026-11-01 | fresh links/KV02 已提交；A6 报告引用已定位并修复，82 项受影响检查通过。 |
| 2026-10-31 | 原始 links/KV 失败已定位并修复，166 检查通过；F2 完整测量的单容量分析为 AMBIGUOUS，继续 fresh links/KV02 与原因解释。 |
| 2026-10-31 | B30b 整合与 merged 检查完成后自查；先前 S5 入口已修正，当前继续真实 probes。 |

**汇总：** PASS 1｜BLOCKED/FAILED/SKIPPED 2｜TODO 17

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 00 baseline audit | PASS | ticket 00 验收已记录完成。 |
| 01 runtime qualification | TODO | P0/P2 R4 references 可复用；完整 qualification 未关闭。 |
| 02 observer and gate | TODO | client gate 已实现；完整 observer qualification 未关闭。 |
| 03 inventory | TODO | source/cache 合同存在；native 完整验收未关闭。 |
| 04 GLM family | SKIPPED | 用户暂停 GLM。 |
| 05 DSV4 family | TODO | operator path已实现；A6四scope失败且exact errors已足够令p90失败。 |
| 06 communication | TODO | F1b held-out与coverage失败；A6曲线及24个pass诊断重建完成，资格仍失败。 |
| 07 NIXL PD | TODO | lifecycle/ingress 已实现；F3实测完整但fit失败；native packed payload资格仍未关闭。 |
| 08 workflow | TODO | run-ahead 与 B29 已验证；完整验收未关闭。 |
| 09 predictor bundle | TODO | non-dummy 路径运行；A6已完成但精度失败，硬件coverage仍未关闭。 |
| 10 H800 cases | TODO | 保留 P0 证据；H200 主线后继续。 |
| 11 H200 cases | TODO | B29 P2-32 数值 PASS；P2-64 repeat1 TTFT FAIL。 |
| 12 D-PD | TODO | lifecycle 证据存在；完整 contention disposition 未关闭。 |
| 13 D-ASYNC | TODO | decision43/B26 已实现；F2单容量解释AMBIGUOUS；独立localK18replays已启动。 |
| 14 D-ROUTING | TODO | balanced routing 与 MATCH 已记录；完整 disposition 未关闭。 |
| 15 D-DP-PD | TODO | P2 placement/reference 存在；完整 disposition 未关闭。 |
| 16 GLM case | SKIPPED | 用户暂停 GLM。 |
| 17 DSV4 case | TODO | P0/P2 bindings 可复用；H80064 后续工作未完成。 |
| 18 native references | TODO | P2 原 R4 PASS；全硬件 qualification 未完成。 |
| 19 comparison | TODO | F1b/F3失败已归档；F1a64和两次F2仍运行，完整资格未关闭。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 前次probe提交检查点8b530816至collective逐点归因完成｜**当前前沿：** 11/19三个既有CPU replay；05/09 A6测量资格

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 仅新增下一步需要的逐点归因，复用production loader/LOO；8条曲线与24条pass汇总后停止扩展。 |
| D2 细节反复重试 | CLEAR | 未重复提交或训练；检验kernel/mode变化与pass波动假设，结果排除单一重新汇总方案。 |
| D3 支线问题修补 | CLEAR | 逐点collective结果属于05/06/09 A6资格；没有与主线无关的修复。 |
| D4 可并行任务停滞 | CLEAR | F1a64与F2两规模保持运行；CPU分析独立推进。 |
| D5 超范围镀金 | CLEAR | 只增加必要分析工具；无生产代码、参数或formal输入变化。 |
| D6 分析空转 | CLEAR | 新结果确认6条失败曲线每个pass均失败，记录停止平滑/择pass的决定；唯一attention实测仍在运行。 |
| D7 路线违背 | CLEAR | D-NO-INTERRUPT/D33预算内提交1H200；约4.52+1.5=6.02/7GPUh，完整个人non-preemptible预留0→1≤32；F诊断不得替换formal输入。 |

**文档检查：** S1独立比较无错误依赖；S2测量精度已有05/09归属；S3保留原资格标准，当前失败不等于标准不可达；S4原median/p90及四指标门限明确；S5当前入口按新证据同步，未发现与决定冲突；S6原ticket切分仍可用。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，验收条件不变。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 05/09：检查唯一probe的creator、local NFS、worker硬件、128完整case和terminal → 完成标志：精确job与输入输出身份记录。
2. 并行：11/19 F1a P2-64 ‖ F2 P2-32 ‖ F2 P2-64 → 完成标志：原launcher终态、完整128请求、每个reference repeat四指标；不重复提交或gate。
3. 05/09：按冻结计划分析16次重建的全部timings、变异和前后半组median → 完成标志：保留每项测量与原值、误差、原因限制及下一步决定；不把诊断视为A6 PASS。
4. 19：汇总F1–F4时保留F1b held-outFAIL及COVERAGE_MISS和F3 invalid-fit；完整运行逐repeat列出四指标。

## 4. 最值得关注的 3 点

1. A6 collective的6条失败曲线在每个pass均失败，重新汇总无法全面修正；kernel/mode和离散payload也需保持对应。
2. 唯一1H200 probe已提交；128案例CPU检查通过，等待真实采样和终态。
3. 三个CPU replay还未结束；正式CC/K19、10.3/11.4ms与三次reference不变，P2-64 TTFT失败和其他qualification缺口仍在。

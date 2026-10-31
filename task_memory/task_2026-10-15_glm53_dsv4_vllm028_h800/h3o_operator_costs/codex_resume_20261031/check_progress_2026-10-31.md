## 1. 规划完成情况

**Modification History**

| Date | Change |
| --- | --- |
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
| 06 communication | TODO | 03已Succeeded，测量验证完成；F1b held-out失败，两次replay均COVERAGE_MISS。 |
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

**结论：** 未偏移，正常推进。F1b 的两项失败有明确结果，不把补表和重跑当成默认下一步。
**检测窗口：** 前次03提交/F2启动自查至2026-11-01本次提交前｜**当前前沿：** 11/19 F1a64和F2双规模；06/19 F1b结果；06/07 F3 invalid-fit结果；05/09资格失败

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 03数据重建校验处理跨Python浮点末位；F1b失败后仅复现两条实际coverage拒绝，未扩展harness。见F1b报告。 |
| D2 细节反复重试 | CLEAR | 成功NIXL02/F2/03没有重跑。F1b两次终态同属缺少joint coordinates；保留失败，不改阈值/参数试配。 |
| D3 支线问题修补 | CLEAR | 下载、fit、heldout和failure reproduction均服务06/11/19；A6状态同步服务05/09。 |
| D4 可并行任务停滞 | CLEAR | F1b两规模同时完成；F1a64和两F2一直独立运行。F3无可用fit不阻塞其他比较。 |
| D5 超范围镀金 | CLEAR | simulator/profiler及正式K19、CC、ingress、references未改；没有新增容量/通信framework。 |
| D6 分析空转 | CLEAR | 03终态、62文件hash验证、274 pooled fit points、128 heldout cases和两次CPU终态已提交2fea0ac0；A6原fold重建exact errors给出不调RF的明确路线，未循环重拟合。 |
| D7 路线违背 | CLEAR | decision46和D-CODEX-DIRECT-RJOB有效；本轮没有新GPU提交；所有F variant保留diagnostic标签。 |

**文档检查：** S1无新增错误依赖，F3不阻塞独立比较；S2两项coverage属于05/09/19，无新owner缺口；S3固定128请求/三次reference/四指标条件可判定，当前variant失败不是改标准的理由；S4门限明确；S5曾命中：spec当前输入合同及05/09/11/19旧入口仍把已完成且失败的A6写为待完成，且F1b运行状态刚过时；现已按A6/F1b报告同步；S6既有ticket切分无新矛盾。
**归因：** 无执行偏离；S5是证据更新后的状态滞后，已直接修正。

## 3. 纠正建议

**分支：** B：本窗口S5状态缺陷已修正，先提交已更新记录和实际证据。

**文档更新（先执行）：**
1. `spec.md`当前合同、`plan.md`当前路线、`progress.md`handoff及05/06/09/11/19当前入口：A6 completed-but-failed、03Succeeded、F1b heldout FAIL和两次COVERAGE_MISS、F3 invalid-fit均已同步（可直接执行，已完成；不改scope、decision或验收）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 06/11/19：提交03测量、fit/heldout、F1b coverage终态、F3 variation和本次自查 → 完成标志：准确路径、命令和失败边界均在同一主分支。
2. 并行：11/19 F1a P2-64 ‖ F2 P2-32 ‖ F2 P2-64 → 完成标志：原launcher终态、完整128请求和每个reference repeat的四项误差；只有完整运行进入gate。
3. 19：汇总F1–F4 → 完成标志：完整运行有absolute/predicted/relative值，F1b/COVERAGE_MISS和F3/invalid-fit单独列出，不能以部分请求补值。
4. 05/09：A6 exact errors已证明插值调参不能过门限，先检查CSA/FULL宽度8/16等相同特征下的既有实测/native输入，再处理7/8 collective曲线及硬件/payload资格缺口 → 完成标志沿用已记录的qualified-input条件；现在不关闭这些ticket。

## 4. 最值得关注的 3 点

1. 03测量完成且数据可重建；F1b目标EP graph-forward预测为实测11.449–19.300倍，正参数并不表示模型合格。
2. F1b完整replay没有可用四指标：P2-32缺embedding32，P2-64缺query16/padded24的attention覆盖；精确拒绝已复现。
3. F3负截距/29.18%p90、A6四scope失败和B29 P2-64 repeat1 TTFT12.254940665%仍在；不能以数值P2-32 PASS宣告formal完成。

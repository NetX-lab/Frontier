## 1. 规划完成情况

**Modification History**

| Date | Change |
| --- | --- |
| 2026-11-01 | Resumed existing F2 runs; checked six completed readings/96 metric rows against unchanged native CSVs; stopped further expansion of the comparison verifier. |
| 2026-11-01 | A6 repeatability accepted as diagnostic; F1a both scales complete with FAIL; only two F2 replays remain active. |
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
| 05 DSV4 family | TODO | 128-step repeatability diagnostic complete; A6 qualification still fails. |
| 06 communication | TODO | F1b held-out与coverage失败；A6曲线及24个pass诊断重建完成，资格仍失败。 |
| 07 NIXL PD | TODO | lifecycle/ingress 已实现；F3实测完整但fit失败；native packed payload资格仍未关闭。 |
| 08 workflow | TODO | run-ahead 与 B29 已验证；完整验收未关闭。 |
| 09 predictor bundle | TODO | Non-dummy path exists; repeated data do not qualify new inputs or a fixed repeat count. |
| 10 H800 cases | TODO | 保留 P0 证据；H200 主线后继续。 |
| 11 H200 cases | TODO | B29 P2-32 数值 PASS；P2-64 repeat1 TTFT FAIL。 |
| 12 D-PD | TODO | lifecycle 证据存在；完整 contention disposition 未关闭。 |
| 13 D-ASYNC | TODO | decision43/B26 已实现；F2单容量解释AMBIGUOUS；独立localK18replays已启动。 |
| 14 D-ROUTING | TODO | balanced routing 与 MATCH 已记录；完整 disposition 未关闭。 |
| 15 D-DP-PD | TODO | P2 placement/reference 存在；完整 disposition 未关闭。 |
| 16 GLM case | SKIPPED | 用户暂停 GLM。 |
| 17 DSV4 case | TODO | P0/P2 bindings 可复用；H80064 后续工作未完成。 |
| 18 native references | TODO | P2 原 R4 PASS；全硬件 qualification 未完成。 |
| 19 comparison | TODO | F1a complete and FAIL at both scales; F1b/F3 failures retained; two F2 replays still running. |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** Commit0a301fcc through six-reading comparison precheck; F2 remains running. **当前前沿：** 11/19 two F2K18 replays;05/09 measurement qualification after comparison completion.

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | Comparison verification is limited to the eight requested readings. Six existing readings/96 metrics pass; three metadata assumptions (E2E key and two pre-B29 fields) were corrected from the retained schemas. Stop adding checks after final F2 verification. |
| D2 细节反复重试 | CLEAR | The pre-recorded fixed-geometry hypothesis returned new measurements. No repeated GPU submission, simulator restart or second gate. |
| D3 支线问题修补 | CLEAR | A6 evidence belongs to05/09; F1a belongs to11/19. No unrelated change. |
| D4 可并行任务停滞 | CLEAR | All three original CPU runs continued during A6 analysis. F1a finished; both independent F2 jobs still run. |
| D5 超范围镀金 | CLEAR | No production feature, data schema, estimator or formal input changed. |
| D6 分析空转 | CLEAR | No A6 analyzer or repeated probe was added. The six completed readings are consolidated; do not re-run gates or expand analysis while waiting for F2. |
| D7 路线违背 | CLEAR | D-NO-INTERRUPT and decision46 preserved. Observed probe launcher window0.319825GPUh keeps approximate ledger4.839825/7; no larger campaign submitted. |

**文档检查：** S1: remaining F2 runs are independent. S2: measurement repeatability and precision have05/09 owners. S3: failed qualification is not proof that the criterion is unreachable; a larger measurement campaign needs a concrete recipe/resource decision. S4: formal median/p90 and four-metric thresholds remain explicit. S5: spec, plan, progress and tickets05/09/11/19 updated to actual terminal results. S6: existing ownership separates measurements, runtime and final acceptance; no new slice needed.
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，验收条件不变。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 11/19: keep F2 P2-32 and P2-64 running; consume watcher23683 outputs once → terminal receipts,128 complete requests and each reference's four metrics.
2. 19: finish F1–F4 comparison → retain F1a latency failures, F1b held-out/COVERAGE_MISS and F3 invalid fit, alongside F2 and completed F4.
3. 05/09: choose a uniform measurement correction from completed evidence → preserve the full1,421-case population and original data, state why the recipe addresses the observed variation, then assess resources. The index-compression control's20.50%half-median difference does not establish that16 repeats suffice; do not launch a full repetition sweep on cost estimates alone.

## 4. 最值得关注的 3 点

1. A6 raw identity/geometry/count checks pass, but large original CSA contrasts do not persist. Keep features and formal tables unchanged; all qualification failures remain.
2. F1a P2-64 now fails TTFT/TPOT/E2E in all three references; six compute hashes and three reference CSVs match B29. Both F2 replays still need terminal evidence.
3. Remaining approximate G1h–G4h budget is2.160175GPUh; two full reconstructions already cost2.210720GPUh before reserve. No larger campaign or diagnostic promotion is authorized by these results.

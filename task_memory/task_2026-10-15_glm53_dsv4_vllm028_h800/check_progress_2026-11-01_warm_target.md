## 1. 规划完成情况

### Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Final evidence update: warm review accepted/No findings and archive0cd8d5d3 complete; primary exact skill/prerequisites import/final status commit and human two-fix/two-replay decision remain TODO. |
| 2026-11-01 | Follow-up checkpoint: accepted skill/root main0cc4118a and completed accounting7/7; next frontier is correction decision plus final review, not residual investigations. |
| 2026-11-01 | WARM-DOCS scoped continuation checkpoint: synchronized completed warm readings/residual evidence; independent skill consistency and primary review/integration remain TODO. |

范围：`requirements.md` D-WARM-TARGET、`plan.md` Approved warm-target continuation 和本次 WARM-DOCS 委派。PASS 仅表示交付已记录完成，不代表质量、正式资格或批准；不是整体 tickets05/09/11/19 完成检查。

**汇总：** PASS 7｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 统一 warm 事件规则与请求身份绑定 | PASS | `warm_target_20261101/openers.json`及 reading report；五个 canonical opener 由八TP日志、DP ownership、proxy identity 绑定，非按延迟挑选。 |
| 六组逐 repeat 配对 warm reading | PASS | `warm_target_20261101/readings/`、`output_verification.json`；96/96 warm metrics PASS，空排除集合精确复现原结果。 |
| 当前 residual 归因与限制 | PASS | `warm_residual_20261101/test_report_2026-11-01_warm_ttft_residual.md`、`validation.json`；9/9检查通过，观测/推断/未分配项分别保留。 |
| Calibration skill 一致性修正 | PASS | SKILL-WARM-REVIEW六 findings CLOSED/no actionable findings/四 walkthroughs PASS；请求root skill完成于main0cc4118a，跨worktree integration尚未完成。 |
| 保留证据READ/payload及P-step accounting | PASS | `warm_accounting_20261101/test_report_2026-11-01_warm_accounting.md`、`validation.json`7/7；主代理接受两个结构缺陷，未批准code corrections。 |
| 两处existing-path修正与两次CPU replay的具体决定 | TODO | 已向human请求approve/reject/defer，尚未批准；subagent消息不授权fix/replay。不重开residual调查。 |
| 父任务当前状态、报告链接和检查点同步 | PASS | `plan.md`、`progress.md`、`summary.md`及 tickets05/09/19；原all-request FAIL保留，完整资格OPEN。 |
| 主代理最终 warm review/evidence archive | PASS | WARM-FINAL-REVIEW ses_e68dddb1effeH8TbRzOxiHxLPx No findings，独立96/96 PASS最大5.952669%；primary接受diagnostic/warm，archive0cd8d5d3完成。 |
| 主代理exact skill/prerequisites import与final status commit | TODO | primary在本次owned-file edits结束后执行；这是唯一剩余integration，不重新做已完成review/archive。 |

证据路径均相对于本task的`h3o_operator_costs/`。完整tickets和48 attention omissions、hardware/NIXL/payload/memory/promotion要求仍OPEN，不纳入上述完成计数。

前次文档核对：Python3.10.6只读71/71 PASS；warm96/96 PASS、原3/3 FAIL保留、12/12显示数值符合1e-6舍入。初次repeat ID检查器误读已修正，未改证据。Skill/accounting follow-up曾60/60文档检查PASS、五段历史保留、19 throughput规则原文不变。本次最终状态只采用primary明确接受/归档更新，不扩大核验。最终review独立重算96指标并检查事件/window/repeat/proxy限制，但未另跑row/hash verifier；不将其说成新全量验证。精确旧命令及限制在`progress.md`的WARM-DOCS verification。

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 前次WARM-DOCS checkpoint至skill/accounting与最终warm review接受、archive0cd8d5d3及本次同步｜**当前前沿：** human两处fix/两次CPU replay决定 ‖ primary exact skill/prerequisites import/final status commit；停止residual调查。

主线：`completed {readings || residual || skill || accounting} -> accepted warm review/evidence archive -> {primary exact skill/prerequisites import + final status commit || human two-fix/two-CPU-replay decision}`；仅human批准后fix→CPU regressions+两次replay。Skill是明确请求，accounting对应已结束调查，父记录对应integration准备；未发现无归属工作。之后仍需合格输入、每轮四指标和formal P2/H800。本次不改source/evidence，不运行模拟或GPU，不扩大检查。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | reading report所需配对reader/空集合核对完成即停；WARM-DOCS只修当前指针，未扩建harness。 |
| D2 细节反复重试 | CLEAR | accounting report记录package import/config-key检查修正，未重跑residual/FIFO或原replays/gates；无无假设循环。 |
| D3 支线问题修补 | CLEAR | warm范围和skill为D-WARM-TARGET明确批准；没有GLM、source、历史trace audit或无关修补。 |
| D4 可并行任务停滞 | CLEAR | skill/accounting/final warm review/archive已完成；primary import/status等待owned-file编辑结束是真实同文件协调，human决定不等待新READ/host证据。 |
| D5 超范围镀金 | CLEAR | accounting只确认existing wire-price/count缺陷，未实现修正或增加READ/host机制、offset、GPU、formal table。 |
| D6 分析空转 | CLEAR | accounting已交付7/7验证和两个具体决定候选；停止residual调查，下一步决定，不扩写调查路线。 |
| D7 路线违背 | CLEAR | 按D-WARM-TARGET保留全部repeats与原FAIL，不按慢请求筛选、不建模一次性warmup、不推定promotion。 |

**文档检查：** 本continuation逐项检查S1–S6；S5当前状态缺陷已直接修正。

| 文档缺陷 | 判定 | 依据 |
| --- | --- | --- |
| S1 Wrong edge | CLEAR | reading/residual/skill/accounting/review/archive完成；primary integration‖human two-fix/two-replay决定独立，CPU corrections/replays依赖真实批准；资格与数值门槛分开。 |
| S2 Unowned blocker | CLEAR | omissions由05/09，NIXL/payload/memory及正式比较由plan和05/09/11/19持有；本次skill/review也有明确主代理归属。 |
| S3 Unreachable criterion | CLEAR | D-WARM-TARGET允许有源代码及本run事件证据的一次性carrier配对排除；ticket19据此改为warm人口规则，仍记录128完整completion且不排除repeat。 |
| S4 Unbounded criterion | CLEAR | 六reading、所有3repeats/4metrics、事件绑定、skill一致性、明确review记录均可判断完成；不要求无限warmup或全量新测。 |
| S5 Stale doc | HIT（已修正） | 先纠正skill TODO/下一调查；最终update又明确warm review No findings/primary接受及archive0cd8d5d3。已把review/archive改完成，只保留exact skill/prerequisites import/final status为integration TODO，human fix/replay批准未获。原纠正历史保留。 |
| S6 Bad slice | CLEAR | 05算子/attention资格、09 bundle/decomposition、19 comparison独立责任；WARM-DOCS仅持有指定父记录，未重做source/evidence/skill。 |

**归因：** 无D1–D7 HIT。S5是新规则/产物后待同步的当前指针，不是实际偏离；执行修正不改变用户决定。

## 3. 纠正建议

**分支：** B：当前文档存在S5，先修正后继续；本次已落实，不扩展批准范围。

**文档更新（先执行）：**
1. `plan.md`当前milestone/dependency：skill/accounting/review/archive完成，只留primary import/status integration及human two-fix/two-replay决定；批准才做fix/replay（可直接执行，已落实）。
2. `progress.md`handoff/Next actions、英文`summary.md`：记录final reviewer ID、96/96最大5.952669%、No findings/primary接受、archive0cd8d5d3及review未重跑row/hash verifier限制（可直接执行，已落实）。
3. tickets05/09/19及本checkpoint：仅integration待完成，不把diagnostic/warm接受当完整资格或fix/replay批准；19 retained-population throughput规则原文保留（可直接执行，已落实）。

**需停止的偏离行为：无。**

**接下去的路线：**
1. Primary exact skill/root main0cc4118a及prerequisites导入current branch，再做final status commit → 完成标志：import/status receipts记录。Review/archive已完成，不重启。
2. 并行：等待human两处existing-path fix（NIXL selected pages×existing physical stride仅wire pricing；COMM EP dispatch3/combine1）及两次CPU replay的approve/reject/defer → 完成标志：human决定明确。尚未批准；只有批准后CPU geometry/ownership、NONE/PIECEWISE/FULL gap-count regressions及两次replay。
3. 停止residual调查；不新增GPU/host证据、READ机制或offset。两个现有路径fix尚未获准。完整输入资格及每轮warm门槛之后才是formal P2/H800，本次不关闭完整tickets。

## 4. 最值得关注的 3 点

1. 六reading、96/96 warm metrics全部PASS，但完整输入资格仍OPEN；原all-request repeat01 FAIL12.254940665%/10.169032726%/10.277736953%与所有repeats完整保留。
2. P2-64 ~31ms只来自price-neighborhood proxy，不是同价长期平均；P2-32只可靠定位到P-side13.4–13.8ms/post-P5.8–6.9ms，不能用probe分配项或offset填平。
3. Skill/root main0cc4118a、accounting7/7和final warm review/archive0cd8d5d3已接受完成；仅primary import/final status integration待完成，human两fix/两replay批准待定。+0.417203 ms wire/+0.046836 ms direct eager P-step不是TTFT承诺；READ/host-zero证据仍不足。

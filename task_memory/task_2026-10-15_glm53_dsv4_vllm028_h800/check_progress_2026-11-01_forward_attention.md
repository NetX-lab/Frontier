# Forward-attention continuation checkpoint

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Both P2-32 runs/gates complete; remaining P2-64 runs active, collector queued. Corrected summary's stale 4 PASS / 2 TODO count to 5 PASS / 1 TODO. |
| 2026-11-01 | Continued original CPU runs; prepared the final retained-reading collector with focused/schema checks. Still 5 PASS / 1 TODO; no new workload or scope. |
| 2026-11-01 | A6-FINAL-DOCS milestone self-check: all four original CPU replays and final collector terminal; 6 PASS / 0 TODO. Corrected S5 current pointers, retained historical observations and formal qualification OPEN. |
| 2026-11-01 | Final A6 actual-evidence review completed with no actionable evidence defect; retained 6 PASS / 0 TODO, review limits and material near-threshold P2-64 TTFT sensitivity. |

## 1. 规划完成情况

范围：`requirements.md` 148–161 行的 D-A6-FORWARD-ATTENTION / D-A6-PASS-REPLAYS 和 `plan.md` 的 Approved A6 continuation。不是 tickets 05/09/11/19 的整体完成检查；PASS 只表示交付完成，不代表数值、质量或批准。

证据目录均相对于本 task 的 `h3o_operator_costs/`；终态/比较在 `a6_pass_replays_20261101/`，median3 验证在 `a6_forward_attention_20261101/`。

**汇总：** PASS 6｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 规划子任务 | 状态 | 证据 |
| --- | --- | --- |
| 432 个补测 workload 再测两轮并完成取数/业务核对 | PASS | `a6_three_run_20261101/{platform_final.json,download.json,repeatability_result.json,business_verification.json}`；864 步身份/形状/计数一致，容器耗时 0.9302778 GPUh。 |
| Forward 级判定实现及回归验证 | PASS | 原生 layer window reference、配置层数权重、固定 folds/门槛；主代理 31 tests PASS，B-FORWARD-REVIEW 无 actionable finding。 |
| 当前完整输入的 forward 判定和全 scope diagnostics | PASS | `a6_forward_attention_20261101/current_phase_heldout.json`，exit 0；1,373 complete / 48 omitted，三个完整类别 PASS，整体仍 NOT_EVALUABLE。 |
| 完整三轮表、两张独立轮次表和四个输入绑定 | PASS | `a6_pass_replays_20261101/{rows_manifest.json,tables_manifest.json,preflight.json}`；7,105 行均有三轮结果；kernel/gap/window 同轮替换，非 attention 输入不变。 |
| Median3 表的固定-fold A6/forward 复核 | PASS | 两阶段 exit 0，终态 08:37:48 UTC；`median3_verification.json` 验证所有 folds/reference/weights/omissions 不变，三个完整类别 p90 1.628628%/1.932193%/2.215912% PASS。不能算整体 PASS，不重复启动。 |
| 四次完整 P2 replay、逐 repeat 四项指标和跨表 spread | PASS | `completion.json` 10:34:20 UTC 记录四个原 run/gate 终态；collector exit 0/1.273275s，10:34:21 UTC 完成六 readings/96 metrics/八 spread/八 native variability 的 PASS_EVIDENCE_CHECKS。P2-64 repeat01 TTFT 数值 FAIL 仍保留。 |

六项批准交付已完成，无 replay/gate/collector 重跑。P2-32 三个 attention 表版本均 PASS；P2-64 三个版本均仅 repeat01 TTFT FAIL，不能将完成数当作数值 PASS。48 个遗漏保持整体 attention NOT_EVALUABLE；正式验收 OPEN。最终 A6-PASS-FINAL-REVIEW（`ses_e69531224ffeMg6U8NpSEOZTF9`）已完成：无 actionable evidence defect，诊断交付可接受，不是正式资格通过。96 metrics/八 spreads/八 variability 已由 reviewer 独立复算；检查/限制见 [continuation report](test_report_2026-11-01_forward_attention_continuation.md)。大 CSV 完整字节哈希依赖实际 collector，source equivalence 依赖 baseline 记录的 diff paths，不是新增完整 source-diff 复核；最终 report/status 文本由主代理检查，不是 reviewer 的最终文本审查。

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** P2-32 终态 checkpoint（10:04 UTC）至最终 collector 终态（10:34:21 UTC）、最终 evidence review 和本次文档同步｜**当前前沿：** 六项交付及最终 review 均完成；仅按既有 qualification 路线和进一步批准推进。

**主线：** 完成的 A6/独立轮次比较及最终 review → 新表 promotion 的进一步 YC 决定及 NIXL/memory/coverage 输入资格 → 每个 admitted repeat 四项指标门槛 → 正式 P2 → H800。检测窗口中的原 P2-64 运行、collector 终态、review 和当前文档同步均对应批准交付，没有无归属工作。

| 偏离模式 | 判定 | 证据与例外 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 最后一项仅需 retained-reading collector；`comparison_command.json` 记录一次实际取数成功，随后停止扩建。当前指针修正避免重复执行，不是新 harness。 |
| D2 细节反复重试 | CLEAR | `completion.json` 和四个原 command 保留一次运行/一次 gate；最终 collector 一次 exit 0。此前 receipt 问题已有具体原因及恢复记录，没有变参重跑。 |
| D3 支线问题修补 | CLEAR | 工作只涉及批准的四个 P2 终态、全量比较和对应文档；未新增 source/GLM/prefill/无关问题修补。 |
| D4 可并行任务停滞 | CLEAR | 四次原 replay 并行；median3 和 collector preparation 在运行期间完成。collector 依赖终态真实存在，最终 review 与文档同步独立。 |
| D5 超范围镀金 | CLEAR | 最终产物正是 requirements 161 行要求的三个 attention 表版本、所有 admitted repeats 和四指标比较；未加 simulator feature、GPU campaign 或 formal table。 |
| D6 分析空转 | CLEAR | 最后一项从 TODO 到 PASS，有 `comparison.json` 实际输出；只记录固定-case sensitivity，不据 spread 重开无依据原因分析。 |
| D7 路线违背 | CLEAR | requirements 160–161 行的 folds/limits/非 attention 输入/clean references 不变；保留全部数值 FAIL/48 omissions，没有选择最好轮次或提前 promotion。 |

**文档检查：** 六类均已检查；S5 当前状态缺陷已在本轮直接修正，其他项未发现本切片缺陷。

| 文档缺陷 | 判定 | 依据 |
| --- | --- | --- |
| S1 Wrong edge | CLEAR | `plan.md` Approved A6 dependency 中 collector 确实需要四个 terminal；median3 与独立轮次比较无虚假串行边。 |
| S2 Unowned blocker | CLEAR | 48 omissions、NIXL/memory/coverage 和每轮门槛归属 plan H3o/Qualified P2、tickets 05/09/11/19；不是本次交付中的无归属阻塞。 |
| S3 Unreachable criterion | CLEAR | requirements 160–161 的有限诊断交付已有对应输出；正式 qualification 未被无 GPU/promotion 批准的诊断冒充。 |
| S4 Unbounded criterion | CLEAR | folds K5/seed1234、median5%/p90 10%、四个 CPU replay、三个表版本/全部 repeats/四指标均有有限完成条件。 |
| S5 Stale doc | HIT（已修正） | 检查时 `plan.md` Approved A6、`progress.md` 当前证据/Next actions、`summary.md` continuation、tickets 05/09 和本 checkpoint 有 pending/5 PASS 1 TODO（progress 另有旧 4/2）；与 10:34:20 `completion.json` 和 10:34:21 `comparison_command.json` 矛盾，已同步 6 PASS / 0 TODO，保留历史快照。 |
| S6 Bad slice | CLEAR | 05 owns operators/qualification，09 owns bundle/coverage；本次六项是有明确接受条件的批准 continuation，最终文档六文件独立于主代理 report/JSON，未重复所有权。 |

**归因：** 无 D1–D7 HIT。S5 是完成后尚未同步的文档状态，不是实际重复执行或绕过接受条件。

## 3. 纠正建议

**分支：** B：本轮发现 S5，先更新文档再继续；当前指针修正已落实，不改变 spec/批准/接受条件。

**文档更新（先执行）：**
1. `plan.md` 当前 H3o/Approved A6：四原 replay、collector 和最终 review 改为 complete，依赖前沿改为进一步决定/资格，保持正式资格 open（可直接执行，已落实）。
2. `progress.md` 当前证据/Next actions、`summary.md` continuation：写入终态时间、六 readings/96 metrics、三个表版本的数值判定和 spread/CV 限制；旧有带时间的过程读数标为历史（可直接执行，已落实）。
3. tickets `05-dsv4-flash-family.md` / `09-predictor-bundle.md` 当前 continuation 及本 checkpoint：6 PASS / 0 TODO，保留每轮 FAIL/48 omissions，不关闭整体 tickets（可直接执行，已落实）。

**需停止的偏离行为：无。**

**接下去的路线：**

1. Continuation、最终 actual-evidence review 和对应记录均完成，无 actionable evidence defect；检查范围与限制见 continuation report，不重复任何完成的 run/gate/collector，不据此宣布正式 qualification。
2. 若考虑 formal table promotion，先取得进一步 YC 决定；本轮不执行 promotion/新 GPU。Done 条件沿用 requirements 的进一步决定，不把 6 PASS 当作授权。
3. 其余主线按 plan Qualified P2/H800 继续：05/09/11/19 的 measured-input/coverage 资格及每个 admitted repeat 四指标 <=10% 后才可正式验收；本轮不新增实验或推定 residual 原因。

## 4. 最值得关注的 3 点

1. 六项 continuation 交付完成，但 P2-64 三个 attention 表版本仍仅 repeat01 TTFT FAIL：B29/pass1/pass2 = 12.254940665%/10.169032726%/10.277736953%，pooled PASS 不覆盖单轮失败。
2. P2-64 TTFT spread 16.267761 ms / 2.377237%B29；新 pass1/pass2 仅比 701.899855 ms 下界低 1.318267/2.166039 ms，对门槛判定有实质影响，不能说无关。Native TTFT CV 0.496152%/4.544622%（n=3）仅描述，不证明 reference sufficiency 或 residual causal mechanism。
3. 最终 review 无 actionable evidence defect，仅接受诊断交付；48 omissions/整体 NOT_EVALUABLE、median3 原 gap/window reference 及 NIXL/memory/正式 P2/H800 资格仍保留，新表 promotion/新 GPU 无批准。

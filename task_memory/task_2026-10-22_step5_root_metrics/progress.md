# D23-ROOT-METRICS Progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record verified baseline, RED evidence, and inspected projection failures. |
| 2026-10-22 | Record reviewer P1 public-result RED/GREEN and stage-only owner correction. |

- completed: created the authorized worktree and branch from exact parent HEAD `61cbac9a3e8c29190b91a5d4e8f7fa2ad9fcd359`; no edits to parent or other worktrees. Current session: `ses_e9f9fd0bdffeZH9Q2Ux5NbCYko`.
- completed: ran the parent's exact inline CPU repro using the requested Python/environment. Exit 1; `/data/ycfeng/tmp/d23_root_metrics_red01.log` records `KeyError: 'step5_layer_e2e'` at `metrics_store.py:4504`, reached through the stage component ledger.
- completed: read root AGENTS, GLOSSARY, the only ADR, module-size exceptions, parent's `core_blocker.md`, execution/stage timing, family registry, attention projection, metric/trace adapters, and cost producer.
- completed: focused regressions went RED before source changes: `8 failed, 7 passed in 2.95s`, exact root KeyError and trace whitelist errors (`/data/ycfeng/tmp/d23_root_metrics_tests_red02.log`). An earlier test setup used an invalid shared TP domain and was corrected before source edits.
- completed: first fix passes all 15 focused tests in 6.11s (`/data/ycfeng/tmp/d23_root_metrics_tests_green01.log`). The mixed stage bills `.8244 ms` root, `.023 ms` measured kernel gap, dense `.3 ms`, and only authoritative stage fields; root proofs retain counts 23 and 7 in emitted events.
- completed: adjacent checks plus the expanded focused suite pass `161 passed, 1 deselected in 25.79s` (`/data/ycfeng/tmp/d23_root_metrics_final_green.log`). The one deselected monkeypatch/cache-isolation test and the excluded stale GLM53 fixture are baseline issues reproduced at parent HEAD; production compatibility shims are not added. Independent focused legacy inclusive tests verify totals and rejection at the same projection seam.
- completed: reran the original fast repro unchanged in timing inputs; root/ledger/stage totals are exactly `.6244 ms`, count `23.0` (`/data/ycfeng/tmp/d23_root_metrics_repro_green01.log`). Ranked hypotheses (1)/(2) were confirmed; exact totals and separate diagnostic child storage exclude double billing in tested scenarios.
- observed: root is already normalized/retained by ExecutionTime and charged once by AttentionTime; diagnostics are stored separately. The metrics enum lacks root, and Step tensor metadata rejects it. Stage trace paths retain per-layer timing records; aggregate attention timing intentionally drops individual proofs, so publication must read each authoritative layer, not fabricate an aggregate proof.
- observed: broad focused legacy selection gives `1 failed, 148 passed, 17 errors in 22.68s`. The GLM53 fixture references a deleted private cache function; the monkeypatched custom-family test lacks cache isolation. Parent baseline reproduces the missing function (`/data/ycfeng/tmp/d23_root_metrics_baseline_failures.log`). These are test-fixture issues, not reasons to change production timing.
- completed: source call-site audit: the predictor's `_assemble_stage` publishes StageExecutionTime; actual P/D reporting requests the full stage and preserves authoritative layers through the existing metrics adapters. Scalar aggregated metric/trace methods remain legacy adapters, not the measured root producer path. Stage attention aggregates intentionally have no individual proof; emitted Step root spans remain layer-scoped even with expansion disabled.
- completed: final verification after the public layer identity adapter adjustment: `161 passed, 1 deselected in 27.48s`, `/data/ycfeng/tmp/d23_root_metrics_final_verified.log`.
- completed: committed the seven scoped code/test/exception files as `3a189ba` (Preserve measured inclusive attention roots in metrics and traces). Net metrics store growth is eight lines. No unrelated files staged.
- completed (initial phase): English summary and exact test report written; final requirements reread. Records committed as `24b889d` after code commit `3a189ba`, checkpoint `9ef9cd0`. This phase's call-site acceptance was incomplete: the positive stage owner fixture supplied a root proof absent in real first/last boundary owners. P1 below supersedes the initial completion for that path.

## P1 execution

- completed: parent reviewer identified reachable first/last measured boundary owner failure after initial merge `f237362`. Continued from local `9ef9cd0`, clean worktree, no parent/platform operations.
- completed: public constructor/training + existing `predict_stage_execution_time` first/last -> Stage ledger RED before source changes: `1 failed in 11.51s`, `/data/ycfeng/tmp/d23_root_metrics_p1_public_red01.log`. Root error at `trace_mapping.py:161`, reached through `metrics_store.py:4370 -> 4504`. The public result had no owner root cost and a valid authoritative layer cost.
- completed: expanded same call-site RED to independently cover first, last, dummy-first and three malformed/non-authoritative owner cases: `6 failed, 24 passed in 47.13s`, `/data/ycfeng/tmp/d23_root_metrics_p1_callsites_red02.log`. The negative authoritative missing-proof regression already passed, proving strict validation was active.
- cause confirmed: owner recursion projected all layer work before filtering stage-owned names. Boundary replacement intentionally reconstructs numerical attention without OperatorCost; the stage contract ignores that non-authoritative owner layer payload. Layer proof loss and layer double billing were falsified by inspecting valid authoritative roots and exact mixed totals.
- completed: move the existing scalar/canonical stage field projection before aggregation and reuse it for scalar or Stage results. Remove recursive owner ledger and old later stage-field projection. No new projection path/flags or proof/predictor/replacement changes. Gain: correct ownership on reachable boundary results and net minus 22 metrics-store lines (5,825 -> 5,803).
- completed: focused call sites GREEN first try: `30 passed in 46.72s`, `/data/ycfeng/tmp/d23_root_metrics_p1_callsites_green01.log`. Expanded affected suite `167 passed, 1 deselected in 56.83s`, `/data/ycfeng/tmp/d23_root_metrics_p1_affected_green01.log`.
- completed: final suite after removing fake-valid mixed owner proof and tightening first/last metric/trace boundary assertions: `167 passed, 1 deselected in 47.57s`, `/data/ycfeng/tmp/d23_root_metrics_p1_final_verified.log`. First/last actual schedule and pending ledger match public durations; roots retain layer cost/count/hash, boundaries occur once. Dummy-first is checked via its public Stage ledger only, not a synthetic request-bearing trace wrapper.
- completed: the initial mixed owner fixture now has no proof. Added owner absence, invalid count, overlapping child/model projection, unrelated FFN/norm, and 99 ms owner layer gap cases; stage ignores them while the same object fails if used as an authoritative layer. Existing actual layer invalid-scope tests stay strict.
- completed: code/tests/exception committed as `7c70461`; four scoped files, 108 insertions / 45 deletions overall, metrics source 19 insertions / 41 deletions. No predictor, attention validator, or parent edits.
- completed: inspected actual public-result values with in-process diagnostics only: `/data/ycfeng/tmp/d23_root_metrics_p1_public_values.log`, `3 passed in 33.63s`. Owner root costs empty; first/last actual layer cost 1.016 ms / 23 kernels, .16 ms once-only boundary, .023 ms layer gap. Totals match runtime to floating-point epsilon; dummy-first .321 ms / 17 kernels and .01 ms cast, .341 ms total.
- completed: P1 cause, exact RED/GREEN commands/results, summary, requirements and scope records finalized in this closing records commit. No debug instrumentation retained in source. Final requirements reread and diff checks passed; no unfinished in-scope work. Parent owns rereview/merge/full CLI/new77; inherited cache fixtures remain outside scope.

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Public predictor P1 RED | PASS | Six call-site failures reproduce proof-free owner error before source edits. |
| Stage-only owner correction | PASS | Existing stage field projection reused; layer proof validation untouched. |
| Affected checks | PASS | 167 pass / one inherited test deselected; first/last metrics and traces verified. |
| Code / records commits and archive | PASS | Code 7c70461 committed; all five records complete in this closing records commit. |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** P1 follow-up request 至今｜**当前前沿：** Return exact artifacts and limits to parent.

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | Existing public fixture extended only to requested boundary metrics/ledger/trace acceptance. |
| D2 细节反复重试 | CLEAR | One production correction; first and final affected runs GREEN. |
| D3 支线问题修补 | CLEAR | No inherited fixture repair or generic gap-span changes. |
| D4 可并行任务停滞 | CLEAR | Explicit no-agents instruction; independent CPU checks ran concurrently. |
| D5 超范围镀金 | CLEAR | Existing owner projection moved, duplicate removed; no new flags/abstractions. |
| D6 分析空转 | CLEAR | Diagnosis produced RED tests and a verified fix. |
| D7 路线违背 | CLEAR | Public RED before fix; no manufactured proof, predictor changes, delegation/platform, or parent edits. |

**文档检查：** S1–S6 均未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. Return to parent → 完成标志：Clean worktree, exact code/record commits, P1 evidence and parent-owned limits returned.

## 4. 最值得关注的 3 点

1. Stage owner no longer goes through layer root validation; authoritative roots still require proof.
2. Public first/last results pass metrics/trace with once-only source boundaries.
3. Rereview/full CLI/new77 remain parent-owned and unverified here; baseline exclusions remain explicit.

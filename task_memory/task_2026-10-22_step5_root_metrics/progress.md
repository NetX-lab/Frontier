# D23-ROOT-METRICS Progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record verified baseline, RED evidence, and inspected projection failures. |

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
- completed: English summary and exact test report written; final requirements reread. Records committed as `24b889d` after code commit `3a189ba`; final git status clean and baseline-to-HEAD diff check passes. Parent retains full CLI/native/admission/legacy77 acceptance and two baseline fixture issues; no unfinished in-scope work.

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Baseline / fast repro | PASS | Exact HEAD and RED traceback recorded. |
| Call-site regression | PASS | Eight pre-fix failures match root blockers. |
| Existing projection fix | PASS | Root-only registry projection and truthful proof serialization. |
| Focused verification | PASS | 161 passing checks; exact original repro GREEN. |
| Self-check / commits / archive | PASS | Code 3a189ba and records 24b889d committed; clean worktree and diff check confirmed. |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** Baseline RED 至今｜**当前前沿：** Return scoped commits and exact limits to parent.

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | Focused tests settle requested metadata, totals, and malformed-scope criteria. |
| D2 细节反复重试 | CLEAR | Initial fixture TP and inclusive-registry corrections each followed concrete errors; production root fix passed first run. |
| D3 支线问题修补 | CLEAR | Baseline stale fixtures investigated once and left unchanged. |
| D4 可并行任务停滞 | CLEAR | Explicit no-agents instruction; independent CPU checks ran concurrently. |
| D5 超范围镀金 | CLEAR | Only existing projection/registry wiring and focused tests added. |
| D6 分析空转 | CLEAR | Diagnosis produced RED tests and a verified fix. |
| D7 路线违背 | CLEAR | No input/scalar changes, fake child costs, disabled ledger, native/GPU runs, or parent edits. |

**文档检查：** S1–S6 均未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. Return to parent → 完成标志：Exact commits, evidence paths, and pending parent acceptance returned.

## 4. 最值得关注的 3 点

1. Original root is retained once with `.6244 ms` and `23 kernels`; child diagnostics remain unbilled.
2. Baseline test-fixture failures are not suppressed or treated as code PASS.
3. Full CLI/native/admission/legacy77 remain parent-owned and unverified here.

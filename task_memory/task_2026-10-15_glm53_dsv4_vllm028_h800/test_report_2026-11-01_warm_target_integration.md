# D-WARM-TARGET integration

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Recorded the warm-reading deliverables, unchanged harness regression, archive verification, and pending phase review. |
| 2026-11-01 | Accepted both phase reviews, archived evidence, integrated the exact root skill/prerequisites and approved README, and completed validation. |

## Acceptance and evidence

Authority: `requirements.md`, D-WARM-TARGET. The target is the warmed serving path. Comparison excludes only requests carrying run-evidenced one-time events, symmetrically within each repeat, and retains all three clean repeats and the immutable all-request reading.

| Deliverable | Observed result | Evidence |
| --- | --- | --- |
| Event-based request identification | Canonical P2-32: no formal opener. Canonical P2-64: repeat01 r006/r017/r027/r035/r036; later repeats none. Formal event bindings are unique, without latency selection. | `h3o_operator_costs/warm_target_20261101/openers.json` and its report |
| Matched warm readings | Six readings, 96/96 warm metrics PASS. Original P2-64 repeat01 TTFT FAIL readings retained. | `h3o_operator_costs/warm_target_20261101/readings/` |
| Residual analysis | Six replay decompositions; 9/9 arithmetic checks PASS. Probe-derived fine spans and price-neighborhood sensitivity remain diagnostic. | `h3o_operator_costs/warm_residual_20261101/test_report_2026-11-01_warm_ttft_residual.md` |
| READ/payload and P-step accounting | 7/7 checks PASS; two existing-path structural discrepancies confirmed; remaining macro residual unresolved | `h3o_operator_costs/warm_accounting_20261101/test_report_2026-11-01_warm_accounting.md` |
| Existing client gate regression | 40 PASS | `h3o_operator_costs/warm_target_20261101/primary_harness_regression_command.json` |
| Initial investigation archive | 18 existing files verified; two documented archival script adjustments; no rerun | `h3o_operator_costs/p2_64_ttft_20261101/{archive_manifest.json,notes.md}` |
| Task records | WARM-DOCS reported 71/71 checks PASS; primary corrected a window-wording mismatch against the reused harness | `plan.md`, `progress.md`, `summary.md`, tickets 05/09/19, and `check_progress_2026-11-01_warm_target.md` |
| Calibration skill consistency | Six findings CLOSED; independent review reports no actionable skill finding; committed at repository root as `0cc4118a` | `/data/ycfeng/Frontier/skills/frontier-calibration/`; `task_memory/task_2026-11-01_calibration_warm_target/review.md` in that checkout |
| Independent final review | No findings; independently recalculated 96/96 warm metrics PASS, maximum error 5.952669% | WARM-FINAL-REVIEW, `ses_e68dddb1effeH8TbRzOxiHxLPx` |
| Cross-branch integration | Complete; 34/34 skill/README files match requested root; both validators PASS | `f31b2b2d`, `24ddbbb6`, `f21cdcdb`; root skill `0cc4118a` and closeout `d4924d2e` |

## Numerical interpretation

The final P2-64 paired pool retains 379 native request-repeat rows and the corresponding Frontier rows. Native TTFT is 720.639680 ms; B29/pass1/pass2 predict 684.545783/701.662204/700.865508 ms, signed errors -5.008591%/-2.633421%/-2.743975%. Repeat01 errors are -4.599811%/-1.970116%/-2.066263%; all later repeats remain represented. P2-32 retains all 384 request-repeat rows, with maximum error 5.952669% across the three generations.

The raw all-request P2-64 repeat01 TTFT errors remain 12.254941%/10.169033%/10.277737%. No reading was replaced in place. The new population reading does not reconstruct scheduling after a hypothetical stronger warmup.

Residual estimates use their declared populations. In particular, the approximately 689 ms price-neighborhood proxy is not a fixed-price long-window prediction and is not a gate reading. Numerical warm PASS does not close the 48 unevaluable attention workloads, remaining NIXL/hardware/memory input qualification, or H800 scope.

## Primary verification

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_e2e_gate.py -q -p no:cacheprovider
```

Python 3.10.6. Expected: unchanged harness tests all pass. Observed: **40 passed in 1.34 s**, exit 0. The command receipt records source commit, UTC times, environment, and SHA-256 of `/data/ycfeng/tmp/opencode-v2-tmp/opencode/warm_target_harness_regression_20261101.log`.

The warm-reading producer independently verified original/empty-set reproduction, both-side pooling totals, source SHA-256, and all 192 all-request/warm CSV rows; exact commands are in its `command_receipts.json`. This integration report does not substitute the unchanged harness tests for those numerical checks.

Primary documentation check found that ticket19 initially described warm throughput over fixed original endpoints. The implementation reuses `metric_values()` (`tests/comparison/calibration/e2e_metrics_gap.py:123–132`), which derives the window from the retained rows. Ticket19 now describes that actual rule. The five current exclusions are interior, so both endpoint rules coincide on these retained P2 readings; no numerical output or source changed.

The archive's CSV writers retain CRLF terminators. The default staged whitespace check flags those terminators; `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check` passes with zero diagnostics. No artifact bytes or persistent Git setting were changed to address that format-only report.

## Phase review and acceptance

WARM-FINAL-REVIEW (`ses_e68dddb1effeH8TbRzOxiHxLPx`) inspected the current requirements/plan/progress/tickets05/09/19, the specified warm/residual scripts and JSON/CSV/reports/receipts, this integration report, and the primary regression receipt. Result: **No findings**. It independently recalculated all 96 warm metrics (maximum error 5.952669%), verified repeat-specific populations, inspected event/source identification without a latency threshold, and recalculated FIFO discrepancies below 0.103 ms. It confirmed the retained-window rule, signed/absolute error distinction, original three FAIL readings, and diagnostic limitations.

The reviewer did not rerun file-level row/hash/event verification; those checks rely on inspected scripts, artifacts and command receipts. It read the 40-PASS regression log without rerunning it. WARM-ACCOUNTING and the separate skill review were outside that review's acceptance scope. Primary separately inspected the decisive accounting source and validation receipt below. The skill has its own accepted phase review at repository-root `task_memory/task_2026-11-01_calibration_warm_target/review.md`.

Primary accepts the retained-reference warm-reading and residual diagnostic deliverables. This acceptance is not formal input admission or a stronger-warmup counterfactual run. Evidence/helper/requirements are archived in `0cd8d5d3`, with status records in `4c3208d1`. Skill integration is complete: prerequisite commits `c3ba1712`, `7077960f`, `33ff10fd`; warm-target skill `f31b2b2d`; approved README `24ddbbb6`; shared skill-task closeout `f21cdcdb`.

### Integration validation and README dependency

The only integration conflict was the branch's older `skills/README.md`, initially preserved. The structure validator then failed on its personal `vLLM-BS` wording. The primary recovered the explicit prior README/generalization request at repository-root `skills/task_memory/task_2026-11-01_frontier_skills_readme/requirements.md:9–11,24–35`, and imported exactly the already-approved `98b680a2:skills/README.md`. No new README prose or validator exception was introduced.

Final commands in the active worktree, Python3.10.6:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 skills/frontier-calibration/tests/pressure_scenarios.py skills/frontier-calibration
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/codex-home/skills/.system/skill-creator/scripts/quick_validate.py skills/frontier-calibration
git diff --check
```

Results: `v2_pressure=PASS`, `Skill is valid!`, and zero whitespace diagnostics. All 33 tracked skill files plus README match `/data/ycfeng/Frontier` byte-for-byte; README SHA-256 is `594fa59a99a6e3b07e01095ea7a29bebe1163f82dfd2b344552e73153004473b`. All eight shared skill-task files also match both checkouts after the closeout cherry-pick. No merge conflict remains. This completes the approved warm-target continuation; only the separate next correction decision below is pending.

## Next correction decision

WARM-ACCOUNTING has completed both checks. Primary inspected `analytical_kv_cache_transfer_predictor.py:19–63,126–135`, `operators/families.py:802–816`, `operators/spec.py:199–200`, `dsv4_runtime.py:31–38`, and the seven-check validation receipt. The source supports these two findings:

1. Grouped NIXL selects 46 physical pages but prices 39,521,664 component bytes instead of 46,103,040 packed bytes per TP handle. Use selected pages times the existing physical stride only for wire pricing, preserving structural/ownership semantics. At the current bandwidth the direct wire increase is 0.417203 ms; mean TTFT impact is unmeasured.
2. The COMM registry inherits one kernel for EP dispatch despite the native three-call source and 129/43-layer witnesses. Recording dispatch count 3, with combine count 1, would add 0.046836 ms to direct eager P-step gap pricing. This is not a supported mean TTFT estimate.

Both are correction proposals, not implemented changes. The remaining READ lifecycle requires per-worker handle/post/poll/report evidence; the recurring host estimate cannot override the explicit host-zero decision. Investigation stops at those evidence limits. The next decision is whether to implement and CPU-verify the two existing-path corrections. Any additional full replay, new modeling, measurement-input replacement or GPU run needs its applicable authorization.

The primary requested approval for these two fixes, affected CPU regression, and two fixed-input P2-32/P2-64 CPU replays (about three hours in parallel, zero GPU). No human approval has yet been recorded; an automated subagent completion is not that decision.

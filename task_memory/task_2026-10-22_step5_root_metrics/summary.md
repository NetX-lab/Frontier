# D23-ROOT-METRICS Completion Summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archive scoped fix, exact evidence, and parent-owned acceptance limits. |
| 2026-10-22 | Close reviewer P1 stage-owner correction with public-result metrics/trace evidence. |

## Outcome

The demonstrated measured-root metrics projection blocker is fixed at the existing attention/ledger seam. The supported scalar-free inclusive root is retained exactly once, while its diagnostic children remain unbilled. Actual stage scheduling publishes the root operation and per-layer OperatorCost into trace events and persisted JSONL without relabeling source evidence or inventing child costs.

Reviewer P1 exposed a reachable owner-only path missed by the initial fixture:
public first/last boundary owners retain attention duration without proof.
The owner now projects only existing stage-owned scalar/canonical fields, never
its irrelevant layer payload; authoritative layers still require valid costs.
Predictor and replacement semantics are unchanged.

Task: `D23-ROOT-METRICS`; session: `ses_e9f9fd0bdffeZH9Q2Ux5NbCYko`.
Worktree: `/data/ycfeng/Frontier/.worktrees/step5-root-metrics`; branch: `feat/step5-root-metrics`.
Baseline: `61cbac9a3e8c29190b91a5d4e8f7fa2ad9fcd359`; initial code: `3a189ba`; P1 code: `7c70461` from local `9ef9cd0`.

## Deliverables

- `frontier/attention/ops.py`: registry-role classification for a scalar-free inclusive attention root.
- `frontier/attention/trace_mapping.py`: measured-root selection and cost/physical-scope validation through the existing projection.
- `frontier/attention/step5.py`: source input/output tensor boundaries for registered roots, without invented internal child costs.
- `frontier/metrics/constants.py`: required root OperationMetrics entry.
- `frontier/metrics/metrics_store.py`: truthful per-layer proof serialization, absent-family filtering, and strict layer projection reused by stage aggregation with owner-only stage fields. P1 removes recursion and duplicate stage projection: net minus 22 lines, 5,803 total (14 below initial baseline).
- `tests/unit/test_measured_attention_root_metrics.py`: 27 cases covering original root, layer/stage ownership, mixed totals/gaps, schedule/trace/JSONL proofs, malformed scopes, renamed registry classification, legacy inclusive paths, absent dense tables, and owner-only payload isolation. The initial mixed owner no longer supplies an artificial valid proof.
- `tests/unit/test_step5_measured_public_predictor.py`: three independent first/last/dummy-first public measured-result cases. First/last pass actual metrics/trace/boundary publication; dummy-first passes its real Stage ledger.
- `docs/general/module-size-exceptions.md`: scoped justification and unchanged deferred split.
- Task records: `requirements.md`, `plan.md`, `progress.md`, `test_report_2026-10-22_root_projection.md`, and this `summary.md`, all in `task_memory/task_2026-10-22_step5_root_metrics/`.

## Validation

- Original pre-fix reproduction: exact KeyError at metrics_store.py:4504.
- Pre-fix regression: eight failures matching the KeyError/trace root rejection, seven passing negative cases.
- Initial focused/adjacent run: `161 passed, 1 deselected in 27.48s`, log `/data/ycfeng/tmp/d23_root_metrics_final_verified.log`.
- P1 pre-fix public call-site RED: `6 failed, 24 passed in 47.13s`, log `/data/ycfeng/tmp/d23_root_metrics_p1_callsites_red02.log`.
- Final P1 affected suite: `167 passed, 1 deselected in 47.57s`, log `/data/ycfeng/tmp/d23_root_metrics_p1_final_verified.log`.
- Public owner root costs are empty while authoritative first/last roots retain 1.016 ms / 23 kernels, boundary .16 ms once, gap .023 ms; ledger/runtime totals agree to floating-point epsilon. Exact evidence `/data/ycfeng/tmp/d23_root_metrics_p1_public_values.log`, three cases pass.
- Original reproduction GREEN: root/ledger/stage exactly `.6244 ms`, count `23.0`, log `/data/ycfeng/tmp/d23_root_metrics_repro_green01.log`.
- Mixed-stage sum `1.8074000000000001 ms` matches runtime; root `.8244 ms`, separate gap `.023 ms`, stage owner scopes once. Exact component evidence: `/data/ycfeng/tmp/d23_root_metrics_mixed_total.log`.
- `git diff --check` passed. No further agents, GPU/native runs, push, README edits, input changes, predictor changes, or parent-worktree edits.

## Open / deferred

Rereview/merge/full CLI/new77 and actual offline/online/native/admission/READ/readiness/source-release acceptance remain parent-owned and unverified here. Two baseline old-test fixture defects remain (deleted private cache name and missing monkeypatch cache isolation); direct baseline evidence and exact exclusions are in the test report. Existing generic trace omission of kernel-gap spans is unchanged; the root proof retains its truthful gap-inclusion flag and the component ledger retains the separate gap charge.

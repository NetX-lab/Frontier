# D23-ROOT-METRICS Completion Summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archive scoped fix, exact evidence, and parent-owned acceptance limits. |

## Outcome

The demonstrated measured-root metrics projection blocker is fixed at the existing attention/ledger seam. The supported scalar-free inclusive root is retained exactly once, while its diagnostic children remain unbilled. Actual stage scheduling publishes the root operation and per-layer OperatorCost into trace events and persisted JSONL without relabeling source evidence or inventing child costs.

Task: `D23-ROOT-METRICS`; session: `ses_e9f9fd0bdffeZH9Q2Ux5NbCYko`.
Worktree: `/data/ycfeng/Frontier/.worktrees/step5-root-metrics`; branch: `feat/step5-root-metrics`.
Baseline: `61cbac9a3e8c29190b91a5d4e8f7fa2ad9fcd359`; verified code commit: `3a189ba`.

## Deliverables

- `frontier/attention/ops.py`: registry-role classification for a scalar-free inclusive attention root.
- `frontier/attention/trace_mapping.py`: measured-root selection and cost/physical-scope validation through the existing projection.
- `frontier/attention/step5.py`: source input/output tensor boundaries for registered roots, without invented internal child costs.
- `frontier/metrics/constants.py`: required root OperationMetrics entry.
- `frontier/metrics/metrics_store.py`: truthful per-layer proof serialization, absent-family filtering, and the existing layer ledger projection reused by stage aggregation. Net growth eight lines; no second ledger owner.
- `tests/unit/test_measured_attention_root_metrics.py`: 23 cases covering original root, layer/stage ownership, mixed stage totals/gaps, actual schedule/trace/JSONL proofs, malformed scopes, renamed registry classification, legacy inclusive paths, and absent dense tables.
- `docs/general/module-size-exceptions.md`: scoped justification and unchanged deferred split.
- Task records: `requirements.md`, `plan.md`, `progress.md`, `test_report_2026-10-22_root_projection.md`, and this `summary.md`, all in `task_memory/task_2026-10-22_step5_root_metrics/`.

## Validation

- Original pre-fix reproduction: exact KeyError at metrics_store.py:4504.
- Pre-fix regression: eight failures matching the KeyError/trace root rejection, seven passing negative cases.
- Final focused/adjacent run: `161 passed, 1 deselected in 27.48s`, log `/data/ycfeng/tmp/d23_root_metrics_final_verified.log`.
- Original reproduction GREEN: root/ledger/stage exactly `.6244 ms`, count `23.0`, log `/data/ycfeng/tmp/d23_root_metrics_repro_green01.log`.
- Mixed-stage sum `1.8074000000000001 ms` matches runtime; root `.8244 ms`, separate gap `.023 ms`, stage owner scopes once. Exact component evidence: `/data/ycfeng/tmp/d23_root_metrics_mixed_total.log`.
- `git diff --check` passed. No further agents, GPU/native runs, push, README edits, input changes, predictor changes, or parent-worktree edits.

## Open / deferred

Full CLI, actual offline/online/native/admission/READ/readiness/source-release and legacy77 acceptance remain parent-owned and unverified here. Two baseline old-test fixture defects remain (deleted private cache name and missing monkeypatch cache isolation); direct baseline evidence and exact exclusions are in the test report. Existing generic trace omission of kernel-gap spans is unchanged; the root proof retains its truthful gap-inclusion flag and the component ledger retains the separate gap charge.

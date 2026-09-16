## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-15 | Reclassified the checklist against the independent v1.1 completion audit; historical execution claims are retained but no longer treated as current acceptance closure. |
| 2026-09-15 | Updated item dispositions and bound final evidence to candidate `b8cecf53`. |
| 2026-09-15 | Created from Frontier_PR33_review_revision_plan_2026-09-15_v1.1 before source changes. |

# PR33 v1.1 Revision Execution Checklist

Specification: `Frontier_PR33_review_revision_plan_2026-09-15_v1.1_en.md`.

## Ordered work

- [~] W0/R12: clean baseline and final paired runs exist, and the historical 9.42x result is not reproduced; the residual hotspot has not received a narrow causal ablation or allocation/RSS closure.
- [~] R01: binding helpers and focused tests exist, but the real `BaseModelConfig` MLA path can still be misclassified and the required public-predictor integration matrix is missing.
- [~] R02: effective capacity reaches `MemoryPlanner` and both automatic constructors are exercised, but direct block monotonicity, non-GDN zero reservation, and structured automatic OOM acceptance are missing.
- [~] R03: scheduler slot paths and private-method tests exist, but a real two-request Simulator lifecycle and reachable cancellation/termination evidence are missing.
- [~] R04: explicit phase/mask validation and positive/rejection tests exist, but cold one-token prefill and no-native/no-artifact side-effect assertions are missing.
- [~] R07: a shared helper covers loading/predictor paths, but the public manager resolver and a complete three-entry parameter matrix remain.
- [~] R05/R06/R08: timing/cache optimizations exist, but dual semantics remain, the operator-ownership seam is absent, and resource/causal evidence is incomplete.
- [~] R09: a genuine synthetic real-constructor E2E exists, but automatic modes, lifecycle, shared experts, TP>1, MLA, and homogeneous EP/PP coverage are missing.
- [~] R10: CPU timer contracts and explicit hardware SKIPs exist, but required GPU test entry points, real execution, and the complete timer cleanup contract are not closed.
- [~] R11: the 58-case matrix passes, but operation/trace artifacts, available non-dummy/golden lanes, and auditable node-level baseline comparison are missing.
- [ ] R12 final: paired unprofiled performance and causal optimization are complete, but significant dense residual cost requires D02 user disposition before closure.
- [ ] R13/R14: R13 cleanup/split analysis is absent; R14 now has an audit report but cannot close while the other packages and evidence gates remain open.

`[~]` means implementation/evidence exists but the v1.1 acceptance condition is
not complete. The exact-HEAD checkpoint below is retained as historical
execution evidence, not as a current all-items PASS.

## Decision gates

- D01 is triggered only by a stable cross-version numerical/discrete difference beyond existing tolerances; complete RCA and ask the user before accepting a semantic treatment.
- D02 is triggered only by significant measured residual simulator implementation cost after causal optimization; it cannot accept the historical ~9.42x degradation.
- D03 is triggered only if call-site inventory proves no reachable cancellation/termination entry point exists for slot cleanup.

## Acceptance evidence

- The independent audit found that not every accepted R item has sufficient
  focused/integration evidence at the exact candidate SHA; see
  `completion_audit_2026-09-16.md`.
- CPU regressions have no unexplained candidate-only failure; baseline failures remain identified by node ID and cause.
- AMD/MI355X, ROCm, NVIDIA, benchmark, and groundtruth checks are PASS only with actual hardware; otherwise retain explicit SKIP reasons.
- No user decision is silently assumed; no merge/rebase/force-push/branch deletion is performed.

## Exact-HEAD evidence checkpoint — 2026-09-15

- Candidate: `b8cecf53f8b81ea8380238971277ba94c6fe4961`; baseline: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`.
- Full unit: **3309 passed, 19 baseline failures, 25 skipped, 576 warnings**; no candidate-only failure.
- Concurrent broad groups: **1234 passed, 19 skipped** after correcting two nonexistent test path names in the initial harness invocation.
- Fidelity: **58 passed, 0 failed**, artifacts `/data/ycfeng/tmp/pr33-r11-fidelity-20260915-b8cecf53`.
- Paired performance: **18/18 successful**, artifacts `/data/ycfeng/tmp/pr33-r12-paired-20260915-final-b8cecf53`; dense residual remains under D02.
- Hardware: `SKIP: AMD/MI355X hardware unavailable`; NVIDIA: `SKIP: no visible NVIDIA device`.

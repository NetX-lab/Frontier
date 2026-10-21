# D20-PD progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Preserve the closed R2 regression and start measured PD implementation. |

- completed: R2 P1 regression closed at `cb09a27`, 636 affected tests passed and independent follow-up review passed. Old worktree is clean at `b6ba7e5`; no further R2 work is authorized by this assignment.
- completed: inspected primary D20 requirements/plan/ticket17, repository development rules, existing transfer events, transfer predictor and Step payload resolver. The current measured guard fails before ownership changes. Existing transfer duration is analytical; the Step payload explicitly assumes whole-prompt/open-tail state.
- completed / A: focused scheduler/Q10 trace returned as D20-PD-trace, session `ses_ea308e44fffegeocAaodWhIRyC`; primary inspected the exact shared commit. `git merge-base --is-ancestor 6cc09dc HEAD` returned 1: this fix is not already present at base6db2250.
- in-progress / B,C: coordinate measured transfer geometry/coverage and cost interface with primary/computation owner. No shared-family or computation-owned source edits.
- pending / D,E: vertical CPU integration tests, code changes, affected regression, commit and small actual measured run after provider availability.

## Verified computed-context slice

- Reason: the actual Step NIXL scheduler produced contexts4105..4113 instead of computed-KV contexts4104..4112. Profile lookup must use actual computed context, not accepted-output progress.
- Method: port the exact model-agnostic `_create_batch` expression from shared6cc09dc. No family-specific callback, new timing or token policy.
- RED: `d20_pd_context_red02` completed all9 outputs and failed only the context assertion. The preceding attempt was a fixture setup error (`_StageContexts` lacks DP peer lookup); the final test uses the existing real four-lane cluster scheduler.
- GREEN: the dedicated offline/online lifecycle and existing Step structural, placement, GLM structural and PD client-TTFT tests passed70/70 in74.19s. Bytes103687956, nine emitted outputs and final pool/frontier balances are unchanged.
- Existing scheduler module is above the repository soft size limit. The touched method has no redundant code to remove; this slice reuses the reviewed upstream expression rather than adding another scheduler abstraction. Existing iteration/allocation/role scheduling mixins remain the functional split; unrelated cleanup would change the approved patch unnecessarily.
- in-progress / C: Request callback extension delegated to `ses_ea2f95d90ffeMAWaEnG4wQPVbS`, ownership only Request and its new dedicated tests. It reuses existing `_num_recomputed_tokens`, which already gives prefill cost shape and grants one output at catch-up; no new token lifecycle.

## Route correction

The active frontier is measured PD implementation, not R2 capture preparation. Existing native evidence remains a separate acceptance prerequisite; missing native proof must not prevent CPU implementation and must not be replaced with a fixture claim. Stop R2 tooling/staging expansion and do not rerun the full legacy matrix in this worktree.

# D20-PD progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Preserve the closed R2 regression and start measured PD implementation. |

- completed: R2 P1 regression closed at `cb09a27`, 636 affected tests passed and independent follow-up review passed. Old worktree is clean at `b6ba7e5`; no further R2 work is authorized by this assignment.
- completed: inspected primary D20 requirements/plan/ticket17, repository development rules, existing transfer events, transfer predictor and Step payload resolver. The current measured guard fails before ownership changes. Existing transfer duration is analytical; the Step payload explicitly assumes whole-prompt/open-tail state.
- completed / A: focused scheduler/Q10 trace returned as D20-PD-trace, session `ses_ea308e44fffegeocAaodWhIRyC`; primary inspected the exact shared commit. `git merge-base --is-ancestor 6cc09dc HEAD` returned 1: this fix is not already present at base6db2250.
- completed / B,C,D (CPU consumer): admitted coverage, zero-query remote reservation, READ-to-GPU-ready cost, recomputation and existing owners are connected. Initial code a082c00 passed196; corrections9625984 passed201 in176.62s and independent follow-up review closed both findings. No production supplier/native qualification is claimed.
- completed / E (stable handoff): exact hooks/configuration and missing inputs are archived. Primary now owns production transfer provider and small integrated main; no waiting for compute or further provider edits. Actual main is NOT_RUN here. T2/P backoff remain outstanding; P still computes the whole prompt.

## Verified computed-context slice

- Reason: the actual Step NIXL scheduler produced contexts4105..4113 instead of computed-KV contexts4104..4112. Profile lookup must use actual computed context, not accepted-output progress.
- Method: port the exact model-agnostic `_create_batch` expression from shared6cc09dc. No family-specific callback, new timing or token policy.
- RED: `d20_pd_context_red02` completed all9 outputs and failed only the context assertion. The preceding attempt was a fixture setup error (`_StageContexts` lacks DP peer lookup); the final test uses the existing real four-lane cluster scheduler.
- GREEN: the dedicated offline/online lifecycle and existing Step structural, placement, GLM structural and PD client-TTFT tests passed70/70 in74.19s. Bytes103687956, nine emitted outputs and final pool/frontier balances are unchanged.
- Existing scheduler module is above the repository soft size limit. The touched method has no redundant code to remove; this slice reuses the reviewed upstream expression rather than adding another scheduler abstraction. Existing iteration/allocation/role scheduling mixins remain the functional split; unrelated cleanup would change the approved patch unnecessarily.
- completed / C: Request callback extension delegated to `ses_ea2f95d90ffeMAWaEnG4wQPVbS`, ownership only Request and its new dedicated tests. RED1, then14 validation failures/1 pass; GREEN26 dedicated and84 affected. It reuses existing `_num_recomputed_tokens`, which already gives prefill cost shape and grants one output at catch-up; no new token lifecycle.

## Measured consumer vertical slice

- RED: the real non-dummy four-lane constructor rejected the coverage-bearing fixture as dummy-only (1 failed,5.52s).
- Decision: user assigned the minimal transfer predictor/info extension to this owner, then approved zero-query reservation in the existing grouped plan. Both are recorded in requirements.md.
- Core gain: the allocation API could not reserve an already-computed remote prefix without claiming a new query. The narrow extension uses the same plan/pool with an explicit positive frontier and no query/lookahead. Ordinary zero-token calls remain invalid; scheduler receiving state bounds this caller.
- A self-check found serial guard discovery. The reserve/can_allocate/plan/allocate chain was then read in full; the original measured scenario passed1/1 in9.03s.
- Fixture route: P4105; retained P ownership; declared D prefix4080; reserve1020 blocks;103051776 modeled bytes/worker lane; covered READ3.5ms; no pre-ready scheduling; first-D eager prefill(query25,context4080);1+8 emitted outputs; empty final P/D allocations/frontiers.
- No analytical fallback: the measured positive test makes analytical get_transfer_time raise if invoked. Timing comes only from the supplied fixture's coverage/cost methods.
- Negative controls: missing provider/request coverage, prompt/TP/device/alignment mismatch, network-only readiness, empty identity, invalid duration, D budget mismatch and rollback after missing cost. A missing cost originally leaked a frontend owner; that owner now releases its selection.
- A speculative received-page zeroing check found cache_block_zeroing=None for this fixture. No unproven zeroing behavior was added or modified.
- Compatibility: exact diff6db2250..9f58dbe confirms non-grouped GLM allowed non-dummy analytical transport. New coverage is limited to grouped measured transfers via the existing capacity classification; shared GLM's three-output lifecycle is tested.
- Budget RED reproduced a lingering source claim. Prefix/budget checks now use the existing every-lane preflight before placement/claims, rather than waiting for reservation mutation.
- green05:79 PASS/4 FAIL from accessing NIXL-only metadata on non-NIXL SimpleNamespace inputs; the access is now confined to NIXL. green06:194 PASS; final green07:196 PASS. The full77-case fidelity matrix was not repeated.

## Independent review corrections

- D20-PD-review, session `ses_ea2b31136ffeNlPFBrPWSNlz0s`, statically found coverage replacement after FIFO retry reservation and missing cleanup on measured completion-validation failure. It did not execute tests.
- RED:3 completion mutations failed owner assertions in `d20_pd_review_red01` (31.56s total, plus1 fixture error because a capacity wait legitimately emits a ReplicaScheduleEvent). After fixing only that expectation, P1 failed DID NOT RAISE RuntimeError in `d20_pd_review_red02` (9.28s).
- Fix9625984: freeze first accepted coverage; reject any changed retry coverage before overwriting it and unwind target reservation/selection/P claim. Keep P pending KV. One READ validation boundary performs measured cleanup before rethrowing, without ready/end metrics; legacy errors preserve their path.
- GREEN:all4 reproduced negatives passed31.87s. Added unchanged-coverage capacity retry completing9 outputs, and actual cluster batch-end publication to assert normal frontend ownership release. Final affected command passed201 in176.62s (`d20_pd_review_green02`).
- Follow-up review closed both findings with no new actionable blocker in this increment. Reviewer executed no tests;201-test evidence is the main agent's run. Exact independent6cc09dc/9f58dbe artifacts verified the historical comparisons. Supplier/main/T2/native and four10% gates remain open.

## Route correction

The active frontier is measured PD implementation, not R2 capture preparation. Existing native evidence remains a separate acceptance prerequisite; missing native proof must not prevent CPU implementation and must not be replaced with a fixture claim. Stop R2 tooling/staging expansion and do not rerun the full legacy matrix in this worktree.

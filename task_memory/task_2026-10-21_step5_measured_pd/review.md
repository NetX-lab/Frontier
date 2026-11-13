# D20-PD consumer review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archive independent findings, actual reproductions and accepted9625984 remediation. |

## Component and reviewer

- Component: existing PD transfer events, reservation/recompute ownership and measured coverage/cost consumption.
- Reviewer: native `reviewer`, task D20-PD-review, session `ses_ea2b31136ffeNlPFBrPWSNlz0s`.
- Initial artifact: `d20_pd_implementation_a082c00.diff`, exact base6db2250..a082c00, source/tests plus the five task records. Review also traced Batch, allocation, C16 FIFO retry and arrival/completion. Reviewer executed no tests.

## Findings and resolution

| Priority | Finding | Actual main-agent evidence | Remediation |
| --- | --- | --- | --- |
| P1 | Retry replaced coverage after reserving the old prefix, pairing mismatched cursor/payload/allocation | red02:1 failed9.28s, no expected error for4080->4096 |9625984 freezes first admission; any retry difference rejects before replacement and releases owners |
| P2 | READ coverage/cost validation errors left D receiving/frontier/frontend and P claim alive | red01:3 true completion-mutation cleanup failures |9625984 validates before publication and aborts measured transfer owners on any validation exception; P pending KV is preserved |

Initial red01 also had one test setup error: capacity wait emits a ReplicaScheduleEvent. The corrected retry test produced the independent P1 RED before source changes. All4 reproduced negatives then passed31.87s. Normal and unchanged-coverage retry paths publish real cluster batch-end callbacks, emit9 tokens and empty all owners. Final affected201 tests passed176.62s.

## Follow-up scope

- Exact incremental artifact: `d20_pd_review_fix_9625984.diff` (a082c00..9625984).
- Independent historical artifacts: `d20_shared_context_6cc09dc.diff` and `d20_shared_legacy_9f58dbe.diff`. All diff files are local reproducible evidence; Git commits preserve their source.
- Follow-up review:both original findings closed; no new actionable blocker in the increment. P1 refs:start event92–103,165–184. P2 refs:end event109–119. Reviewer checked all four negatives, unchanged-coverage retry and frontend release assertions, but executed no tests.
- Independent artifacts confirmed the exact shared context expression and9f58dbe's removal of the non-dummy prohibition. Actual main, supplier, T2, P backoff, canonical/native/profile admission and all four10% gates remain outside this CPU review acceptance.

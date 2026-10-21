# Plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Established bounded implementation and verification sequence. |

Dependency: native rules -> existing allocation/transfer implementation -> CPU
acceptance -> commit -> one clean strict 77-case comparison -> final receipt.

1. Inspect pinned coordinator/scheduler/connector group selection and retention;
   compare one unchunked 6144 prefill to ordinary decode pruning. Establish exact
   source-held/selected/destination group tables with CPU source-body controls.
2. Extend existing scheduler reservation and transfer payload consumers without
   duplicate allocation state; use capacity component pages for payload bytes.
3. Test actual Request, Batch, allocation manager and transfer event consumers,
   including rollback, completion preflight, capacity refusal and release.
4. Run focused V4/GLM/context checks; retain execution guards and capture failures.
5. Commit only owned code and tests, then run unchanged strict 77-case comparison
   once from clean source. Record exact diff, commands, results and limitations.

No parallel agent work; no external operations. Native semantic contradiction
requiring a material architecture change is escalated to the parent with one
concrete blocker rather than expanding investigation.

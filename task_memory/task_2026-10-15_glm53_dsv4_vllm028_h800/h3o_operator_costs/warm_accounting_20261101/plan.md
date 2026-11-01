# WARM-ACCOUNTING plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-11-01 | Defined two bounded checks; stop each after classification. |

Dependency: read authority -> {NIXL source/bytes/lifecycle check, P-step
retained-trace segment/input/count check} -> arithmetic validation and report.

1. NIXL: preserve native commit/source identity; compare actual grouped
   structural ledger bytes with native packed registration/descriptor bytes;
   determine what lifecycle timestamps are and are not retained.
2. P-step: join only current forward segments/scopes and relevant TP0 native
   NONE forwards; inspect current gap/attention stream ownership and kernel
   counts without rerunning a predictor. Separate device work from host and
   rendezvous/peer waits. Stop once an actual defect or observation gap is clear.
3. Report smallest evidence-supported proposals, expected effects only where
   justified, independent validation and cost. No implementation or promotion.

Checks use Python3.10, no GPU calls, ulimit -v31457280 and timeout60s for large
reads. Scripts and all outputs stay in this directory; no main-record edits.

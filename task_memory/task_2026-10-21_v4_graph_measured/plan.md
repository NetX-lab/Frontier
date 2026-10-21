# Plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Established bounded implementation and acceptance sequence. |
| 2026-10-21 | All four steps completed; implementation621b198 and sole strict77 IDENTICAL. |
| 2026-10-21 | Added the approved review-correction frontier; original completion remains historical. |

1. Inspect pinned native V2 padding/metadata and existing measured schema/readers.
2. Implement graph-aware physical workloads without changing request geometry; implement exact, scope-qualified measured consumption through existing interfaces.
3. Verify focused positive mathematics and negative scope/phase/TP/routing cases, including real locally accepted compute rows. Preserve old supported behavior.
4. Self-check progress; commit tested changes; run one clean strict77; record receipts and limitations.

Dependencies: 1 -> 2 -> 3 -> 4. Attention GPU qualification is independent and does not block CPU implementation. No formal calibrated E2E claim follows from fixtures or isolated rows.

## Review correction

Approved seams: `read_isolated_scope` and the existing structural execution entry.
For each requested defect, reproduce a CPU red test before its minimal source fix;
keep the known error conversion narrow. Then run affected checks and the accepted
local data consumer, commit one correction, run clean77 against the absolute
baseline and archive receipts. B assertions are recommendation-only; timer audits
and all native/hardware admission limits remain outside this correction.

| Step | Status | Acceptance |
| --- | --- | --- |
| R1 layout | completed | Red counterexamples, complete exact isolated layout and source cache/head agreement; actual accepted CSV negatives pass. |
| R2 errors | completed | Known missing-table/TP/EP/typed-contract errors are typed with original causes; unrelated reader failure remains unchanged. |
| R3 embedding | completed | Actual17/descriptor40 uses40-row embedding all-reduce while actual request statistics remain17. |
| R4 affected verification | completed | 208 PASS/8 SKIP;102 exact real scopes/68 typed negatives and two mixed-layout controls PASS. B assertion recommendation recorded without B edits. |
| R5 commit/strict/archive | in-progress | One source correction commit; its clean strict77 is IDENTICAL against the required absolute baseline; selected receipts and final records archived. |

Dependencies: {R1,R2,R3} -> R4 -> R5. User prohibits delegation; these repairs
share the same predictor/test files. No broad unit sweep or GPU work is needed.

# Plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Established bounded implementation and acceptance sequence. |
| 2026-10-21 | All four steps completed; implementation621b198 and sole strict77 IDENTICAL. |
| 2026-10-21 | Added the approved review-correction frontier; original completion remains historical. |
| 2026-10-21 | Correction282cb11 clean strict77 IDENTICAL77/77; review frontier complete, records-only archive. |
| 2026-10-21 | Reopened only final P2 known unknown-operator coverage conversion. |
| 2026-10-21 | Final P2e3ce166 sole clean77 IDENTICAL77/77;U1–U3 complete and prior receipts preserved. |

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
| R5 commit/strict/archive | completed | One source correction282cb11; its sole clean strict77 is IDENTICAL77/77 against the required absolute baseline; selected receipts and final records archived. |

Dependencies: {R1,R2,R3} -> R4 -> R5. User prohibits delegation; these repairs
share the same predictor/test files. No broad unit sweep or GPU work is needed.

## Final P2 follow-up

1. U1: reproduce two actual shared/MARLIN reader failures at the existing
   `read_isolated_scope` seam, copying complete real metadata under an unknown name.
2. U2: add only the known validator error prefix, verify original causes and the
   existing unexpected-ValueError control, and extend the existing real-data check.
3. U3: freeze/commit source, run its sole clean77 against the required absolute
   baseline, preserve282cb11 receipts exactly and archive the new result separately
   within the existing receipt file. No B/timer/consumption-scope changes.

Dependency: U1 -> U2 -> U3. Status: U1/U2/U3 completed. Final sourcee3ce166's
sole clean77 is IDENTICAL77/77 with clean source/cache; the282cb11 receipt is
unchanged. Only records are archived afterward. No broad test sweep was repeated.

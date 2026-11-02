# W128-V4-ATTENTION-INPUT requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded the delegated private-input scope and approval boundary. |

[Original Request] Implement W128-V4-ATTENTION-INPUT on `feat/v4-attention-input`, baseline `972f0cb1`, after reading the parent design. YC approved CPU implementation under D-V4-KERNEL-ONLY-GPU-COST.

- Selected attention CSV discovers `dsv4_attention.json` when an explicit prefill layout is omitted. Preserve explicit-layout API and typed missing-layout behavior. Invalid present sidecars reject without fallback.
- Derive per-table/per-layer layouts with the existing independent builder from accepted plan and non-timing raw metadata. Bind CSV, plan, raw, summary, model and source lineage.
- Preparation validates the original ACCEPTED_ISOLATED_DATA receipt/package, copies immutable originals to a fresh candidate, and never issues PASS admission. A separate externally reviewed isolated-only admission binds the candidate and evidence. Test admissions are fixtures, not production qualification.
- Own only two core files (180 changed lines) and three named non-core files (350 changed lines target). Move/reuse the accepted preparation in the existing checker. No further delegation.
- Verify all 94 medians by explicit and automatic entry paths, scoped negatives, original affected unit tests and legacy 102/72 checks. Use only this repository's writer; external retained inputs are explicit and SHA validated.
- Run at most one clean fixed-source 77-case candidate matrix with jobs4/clean-cache/continue-on-failure. Root owns strict comparison, actual representation admission and independent review.
- No GPU/platform/SSH/cloud/Docker/native/C26/default data edits, serving totals, coverage promotion, overlap assertions or shared config changes. Native06 R4 remains unchanged.
- Commit verified code and local records; deliver commit, scope, checks, candidate paths and outstanding root gates with a clean worktree.

Parent requirements/design/plan remain root-owned. This record narrows, not expands, their authorization.

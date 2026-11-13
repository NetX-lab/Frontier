# Plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-20 | Completed steps 1–5; source commit 6db2cd9, exact comparisons and parent status notes verified. |
| 2026-10-20 | Created the scoped implementation and CPU verification plan. |

1. Inspect Request/event timestamp causality and public comparison consumers.
2. Add `Request.client_ttft` from the existing first-output timestamp; export PD-only `client_ttft` in milliseconds, preserving historical metrics and non-PD schemas.
3. Test transfer-gap, one-output, co-location, legacy prefill latency, CSV units and actual transfer-before-D-admission ordering.
4. Freeze candidate source; run affected unit tests, structural Step/GLM PD driver cases and legacy 77-case baseline/candidate comparisons against `8585748`.
5. Record exact commands/results and scope limits; commit owned code/docs and update parent progress/ticket notes.

Dependencies: 1 -> 2 -> 3 -> {4.unit, 4.structural, 4.legacy} -> 5. Verification runs may be concurrent; do not edit source while identity-checked drivers run. Acceptance: non-PD outputs bit-identical; PD old values identical with only the explicit client field added; all new values end at the actual first D output and follow transfer completion. Structural dummy timing does not qualify measured H800 accuracy. T2 still needs ticket 10 profiles and representative ticket 24 cells.

## Completion

All five implementation/verification steps are complete. The scoped simulator
endpoint is committed as `6db2cd9`; final commands, counts, initial failures and
resolutions are in the test report. Parent progress and ticket 14/17 notes are
updated. The existing gate is not modified: the raw comparator reports the
approved additive schema difference, while every old output is exactly
verified. Native gate integration, T2 and clean-cache/performance qualification
remain explicitly outside this completion. Selected evidence: 55 files,
1,794,413 bytes, plus a retained SHA-256 manifest, under `evidence/`.

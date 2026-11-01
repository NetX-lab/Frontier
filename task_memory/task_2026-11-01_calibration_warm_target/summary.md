# Calibration warm-target skill summary

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Recorded delivered documentation and pending final review/integration. |

The requested `/data/ycfeng/Frontier/skills/frontier-calibration` documentation now defines the warmed comparison scope in its contract and makes the execution routes consistent with that scope. It preserves every valid repeat and immutable all-request reading, specifies repeat-specific paired warm calculations, distinguishes validity from readiness, and fixes preflight timing, NIXL log-count scope and policy-only build identity.

Deliverables are `skill_warm_fix.diff`, `test_report_2026-11-01_skill_warm_fix.md`, `review.md`, and `check_progress_skill_warm_fix.md` in this directory. Existing structure/skill checks and direct arithmetic checks pass. Independent review closed all six findings with no actionable skill defect and passed four workflow walkthroughs. Root commit `0cc4118a` is integrated into the active GLM/V4 branch as `f31b2b2d`, with prerequisite skill changes and the separately approved README (`24ddbbb6`). All 34 skill/README files match root bytes; integrated structure/skill checks pass. This slice is complete. No runtime calculator, simulator change or native/GPU run belongs to it; the parent task owns the actual P2 warm-reading evidence and remaining input qualification.

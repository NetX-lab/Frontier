# D20 actual main requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record D20-ACTUAL-MAIN scope and initial production blocker. |
| 2026-10-21 | Parent authorizes the focused graph-policy production correction and CLI regression. |

- [Original Request] Task D20-ACTUAL-MAIN: work only in the step5-d20-integration worktree. Execute real `python -m frontier.main` measured Step P16/D16, TP8/DP2/EP16/PP1, using the built collective_sim backend.
- [Original Request] Complete at least one 6144-input/1024-output request in offline and online modes, with first D tail16/cached6128, normal lazy compute and NIXL delegates, final metrics and completed transfers.
- [Original Request] Reuse the existing synthetic compute fixture and the explicit 3.5 ms READ contract from `tests/unit/test_nixl_read_cost.py`. No simulator patch/subclass, native admission, accuracy claim, GPU/auth/network/SSH or new agents.
- [Original Request] Use current vllm_v1 and vllm_load_balancing/v0.28.0, no prefix/chunk/MTP, full_decode_only, MoETP1. The fixture capacity8192 differs from canonical7168 and must be an explicit limitation, never silently changed.
- [Original Request] Own only tests/integration and this actual_main directory. Return exact diagnosed production issues to parent; parent owns source fixes. Commit verified tests/records.
- [Original Request] Follow-up: parent authorizes ownership of shared_prediction_model_manager.py and Step measured mapping/cost files when needed, focused tests, runner and this directory. Diagnose existing normalization, preserve public full_decode_only semantics (pure decode full, prefill none), add a real CLI regression, and continue through full6144/1024 lifecycle verification. No new policy enum, flags, guards or fallbacks. No77-case matrix, GPU/auth/network or new agents. Different-ownership defects return to parent.

The parent-supplied manifest is `/data/ycfeng/tmp/d20_measured_compute_fixture_h80032_parent01/fixture_manifest.json`. This is CPU synthetic source-contract evidence only. Formal128/QPS1/three-repeat/four-metric acceptance remains outside this test.

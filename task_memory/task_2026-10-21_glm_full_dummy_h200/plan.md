# W128-FULL plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Established the producer-only implementation and verification sequence. |

Dependency: source inspection -> native FULL producer -> {CPU contracts,
existing DSA tests, offline launch/schema documentation} -> commit and handoff.
No additional subagent or GPU work is required in this delegated slice.

1. Inspect pinned native dummy dispatch, capture, and replay mechanisms.
2. Reuse reduced-model planning and native DSA module inventory. Instrument
   native whole-model graph capture with persistent external CUDA events.
3. Restage actual-query-one dummy metadata through native `_dummy_run` for each
   requested capture bucket. Only the local graph descriptor substitutes for the
   DP maximum-width decision; do not measure or claim DP communication here.
4. Emit isolated FULL-dummy outputs, never regular-serving rows. Check slot and
   query/sequence padding witnesses; distinguish wall time and kernel-only time.
5. Verify CPU contracts, existing focused tests, and offline CLI. No simulator
   changes means no repeat of the numerical fidelity matrix for this slice.
6. Document runtime qualification requirements and consumer admission scope;
   preserve the current typed miss and commit explicit files.

# Plan — D29-DETACHED-EMISSION-OWNERSHIP

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded the read-only plan; all steps completed in one pass. |

Frozen source `7540414341d5b81a86b18b9562d1506c0ba74285`
(`/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields`).

## Scope

Read-only. No source, test, peer or baseline-worktree edit; no runtime campaign, no
GPU, no network, no allocation, no pyc inventory work.

## Steps

1. Read the existing emission profile and inventories
   (`/data/ycfeng/tmp/d29_emission_immutable_work/baseline/{emission.pstats,inventory.json,result.json}`).
2. Trace ownership and callers in `ep_wave_metrics.py`, `op_trace_utils.py` and the
   `ExecutionTime` getters; confirm which work repeats and which is required.
3. Establish alias/escape facts from the retained probe assertions and the earlier
   ownership checks; distinguish a transferred private subtree from a borrowed
   mutable component.
4. Bound the maximum removable duplicate from measured caller costs (H3 delta as
   the calibration point for `asdict`; aggregate profiled `json.loads`).
5. Write the report (<500 words) with a no-seam verdict or one concrete control,
   plus evidence files and limits.

## Acceptance criteria

- Ranked answer to H1/H2/H3 with file:line references.
- Alias/escape evidence, including that getter detachment does not imply
  independence among emitted events or the ledger.
- A profile-derived maximum-gain bound, marked estimate where unmeasured.
- No production change; records committed alone.

## Verification

- Every cited line re-read from the frozen revision; `git status` of the source
  worktree empty at the end.
- Evidence files regenerated from existing artifacts only
  (`evidence/profile_callers.txt`, `evidence/artifact_facts.json`,
  `evidence/source_refs.md`).

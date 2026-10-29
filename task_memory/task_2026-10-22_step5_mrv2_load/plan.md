# D23-MRV2-LOAD-DIAG Plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Define the bounded attribution loop and handoff. |
| 2026-10-22 | Add authorized reuse implementation and regression sequence. |

1. Inspect contracts and retained actual05 evidence; record hypotheses.
2. Build a seconds-scale profile of actual Step attention loading and role union
   on source-preserving subsets, plus full small KERNEL_ONLY source rows.
3. Inspect warm-cache lookup at the existing manager seam using new scratch copies
   only. Do not fit any missing model or write original lock/cache files.
4. Record counts/times, source-backed full-call multiplicities and attribution
   limits; propose the smallest correction without editing production.
5. Check progress, verify ownership/source preservation, and commit artifacts.

Dependency: 1 -> 2 -> 3 -> 4 -> 5. Runtime optimization and full actual-main
acceptance remain parent-owned. No parallel work or further delegation here.

Passing conditions: unchanged source rows/columns/configuration, reproducible
production-call profile, no full initialization, and no production/input/cache
changes. Timing estimates must be labelled separately from measured wall time.

## D23-MRV2-LOAD-REUSE
1. Inspect consumers and existing mutable same-manager contracts; add a
   repeated P/D load/hash regression at the parent-approved `_fit`/union seam.
   Run RED before production changes; capture unchanged baseline frame/attrs/hash.
2. Implement current-measurement union reuse after per-distinct-source SHA checks,
   sharing identical role validation within a lookup. Reuse the resulting root
   signature hash only for the current frame and hash-relevant fit config.
3. Tie frame release to existing real manager initialization in `finally`.
   Verify normal and exceptional exits; keep partial-manager APIs supported.
4. Run focused contracts: source edits/restoration, shape/fit config edits,
   malformed input, model/geometry/topology/measurement identities, distinct
   paths/nested scopes/order. Compare exact baseline frame/attrs/hashes and
   rerun unchanged fast profile; require zero fit/source preservation.
5. Record evidence, check progress and commit. Parent performs integration and
   full acceptance. No full CLI, larger timeout or runtime-seam changes here.

Dependency: 1 -> 2 -> 3 -> 4 -> 5. The approved seams are the existing
`test_step5_attention_input_union._fit`, union loader and manager `__init__`.

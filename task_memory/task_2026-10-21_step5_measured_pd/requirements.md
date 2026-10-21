# D20-PD: Measured PD execution

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record the approved CPU implementation and exclusive ownership. |

## Original requests and decisions

**[Original Request] D20-PD, 2026-10-21:** Close the current R2 P1 regression, preserve that branch, and implement measured PD in `feat/step5-measured-pd` from `6db2250`. Implement the existing PD events/scheduler path, not a second engine or another diagnostic deliverable. No GPU, authentication, platform, cloud, SSH or push operations.

**[Original Request] CPU approval:** The user approved CPU shared integration of Step measured attention/NIXL consumption, predictor/bundle and client metrics. Native/profile admission and four independent 10% accuracy gates remain unchanged. CPU fixture costs do not establish native qualification.

**[Original Request] Ownership:** Own `frontier/events/kv_cache_transfer*.py`, scheduler PD/NIXL owners, Request/Batch only as needed, and dedicated tests. Do not edit attention registries, training, computation predictors, prediction managers, typed loaders or client metrics/comparators. Tell primary about required interfaces outside that scope.

**[Original Request] Semantics:** Preserve the existing reservation -> READ -> ready -> first decode -> release ownership and all legacy dummy/GLM/DS behavior. Inspect `6cc09dc` before reusing its model-agnostic computed-frontier correction. Preserve Step transfer bytes and emitted-token accounting. Do not inject native future batches/routing. The R1 candidate rule is `16 * floor((prompt - 16) / 16)` transferred tokens and 16..31 recomputed tokens for prompts 128..7681; canonical 6144 -> 6128 + 16 remains pending fresh proof. T2 assessment remains required.

## Acceptance

1. Supported measured costs are exercised through actual PD events and scheduler lifecycle, including correct first-D query/context shape, token counts, capacity and release balances.
2. Wrong/missing profile, geometry or lifecycle coverage fails explicitly; no production analytical/dummy substitution or dependence on an R2 claimed PASS.
3. Labelled CPU contract fixtures test actual owners and meaningful negative controls. Run a small actual `frontier.main` execution when the computation owner's interface is ready.
4. Commit verified steps and report commands, test counts and outstanding native inputs. Primary owns consolidation and the final 77-case fidelity matrix; do not repeat it here.

## References and preserved R2 disposition

- Primary task: `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/{requirements.md,plan.md,tickets/17-single-request-pd-handoff.md}`; D20 and plan section 6 govern this implementation.
- Preserved R2 branch: `feat/step5-r2-correction`, HEAD `b6ba7e5`; P1 code `cb09a27`, 636 passing affected tests, independent follow-up review passed. Historical evidence remains UNVERIFIED. No further R2 features or staging are part of D20-PD.

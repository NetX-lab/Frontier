# 11: Independent native routing and selector statistics

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-22 | YC approved position_cyclic and native/image implementation; retain independent-statistics prerequisites. |
| 2026-10-22 | D29 adds immediate routing-policy alignment and supersedes old no-balancing criteria for current calibration. |
| 2026-10-20 | Changed the native prerequisite to `06.simulation_scope` (decision D2). |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Approved planning ticket created. |

**What to build:** An independently sampled native occupancy, padding, and selector-locality evidence set that can support predictive statistics and an existing D-ROUTING sufficiency finding or amendment without importing holdout-run internals.

**Blocked by:** `06.simulation_scope` (decision D2); `07` validated observer; `15.controlled_structural_path`.

**External prerequisites:** Statistical acceptance uses06.simulation_scope,
validated07 and15.controlled_structural_path. The immediate11.routing_policy
source/configuration check can proceed before07 and supplies D29's prerequisite
to new08/09/23 native generations. Preserve independent sampling and original
run identities. A necessary native implementation/reference change requires
its evidence-backed decision before code or GPU execution.

**Status:** statistics blocked;11.routing_policy has approved semantics and active implementation design under D29.

**D31 status (2026-10-25):** Paused. While both systems use `position_cyclic` routing (D29, D31.3), native routing statistics are not needed.

**Current routing-policy scope:** compare effective Frontier/native assignment,
DP aggregation, EP ownership, integer remainders, padding and independent
reference semantics. Frontier actual06 explicitly uses balanced; locked native
built-ins uniform_random/normal_routing do not establish equivalent balance.
Capture03 remains sealed/unsubmitted and not balanced. YC approved
`position_cyclic`: `(logical_position * top_k + slot) % num_experts`, accepting
same-position concentration while preserving P/D/chunk/reference invariance.
Retain selected-weight semantics and native mask; physical rows do not imply
live expert assignments. Native/Frontier/profile implementation and internal
derived-image publication are approved. Native and Frontier designs determine
the minimal numerical code scope before implementation; parent owns integration
and fresh-generation acceptance. Policy verification is workflow work, not
CPU-gap measurement or independent-statistics acceptance.

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S07.
**Shared owners:** C04; existing D-ROUTING evidence consumer.

## Historical review disposition — 2026-10-18

NOT STARTED. Measure occupancy/locality to explain complete-path costs, retaining independent fit/holdout membership. No forced-balanced router or evaluated-run top-k list may substitute for prediction.

## Acceptance

- [ ] Deliver the independent routing/selector dataset, sampling and partition manifest, measured uncertainty/coverage summary, and D-ROUTING evidence card or reuse-sufficiency finding under §§3, 7, and 9.
- [ ] Bind D29's approved effective balanced policy on both sides; record initialization seed, metadata, precision, backend, placement, source and parameter/buffer identities. Preserve old original-router datasets and require a new decision for native implementation/reference changes.
- [ ] Independently sample across seeds, text-content families, global layers, prefill/decode phases, token shapes, and reachable selector states; measure expert occupancy, useful/padded work, rank imbalance, and selector locality rather than assume uniform distributions.
- [ ] Distinguish routed experts from shared/dense FFN work and logical selected regions from token/page units; account for provider/KV replication and padded physical extent without crediting padding as live tokens.
- [ ] Use validated causal observer records to connect statistics to request/step/rank/group identities and quantify perturbation limits; preserve missing or ambiguous observations explicitly.
- [ ] Freeze independent fitting and validation partitions and record any statistical fit's legal predictive inputs; compared holdout expert histograms, exact top-k lists, future batches, transfer timings, and completions must never become predictive features.
- [ ] Report measured variability and limits of dummy-weight evidence; do not claim checkpoint-quality behavior, production-routing representativeness, or adequacy for unmeasured content/shape classes.
- [ ] Evaluate the existing workload representation with distinguishing evidence; if insufficient, provide native findings, current owner, minimum change, omitted details, experiment, and consumers to the same D-ROUTING decision rather than create a parallel authority.
- [ ] Reject unverified/mismatched routing policy, held-out-feature leakage, wrong source/runtime identities, incomplete rank/request records and unsupported locality extrapolation. Controlled balanced evidence cannot be relabeled production-routing evidence.
- [ ] Any collection/statistics-tool changes satisfy §10's ≥50-scenario fidelity comparisons and preregistered wall-time/memory budgets with over-budget dispositions; ≥32-GPU auxiliary validator errors remain soft and leave affected evidence unverified.

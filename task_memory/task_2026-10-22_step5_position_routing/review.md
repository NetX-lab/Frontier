# Frontier position-routing design review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record six findings and assign disjoint design corrections before scope approval. |

Reviewer: `ses_e9e0793b2ffe5YWSjciM0uRzna`, task
`D29-FRONTIER-DESIGN-REVIEW`. Reviewed artifact: initial `design.md` at parent
`0447681`, requirements/plans, native design and relevant native/Frontier source.
Read-only review; no test, shell, GPU or platform run. The initial 32-file /
1699-line estimate is **not the smallest complete implementation scope**.

| Finding | Required correction | Owner |
| --- | --- | --- |
| P1: optional companions contradict full acceptance (`design.md:587–601,670–679`) | Actual-main fixture, source-local pricing and physical communication are mandatory; remove counts-only completion alternatives. | Core design / parent |
| P1: profiler launcher still uses base image (`profiling/common/step5_job.py:12–28`, `submit_profiling_job.py:23–25,71`) | Propagate the same derived adapter/image/source through staging, submission and worker/row verification; preserve historical plans. | Profile source design |
| P1: added physical column breaks padded-schema equality (`moe_routing_workload.py:593–604`, both trainers) | Share schema recognition across feature builder, trainers and BLOCK_SIZE_M ranges; preserve old schemas. | Core design |
| P2: descriptor cache omits lane identity (`design.md:187–192,301–303`) | Key every cached source-lane field, or cache only lane-independent counts and bind each wave's sources separately. | Core design |
| P2: registry classifies but does not own count behavior (`design.md:169–195,243–249,278–283`) | One public materializer invokes registered behavior; keep old ratio algorithms in their existing owner. | Core design |
| P2: bounds do not enforce empty-scope coverage (`sklearn_execution_time_predictor.py:4915–4979`) | Enforce measured empty/all-padding coverage at existing Step lookup, including an absent width inside bounds; retain paired proofs. | Core design |

## Confirmed necessary pricing boundary

Native `naive_dp_ep.py:181–208,241–245` dispatches physical activations, weights,
IDs and scales. `all2all.py:76–97,128–152,193–209` selects the no-SP DP group and
actual sizes. GateLinear/router execute before this gather. Frontier currently
prices assignment counts (`families.py:540–555`) using EP16 all-to-all
(`moe_operator_times.py:244–272`). A single global physical frame cannot express
both source-local gate/router and gathered expert scope.

Existing communication registry builders and ATTN_DP/allgather/reduce-scatter
support are the first reuse targets. Actual unequal sizes and multiple payloads
must be traced to supported primitives; aggregate/DP averaging is not evidence
of equivalence. Detailed minimal mapping is assigned to the pricing design.

## Scope removed and retained

New-policy PD-AF, MTP and legacy fallback expansion is unnecessary for current
Step P/D acceptance and is to be deferred. Keep all existing ratio behavior.
If a fallback is retained, its real virtual-source guard cannot simply be
removed. Thin forwarding through current Ray/multiprocessing paths is justified
because they share the same API and Ray is the CLI default.

## Correction assignments and acceptance

- `design.md`: `ses_e9e28d5b1ffeSkTePirmacKeiT`, registry/source/cache/schema/exact
  empty coverage and mandatory fixture, with optional unrelated adapters removed.
- `pricing_design.md`: `ses_e9df416bdffeEQClYWozYAYoWL`, exact source-local versus
  gathered pricing, physical payload/sizes and existing collective builders.
- `profile_source_design.md`: `ses_e9df3cc43ffer0nUqjD49IsZd0`, complete derived
  profiling image/source entry and provenance through actual launchers.

Each agent edits only its design artifact. The parent must reconcile interfaces,
deduplicate actual file paths and hunks, verify the full budget, and request one
numerical scope decision before implementation. Native's reviewed 19-file scope
and D30's seven-file proposal remain separate inputs to this union.

No implementation, new-policy timing/kernel count, legacy comparison or G3
acceptance follows from this review. The author's 200 math cases were not
independently rerun by the reviewer.

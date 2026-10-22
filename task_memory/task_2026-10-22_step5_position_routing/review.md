# D29 position-routing reviews

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Assign two foundation implementation findings and supply the exact37-path diff for review completion. |
| 2026-10-23 | Record four native implementation findings and repair assignment; model/scope is approved. |
| 2026-10-23 | Final producer delta has No findings; all design findings closed, parent107-path union awaits model/scope decision. |
| 2026-10-22 | Record source-proof delta review and parent communication-model/minimality findings. |
| 2026-10-22 | Add parent interface findings after revised core and profile-source delivery. |
| 2026-10-22 | Record six findings and assign disjoint design corrections before scope approval. |

## Current implementation review

### CORE/SOURCE and initial PRICING contracts

Reviewer: `ses_e9abd8364ffee5sQKdpiQh3yH4`, D29-FOUNDATION-SOURCE-REVIEW.
Frozen sourcebd97ac5 combines CORE02f061d and PRICING7b86614; native19 is excluded
from this review. Inspected29 CORE/SOURCE and8 PRICING paths, approved designs,
requirements/plan/interface/accounting and parent306-case receipts. Declared
unfinished consumers/schema/model/fixture were not treated as regressions.

| ID | Finding and evidence | Required correction | Owner/status |
| --- | --- | --- | --- |
| F1 P2 | moe_impl.py:306–307 checks the router class/inherited select_experts, but not actual _select_experts; replacing only the implementation bypasses provenance. | Add actual bound _select_experts to existing verifier; preserve module/file/class/base method in a foreign-implementation negative test. | CORE/SOURCE, assigned |
| F2 P2 | collective_plan.py:35–54,92–102 accepts externally mutable sizes/tensors/cost arrays despite frozen dataclasses. | Normalize relevant nested sequences to tuples before validation; original-list mutation must not change plan/cost invariants. | PRICING, assigned |

Both corrections remain within their existing ownership/quotas. No numerical,
native or gate change is approved by these findings. Parent306 checks remain
valid test observations, not acceptance despite known source findings.

The first review did not obtain the exact diff and therefore did not establish
deleted-hunk/baseline/change-attribution completeness. Parent supplies
`repairs/review_evidence/foundation.diff`, changed-file list and diff receipt:
exact1aee0af11cd58691e64603b99eca0ac459a5b350 to
bd97ac520fd0ba796ecd9d1905fdff352e859680,37 paths,158722 bytes,
SHA25625299e3a077042e2d17a38ac6409524472be93f826b4ea2d1c4b1d1ca27eb069.
The same reviewer continues only that missing portion while both authors repair
their owned findings and continue the remaining implementation.

### Native19

Component: native19 at2c4cc9e, compared with1aee0af. Reviewer:
`ses_e9afdb7c4ffejb3hbkjZCeHFKS`, D29-NATIVE-IMPLEMENTATION-REVIEW. Artifacts:
`/data/ycfeng/tmp/d29_native_parent_review01.diff`, native task's native_source.patch,
candidate/snapshots, requirements/design/scope/test reports and pinned native
router/runner/graph/model call paths. Review is read-only; root separately
reproduces418 checks and verifies1017/1582 gross lines.

| ID | Finding and evidence | Required correction | Status |
| --- | --- | --- | --- |
| N1 P1 | T2:208–211 indexes a[1], but native MoERunner calls select_experts by keyword. | Explicit native-compatible signature and keyword-call capture test. | Closed74d5c48 |
| N2 P1 | T2:126–142 tests dummy writes immediately after capture, which has not replayed. | Replay dummy after PW/FULL capture, then assert outputs/counters. | Closed74d5c48 |
| N3 P2 | r2_runtime:116–118 checks class.forward, allowing an instance forward replacement. | Check actual bound function identity/origin and negative instance override. | Closed74d5c48 |
| N4 P2 | r2_runtime:130–155 does not validate quant/kernel classes; object() passes a positive fixture. | Validate actual supported objects/source/nonmonolithic path per layer group. | Closed74d5c48 |

Native owner session `ses_e9b3f65b5ffeMAo4OpbcxD9LIV` repairs all four findings in
74d5c48/6827507, within19-path1582 allocation (actual1269). Parent485 checks pass;
same reviewer closes N1–N4 with no additional actionable finding. Root verifies
exact Git diff/budget and12 backend sources against installed-image hashes.
Native19 source/CPU phase is accepted for integration; runtime qualification
remains open. Reviewer found no
Step formula/mask/scaling-duplication defect in the four-file native patch and
accepted sparse overlay/source descriptions. Thin graph evidence remains only
outer setter→opaque MoE→router, not MRv2 serving inputs/attention/connector or
full backend graph qualification. D30 and original gates are retained.

## Design closure

Reviewer `ses_e9e15d18cffemTkCOdK4P941nU`, D29-FINITE-PRODUCER-DELTA,
reports No findings after checking pricing5.4/5.5/6.Closure and actual callers.
The shared selected-call planner reaches validator and fixtures; actual worker
and eager probe bind the derived source, effective settings, positions/mask and
per-input/layer keys. CORE/probe share one source_frame_tensors implementation.
This closes the two final producer findings in design only. Native graphs,
actual source/count evidence, qualified model parameters and E2E remain unverified.
Parent accepts the sole-writer map and H11 deduction; `scope.json` records the
107-path/9942-line union. YC approved the numerical model and implementation
scope through D29-SCOPE-107-10000; additional collection resource/run approval
remains a readiness decision.

## Historical initial review

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
complete D29 numerical scope decision before its implementation. Native's
reviewed19-file scope is part of that union. D30's complete seven-file correction
is independent and can be decided now; coordinate its overlapping file at actual
integration instead of waiting for D29 design closure.

No implementation, new-policy timing/kernel count, legacy comparison or G3
acceptance follows from this review. The author's 200 math cases were not
independently rerun by the reviewer.

## Parent cross-design findings

The revised core subtotal is 31 paths / 2068 gross lines; the delivered source
binding table has 13 paths / 673 lines. These are still not a complete union.

1. **P1 — Unowned Frontier native-router adapter.** Revised core
   `design.md:522–529` and B2 defer the actual importable adapter to profile-source
   ownership. `profile_source_design.md:65,225–245` and S1–S13 leave that same
   `moe_impl.py` adapter to core; native19 supplies only the vLLM interface.
   Core must own and budget factory/context/select_experts integration while
   source binding owns its provenance checks.
2. **P1 — Specify the actual row/width protocol.** Parent inspected
   `moe_wrapper.py:810–869,872–929`: `_build_profile_result` stores one
   `num_tokens`, and `profile` merges every operation timing into that row.
   Merely adding source/gathered labels cannot bind different source-local and
   gathered physical widths. Core must specify and budget actual row/cell
   emission, current executor/CSV flattening and training admission, preserving
   legacy output and each measured scope's real width.

Assigned to `ses_e9e28d5b1ffeSkTePirmacKeiT` as a focused design follow-up; no
code edits or new tests authorized. The source-binding review is independent.
Its note about isolated Unspecified versus native Simulated routing remains a
kernel-identity reconciliation requirement, not permission to force a backend.

## Profile-source delta review

Reviewer `ses_e9e0793b2ffe5YWSjciM0uRzna` resolves the missing-launcher P1 at
design level. One P2 remains: the actual timed GateLinear, RMSNorm and FP8/BF16
oracle callables are not all covered by the serving 53-pin set. The source owner
must check their bound callable/module origins against generated source receipts
in the existing producer seams, and specify who provides those expected hashes.
Shared source/image facts are common; producer-specific actual import maps differ.
Legacy serialization must omit absent native facts entirely. This correction is
assigned to `ses_e9df3cc43ffer0nUqjD49IsZd0`; no implementation acceptance follows.

## Parent pricing review and assignments

1. **P1 — Numerical-source change needs its own decision.** The proposed exact
   measured-plan lookup inside CollectiveSimCCBackend changes the previously
   retained simulation approach (main requirements lines1162/1182/1232). A
   numerical code budget or unchanged class name does not authorize this switch.
2. **P2 — Smallest-completion claim is not established.** The existing HTSIM
   flow layer supports individual byte sizes, dependency triggers and multiple
   groups in one traffic matrix. Public scalar limitations do not prove a
   faithful simulator extension must be larger than lookup. Native grouping and
   GPU-local work remain unresolved; arbitrary simultaneous/serial lowering is
   not accepted. Parent evidence is in pricing sections1/3.2.
3. **Arithmetic correction.** Parent parsed the 37 table rows: +1802/-185 =1987
   gross lines. Section D sums to +585/-14 =599, not699. The authored summary
   has been corrected without changing any individual file allowance.

Independent pricing review `ses_e9e15d18cffemTkCOdK4P941nU` owns native compute
scope/phase/padding and required integration checks. Parent owns communication
architecture comparison. Code/GPU correctness and a final combined budget remain
unverified; ordinary file reads and budget checks do not establish either.

### Pricing review findings and correction assignment

The reviewer confirms the source/gathered split, physical dummy buffers, native
inclusive movement ownership and routed TP reduction after combine. Four P2
interfaces remain open:

1. Core's four-target CSV/coverage rule conflicts with new precision-specific
   prepare/inclusive-expert scopes. Use a registry-selected target set; preserve
   legacy four-target rules.
2. `STEP5_FFN_FAMILY` is bound to SHARED, while the profiler requires ROUTED.
   Define the real routed quantization contract and applicable enumeration.
3. Actual metrics/trace consumers need new names and correct frames:
   `OperationMetrics`, `metrics/ep_wave_metrics.py`, `metrics/op_trace_utils.py`
   and `attention/step5.py`; verify the enabled reporting paths.
4. Collector return-output `reduce_scatterv` is not automatically equivalent to
   native preallocated `combine_into_output`. Resolve the reached registration,
   alias and staging-copy branch before admitting matching numerical evidence.

Assigned back to `ses_e9df416bdffeEQClYWozYAYoWL`, together with a concrete
existing-simulator alternative. No code edit or new GPU run is authorized. Its
new comparison replaces the parent's initial communication exploration. The
reviewer's repeated 2087 arithmetic claim does not supersede the parent's
executed 1987 check; the correction applies to the received table only, and the
revised complete table must be checked again.

## Source emitter closure

The source delta reviewer closed actual callable checks but correctly found no
concrete full-inventory emitter in either old exporter. Parent supplied the exact
post-build command in `profile_source_design.md` and linked it from native design.
It has five explicit inputs, uses the finalized local image without source
mounts, hashes all installed Python sources, and exclusively writes an external
receipt. Its51 Python lines include29 nested container lines. Parent syntax check
passes without Docker or command execution.

Reviewer `ses_e9e0793b2ffe5YWSjciM0uRzna`, task
`D29-PROFILE-EMITTER-DELTA`, confirms correct Docker/Python argv and stdin,
origin/root confinement, externally supplied image identity, no self-digest
cycle, schema agreement and explicit ownership. **Emitter P2 resolved at design
level; no new actionable findings.** Actual image/export, source inventory and
serving-pin subset checks remain unverified. No maintained exporter hunk is
hidden by this operational command; future code changes would need scope updates.

## Core delta review and remaining correction

Reviewer `ses_e9e0793b2ffe5YWSjciM0uRzna`, task
`D29-CORE-DESIGN-DELTA`, finds three concrete seams still needing design text:

1. Predictor constructor `sklearn_moe_execution_time_predictor.py:519–525`
   rejects the new policy through an old four-name set; resolve the existing
   registry there and include it in A7.
2. Trainer pre-loader TP/EP/gating helpers assume MOE_FAMILY. New prepare
   callers must consume returned binding parallel/precision/context facts;
   preserve legacy helpers, without a second generic family registry.
3. Count replay must locally switch the existing TimerStatsStore singleton to
   RECORD_FUNCTION, clean stats and restore its previous method even on error.
   C4 must exercise actual annotations and paired counts, not stubbed mappings.

Earlier six findings, adapter ownership and row protocol are substantially
closed at design level. Registry count behavior, snapshots, lane-aware cache,
schema/ranges, target scope and removal of duplicate expert-adapter allowances
are closed. Exact coverage is restricted to applicable new-policy keys; actual
census coverage is still required. Core33-file subtotal is not final union
approval. Assign these three explicit corrections to the original core author;
pricing's missing modeling inputs stay separate.

## Communication-model prerequisite review

Reviewer `ses_e9e15d18cffemTkCOdK4P941nU`, task
`D29-CC-DESIGN-SCOPE-REVIEW`, closes the four prior pricing interface findings
at design level: applicable target contract, MOE_FAMILY/ROUTED EP-agnostic
prepare, reporting/frame ownership, and actual ordinary/preallocated combine.

One P2 remains: `pricing_design.md:222–260,822–827` incorrectly makes a lossless
NCCL channel/fusion/chunk schedule mandatory before completing a simulator
design. Main requirements demand retained collective_sim and four independent
E2E errors<=10%, not exact NCCL microarchitecture. Existing algorithm/channel
configuration, FlowDef/triggers and sum/max combination are model abstractions.

Correction: keep the exact workload plan (physical/live rows, tensor dtype/bytes,
size-vector/root/group order, all groups, phase boundaries); define a finite
scheduling/protocol/local-work/overlap model and label its assumptions. Parent
and YC decide material alternatives. Native counts/scopes/transport/duration
and E2E qualification follow implementation. Add channel/timed-local machinery
only if error evidence establishes its need. Zero intra-group NVLink traffic
does not establish zero GPU-local work, and primitive counts are not launches.

The50-path/2782 known-interface subtotal is not a complete implementation budget;
measured lookup remains unapproved. Assign a bounded model proposal and complete
scope to the pricing author. This review is documentary, not runtime acceptance.

## Core three-seam closure

Reviewer `ses_e9e0793b2ffe5YWSjciM0uRzna`, task
`D29-CORE-THREE-SEAM-DELTA`, closes all three findings with no new actionable
delta: constructor registry resolution before parent initialization; selected
binding facts before trainer/preflight legacy helpers; actual singleton timer
switch/restore/clear and real annotation/parser/count checks. No common timer,
tracer or singleton module change is required. Parent independently verifies
33 paths/+2625/-341=2966. These are design conclusions, not implementation tests.

Parent also updates source-table overlap annotations to the final core budget:
S8+core is140/31; S12+core225/0; S14+core113/18. S16's22/1 provenance allowance
is additive to pricing, not the retired core expert adapter. Source subtotal
remains16/+835/-44=879, with three shared core paths. Complete union still
awaits the finite pricing model.

## Finite model review and producer closure

Reviewer `ses_e9e15d18cffemTkCOdK4P941nU`, task
`D29-FINITE-MODEL-REVIEW`, accepts the finite lowering/accounting/fit as
decision-ready assumptions, including preservation of negative residuals and
independent qualification. This is not numerical approval. One P2 remains:
`validate_result.py:25–38` rebuilds unfiltered expected native calls although
prepare/benchmark select a subset; worker262–269 consumes that verdict. Extend
the same planner and reuse `test_step5_comm_transport.fixture_result` for subset
and missing-selected-call controls.

Parent finds a second concrete producer seam: worker87–100/248 still calls
`step5_dispatch_probe`, whose446–456 directly invokes Step5MoE under a plain
ForwardContext without logical positions or padding mask. Native T2's earlier
review already excluded using this unchanged probe as position-cyclic evidence.
The new finite collection needs explicit routing/source/position-mask binding
for this actual worker path, or it cannot claim the same derived runtime. The
pricing owner is completing the necessary bounded interface scope; no model
variant, extra launcher or copied graph suite was requested.

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-27 | Added and closed SCOPE-042 payload admission ordering gates for dispatch/combine and migrated legacy raw-width fixtures to typed lanes. |
| 2026-08-27 | Reconciled SCOPE-041 active-role gates with the fresh 309-test matrix and formal Simulator reachability audit; retained heterogeneous aggregate reuse as a future boundary. |
| 2026-08-27 | Added active-role topology propagation gates for the inherited MoE admission chain (SCOPE-041 residual). |
| 2026-08-27 | Closed SCOPE-040b role-capability and SCOPE-041 topology gates with fresh evidence; final static audit remains. |
| 2026-08-27 | Added strict descriptor/predictor EP and router top-k consistency gates (SCOPE-041). |
| 2026-08-27 | Added SCOPE-040b PD/PD-AF role-capability and unavailable-role fail-fast gates. |
| 2026-08-27 | Added SCOPE-040 stable routing-map attribute and constructor RED/GREEN gates. |
| 2026-08-27 | Closed SCOPE-039 dummy attention-only gates with `234` focused passes and the shared DECODE_FFN role guard. |
| 2026-08-27 | Recorded SCOPE-039 RED evidence: both shared-domain roles returned `50.0 ms` post-attention for `include_ffn=False`. |
| 2026-08-27 | Added SCOPE-039 dummy attention-only RED/GREEN and role-preservation gates. |
| 2026-08-27 | Closed SCOPE-036/037/038 gates with fresh matrices and added the shared-collective direct-case evidence. |
| 2026-08-27 | Added lane-local versus aggregate conservation gates and the source-batch explicit-lane regression boundary. |
| 2026-08-27 | Added direction-A aggregate classification, capability, and zero-lookup gates after the PDD RCA. |
| 2026-08-27 | Closed the token-ledger/admission verification gates and recorded the descriptor-context topology residual watch. |
| 2026-08-26 | Resolved the EP>1 aggregate admission gate with narrow A' and added early lookup-order acceptance criteria. |
| 2026-08-26 | Added raw-width conservation acceptance and the EP>1 aggregate dummy-admission decision gate. |
| 2026-08-26 | Added the dummy terminal-MTP physical-lane gate, its mode-specific RCA, and the final focused-verification/commit/review checklist. |
| 2026-08-26 | Added the PDD attention-only identity gate and recorded the aggregate all-to-all/structural-MTP audit evidence. |
| 2026-08-26 | Added the global-layer identity propagation gate and recorded its RED/GREEN contract. |
| 2026-08-26 | Closed the terminal MTP EP>1 hook gates with `178` focused tests and numeric phase/layer evidence. |
| 2026-08-26 | Added the terminal MTP overshoot EP>1 descriptor RED/GREEN gates and numeric evidence requirements. |
| 2026-08-26 | Closed the typed-trace gate: descriptor-only workload input, output-only map projection, and `168`-test evidence. |
| 2026-08-26 | Recorded the implementation handoff audit and the remaining typed-lane verification boundaries. |
| 2026-08-26 | Recorded the closed MONOLITHIC initial-decode frontier gate and its numeric probe contract. |
| 2026-08-26 | Added the canonical token-ledger gates and recorded approval of the shared compute-contract and structural MTP shape repairs. |
| 2026-08-25 | Split the physical MoE MTP lane gate from the unresolved generic MTP shape-adapter decision. |
| 2026-08-25 | Recorded the explicit scalar compatibility boundary and aggregate-domain validation gate. |
| 2026-08-25 | Added A' typed-lane, scheduler-identity, zero-lane, communication, and pure-MTP gates for the post-PR17 integration repair. |
| 2026-08-22 | Added the Option-B warning/coverage gates for standard-attention explicit KV filtering and archived the final PR #21 review status. |
| 2026-08-22 | Advanced the PR #21 completion gates with final synthetic-stack, routing-environment, memory-accounting, and static-hygiene evidence; recorded APPROVE/CLEAR review closure and push authorization. |
| 2026-08-22 | Reopened the exact-row and MTP TP-consumer completion gates after the resumed Claude review confirmed two P1 defects. |
| 2026-08-22 | Added and closed the universal exact-row default-persistence gate for PR #20. |
| 2026-08-21 | Closed the scoped implementation/verification gates with 111- and 265-test matrices plus direct PDD evidence; recorded deferred data gates. |
| 2026-08-21 | Added the accepted option A EP-domain gate and its verification requirements. |
| 2026-08-21 | Created task-specific gates for the analysis-only phase and the later implementation phase. |
| 2026-08-21 | Added staged dependency-boundary gates and removed the assumption that a separate adapter is required. |
| 2026-08-21 | Marked the staged boundary and `operator_query_binding` seam as maintainer-confirmed; added the final file-map gate. |
| 2026-08-21 | Added function-level ownership and seam-depth gates after the final read-only source audit. |

# Harness - Scoped PR Gates

## Current-phase gates

- [x] Work occurs in `.worktrees/pr4-scoped-lookup-boundary-20260821` on a
  dedicated branch based on `origin/main`.
- [x] The worktree is clean before documentation changes.
- [x] Focused baseline tests pass: `37 passed`.
- [x] Legacy PR #4 behavior and current-main residual sites have file/line
  evidence.
- [x] New task-memory documents link to the old review archive.
- [x] Maintainer accepts the staged dependency-boundary interpretation.
- [x] Maintainer selects `operator_query_binding` / `bind_operator_query` and
  rejects an independent profiling-facing adapter by default.
- [x] Function-level lookup, TP-classification, and sampling call sites are
  recorded in `findings.md` and `plan.md`.
- [x] Maintainer confirms the implementation file map and acceptance gates.
- [x] A1 production/test changes stay inside the confirmed file map and were
  started only after the maintainer's continuation instruction.

## Later implementation gates

### Lookup and cache

- Validate identity/schema/physical/relational constraints before all cache and
  model access.
- Exact measured value precedes process-local runtime cache, which precedes the
  canonical model.
- A legal miss calls the model once; a repeated identical query does not.
- Invalid identity, bounds, schema, NaN, infinity, or negative values fail before
  model calls and cache writes.
- No nearest-row, clamp, default, silent fallback, or CSV mutation.
- Runtime cache keys include all timing determinants but exclude run paths,
  timestamps, publication metadata, and other non-performance fields.
- A1 treats bounds, relational constraints, and identity as optional metadata
  until current-main producers emit and persist those fields; tests must not
  present them as a complete producer-side contract.
- Both unified runtime training/cache entry points persist scalar-numeric exact
  measured rows by default, while preserving an explicit opt-out for a future
  unsupported key schema.

### Boundary

- Supported runtime modules do not invoke profiling benchmark runners, GPU
  wrappers, or profiling CLI entry points.
- Any remaining CPU/PP CSV schema/validation import is side-effect-free,
  explicitly allowlisted, and shared rather than duplicated.
- Runtime cache miss can locate current raw CSV, fit the configured estimator,
  and serialize the runtime-owned model without calling a benchmark generator.
- The non-KV-cache-memory and MTP structural-config exceptions are explicit and
  separately tested.

### A' typed-lane gates

- [ ] `LayerEPWorkload` is the sole aggregate routing/workload owner.
- [ ] Exactly one canonical seam constructs `EPLaneWorkload`; no caller-local
  descriptor constructor or duplicate mutable map exists.
- [ ] The descriptor contains physical topology and routing data only:
  `ep_id`, EP size, total/local expert width, owned IDs, fixed-width local
  counts, routed token count, and router top-k.
- [ ] Scheduler lifecycle identity (`schedule_epoch`, AFD stage index,
  admission ticket, stale-wave identity, metrics operation identity, and exact
  stage/KV provenance) remains owned by scheduler/stage context.
- [ ] Sparse maps are normalized to fixed topology width at construction;
  `len(map)` never determines profiling width and no
  `validate_expert_width=False` escape hatch remains.
- [ ] EP=1 and EP>1 use the same descriptor/predictor interface; only the
  physical collective cost degenerates for EP=1.
- [ ] Every physical lane, including zero-routed lanes, remains in the
  participant/barrier set. Zero routed-dependent phases return `0.0` without
  positive-load model lookup.
- [ ] `EPBatchGroupPlan` and `EPBatchGroup` carry the descriptor and expose any
  raw map only as a read-only compatibility projection.
- [ ] Predictor base/concrete/disaggregation/mocks/tests share one typed
  contract. Predictor code does not re-split global maps or infer topology.
- [ ] The only retained scalar path is the documented standard one-feature or
  shared-domain pre-routing path; load-aware and physical EP consumers require
  `EPLaneWorkload`, and raw/partial maps fail before model access.
- [ ] Communication payload consumers read the typed projection instead of
  using entity type as the domain discriminator.
- [ ] Physical MoE MTP phase timing uses a pure `EPLaneWorkload` seam and never
  constructs a synthetic `EPBatchGroup` or copies scheduler lifecycle identity.
- [x] The generic target-embedded MTP shape replay remains the approved narrow,
  scheduler-independent adapter from SCOPE-026; no shared predictor-interface
  expansion is introduced by the token-ledger repair.
- [ ] Dense `predict_stage_execution_time()` callers retain their existing
  signature; lane context is obtained from the typed batch/entity path.

### Terminal MTP overshoot gates

- [x] The RED regression reaches the real terminal metadata builder and fails
  only because EP>1 MoE terminal replay lacks an `EPLaneWorkload` descriptor.
- [x] The default terminal-row hook preserves dense/EP=1 behavior and the MoE
  override is selected only for an actual MoE layer on the supported MTP
  cluster path.
- [x] The MoE hook executes one shared attention-only prediction and one typed
  lane phase call per physical participant, including zero-routed lanes.
- [x] The five lane phases use the existing `predict_moe_lane_phase_times()`
  decomposition and lane-wise `max()`; no raw map, fabricated descriptor, or
  synthetic `EPBatchGroup` enters the predictor.
- [x] `stage_id`/`pipeline_stage`, `cluster_type`, `layer_id`, and `num_layers`
  are passed explicitly. Per-layer physical phases scale with `num_layers`,
  while pipeline and CPU overhead remain single batch-level terms.
- [x] Numeric evidence records terminal row count, verification width, each
  lane ID/local width/routed count, zero-lane model calls, phase values,
  phase-wise maxima, total time, and stage/layer identity.
- [x] Global `layer_id` remains distinct from `pipeline_stage`/`stage_id` in
  the public predictor, internal execution-time probe, attention probe, and
  terminal MTP hook; no layer identity is derived from stage identity.
- [x] The internal `layer_id=0` default is documented as compatibility for
  callers without a real global layer identity, while callers that have one
  pass it explicitly.

### PDD identity and boundary audit gates

- [x] PDD `include_ffn=False` attention-only prediction forwards the explicit
  global `layer_id` into its attention profiling lookup.
- [x] The helper keeps a compatibility default only for direct callers that
  lack a global identity and never derives a layer from `stage_id`.
- The RED/GREEN matrix records `0 -> 17` and `64` passing tests.
- [x] EP>1 routed aggregate all-to-all dispatch/combine fails at the existing
  payload boundary without `EPLaneWorkload`; no duplicate caller-side guard is
  added.
- [x] Structural MTP local-config coverage is recorded with loaded layer
  counts and explicit missing-path failures; no guessed JSON is introduced.

### MTP/MoE token-ledger gates

- [x] For ordinary target-embedded MTP, `Batch.num_tokens` represents the full
  verification width and the shared compute helper does not add planned draft
  metadata a second time.
- [x] Structural target verification uses the complete `verify_tokens` shape;
  rejected drafts affect only post-forward outcome/progression accounting.
- [x] Gating and all pre-routing shared compute use the canonical physical
  token count, while `LayerEPWorkload` materialization uses
  `Batch.total_num_tokens`.
- [x] `total_routed_assignments == routing_token_count * router_topk`, and the
  sum of typed lane routed counts equals that assignment count.
- [x] Routed-dependent zero-lane phases return `0.0` without positive-load
  model lookup, while physical participant/barrier membership remains intact.
- [x] Explicit AFD/CUDA Graph compute padding retains precedence over the
  ordinary MTP width rule; transfer sizing keeps its raw physical payload
  contract.
- [x] The MONOLITHIC initial decode `max(planned_drafts, 1)` path is directly
  audited: with `num_prefill_tokens=8` and the boundary state
  `num_processed_tokens=9`, the scheduler frontier is `8` and planned values
  `[0,1,2,4]` produce next widths `[1,1,2,4]`; the ordinary full verification
  width resumes after the frontier advances. The rule remains in the existing
  scheduler seam and is covered by a focused regression.

### Registry

- Unknown names fail at exact admission.
- The admission API is named `bind_operator_query`; exactness is a tested
  behavior, not a second module or catalog.
- No family/TP/dataset decision uses a name prefix, substring, split, or local
  literal set.
- Explicit aliases and many-to-one physical mappings have collision checks.
- The existing unified registry remains the only ordinary operator catalog; an
  independent profiling-facing adapter is not added without a concrete owner
  mismatch.
- Ordinary existing-owner operator extension requires one declaration.

### Plan handoff

- The first implementation edit is limited to the candidate file map in
  `plan.md`.
- A new `operator_query_binding.py` file requires evidence that the existing
  `binding.py` is overloaded or creates an import cycle.
- Any file outside the map requires a root-cause note and renewed scope review.

### Profiling domain

- `Profiling Domain >= Runtime Domain` for every tested family and axis.
- MoE default EP sampling includes every positive divisor of `num_experts`; an
  explicit CLI EP list passes the same divisibility validation, and the
  resolved per-model domain is recorded before measurement.
- Sampling is deterministic and bounded by physical capacity.
- Automatic standard-attention decode sampling stays at or below
  `max_seq_len - 1`. An explicit standard-attention decode KV value may reach
  `max_model_len - 1` after reserving the current token.
- Standard-attention physical filtering runs independently for every selected
  `(model, tensor_parallel_size)` target. Every dropped explicit row emits a
  `RuntimeWarning` before worker dispatch and the parent process prints the
  requested/retained union plus target capacities.
- The standard-attention planner raises `ValueError` only when a requested
  explicit KV value is dropped by every selected target. Values outside the
  runtime boundary remain fail-fast.
- Coupled/derived features come from real executable inputs.
- Active simulation never measures, trains, or writes profiling CSVs.

### Verification and reporting

- Direct case-driven runs close the evidence chain with numeric values and model
  call/cache-hit counts.
- Persistent tests are narrow and justified by regression risk.
- Reports live under this task directory and record command, interpreter,
  environment, acceptance criteria, and observed values.
- Warning checks record the exact discarded values, dropped combination count,
  target identity, physical capacity, retained union, and the all-target
  fail-fast message.
- No `README.md`, canonical data, remote PR state, or main checkout changes.

### 2026-08-26 continuation gates

- [x] Dummy terminal-MTP lane phases use the typed descriptor in both modes;
  dummy mode reuses `_get_dummy_execution_time()` and non-dummy mode retains
  `_get_execution_time_internal()`.
- [x] Zero-routed dummy lanes keep dispatch/combine and shared pre-routing
  work, return routed shuffling/grouped-GEMM `0.0`, and perform no positive-load
  model lookup.
- [x] MoE TP all-reduce comments and registry semantics agree on the shared
  source-batch pre-routing width; only EP all-to-all is lane-local.
- [x] Aggregate materialization routed assignment conservation uses
  `Batch.total_num_tokens * router_topk`; compute-effective width remains
  reserved for compute/gating lookup features.
- [x] The predictor checks `batch.total_num_tokens ==
  lane_workload.routed_token_count` only for a batch-attached lane entity.
  A source `Batch` plus an explicit lane is a lane subset and receives no
  predictor-derived `source_width * router_topk` expectation.
- [x] Descriptor/predictor topology equality is enforced by the shared
  routed-admission helper; mismatches fail before mode-specific work and no
  duplicate caller-side validator exists.
- [x] Narrow A' is the public EP>1 aggregate admission contract: every routed
  MoE call, including an implicit multi-layer aggregate, requires
  `EPLaneWorkload` in dummy and non-dummy modes before dummy return,
  measurement activation, attention/model lookup, or backend lookup.
- [x] The admission helper runs only after each concrete predictor has resolved
  its existing routed/dense/attention-only classification. Implicit aggregates
  use the model-level `is_moe` flag; concrete single-layer calls use
  `is_moe_layer(layer_id)` or an explicit selector, and a missing required
  predicate fails at the public boundary. The helper does not reinterpret
  `DECODE_ATTN` or stage identity.
- [x] Attention-only, dense, EP=1, valid typed-lane, and zero-routed-lane
  calls retain their existing behavior; the communication payload builder
  remains the final structural descriptor check.
- [x] Focused A' matrix evidence is recorded as `350 passed in 10.68s`, with
  direct dummy totals `27.0/33.0/33.0 ms` for generic/Step2Mini/Step3.
- [ ] Re-run the complete focused command, compile/whitespace checks, and
  documentation gates at the final dirty state.
- [ ] Stage and commit production, tests, and task docs as coherent
  sub-steps, then obtain an independent code review.
- [ ] Complete the read-only PR20/PR21 merge-readiness audit. Keep all remote
  PR state unchanged until a separate authorization gate.

### Routed aggregate correction gates

- [x] PDD `EP=2`, `num_layers>1`, `include_moe=None`, and no lane fails with
  `ValueError` before `select`, `require`, `activate`, communication,
  overhead, attention, model, or backend lookup in both execution modes.
- [x] MONOLITHIC and PDD implicit aggregates use the same routed/dense
  classification in dummy and non-dummy paths; no downstream resolver is the
  first failure site.
- [x] A concrete model-level MoE call without callable `is_moe_layer` fails
  explicitly before timing/lookup unless `include_moe` is supplied.

### Lane-local conservation correction gates

- [x] `LayerEPWorkload` remains the sole owner of aggregate assignment
  conservation and materializes every physical lane, including zero lanes.
- [x] Attached lane entities reject a local-width mismatch before any model or
  communication lookup; zero is valid when both entity and descriptor widths
  are zero.
- [x] Source-batch plus explicit partial/zero lane calls pass without a guessed
  lane expected count, including the terminal MTP lane subset path.
- [x] The focused report records source width, lane routed count, aggregate
  assignment count, downstream lookup count, and barrier participants.

### Disaggregation dummy attention-only gates (SCOPE-039)

- [x] The RED regression reaches the public disaggregation dummy path and fails
  because `_get_dummy_execution_time_for_cluster()` has no `include_ffn`
  selector; the failure records non-zero FFN fields rather than a fixture or
  import error (`PREFILL` and unified `DECODE`: `50.0 ms` post-attention).
- [x] The public `include_ffn=False` value is forwarded through the existing
  dummy owner without changing the predictor signature or adding a second
  timing owner.
- [x] Shared-domain `PREFILL` and unified `DECODE` dummy results retain
  attention and batch-level overhead while setting all FFN/MoE fields,
  post-attention time, and FFN communication components to zero.
- [x] `DECODE_ATTN` retains its established post-attention layernorm/residual
  behavior; `DECODE_FFN` remains an invalid attention-only public role.
- [x] Dense/MoE classification, typed EP admission, zero-routed lanes,
  non-dummy lookup order, and the affected predictor matrix remain green.
- [x] Fresh RED/GREEN numeric evidence and the final SCOPE-039 status are
  recorded in `issues.md`, `progress.md`, `review.md`, and the test report.

### Aggregate routing-map constructor gates (SCOPE-040)

- [x] The `cluster_type=None` constructor deletion failure is reproduced with
  a deterministic probe and recorded before the production edit.
- [x] The RED regression isolates the repeated `del` operation rather than a
  model, routing, or environment failure.
- [x] All three routing attributes remain present and independently readable
  for aggregate, explicit-role, and `DECODE_ATTN` instances.
- [x] Role maps populate only for materialized MoE roles; no routing data is
  synthesized for attention-only roles.
- [x] Existing disaggregation/typed-EP matrices, compile, whitespace, and
  conflict-marker checks remain green after the correction.
- [x] Fresh RED/GREEN numeric evidence and the final SCOPE-040 status are
  recorded in `issues.md`, `progress.md`, `review.md`, and the test report.

### Aggregate role-capability gates (SCOPE-040b)

- [x] A legal PD-shaped aggregate materializes only PREFILL and unified
  DECODE; it never dereferences or synthesizes DECODE_FFN.
- [x] A legal PD-AF-shaped aggregate materializes PREFILL and DECODE_FFN while
  leaving DECODE_ATTN outside routed maps.
- [x] The role-to-config-attribute mapping is the single declaration-driven
  capability source; no mode-string branch or duplicate validator is added.
- [x] An explicitly requested unavailable role preserves the existing
  fail-fast error and performs zero downstream model/routing lookup.
- [x] Every constructor instance exposes all three stable routing attributes,
  with `None` for non-materialized roles and no timing/scheduler change.
- [x] Fresh RED/GREEN numeric evidence, focused matrices, compile, whitespace,
  and conflict-marker checks are recorded before SCOPE-040b closes.

### Descriptor/predictor topology gates (SCOPE-041)

- [x] The active predictor EP size and router top-k are read from the existing
  predictor/role topology owner; the lane descriptor remains scheduler-built.
- [x] A routed descriptor with EP `4` against predictor EP `2` fails with a
  topology-specific `ValueError` before any dummy, measurement, model,
  backend, or communication lookup.
- [x] A routed descriptor with top-k `1` against predictor top-k `2` fails at
  the same boundary and produces downstream lookup count `0`.
- [x] Dummy and non-dummy MONOLITHIC/PDD entry points share this ordering;
  valid EP=1, valid EP>1, partial source-batch subsets, and zero-routed lanes
  retain their existing behavior.
- [x] Aggregate conservation remains owned by `LayerEPWorkload`, physical
  lane construction remains scheduler/materializer-owned, and the payload
  builder remains the final structural descriptor check.
- [ ] Fresh final matrix, compile, whitespace, conflict-marker, and static
  owner audits are recorded in the task report before SCOPE-041 closes.

### Active-role topology propagation gates (SCOPE-041)

- [x] The non-dummy aggregate disaggregation regression demonstrates that the
  first admission uses active `DECODE_FFN` `EP=2/top-k=2`, while the pre-fix
  second inherited admission incorrectly reads representative top-k `1`.
- [x] The MoE API carries optional active `EP/top-k` context without changing
  direct MONOLITHIC/default call behavior or introducing mutable predictor
  state.
- [x] Every disaggregation routed role forwards the same active context to the
  inherited method, and every admission within that method uses it.
- [x] A valid active-role descriptor passes both admission points in non-dummy
  mode; true EP or
  top-k mismatches still fail before timing, measurement, model, backend, or
  communication lookup.
- [x] The focused RED/GREEN evidence records descriptor, representative,
  active-role values and downstream call counts before SCOPE-041 is re-closed.

The formal `Simulator` constructs one predictor per disaggregation role, so
downstream phase/gating/communication helpers read role-local topology from
their own predictor instance. A manually reused `cluster_type=None` aggregate
predictor executing a heterogeneous role has no supported downstream context
contract and remains outside this gate; adding such a contract requires a
separate immutable execution-context design.

### EP payload admission ordering gates (SCOPE-042)

- [x] `_validate_ep_barrier_arrival()` remains the scheduler-owned identity and
  waiting-room validator; no physical descriptor check is duplicated there.
- [x] Dispatch and combine invoke the existing payload owner on the complete
  prospective lane set before architecture collective resolution,
  communication predictor/backend lookup, trace publication, or final-lane
  commit.
- [x] Missing descriptors and entity-width mismatches fail with the typed
  `ValueError` on both entrances, and `predict_alltoall_time` remains uncalled.
- [x] A prior valid lane may remain in the waiting room when the final malformed
  lane fails; this option-1 behavior is explicit and transactional.
- [x] Successful and predictor-error fixtures carry valid `EPLaneWorkload`
  descriptors; raw-width fixtures remain only where identity rejection is the
  behavior under test.
- [x] The dispatch/combine payload and PD-AF invariant matrix passes `392` tests
  with `19` explicit skips.

## Scoped completion checkpoint

- [x] Wave A1/A2 lookup precedence, validation, finite consumers, and grouped-
  GEMM in-flight token preservation are verified.
- [x] Wave B staged import boundary and runtime-owned MoE feature extraction are
  verified with explicit shared-helper and named-exception allowlists.
- [x] Wave C ordinary unified-registry admission and the three target-embedded
  MTP TP consumers are verified through the existing MTP registry accessor and
  extension regression on PR #20.
- [x] Wave D primitive/mixed-Attention envelopes, MoE routing/load, all
  runtime-legal EP divisor resolution, and decode memory accounting are
  verified at the code-side domain; the repaired PR #21 synthetic stack passes
  `322` tests with `2` known
  logger cases deselected.
- [x] Standard-attention Option-B filtering emits target-local discard warnings,
  prints retained coverage, and fails only when every target drops a requested
  explicit KV value.
- [x] Wave E direct probe passes with model-call/cache evidence; the PR #21
  targeted sampling/EP matrix passes `33`, and the separate MoE routing/load
  matrix passes `13`.
- [x] Sequential PDD direct case completes `2/2` requests, processes `24`
  tokens, and records `2` KV transfers totaling `8,388,608` bytes.
- [x] `SCOPE-014` default exact-row persistence survives both unified cache
  round trips, including both one-feature MoE producers; measured `4.0` beats
  stale `99.0` after cache reload.
- [x] The resumed Claude Opus 5 review completed with `REQUEST CHANGES`, and
  independent probes reproduced both P1 findings.
- [x] Maintainer selected the focused MTP TP-consumer repair; the broader
  unified-family/interface audit remains in `future.md`.
- [ ] `SCOPE-011` producer-side persistence for optional bounds/domain/
  constraint/identity metadata remains open.
- [ ] Canonical CSV regeneration for newly exposed EP values remains an
  offline profiling/data-publication follow-up.

### PR #21 delivery gate

- [x] PR #20 base remains frozen at `18d1a23e` and unmerged.
- [x] PR #21 code tip is `8cc267ea`; the prior synthetic stack remains
  conflict-free through tree `dc808182`.
- [x] Local broad/targeted/routing verification, numeric probes, compile, diff,
  and path audits are recorded in the PR #21 test report.
- [x] Push the five local PR #21 commits and update the remote PR body after
  explicit maintainer authorization.
- [x] Keep PR #20 and PR #21 open and unmerged after the authorized push.

## Prohibited changes

- Blanket fallback handlers or calibration factors used to hide lookup misses.
- A new Contract/Data Guarantee abstraction layer.
- N3/N4/N5 profiling database/publication/provenance work.
- Bulk refactors unrelated to lookup, boundary, registry admission, or domain
  coverage.

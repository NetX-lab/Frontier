## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
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

### MTP/MoE token-ledger gates

- [ ] For ordinary target-embedded MTP, `Batch.num_tokens` represents the full
  verification width and the shared compute helper does not add planned draft
  metadata a second time.
- [ ] Structural target verification uses the complete `verify_tokens` shape;
  rejected drafts affect only post-forward outcome/progression accounting.
- [ ] Gating and all pre-routing shared compute use the canonical physical
  token count, while `LayerEPWorkload` materialization uses
  `Batch.total_num_tokens`.
- [ ] `total_routed_assignments == routing_token_count * router_topk`, and the
  sum of typed lane routed counts equals that assignment count.
- [ ] Routed-dependent zero-lane phases return `0.0` without positive-load
  model lookup, while physical participant/barrier membership remains intact.
- [ ] Explicit AFD/CUDA Graph compute padding retains precedence over the
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

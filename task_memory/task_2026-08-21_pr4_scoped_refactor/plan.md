## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-26 | Resolved the EP>1 aggregate admission gate as narrow A' and queued the shared early-admission TDD sub-step. |
| 2026-08-26 | Added the raw/effective width conservation repair and the explicit EP>1 aggregate admission decision gate. |
| 2026-08-26 | Recorded the dummy terminal-MTP physical-lane RCA, typed dummy seam gate, fresh focused matrix, and remaining commit/review gates. |
| 2026-08-26 | Completed the PDD attention-only identity repair with `64` focused GREEN tests; focused reporting and merge-readiness remain. |
| 2026-08-26 | Added the EP>1 aggregate/structural-MTP audit gate and the focused PDD attention-only layer-identity repair sub-step. |
| 2026-08-26 | Added and completed the independent predictor-layer-identity audit after the terminal MTP hook. |
| 2026-08-26 | Completed the terminal MTP EP>1 hook, focused GREEN matrix, numeric report, and implementation commits; merge-readiness remains pending. |
| 2026-08-26 | Added the terminal MTP overshoot EP>1 RED/GREEN hook sub-step and its numeric acceptance gates. |
| 2026-08-26 | Completed and committed the typed-trace observability sub-step as `183cfd61`; predictor-interface audit is next. |
| 2026-08-26 | Recorded the implementation handoff audit and promoted the trace-helper migration to the next concrete sub-step. |
| 2026-08-26 | Closed the MONOLITHIC initial-decode boundary audit and added its focused regression gate. |
| 2026-08-26 | Added the approved token-ledger repair sub-steps, RED/GREEN gates, and MONOLITHIC boundary audit. |
| 2026-08-25 | Added the MTP scope decision gate and the typed trace-helper completion step after implementation audit. |
| 2026-08-25 | Added the docs-first A' implementation plan for post-PR17 typed EP lanes, predictor/communication migration, pure MTP phases, and PR20/PR21 refresh gates. |
| 2026-08-22 | Synchronized PR #20/#21 descriptions to their published heads and archived the producer-metadata and MTP interface audits. |
| 2026-08-22 | Completed the Option-B standard-attention explicit-KV repair, warning visibility audit, two-round review archive, and targeted verification. |
| 2026-08-22 | Advanced stacked PR #21 through focused repair verification, closed the decode memory-accounting review watch, completed APPROVE/CLEAR review synthesis, and received authorization to push. |
| 2026-08-22 | Completed the focused PR #20 repairs and verification, recorded the pushed HEAD, and closed the fresh 20-minute Claude invocation as a timeout without verdict. |
| 2026-08-22 | Reopened the one-feature MoE exact-row and MTP TP-consumer sub-steps after Claude review and independent verification. |
| 2026-08-22 | Recorded option A and the verified PR #20 exact-row default-persistence sub-step. |
| 2026-08-22 | Restored the two-PR continuation, recorded remote state, and opened the finite exact-row producer decision gate on PR #20. |
| 2026-08-21 | Re-ran the committed-HEAD closing matrix, direct lookup probe, and sequential PDD case; recorded fresh output paths and numeric evidence. |
| 2026-08-21 | Completed Wave E direct lookup/PDD verification; recorded residual producer-metadata and canonical-CSV gates without widening the PR. |
| 2026-08-21 | Implemented option A for the MoE EP envelope: all runtime-legal divisors derive the helper and omitted CLI domain. |
| 2026-08-21 | Completed the MoE routing/load slice; isolated the legal-EP policy decision before joint TP/EP sampling changes. |
| 2026-08-21 | Completed the Wave D mixed-Attention envelope slice for legacy, online-grid, and true-mixed generators; MoE routing/load coverage remains pending. |
| 2026-08-21 | Completed the first Wave D sampling-domain endpoint slice for token, sequence, prefill-chunk, batch, and MoE token grids. |
| 2026-08-21 | Completed Wave C registry consumer migration and recorded the post-commit 235-test regression matrix. |
| 2026-08-21 | Completed the profiling-plan catalog cleanup; Wave C TP consumer migration remains in progress. |
| 2026-08-21 | Completed the staged import-boundary check and extracted the pure MoE runtime feature object; registry-consumer migration is next. |
| 2026-08-21 | Closed A1 lookup-contract verification; documented optional metadata limits and the A2 handoff. |
| 2026-08-21 | Entered implementation after maintainer approval; A1 lookup contract is in progress. |
| 2026-08-21 | Created the analysis-only plan, decision gates, scope boundaries, and acceptance criteria. |
| 2026-08-21 | Revised the plan after the dependency-cost and adapter-necessity audit; removed the independent adapter as a prerequisite. |
| 2026-08-21 | Closed the dependency and naming decisions; froze the `operator_query_binding` seam and the staged implementation gates. |
| 2026-08-21 | Added function-level call-site ownership and the ordered implementation handoff after the final source audit. |

# Plan - Scoped Lookup Boundary PR

## Active A' integration plan (post-PR17)

The earlier PR #20/#21 lookup and profiling waves remain historical context.
This active plan governs the conflict-resolution repair against the synthetic
post-PR17 integration head `ad2b558d`. The six dirty probe files are evidence
inputs only and must be rewritten to satisfy the design in `design.md`.

### Dependency map

```text
docs-freeze
  -> descriptor-contract-tests
  -> canonical-materializer
  -> scheduler-plan/entity-propagation
  -> predictor-interface
  -> communication-consumers
  -> MTP-contract-decision (historical; resolved by the 2026-08-26 narrow A' approval)
  -> MTP-pure-path (narrow physical MoE/EP phase)
  -> typed-trace-observability
  -> raw-width-conservation
  -> EP>1-aggregate-admission
  -> focused-verification
  -> PR20/PR21 merge-readiness
```

### Sub-steps and gates

1. **`docs-freeze`**
   - Files: `requirements.md`, `issues.md`, `design.md`, `plan.md`,
     `harness.md`, `progress.md`, `review.md`, `future.md`.
   - RED/GREEN: documentation placeholder scan, required-section scan, and
     cross-file terminology check.
   - Boundary: record RCA and A' only; do not alter production or tests.
   - Acceptance: every approved invariant has one owner and one verification
     gate; the provisional probe is explicitly labelled.
   - Commit: documentation-only commit on the task branch, then carry it to
     the integration branch before merge readiness.

2. **`descriptor-contract-tests`**
   - Files: `tests/unit/test_moe_ep_workload_materializer.py` and a focused
     typed-lane predictor test module under `tests/unit/`.
   - RED command: `PYTHONPATH=$PWD python -m pytest -q -p no:cacheprovider
     tests/unit/test_moe_ep_workload_materializer.py
     tests/unit/test_predictor_effective_tokens.py`.
   - Boundary: test immutable fields, fixed local width, sparse-zero
     densification, complete-map conservation, EP=1/EP>1 symmetry, and
     partial-map fail-fast behavior.
   - GREEN command: same command after the materializer implementation.
   - Acceptance: tests fail for the missing A' behavior before production edits
     and pass without a width escape hatch.
   - Commit: one contract-test commit only.

3. **`canonical-materializer`**
   - Files: `frontier/moe_ep_workload.py`.
   - RED source: the contract tests above.
   - Boundary: one immutable `EPLaneWorkload` constructor/factory owned by
     `LayerEPWorkload`; no scheduler identity fields and no predictor helpers.
   - GREEN: materializer contract tests plus `py_compile` for the module.
   - Acceptance: all lanes, including zero lanes, are materialized with fixed
     topology width and exact token conservation.
   - Commit: one production materializer commit.

4. **`scheduler-plan/entity-propagation`**
   - Files: `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`,
     `frontier/entities/batch.py`,
     `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py`,
     and the narrow stage ledger projection site.
   - RED command: existing EP scheduler/entity tests plus new descriptor
     identity assertions.
   - Boundary: plans/entities carry the descriptor; scheduler retains stage
     identity, events, participant sets, and barriers; raw maps become
     read-only projections.
   - GREEN: focused scheduler/entity matrix and PR17-sensitive stage/KV tests.
   - Acceptance: exact stage/KV provenance remains unchanged and every physical
     lane reaches the same barrier set.
   - Commit: one scheduler/entity propagation commit.

5. **`predictor-interface`**
   - Files: `frontier/execution_time_predictor/base_execution_time_predictor.py`,
     `sklearn_execution_time_predictor.py`,
     `sklearn_moe_execution_time_predictor.py`,
     `sklearn_disaggregation_execution_time_predictor.py`, mocks, and focused
     predictor tests.
   - RED command: predictor contract matrix covering regular `Batch`,
     `EPBatchGroup`, EP=1, EP>1, sparse maps, and zero lanes.
   - Boundary: predictor consumes typed descriptors; it does not split maps,
     infer topology, or own runtime identity. Keep dense
     `predict_stage_execution_time()` signature stable.
   - GREEN: focused predictor/disaggregation matrix and direct model-call probe.
   - Acceptance: local-width features are fixed, zero routed lanes make zero
     routed-dependent calls, and malformed identity/topology fails before model
     access.
   - Commit: one predictor-interface commit.

6. **`communication-consumers`** (runs after `scheduler-plan/entity-propagation`
   and `predictor-interface`)
   - Files: `frontier/operators/families.py` and any direct payload consumers
     identified by the typed projection search.
   - RED command: EP collective payload tests for ordinary and typed lane
     batches.
   - Boundary: use the descriptor-backed projection; retain dict output only
     as an explicit compatibility view.
   - GREEN: communication and scheduler event matrix.
   - Acceptance: payload bytes use the correct local/global domain without
     `isinstance(EPBatchGroup)` branching.
   - Commit: one communication-consumer commit.

7. **`MTP-contract-decision`**
   - Evidence: `issues.md:SCOPE-026` documents the implemented physical lane
     seam and the still-existing generic block-shape replay.
   - Boundary: obtain the maintainer's choice between the narrow physical-lane
     contract and a new shared pure-MTP descriptor interface.
   - Acceptance: the selected contract is reflected consistently in
     `design.md`, `harness.md`, tests, and the remaining file map.
   - No production edit occurs before this gate is resolved.

8. **`MTP-pure-path`**
   - Files: `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`
     and its MTP-focused tests.
   - RED command: MTP phase tests that reject synthetic scheduler entity
     construction and assert lane-wise phase values.
   - Boundary: pure descriptor and phase calculation only; preserve existing
     decomposition and lane-wise max.
   - GREEN: MTP-focused matrix plus the target-embedded registry tests.
   - Acceptance: the selected SCOPE-026 contract is met; the narrow option
     forbids synthetic physical EP entities while retaining the explicit
     generic shape adapter.
   - Commit: one MTP-path commit.

9. **`typed-trace-observability`** (completed in `183cfd61`)
   - Files: `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`,
     `frontier/events/replica_stage_schedule_event.py`, and focused trace tests.
   - Boundary: pass `EPLaneWorkload` to the trace helper and create the raw-map
     projection only inside the serialization/logging boundary.
   - Acceptance: no production trace helper accepts a raw expert-token map.

10. **`focused-verification`**
   - Commands: focused tests, PR17-sensitive tests, `py_compile`,
     `git diff --check`, and direct EP=1/EP=2 zero-lane probes.
   - Acceptance: numeric evidence records model-call counts, local widths,
     routed-token sums, barrier participant sets, and exact stage/KV identities.
   - Commit: verification report/documentation update only.

11. **`PR20/PR21 merge-readiness`**
   - Boundary: refresh PR20's post-PR17 conflict resolution first, then
     rebase/refresh stacked PR21 against the corrected PR20 base. Do not push,
     merge, or alter remote PR state without a separate authorization gate.
   - Acceptance: local heads, base relationship, and diff ownership are
     explicit; no unresolved A' gate remains.

## Approved token-ledger repair addendum (2026-08-26)

This addendum supersedes the earlier open `MTP-contract-decision` wording for
the confirmed narrow A' path. The maintainer approved the shared `Batch`
compute-contract correction and the structural verification-shape correction.

### Dependency map

```text
token-ledger-RED-tests
  -> shared-Batch-compute-contract
  -> structural-MTP-verify-shape
  -> monolithic-initial-boundary-audit
  -> focused-token-ledger-verification
  -> PR20/PR21 merge-readiness refresh
```

The shared helper and structural replay edits are sequential because both
depend on the same canonical ledger. The boundary audit runs after those edits
and before the final matrix; it may remain unchanged if the existing scheduler
frontier is proven to represent a distinct, explicit shape.

### Sub-steps

1. **`token-ledger-RED-tests`**
   - Add the smallest regression cases for target-embedded MTP:
     `planned=[2,1]`, `verify=[3,2]`, compute width `5`, transfer width `5`,
     and routing assignments `10` at top-k 2.
   - Assert that rejection metadata does not prune the first target-forward
     structural shape and that the materializer consumes the pre-routing count.
   - Run the new tests before production edits and record the expected failures.

2. **`shared-Batch-compute-contract`**
   - Update `Batch.get_effective_total_tokens_for_compute()` only at its
     target-embedded MTP branch. Preserve explicit decode CUDA Graph and AFD
     stage precedence and all non-MTP behavior.
   - Verify the shared helper through compute, gating, communication, and stage
     ledger callers rather than adding caller-specific bypasses.

3. **`structural-MTP-verify-shape`**
   - Update the first structural MTP verification block to use complete
     `verify_tokens_per_request` values. Keep later MTP block selection and
     acceptance/progression metadata unchanged.
   - Preserve the generic scheduler-independent shape adapter and the pure
     typed MoE lane phase seam.

4. **`monolithic-initial-boundary-audit`**
   - Exercise the `max(planned_drafts, 1)` MONOLITHIC initial decode path and
     compare its scheduler-visible width with the metadata/forward ledger.
   - Encode a rule only if the direct probe establishes a distinct physical
     boundary; keep the rule in the existing scheduler/Batch seam.

   - **Audit result:** The production `Request`/scheduler probe confirmed a
     distinct one-time admission frontier (`num_processed_tokens =
     num_prefill_tokens + 1`, scheduler frontier at `num_prefill_tokens`).
     Keep the existing branch and add a focused regression; do not change the
     shared Batch compute or MoE materializer contracts.

5. **`focused-token-ledger-verification`**
   - Run the MTP, MoE, EP=1/EP>1, zero-lane, communication, scheduler, and
     PR17-sensitive tests, plus `py_compile` and `git diff --check`.
   - Record actual model-call counts, token widths, assignment conservation,
     lane participants, and boundary observations in the task report.

Each production sub-step is committed only after its RED/GREEN verification;
the pre-existing dirty A' probe files remain in scope and are not reverted.

## Status

### 2026-08-26 implementation handoff checkpoint

The documentation-first checkpoint is complete. The remaining implementation
must proceed in dependency order:

```text
descriptor-contract-tests
  -> canonical-materializer
  -> scheduler-plan/entity-propagation
  -> predictor-interface
  -> communication-consumers
  -> MTP-pure-path
  -> typed-trace-observability (complete: `183cfd61`)
  -> focused-verification
  -> PR20/PR21 merge-readiness
```

The worktree contains provisional implementation for the remaining dirty
sub-steps; each slice must be independently re-vetted against the design and
committed only after its focused RED/GREEN gate passes. The raw-map trace
boundary is closed; predictor-interface audit is the next concrete gate.

**Current phase:** PR #20 remains maintainer-approved, open, and unmerged at
`18d1a23e`. PR #21 is published at `8cc267ea`, based on PR #20, and remains
open and unmerged. Both remote PR descriptions now match their published
heads and record the latest Option-B evidence. No merge or auto-merge is
enabled.

### 2026-08-26 continuation checkpoint

The approved narrow A' production path is implemented through the terminal
MTP dummy boundary. The remaining execution chain is now:

```text
dummy-terminal-lane-gate
  -> focused-matrix-refresh
  -> coherent-substep-commits
  -> independent-code-review
  -> PR20/PR21-read-only-merge-audit
```

- `dummy-terminal-lane-gate`: complete. Dummy lane phase prediction reuses
  `_get_dummy_execution_time()` with the typed descriptor; non-dummy prediction
  retains `_get_execution_time_internal()`.
- `focused-matrix-refresh`: evidence captured as `350 passed in 10.68s`, with
  direct dummy phase values `27.0/33.0/33.0 ms` for generic/Step2Mini/Step3
  profiles and zero-lane routed compute `0.0 ms`.
- `coherent-substep-commits`: pending. Production, regression, and task-doc
  files must be staged by ownership and committed only after their focused
  commands are rerun.
- `independent-code-review`: pending before merge-readiness; reviewer findings
  will be independently checked against this plan and the actual diff.
- `PR20/PR21-read-only-merge-audit`: pending. No remote write or PR state
  mutation is authorized by this plan.

**Implementation status:** The dependency-boundary and naming decisions are
confirmed. The maintainer approved the bounded file map and acceptance gates;
Wave A1 registry admission, the validated on-demand contract, finite dense/MoE
consumer migration, and the staged import-boundary check are implemented and
verified. The pure `MoELoadImbalanceInput` feature object now has one
runtime-owned definition with a profiling compatibility re-export. Wave C's
ordinary registry admission and target-embedded MTP TP-consumer sub-step are
complete: all three consumers read the existing structural MTP registry
accessor. Wave D's primitive, mixed-Attention, and MoE routing/load slices are complete:
default/explicit token domains, sequence lengths, prefill chunks, attention
batch sizes, legacy mixed-prefill KV/context pairs, online-grid extensions,
true-mixed prefill/decode endpoints, and the profiling-owned MoE token config
now include their requested upper endpoints. MoE routing identity and load
distribution edge cases now fail fast and preserve the requested route. The
MoE EP envelope now includes every positive divisor accepted by runtime
divisibility validation; omitted CLI selections resolve that domain per model,
while explicit smoke selections remain supported. Wave E direct predictor and
heterogeneous PDD cases are also verified. Canonical CSV regeneration for
newly covered divisors and a producer-side persistence contract for optional
domain metadata remain outside this code-only slice. A read-only producer audit
confirmed that optional bounds/domain/constraint/identity fields still have no
production producer or serialized-cache round trip; this is a pre-existing
cross-cutting follow-up, not a PR #21 regression. The maintainer selected
universal scalar-numeric exact-row persistence for PR #20. Both unified
runtime training/cache entry points and both current MoE producers now follow
that contract. The independent MTP audit confirmed that the two physical
target-embedded MTP linear operators are not members of the generic
`OperatorFamilySpec` registry, while the focused PR #20 repair correctly
unifies the three TP consumers through the dedicated MTP accessor. Full MTP
family/interface unification remains a separately scoped cross-cutting design
task in `future.md`.

**Worktree:** `.worktrees/pr4-scoped-lookup-boundary-20260821`

**Branch:** `refactor/pr4-scoped-lookup-boundary-20260821`

**Base:** `origin/main` at `9e9fa94c`.

### Post-PR17 implementation audit status (2026-08-25; superseded by the 2026-08-26 approval)

The synthetic post-PR17 integration branch has completed the descriptor,
scheduler/entity, predictor, communication, and MoE-specific MTP lane slices.
The historical remaining gate was SCOPE-026. The maintainer subsequently
selected the narrow physical-lane contract, so the generic block/terminal
shape replay remains a named scheduler-independent adapter. The raw-map trace
helper is a separate localized follow-up, and PR20/PR21 refresh and
merge-readiness remain pending.

## 2026-08-22 two-PR continuation

- Remote PR #20, `refactor/pr4-lookup-registry-core-20260822 -> main`, owns the
  lookup, cache, runtime/profiling boundary, and registry-consumer slice.
- Remote PR #21,
  `refactor/pr4-profiling-envelope-ep-20260822 -> refactor/pr4-lookup-registry-core-20260822`,
  owns the sampling-envelope, MoE routing/load, and legal-EP slice.
- Both PRs are open and unmerged. PR #20 is the active review/fix target; PR
  #21 remains stacked and receives its integration refresh after the base is
  accepted.
- The resumed PR #20 audit found partial producer admission: mixed/MLA/CPU/
  KV-cache paths persisted exact rows, while ordinary Linear, standard
  Attention, and legacy one-feature MoE used the false default.
- The maintainer selected option A. Commit `d45d836c` changes both unified
  runtime training/cache entry points to `persist_exact_lookup=True` by
  default. The RED regression failed twice with missing
  `_frontier_exact_lookup`; GREEN passed both cache round trips.
- The committed-HEAD direct probe preserved the lookup order across a pickle
  round trip: exact measured `4.0` beat stale runtime cache `99.0`, runtime
  cache returned `8.25`, and an unmeasured key returned model value `7.5`
  twice with one model call.
- Commit `d45d836c` was pushed through the HTTPS proxy and PR #20's body now
  records the RED/GREEN, focused, broader, and direct-probe evidence. PR #20
  reports `OPEN/CLEAN/MERGEABLE`.
- PR #21 remains `OPEN/CLEAN/MERGEABLE` at `03952bf1`. A local synthetic merge
  with the new base produced tree
  `2a1ed9c5bb1b1ed412f5cef7f3f8e7e262bb9506` without conflict and retains the
  same nine profiling-only diff paths.
- The first 20-minute Claude invocation timed out before verdict. The
  maintainer-requested resume exited `0` and returned `REQUEST CHANGES`.
- Before repair, independent replay confirmed `SCOPE-014`: measured `4.0`
  became `99.0`
  without one-feature persistence, an absolute error of `95.0` and relative
  error of `2375%`.
- Before repair, independent registry extension confirmed `SCOPE-016`:
  predictor, shared manager, and trainer rejected the new registry member, so
  `3/3` TP consumers were tied to the duplicated local set.
- Commit `3e9c0374` removed the two MoE feature-count overrides. Commit
  `18d1a23e` routed all three target-embedded MTP TP consumers through
  `get_target_embedded_mtp_linear_ops()`.
- Fresh verification passed four targeted cases, the affected-file matrix
  (`62 passed`), and the broad PR #20 matrix
  (`289 passed, 2 deselected`). The direct probe returned measured `4.0`
  ahead of stale `99.0` with absolute error `0.0`, and the MTP registry
  extension reached `3/3` consumers with TP `2`.
- The branch was pushed through the HTTPS proxy. Local and remote PR #20
  heads both equal `18d1a23e1a04f3dcad2d2d1376451190d2b5300a`.
- The fresh Claude re-review used Opus 5, `effort=max`, and `timeout 1200s`.
  It used seven tools but emitted no final response before timeout exit `124`;
  therefore no external verdict is available from that invocation.
- PR #20 remains `OPEN/CLEAN/MERGEABLE`. PR #21 remains open at `03952bf1`;
  GitHub currently reports its merge state as `UNKNOWN`, while a local
  synthetic merge produced tree `42526bf389c62698988fe09ec8417a8e7ad908ff`
  without conflict and retained the same nine profiling-only paths.
- Both PRs remain unmerged. The next action is maintainer review; another
  Claude continuation or longer run requires explicit authorization.

## Objective

Produce one small PR that fixes the lookup-domain failure at its cause, makes
the profiling/runtime boundary file-based, and makes operator admission
registry-driven. The PR must be understandable without the old PR #4 branch
and must not absorb the old branch's N3/N4/N5 publication work.

## Evidence phases

| Phase | Status | Purpose |
|---|---|---|
| 1. Baseline and source inventory | Complete | Confirm the clean worktree, focused tests, legacy diff, current lookup sites, profiling imports, and prefix branches. |
| 2. Gap analysis and scope record | Complete in this document set | Separate confirmed root bugs from old-branch scope creep and record numeric evidence. |
| 3. Maintainer dependency-boundary decision | Complete | Staged, side-effect/cost-based import boundary accepted; shared CSV helpers and named exceptions remain explicit. |
| 4. Maintainer operator-query naming/seam decision | Complete | `operator_query_binding` / `bind_operator_query` selected; existing unified registry remains the only ordinary operator catalog. |
| 5. Implementation plan confirmation | Complete | Function-level ownership and the bounded file map were accepted with the instruction to continue. |
| 6. Implementation and direct verification | Complete for focused PR #20 repair | Waves A1, A2, B, C, D, and E are verified. `SCOPE-014` and `SCOPE-016` are resolved; producer-side optional metadata and canonical measured rows remain tracked residuals. |
| 7. Fresh external review after repair | Invocation complete; verdict unavailable | Ask Claude used the exact Opus 5/max/1200-second configuration and timed out with exit `124` before final synthesis. |

## Implementation waves and completed slices

### Wave A - One validated lookup path

1. Define one runtime-owned lookup entry point for ordinary finite and
   high-dimensional prediction queries.
2. Validate operator identity, model/device/TP/EP/family identity, feature
   names and order, numeric/finite values, physical bounds, and relational
   constraints before any cache or model access.
3. Resolve values in this order:
   `validated exact measured value -> process-local runtime cache -> canonical
   model prediction`.
4. Cache only a validated model prediction in the process-local runtime cache.
   Never mutate profiling CSVs, measured-row metadata, or persisted model
   identity.
5. Reject NaN, infinity, negative timing, nearest-row substitution, clamping,
   defaults, and unvalidated fallback. Preserve the intentional optional
   CPU/PP missing-profile `0.0` behavior as an explicitly owned path.
6. Replace direct dictionary indexing in the affected ordinary paths with the
   common entry point; keep communication backend calls and fused-operation
   semantics in their existing owners.

The first code slice is limited to the predictor lookup implementation and its
direct call sites: `_get_on_demand_prediction`, the one-feature linear/FFN
methods at `sklearn_execution_time_predictor.py:4394-4524`, the dense
attention/speculative methods at `:5499-5855`, the MLA operator-time method at
`:6483-6524`, and the legacy one-feature MoE branches at
`sklearn_moe_execution_time_predictor.py:1173-1227,1440-1516,1757-1910`.
The optional CPU/PP helpers are not routed through this slice.

### Wave A1 completion record

- `bind_operator_query` admits exact physical names and unique profiling aliases
  through the existing registry and fails on unknown, ambiguous, mismatched, or
  disabled ownership.
- `_get_on_demand_prediction` validates the declared schema and model schema
  before cache access, normalizes finite non-negative numeric features, checks
  explicitly supplied bounds/constraints/identity metadata, and resolves
  `exact measured -> runtime cache -> model` in that order.
- Exact measured values remain read-only; only a validated model result enters
  the process-local runtime cache. Negative, non-finite, malformed, and
  multi-result estimator outputs fail fast instead of being clamped.
- `_feature_bounds`, `_feature_domain`, `_feature_constraints`, and identity
  metadata are optional extension points in this slice. Current-main producer
  code does not emit them yet, so A1 does not claim complete producer-side
  runtime-domain publication. A2/B must add a concrete producer before making
  any of these fields mandatory.
- Fresh evidence is recorded in
  `test_report_2026-08-21_a1_lookup_contract.md`; the combined attention logger
  failures are reproduced on clean `origin/main` and remain outside A1.

### Wave B - Runtime predictor lifecycle and staged file boundary

1. Keep serialized model-cache lookup in runtime initialization.
2. On a cache miss, resolve the configured raw CSV path, load the current
   schema, fit the configured estimator, and publish only the runtime-owned
   serialized predictor artifact.
3. Prohibit runtime imports of profiling benchmark runners, GPU wrappers, and
   profiling CLI entry points. Do not promise a literal zero import before the
   migration cost is justified.
4. Keep side-effect-free CPU/PP CSV schema and validation helpers as an
   explicitly allowlisted shared implementation. Moving them would touch both
   producer and consumer modules and would duplicate the same CSV invariants.
5. Extract only the pure MoE runtime feature object if a small, direct split is
   confirmed; leave profiling sampling configuration in its current owner.
6. Treat the non-KV-cache-memory and MTP structural-config paths as named,
   separately tested exceptions; do not silently expand them to compute timing.
7. Add an import-boundary check for forbidden benchmark/GPU modules and an
   allowlist check for the temporary shared helpers.

The import audit starts with the actual consumers currently importing
`frontier.profiling`: the CPU/PP validators in
`sklearn_execution_time_predictor.py` and `shared_prediction_model_manager.py`,
the `MoELoadImbalanceInput` import in
`sklearn_moe_execution_time_predictor.py:1385`, the non-KV-memory imports in
`scheduler/replica_scheduler/base_replica_scheduler.py:109-114`, and the MTP
structural config import in `spec_decode/mtp_runtime.py:16`. Only the first
three are compute-predictor candidates; the latter two remain named
exceptions.

**Wave B completion:** The runtime predictor now imports the pure
`frontier.moe_load_imbalance` feature module. The profiling module keeps
`MoELoadImbalanceInput` as a compatibility re-export, while its sampling
configuration remains profiling-owned. An AST boundary test confirms that
predictor imports contain only the three allowlisted CSV/schema helpers and
that training Python modules have no profiling implementation imports. The
non-KV-memory and MTP imports remain explicit exceptions.

### Wave C - `operator_query_binding` registry admission (no adapter by default)

1. Reuse `OperatorSpec`, `OperatorFamilySpec`, and existing attention/MoE
   registry APIs as the authoritative source.
2. Add `bind_operator_query(...)` to the existing
   `frontier/operators/binding.py` surface first. Create
   `frontier/operators/operator_query_binding.py` only if the implementation
   proves that the existing file has become overloaded.
3. Add one explicit alias/many-to-one mapping table only where current timing
   names cannot be obtained from `profiling_name()`.
4. Keep timing ownership and TP policy declarative without adding a second
   profiling-facing operator catalog.
5. Route TP selection, dataset selection, model binding, and predictor-family
   dispatch through `bind_operator_query` or the existing family-owned hooks it
   selects.
6. Unknown names, unsupported owner states, alias collisions, and conflicting
   physical-to-profiling mappings fail at admission time.
7. Remove only the classification branches that make operator-family or TP
   decisions from name shape. Format parsing such as `time_stats.*` remains
   outside this change.

The concrete TP/classification consumers are
`SklearnExecutionTimePredictor._get_linear_op_tp_key`,
`ExecutionTimePredictionModelManager._get_linear_op_tp_key`,
`LinearOpTrainer._get_training_tp_key`, and
`AttentionTrainer._get_compute_tp_key`. `attention_tp_policy.py` already derives
the non-linear attention set from the family; its remaining `ATTENTION_LINEAR_OPS`
literal set is a registry-consumer cleanup, not a new catalog. The standalone
`MoETrainer` already resolves family membership through `MOE_FAMILY`; its
load-imbalance feature-mode decision is included only if a declarative field can
be added without moving the profiling sampler.

`profiling/linear_op/profiling_plan.py` is included only to replace its ordinary
attention lists with family/role enumeration. MTP names remain owned by the
existing `spec_decode/mtp_registry.py`, which is an explicit structural
exception rather than a second profiling-facing adapter.

The PR #20 review reopened one bounded Wave C sub-step: the predictor, shared
manager, and Linear trainer TP-key methods must read target-embedded MTP linear
membership from `get_target_embedded_mtp_linear_ops()`. Other MTP occurrences
serve distinct enumeration, schema, profiling-kernel, quantization, or runtime
policy roles and remain outside the focused option unless the maintainer
authorizes a broader audit.

### Wave D - Deterministic profiling coverage

1. Keep profiling as an offline CSV producer; do not let active simulation
   measure, train, or write profiling data.
2. Build per-family sampling envelopes from executable configuration and
   operator-specific runtime features.
3. Ensure every profiling axis reaches at least the runtime maximum plus the
   first legal canonical anchor, subject only to explicit physical capacity.
4. Sample boundaries, single-axis sweeps, pairwise interactions, and
   operator-specific risk cases deterministically. Derive coupled features from
   real joint inputs.
5. Record the measured domain and runtime domain separately so a coverage gap
   is an actionable error, not a silent lookup miss.
6. Derive the MoE EP axis from every positive runtime-legal divisor. Resolve
   omitted CLI EP selections per model, validate explicit selections against
   the same domain, and leave canonical CSV regeneration to the offline
   profiling workflow.

The first sampler audit targets `get_num_tokens_to_profile` (including the
currently ignored oversized `extra_num_tokens`),
`get_seq_lengths_to_profile` (currently excludes `max_seq_len`), and
`get_attention_input_combinations`/`get_attention_prefill_chunk_sizes_to_profile`
for joint batch/KV/prefill coverage. MoE token/expert/TP envelopes remain in
`profiling/moe/moe_input.py`; the EP legality resolver is pure and derives its
domain from the runtime divisibility invariant without moving sampling
configuration into runtime.

### Wave E - Focused verification

1. Add the smallest regression tests justified by the root bugs: legal lookup
   miss, repeated runtime-cache hit, invalid identity/schema/bounds, malformed
   prediction, unknown operator, alias collision, and import boundary.
2. Run direct simulator cases for one Linear, one Attention, and one MoE
   operator, plus one heterogeneous role identity if the implementation touches
   shared model management.
3. Record exact commands, environment, model-call counts, cache-hit counts,
   predicted values, and failure messages in a task-local test report.

**Wave E completion:** The reproducible direct probe covers Linear, Attention
decode/prefill, MoE grouped-GEMM, exact-measured precedence, and NaN rejection.
The focused matrix passed `111` tests and the broader final cross-wave matrix
passed `306` tests (the earlier narrower cross-wave matrix passed `265`). A
sequential PDD case completed two requests with separate PREFILL and
DECODE predictor objects sharing one model manager; it processed `24` tokens
and recorded `2` KV transfers (`8,388,608` bytes total). The full command and
numeric output are in
`test_report_2026-08-21_wave_e_direct_cases.md`.

The committed-HEAD rerun independently passed the 26-file matrix with
`306 passed in 19.79s`; the direct probe printed `direct_probe PASS`; and the
fresh PDD artifacts are under
`/data/ycfeng/tmp/pr4_completion_pdd_20260821/`. These values are a closing
verification refresh, not a change to the scoped acceptance criteria.

Implementation order is intentionally sequential: (A1) binding and lookup
contract tests, (A2) finite/on-demand consumers, (B) import boundary and any
small MoE feature extraction, (C) registry consumers, (D) sampler coverage,
then (E) direct Linear/Attention/MoE runs. A failed gate stops the next wave;
there is no temporary fallback wave.

## Planned file ownership (implementation review gate)

This was the bounded pre-implementation ownership map. The maintainer
confirmed it before the approved waves; the resulting edits and verification
remain limited to this map plus the task-local reports.

| Area | Candidate files | Allowed responsibility |
|---|---|---|
| Registry binding | `frontier/operators/binding.py`; possibly a later `frontier/operators/operator_query_binding.py` | Define `OperatorQueryBinding` only if a value object is needed; implement `bind_operator_query`; perform exact registry membership and explicit mapping checks. No CSV or predictor imports. The new file is forbidden unless `binding.py` is demonstrably overloaded or cyclic. |
| Registry declarations | `frontier/operators/spec.py`, `frontier/operators/families.py`, and only the relevant family module | Add or correct declarative metadata required by an existing operator. Do not duplicate the registry in a profiling package. |
| Predictor consumers | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` (`_get_on_demand_prediction`, finite linear/attention/MLA methods listed above), `shared_prediction_model_manager.py` (`_get_linear_op_tp_key`), and `sklearn_moe_execution_time_predictor.py` (gating/shuffling/grouped-GEMM methods listed above) | Replace prefix/literal-set classification and direct lookup branches with the shared binding/validation path; preserve owner-specific numerical logic and calibration. |
| Trainer consumers | `frontier/training/linear_op_trainer.py` (`_get_training_tp_key`), `frontier/training/attention_trainer.py` (`_get_compute_tp_key`), and `frontier/training/moe_trainer.py` only if its load-imbalance mode can be declared without moving sampling config | Consume registry metadata; do not add a second trainer catalog. |
| Profiling plan/sampling | `frontier/profiling/linear_op/profiling_plan.py`; `frontier/profiling/utils/__init__.py`; `frontier/profiling/moe/moe_input.py`; relevant attention sampling caller | Derive ordinary operator lists from registries and extend deterministic runtime-domain envelopes only. Do not add publication/provenance machinery or runtime imports. |
| Boundary helpers | Existing CPU/PP schema and validation modules; optional small MoE feature extraction | Keep the allowlisted shared helpers; extract only a pure runtime object if the diff remains local and sampling configuration stays profiling-owned. |
| Verification | Smallest justified files under `tests/unit/`, `tests/integration/`, or `tests/e2e/`; task-local reports under `task_memory/...` | Cover exact binding, lookup precedence, fail-fast errors, import boundary, and profiling-domain coverage with direct numeric evidence. |

Any file outside this map requires a new root-cause explanation and an explicit
scope review before editing.

### File-map stop conditions

- `frontier/operators/spec.py`/`families.py` are edited only when an existing
  operator lacks declarative metadata needed by a current consumer. Adding a
  generic timing-owner catalog is out of scope.
- `frontier/spec_decode/mtp_registry.py` is edited only for an explicit MTP
  mapping already represented by its structural registry; it is not folded
  into a new profiling adapter.
- A pure `MoELoadImbalanceInput` extraction is optional. If it changes more
  than the class, its direct runtime import, and focused tests, defer it and
  keep the current shared import documented.
- No CSV schema/validator relocation is attempted unless a line-count and
  import-cycle check shows a local, side-effect-free extraction.

## Acceptance criteria for implementation phase

- A legal runtime key absent from a finite materialized table reaches the
  canonical model exactly once and is reused from the process-local cache on a
  repeated query.
- An illegal or identity-incompatible key fails before model invocation and
  before a cache write.
- A malformed/non-finite/negative model result fails rather than being clamped.
- Runtime does not load profiling benchmark runners or GPU wrappers on the
  supported simulation path. Any remaining CSV-helper, non-KV-memory, or MTP
  imports are explicit, side-effect-free/exception paths and are documented.
- Adding an ordinary existing-owner operator requires one unified-registry
  declaration and no consumer-local prefix or literal-set edit. A second
  adapter entry is not required.
- Profiling CSV coverage is demonstrably a superset of the runtime domain for
  the tested family and configuration on the code-side sampling envelope;
  canonical rows for newly exposed EP values remain a separately tracked
  offline data task.
- Focused tests and direct runs pass with numeric evidence; no canonical data,
  README, remote PR, or main checkout is modified.

## Explicit exclusions

- N3/N4/N5 incremental profiling database and preflight/report workflow.
- Canonical profiling-data publication, merge provenance, sidecars, aliases,
  or historical provenance repair.
- Broad heterogeneous PDD/PD-AF artifact publication matrices, except a narrow
  identity check required to prove the lookup/cache boundary.
- New Contract/Data Guarantee abstraction layers.
- Predictor accuracy calibration, arbitrary scale factors, nearest-row or
  default fallbacks.
- Remote GitHub PR operations, merge, push, or changes to `README.md`.

## Final planning gate before implementation

The dependency boundary and operator-query name are confirmed. The maintainer
accepted this file's ownership map and the following stop conditions before
implementation:

1. `bind_operator_query` remains runtime/registry-facing and imports no
   benchmark runner, GPU wrapper, profiling CLI, CSV reader, sklearn class, or
   cache implementation.
2. Existing side-effect-free CPU/PP schema helpers and the named non-KV-memory
   and MTP exceptions remain explicit rather than becoming a blanket import
   rule.
3. A concrete current-main owner gap is required before any minimal bridge or
   new file is added.
4. The first implementation wave is limited to the files above; N3/N4/N5,
   publication, canonical data, and broad model migrations stay deferred.

See `design.md`, `findings.md`, and `issues.md` for the evidence behind these
gates.

## 2026-08-22 PR #21 focused repair continuation

- **Scope:** Standard decode context bounds, mixed-Attention planning
  validation, MoE routing dimensions, and multi-model MoE confirmation/progress
  accounting.
- **Commits:** `583a6181`, `e7fa3a3e`, `f0f5e515`, `a7498d15`, and
  `b8149c32` on
  `refactor/pr4-profiling-envelope-ep-20260822`.
- **Composition:** PR #20 base `18d1a23e` plus PR #21 tip produces synthetic
  tree `dc808182` without conflict. The stacked diff is 11 profiling/test
  paths, `1073` insertions, and `125` deletions.
- **Verification gates:** Synthetic-stack broad matrix passed `322` tests with
  `2` known logger cases deselected; PR #21 sampling/EP targeted tests passed
  `33`; MoE routing/load tests passed `13` under `/usr/bin/python` with
  Torch `2.5.1+cu124`; seven production files passed `py_compile`;
  architecture re-review returned `CLEAR`; diff, changed-path, and
  conflict-marker audits passed.
- **Numeric evidence:** Standard decode resolves `kv_cache_size=999` for
  `max_seq_len=1000`, giving wrapper `total_len=1000`. Invalid mixed axes
  fail at the planning boundary. Decode memory filtering rejects
  `kv_cache_size=32` at capacity `32` and accepts it at capacity `33`.
  Default MoE confirmation reports `12,432 + 37,296 = 49,728`
  configurations. Extreme-skew routing keeps `423/512` rows in the hot
  subset. The skewed expert-0 count is
  Torch-version dependent (`69/1024` on Torch `2.5.1+cu124`,
  `61/1024` on Torch `2.12.1+cpu`); the contract asserts reachability rather
  than a backend-specific exact count.
- **Next gate:** Maintainer review of PR #21 and an explicit merge decision.
  Keep PR #20 and PR #21 open until that decision; no merge was performed in
  this step.

## 2026-08-22 PR #21 Option-B explicit-KV feedback repair

- **Scope:** Standard-attention explicit decode KV values only. The repair
  separates the profiling envelope (`max_seq_len`) from the runtime context
  boundary (`max_model_len`) and does not change mixed/true-mixed capacity
  semantics.
- **Behavior:** Automatic standard decode remains bounded by
  `max_seq_len - 1`. An explicit value is logically legal through
  `max_model_len - 1` after reserving the current decode token. Each selected
  `(model, TP)` target filters against its physical token capacity.
- **Discard contract:** A target-local explicit drop emits `RuntimeWarning`
  before worker dispatch and includes the dropped KV values, combination count,
  model, TP, capacity, and retained values. The parent process prints the
  requested/retained union and target-capacity map.
- **Fail-fast contract:** Retained explicit values are unioned across targets.
  `ValueError` occurs only for a requested value absent from that union, and
  the error lists the missing values plus every target capacity. Explicit
  values outside `max_model_len - 1` continue to fail before physical filtering.
- **Implementation:** Commit `8cc267ea` adds the target-local filter,
  warning/summary path, global coverage validator, and focused regression
  file `tests/unit/test_attention_memory_filter_contract.py`.
- **Verification:** The focused warning/coverage tests passed `3`; the
  combined PR #21 sampling/EP/routing matrix passed `59`; compileall and
  `git diff --check` passed. A direct probe captured one warning for two
  discarded combinations (`KV=[300, 511]`, capacity `256`), retained union
  `[255, 300, 511]` across capacities `256/512`, and raised the expected
  all-target error for `KV=[511]`.
- **Review disposition:** Round-1 code/spec was `APPROVE`; Round-2
  code/spec was `CLEAR/APPROVE`; Round-2 architecture was `CLEAR with WATCH`.
  The Round-1 architecture artifact targeted the preceding `a8d75b84` tip and
  is retained as pre-final evidence, not as the final `8cc267ea` verdict.
  Ask Claude's concurrent `1800s` run timed out with no final verdict.
- **Next gate:** Keep PR #20 at `18d1a23e` and PR #21 at `8cc267ea` open and
  unmerged. Treat the true-mixed block-rounding mismatch, duplicate targets,
  direct-CLI default derivation, broad automatic exception catches, and strict
  Python integer validation as deferred follow-ups.

## 2026-08-22 post-publication audits and description sync

- **Producer metadata audit:** `SCOPE-011` remains open and outside PR #21.
  Production assignments for `_feature_bounds`, `_feature_domain`,
  `_feature_constraints`, `_identity`, and `_model_identity` are all absent.
  The standalone `BaseTrainer` path also persists only the raw estimator.
  Repairing this requires a shared descriptor contract across producers,
  runtime `model_info`, serializer/cache reload, and consumer verification.
- **MTP interface audit:** `mtp_fusion_proj` and `lm_head_linear` are separate
  physical `ColumnParallelLinear` operators. They are not generic-family
  aliases and do not bind through `bind_operator_query`; their dedicated
  MTP registry accessor correctly supplies TP policy to the three repaired
  consumers. Registry extension does not yet reach every model-name and
  profiling-plan enumeration site, so full one-entry extension remains a
  medium-priority follow-up.
- **Remote description sync:** PR #20 body now reports head `18d1a23e`; PR #21
  body now reports head `8cc267ea`, the warning/union/all-target evidence, and
  the Claude timeout as inconclusive. Both PRs remain `OPEN/CLEAN` and
  unmerged.

## Terminal MTP overshoot continuation (SCOPE-030)

### Dependency map

```text
terminal-mtp-red
  -> terminal-row-hook
  -> focused-terminal-green
  -> predictor-layer-identity-audit
  -> focused-report-and-commit
  -> PR20/PR21 merge-readiness
```

### Sub-steps and gates

12. **`terminal-mtp-red`** (completed)
    - File: `tests/unit/test_mtp_terminal_overshoot_ep_replay.py`.
    - Boundary: exercise the real scheduler metadata builder and the real
      non-dummy EP=2 MoE predictor path. Assert terminal metadata, physical
      lane expectations, stage/layer propagation, and lane assignment
      conservation.
    - Evidence: after correcting the fixture's phase arithmetic, the test
      fails only with the missing `EPLaneWorkload` descriptor error from the
      canonical all-to-all payload builder.

13. **`terminal-row-hook`** (completed)
    - Files: `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`
      and `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`.
    - Boundary: add a default dense terminal-row hook and a MoE override that
      reuses `predict_moe_lane_phase_times()` and `LayerEPWorkload.lane()`. Keep
      the generic synthetic batch as a scheduler-independent shape adapter.
    - Contract: shared attention/pipeline/CPU scope is evaluated once;
      physical five-phase lane values are aggregated with `max()` per phase;
      `num_layers` scales only per-layer physical phases; `stage_id` and
      `layer_id` remain distinct.
    - Forbidden: raw expert maps, fabricated descriptors, synthetic
      `EPBatchGroup`, a second phase aggregator, or a new shared MTP descriptor.

14. **`focused-terminal-green`** (completed)
    - Run the terminal regression together with the existing structural MTP,
      typed-lane, predictor-effective-token, communication, and disaggregation
      matrices listed in `progress.md`.
    - Record terminal row count, verification width, lane IDs/local widths,
      routed counts, zero-lane model-call counts, five phase values, phase-wise
      maxima, total time, and stage/layer identity in the task test report.
    - **Evidence:** The terminal regression and structural/typed-lane matrix
      pass (`32 passed` for the terminal/structural/typed subset; `178 passed`
      for the full focused matrix). Numeric terminal evidence is recorded in
      `test_report_2026-08-26_terminal_mtp_ep_repair.md`.

15. **`predictor-layer-identity-audit`** (completed)
    - Files: `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`
      and `tests/unit/test_moe_predictor_layer_id_semantics.py`.
    - Boundary: preserve separate `pipeline_stage` and global `layer_id`
      identities through the public predictor, internal execution-time probe,
      attention probe, and terminal MTP hook. Keep the internal layer-zero
      default only for legacy callers without a real global layer identity.
    - RED: assert that a non-zero public `layer_id` reaches the internal method
      and terminal hook; the pre-fix internal call omits the keyword.
    - GREEN: use the explicit layer identity for attention and terminal calls;
      do not derive it from stage identity or add a fallback wrapper.
    - Evidence: the RED regression observed `KeyError: 'layer_id'`; the focused
      layer/terminal/structural subset passed `32` tests after the repair.

16. **`all-to-all-structural-MTP-audit`** (completed)
    - Boundary: audit the existing EP>1 payload admission and local
      structural-MTP registry/config coverage without adding a caller-side
      guard or guessed configuration asset.
    - Evidence: aggregate dispatch and combine both fail at payload
      construction without `EPLaneWorkload`; configs load with layer counts
      `48/2/20`, while two missing JSON paths fail explicitly.

17. **`pdd-attention-only-identity-repair`** (completed)
    - Files: `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`
      and `tests/unit/test_sklearn_disaggregation_execution_time_predictor.py`.
    - RED: call the real public attention-only path with a non-zero global
      `layer_id` and assert the attention lookup receives it; observe the
      current hard-coded-zero failure.
    - GREEN: add one explicit private-helper parameter and forward the public
      identity. Preserve the public signature, stage/communication behavior,
      and the compatibility default for callers without an identity.
    - Forbidden: stage-to-layer inference, fallback scaling, wrapper layers,
      or unrelated predictor refactoring.
    - Evidence: RED observed `assert 0 == 17`; GREEN passed the focused
      disaggregation/layer-identity/terminal-MTP/structural-MTP matrix with
      `64 passed in 2.77s`.

18. **`raw-width-conservation`** (completed)
    - File: `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`.
    - RED: explicit physical lane with source `Batch.total_num_tokens=5`,
      compute-effective width `8`, and `router_topk=2` must conserve `10`
      assignments; the pre-fix fallback expected `16`.
    - GREEN: use the source batch's physical width for conservation while
      leaving compute-effective lookup helpers unchanged.
    - Evidence: RED observed `allocated 10, expected 16`; the typed predictor
      contract module passed `28` tests after the correction.
    - Commit: one production/test sub-step after the focused matrix is rerun.

19. **`EP>1-aggregate-admission`** (decision resolved; implementation next)
    - Decision: narrow A'. After concrete predictor classification marks a call
      as routed MoE, EP>1 requires an `EPLaneWorkload` in both dummy and
      non-dummy modes.
    - Boundary: add one protected helper in the shared MoE predictor and call
      it from the MONOLITHIC and disaggregation public predictor seams. The
      helper runs before dummy return, measurement activation, and any model,
      attention, routing, or communication lookup.
    - Preservation: `DECODE_ATTN`, explicit dense/attention-only calls,
      mixed-layer aggregate semantics, EP=1, valid lanes, and zero-routed lanes
      retain their existing behavior. The communication payload builder keeps
      its final descriptor invariant check.
    - RED/GREEN: focused tests must prove missing-lane dummy and non-dummy
      calls fail with zero lookup calls, while the preserved cases continue to
      pass. No synthetic lane, raw-map inference, caller-side duplicate guard,
      or scaling factor is permitted.
    - Dependency: this sub-step follows the completed raw-width conservation
      repair and precedes the final focused matrix and PR20/PR21 audit.

20. **`focused-report-and-commit`** (pending admission GREEN)
    - Update the task report with the layer-identity evidence, run the fresh
    combined A' matrix, compileall, documentation gates, and `git diff --check`.
    - Commit each coherent production/test/docs sub-step separately before the
      local PR20/PR21 merge-readiness audit. No remote operation is included.

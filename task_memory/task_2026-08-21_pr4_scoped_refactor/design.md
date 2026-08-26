## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-26 | Implemented the PDD attention-only layer-identity propagation contract after the focused RED/GREEN gate. |
| 2026-08-26 | Added the PDD attention-only identity seam contract and the EP>1 aggregate/structural-MTP audit boundaries. |
| 2026-08-26 | Added the global-layer versus pipeline-stage identity invariant and the predictor propagation contract. |
| 2026-08-26 | Closed the terminal MTP physical-lane seam with a shared one-layer attention probe and lane-wise barrier aggregation. |
| 2026-08-26 | Defined the terminal MTP overshoot physical-lane hook and its shared-phase/lane-barrier aggregation contract. |
| 2026-08-26 | Recorded the pre-implementation audit boundary for the remaining typed-lane, predictor, communication, MTP, and trace changes. |
| 2026-08-26 | Documented the verified MONOLITHIC initial-decode scheduler frontier boundary. |
| 2026-08-26 | Added the approved canonical MTP/MoE token ledger and shared compute-contract repair design. |
| 2026-08-25 | Added the implementation audit separating the physical MoE MTP lane seam from the existing generic MTP shape replay. |
| 2026-08-25 | Closed aggregate lane-domain validation and documented the bounded scalar compatibility path. |
| 2026-08-25 | Froze the post-PR17 A' typed-lane architecture, physical/runtime ownership boundary, EP symmetry, zero-lane semantics, and pure MTP contract. |
| 2026-08-22 | Added the standard-attention Option-B contract: target-local physical filtering, mandatory discard warnings, retained-value union, and all-target fail-fast. |
| 2026-08-22 | Recorded the two confirmed PR #20 gaps between the selected design and the current one-feature MoE/MTP TP-consumer implementation. |
| 2026-08-22 | Added the maintainer-selected universal exact-row producer policy for PR #20. |
| 2026-08-21 | Updated the design record from pre-implementation gate language to the verified scoped implementation and explicit residual-data boundaries. |
| 2026-08-21 | Recorded option A for the MoE EP sampling envelope and the per-model CLI resolution. |
| 2026-08-21 | Completed the optional pure MoE feature extraction and documented the enforced staged import allowlist. |
| 2026-08-21 | Recorded the initial evidence-backed target design and seam decision gate. |
| 2026-08-21 | Replaced the default adapter design with a staged dependency boundary and reuse of the existing unified registry. |
| 2026-08-21 | Confirmed `operator_query_binding` / `bind_operator_query` and the staged dependency boundary. |
| 2026-08-21 | Applied the deep-module review to the binding seam and recorded the function-level lookup ownership. |

# Design - Scoped Lookup and Boundary Refactor

## Design intent

Keep the first PR deep in one problem: a runtime query must be bound to a
validated, explicit registry declaration even when a finite prediction table
does not contain the requested key. The PR should not become a second runtime
model, a profiling publication system, or a generic data-contract framework.

## Canonical MTP/MoE token ledger (approved 2026-08-26)

The target-embedded MTP path uses one physical-width ledger and several
phase-specific projections. The projections are intentionally different and
have distinct owners:

| Projection | Definition | Owner | Consumers |
|------------|------------|-------|-----------|
| `planned_draft_tokens` | Reserved speculative slots selected before target verification | MTP runtime/config and scheduler metadata | admission, block reservation, active-block selection, acceptance/outcome bookkeeping, metrics |
| `verify_tokens` | `1 + planned_draft_tokens` for each target-embedded request | MTP runtime outcome and `SpecDecodeBatchMetadata` | target attention/verification and structural replay shape |
| `Batch.num_tokens` / `Batch.total_num_tokens` | Scheduler-visible physical rows for the current forward; for the ordinary target-embedded path it equals the verification width | scheduler and `Batch` | attention, compute shape, pre-routing MoE gating, routing materialization |
| `total_routed_assignments` | `routing_token_count * router_topk` after gating | `LayerEPWorkload` materializer | dispatch, expert compute, combine, EP conservation/barriers |
| `EPLaneWorkload.routed_token_count` | Assignment count for one physical EP lane | typed lane descriptor | lane-local grouped GEMM, shuffling, all-to-all payloads, lane phase timing |
| acceptance outcome | accepted/rejected/committed counts after target verification | MTP runtime/request progression | next scheduling iteration and request metrics |

For a step with `planned=[2,1]` and `router_topk=2`:

```text
verify width             = [3,2]
pre-routing forward rows = sum([3,2]) = 5
router assignments       = 5 * 2 = 10
```

`planned_draft_tokens` is not an additional physical token vector once the
verification width has been placed in `Batch.num_tokens`. The shared
`Batch.get_effective_total_tokens_for_compute()` helper therefore returns the
existing physical width for target-embedded MTP, except where an explicit
AFD/CUDA Graph metadata contract supplies a padded compute shape. The transfer
helper keeps its existing raw physical payload rule.

The structural MTP replay uses the complete `verify_tokens` vector for the
first target-verification block. Rejection is an outcome observed after that
forward and cannot be subtracted from the already executed target shape. The
generic replay remains the narrow, scheduler-independent shape adapter defined
in SCOPE-026; the physical MoE portion continues to use the pure typed
`EPLaneWorkload` phase seam.

The conservation boundary is explicit:

```text
Batch.total_num_tokens
    -> routing_token_count
    -> routing_token_count * router_topk
    -> sum(EPLaneWorkload.routed_token_count)
```

No consumer may substitute the assignment count for the pre-routing count, or
add planned metadata to a batch width that already contains verification rows.

### Layer identity propagation invariant

`pipeline_stage`/`stage_id` and `layer_id` have separate owners and meanings:

| Identity | Meaning | Owner | Valid consumers |
|----------|---------|-------|-----------------|
| `stage_id` / `pipeline_stage` | Pipeline partition used for stage-local communication and scheduling | scheduler/predictor stage context | stage communication, stage boundary and pipeline overhead |
| `layer_id` | Global transformer-layer identity used for mixed-layer classification, routing lookup, attention rows, and terminal MTP policy | public predictor call and model configuration | `is_moe_layer`, routing materialization, attention prediction, terminal-row hook |

The propagation contract is explicit:

```text
predict_stage_execution_time(layer_id)
    -> _get_moe_tokens_input(layer_id)
    -> _get_execution_time_internal(layer_id)
        -> predict_attention_layer_time(layer_id)
        -> _get_mtp_terminal_overshoot_time(layer_id)
```

The internal method keeps a default `layer_id=0` only for legacy internal
post-attention callers that genuinely lack a layer identity. A caller that has
the global identity passes it explicitly. The implementation never derives a
layer from `pipeline_stage`, a name prefix, or an entity type. This preserves
the PR17 stage/KV identity boundary while making layer-aware predictor paths
correct for non-zero and mixed layers.

### MONOLITHIC initial-decode frontier boundary

The MONOLITHIC scheduler has one explicit boundary projection that precedes
ordinary batch physical-width accounting. `Request.on_batch_end()` grants the
first output token when the final prefill callback completes, so
`num_processed_tokens` can be `num_prefill_tokens + 1` before a decode batch is
admitted. `_get_scheduler_num_computed_tokens()` deliberately reports the
prefill frontier at that point. `_get_request_next_num_tokens()` consequently
returns `max(planned_drafts, 1)` for the first target-embedded MTP admission,
which schedules the still-unadmitted draft slots and avoids replaying the
already-granted boundary token.

This is a scheduler-frontier rule, not a second physical token ledger. Once the
frontier advances, the helper returns the ordinary `1 + planned_drafts`
verification width. Batch formation, compute lookup, MoE gating, and lane
materialization continue to consume the physical `Batch.num_tokens` width and
never infer it from this admission rule. The direct probe recorded in
`issues.md:SCOPE-028` is the regression contract for this distinction.

## Target ownership

```text
profiling generators  --writes-->  benchmark CSV files
                                      |
                                      v
runtime path: cache lookup -> raw CSV load/fit on miss -> predictor model
                                      |
                                      v
simulation query -> bind_operator_query -> validation -> exact value/cache/model prediction
```

The data flow is intentionally one-way, but the Python import boundary is
staged rather than absolutist. Runtime must not invoke a benchmark generator,
GPU measurement wrapper, or profiling CLI. Side-effect-free CSV schema and
validation helpers may remain shared temporarily when moving them would touch
both producer and consumer and would create a new contract layer. The
non-KV-cache-memory and MTP structural-config paths remain separately named
exceptions and must not become implicit exceptions for compute timing.

## Query contract

Every profile-backed query uses the same semantic sequence:

1. Resolve an exact registered operator and active measurement identity.
2. Validate feature schema/order, finite numeric values, physical bounds, and
   relational constraints.
3. Return a validated exact measured value when present.
4. Return a validated process-local runtime-cache value when present.
5. Pass the original validated feature vector to the canonical estimator.
6. Validate that the estimator result is finite and non-negative.
7. Store only that model prediction in the process-local runtime cache.

No nearest row, clamp, default, silent skip, or runtime write-back to profiling
CSV is allowed. Optional CPU/PP missing-profile zero remains an explicit owner
policy, not a generic prediction fallback.

## Exact-row producer policy

Both unified runtime training/cache entry points persist scalar-numeric
measured rows by default. The estimator cache therefore carries
`_frontier_exact_lookup` across process boundaries, including when an older
cache artifact is first loaded and rewritten.

The default is authoritative because measured-row precedence is part of the
shared lookup contract rather than a family-specific optimization. Ordinary
Linear, standard Attention, one-feature MoE, mixed Attention, MLA, CPU, and
KV-cache producers follow the same mechanism without separate admission
lists. The parameter remains available as an explicit opt-out for a future
producer whose feature key is not representable as a scalar numeric tuple;
the current exact-key builder fails fast for such a schema.

This policy changes only persisted estimator metadata. It leaves unmeasured
legal queries on the existing process-local runtime-cache/model path and does
not write profiling CSVs.

**Current PR #20 gap:** The two one-feature MoE producer call sites still pass
`persist_exact_lookup=len(feature_cols) > 1`, overriding the unified
default. The selected design therefore remains the acceptance target, while
`SCOPE-014` stays open until those producers retain the measured rows through
a cache round trip.

## Operator query binding decision

The current source already has a useful unified registry
(`frontier/operators/spec.py`, `frontier/operators/families.py`) and
family-specific mapping APIs. The selected seam is named
`operator_query_binding`, with `bind_operator_query` as its public operation.
The name describes the domain action; exact membership and fail-fast behavior
are invariants of that operation rather than part of the module name.

### Selected design: reuse the unified registry with `bind_operator_query`

Keep `OperatorSpec`, `OperatorFamilySpec`, and the existing attention/MoE
mapping APIs as the only operator source of truth. Add a narrow query-binding
operation in the existing `frontier/operators/binding.py` surface first. It
must:

- resolve a registered physical name to its declared family and
  `profiling_name()`;
- apply `tp_mode` from the registered `OperatorSpec`;
- handle timing-only aliases and many-to-one CSV columns through one explicit
  mapping table, with collision checks;
- fail fast for unknown names, unsupported owner states, and ambiguous aliases.

The initial API should be `bind_operator_query(...)` and, if a return object is
needed, it should be named `OperatorQueryBinding`. The implementation must not
create a second catalog. A separate `operator_query_binding.py` file is only a
later extraction if the existing `binding.py` becomes demonstrably overloaded.

The operation is deliberately narrower than a predictor query validator. It
does not read CSV, fit a model, inspect either runtime cache, or validate the
numeric feature vector. Those remain in their existing owners.

### Seam depth and interface

`bind_operator_query` earns a seam only if callers can stop carrying family and
TP classification rules themselves. Its intended interface is one small
operation with a result that contains the declarative facts a predictor or
trainer needs:

```text
bind_operator_query(operator_name, context) -> OperatorQueryBinding
```

The result may expose `family`, `operator`, `profiling_name`, `tp_mode`, and an
explicit physical-to-timing mapping. The interface also includes these
invariants: membership is exact, aliases are collision-checked, unsupported
owner states raise, and no profiling/predictor/cache side effect occurs. A
caller should not need to know whether the result came from a generic family,
an attention family, or an explicit alias entry.

The module is intentionally not a pass-through wrapper around
`get_operator_family`. If the implementation cannot centralize alias conflict
checks and TP-policy admission behind this interface, the seam is too shallow
and the change must stop for a new scope decision. Conversely, the interface
must not grow CSV paths, feature vectors, sklearn estimators, or runtime-cache
handles; those would make it a second predictor contract rather than a deep
registry seam.

This keeps the extension gate close to one registry declaration and avoids
coupling the runtime registry to sklearn classes, CSV paths, or benchmark
execution. It also directly addresses the current prefix branches without
adding an adapter object that profiling does not consume.

### Contingency: minimal bridge only if the binding cannot express a real owner gap

If a concrete current-main case proves that a physical operator maps to a
timing owner that cannot be represented by the existing `profiling_key`, family,
or explicit alias table, introduce the smallest bridge for that case only. The
bridge must be derived from the existing registry, must not become a second
operator catalog, and must be approved with a file/line-level migration
estimate before implementation.

### C. Independent profiling-facing adapter (deferred, not default)

An independent adapter is not warranted by the current evidence. It would
duplicate the registry's names, TP policy, and profiling keys, force a second
declaration for every operator, and add synchronization checks without changing
the CSV-only producer/consumer boundary. It may be revisited only after a
measured owner mismatch survives option A. The historical R-025 adapter
proposal depended on a separate D24/N5 timing catalog that was added only in the
old scope-crept worktree and is absent from current `origin/main`; importing
that catalog would expand this PR rather than solve its lookup root cause.

**Decision status:** Confirmed by the maintainer on 2026-08-21. The staged
dependency boundary and ordinary `operator_query_binding` /
`bind_operator_query` seam are implemented. The PR #20 review reopened one
target-embedded MTP TP-membership gap: predictor, shared manager, and Linear
trainer still use a local two-name set instead of
`get_target_embedded_mtp_linear_ops()`. An independent profiling-facing
adapter remains deferred because no owner gap was found. The direct lookup,
registry, sampling, and PDD lifecycle evidence is archived in the Wave E
report. Producer-side optional metadata and canonical measured rows remain
explicit follow-up work.

## Dependency ownership rules for this PR

1. **Benchmark implementation:** `frontier/profiling` GPU wrappers, benchmark
   runners, and CLI entry points remain producer-only. Runtime and training
   code must not invoke them.
2. **CSV consumption:** Existing side-effect-free schema/validation helpers may
   be imported by predictor code as a temporary shared implementation. They
   must not import a benchmark runner or write CSV files.
3. **Runtime-only feature construction:** The pure MoE load-imbalance feature
   object lives in `frontier/moe_load_imbalance.py`. The profiling input module
   re-exports it for compatibility, while sampling configuration stays in
   profiling. This is a feature-derivation module, not a data-contract layer.
4. **Named exceptions:** Non-KV-cache-memory estimation and MTP structural
   config retain their current imports until separate ownership work is
   approved. Their imports are isolated and explicitly tested.

The staged boundary is enforced by
`tests/unit/test_profiling_runtime_boundary.py`: predictor modules may import
only the current CPU/PP CSV/schema helpers, and training modules may not import
profiling implementation code. Benchmark runners, GPU wrappers, and profiling
CLI modules remain producer-only.

This ownership rule is the smallest change that honors the file-based data
boundary while avoiding a broad, duplicate schema migration.

## Explicit owner states

- Ordinary Linear/Attention/MoE entries are profile-backed and use their owner
  schemas/resolvers.
- Registered MLA entries remain explicit Attention-owned entries even if an
  exact timing identity is not available; an exact identity request fails with
  a clear error rather than a prefix-derived family.
- Communication entries are CC-backend-owned; missing backend configuration
  fails fast.
- CPU auxiliary entries retain their intentional optional-zero behavior.
- Unknown names are not assigned a generic `compute`/`attention`/`moe` family.

## Profiling sampling model

The profiler owns deterministic sample planning and CSV output. Each timing
family has its own multi-feature envelope. The envelope is built from actual
runtime/configuration maxima and extends through the first legal canonical
anchor above those maxima, bounded only by physical capacity. Boundary,
single-axis, pairwise, and operator-specific risk points are selected without
an unbounded Cartesian product. Derived values are computed from executable
joint inputs, not independently fabricated columns.

For MoE expert parallelism, the runtime topology rule is divisibility of
`total_expert_num`. The selected policy is to make every positive divisor part
of the default profiling domain. `frontier/profiling/moe/moe_input.py` owns the
pure resolver used by the profiling helper, while `frontier/profiling/moe/main.py`
resolves omitted CLI selections separately for each model. Explicit CLI lists
remain a bounded operator choice and are rejected when they are not legal
divisors. The profiler's representative local-rank implementation does not
require an EP-sized GPU allocation; actual CSV measurement for newly exposed
values remains an offline operational step.

### Standard attention explicit-KV policy

The standard (non-mixed) attention path uses two different length concepts:

- `max_seq_len` bounds the automatic profiling envelope. Automatic decode
  points therefore stop at `max_seq_len - 1`, because the current decode token
  is counted separately.
- `max_model_len` bounds the runtime context. An explicitly requested decode
  KV value is legal when `kv_cache_size + 1 <= max_model_len`, even when it is
  above `max_seq_len - 1`.

Option B applies physical capacity after that logical runtime validation. Each
`(model, tensor_parallel_size)` target filters its own standard-attention rows
against its computed KV capacity. A target-local drop does not force the user
to edit the request when another selected target can measure the value. The
filter emits a `RuntimeWarning` before worker dispatch with the dropped KV
values, dropped combination count, model, TP size, physical capacity, and
retained values. The main process also prints the requested/retained union and
all target capacities.

The retained explicit values are unioned across targets. The planner raises
`ValueError` only when a requested explicit value is absent from that union,
and the error lists the missing values and every target capacity. This keeps
explicit illegal runtime values fail-fast while avoiding a user-facing
parameter-repair loop for a target-local physical shortfall. The contract is
limited to standard attention in this PR; mixed and true-mixed block-aware
capacity semantics remain separate audit items.

## Why this is scoped

The design does not move canonical data, repair old aliases, introduce a
publication journal, or redesign every profiling producer. It fixes the shared
admission/query boundary first, then leaves data expansion and publication to
separate PRs with their own evidence.

## A' typed EP lane design (post-PR17 merge)

### Ownership graph

```text
LayerEPWorkload (aggregate routing; pure materializer)
        |
        +-- canonical lane constructor --> EPLaneWorkload (one physical lane)
        |                                      |
        |                                      +--> EPBatchGroupPlan
        |                                      |       |
        |                                      |       +--> EPBatchGroup
        |                                      |                 |
        |                                      |                 +--> read-only map projection
        |                                      |
        |                                      +--> predictor / CC payload consumers
        |
        +-- participant ids, barriers, events, stage identity --> scheduler
```

`LayerEPWorkload` owns aggregate routing and the expert-to-lane partition. It
is the only owner allowed to construct a physical lane descriptor from a
global map or from its materialized per-lane map. `EPLaneWorkload` is immutable
and contains no scheduler lifecycle state. `EPBatchGroupPlan` and
`EPBatchGroup` carry the descriptor by value/reference and expose
`per_expert_tokens` only as a descriptor-backed compatibility projection.

The scheduler remains the owner of `source_batch_ids` as admission provenance,
`schedule_epoch`, AFD stage index, admission tickets, stale-wave identity,
metrics operation identity, event ordering, and participant/barrier state.
Those values may be used to validate a call at the scheduler boundary, but
they do not become predictor feature or descriptor fields.

### Physical descriptor contract

`EPLaneWorkload` has one canonical constructor and the following invariants:

- `ep_id` is a member of the materialized participant topology.
- `moe_expert_parallel_size` and `total_expert_num` are positive integers and
  `total_expert_num % moe_expert_parallel_size == 0`.
- `local_expert_width` is derived from topology, never from map length.
- `owned_expert_ids` is the complete canonical ownership interval for `ep_id`.
- `local_token_counts` is a fixed-width dense vector in ownership order;
  omitted sparse entries are explicit zeros and out-of-owner IDs fail fast.
- `routed_token_count` is the sum of the dense local vector and
  `router_topk` remains the source routing parameter.
- The descriptor is frozen and contains no mutable scheduler/entity identity.

The aggregate constructor validates global-map completeness and token
conservation before creating all physical lanes. A caller with a local map
must provide the descriptor produced by that constructor; a partial map with
no lane identity fails before predictor/model access. No `bool`/numeric
coercion or `validate_expert_width=False` escape hatch is permitted.

### EP=1 and EP>1 symmetry

Both modes call the same constructor and predictor interface. For `EP=1`, the
participant set is `(0,)`, local width equals total width, and dispatch/combine
payloads are zero. For `EP>1`, every lane is materialized with its local width,
and lane phase aggregation uses the same complete-participant barrier. The
only conditional behavior is the physical collective cost; no caller branches
on raw map shape or entity class.

### Zero-routed lane semantics

A zero-routed lane is still a physical participant. It remains in dispatch,
combine, post-combine, and phase-barrier accounting. Its routed-dependent
compute/shuffling phase returns `0.0` after descriptor validation and performs
no positive-load model lookup. Shared pre-routing or fixed per-lane phases use
the source workload and collective contract. This preserves both physical
barrier correctness and the profiling domain.

### Predictor and communication contracts

`predict_moe_layer_time()` accepts the typed descriptor/plan path and computes
lane phases from the descriptor. It does not split a global map or infer local
width. `predict_stage_execution_time()` keeps its existing dense-compatible
signature; stage callers obtain the descriptor from the batch/entity context.
Base, concrete, disaggregation, mock, and test interfaces change together.

Communication payload helpers consume the descriptor-backed projection through
one helper; they do not use `isinstance(EPBatchGroup)` as their domain test.
The projection remains a dict-shaped view only while downstream consumers are
migrated, and no second mutable map is stored.

### Scalar compatibility boundary

The typed descriptor is mandatory for every physical, load-aware EP lane path.
The remaining integer token path is deliberately narrower and does not accept
or reconstruct a raw expert-token map:

- Standard one-feature MoE lookups use the source batch's pre-routing token
  count. This keeps the existing scalar profiling schema usable for ordinary
  non-lane callers whose model has no load-imbalance feature columns.
- Shared-domain communication uses the source batch's effective pre-routing
  token count when no physical lane is present. This is the all-reduce contract;
  it does not describe a local expert map.
- EP lane entities, load-aware shuffling/grouped-GEMM, dispatch/combine payloads,
  and EP>1 aggregate calls that need local load all require an
  `EPLaneWorkload`. A regular EP>1 aggregate fails before model or backend
  access until the scheduler materializes its lanes.
- Raw mappings and partial local maps remain invalid predictor inputs. The only
  mapping-shaped value exposed by an entity is the read-only compatibility view
  derived from its descriptor.

This boundary preserves the existing one-feature/shared-domain callers while
preventing a scalar fallback from silently selecting a physical expert domain.

### Pure MTP phase path

MTP decoder timing uses the materialized lane descriptor and a pure phase
calculation seam. It retains the existing phase decomposition and lane-wise
`max()` semantics without constructing synthetic `Request`, `Batch`, or
`EPBatchGroup` entities. MTP receives only physical descriptor data and
explicit model/config inputs; scheduler identity remains absent.

#### Implementation audit boundary (2026-08-25)

The physical MoE/EP part of this contract is implemented by
`predict_moe_lane_phase_times()`: it receives the source batch plus one
immutable `EPLaneWorkload` and does not construct an `EPBatchGroup` or copy
scheduler identity. The surrounding generic target-embedded MTP replay still
uses a short-lived block-shape `Batch` and copied request progress to reuse the
existing attention/token-shape predictor protocol. That object never enters a
scheduler queue and is not a physical lane descriptor.

The broad “no synthetic `Batch`” sentence above belongs to the rejected full
pure-MTP alternative. The approved narrow A' contract is the “pure physical
MoE MTP lane phase”; the generic shape adapter remains an explicit,
scheduler-independent compatibility boundary. No shared MTP interface
expansion is in scope.

### Extension gate

Adding the next EP-aware caller requires consuming the existing descriptor
accessor only. Adding a new topology or expert-count policy changes the
materializer/registry declaration and its contract tests, not scattered
predictor branches. Any implementation that needs a second raw-map owner,
caller-specific width flag, or synthetic scheduler entity fails the A' gate.

## Implementation handoff audit (2026-08-26)

The approved design is now the source of truth for the remaining dirty
implementation slice. The audit separates the work into these ownership
boundaries:

- `moe_ep_workload.py` validates and materializes the immutable physical lane;
  scheduler/entity code only transports it and owns lifecycle identity.
- Predictor code consumes a descriptor for every physical or load-aware lane;
  the scalar path remains limited to non-lane pre-routing lookups.
- Communication builders receive the descriptor through `CommPayloadContext`;
  they may expose a dict-shaped value only at an explicit output boundary.
- Physical MoE MTP phase calculation calls the typed lane seam. Generic MTP
  block-shape replay remains the already-approved scheduler-independent adapter.
- The remaining production trace helper is the only known raw-map input site;
  its caller must pass the descriptor and create the map projection inside the
  logging/serialization boundary.

The implementation must preserve the post-PR17 scheduler ownership of stage,
KV, admission, barrier, and stale-wave identity. Any change that introduces a
second mutable map owner, reconstructs a physical lane from `len(map)`, or
uses a caller-specific token-width bypass violates this design.

### Terminal MTP overshoot physical-lane seam (approved continuation)

Terminal overshoot replay has two different responsibilities that remain
separate:

1. The generic target-embedded MTP adapter owns terminal token vectors,
   copied request progress, and speculative metadata. It continues to create a
   short-lived `Batch` because the existing attention and metadata predictor
   contract consumes that shape. The object never enters scheduler admission,
   an event queue, or an EP participant set.
2. The MoE predictor owns physical EP timing. It receives a source batch and
   explicit `stage_id`/`pipeline_stage`, `cluster_type`, `layer_id`, and
   `num_layers`; it obtains physical lanes only from
   `LayerEPWorkload.lane(ep_id)`.

The terminal-row hook follows this composition:

```text
terminal synthetic Batch
    -> dense/default terminal-row hook
    -> MoE override for an actual EP>1 MoE layer
         -> attention-only stage prediction (shared attention, pipeline, CPU)
         -> LayerEPWorkload materialization
         -> predict_moe_lane_phase_times(one EPLaneWorkload per participant)
         -> max each of the five physical phases
         -> scale only per-layer phase work by num_layers
```

The attention-only result supplies the shared attention model time, one
pipeline boundary, and the batch-level CPU overhead. The five phase values
(`pre-dispatch`, `dispatch`, `routed compute`, `combine`, `post-combine`) are
already the canonical physical decomposition. Their lane-wise maxima model the
dispatch/combine barrier and preserve zero-routed lanes as participants. A
zero-routed lane therefore makes no positive-load shuffling or grouped-GEMM
lookup, while shared pre/post phases still execute.

The hook returns the same total-time unit as the existing generic terminal
path. It does not fabricate an `EPLaneWorkload`, pass a raw expert map, create
an `EPBatchGroup`, or duplicate the phase decomposition. `EP=1` keeps the same
typed lane contract and only observes zero physical collective cost; dense and
non-MoE layers use the default hook. `stage_id` remains the pipeline-stage
identity and `layer_id` remains the predictor routing-layer identity; the
implementation never substitutes one for the other.

The shared helper deliberately asks the attention-only predictor for
`num_layers=1`. That result contains one shared attention scope, one pipeline
boundary, and one batch-level CPU/overhead scope. The terminal caller then
multiplies only the five physical lane maxima by its requested `num_layers`.
The structural MTP caller consumes the same decomposition with
`model_time_ms`, preserving its existing proposer accounting; the terminal
caller consumes `total_time` so its CPU/overhead contract matches the generic
terminal path without multiplying that overhead by layer count.

## Boundary audit and PDD identity seam (2026-08-26)

The existing communication operator contract remains the only admission
boundary for routed EP traffic. An aggregate `CommPayloadContext` is valid for
attention-only, dense, and shared-domain work; an EP>1 routed all-to-all
dispatch or combine payload requires an explicit `EPLaneWorkload` and fails at
payload construction when it is absent. Callers do not add a second guard or
reconstruct a lane from a raw map.

The structural MTP registry audit is a configuration-coverage check, not a new
runtime policy. The three locally present configs load their declared layer
counts (`48`, `2`, and `20`) and identify layer zero as MoE. Two registry names
have no local structural JSON and continue to fail explicitly with the existing
missing-path error. The repair does not invent configuration assets or broaden
the registry.

For PDD shared-domain attention-only prediction, the public
`predict_stage_execution_time()` `layer_id` is the global transformer-layer
identity. `_predict_attention_only_stage_execution_time()` receives that value
explicitly and forwards it to `predict_attention_layer_time()`. Its optional
`layer_id=0` compatibility default applies only to direct legacy helper calls
that have no real identity; `stage_id` remains pipeline placement and is never
used to derive a layer.

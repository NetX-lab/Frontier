## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
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

### Pure MTP phase path

MTP decoder timing uses the materialized lane descriptor and a pure phase
calculation seam. It retains the existing phase decomposition and lane-wise
`max()` semantics without constructing synthetic `Request`, `Batch`, or
`EPBatchGroup` entities. MTP receives only physical descriptor data and
explicit model/config inputs; scheduler identity remains absent.

### Extension gate

Adding the next EP-aware caller requires consuming the existing descriptor
accessor only. Adding a new topology or expert-count policy changes the
materializer/registry declaration and its contract tests, not scattered
predictor branches. Any implementation that needs a second raw-map owner,
caller-specific width flag, or synthetic scheduler entity fails the A' gate.

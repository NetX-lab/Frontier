## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Completed the read-only cleanup-first and functional split analysis for seven large critical modules; recorded two additional bounded cleanup proposals and preserved-state constraints. |
| 2026-09-17 | Appended S6 exact raw-type profiling policy proposal and the existing singleton GDN variant source; no implementation or test execution. |

# Large Critical Module Review

## Scope and method

- Fixed main: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`; fixed candidate: `c288a19f59bec09529ee18d782fa57218da2c781`.
- Main agent owns implementation and tests. This sidecar writes only this report; no production/test edits, suites, remote changes, or subagents.
- Plan: `fixed-diff inventory -> reconcile recorded cleanup -> inspect remaining added hunks and owners -> bounded cleanup proposals -> correctness-preserving split sequence`.
- Source of standards: `AGENTS.md`, Development Gates / Python Module Size and Naming, and the current behavior-preserving cleanup request. Code-review supplies the Standards lens; codebase-design supplies the functional seam analysis. File-based planning is kept here under the single-file authorization.
- Reviewed scope: sklearn base predictor, MoE predictor, disaggregation predictor, shared prediction manager, metrics store, vLLM V1 scheduler, and aggregate configuration module.
- Status: completed read-only review. More than 2,000 lines triggers cleanup-first plus split analysis, not immediate extraction or a repository-wide rewrite.

## Inspection notes

- At inspection start HEAD was `0c4d59bf66f62692ff4f477c8839c37b45219dd7`; A1 production/tests edits were present and left untouched. All issue locations will use frozen-candidate lines unless explicitly marked otherwise.
- Existing records already cover MoE routing construction, GDN admission/state/feature invariants, manager measurement registries/path handling, metrics ownership, and attention identity consolidation. They will not be relabeled as new findings.
- An optional `review.md` was not present; the existing `metrics_review.md`, `refactoring_record.md`, `standards_review.md`, and `attention_review.md` provide the actual review record. No suite was run.
- At completion inspection, HEAD was `6159532511b9ac45a31dc61ea03bf124a60dd9df`; the tracked worktree was clean before this report update. Main's recorded A1/P1f implementation and verification supersede the initial dirty-state observation. This sidecar did not implement or independently verify those changes.

## Summary

All seven files already exceeded 2,000 lines on main. Their size alone is not a new candidate defect: the frozen candidate reduces the base and disaggregation predictors. The important remaining work is to clarify ownership before moving functions. Existing P1 cleanup and S2/S3/S8/M1–M5 findings account for most candidate-added complexity.

Two additional opportunities remain: remove the manager-local hybrid/MLA classification shortcut in favor of the existing runtime-family resolver (L1), and remove the unused aggregate degree of freedom from private MoE/disaggregation layer prediction (L2). Neither requires new modeling semantics. L2 must retain real stage width for MTP and the public stage interface.

Documented-standard concern: L1 duplicates canonical family policy. L2 is a maintainability judgment about a misleading private interface, not an observed numerical failure. No additional confirmed defect was found in the candidate scheduler lifecycle or large config additions beyond already-recorded issues.

Recommend continuing the planned fidelity campaign without a broad split. The functional split sequences below satisfy the analysis requirement; implementation is a separately scoped step after cleanup/characterization, not a prerequisite introduced by this review.

## Frozen-diff inventory

All paths below are relative to the worktree root. Locations refer to `c288a19f`, not moving HEAD.

| Reviewed module | Main lines | Candidate lines | Actual added/deleted lines | Candidate scope and priority |
| --- | ---: | ---: | ---: | --- |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 8,315 | 8,263 | +157 / -209 | High: family/measurement selection, GDN dispatch, homogeneous stage reuse and stage-owned work. |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | 3,523 | 3,584 | +202 / -141 | High: routing generation/workload cache, per-layer prediction and stage-local attention reuse. |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | 3,058 | 2,985 | +107 / -180 | High: shared routing generator, physical-layer role prediction and once-only overhead/communication. |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | 4,570 | 4,646 | +176 / -100 | Initialization/artifact critical: DEVICE_EVENT registries, input paths, GDN artifact loading, hybrid family routing. |
| `frontier/metrics/metrics_store.py` | 5,270 | 5,640 | +403 / -33 | Reporting critical: layer-specific traces, stage operation metrics, component ledger and lane evidence. |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | 5,073 | 5,139 | +81 / -15 | High: constructor-owned GDN slots, KV/GDN admission and release, queue-commit ordering. |
| `frontier/config/config.py` | 5,663 | 5,725 | +62 / -0 | Initialization critical: GDN artifact path, ReplicaConfig and SimulationConfig admission guards. |

Reproduction: `git log 0515589ac7f49ac5288a5f55b0ce38b0ede29bb2..c288a19f59bec09529ee18d782fa57218da2c781 --oneline` was inspected, including stage ownership/reuse, GDN admission, routing and measurement-path commits. Use `git diff --unified=3 0515589ac7f49ac5288a5f55b0ce38b0ede29bb2...c288a19f59bec09529ee18d782fa57218da2c781 -- <reviewed-path>` for each row; counts use the same range with `--numstat`. Main/candidate file sizes use `git show <revision>:<path> | wc -l`.

## Reconciliation with existing findings

- `refactoring_record.md` / `progress.md`: P1a–P1f already cover component ownership, memory/model binding, required GDN fields, MoE constructor/routing dispatch, free-slot allocation and raw Qwen identity. Do not reopen these as new issues.
- `standards_review.md`: S2 covers manager GDN path/default duplication; S3 covers lazy constructor-owned measurement registries and implicit platform fallback; S8 covers required config fields read defensively. These remain the first cleanup steps for their owners.
- `metrics_review.md`: M1–M5 cover obsolete trace proxies, dense metadata copies, registry-driven FFN/GDN operator enumeration and repeated EP context construction. Its S1 residual-op metadata discrepancy still requires main's isolated reproduction/RCA; this review does not claim it resolved or change its expected outputs.
- Existing main debt such as large PP/MTP helpers, broad scheduler methods and config flattening is considered for split sequencing only. A mixin or forwarding wrapper that still needs the entire parent's private state would not improve locality.

## Additional bounded cleanup proposals

| Module / issue | Problem and root cause | Why unnecessarily complex | Refactor and reused abstraction | Preserved behavior / focused verification |
| --- | --- | --- | --- | --- |
| L1 — shared manager, `_is_mla_family`, `:2355–2368`; callers `:2115`, `:3280` | Candidate avoids the hybrid-rejecting whole-model binder by probing `get_num_gdn_layers` and returning False locally. | It recreates the model-wide full-attention selection policy already owned by `attention/model_binding.py:210`, and probes a method merely to choose a different classification path. | Keep the existing outer `model_config is None -> False` behavior for this bounded change; for supplied configs compare `resolve_runtime_attention_family(model_config).family_id` with `LATENT_MLA_ATTENTION_FAMILY.family_id`. No new helper, cache or registry. | Dense/MFA/hybrid full-attention remain non-MLA; MLA remains MLA. Compare selected training operators, structural filtering, feature/target columns and artifact identities. Use the shared-manager MLA/dense tests, hybrid family tests and GDN constructor acceptance listed below. |
| L2 — MoE `_predict_moe_layer_execution_time`, `:3298`; disaggregation `_predict_disaggregated_layer_execution_time`, `:1391` | The stage refactor retained an aggregate `num_layers` parameter inside methods now explicitly predicting one physical layer. | Every production caller supplies literal `1`; the classifier still documents a multi-layer path and disaggregation threads the constant through operator validation and logs. | Remove the private aggregate parameter in separate small steps; have each layer call use single-layer classification explicitly. Reuse `_resolve_moe_layer_classification` initially with `num_layers=1`; do not widen the initial patch into all helper signatures. Retain public stage width, actual layer ID, `include_stage_owned` and MoE `stage_num_layers`. | Preserve classifier precedence, independent routing by physical layer, once-only CPU/PP/MTP accounting and exact stage snapshots. Update the direct private oracle fixture's call shape, not its values. Use layer semantics, scaling, typed EP, attention-cache and terminal-MTP tests. |

### L1 evidence and limits

- The changed shortcut is at `:2359–2364`. Both production callers already have `replica_config.model_config`; it is not an artifact-only loader with an absent required model. Removing the reflective method probe does not require making the model globally optional or installing a fabricated GDN count.
- The canonical resolver deliberately selects the unique full-attention family for supported hybrids, while `bind_attention_family` deliberately rejects them. Its implementation (`attention/model_binding.py:210–239`) reads layer specs, not manager state, so this replacement does not recursively re-enter training or prediction. Per-layer numerical prediction must continue using per-layer binding.
- The supported Qwen hybrid fixture uses a real `BaseModelConfig` (`tests/unit/test_gdn_hybrid_e2e_increment14ab.py:67`); constructor validation excludes an all-GDN layout. The MLA manager fixture is a legitimate structural `SimpleNamespace` (`tests/unit/test_shared_prediction_model_manager_eager_attention_mla.py:40`), which the canonical homogeneous binder already accepts. Do not force that fixture to implement unrelated GDN methods.
- This equivalence is established for admitted current configurations, not arbitrary objects claiming positive GDN counts without a valid topology. Preserve invalid-input propagation; do not introduce catch-and-return-False behavior. Keeping the old None result avoids unrelated interface tightening.
- Focused paths: `tests/unit/test_shared_prediction_model_manager_eager_attention_mla.py`, `tests/unit/test_shared_prediction_model_manager_eager_attention_decode.py`, `tests/unit/test_hybrid_runtime_family.py`, `tests/unit/test_gdn_hybrid_e2e_increment14ab.py`. These are recommendations, not executions by this sidecar.

### L2 evidence and limits

- Frozen `git grep` finds the MoE private method called only by its stage method (`:3288–3292`, literal `1`). Disaggregation has two production calls (`:1376–1385`, both literal `1`) and the direct test oracle in `tests/unit/test_dense_execution_time_layer_scaling.py:260–279`, also using `1`.
- The common classifier (`sklearn_moe_execution_time_predictor.py:432–474`) uses `num_layers != 1` to bypass layer classification. That path is not selected by these stage implementations. Preserve explicit `include_moe`, `include_ffn` and model-owned `is_moe_layer` precedence; a public multi-layer stage still classifies every actual physical layer.
- Disaggregation `_predict_one_op_time` (`:1164–1191`) only validates this count and includes it in diagnostic context; it does not scale the returned time. Initial cleanup can keep its signature and pass the proven constant. Removing all repeated arguments is a later bounded patch, not justification to touch dozens of call sites before fidelity.
- Do not apply the same deletion to base `_predict_dense_layer_execution_time` (`:7906–7923`): its count defines the actual terminal-MTP stage range. MoE separately passes `stage_num_layers=num_layers` (`:3292`); that is real state, not a duplicate of the constant layer count.
- Focused paths: `tests/unit/test_moe_predictor_layer_id_semantics.py`, `tests/unit/test_dense_execution_time_layer_scaling.py`, `tests/unit/test_pd_decode_moe_layer_accounting.py`, `tests/unit/test_typed_ep_predictor_contract.py`, `tests/unit/test_attention_query_cache.py`, `tests/unit/test_mtp_terminal_overshoot_ep_replay.py`. Compare exact per-layer/stage operator and communication maps, layer IDs, snapshot counts and once-only work, not just successful return.

## Functional split analysis, after cleanup

The following are concrete responsibilities and sequencing proposals, not approved new public interfaces or immediate file moves. All seven retain substantial main-owned functionality after candidate cleanup. Their current owner is temporarily retained for the correctness reasons below; this is not a claim that their present size is ideal.

### 1. Base sklearn predictor

- Inspected responsibilities: family/measurement initialization `:379–930`; PP accounting `:1918–2240`; profile preparation/training `:2246–4157`; lookup `:4410–4615`; MTP `:5208–5862`; attention prediction `:6946–7542`; stage assembly entry `:7890`.
- Cleanup first: finish authoritative path/measurement selection reuse; retain candidate's homogeneous numerical reuse and finalized payload sharing. Do not create another family resolver or prediction cache.
- Retain current owner now because measurement activation, precision-specific model selection, per-batch overhead and stage-owned terminal replay must agree before timings are published. Extracting arbitrary helper methods risks stale active-family state or counting stage work twice.
- Proposed sequence: `canonical measurement/path inputs -> dataset-to-feature preparation -> estimator fit/cache operation -> runtime lookup -> optional later PP/MTP accounting extraction`. Put shared preparation/fit implementation under the existing predictor package and have existing training callers reuse it when contracts match; leave stage orchestration in the predictor.
- Existing owners checked: `measurement_input_paths.py`, attention/operator metadata, `BaseTrainer`, `AttentionTrainer`, `LinearOpTrainer`, `MoETrainer`. Standalone trainers have dataset/output ownership and their own cache identities; they are not drop-in replacements for runtime training. Characterize target columns, hash inputs, exact lookup and model results before sharing a fit implementation. Do not introduce a second trainer hierarchy.

### 2. MoE predictor

- Inspected responsibilities: typed admission/classification `:260–474`; routing initialization `:702–825`; workload cache `:930–981`; attention reuse `:2199–2246`; per-layer time construction `:2248`; stage orchestration `:3272`.
- Cleanup first: retain P1d's constructor/routing simplifications; apply L2 only if it fits main's current sub-step. Reuse `generate_moe_routing_ratios`, `materialize_layer_ep_workload` and `EPLaneWorkload` rather than extracting another routing service.
- Retain current owner because one stage-local attention cache belongs to one batch/stage invocation, while EP routing and integerized lane workloads vary by actual layer. Moving both into a generic global cache would merge distinct identities.
- Proposed sequence: `single-layer private contract -> isolate construction of MoE time components from an admitted EPLaneWorkload -> leave stage traversal/cache lifetime in predictor`. A child function should consume explicit bound model/parallel inputs and return existing `MoETime`/communication components, not accept the entire predictor or expose mutable parent dictionaries.
- Preserve bounded workload LRU capacity and copied attention result isolation. Verify cache hit/miss equivalence alongside mixed-layer and stage-total outputs before extracting numerical construction.

### 3. Disaggregation predictor

- Inspected responsibilities: role replica binding `:91–285`; routing setup `:286–523`; role-aware dummy timing `:577–910`; zero EP-barrier path `:961`; communication/overhead `:1017–1162`; attention-only builder `:1218`; stage/layer dispatch `:1354–2985`.
- Cleanup first: L2 and existing canonical routing use. The real routing path already shares layer allocations; no repeated-generation hoist is warranted. Keep dummy uniform routing distinct from configured real distributions.
- Retain current owner because role admission, measurement activation, attention/FFN selection, zero-work EP synchronization and once-only PP/CPU accounting have a defined order. The long method currently keeps that order explicit.
- Proposed sequence: `single-layer contract -> explicit admitted role inputs -> attention/full/FFN time-component construction -> common final ExecutionTime construction`. Keep one role dispatch/admission owner and reuse `_assemble_stage`, `AttentionTime`, `OverheadTime`, `CommunicationTime` and typed EP workloads. Extract already-distinct role calculations, not a parallel legacy/new stage execution path.
- Characterize PREFILL, DECODE, DECODE_ATTN and DECODE_FFN separately, including post-attention-only and zero EP barrier cases. Do not use this split to change supported architecture admission.

### 4. Shared prediction model manager

- Inspected responsibilities: constructor registries `:420–467`; family/path selection and training orchestration `:543–843`; GDN load-only path `:846–899`; typed contracts `:920–1346`; family training/filtering `:1348–3950`; cache identity and public registry views `:3952–4646`.
- Cleanup first: S2/S3 and L1. Constructor-owned family dictionaries should be used directly before any registry extraction. Keep GDN artifact validation/load separate from estimator fitting.
- Retain current owner because training deduplication spans cluster roles, precision, measurement family and typed FFN contracts. Splitting by family alone would duplicate that identity policy or publish partially populated views.
- Proposed sequence: `constructor-owned registries and canonical paths -> shared dataset/fit operations with base predictor -> artifact identity/load-store operations -> registry storage and published views`. Manager remains the orchestration/deduplication owner; implementation children consume explicit family/contract identity. Reuse `cache_io.py`, `measurement_input_paths.py` and existing trainer metadata, not a new path resolver or cache format.
- Keep family activation/view behavior and complete artifact keys stable. The shared preparation work with the base predictor is one dependent sub-step, not two parallel competing implementations.

### 5. MetricsStore

- Inspected responsibilities: trace projection `:525–1526`, including `_emit_stage_layer_traces:1096`; plotting/output `:1762–2218`; request lifecycle `:2806–3543`; stage operation metrics `:3766–4148`; stage ledger `:4185–4865`; transfer ledger `:4866–5640`.
- Cleanup first: M1–M5; independently settle the existing S1 residual metadata discrepancy before moving that path. Reuse canonical operator ownership rather than unifying through private scalar proxying.
- Retain current owner because event ordering, once-only stage fields, deferred trace decisions and pending ledger rows join schedule/completion callbacks. Moving just the append method would leave two owners for the same lifecycle.
- Proposed sequence: `canonical layer/operator projection -> stage trace projection -> complete stage-ledger lifecycle -> complete transfer-ledger lifecycle -> later plot/export separation`. `TraceStore` already owns buffering/persistence; any projection extraction must feed it, not invent another writer. Ledger extraction must take both row creation and completion/removal, with existing rounding and schemas unchanged.
- Stage and historical scalar inputs are different contracts, not automatically duplicate execution paths. Preserve real layer IDs, EP lane work versus barrier maxima, disabled-reporting gates and output inventories. See `metrics_review.md` for focused tests and the enabled-reporting performance limitation.

### 6. vLLM V1 replica scheduler

- Inspected additions: constructor slot manager `:129`; free/release `:1399–1444`; GDN/KV allocation `:2833–3026`; preemption `:3094`; waiting admission/queue commit `:3591–3856`. Existing bulk includes PP/MTP waits, prefix cache, graph/spec metadata and core scheduling loops.
- Cleanup first: P1c/P1e already address required fields and free-slot mechanics. No additional confirmed candidate-only GDN defensive workaround remains from this review.
- Retain current owner because KV allocation, GDN reservation/rollback, queue removal and RUNNING admission form one transaction. Candidate's allocate-before-queue-commit order protects failure recovery and must not be reversed for a shorter loop.
- Proposed sequence: `pure metadata construction using existing scheduler/utils responsibilities -> PP/MTP wait-state lifecycle as one owner -> only then consider request-selection extraction`. Inspect/reuse `afd_metadata.py`, `batch_builders.py`, `prefix_cache.py`, `request_selection.py` and diagnostics helpers for their actual matching responsibilities; do not create parallel generic wrappers. GDN slots already have `attention/gdn/state.py` as their owner.
- Keep allocation/release/queue commit together until a complete transactional interface can be characterized. Preserve preemption order, continuation slot ownership and rollback propagation. Focused paths include `tests/unit/test_gdn_scheduler_slots.py`, `tests/unit/test_pdaf_decode_attn_preemption.py` and `tests/unit/test_prefix_cache_scheduler_frontier.py`.

### 7. Aggregate configuration

- Inspected additions: `ReplicaConfig.__post_init__` GDN guards `:2085–2099`, predictor GDN path `:2155`, `SimulationConfig.__post_init__` GDN guards `:5362–5403`. Existing bulk is leaf config definitions, `ClusterConfig` materialization and global simulation validation.
- Cleanup first: S8's required-field access and S2's path default reuse. Real optional role configs and scheduler capabilities remain optional; do not replace the guard's role list with a helper unless its inactive/alias behavior matches exactly.
- Retain current owner because dataclass defaults, polymorphic registration, flattened CLI generation, role inheritance, runtime backend binding and validation order are coupled. Moving classes can change import registration or serialized identity without changing their bodies.
- Proposed sequence: `leaf dataclass groups -> ReplicaConfig -> ClusterConfig materialization -> SimulationConfig orchestration last`. Existing `frontier/config/` is the destination package; preserve both `frontier.config` and `frontier.config.config` import paths. Reuse `flat_dataclass.py`, `base_poly_config.py`, `parallel_semantics.py` and `utils.py`; do not introduce another schema/serializer.
- Characterize CLI flag names/defaults, config serialization and valid/invalid role constructors at each move. `frontier/config/__init__.py` intentionally imports base classes before re-exporting `config.py`; preserve initialization order. Focused paths include `tests/unit/test_gdn_runtime_guards.py`, `tests/unit/test_pdaf_config_contract.py` and `tests/unit/test_pd_transfer_types_and_configs.py`.

## Retained optionality, state and numerical constraints

| Pattern | Why it remains / implementation constraint |
| --- | --- |
| Shared manager or `_gdn_predictor` absent | Ordinary monolithic/dense or dummy construction does not require shared GDN artifacts. Actual GDN dispatch must continue to fail clearly when its required predictor is missing. |
| `include_moe=None`, optional lane workload and role configs | None can request model-owned classification, identify a non-lane call, or denote an inactive deployment role. Explicit False is not equivalent. |
| MoE workload LRU (`:930–981`) | Bounded at 256 entries and keyed by role/replica/layer/token/top-k/expert/EP identity. It avoids repeated deterministic materialization; not an unbounded or redundant cache. |
| Stage-local attention cache and copies (`:2199–2246`) | Cache lifetime fixes batch context; family/variant keys share numerics only within that context. Cloning mutable timing/operator maps prevents one physical layer from mutating another. |
| Disaggregation's second routing normalization (`:519–521`) | The canonical generator uses NumPy summation; the adapter uses Python summation. Removing the second pass can change floating-point ratios and downstream token integerization. Main's recorded 3,456 sampled histograms with no mismatch are useful but not a proof; retain in this cleanup. |
| GDN slot-manager None and rollback catch/rethrow | Non-GDN models allocate no slots. A KV failure after reservation must release the slot and propagate the original exception. This is a transaction, not exception swallowing. |
| `_expanded_trace_batches`, pending ledgers, Stage/scalar projections | They track reporting lifecycle or distinct admitted input contracts. Do not delete them as generic caches/compatibility code without tracing creation, completion and suppression ownership. |
| Manager measurement/precision/typed-contract registries | Distinct artifact identities are real. Remove lazy reconstruction, not the distinctions; legacy and typed entries cannot be merged solely because dictionaries look similar. |

## Sequencing, evidence and handoff

Recommended dependency order: `existing P1 cleanup -> planned runtime fidelity/timing`; independently, `S2/S3/S8 and L1 -> L2 if selected -> focused before/after checks`. Future splitting: `canonical contracts -> shared predictor/manager preparation -> independent metrics and scheduler functional extractions -> config orchestration last`. Independent inspections may run in parallel; shared-owner code moves should not.

- Observed evidence is source/diff/caller/fixture inspection only. No production/test edits, suites, benchmarks, GPU operations, remote changes or subagents were performed by this sidecar. Static inspection does not establish numerical or performance parity.
- Existing verification numbers in task records belong to main; they are not new results from this review. No performance gain is asserted for L1/L2 or for splitting files.
- Two guessed test filenames were absent during discovery; `rg --files` identified the actual paths above. This was an inspection-command error, not a test failure or an environment blocker.
- New semantic-choice blockers: none for L1/L2 on admitted current inputs. Routing normalization changes, supported legacy-view removal, or moving a shared interface beyond the current sub-step would need separate evidence/scope approval. No grill-me pause is needed for this read-only deliverable.
- Sidecar pending work: none. Main next steps: (1) reconcile L1/L2 against its latest edits; (2) select bounded cleanup with the listed unchanged-output checks; (3) continue phase fidelity and retain these split proposals as implementation-scoped follow-up, without delaying fidelity merely to reach a line-count target.

## Addendum — S6 minimal raw-type policy reuse

Requested follow-up: preserve Gemma norm's exact `qwen3_next` / `qwen3_5_moe_text` admission and MXFP4's narrower exact `qwen3_5_moe_text` admission. This is an implementation proposal for main, not a change to the approved policy. Addendum line references use the inspected `61595325` source; the two profiling checks are unchanged from the frozen candidate.

### Observed contracts and why ordinary profile resolution is insufficient

- `frontier/profiling/linear_op/linear_op_impl.py:63–69`, `_uses_gemma_rms_norm`: True only when `norm == "rms_norm"` and the unmodified model type is one of the two exact strings. Main already supported `qwen3_next`; the candidate adds `qwen3_5_moe_text`. `_build_untimed_norm` and other norm construction consumers use this choice.
- `frontier/profiling/moe/moe_vllm_kernel.py:457–466`, `validate_mxfp4_runtime`: exact raw model type first, then ROCm, then the functional vLLM API. `profile_fused_moe_kernel:732–733` passes its raw `model_type`. Keep this ordering and all three checks.
- `frontier/model_architectures.py:1164–1177`, `ModelArchitectureRegistry.resolve`, prioritizes an explicit profile; its Qwen matcher lowercases the raw type (`:1042`, `:1067`). Topology identity additionally strips whitespace (`:1071–1086`) and can accept profile/architecture-only identity. These policies are intentionally broader/different than the native profiler gates.
- Consequently, using `config.get_model_architecture_profile().<boolean>`, `is_qwen3_5_profile_config`, or `_matches_qwen3_5_moe` directly would change admission. An exact Qwen type with explicit `generic` must retain native profiling eligibility; uppercase, padded, profile-only and architecture-alias-only inputs must not gain it. Do not fabricate a config to re-enter `resolve`, mutate the explicit profile, or reclassify Qwen3-next as GDN.

### Recommended existing-registry extension

Extend the existing `ModelArchitectureProfile` metadata (`:408`) with two concrete tuple fields, both defaulting to empty: `gemma_rms_norm_model_types` and `mxfp4_moe_model_types`. Use the following declarations on the existing entries; no new profile, secondary registry, capability-name dispatcher or generic policy framework is needed.

| Existing profile entry | `gemma_rms_norm_model_types` | `mxfp4_moe_model_types` |
| --- | --- | --- |
| `ModelArchitectureProfile.generic()` (`:493`) | `("qwen3_next",)` | `()` |
| `ModelArchitectureProfile.qwen3_5_moe()` (`:610`) | `("qwen3_5_moe_text",)` | `("qwen3_5_moe_text",)` |
| Other existing entries | `()` | `()` |

Qwen3-next currently uses the generic architecture entry; placing its exact legacy native-norm rule there preserves that classification. It does **not** make all generic models Gemma-norm models. The type strings remain labels in the authoritative owner, not branches in consumers.

Add two narrow boolean queries on `ModelArchitectureRegistry`, alongside `iter_profiles`: `uses_gemma_rms_norm(model_type: str | None)` and `supports_mxfp4_moe(model_type: str | None)`. Each computes `any(model_type in profile.<its_specific_tuple> for profile in self.iter_profiles())`. This queries declared raw-type support across existing entries, not the selected runtime profile. Use no `.lower()`, `.strip()`, string coercion, profile resolution, fallback profile, new dictionary, dynamic `getattr`, cache, or additional mutable state. Exact membership preserves None/empty/unknown rejection.

Consumers then keep their local non-architecture checks and delegate only the type policy:

```python
# linear_op_impl.py: retain the existing norm guard and raw model_type value.
norm_is_rms and MODEL_ARCHITECTURE_REGISTRY.uses_gemma_rms_norm(model_type)

# moe_vllm_kernel.py: retain the existing failure message and following guards.
if not MODEL_ARCHITECTURE_REGISTRY.supports_mxfp4_moe(model_type):
    raise ValueError("MXFP4 profiling requires model_type='qwen3_5_moe_text'")
```

The illustrative variables are not a request for additional retained state. The existing small norm helper can remain the norm/type conjunction. Preserve CPU-safe metadata imports: registry metadata must not import Torch, vLLM or experimental SGLang. Leave accelerator detection with `profiling/common/accelerator.py`, vLLM API availability with the native adapter, and packed-layout ABI checks where they are. A later model adds its supported exact types through its existing profile entry; no consumer family switch changes.

This is a bounded extension of a shared registry contract, so main should include it in an explicit implementation sub-step. A standalone second raw-type table or a new Qwen3-next runtime profile is unnecessary and would enlarge the scope.

### Required before/after admission characterization

The table describes type eligibility only; MXFP4 still additionally requires ROCm and the functional vLLM API. Repeat norm rows with `layer_norm`, absent norm and `rms_norm`; only the latter permits Gemma norm.

| Raw `model_type` | Explicit profile / alias | Gemma norm with RMS norm | MXFP4 type eligibility |
| --- | --- | --- | --- |
| `qwen3_next` | absent or `generic` | True | False |
| `qwen3_5_moe_text` | absent, `generic`, or Qwen profile | True | True |
| `QWEN3_5_MOE_TEXT` | any | False | False |
| ` qwen3_5_moe_text ` or ` qwen3_next ` | any | False | False |
| None / empty | Qwen profile or exact architecture alias only | False | False |
| `other` | Qwen profile | False | False |

The existing `tests/unit/test_moe_mxfp4_increment10.py` covers mode, layout and backend contracts but does not itself characterize this complete raw-type matrix. `tests/unit/test_linear_op_profiling_output_metadata.py` verifies output metadata, not norm selection. Add the minimal direct predicate/gate checks to appropriate existing tests and retain these files as regressions, alongside `tests/unit/test_model_architecture_registry.py`. For the MXFP4 gate, control platform/API availability in CPU tests and verify rejection precedence; do not initialize native GPU modules merely to test identity. No suite or characterization was executed here, and CPU admission checks cannot establish native numerical parity.

## Addendum — Binder GDN variant without a local literal

- Remaining literal: `frontier/attention/model_binding.py:289`, inside `resolve_attention_topology`.
- Existing authoritative source: `frontier/attention/families.py:221–224`, `GATED_DELTA_NET_ATTENTION_FAMILY.supported_variants == ("qwen3_5",)`. `AttentionFamilySpec` inherits this field from `frontier/operators/spec.py:258`; inspected family/registry interfaces provide no separate model-to-variant resolver. Do not invent one or claim a resolver already exists.
- Minimal current-contract reuse: after the accepted Qwen topology branch, explicitly unpack the **only** registered variant, then use it for GDN layer specs:

```python
(gdn_variant,) = GATED_DELTA_NET_ATTENTION_FAMILY.supported_variants
# Existing schedule iteration and homogeneous branch remain unchanged.
LayerAttentionSpec(
    layer_id, GATED_DELTA_NET_ATTENTION_FAMILY.family_id, gdn_variant
)
```

- This is singleton cardinality enforcement, not first-element selection. `supported_variants[0]` or `next(iter(...))` would silently pick one if the registry grows. Unpacking instead fails when the current single-variant contract ceases to hold, forcing an explicit model-to-variant policy at that future extension. The field already exists; no default-variant field, additional enum, registry or cache is needed today.
- Do not derive the value by trimming `profile_id`, call `config.get_layer_attention_specs()` from its own resolver, or resolve/validate the architecture profile during topology construction. Those approaches respectively encode a naming convention or risk recursion. Reading the already-imported immutable family descriptor adds neither.
- Focused checks for main: preserve the exact GDN/full-layer spec tuple in `tests/unit/test_gdn_semantic_core.py:128`; retain homogeneous and hybrid binding tests; characterize singleton use and fail-fast behavior for a family descriptor with multiple variants. Do not alter numerical expectations. This proposal is source-inspected only and has not been implemented or tested by this sidecar.

Addendum completion: both requested proposals are recorded. Production/tests and other agents' edits remain untouched; sidecar pending work is empty. Main owns the registry-extension decision, implementation, and before/after verification.

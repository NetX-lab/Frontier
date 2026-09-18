## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Completed the independent fixed-diff Phase 2 inventory, bounded cleanup proposals, retained contracts, and source-level residual trace discrepancy. |
| 2026-09-17 | Added the specifically authorized S1 regression in the existing test file; reproduced six frozen-candidate failures and three passing native-main scalar controls. |

# Phase 2 metrics and scheduler reporting review

Status: complete review and authorized S1 red-regression handoff; production proposals are not implemented. Reviewer: Codex metrics sidecar. Fixed main: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`; fixed candidate: `c288a19f59bec09529ee18d782fa57218da2c781`. Concurrent cleanup edits are not the review source. The initial review was report-only. The subsequent explicit authorization permits one regression in `tests/unit/test_metrics_stage_execution_time.py`, focused reproduction and an RCA appendix here. No production edits, suites, commits or subagents. Source line references refer to the fixed candidate unless identified as new-test lines.

## Summary

Reviewed every changed hunk in four metrics files and thirteen scheduler files. Five bounded cleanup proposals are supported by producer/caller and registry inspection: remove orphaned private trace annotations and test-only legacy helpers; simplify the dense reporting adapter's model contract and avoid a discarded stage copy; reuse the existing FFN family iterator; replace the added GDN name tuple with its authoritative attention-family membership; hoist EP trace context construction to phase scope without persistent caching.

One separate correctness discrepancy was found and subsequently reproduced: the candidate unconditionally derives tensor metadata for COMPUTE events even when a residual event supplies complete metadata. The new focused test fails in all six frozen-candidate cases, at quantization lookup for the unsupported residual name, before shape lookup. Pinned main passes all three scalar controls using explicit residual metadata. Main has no Stage class; no main Stage parity is asserted. See the S1 execution appendix for exact commands, initial fixture correction, errors and observations.

Retain the approved physical-layer/stage contract, scalar timing interface, per-layer dense prediction, EP lane work versus barrier maxima, reporting-off capture avoidance, shared source-batch expansion decisions, ledger field names and rounding. No new semantic choice requires grilling on the evidence inspected.

The main agent/user reports fresh frozen-candidate unit results of 3587 PASS / 18 FAIL / 25 SKIP, with the same 18 failing nodes on pinned main, and equal before/after non-dummy artifacts including symmetric file sets. `progress.md` records these results (90 stable comparisons and 106-file symmetric metrics inventory). They validate completed runtime work, not the unimplemented proposals below or this newly added red test. The follow-up adds only the authorized test; production is untouched and test processes have ended before handoff for paired timing.

## Review plan

Dependency: fixed diff inventory -> reporting producer/caller contracts -> existing ownership/registry reuse -> minimal proposals and retained behavior -> test/path inventory. Read-only source inspection may proceed independently of the main agent's memory/predictor changes.

## Progress

- Read the current task plan/progress and retained D01/D02/D03 contract baseline in `spec_review.md`.
- Screened the fixed file inventory: four metrics files and thirteen scheduler files. Runtime-only scheduler changes will be explicitly classified, not treated as reporting cleanup targets.
- Reviewed constants, trace metadata and metrics-payload builders against production callers and existing abstractions before recording the findings below.
- Read every changed hunk in all seventeen metrics/scheduler files. Three scheduler files (`base_replica_scheduler.py`, `vllm_v1_engine_replica_scheduler.py`, `memory_planner.py`) only change runtime memory/lifecycle behavior and are screened out of reporting implementation proposals.
- Confirmed with fixed-tree `git grep` that `_trace_execution_time_override` and `_trace_dense_*` no longer have production writers. `predict_dense_reference()` and `_create_corrected_execution_time_for_metrics()` now have test callers only. Remaining scalar interfaces are retained explicitly below.
- Discovery correction: `frontier/operators/ops.py` does not exist; the authoritative shared declaration module is `frontier/operators/spec.py`, inspected with its registries/families.
- Incorporated the main agent's completed runtime verification; did not repeat suites or extend investigation after the final prioritization request.

## Per-file inventory

Paths are repository-relative; the repository root is `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`. Counts are fixed three-dot diff additions/deletions, not current cleanup changes.

| File | + / - | Inspected contract and disposition |
| --- | --- | --- |
| `frontier/metrics/constants.py` | 6 / 0 | GDN and DP metric labels use the existing enum/schema mechanism. Retain; labels themselves are not duplicate ownership policy. |
| `frontier/metrics/ep_wave_metrics.py` | 86 / 0 | `record_ep_wave`: real lane operators, phase starts, trace/operation/ledger gates. M5: reduce per-operator context reconstruction. Preserve actual lane work. |
| `frontier/metrics/metrics_store.py` | 403 / 33 | Stage trace projection, operation metrics, ledger projection, batch expansion, legacy consumers. M1/M3 and correctness item S1. No wholesale Stage/scalar or ledger rewrite. |
| `frontier/metrics/op_trace_utils.py` | 16 / 0 | GDN visible-payload metadata and DP collective metadata. M4: replace GDN name classification; S1 resolver evidence. Preserve tensor shapes. |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` | 13 / 8 | Reporting wrappers and metrics-store forwarding. M1/M2: remove unused wrapper and make required model config direct. |
| `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | 28 / 0 | Changed GDN capacity/runtime guards only. No reporting cleanup; main agent owns runtime review. |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | 81 / 15 | Changed recurrent-state lifecycle only. No reporting cleanup; preserve D03. |
| `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py` | 3 / 1 | Supplies the actual physical `layer_id`; retain. Filename spelling is the existing repository path. |
| `frontier/scheduler/utils/decode_collective.py` | 3 / 4 | Full-stage correction uses the unified adapter; old trace override writer removed. M1 caller evidence, M2 unused-argument adjustment only. |
| `frontier/scheduler/utils/dense_metrics.py` | 26 / 28 | Actual dense-layer predictions replace representative-first-layer annotations. M1/M2; preserve each physical layer's distinct prediction. |
| `frontier/scheduler/utils/ep_wave.py` | 4 / 0 | Threads `capture_lane_timings` into wave materialization. Retain the reporting-disabled no-capture contract. |
| `frontier/scheduler/utils/ep_wave_schedule.py` | 7 / 0 | Selects capture from reporting demand and forwards completed plan to metrics. Retain optional standalone reporting sink; no proven invariant permitting its removal. |
| `frontier/scheduler/utils/execution_time_metrics.py` | 30 / 56 | Stage-preserving metrics copy and validated single-layer extraction replace generic attribute copying. Retain public behavior and source isolation; M1 tests should call these live helpers. |
| `frontier/scheduler/utils/expert_parallel.py` | 22 / 2 | Conditional lane-record retention; requires a one-layer Stage when captured. Retain validation and no-capture path. |
| `frontier/scheduler/utils/memory_planner.py` | 89 / 9 | Changed runtime state/memory accounting only. No reporting cleanup; main agent owns this file. |
| `frontier/scheduler/utils/prefill_collective.py` | 10 / 3 | Reconstructs participant-specific full-stage attention/dense reporting at completion. Retain full scope and batch context; M2 wrapper arguments only. |
| `frontier/scheduler/utils/sync_entry.py` | 6 / 0 | Carries the optional metrics sink through decode/prefill entry. Retain; do not invent a required reporting service for standalone scheduling helpers. |

## Findings and bounded cleanup proposals

### Behavior-preserving cleanup record

| ID / Module | Problem | Root cause / unnecessary complexity | Minimal refactor | Reused existing abstraction | Verification to run after implementation |
| --- | --- | --- | --- | --- | --- |
| M1 — `metrics_store.py`, `dense_metrics.py`, cluster wrapper | Reflection for orphaned trace overrides; dead dense-reference and correction entry points | Branch removed production annotation writers when Stage became the reporting payload, but retained readers and tests of obsolete helpers | Pass `execution_time` directly at the old override read around 3871. Delete the entire `_trace_dense_*` augmentation blocks around 1054–1068 and 1458–1476; do not replace these MoE-branch blocks with scalar FFN reads, which could introduce output. Leave the existing scalar dense `else` paths intact. Delete `predict_dense_reference()` and `_create_corrected_execution_time_for_metrics()` after updating their test-only callers. Keep neutral annotation-isolation tests. | `StageExecutionTime`, `build_metrics_execution_time`, `build_single_layer_metrics_execution_time` | `test_execution_time_metrics_ownership.py`, `test_execution_time_op_times.py`, `test_metrics_full_stage_scope.py`, `test_stage_reporting_contract.py`; verify copied sources, IDs, layer scope, output equality. |
| M2 — `dense_metrics.py:108`, cluster wrapper around 1141 | Partial-object model lookup, whole-model first-dense search, discarded Stage copy, ignored wrapper arguments | Historical representative-layer workaround survived after changing to actual-layer prediction | Access `self._config.replica_config.model_config` directly. Replace `first_dense_layer_id()` with the equivalent typed mixed-model condition `0 < model_config.get_num_moe_layers() < model_config.num_layers`. Keep scalar/nonmixed branches returning an isolated metrics copy. For a mixed Stage, read the original finalized Stage and build the replacement once. Drop unused `actual_execution_time_ms`/`original_start_time` from the live internal wrapper and its two collective call sites. | `BaseModelConfig.get_num_moe_layers()` and `is_moe_layer()`; finalized Stage constructor | `test_stage_reporting_contract.py::test_dense_metrics_adapter_predicts_each_actual_dense_layer_and_preserves_owner`, ownership/finalization tests, decode/prefill EP materialization tests. Preserve non-MoE/all-MoE/zero-MoE no-reprediction cases and exact source isolation. |
| M3 — `metrics_store.py:3766` | Dense FFN operation metrics manually enumerate three operators while trace projection uses the family | Duplicate interpretation of the same FFN ownership table | Replace dense up/act/down pushes with `_iter_family_execution_times(FFN_FAMILY, layer)` and `OperationMetrics(op_name)`, retaining separate MLP all-reduce. Remove the unused `get_attention_family(family_id)` result at 1118; attention iteration already resolves that family. | Existing FFN family and iterator already used by MoE/share-expert metrics and Stage traces | `test_metrics_stage_execution_time.py`, `test_stage_reporting_contract.py`; assert same order, zero handling, values, layer IDs and independent operation-metrics enablement. |
| M4 — `op_trace_utils.py:534` | Four GDN operator names classified again locally | Added family support copied membership into a consumer | Use the existing `_get_family_operator_by_name` lookup with `GATED_DELTA_NET_ATTENTION_FAMILY` (also exported as `GDN_ATTENTION_FAMILY`) instead of the name tuple. Keep the existing visible `[tokens, hidden_size]` input/output shape. | `frontier/attention/families.py` and local family lookup helper | `test_op_trace_utils.py`, `test_attention_trace_mapping.py`, `test_metrics_stage_execution_time.py`, `test_mla_core_native_op_tracing.py`; GDN and existing dense/MLA metadata equality. |
| M5 — `ep_wave_metrics.py:50–74` | Context, token sum and parallel metadata rebuilt for every positive-duration operator | Phase-invariant metadata is constructed inside the innermost loop | Construct context once per traced phase with positive work; construct operator-specific tensor metadata per operator as before. Compute lane token sum once when tracing is requested. Keep data local; add no persistent cache. | `OpTraceContext`, `build_parallel_context`, existing phase/lane records | `test_stage_reporting_contract.py::test_ep_lane_reporting_preserves_work_and_barrier_gaps`, EP trace/materialization tests. Compare every timestamp, token count, shape, order and lane duration. |

Implementation boundaries:

1. M1 is dead private protocol removal, not permission to delete the scalar `ExecutionTime` interface. Its in-tree production writer search is empty; the remaining annotation assignments are isolation fixtures. The dead helper references are `test_execution_time_metrics_ownership.py:141,163` and `test_execution_time_op_times.py:863,896`.
2. M2's mixed-model predicate preserves the original helper's semantics, including zero MoE layers. `get_num_moe_layers()` at `frontier/config/model_config.py:629` reuses cached canonical layer IDs; do not parse enums or classify model names again. Production uses a configured replica model. Repair incomplete fixtures instead of preserving `getattr(getattr(...))`.
3. M2 must still predict each actual dense layer because its input Stage excludes FFN work. In the existing focused test, layers 7 and 9 have different 11 ms / 17 ms results and the stage owner remains 13 ms. A first-layer reference or stage-wide multiplier would undo D01. Copy removal requires finalized-source isolation, not shared mutable payloads.
4. M3 has matching FFN order (up, act, down) in `frontier/operators/families.py`. Do not mechanically merge trace and operation-metric emitters: their zero rows, legacy ADD handling, EP naming and role gates differ.
5. M4 must use the attention family. The general operator registry does not register GDN; moving GDN into that registry is unnecessary shared-contract expansion.
6. M5 should preserve the current absence of context construction when a phase has no positive operators or tracing is disabled. Ledger-only and operation-only modes do not require trace validation or trace contexts.
7. These are separate bounded sub-steps. Apply M1 in small dead-reader/helper units; M2's wrapper signature change touches its actual callers and should be an explicitly scoped sub-step. The large `metrics_store.py` warrants cleanup-first, not a new broad module split or another projection wrapper.

### S1 — Separate correctness discrepancy: explicit residual metadata is no longer sufficient

**Evidence:** Main `metrics_store.py:554–567` derives metadata only when `extra_meta is None`. Candidate `metrics_store.py:616–632` instead always calls `compute_op_trace_meta()` for positive COMPUTE/COMM events, then merges extras. The unchanged residual emitters at candidate 729–746 provide `residual_family` and `spec_decode_component` for `decode_draft_proposer` and `mtp_terminal_overshoot`. Neither has a compute-shape case in `op_trace_utils.py`; its unsupported branch at 621 raises `ValueError`. Zero duration returns before resolution, so zero-residual trace tests do not exercise the discrepancy.

**Reachability:** `sklearn_execution_time_predictor.py:8132–8153,8247–8248` and `sklearn_moe_execution_time_predictor.py:2496–2516,2574–2575` compute and pass these runtime residual durations. Tracing therefore takes the unsupported resolver path when either is positive. This predictor inspection is static evidence; the follow-up appendix separately demonstrates actual public-metrics caller failures with valid constructed timing records. The eight non-dummy cases are not claimed to exercise this branch.

**Cause:** `extra_meta` used to mean complete explicit event metadata. New physical-layer trace callers also use it for supplemental family/layer identity, and the candidate globally changed the resolver rule to accommodate those callers. The two different meanings were conflated.

**Minimal proposal:** Separate complete event metadata from additive layer metadata in the local trace emitter contract. Explicit residual metadata must bypass tensor-shape derivation; physical-layer COMPUTE/COMM events must still derive their existing tensor metadata and merge family/layer identity. Keep residual event names, type, duration, cursor position and tags. Do not catch `ValueError`, invent tensor shapes, hard-code a residual-name allowlist in the generic resolver, or restore the old global condition in a way that drops layer tensor metadata.

**Focused verification required:** Extend the existing Stage op-level trace tests with a real configured trace store and positive proposer/terminal owner durations. Assert both residual events are emitted once with unchanged tags/durations and subsequent physical-layer events retain tensor and family metadata. Cover scalar and Stage emitter inputs. The existing ledger test uses owner values 17 ms / 19 ms but validates ledger projection, not this trace path. `test_spec_decode_trace_liveness.py` tests trace-input exhaustion, not emitted operator metadata, and is not sufficient coverage by itself.

**Decision boundary:** This restores an existing supported residual reporting contract without choosing new simulator semantics; no grill-me question is established. Keep it a separate correctness fix with reproduction/RCA evidence rather than labeling it output-preserving cleanup. An unproduced `_trace_related_collective_waits` hook is not asserted as another reachable regression.

## Retained behavior and verification

| Inspected pattern retained | Why it is required |
| --- | --- |
| Stage versus scalar `ExecutionTime` handling | Stage is the physical-layer aggregate contract; scalar remains an actual construction/helper interface. Removing orphaned private overrides does not authorize removing scalar support. |
| Actual physical layer IDs and attention family/variant metadata | D01 explicitly approved uniform physical-layer ownership. Preserve real noncontiguous IDs, family/variant aggregate grouping and owner-once accounting. |
| Dense-layer predictions and participant-specific PREFILL full-stage prediction | The input omits dense FFN work or describes only the final executed layer. Each real dense layer and participant batch context must be represented; no representative-first-layer replacement. |
| `_expanded_trace_batches` and request expansion quotas | Attention and multiple EP wave reports for one source batch must share an expansion decision even after the request quota is reached. The set is constructor-owned functional state, not an incomplete-mock workaround. |
| `capture_lane_timings=False`, empty lane records, optional metrics sink | Reporting-off execution avoids retaining Stage/Batch records, and standalone scheduling helpers can omit reporting. No proof justifies making those states invalid. |
| EP one-layer validation and `layer_execution_times[0]` in the reporter | The materializer checks a singleton physical-layer Stage before retention. This is not an aggregate-first-layer shortcut. |
| EP phase maxima separate from lane operator durations | Maxima determine synchronization barriers; traces/metrics must record each lane's actual work and gaps, not charge every lane the maximum. |
| Optional ledger accumulator | It represents an actually disabled output family, not absent required construction state. |
| Existing Stage ledger scalar projection and owner-field source | Preserve legacy CSV keys, per-component rounding order, family slot replacement and stage-owner fields charged once. `StageExecutionTime.stage_owned_fields` is authoritative; do not duplicate it or replace the ledger with a naive sum of op times. |
| Internal EP reporter access to MetricsStore internals | It is a small internal collaborator, not an independently supported public adapter. No evidence justifies a new proxy/service wrapper merely to conceal private field access. |
| Lazy operator-specific attention metadata in `compute_op_trace_meta` | Dense and MLA shape requirements differ. Eagerly resolving both would validate irrelevant dimensions and change accepted inputs. |

No recommendation is made to gate all scheduler reporting predictions off, unify all scalar/Stage projection code, change global expansion retention, or add caches. Those changes lack the bounded side-effect/compatibility evidence needed here.

## Verification handoff and exact paths

The initial review executed no tests, suites, simulator runs or timing jobs. The explicitly authorized S1 follow-up below executes only the new parameterized regression and corresponding native-main scalar controls. All other proposed checks remain for the implementing main agent. The fresh runtime suite results supplied by the user are acknowledged, not repeated or attributed to this sidecar.

All of these existing paths were verified in the fixed tree:

```text
tests/unit/test_stage_reporting_contract.py
tests/unit/test_execution_time_metrics_ownership.py
tests/unit/test_execution_time_op_times.py
tests/unit/test_stage_finalized_contract.py
tests/unit/test_metrics_stage_execution_time.py
tests/unit/test_metrics_full_stage_scope.py
tests/unit/test_op_trace_utils.py
tests/unit/test_attention_trace_mapping.py
tests/unit/test_mla_core_native_op_tracing.py
tests/unit/test_ep_trace.py
tests/unit/test_typed_ep_trace_contract.py
tests/unit/test_prefill_ep_wave_materialization.py
tests/unit/test_decode_ep_wave_materialization.py
tests/unit/test_pdaf_prefill_model_time.py
tests/unit/test_pd_decode_moe_layer_accounting.py
```

Minimum shared Stage checks: real IDs `[4, 7, 9, 12, 14]`; distinct family/variant aggregate durations; operator metrics and ledger enabled without utilization; all reporting disabled builds no projection records; stage owner charged once; EP actual lane totals 13 ms / 12 ms with phase starts `[1, 1.003, 1.005, 1.010, 1.014]`; shared expansion decisions regardless of attention/wave reporting order. Preserve values and artifacts rather than changing expected results.

Example targeted command for M3 (not executed):

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 \
  /data/ycfeng/tmp/quality-review-env/bin/python -m pytest \
  tests/unit/test_metrics_stage_execution_time.py \
  tests/unit/test_stage_reporting_contract.py \
  -q -p no:cacheprovider
```

Run from the repository root above, with the main agent's already-validated environment; capture interpreter/package details with the actual check. For broader acceptance use the exact source flags and artifact checks already recorded in `task_memory/task_2026-09-17_candidate_quality_cleanup/spec_review.md`:

- Eight-case harness: `tests/integration/test_pr33_nondummy_acceptance.py`; compare both artifact-set directions and contents against `../quality-baseline-c288a19f` after relevant changes. Do not replace reporting-enabled verification with successful process exit.
- Fidelity harness: `tests/integration/run_scheduler_refactor_fidelity.py`; keep approved main/candidate differences classified and use frozen candidate for cleanup preservation.
- Paired harness: `tests/performance/measure_pr33_paired.py`, using `tests/performance/sim_walltime_scaling/run_case.py`. Existing paired scenarios disable reporting and cannot measure M3–M5 enabled-reporting overhead. No performance improvement is asserted here.
- Current validation record: `task_memory/task_2026-09-17_candidate_quality_cleanup/progress.md`; historical/source/artifact baseline: sibling `spec_review.md`. Do not conflate the fresh 18 shared failures with historical main's 19-failure report.

## Supporting inspection commands

The review used read-only fixed-tree commands; no source execution was needed. Representative exact commands and all scoped files are recorded here and in the inventory above:

```bash
git diff --numstat 0515589ac7f49ac5288a5f55b0ce38b0ede29bb2...c288a19f59bec09529ee18d782fa57218da2c781 -- frontier/metrics frontier/scheduler
git diff --unified=5 0515589ac7f49ac5288a5f55b0ce38b0ede29bb2...c288a19f59bec09529ee18d782fa57218da2c781 -- frontier/metrics frontier/scheduler
git show c288a19f:frontier/metrics/metrics_store.py
git show 0515589a:frontier/metrics/metrics_store.py
git show c288a19f:frontier/metrics/op_trace_utils.py
git show c288a19f:frontier/metrics/ep_wave_metrics.py
git show c288a19f:frontier/scheduler/utils/dense_metrics.py
git show c288a19f:frontier/config/model_config.py
git grep -n -E '_trace_execution_time_override|_trace_dense_|predict_dense_reference|_create_corrected_execution_time_for_metrics' c288a19f -- frontier tests/unit
git grep -n -E 'capture_lane_timings|record_ep_wave|metrics_store' c288a19f -- frontier/scheduler/utils/ep_wave.py frontier/scheduler/utils/ep_wave_schedule.py frontier/scheduler/utils/expert_parallel.py frontier/scheduler/utils/sync_entry.py
git grep -n -E 'decode_draft_proposer|mtp_terminal_overshoot|Unsupported compute op' c288a19f -- frontier/metrics frontier/execution_time_predictor
git ls-tree -r --name-only c288a19f tests/unit
git status --short
```

## Exact minimal edit handoff

These are proposals except for the S1 regression explicitly marked added. No replacement emitter or test-only behavior has been introduced in production.

| Item | Production edit | Minimum test edit |
| --- | --- | --- |
| S1 | In `MetricsStore._emit_op_level_traces`' local `emit`, distinguish caller-complete metadata from additive layer metadata, for example a keyword-only `resolved_meta` argument. When provided, copy it instead of calling `compute_op_trace_meta`; then merge the existing additive `extra_meta`. Pass the two residual tag dictionaries through the complete-metadata argument. Keep layer call sites on the derived-metadata path. No registry extension, residual-name switch or exception suppression. | **Added:** `test_residual_traces_preserve_owner_and_physical_layer_metadata` in `tests/unit/test_metrics_stage_execution_time.py:204`; three residual combinations times two timing shapes. Existing `_layer` accepts residual constructor arguments. Leave the test red until the main agent fixes production. |
| M1a | Remove orphaned trace override/augmentation readers in `frontier/metrics/metrics_store.py`, as detailed in M1. | In `test_execution_time_metrics_ownership.py`, rename diagnostic-isolation attributes to a neutral test annotation; retain source isolation and entity-ID assertions. No new obsolete-override compatibility test. |
| M1b | Remove `predict_dense_reference` from `dense_metrics.py` and `_create_corrected_execution_time_for_metrics` from `base_cluster_scheduler.py`. | In `test_execution_time_op_times.py`, change the two `test_corrected_execution_time_copy_preserves_*` tests to call `build_metrics_execution_time(original_execution_time)` directly, preserving explicit-zero versus omitted split-TP assertions. In ownership tests, migrate the two `test_dense_reference_*` cases to `build_prefill_metrics_execution_time` with a mixed model and actual dense-layer ID; retain singleton extraction and multi-layer rejection. |
| M2 | Replace first-dense scanning with `get_num_moe_layers`, directly read required replica/model config, avoid the discarded mixed-Stage copy. Remove only the two unused arguments from the live wrapper and its DECODE/PREFILL calls; do not remove the callers' real elapsed-time or start-time calculations. | Add `get_num_moe_layers` to the valid lightweight model doubles in ownership/reporting tests. Adjust the explicit wrapper double in `test_pd_decode_moe_layer_accounting.py` if its signature still includes those two arguments. Keep `test_pdaf_prefill_model_time.py` assertions about the third argument/full-stage scope unchanged. Retain or add a parameterized nonmixed no-prediction assertion. |
| M3 | Replace the three manual dense FFN pushes with `_iter_family_execution_times(FFN_FAMILY, layer)`; keep the existing all-reduce push. Delete unused family resolution at 1118. | Extend `test_stage_operation_metrics_emit_each_layer_once_with_its_attention_family` with explicit up/act/down/all-reduce values and expected per-layer series. Keep existing operation-metrics-independent-of-utilization test. |
| M4 | Import `GDN_ATTENTION_FAMILY` in `op_trace_utils.py`; replace only the four-name membership tuple with `_get_family_operator_by_name(GDN_ATTENTION_FAMILY, op_name) is not None`. `AttentionFamilySpec` inherits `OperatorFamilySpec`, so no helper signature widening is needed. | Existing GDN trace test already covers all four names. Add explicit visible input/output `[1, 16]` shape assertions there; preserve durations and family identity. |
| M5 | Construct `OpTraceContext` and parallel metadata once per traced phase with positive operators; leave operator-specific metadata and cursor advancement in order. | Extend `test_ep_lane_reporting_preserves_work_and_barrier_gaps` with routed versus nonrouted effective tokens and tensor/parallel metadata assertions. Keep reporting-off and zero-work phase coverage; no new persistent cache or timing benchmark harness. |

## S1 execution appendix — independently reproduced, production fix pending

### Authorization, seam and test scope

The follow-up explicitly authorized editing only `tests/unit/test_metrics_stage_execution_time.py` and appending evidence here, with no production edits or commit. The existing file is suitable: it already tests the MetricsStore/Stage reporting seam. The new case uses the public `MetricsStore.on_replica_stage_schedule`, real `SimulationConfig`/`ClusterConfig`/`ReplicaConfig`, real `ExecutionTime`/`StageExecutionTime` and `BatchStage`; only the injected output sink collects events. No private emitter is mocked or patched.

The test adds 87 lines including one import and the fixture's residual keyword forwarding. It covers `(proposer_ms, terminal_ms) = (3, 0), (0, 5), (3, 5)` in scalar and Stage form. The Stage has physical layers 7 and 9 with 2 ms / 7 ms attention work; the non-owner layer deliberately contains 99 ms residual values, which must not be charged. Expected residual names, tags, event types, single-owner durations and cursor positions are asserted, followed by tensor metadata and actual physical layer/family/variant identity. No xfail, swallowed exception or changed production expectation is used.

### Observed results and RCA refinement

| Run | Observed result | Meaning |
| --- | --- | --- |
| Initial frozen-candidate new test | 6 FAIL in 6.99 s | All cases entered the real emitter and failed at residual precision lookup. |
| Initial native-main scalar control | FAIL after residual emission: missing structured `attn_kv_cache_save` | The initial fixture's sparse attention map did not satisfy main's supported attention tracing contract. This was a fixture issue, not S1. The log is preserved. |
| Final frozen-candidate test with complete attention-family map | **6 FAIL in 6.87 s**, exit 1 | Same S1 exceptions with a valid common attention payload. Four cases fail on proposer; the two terminal-only cases fail on terminal. |
| Final native-main scalar controls with complete map | **3 PASS**, exit 0 | All three residual combinations emit the expected residual tags/durations followed by attention tensor metadata. |

The fixture correction fills zero-valued required attention operators from `DENSE_ATTENTION_FAMILY.e2e_trace_ops()` on both sides. It does not change the nonzero durations or expected outputs and introduces no production compatibility branch.

Exact final frozen-candidate error path:

```text
test_metrics_stage_execution_time.py:247 -> MetricsStore.on_replica_stage_schedule
metrics_store.py:3876 -> _emit_op_level_traces
metrics_store.py:729 or :738 -> emit
metrics_store.py:629 -> compute_op_trace_meta
op_trace_utils.py:272 -> _precision_for_op
op_trace_utils.py:251 -> QuantizationManager.get_precision
quantization_manager.py:530 -> ValueError
Unsupported operation 'decode_draft_proposer'. See registry: data/config/op_quantization/supported_operations.json
Unsupported operation 'mtp_terminal_overshoot'. See registry: data/config/op_quantization/supported_operations.json
```

This refines the static finding: the resolver rejects the residual at **precision lookup**, before the also-unsupported compute-shape branch. The cause remains the candidate's unconditional tensor-metadata derivation for caller-described residual events, not a request to add those residuals to the operator registry. Scalar failure rules out Stage ownership as the cause; terminal-only cases rule out the proposer exception merely masking the second defect; main's complete-map controls show that the old explicit-metadata contract supports these values.

Native-main event observations (start times in seconds, durations in milliseconds):

| Control | Emitted sequence `(name, start, duration)` |
| --- | --- |
| proposer | `(decode_draft_proposer, 1.000, 3)`; `(attn_prefill, 1.003, 2)` |
| terminal | `(mtp_terminal_overshoot, 1.000, 5)`; `(attn_prefill, 1.005, 2)` |
| both | `(decode_draft_proposer, 1.000, 3)`; `(mtp_terminal_overshoot, 1.003, 5)`; `(attn_prefill, 1.008, 2)` |

The native-main controls use its actual scalar constructor (no candidate-only identity keywords). Main does not export `StageExecutionTime`; the candidate test module cannot be imported unchanged against main. No Stage compatibility adapter, injected replacement class or main Stage success claim is used. These are focused native-main controls of the same residual seam, not a main run of the candidate-only pytest module. Stage post-fix assertions remain unexecuted because the intentionally red test aborts at the residual before physical-layer emission.

### Environment, source and artifact inventory

- Interpreter: `/data/ycfeng/tmp/quality-review-env/bin/python`, Python **3.12.3**, GCC 13.3.0; dedicated venv, no conda activation. CPU only; OMP/OpenBLAS/MKL threads each 1.
- Frozen production root: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/quality-baseline-c288a19f`; `git rev-parse HEAD` confirmed `c288a19f59bec09529ee18d782fa57218da2c781`.
- Main production root: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915`; HEAD confirmed `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`. Native control prints the resolved MetricsStore source file.
- Current new test: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_metrics_stage_execution_time.py:204`.
- Initial logs, preserved: `/data/ycfeng/tmp/quality-s1-frozen-regression-20260917.log`, `/data/ycfeng/tmp/quality-s1-main-control-20260917.log`.
- Final frozen log: `/data/ycfeng/tmp/quality-s1-frozen-complete-20260917.log`; pytest temporary root: `/data/ycfeng/tmp/quality-s1-frozen-complete-20260917`.
- Final main log: `/data/ycfeng/tmp/quality-s1-main-complete-20260917.log`; output root: `/data/ycfeng/tmp/quality-s1-main-complete-20260917`.
- Read-only `git diff --check -- tests/unit/test_metrics_stage_execution_time.py` passed. Unrelated production/test/progress changes were preserved. No commit was made.

### Exact final frozen command

Executed from the frozen production root, importing the current test as a test-only input. `--import-mode=importlib`, explicit `PYTHONPATH` and frozen working directory keep production imports on the frozen revision. The traceback paths confirm that source choice.

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/quality-baseline-c288a19f
set -o pipefail
env PYTHONPATH=/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/quality-baseline-c288a19f TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_metrics_stage_execution_time.py::test_residual_traces_preserve_owner_and_physical_layer_metadata --rootdir=/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/quality-baseline-c288a19f --import-mode=importlib -q -p no:cacheprovider --tb=short --basetemp=/data/ycfeng/tmp/quality-s1-frozen-complete-20260917 2>&1 | tee /data/ycfeng/tmp/quality-s1-frozen-complete-20260917.log
```

The initial frozen invocation used the `quality-s1-frozen-regression-20260917` basename and did not enable pipefail, so its shell exit reflected `tee`; its pytest summary, not that shell status, establishes the initial failure. The final invocation enables pipefail and returns exit 1. Use fresh temporary/log names for subsequent reruns.

### Exact final main command

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915
set -o pipefail
env PYTHONPATH=/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915 TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python - <<'PY' 2>&1 | tee /data/ycfeng/tmp/quality-s1-main-complete-20260917.log
import inspect
import json
import sys
from types import SimpleNamespace
from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.config import global_vars
from frontier.config.config import ClusterConfig, MetricsConfig, ReplicaConfig, SimulationConfig
from frontier.entities import BatchStage, ExecutionTime, Request
from frontier.metrics.metrics_store import MetricsStore
from frontier.types import ClusterType

print('python', sys.version)
print('production_source', inspect.getfile(MetricsStore))
for name, proposer, terminal in [('proposer', 3.0, 0.0), ('terminal', 0.0, 5.0), ('both', 3.0, 5.0)]:
    global_vars.reset_global_vars()
    cluster = ClusterConfig(replica_config=ReplicaConfig(device='a100', network_device='a100_pairwise_nvlink', model_name='meta-llama/Llama-2-7b-hf'))
    config = SimulationConfig(cluster_config=cluster, metrics_config=MetricsConfig(output_dir='/data/ycfeng/tmp/quality-s1-main-complete-20260917/' + name, write_metrics=False, store_plots=False, enable_op_level_tracing=True, enable_per_layer_expansion=True, num_requests_to_trace_per_layer=1))
    events = []
    store = MetricsStore(config, {ClusterType.MONOLITHIC: cluster}, SimpleNamespace(log_event=events.append))
    zero_fields = ('attention_rope_execution_time', 'attention_kv_cache_save_execution_time', 'attention_decode_execution_time', 'attention_prefill_execution_time', 'attention_layer_pre_proj_execution_time', 'attention_layer_post_proj_execution_time', 'attn_norm_time', 'mlp_norm_time', 'add_time', 'tensor_parallel_communication_time', 'pipeline_parallel_communication_time', 'expert_parallel_communication_time', 'moe_gating_time', 'moe_shuffling_time', 'schedule_time', 'sampler_e2e_time', 'prepare_inputs_e2e_time', 'process_model_outputs_time', 'ray_comm_time')
    op_times = {operator.name: 0.0 for operator in DENSE_ATTENTION_FAMILY.e2e_trace_ops()}
    op_times['attn_prefill'] = 2.0
    timing = ExecutionTime(**dict.fromkeys(zero_fields, 0.0), num_layers_per_pipeline_stage=1, is_moe=False, op_times=op_times, decode_draft_proposer_time=proposer, mtp_terminal_overshoot_time=terminal)
    batch = BatchStage(batch_id=73, replica_id=0, pipeline_stage=0, execution_time=timing.total_time, model_execution_time=timing.model_time, requests=[Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)], num_tokens=[8], cluster_type=ClusterType.MONOLITHIC)
    batch.on_schedule(1.0)
    store.on_replica_stage_schedule(time=1.0, replica_id=0, stage_id=0, batch_stage=batch, execution_time=timing, cluster_type=ClusterType.MONOLITHIC)
    expected = [(op, ms, family, component) for op, ms, family, component in [('decode_draft_proposer', proposer, 'mtp_draft_proposer', 'draft_proposer'), ('mtp_terminal_overshoot', terminal, 'mtp_terminal_overshoot_compute', 'terminal_overshoot')] if ms > 0]
    assert [event.name for event in events] == [row[0] for row in expected] + ['attn_prefill']
    cursor = 1.0
    for event, (op, duration, family, component) in zip(events, expected):
        assert (event.name, event.type, event.layer_id, event.duration_ms) == (op, 'COMPUTE', -1, duration)
        assert abs(event.ts_start - cursor) < 1e-12
        assert event.meta['residual_family'] == family
        assert event.meta['spec_decode_component'] == component
        cursor += duration * 1e-3
    assert events[-1].duration_ms == 2.0
    assert abs(events[-1].ts_start - cursor) < 1e-12
    assert events[-1].meta['tensor_shape'] and events[-1].meta['tensor_size_bytes']
    print('PASS', name, json.dumps([{'name': event.name, 'start': event.ts_start, 'duration_ms': event.duration_ms} for event in events]))
global_vars.reset_global_vars()
PY
```

The initial native-main command omitted the family import/zero-map initialization and supplied only `op_times={'attn_prefill': 2.0}`; its preserved failure directly motivated the valid common fixture used in the final command and test.

## Handoff

- Completed: fixed-diff inventory, exact M1–M5 proposals, retained contracts, S1 reproduction, one parameterized red regression and exact main/frozen evidence.
- Sidecar pending work: none. New issues beyond the confirmed S1: none. Production remains untouched; the authorized test edit is uncommitted and other workers' edits are preserved.
- All focused processes ended; no additional tests are planned during the main agent's isolated paired timing.
- Main-agent next steps: (1) after runtime timing, fix S1's local metadata contract; (2) run the new focused test until all six cases pass, preserving physical-layer metadata assertions; (3) proceed with bounded metrics cleanup and artifact comparison. The red regression is intentionally not claimed as passing or a production fix.

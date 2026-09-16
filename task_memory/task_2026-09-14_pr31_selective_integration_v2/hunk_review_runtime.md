## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Independently reviewed baseline-to-working-tree runtime hunks; identified ledger flag and routed EP reporting gaps; implemented assigned conditional EP timing transport. |

# Runtime hunk reassessment

Reviewer: `/root/w03_contract_review`. Baseline: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`. Scope: every current production hunk under `frontier/entities/`, `frontier/scheduler/`, and `frontier/metrics/`, plus the two explicitly assigned sync-event transport call sites. Root is concurrently completing W08, so the exact current hunk inventory below is a review checkpoint, not a claim that later edits were reviewed. Historical review documents were preserved.

The requested production ledger is named `changed_hunk_review.md` in this checkout; `production_hunk_ledger.md` was not found. It contains the original H001–H471 inventory. `changed_file_inventory.txt` was also inspected. This lane adjudicates the runtime subset of those original entries and every new runtime hunk below; it does not claim coverage of other lanes' original entries.

## Findings and disposition

**F1 — BUGFIX required, ledger completion depends on utilization.** `MetricsStore.on_replica_stage_schedule()` captures ledger rows when requested, independently of utilization. `on_batch_stage_end()` still returned before consuming pending rows when utilization was off. This left actual completion timestamps and completed rows missing. Root was notified immediately and moved the utilization guard to meter updates only. The new normal-constructor regression now passes: completion at `1.25 s`, all three pending maps empty, one completed `28 ms` row. The nine-case reporting suite was subsequently extended to eleven with EP transport tests.

**F2 — BUGFIX in progress, actual routed EP output was absent.** `predict_ep_wave_phase_times()` predicted each physical lane with `include_attention=False`, reduced it to five scalar phases, and discarded the typed result. `prepare_moe_wave_from_inputs()` passed only Python logger callbacks; `ep_trace.log_workload_trace()` writes `logger.info`, not `TraceStore` or operation series. The final PREFILL/DECODE stage report uses `include_ffn=False`, and the dense adapter fills only dense FFNs. Therefore the separate wave logs did not satisfy requested operator metrics, typed traces, or lane ledger output. Source causality was confirmed by root. The conditional exact-payload transport described below is implemented; root owns projection and end-to-end integration. Until those checks pass, W08 routed reporting remains open.

**F3 — DELETE/REFACTOR candidate, mock-justified GDN fallback remains.** The added `_can_allocate_request()` comment explicitly says a missing `_gdn_state_slot_manager` supports lightweight instances constructed before initialization. Normal `VLLMv1EngineReplicaScheduler.__init__()` always defines it. W02 forbids production fallback justified by mock construction. The same `getattr(..., None)` pattern appears in resource cleanup/allocation. Removing this requires migrating partial-constructor test fixtures; root was notified. No numerical failure is claimed from this source finding.

**F4 — DEFER pending supported-scope adjudication, pure-GDN automatic capacity.** `build_sequence_mixer_schedule()` accepts zero-full-attention schedules, including an interval greater than layer count. `MemoryPlanner.get_num_blocks()` correctly reserves fixed state and returns zero KV blocks when no KV-carrying layer exists, but `BaseReplicaScheduler` requires at least one initial KV chunk and raises `insufficient_initial_block_budget`. This is a constructor limitation for a source-admitted all-GDN configuration, not a reproduced failure in the checked-in hybrid model. Root was notified; no capacity PASS should be generalized to all-GDN schedules without either explicit admission scope or a supported no-KV admission path.

**N02 correction — prior getter-mutation accusation rejected.** At the initial task HEAD `41777755`, `ExecutionTime.communication_time_component` already returned a deepcopy. The old Stage getter modified that detached object, not the source component. The initial interpretation was wrong and was withdrawn in `w03_contract_review.md`. The defensible W03 changes are uniform scope, finalized publication, explicit identities, elimination of first-layer/private facades, and source/sibling isolation; they must not be described as reproducing that nonexistent getter alias.

## Contract groups used by the hunk inventory

| Group | Disposition | Actual callers and inspected evidence |
| --- | --- | --- |
| E0 | RETAIN | `entities.__init__` exports StageExecutionTime to predictor, scheduler and metrics consumers. |
| E1 | BUGFIX / REFACTOR / DELETE | `ExecutionTime` now represents one layer with or without identity; every formerly multiplied scalar and model total has the same scope. All-or-none identity validation rejects partial IDs. Supported mutators check publication before updating; snapshots clone mutable maps/components once and reuse finalized records. `_scaled_time_attr_value` was deleted. `as_single_layer` preserves entity IDs. Generic stale scalar docstrings still say aggregated and should be corrected as documentation cleanup, not mistaken for active behavior. |
| E2 | REFACTOR | `StageExecutionTime` validates ordered unique real IDs, finalizes input numerics, sums distinct layer blocks, charges PP/draft/terminal and CPU/diagnostic owner fields once, and caches model total without mutation-version scans. Component views reuse dataclass fields and return independent projections; mixed MLP/MoE component access rejects ambiguity. The public scalar lists declare projection scope; numerical component fields remain in existing dataclasses/operator maps. No `__getattr__` or private-name catch-all remains. Every `_first_layer` call belongs to an explicitly named singleton probe whose guard rejects multi-layer stages. |
| M0 | RETAIN | Four registered GDN operation names extend `OperationMetrics`; `compute_op_trace_meta()` reports visible token/hidden input/output shapes without inventing native recurrent layouts. |
| M1 | BUGFIX / REFACTOR | Metrics imports and family iterator preserve physical operator ownership. The ledger name map is explicitly an external CSV adapter. `_iter_layer_attention_times()` consults family attrs to avoid duplicate model projections. Stage traces select actual IDs or family/variant/operator aggregates according to expansion and quota; metadata is merged with real tensor/parallel metadata. |
| M2 | BUGFIX | `_push_stage_layer_operation_metrics()` iterates real layers; utilization no longer suppresses requested operation insertion. New `on_batch_stage_end()` guard repair closes F1. Legacy `ExecutionTime` branch remains a genuine layer path; stage dispatch returns before its private `_is_moe` read. No Stage private facade is used. |
| M3 | REFACTOR | Stage ledger sums per-layer legacy projections, replaces overlapping alias fields for canonical family ops, and selects owner fields from `StageExecutionTime.stage_owned_fields`. The 28 ms mixed fixture and 285/298/321 ms owner fixture prove this scoped behavior. Generic equality of flat operator sum and critical path is not asserted; existing scheduling formulas are retained. |
| S0 | BUGFIX / REFACTOR | BaseClusterScheduler adapters preserve generic stage scope. Metrics copy preserves maps, MLA fields and entity IDs rather than reconstructing a partial legacy scalar object. Singleton helper unwraps only a verified one-layer Stage. Dense adapter predicts each actual dense ID, preserves routed IDs and owner, and deletes first-dense-layer trace hints. |
| S1 | BUGFIX | ReplicaStageScheduler derives PP offset and PD-AF current layer. Final decode full-stage prediction includes PP start; final prefill metrics no longer use the last-layer-only object. These are real caller migrations, not dummy identity defaults. Fullstage attention and actual dense FFNs remain separate from physical routed-lane work. |
| S2 | RETAIN / BUGFIX | BaseReplicaScheduler resolves one maximum admitted cap from supported ordinary/phase caps. MemoryPlanner and GDN state slot manager share that value. Fixed state reserves `cap × state bytes`; selected resident stage contributes actual KV-carrying layer count; parameter bytes use ParamCounter's precision-aware result. Existing parameter/planning owner exposes stage counts, replacing private cross-module inspection. F4 limits claims about all-GDN topology. |
| S3 | RETAIN / BUGFIX | VLLM-v1 normal construction installs slot manager only for admitted supported GDN topology. New KV allocation rolls back on slot failure; queue/counter commit follows successful allocation. Continuations require/resume owned state; cleanup releases idempotently including orphaned IDs; unsupported state-dropping preemption validates before mutating victim state. F3 is the remaining mock-justified fallback cleanup. |
| T0 | BUGFIX in progress | Existing sync events pass optional metrics_store explicitly through sync_entry to the wave owner. `EPWavePhaseTimes.lane_records` defaults empty; capture retains actual finalized singleton stage and existing lane batch only when requested. Existing six numerical phase tuples and all event counts/timestamps remain unchanged. Root owns reporter predicate, callback plumbing at BaseClusterScheduler/schedule_layer_wave, and actual projection. |

## Bounded routed-lane repair design and exact interfaces

Assigned implementation files: `frontier/events/prefill_sync_event.py`, `frontier/events/decode_sync_event.py`, `frontier/scheduler/utils/sync_entry.py`, `frontier/scheduler/utils/ep_wave.py`, and `frontier/scheduler/utils/expert_parallel.py`.

- Both existing sync-event handlers pass `metrics_store=metrics_store` to `on_prefill_sync` / `on_decode_sync`.
- `enter_prefill_sync` / `enter_decode_sync` accept keyword-only `metrics_store=None` and forward it to `_on_*_ep_wave_ready`.
- `prepare_moe_wave`, `prepare_moe_wave_from_inputs`, and `predict_ep_wave_phase_times` accept `capture_lane_timings: bool=False`.
- Frozen `EPWaveLaneTiming(ep_id, batch, execution_time)` holds the actual physical lane and finalized singleton stage. `EPWavePhaseTimes` appends `lane_records: tuple[EPWaveLaneTiming,...]=()`. Source search found named-attribute consumers, not tuple-unpacking consumers. With capture disabled, no lane-record list is allocated and no timing object is retained after its numerical phase extraction.
- Root's `ep_wave_reporting_enabled` predicate is trace enabled OR write-metrics AND (operation metrics OR ledger requested). Root consumes the existing `EPWavePlan` once in `schedule_layer_wave`, not once per source batch and not by adding DES events.

Real phase starts are pre-dispatch=`wave start`, dispatch=`wave start + max(pre-dispatch)`, routed=`dispatch barrier end`, combine=`dispatch barrier end + max(routed)`, and post-combine=`combine barrier end`. Each lane's operator durations come from its actual payload; waits are gaps. Do not synthesize a lane by choosing independent per-operator maxima, or replay generic serial stage timing across collective barriers. Reuse the existing family/memory projection and trace metadata machinery, with a phase view owned by ExecutionTime matching its existing five phase getters. Wave-lane records must omit full-stage CPU/PP/draft ownership already reported by the stage. Root owns this remaining implementation and its validation.

## Verification evidence and limits

This lane ran no CPU tests during root's performance window. Earlier evidence remains explicitly attributed:

- `test_report_2026-09-16_w02_lifecycle.md`: 26 PASS / 7.91 s, then 185 PASS / 13.73 s; real constructor/automatic capacity, rollback, continuation and slot reuse. Raw logs `/data/ycfeng/tmp/pr33-w02-fixed.log` and `/data/ycfeng/tmp/pr33-w02-regression.log`.
- `test_report_2026-09-16_w03_fixture_migration.md`: separately executed 203 PASS / 4.26 s and 45 PASS / 3.19 s; not a combined full-suite result.
- `w04_cache_review.md`: 52 PASS / 3.79 s, exact stage-local numerical reuse and per-layer routing; no performance acceptance inferred.
- `w08_reporting_review.md`: original isolated eight-case pass, 2.79 s. Later combined root log `/data/ycfeng/tmp/pr33-w08-e2e-final.log` showed 7 FAIL / 15 PASS due to fixture-global `IS_MOE` leakage. The new test file now uses normal `global_vars.reset_global_vars()` setup/teardown, matching existing repository test practice.
- After the window, the updated focused command `PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 python -m pytest tests/unit/test_stage_reporting_contract.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w08-reporting-final --tb=short` observed **11 PASS in 2.96 s**, covering F1, isolation, and conditional actual-lane retention.
- The transport integration command using `test_forward_sync_state.py`, `test_prefill_ep_wave_materialization.py`, `test_decode_ep_wave_materialization.py`, and `test_replica_identity_contract.py` initially observed **40 PASS / 4 FAIL in 3.16 s**, all four failures caused by root's not-yet-landed BaseClusterScheduler `metrics_store` keyword interfaces. This is visible intermediate integration evidence, not a waived failure. The exact command used the same environment and `--basetemp /data/ycfeng/tmp/pr33-w08-ep-transport --tb=short`.

Environment for this lane's observed tests: `/usr/bin/python`, Python 3.12.3, local CPU, no conda activation. No native hardware claim, performance result, whole-suite result, or overall W11 completion is made.

## Responsibility and split decision

`metrics_store.py` remains a large metrics/export owner. The bounded W08 seam is the established per-layer family/operator projection plus explicit external-schema adaptation; extracting a generic metrics engine would duplicate ownership. The physical EP-wave reporter is a narrow additional projection with real phase identity, and should share those views rather than introduce another execution IR. The VLLM-v1 scheduler remains the owner of queue admission, KV/state transaction, and supported continuation semantics; a future lifecycle extraction is coherent only after the transaction boundary and its callers move together. BaseClusterScheduler already delegates pure timing and wave helpers; optional reporting transport belongs in those existing helpers. Existing source stays functionally grouped; unrelated formatter churn is deferred. Exact baseline/current LOC are recorded below.

## Original inventory disposition

All **72 original runtime hunks** are mapped below. A range lists every consecutive H ID; it is not a sampling claim. Original bodies superseded by the current contract are adjudicated against that replacement, including deletion of mutation tracking and broad private delegation. Original ledger rows remain preserved for audit.

| Original IDs | File | Disposition | Contract evidence |
| --- | --- | --- | --- |
| H059–H060 (2) | `frontier/entities/__init__.py` | RETAIN | E0, findings above |
| H061–H091 (31) | `frontier/entities/execution_time.py` | BUGFIX / REFACTOR / DELETE | E1, findings above |
| H092 (1) | `frontier/entities/stage_execution_time.py` | REFACTOR | E2, findings above |
| H216 (1) | `frontier/metrics/constants.py` | RETAIN | M0, findings above |
| H217–H227 (11) | `frontier/metrics/metrics_store.py` | BUGFIX / REFACTOR | M1–M3, findings above |
| H228 (1) | `frontier/metrics/op_trace_utils.py` | RETAIN | M0, findings above |
| H426–H427 (2) | `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | RETAIN / BUGFIX | S2, findings above |
| H428–H436 (9) | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | RETAIN / BUGFIX; F3 cleanup | S3, findings above |
| H437–H450 (14) | `frontier/scheduler/utils/memory_planner.py` | RETAIN / BUGFIX; F4 limited scope | S2, findings above |

## Exact current hunk checkpoint

Snapshot captured after the conditional lane transport and root wave-call interfaces landed, before completion of the new MetricsStore wave reporter. Every listed header is from `git diff --unified=0 0515589 -- <scope>`. Subsequent root reporter edits require the final owner reconciliation.

| File | Group | Exact zero-context hunk headers |
| --- | --- | --- |
| `frontier/entities/__init__.py` | E0 | `@@ -4,0 +5 @@`; `@@ -19,0 +21 @@` |
| `frontier/entities/execution_time.py` | E1 | `@@ -2 +2 @@`; `@@ -27,0 +28,26 @@`; `@@ -30 +56 @@`; `@@ -36,2 +62,3 @@`; `@@ -97,0 +125,3 @@`; `@@ -100,0 +131,26 @@`; `@@ -101,0 +158,4 @@`; `@@ -470,0 +531 @@`; `@@ -478,0 +540 @@`; `@@ -501,6 +562,0 @@`; `@@ -584,0 +641 @@`; `@@ -607,0 +665 @@`; `@@ -627,0 +686 @@`; `@@ -653,0 +713 @@`; `@@ -674,0 +735 @@`; `@@ -692,0 +754 @@`; `@@ -737 +799 @@`; `@@ -751 +813 @@`; `@@ -763 +825 @@`; `@@ -801,6 +863,2 @@`; `@@ -808,28 +866,13 @@`; `@@ -862,11 +905,2 @@`; `@@ -882,12 +916,2 @@`; `@@ -1043,2 +1067,89 @@`; `@@ -1051 +1162 @@`; `@@ -1061 +1172 @@`; `@@ -1071 +1182 @@`; `@@ -1080 +1191 @@`; `@@ -1086 +1197 @@`; `@@ -1091 +1202 @@`; `@@ -1101 +1212 @@`; `@@ -1109 +1220 @@`; `@@ -1117 +1228 @@`; `@@ -1122 +1233 @@`; `@@ -1130 +1241 @@`; `@@ -1138 +1249 @@`; `@@ -1146 +1257 @@`; `@@ -1154 +1265 @@`; `@@ -1162 +1273 @@`; `@@ -1170 +1281 @@`; `@@ -1178 +1289 @@`; `@@ -1186 +1297 @@`; `@@ -1194 +1305 @@`; `@@ -1202 +1313 @@`; `@@ -1210 +1321 @@`; `@@ -1218 +1329 @@`; `@@ -1235 +1346 @@`; `@@ -1243 +1354 @@`; `@@ -1320 +1431 @@`; `@@ -1325 +1436 @@`; `@@ -1333 +1444 @@`; `@@ -1343 +1454 @@`; `@@ -1354 +1465 @@`; `@@ -1370 +1481 @@`; `@@ -1380 +1491 @@`; `@@ -1390 +1501 @@`; `@@ -1400 +1511 @@`; `@@ -1410 +1521 @@`; `@@ -1420 +1531 @@`; `@@ -1454 +1565 @@` |
| `frontier/entities/stage_execution_time.py` | E2 | `@@ -0,0 +1,436 @@` |
| `frontier/events/decode_sync_event.py` | T0 | `@@ -63,0 +64 @@` |
| `frontier/events/prefill_sync_event.py` | T0 | `@@ -63,0 +64 @@` |
| `frontier/metrics/constants.py` | M0 | `@@ -22,0 +23,4 @@`; `@@ -44,0 +49,2 @@` |
| `frontier/metrics/metrics_store.py` | M1–M3 | `@@ -11 +11 @@`; `@@ -15,0 +16 @@`; `@@ -79,0 +81,12 @@`; `@@ -105,0 +119,25 @@`; `@@ -565,2 +603,5 @@`; `@@ -687 +728,7 @@`; `@@ -695 +742,31 @@`; `@@ -987,0 +1065,133 @@`; `@@ -3524,0 +3735,73 @@`; `@@ -3601,13 +3884,11 @@`; `@@ -3623 +3904,7 @@`; `@@ -3646 +3933 @@`; `@@ -3648,0 +3936,12 @@`; `@@ -3828,11 +4127,10 @@`; `@@ -3859,0 +4158,35 @@` |
| `frontier/metrics/op_trace_utils.py` | M0 | `@@ -533,0 +534,14 @@` |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` | S0 / T0 | `@@ -132 +132 @@`; `@@ -971 +971 @@`; `@@ -989,0 +990 @@`; `@@ -992 +993 @@`; `@@ -1004,0 +1006 @@`; `@@ -1043 +1045 @@`; `@@ -1055,0 +1058 @@`; `@@ -1095 +1098 @@`; `@@ -1098 +1101 @@`; `@@ -1146 +1149 @@`; `@@ -1165 +1168 @@`; `@@ -1202,0 +1206 @@`; `@@ -1216,0 +1221 @@` |
| `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | S2 | `@@ -60,0 +61,3 @@`; `@@ -64,0 +68,5 @@`; `@@ -387,0 +396,20 @@` |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | S3 | `@@ -28,0 +29,2 @@`; `@@ -129,0 +132,18 @@`; `@@ -1403,0 +1424,3 @@`; `@@ -1405,0 +1429,3 @@`; `@@ -1412,0 +1439,6 @@`; `@@ -2821,0 +2854,8 @@`; `@@ -2946,0 +2987,10 @@`; `@@ -2953,0 +3004,7 @@`; `@@ -3050,0 +3108,8 @@`; `@@ -3691 +3756,16 @@`; `@@ -3707,14 +3786,0 @@` |
| `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py` | S1 | `@@ -400 +400,3 @@` |
| `frontier/scheduler/utils/decode_collective.py` | S1 | `@@ -149 +149 @@`; `@@ -182,2 +182,2 @@`; `@@ -185 +184,0 @@` |
| `frontier/scheduler/utils/dense_metrics.py` | S0 | `@@ -6,0 +7,2 @@`; `@@ -7,0 +10 @@`; `@@ -43 +46 @@`; `@@ -49,0 +53 @@`; `@@ -56,3 +60,3 @@`; `@@ -114,13 +118,7 @@`; `@@ -128,11 +126,11 @@` |
| `frontier/scheduler/utils/ep_wave.py` | T0 | `@@ -43,0 +44 @@`; `@@ -71,0 +73 @@`; `@@ -91,0 +94 @@`; `@@ -131,0 +135 @@` |
| `frontier/scheduler/utils/ep_wave_schedule.py` | T0 | `@@ -24,0 +25 @@`; `@@ -67,0 +69 @@`; `@@ -68,0 +71,5 @@` |
| `frontier/scheduler/utils/execution_time_metrics.py` | S0 | `@@ -1 +1,3 @@`; `@@ -3,0 +6,7 @@`; `@@ -4,0 +14,9 @@`; `@@ -6,2 +23,0 @@`; `@@ -9,53 +25,11 @@` |
| `frontier/scheduler/utils/expert_parallel.py` | T0 | `@@ -10,0 +11 @@`; `@@ -14 +15 @@`; `@@ -56,0 +58,9 @@`; `@@ -65,0 +76 @@`; `@@ -91,0 +103 @@`; `@@ -94,0 +107 @@`; `@@ -109,0 +123,4 @@`; `@@ -124 +141,4 @@` |
| `frontier/scheduler/utils/memory_planner.py` | S2 | `@@ -4 +4,2 @@`; `@@ -18,0 +20 @@`; `@@ -22,0 +25,3 @@`; `@@ -97 +102,20 @@`; `@@ -119 +143 @@`; `@@ -130 +154,2 @@`; `@@ -132 +157 @@`; `@@ -133,0 +159,44 @@`; `@@ -153 +222,3 @@`; `@@ -178 +249,4 @@`; `@@ -187,0 +262 @@`; `@@ -194,0 +270,3 @@`; `@@ -196 +274 @@`; `@@ -210 +288,3 @@` |
| `frontier/scheduler/utils/prefill_collective.py` | S1 | `@@ -90 +90 @@`; `@@ -229,0 +230,7 @@`; `@@ -231 +238 @@`; `@@ -233 +240 @@` |
| `frontier/scheduler/utils/sync_entry.py` | T0 | `@@ -26,0 +27,2 @@`; `@@ -148,0 +151 @@`; `@@ -161,0 +165,2 @@`; `@@ -282,0 +288 @@` |

Checkpoint covers **163 current hunks in 21 production files**.

## Module size and seam assessment

| File | Baseline lines | Current lines | Existing responsibility / split action |
| --- | ---: | ---: | --- |
| `frontier/entities/execution_time.py` | 1493 | 1604 | E1; bounded owned projection/lifecycle seams described above |
| `frontier/entities/stage_execution_time.py` | 0 | 436 | E2; bounded owned projection/lifecycle seams described above |
| `frontier/metrics/metrics_store.py` | 5270 | 5603 | M1–M3; bounded owned projection/lifecycle seams described above |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` | 1871 | 1876 | S0 / T0; bounded owned projection/lifecycle seams described above |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | 5073 | 5139 | S3; bounded owned projection/lifecycle seams described above |

## Later integration checkpoint

The exact five-file command adds `tests/unit/test_stage_reporting_contract.py` to the four transport files listed above, with `--basetemp /data/ycfeng/tmp/pr33-w08-ep-transport-final --tb=short`. Observed **53 PASS / 2 FAIL in 3.75 s**. Both remaining failures are existing placeholder-consumption test fixtures passing a `SimpleNamespace` metrics store without the new explicit `ep_wave_reporting_enabled` property. The source interfaces now connect; root was asked to migrate the two fixtures to `ep_wave_reporting_enabled=False`, not introduce a production duck-typing fallback. Requested-reporting integration remains pending the root-owned reporter.

## Completed transport fixture migration and phase-table reassessment

The two authorized placeholder tests now explicitly pass `SimpleNamespace(ep_wave_reporting_enabled=False)` at the late sync event boundary. No production compatibility fallback was added. The combined command is:

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 python -m pytest tests/unit/test_forward_sync_state.py tests/unit/test_prefill_ep_wave_materialization.py tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_replica_identity_contract.py tests/unit/test_stage_reporting_contract.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w08-ep-transport-fixtures --tb=short
```

Observed **55 PASS in 4.68 s**, superseding the 53/2 fixture integration checkpoint. Environment remains `/usr/bin/python`, Python 3.12.3, local CPU without conda activation.

An independent direct numerical experiment extracted the five prior phase method definitions from `git show HEAD:frontier/entities/execution_time.py` with Python AST and executed those old functions against the current normally constructed `ExecutionTime` objects. This isolates phase decomposition from the separately approved uniform single-layer contract. Current `_MOE_PHASE_COMPONENTS` uses the same component/operator-map precedence as the old methods: public gating getters read the MoE component, not stale flat `_moe_gating_*` fields; allgather uses the same `tensor_parallel_allgather_time` alias; routed allreduce resolves through the same `_get_moe_tp_allreduce_time` helper. Dispatch/combine still require exact structured communication names.

| Case | Old pre-dispatch ms | New pre-dispatch ms | Evidence |
| --- | ---: | ---: | --- |
| Legacy gating 14, no split arguments | 14 | 14 | Existing constructor splits the component into 7 + 7. |
| Explicit split 3 + 5, legacy 99 | 8 | 8 | Supplied split remains authoritative. |
| Only explicit linear 3, legacy 99 | 3 | 3 | Existing legacy fallback requires both split fields zero. |
| Structured split 3 + 5, legacy 99 | 8 | 8 | Canonical operator map wins. |
| Structured zero split, legacy 14 | 0 | 0 | Canonical zero wins over constructor component fallback in both old and new getters. |
| Legacy override gating 14, shuffling 2 | 16 | 16 | Override's effective component values are preserved, despite legacy raw scalar fields. |

Every case also compared dispatch, routed, combine, and post-combine; all five phases matched exactly. Dispatch/combine were 17/19 ms; the override case used routed GEMM 23 ms. A further 100 deterministic mixed positive scalar cases (`random.Random(173)`, each phase component uniform `[0,10)`) matched to `1e-12 ms`; maximum observed difference was **1.4210854715202004e-14 ms**. This is a real floating summation-order difference: the old pre-dispatch expression first groups linear+topk, whereas the new flattened table sums them in sequence. No physical-duration or ownership change was observed. Bit-for-bit timing identity is therefore not claimed.

Reachable producers inspected: `sklearn_moe_execution_time_predictor._get_execution_time_internal` passes explicit split fields for real predictions; its dummy path passes legacy aggregate gating; base/disaggregation dummy constructors also pass legacy gating. `override_moe_times` currently has unit-test callers and no production caller found. All those supported forms are numerically covered above. This review found no reachable gating semantic regression and made no production edit. The new reporter itself remains root-owned and must be checked independently for event identity, phase timestamps, metadata, and export cardinality.

## Final EP reporter and remaining original runtime IDs

At the final reporter review checkpoint (`5a1cc2af` plus working edits), `frontier/metrics/ep_wave_metrics.py` is a coherent extracted projection owned by the existing MetricsStore entrypoint. It consumes the exact lane records retained by the existing wave plan; it does not predict additional work or introduce events. Five phase starts match the existing barrier timing. Each lane cursor advances only by its actual phase operators, leaving barrier waits as gaps. CPU, PP, draft and terminal stage owners do not appear in the lane phase table. Ledger rows are explicitly `ep_wave_lane`, carry actual IDs/source batches/phase scopes, and export separately to `frontier_ep_wave_lane_ledger.jsonl`; they are not injected into the full-stage sum. Operation insertion is independent of utilization; trace-only reporting remains available when write-metrics is false. New DP input/output communication shape cases reuse the existing token×hidden collective convention in `op_trace_utils.py`. These additions are **BUGFIX / REFACTOR** under M1–M3/T0, with the phase table under E1.

The expansion decision is now remembered by actual source batch IDs in `_expanded_trace_batches`. EP and attention-stage projections consult the same decision, so a wave cannot consume the request quota and suppress its own attention trace. New independent tests exercise both arrival orders with quota one, source batch 73, and subsequent unrelated batch 74: both projections for 73 retain actual layer IDs, while 74 emits three family aggregates with layer_id=-1. This is actual output validation, not an internal-set assertion.

Exact command: `PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 python -m pytest tests/unit/test_stage_reporting_contract.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w08-quota-review --tb=short`. Observed **14 PASS in 3.21 s** (Python 3.12.3 CPU). Root's separately attributed combined reporting/hybrid execution observed **18 PASS in 12.06 s** at `/data/ycfeng/tmp/pr33-w08-reporting-hybrid-2.log`; those 18 precede these two appended quota cases. No whole-suite or final W10 inference is made.

The following seven original entries extend the initial directory scope. Together with the earlier 72, this lane now adjudicates **79 original H IDs**.

| Original ID | Exact baseline/current hunk | Disposition | Caller and evidence |
| --- | --- | --- | --- |
| H214 | `analytical_kv_cache_transfer_predictor.py @@ -3,0 +4 @@` | RETAIN | Imports the existing shared GDN runtime guard, without a parallel classifier. |
| H215 | `analytical_kv_cache_transfer_predictor.py @@ -52,0 +54 @@` | RETAIN | `_calculate_kv_cache_size_for_tokens` rejects unsupported recurrent P-to-D state transfer before selecting ordinary KV family/layout; normal dense/MLA paths remain unchanged. Guard tests and supported sequential PDD fixtures cover this boundary. |
| H451 | `simulator.py @@ -66,0 +67,6 @@` | RETAIN | Normal Simulator construction supplies cluster replica count before predictor/routing-domain construction. Real constructor E2Es consume this topology. |
| H452 | `simulator.py @@ -182 +188,4 @@` | RETAIN | Documents the real hybrid-only shared-manager branch and preserves ordinary monolithic independent training. |
| H453 | `simulator.py @@ -186,0 +196,18 @@` | RETAIN | Hybrid GDN constructor loads typed ordinary/GDN artifacts through the existing manager and its canonical training path API. Homogeneous monolithic models retain the established path. W09 and hybrid normal-constructor tests observe real loading and execution. |
| H454 | `simulator.py @@ -196 +223 @@` | RETAIN | Passes the selected manager through the existing predictor registry; no extra training framework. |
| H455 | `simulator.py @@ -197,0 +225,2 @@` | BUGFIX / RETAIN | Passes canonical paths and real cluster replica keys, preserving the independently reproduced actual-replica-ID routing repair; RF wrapper forwards actual IDs only to compatible MoE/disaggregation constructors. |

F3 is now **resolved**: normal constructor-owned `_gdn_state_slot_manager` reads replace all six fallback reads, with legitimate fixture initialization updates. F4 is **withdrawn as a reachable production defect**: the schedule helper alone accepts all-GDN descriptors, but the normal runtime model constructor reaches `resolve_runtime_attention_family`, which rejects an empty full-attention family set. The earlier inference stopped too early in construction. No pure-GDN support expansion or new capacity workaround is warranted. These corrections preserve the original intermediate findings rather than silently rewriting them.

## c9f8f904 final homogeneous assembly check

Read-only independent review of the exact homogeneous aggregation change and existing `from_execution_time` contract found no correctness issue. Every physical model spec is resolved and range-validated before the fast path; same-object identity plus family/variant equality prevents conflating distinct routed layers; finalized expansion preserves separate actual identity records, once-only stage owners and mutable-source isolation. The existing four-role independent construction oracle and new calculation-count/range/mixed-identity tests are appropriate. The 132-pass focused result is attributed to `test_report_2026-09-16_predictor_owner_once.md`; this lane ran no checks or benchmarks in the final timing window. Full explanation and updated final-source CPU/non-dummy/hunk status are in `review_reassessment_2026-09-16.md`.

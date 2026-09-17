## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Enumerated every production diff path at the frozen revisions; ranked dependency and risk. |
| 2026-09-17 | Reconciled every frozen path with inspected contracts, completed cleanup evidence, retained rationale and explicit remaining work. |
| 2026-09-17 | Closed A5 against commit 5cb8794f and its focused report; advanced the frozen cleanup checkpoint to 38 cleaned / 68 retained paths. |

# Frozen diff inventory

Comparison: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2...c288a19f59bec09529ee18d782fa57218da2c781`.

Ranks: R1 = runtime contracts; R2 = metrics/reporting; R3 = profiling; R4 = training/config. Original ranks and frozen numstat values are retained; a config path's R4 label does not exclude its runtime impact.

## Reconciliation scope and interpretation

The inventory covers exactly the **106 changed paths under `frontier/`: 105 Python files and one profiling README**. It is not an inventory of every test, asset or task document changed by the branch. Reconciliation source checkpoint: `5cb8794f49fb3e27e84316a3f87ebadbcb9d1574`; production/tests are frozen for main's final regression. The original frozen comparison above remains unchanged.

- **C**: inspected, with an implemented cleanup and linked verification. The reason column also states important retained behavior. This does not mean every unchanged line in the module is debt-free.
- **R**: inspected and retained, with a concrete contract/boundary reason; no production cleanup in this task at the checkpoint. R is not an inference from test counts.
- **A5 completed**: the duplicated default is now reused from its existing owner; the linked gate-default report supersedes the earlier pending status.
- Review links establish hunk/caller inspection; test-report links establish only the checks actually described there. Group regression is not asserted as a separate test for every row or as native GPU evidence.
- [Record][REC] is the shared issue/retained-pattern record. [Standards][STD], [Attention][ATT], [Metrics][MET], [Large modules][LARGE] and [Spec][SPEC] contain the original issue IDs and detailed evidence. Later implementation reports supersede their historical pending wording. This reconciliation changes no production/tests and runs no suite.

## Per-path disposition

| File | Added | Removed | Rank | Disposition / inspected contract | Review, refactor and verification evidence |
| --- | ---: | ---: | --- | --- | --- |
| `frontier/attention/__init__.py` | 10 | 0 | R1 | R — exports canonical family, layer binding and runtime-family interfaces; no independent policy. | [ATT], [P1 identity]; export hunk rechecked. |
| `frontier/attention/families.py` | 85 | 0 | R1 | R — authoritative GDN schema, phase operators and singleton variant; consumers now reuse this declaration. | [STD] S4, [ATT] A4, [P3 schema], [P3 small]. |
| `frontier/attention/gdn/__init__.py` | 38 | 0 | R1 | R — CPU-safe semantic exports, including the supported identity adapter; no kernel dependency. | [ATT], [P1 identity]; export hunk rechecked. |
| `frontier/attention/gdn/config.py` | 265 | 0 | R1 | C — A1/A4 central identity and family ID; A5 shape resolver uses GatedDeltaNetConfig.output_gate_type. Retain external missing-field adaptation, partial-shape validation, schedule precedence and cycle-safe exported adapter. | [ATT] A1/A4/A5, [REC], [P1 identity], [P4 gate defaults]. |
| `frontier/attention/gdn/features.py` | 252 | 0 | R1 | C — use required Batch/Request members; task enumeration remains family-derived. Physical versus logical sizes and phase/state validation are real. | [REC] GDNBatchFeatures, [P1c], [STD] S4, [P3 schema]. |
| `frontier/attention/gdn/guards.py` | 48 | 0 | R1 | C — direct required model capability; preserve fail-fast unsupported PP/PD/EP/DP/prefix/spec and state-drop paths. | [REC] runtime guards, [SPEC] D03, [P1c], [P1 fixtures]. |
| `frontier/attention/gdn/memory.py` | 77 | 0 | R1 | R — immutable fixed-state shapes/byte widths; positivity/type checks validate a public value contract, not absent members. History-independent bytes retained. | [ATT] shape contract, [SPEC], [P1b]; full new module rechecked. |
| `frontier/attention/gdn/state.py` | 68 | 0 | R1 | C — heap replaces repeated sorting/front removal; request ownership, minimum-free-ID allocation and idempotent release retained. | [REC] slot manager, [P1e] exact 1,000-request transcript. |
| `frontier/attention/model_binding.py` | 113 | 0 | R1 | C — existing homogeneous binder avoids re-entry; family ID and singleton variant come from family owner. Whole-model rejection differs intentionally from runtime KV-family selection. | [ATT] A2/A4, [P1 identity], [P3 small]; cached-getter fixture corrected at actual resolver. |
| `frontier/attention/ops.py` | 1 | 0 | R1 | R — FIXED_STATE is a declared memory-layout enum member, not another family classifier. | [STD] associated registries, [P3 schema]. |
| `frontier/attention/profiling_mapping.py` | 23 | 0 | R1 | R — reuse GDN feature columns; selected external identity validation and distinct ordinary/shared feature APIs remain legitimate boundaries. | [STD] associated registries, [P3 schema]; identity and FIXED_STATE hunks rechecked. |
| `frontier/attention/trace_mapping.py` | 2 | 8 | R2 | R — structured times already represent one physical layer; removing stage multiplication is approved D01, not cleanup to reverse. | [SPEC] D01, [P2 integrated]; exact scaling hunk rechecked. |
| `frontier/config/config.py` | 62 | 0 | R4 | C — required world size, node capacity and speculative config accessed directly; optional roles/scheduler capabilities and guard ordering retained. Single predictor GDN path default remains here. | [STD] S2/S8, [LARGE] config split analysis, [P4 config], [P4 manager]. |
| `frontier/config/device_sku_config.py` | 14 | 0 | R4 | R — declarative MI355X SKU and canonical gpu_platform; no consumer-local device classification needed. Hardware values not remeasured. | [STD] per-module inventory and retained device boundary, [P3 integrated] CPU limit. |
| `frontier/config/model_config.py` | 156 | 8 | R4 | C — A5 runtime field/HF loader reuse the shape-owned gate default. Required supplied-model interface, topology/shape caches and profile snapshot retained; lazy resolution preserves validation timing and replace behavior. | [ATT] A3/A5 and cache/mutation matrix, [P1 identity], [P4 GDN], [P4 gate defaults]. |
| `frontier/config/node_sku_config.py` | 12 | 0 | R4 | R — declarative MI355X UBB capacity is separate from TP and supplies required node capacity to config guards. | [STD] S8/inventory, [P4 config] constructed node/topology cases. |
| `frontier/config/quantization_manager.py` | 50 | 6 | R4 | C — family-owned GDN operator set and direct supplied-model capability. Preserve null-versus-empty selectors and emitted identity. | [STD] S4, [REC], [P3 schema] exact four-name set. |
| `frontier/config/utils.py` | 6 | 0 | R4 | R — omit internal caches from serialization; no duplicate serializer or altered public identity. | [STD] inventory, [ATT] snapshot/schema exclusion, [P3 model]. |
| `frontier/entities/__init__.py` | 2 | 0 | R1 | R — exports StageExecutionTime through the existing entities package; no parallel timing implementation. | [SPEC] timing contract, [P1a]; exact export hunk rechecked. |
| `frontier/entities/execution_time.py` | 259 | 149 | R1 | C — direct constructor-owned component snapshots and removal of unread aggregate bookkeeping. Keep real optional maps, pre-binding identity and finalized copy isolation. | [REC], [SPEC] D01, [P1a], [P1d], [P2 scheduler]. |
| `frontier/entities/stage_execution_time.py` | 436 | 0 | R1 | C — remove redundant None filtering after identity validation. Ordered real layers, once-only stage owner, bounded scalar projection and singleton-only probes remain explicit contracts. | [REC], [SPEC] D01, [P1a], [P2 integrated]. |
| `frontier/events/decode_sync_event.py` | 1 | 0 | R1 | R — forwards the existing metrics sink into decode sync; reporting demand reaches EP capture without changing scheduling. | Exact one-line hunk and sync_entry consumer rechecked; [MET] sync entry, [P2 integrated]. |
| `frontier/events/prefill_sync_event.py` | 1 | 0 | R1 | R — forwards the existing metrics sink into prefill sync; no additional event state or resolver. | Exact one-line hunk and sync_entry consumer rechecked; [MET] sync entry, [P2 integrated]. |
| `frontier/execution_time_predictor/attention_tp_policy.py` | 2 | 2 | R1 | R — consumes the renamed required attention_linear_ops declaration; classification stays in architecture registry. | Exact rename hunk rechecked; [ATT] operator contracts, [P4 config]. |
| `frontier/execution_time_predictor/base_execution_time_predictor.py` | 38 | 4 | R1 | R — _assemble_stage binds model-owned IDs and shares only identical numerical objects with identical family/variant. Dummy results are single-layer; retired DP scalar seam stays zero. | [SPEC] assembly, [P1a], [P1 private], [P2 trace]; assembly and dense-scaling oracle tests rechecked. |
| `frontier/execution_time_predictor/cache_io.py` | 32 | 4 | R4 | R — atomic pickle/JSON publication and reuse fingerprint; exception cleanup removes temporary output and rethrows, not fallback success. | [STD] retained atomicity, [P4 GDN] artifact/cross-load checks. |
| `frontier/execution_time_predictor/gdn_predictor.py` | 318 | 0 | R4 | C — task registry and direct supplied-model methods/embedding_dim; retain whole-model None, legitimate absent GDN shape, selectors and external artifact validation. | [STD] S1/S4, [REC], [P3 schema], [P4 GDN] 56 file pairs / 192 predictions. |
| `frontier/execution_time_predictor/measurement_input_paths.py` | 163 | 0 | R4 | C — existing substitution owner reused by GDN manager. Optional network substitution preserves compute-only literals; empty/absent override and legacy tuple precedence remain distinct. | [STD] S2, [P4 manager] M3 exact path checks. |
| `frontier/execution_time_predictor/random_forrest_execution_time_predictor.py` | 10 | 3 | R1 | R — pass actual process-global Replica IDs to monolithic MoE as well as disaggregation; cluster_config remains disaggregation-only. | Exact constructor forwarding hunk rechecked; [SPEC] caller integration, [P1d], [P1 fixtures]. |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | 176 | 100 | R4 | C — canonical runtime-family/device/path inputs and constructor-owned registries/cluster map. Retain dataset-only None and distinct typed, legacy, precision and measurement identities. | [STD] S2/S3, [LARGE] L1 and split sequence, [REC], [P4 manager]. |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | 107 | 180 | R1 | C — shared routing and private single-layer contract; preserve role admission, zero-work EP barrier, normalization order and once-only CPU/PP/MTP work. | [REC] MoE predictor, [LARGE] L2/split, [P1d], [P1 private]. |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 157 | 209 | R1 | R — candidate homogeneous numerical reuse and private layer construction already remove recursive public prediction. GDN dispatch, optional artifact state and real MTP stage width retained. | [LARGE] base predictor, [SPEC] caches/ownership, [P1 private], [P2 integrated]. |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | 202 | 141 | R1 | C — direct initialized capacity, one routing owner, role override and single-layer private interface. Keep 256-entry workload LRU and stage-local attention copies; they have different identities/lifetimes. | [REC], [LARGE] L2/caches, [P1d], [P1 private]. |
| `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py` | 2 | 0 | R1 | R — canonical GDN PD guard precedes byte accounting; do not invent recurrent-state transfer support or alter dense bytes. | Exact guard hunk rechecked; [SPEC] D03, [P1 fixtures]. |
| `frontier/metrics/constants.py` | 6 | 0 | R2 | R — existing enum/schema holds new labels. Retired DP public zero timings/labels are not authorization for positive tensor traces. | [MET] inventory, [P2 trace] DP RCA, [REC] retained FFN labels. |
| `frontier/metrics/ep_wave_metrics.py` | 86 | 0 | R2 | C — phase-local context/parallel metadata and trace-only lane sum; keep zero-work/reporting-off gates and per-event copies/per-operator shapes. | [MET] M5, [REC], [P2 trace] 64-case equality and call counts. |
| `frontier/metrics/metrics_store.py` | 403 | 33 | R2 | C — remove dead trace readers; restore explicit residual metadata versus additive layer identity. Retain scalar/Stage contracts, expansion/ledger lifecycle and FFN label adaptation. | [MET] S1/M1/M3, [REC], [P2 residual], [P2 readers], [P2 integrated], [LARGE] split. |
| `frontier/metrics/op_trace_utils.py` | 16 | 0 | R2 | C — GDN membership uses family lookup; remove unreachable retired-DP shape names, not the public zero timing seam. Tensor metadata unchanged. | [MET] M4, [REC], [P2 trace] four-op metadata and retired-zero test. |
| `frontier/model_architectures.py` | 142 | 14 | R1 | C — architecture-local nonrecursive identity and exact raw-type native policy. Preserve distinct normalization/explicit-profile rules, structural errors and warning snapshot behavior. | [ATT] A1 matrix, [LARGE] S6 addendum, [P1 identity], [P3 architecture]. |
| `frontier/moe_ep_workload.py` | 78 | 10 | R1 | C — existing Hamilton histogram operation now shared with experimental replay; independent physical-layer routing and arithmetic/tie order retained. | [STD] S7, [LARGE] workload ownership, [P3 routing] exact workloads/replay dictionaries. |
| `frontier/operators/binding.py` | 5 | 5 | R1 | R — required declaration rename propagated; optional profile for generic operators and explicit malformed external-profile diagnostic remain public boundary behavior. | [ATT] operator aliases/optional profile, [STD], [P4 config]; changed hunk rechecked. |
| `frontier/operators/typed_contracts.py` | 11 | 11 | R1 | C — direct required declaration during internal registry iteration. Retain validation of external profile inputs and supported typed/untyped CSV distinctions. | [STD] smaller observation 5, [P4 config] 66 metadata outcomes. |
| `frontier/profiling/attention/backends/__init__.py` | 14 | 0 | R3 | R — explicit VLLM_ROCM admission/lazy backend loading preserves CPU imports. | [STD] inventory, [P3 orchestration]. |
| `frontier/profiling/attention/backends/vllm_rocm_attention_wrapper.py` | 329 | 0 | R3 | C — reuse admitted single-phase sequence plan. Dual metadata, empty/before-begin/after-end states and both timer scopes are real lifecycle semantics. | [STD] J1, [REC], [P3 orchestration] exact metadata/slots/output/timer snapshots. |
| `frontier/profiling/collectives/benchmark_runner.py` | 13 | 6 | R3 | C — reuse existing collective environment setup owner before initialization; nonparticipating-rank None retained. | [STD] smaller observation 3, [P3 small] four environment values and device call. |
| `frontier/profiling/collectives/collectives_impl.py` | 5 | 0 | R3 | R — explicit dtype propagation through existing collective dispatch; no new platform policy. | [STD] inventory, [P3 small], [P3 integrated]. |
| `frontier/profiling/collectives/collectives_input.py` | 32 | 4 | R3 | R — precision-aware input/schema validation protects external rows rather than incomplete internal state. | [STD] inventory, [P3 integrated]. |
| `frontier/profiling/collectives/collectives_wrapper.py` | 7 | 2 | R3 | R — forwards native dtype/backend explicitly; separate wrapper still owns existing collective call contract. | [STD] inventory, [P3 small], [P3 integrated]. |
| `frontier/profiling/collectives/main.py` | 198 | 55 | R3 | R — existing setup owner is reused by runner; CPU planning/native dispatch, rank errors and group ownership remain coordinated here. | [STD] collective boundary, [P3 small], [P3 integrated]. |
| `frontier/profiling/common/accelerator.py` | 232 | 0 | R3 | R — shared device discovery/visibility owner; optional native dependencies and identity-to-SKU resolution are genuine CPU-safe boundary handling. | [STD] platform/optionality inventory, [P3 integrated]. |
| `frontier/profiling/common/constants.py` | 4 | 0 | R3 | R — VLLM_ROCM declared through existing backend enum; no secondary backend registry. | [STD] inventory, [P3 orchestration]. |
| `frontier/profiling/common/cuda_timer.py` | 11 | 106 | R3 | R — thin supported CudaTimer compatibility entry delegates to the common timer; actual callers require the name. | [STD] retained timer contracts, [P3 integrated]. |
| `frontier/profiling/common/device_timer.py` | 130 | 0 | R3 | R — single timer owner; disabled/unnamed/context modes and failed-sample suppression are supported behavior, not missing-constructor workarounds. | [STD] retained timer contracts, [P3 integrated]. |
| `frontier/profiling/common/layers/layernorm.py` | 29 | 4 | R3 | R — old/current vLLM constructor and configuration-context compatibility reflects supported native API differences. | [STD] inventory/native boundary, [P3 architecture], [P3 integrated]. |
| `frontier/profiling/common/layers/rotary_embedding.py` | 54 | 1 | R3 | R — partial rotary and native signature compatibility retained; not an alternative architecture classifier. | [STD] inventory/native boundary, [P3 integrated]. |
| `frontier/profiling/common/model_config.py` | 108 | 2 | R3 | C — canonical runtime family resolver and parsing replace duplicate policy/overlays; A5 constructor reuses shape-owned gate default. Preserve raw type/null/string overlays and mutation rebind. | [STD] S5, [ATT] mutation contract, [REC], [P3 model] 22-model exact snapshot, [P4 gate defaults]. |
| `frontier/profiling/common/timer_stats_store.py` | 22 | 7 | R3 | R — measurement-family labels plus empty/failed-sample behavior are observation schema, not removable optional state. | [STD] inventory, [P3 integrated]. |
| `frontier/profiling/common/vllm_compat.py` | 34 | 0 | R3 | R — shared optional native config-context adapter; retain supported API boundary without experimental dependency. | [STD] inventory/native boundary, [P3 integrated]. |
| `frontier/profiling/experimental/__init__.py` | 1 | 0 | R3 | R — lightweight explicit experimental package boundary. | [STD] inventory and standard/experimental separation. |
| `frontier/profiling/experimental/sglang/__init__.py` | 8 | 0 | R3 | R — CPU-safe experimental exports; no standard profiling dependency introduced. | [STD] inventory, [P3 orchestration]. |
| `frontier/profiling/experimental/sglang/attention.py` | 277 | 0 | R3 | R — logical/physical workload identity and independent reference tensors are required for replay correctness. | [STD] inventory, [P3 orchestration] replay checks, [P3 integrated]. |
| `frontier/profiling/experimental/sglang/dense.py` | 144 | 0 | R3 | R — validated primitive shapes; builder does not return metadata, so existing metadata derivation is not removed by J2. | [STD] inventory, [P3 orchestration] builder-contract evidence. |
| `frontier/profiling/experimental/sglang/gdn.py` | 178 | 0 | R3 | R — cold/carried state, reset and independent references retained; builder already returns the metadata now consumed by J2. | [STD] inventory, [P3 orchestration]. |
| `frontier/profiling/experimental/sglang/gdn_trace.py` | 318 | 0 | R3 | R — external trace validation and carried-state provenance distinguish actual replay workloads; missing/invalid trace checks remain. | [STD] inventory/external-boundary reasons, [P3 integrated]. |
| `frontier/profiling/experimental/sglang/graph_replay.py` | 513 | 0 | R3 | C — explicit profile_graph arguments and named internal replay record replace positional overloading/GDN re-derivation. Snapshot/reset/check owner and optional trace lifecycle retained. | [STD] J2, [REC], [P3 orchestration] 12 exact snapshots / 1,802 events. |
| `frontier/profiling/experimental/sglang/moe.py` | 342 | 0 | R3 | R — routed shape/assignment validation and local assignment builder remain authoritative for native primitive inputs. | [STD] inventory, [P3 routing], [P3 integrated]. |
| `frontier/profiling/experimental/sglang/routed_moe_replay.py` | 269 | 0 | R3 | C — consume existing runtime Hamilton histogram owner after parity proof. Retain exported reconstruction forms and graph owner; no dummy EP state. | [STD] S7 superseded by [P3 routing], [REC]; 893 exact replay dictionaries. |
| `frontier/profiling/gdn/README.md` | 22 | 0 | R3 | R — standard timing/provenance documentation agrees with DEVICE_EVENT versus experimental replay boundary. No README edit. | [STD] inventory; documentation inspection, not a test claim. |
| `frontier/profiling/gdn/__init__.py` | 20 | 0 | R3 | R — lightweight standard GDN exports preserve optional native-import boundary. | [STD] inventory, [P3 schema]. |
| `frontier/profiling/gdn/inputs.py` | 278 | 0 | R3 | C — use family-owned ordered schema and remove zero-mean fallback after positive-query validation. Optional mask and physical-size normalization retained. | [STD] S4, [REC], [P3 schema] exact 27-column tuple. |
| `frontier/profiling/gdn/main.py` | 147 | 0 | R3 | R — campaign preflight and standard output identity remain separate from experimental replay; native execution required for numerical claims. | [STD] inventory, [P3 schema], [P3 integrated]. |
| `frontier/profiling/gdn/vllm_wrapper.py` | 741 | 0 | R3 | R — backend/API admission, rank error aggregation, state allocation/reset and caller-owned versus locally-owned group cleanup are real contracts. | [STD] native/GDN retained boundaries, [P3 integrated]; no native execution claim. |
| `frontier/profiling/linear_op/linear_op_impl.py` | 10 | 9 | R3 | C — architecture registry owns exact raw model-type norm policy; required model fields direct. Preserve norm guard, error order and bootstrap-dependent local import. | [STD] S6, [LARGE] S6 proposal, [P3 architecture] 240 norm outcomes. |
| `frontier/profiling/linear_op/linear_op_wrapper.py` | 1 | 1 | R3 | R — native dtype/config-context adaptation uses existing owner; no parallel normalization policy. | [STD] inventory, [P3 integrated]. |
| `frontier/profiling/linear_op/main.py` | 7 | 51 | R3 | R — candidate already centralizes platform discovery and DEVICE_EVENT output selection; retain orchestration boundary. | [STD] inventory, [P3 integrated]. |
| `frontier/profiling/linear_op/profiling_plan.py` | 10 | 10 | R3 | R — attention_linear_ops rename follows required profile declaration and existing plan dispatch. | [STD] inventory, [ATT] registry ownership, [P3 integrated]. |
| `frontier/profiling/moe/main.py` | 8 | 55 | R3 | R — shared platform discovery and explicit profile/quantization selectors; no replacement resolver warranted. | [STD] inventory, [P3 integrated]. |
| `frontier/profiling/moe/moe_impl.py` | 36 | 14 | R3 | R — native model/quantization/dtype propagation retained; valid selector distinctions are not missing state. | [STD] inventory, [P3 architecture], [P3 integrated]. |
| `frontier/profiling/moe/moe_vllm_kernel.py` | 416 | 42 | R3 | C — exact raw-type MXFP4 policy queried from architecture owner. Keep platform/API/ABI guards, one-shape weight cache and branch allocation sentinels pending any separately proven simplification. | [STD] S6/smaller observation 4, [P3 architecture] 90 gate outcomes. |
| `frontier/profiling/moe/moe_wrapper.py` | 47 | 10 | R3 | R — runtime/model identity and measurement-family propagation retained at native wrapper boundary. | [STD] inventory, [P3 integrated]. |
| `frontier/profiling/utils/__init__.py` | 33 | 4 | R3 | R — lazy native-related utility exports preserve CPU import safety; not lazy reconstruction of required runtime state. | [STD] inventory/import boundary, [P3 integrated]. |
| `frontier/profiling/utils/confirmation.py` | 2 | 2 | R3 | R — accelerator naming/discovery already delegated to shared owner. | [STD] inventory. |
| `frontier/profiling/utils/singleton.py` | 4 | 0 | R3 | R — explicit clear operation serves the existing singleton lifecycle/testing interface; no production fallback introduced. | [STD] inventory. |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` | 13 | 8 | R1 | C — remove test-only correction wrapper/unused arguments; direct configured model feeds live helper. Metrics forwarding remains. | [MET] M1/M2, [REC], [P2 scheduler]. |
| `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | 28 | 0 | R1 | R — one admitted capacity feeds both memory reservation and slot allocation; constructor runtime guard retained. | [MET] runtime-screened path, [SPEC] capacity invariant, [P1b], [P1c]. |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | 81 | 15 | R1 | C — direct initialized preemption state; keep optional non-GDN slot manager and transactional KV/state/queue ordering with rollback/rethrow. | [REC], [SPEC] D03, [P1c], [LARGE] scheduler split sequencing. |
| `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py` | 3 | 1 | R1 | R — supply actual pipeline/AF physical layer ID; existing optional DECODE_FFN ID validation is outside added hunk. | [MET] inventory, [SPEC] D01, [P1d], [P2 integrated]; exact hunk rechecked. |
| `frontier/scheduler/utils/decode_collective.py` | 3 | 4 | R1 | C — remove unused elapsed/start wrapper arguments; preserve actual stage completion timing and full-stage reporting. | [MET] M1/M2, [P2 scheduler] exact snapshots. |
| `frontier/scheduler/utils/dense_metrics.py` | 26 | 28 | R2 | C — remove test-only first-dense helper and discarded copy; canonical model count and per-actual-layer prediction. Scalar/nonmixed source isolation retained. | [MET] M1/M2, [REC], [P2 scheduler] five exact cases. |
| `frontier/scheduler/utils/ep_wave.py` | 4 | 0 | R1 | R — forwards capture_lane_timings; reporting-off avoids retained lane payloads without dropping scheduled work. | [MET] inventory/retained capture, [P2 scheduler], [P2 integrated]. |
| `frontier/scheduler/utils/ep_wave_schedule.py` | 7 | 0 | R1 | R — capture follows reporting demand and completed plan goes to optional sink; standalone helper may omit reporting. | [MET] inventory/retained optional sink, [P2 trace], [P2 integrated]. |
| `frontier/scheduler/utils/execution_time_metrics.py` | 30 | 56 | R2 | R — existing live Stage-copy and singleton extraction contracts replace generic copying; scalar support, explicit-zero semantics and source isolation retained. | [MET] M1 live helper, [P2 scheduler] migrated caller tests. |
| `frontier/scheduler/utils/expert_parallel.py` | 22 | 2 | R1 | R — conditional lane retention validates singleton Stage; lane work is distinct from synchronization maxima. | [MET] inventory/retained lane semantics, [P2 scheduler], [P2 trace]. |
| `frontier/scheduler/utils/memory_planner.py` | 89 | 9 | R1 | C — canonical runtime family/per-layer/GDN APIs replace reclassification/reflection. No-GDN shape and capacity required only for reservation remain optional. | [REC], [SPEC], [P1b] four exact memory snapshots. |
| `frontier/scheduler/utils/prefill_collective.py` | 10 | 3 | R1 | C — remove unused reporting wrapper arguments; retain participant-specific full-stage attention and actual dense layers. | [MET] M2, [P2 scheduler], [P2 integrated]. |
| `frontier/scheduler/utils/sync_entry.py` | 6 | 0 | R1 | R — optional metrics sink connects event/collective/wave reporting; do not require a service for standalone scheduling. | [MET] inventory; event callers rechecked, [P2 integrated]. |
| `frontier/simulator.py` | 31 | 2 | R1 | C — one predictor construction loop/direct model capability; retain genuine optional shared manager and actual Replica ID propagation. | [REC], [SPEC] integration, [P1a], [P1 timing], [P2 integrated]. |
| `frontier/training/__init__.py` | 2 | 1 | R4 | R — export GDNTrainer and update package terminology; no new policy or wrapper. | [STD] inventory, [P4 GDN]. |
| `frontier/training/attention_trainer.py` | 1 | 1 | R4 | R — consume required attention_linear_ops rename through existing architecture profile. | [STD] inventory, [P4 manager] family selections. |
| `frontier/training/base_trainer.py` | 1 | 0 | R4 | R — DEVICE_EVENT admitted via existing measurement enum; do not collapse distinct measurement identities. | [STD] inventory, [P3 integrated], [P4 GDN]. |
| `frontier/training/cli.py` | 61 | 0 | R4 | R — explicit GDN trainer entry/selectors; whole-model None is supported dataset-only operation. | [STD] S1/inventory, [P4 GDN] dataset/selector contracts. |
| `frontier/training/gdn_trainer.py` | 301 | 0 | R4 | C — direct supplied-model methods and removal of unread GDN-only df state. Preserve selector precedence, whole-model None, estimator metadata and atomic saves. | [STD] S1/smaller observation 1, [REC], [P4 GDN] manifests/estimators/cross-loads. |
| `frontier/types/device_sku_type.py` | 1 | 0 | R1 | R — MI355X enum entry extends existing SKU mechanism, not a local name gate. | [STD] associated registries. |
| `frontier/types/measurement_type.py` | 1 | 0 | R1 | R — DEVICE_EVENT is distinct from CUDA_EVENT/KERNEL_ONLY; measured/provenance identities must remain separate. | [STD] associated registries, [P4 manager] selector matrix. |
| `frontier/types/node_sku_type.py` | 1 | 0 | R1 | R — MI355X UBB enum entry extends existing node registry. | [STD] associated registries, [P4 config]. |
| `frontier/utils/param_counter.py` | 121 | 10 | R1 | C — canonical topology/GDN contract replaces defensive reconstruction; retain D57 parameter-byte approximation and FP32 GDN A_log. | [REC], [SPEC] memory authority, [P1b] exact parameter/KV/state bytes. |

## Remaining work and reconciled decisions

Inspection gaps: **none identified** after path-by-path review reconciliation and the supplementary hunk/consumer checks above. This is changed-hunk coverage, not a claim to have reviewed every unchanged line of seven large modules. Their existing-boundary reasons and functional split sequences are recorded in [Large modules][LARGE]; no broad extraction is silently marked implemented.

Implementation follow-up: **A5 completed and committed as `5cb8794f`**. The raw shape adapter, runtime field/HF loader and profiling constructor now reuse `GatedDeltaNetConfig.output_gate_type`; exactly four default expressions changed without new registry, import or dependency. [P4 gate defaults] supersedes the earlier pending wording in [ATT] and [P4 config]. Recorded evidence: 31 existing baseline tests plus 43 corrected pre-change characterization tests; 74 combined after PASS; 54 complete output snapshots are byte-identical (259,793 bytes each), including all 10 error outcomes. Omission versus explicit null, raw values, swish normalization, wrapped error causes/stages and ordered layer identities remain unchanged. The initial eight characterization failures were incorrect expectations about existing error wrapping, corrected before production edits; they were not a candidate defect. Genuine external missing-field adaptation and absent GDN shape remain supported. No A5 or inventory implementation follow-up remains; this reconciliation only reads the existing evidence and runs no tests.

Other original proposals are dispositioned rather than silently dropped:

- **Metrics M3**: direct `OperationMetrics(op_name)` was rejected after RCA because trace `mlp_act` differs from metric `mlp_activation`. Keep the explicit three-field schema projection; the unused family lookup was removed. [REC], [P2 readers].
- **Attention A1/A2/A3/A4**: implemented, including actual cache-test seam and singleton variant unpacking, not `supported_variants[0]`. A direct identity re-export failed with a demonstrated import cycle; the existing exported adapter now defers to the canonical owner. [P1 identity], [P3 small].
- **S6 imports/admission**: registry metadata is centralized, but raw-type membership is intentionally not resolved-profile membership. Bootstrap-dependent local import remains backed by actual fresh-interpreter cycle evidence. [P3 architecture].
- **S7 routing**: original numerical uncertainty is closed for the characterized supported inputs by exact comparisons and shared implementation. The separate disaggregation normalization pass remains because removal can change floating arithmetic; no unapproved normalization simplification. [P3 routing], [LARGE].
- **J1 larger state collapse / MXFP4 branch-local weights / exported replay wrappers**: inspected lower-value ideas are not required fixes. Existing lifecycle timer scopes, mutually exclusive weight ownership/random-call order, and public exported forms justify retaining the current boundaries without a larger unverified rewrite. [STD], [P3 orchestration].
- **Positive DP trace fixture**: unsupported current runtime, not a production bug. Only unreachable shape labels were removed; zero-return scalar seam retained. [P2 trace].

Verification gaps remain separate from inspection coverage: main owns final P4/P5 broad regression, final at-least-50-case fidelity, final non-dummy comparison and paired timing against the frozen cleanup source. Existing [P2 integrated] records 159 PASS, eight non-dummy cases, 90 stable artifact comparisons, 106 symmetric metric files and 16 summary pairs. Those **106 metric files are unrelated to this inventory's 106 source paths**. [P3 integrated] records 721 PASS / 5 SKIP / 2 known main-equivalent missing-FlashInfer CLI failures, not an all-green suite. CPU stand-ins and synthetic artifacts do not establish native ROCm/CUDA numerics or production-profile fidelity; [P1 timing] measures reporting-disabled Simulator.run only. No new semantic-choice blocker was found by this reconciliation.

## Coverage checks

Observed on September 17, 2026: **106 rows / 106 unique paths / 105 Python files / 1 README; 38 C / 68 R at the stated checkpoint**. Exact frozen path/numstat comparison and original four-column comparison both exit 0: no missing, extra or duplicate paths and no changed ranks/counts. The C-path set also exactly equals the production paths changed between frozen candidate and checkpoint. All 31 evidence-link targets exist, all table reference labels resolve, and scoped `git diff --check` passes. These are inventory/document checks, not test execution or numerical verification. Only this inventory was edited by the reconciliation worker; production/tests and shared records remain untouched.

The executable checks below compare the first four columns against the frozen Git inventory and the pre-reconciliation table. They also expose omissions/duplicates rather than deriving coverage from a count alone. Run from the worktree root; they are read-only and do not run tests.

```bash
git diff --name-only 0515589ac7f49ac5288a5f55b0ce38b0ede29bb2...c288a19f59bec09529ee18d782fa57218da2c781 -- frontier
diff -u \
  <(git diff --numstat 0515589ac7f49ac5288a5f55b0ce38b0ede29bb2...c288a19f59bec09529ee18d782fa57218da2c781 -- frontier | sort) \
  <(awk -F '|' '/^\| `frontier\// {p=$2; a=$3; d=$4; gsub(/[ `]/,"",p); gsub(/ /,"",a); gsub(/ /,"",d); print a "\t" d "\t" p}' task_memory/task_2026-09-17_candidate_quality_cleanup/inventory.md | sort)
diff -u \
  <(git show 540f23150d12670e268bd0ef74b9af30a00bd88a:task_memory/task_2026-09-17_candidate_quality_cleanup/inventory.md | awk -F '|' '/^\| `frontier\// {print $2 "|" $3 "|" $4 "|" $5}') \
  <(awk -F '|' '/^\| `frontier\// {print $2 "|" $3 "|" $4 "|" $5}' task_memory/task_2026-09-17_candidate_quality_cleanup/inventory.md)
awk -F '|' '/^\| `frontier\// {seen[$2]++; n++; if ($2 ~ /\.py`/) py++; if ($2 ~ /README\.md`/) doc++} END {for (p in seen) {u++; if (seen[p] != 1) print "DUPLICATE", p} print "rows=" n, "unique=" u, "python=" py, "readme=" doc; exit !(n == 106 && u == 106 && py == 105 && doc == 1)}' task_memory/task_2026-09-17_candidate_quality_cleanup/inventory.md
git diff --check -- task_memory/task_2026-09-17_candidate_quality_cleanup/inventory.md
```

## Evidence links

All links below are existing task records in this directory. Original review proposals are historical; use the linked implementation report for current completion/evidence limits.

[REC]: refactoring_record.md
[STD]: standards_review.md
[ATT]: attention_review.md
[MET]: metrics_review.md
[LARGE]: large_module_review.md
[SPEC]: spec_review.md
[P1a]: test_report_2026-09-17_p1a.md
[P1b]: test_report_2026-09-17_p1b.md
[P1c]: test_report_2026-09-17_p1c.md
[P1d]: test_report_2026-09-17_p1d.md
[P1e]: test_report_2026-09-17_p1e.md
[P1 fixtures]: test_report_2026-09-17_p1_fixtures.md
[P1 identity]: test_report_2026-09-17_p1_identity.md
[P1 private]: test_report_2026-09-17_p1_private_layer.md
[P1 timing]: test_report_2026-09-17_p1_timing.md
[P2 residual]: test_report_2026-09-17_p2_residual_metadata.md
[P2 readers]: test_report_2026-09-17_p2_trace_readers.md
[P2 scheduler]: test_report_2026-09-17_p2_scheduler.md
[P2 trace]: test_report_2026-09-17_p2_trace_context.md
[P2 integrated]: test_report_2026-09-17_p2_integrated.md
[P3 schema]: test_report_2026-09-17_p3_schema.md
[P3 small]: test_report_2026-09-17_p3_small_contracts.md
[P3 model]: test_report_2026-09-17_p3_model.md
[P3 architecture]: test_report_2026-09-17_p3_architecture.md
[P3 routing]: test_report_2026-09-17_p3_routing.md
[P3 orchestration]: test_report_2026-09-17_p3_orchestration.md
[P3 integrated]: test_report_2026-09-17_p3_integrated.md
[P4 manager]: test_report_2026-09-17_p4_manager.md
[P4 GDN]: test_report_2026-09-17_p4_gdn.md
[P4 config]: test_report_2026-09-17_p4_config_contracts.md
[P4 gate defaults]: test_report_2026-09-17_p4_gate_defaults.md

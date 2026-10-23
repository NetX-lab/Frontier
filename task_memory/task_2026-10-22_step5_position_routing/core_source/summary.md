# CORE/SOURCE implementation handoff

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive verified interfaces, source/runtime slices and dependency boundaries. |
| 2026-10-23 | Supersede early partial accounting with delivered consumers, peer imports and strict compute/buffer fixture inputs. |

TaskID: **D29-CORE-SOURCE-IMPLEMENTATION**. Branch `feat/step5-position-core`; worktree `/data/ycfeng/Frontier/.worktrees/step5-position-core`; baseline `1aee0af`. Authority: [requirements.md](requirements.md), central D29-SCOPE-107-10000 and44-path sole-writer allocation. **Partial integration handoff, not completed D29 acceptance.**

## Delivered implementation

- `c995fdc`, `ac48363`, `04294dcd`: registry-owned position counts, immutable complete source/lane descriptors, physical/live separation, single source_frame_tensors/cell schema, constructor-before-parent admission, source-aware scheduler/cache, full installed-source verifier, original singleton annotation/parser replay/restore/clear and unchanged event durations. Legacy source identity omits native_facts.
- `b39fc51c`, `e255edee`, `bf458dbd`, `f7c02c7b`, `41bbf035`: actual selected registry/schema consumers, both preflights/loaders/trainers and cold/warm paired cache, actual per-row source/native layouts and invariant binding identity, three executor cell forwarding, source-local gate/router/reached prepare and gathered inclusive expert wrapper, source-width planning and strict source/EP association. No legacy shuffle charge or alternate registry.
- `795a583a`: F1/P2 actual bound native_router._select_experts provenance correction. Constructor-level negative test preserves router class/module/file/hash/public select_experts and replaces only internal bound implementation; decisive RED DID NOT RAISE then GREEN20 PASS. Existing verifier only; no native semantic changes.
- `1efe927a`, `884390e4`, `396a3806`: retained actual source/native/collective point binding into H04 and ExecutionTime, explicit fresh position fixture/runner and exact positive source/shared duration/count pairs. Actual BF16 shared estimator composition and live0 expert pairs tested; collective leaf remains explicitly CPU-only.
- `26946b85`: full observed compute_context required in plan.buffer_json and exact buffer/count identity; same-context CPU activation/scales/topk/output/actual graph inputs, explicit CPU CollectivePlanCost payload and existing profiling_data_dir forwarding. CPU payload is not a finite calibrated/native model or importer bypass.

Authorized peer histories remain intact:7b86614/2c4cc9e merges3f044142/6011b269; source/native67d5b5e merge4d201c8f (includes repair6827507); PRICING8e61c2d merge dc538ac0;27b3499 mergef4dbfc22;bfe6f2ae mergeb4f7a1d6. Import authority is not final source/runtime acceptance. F4 frozen-fit normalization belongs only to PRICING collective_model_context; full per-rank branch/buffer/policy/count/cache observations are retained by CORE.

## Observed verification

- Count/source/runtime/selected union:215 PASS/19 SKIP at foundation; automatic canonical source/consumer union83 PASS/4 SKIP; later source/runtime70 PASS/4 SKIP. These overlap and are not summed.
- Actual source/timer/provenance20 PASS (Python3.12.13 asyncopd-py312), including F1/P2 real seam. Actual source/gathered wrapper with CPU tensors and exact pricing feature function3 PASS under narrow import isolation; not GPU/kernel proof.
- Both automatic shared and independent trainers: fresh252-row/two-selected-group CSV cold/warm fits PASS for both measurement modes; complete exact paired keys and retained physical receipts. Latest source/runtime/fixture36 PASS/3 SKIP; strict compute/buffer/command6 PASS/3 SKIP; final buffer2 PASS. Missing/foreign context rejected, positive live0 evidence retained, legacy defaults preserved.
- Source/package152 PASS/1 SKIP/4 FAIL: four retained unmodified request-free tests fail only because frontier-py310 lacks torch. Exact errors/logs in dated report; no fallback/dependency install. Old peer inventory/fake-family assumptions remain separate in earlier report, not hidden by narrowed tests.
- Authentic published source/static binding:2907 files,53 installed pins,4 derived files,12 backend files match root publication/export/receipt; index/base/missing receipt/wrong generation rejected. Actual published Linux/amd64 digest f037d958... is used, never local index10cdd7af. Fresh P/D package548 sealed files each, MoE36 planned cells/method, admitted=false. Source/static package proof is not installed-runtime/GPU qualification.
-44-path accounting, every per-file ceiling, Python AST and whitespace PASS: **+3150/-155=3305/4404 gross;1099 remaining.** No quota borrowing, new handwritten path, peer edit or platform operation. Task records excluded. Isolated development submodule3dac53c checkout is intentionally unstaged; final gitlink is root-owned.

Exact commands, environments, logs, prior failures and evidence limits live in [test_report_2026-10-23_compute_buffer_fixture.md](test_report_2026-10-23_compute_buffer_fixture.md), [test_report_2026-10-23_source_exact_pairs.md](test_report_2026-10-23_source_exact_pairs.md), [test_report_2026-10-23_runtime_points.md](test_report_2026-10-23_runtime_points.md), and the earlier dated reports. Interfaces and size-note delta: [interface_receipt.md](interface_receipt.md).

## Exact44-owned-path accounting

Gross=baseline1aee0af added+deleted; H04/H17 excluded; each retained core/source row grouped by exact path.

| Owned changed path | Gross / allowance |
| --- | ---: |
| `frontier/config/replica_config.py` |16/16|
| `frontier/execution_time_predictor/layer_contract_resolution.py` |11/45|
| `frontier/execution_time_predictor/moe_dataset_training.py` |160/164|
| `frontier/execution_time_predictor/moe_routing_workload.py` |124/145|
| `frontier/execution_time_predictor/prediction_family_trainers.py` |80/158|
| `frontier/execution_time_predictor/profiling_dataframe_loaders.py` |9/27|
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` |32/38|
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` |98/188|
| `frontier/execution_time_predictor/step5.py` |34/81|
| `frontier/moe_ep_workload.py` |48/87|
| `frontier/moe_load_imbalance.py` |51/56|
| `frontier/moe_routing_policy.py` |207/212|
| `frontier/moe_routing_runtime.py` |104/123|
| `frontier/profiling/common/layers/layernorm.py` |12/21|
| `frontier/profiling/common/runtime_identity.py` |133/176|
| `frontier/profiling/common/step5_job.py` |67/110|
| `frontier/profiling/linear_op/main.py` |29/29|
| `frontier/profiling/moe/inputs.py` |123/138|
| `frontier/profiling/moe/load_distribution.py` |1/5|
| `frontier/profiling/moe/main.py` |116/171|
| `frontier/profiling/moe/moe_impl.py` |53/172|
| `frontier/profiling/moe/moe_wrapper.py` |168/284|
| `frontier/profiling/step5/job.py` |49/82|
| `frontier/profiling/step5/vllm_wrapper.py` |15/29|
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` |11/11|
| `frontier/scheduler/utils/ep_trace.py` |11/28|
| `frontier/scheduler/utils/ep_wave.py` |5/22|
| `frontier/scheduler/utils/layer_workload.py` |22/43|
| `tests/integration/run_step5_measured_main.py` |37/91|
| `tests/integration/step5_measured_compute_fixture.py` |148/230|
| `tests/integration/step5_profiling/run_on_worker.sh` |21/26|
| `tests/integration/step5_profiling/submit_profiling_job.py` |3/9|
| `tests/integration/step5_profiling/verify_fixed_probe.py` |41/65|
| `tests/integration/test_step5_measured_main.py` |153/153|
| `tests/unit/test_moe_grouped_gemm_padded_features.py` |173/180|
| `tests/unit/test_moe_position_routing.py` |159/160|
| `tests/unit/test_moe_profiling_output_metadata.py` |85/85|
| `tests/unit/test_moe_routing_conservation.py` |63/71|
| `tests/unit/test_moe_routing_runtime.py` |59/60|
| `tests/unit/test_step5_attention_profiling.py` |15/30|
| `tests/unit/test_step5_fixed_probe.py` |150/155|
| `tests/unit/test_step5_measured_ffn.py` |108/108|
| `tests/unit/test_step5_moe_profiling.py` |276/290|
| `tests/unit/test_step5_request_free_profiling.py` |25/30|

## Remaining concrete integration boundary

1. **Unowned registry:** prediction_model_registry.py/prediction_model_identity.py currently store only name/typed identity within precision/family. Same FP32 gate contract in FP8/BF16 native bindings overwrites; current runtime cannot retain/select both actual source contexts. Root must coordinate measurement-binding retention/query in the existing registry and actual-point lookup. No CORE aliases/parallel registry.
2. **Unowned backend/config:** production predict_collective_plan intentionally rejects cpu_fixture/candidate inputs. CORE emits explicit CPU plan/cost payload at manifest.collective_plan_input_file and forwards existing profiling_data_dir, but no existing CLI consumer accepts that payload as CPU. Root must integrate an explicit CPU-evidence path in the same backend, preserving native qualification requirements. No synthetic parameters relabeled native.
3. **C2 native producer:** H04 repair requires complete observed compute_context in buffer_receipt and cache/count key. Native collector currently lacks observed compute-to-buffer association. CORE strict reader and CPU same-context writer are delivered; native evidence producer and repaired C1/C2 tip must come from root/PRICING. Shared receipt or frozen fit identity alone is insufficient.
4. **Fresh actual-main unfinished:** full generator produced37044 MoE rows per method but timed out in preflight at600s before manifest. Direct actual CLI seam01 outer120s interrupted before internal2400s; seam02 runs with no outer timeout, internal2400s unchanged, result pending in `/data/ycfeng/tmp/d29-core-position-main-seam02`. Both attempts predate the new complete CPU plan payload. No completed offline/online metrics/ledger validation, no full native/G3 qualification, no final fidelity/accuracy claim.

Workflow retains old fixtures,6144/1024 semantic request,D26 P6128+D16,D27 exact READ payload,D30 readiness/sync and original2400s limit. Root alone owns full-unit/strict77/independent review/native/image/GPU/E2E acceptance. Work stops at specific dependency boundary, **not quota exhaustion**. No GPU/platform/auth/cloud/SSH/Docker/push/deletion or extra agent operation occurred.

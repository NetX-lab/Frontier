# CORE/SOURCE partial implementation handoff

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive verified interfaces/runtime/source slices and precise dependency boundary. |

TaskID: **D29-CORE-SOURCE-IMPLEMENTATION**. Branch `feat/step5-position-core`, worktree `/data/ycfeng/Frontier/.worktrees/step5-position-core`, baseline `1aee0af`. Authority: [requirements](requirements.md), central D29-SCOPE-107-10000 and sole-writer matrix. This is a **partial handoff**, not complete D29 acceptance.

## Completed deliverables

1. `c995fdc`: registry-owned interval position counts; frozen source/lane descriptors; physical/live workload separation; canonical `inputs.py` source tensors and one cell schema; full installed-source module/callable verifier; legacy identity omission.713 gross code/test lines.
2. `ac48363`: policy resolved before parent initialization; shared scheduler actual admitted DP source map; complete immutable-source cache; no per-wave Hamilton/ratio reset in position mode; legacy paths retained.230 additional gross lines.
3. Source/timer follow-up: existing image/launcher/copy/seal propagation, GateLinear/router/lazy native norm callable evidence, per-row shared-native identity attachment, real singleton RECORD_FUNCTION replay/parser/restore/clear and preserved CUDA_EVENT duration fields, masked sentinel filtering.622 additional gross lines; commit recorded in progress after creation.

Interface contracts and root size-note proposal: [interface_receipt.md](interface_receipt.md). Test evidence: [interface report](test_report_2026-10-23_interfaces.md), [runtime/source report](test_report_2026-10-23_runtime_source.md), [source/timer report](test_report_2026-10-23_source_timer.md).

## Observed validation

- frontier-py310 Python3.10.6 actual source/cache/constructor and selected legacy checks:215 PASS,19 SKIP,4 deselected.
- asyncopd-py312 Python3.12.13 real CPU timer/router/callable provenance checks:18 PASS,11 deselected.
- actual source parser/linear attachment/fixed-probe tests:35 PASS.
- Actual old stage/verifier:PASS,540 sealed files,18 attention candidate rows,16 linear rows, both methods dry-run PASS,0 platform operations,admitted=false. Retained result `/data/ycfeng/tmp/d29-core-actual-stage-verifier-01/verification.json`.
- Exact owned paths/per-file ceiling, AST and whitespace:PASS.29/44 paths changed,+1470/-95=**1565/4404 gross**,2839 remaining. No peer paths, root-only size-note, README, platform or unallocated code file changed.
- Preserved unavailable baseline checks separately: four unmodified request-free tests fail because frontier-py310 lacks torch; asyncopd-py312 lacks fasteners for full simulator imports. No dependency install or behavior/test fallback.

## Exact 44-owned-path accounting

Gross is the baseline1aee0af added+deleted code/test/config diff; task records excluded. Zero does not mean a required responsibility is complete. H04/H17 are explicitly PRICING-owned and excluded.

| Owned path | Gross | Status |
| --- | ---: | --- |
| `frontier/config/replica_config.py` |16| implemented |
| `frontier/execution_time_predictor/layer_contract_resolution.py` |0| selected preflight pending |
| `frontier/execution_time_predictor/moe_dataset_training.py` |0| selected trainer pending |
| `frontier/execution_time_predictor/moe_routing_workload.py` |32| source/cache done; native schema pending |
| `frontier/execution_time_predictor/prediction_family_trainers.py` |0| selected trainer pending |
| `frontier/execution_time_predictor/profiling_dataframe_loaders.py` |0| selected operator facts pending |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` |32| admission/runtime done |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` |17| constructor done; selected price dispatch pending |
| `frontier/execution_time_predictor/step5.py` |0| selected native/scope prerequisites pending |
| `frontier/moe_ep_workload.py` |48| implemented |
| `frontier/moe_load_imbalance.py` |0| native schema classifier pending |
| `frontier/moe_routing_policy.py` |196| implemented |
| `frontier/moe_routing_runtime.py` |16| registry runtime done; selected frame filter pending |
| `frontier/profiling/common/layers/layernorm.py` |12| callable verification done |
| `frontier/profiling/common/runtime_identity.py` |133| verifier/identity implemented |
| `frontier/profiling/common/step5_job.py` |67| source interface done; full selected package unverified |
| `frontier/profiling/linear_op/main.py` |29| identity/provenance implemented |
| `frontier/profiling/moe/inputs.py` |123| implemented |
| `frontier/profiling/moe/load_distribution.py` |1| sentinel fix done |
| `frontier/profiling/moe/main.py` |0| selected census/row validation/identity pending |
| `frontier/profiling/moe/moe_impl.py` |52| actual factory/source adapter done; reached prepare connector pending |
| `frontier/profiling/moe/moe_wrapper.py` |41| singleton timer/mask done; selected cell/native integration pending |
| `frontier/profiling/step5/job.py` |49| explicit plan/source interface done; selected package unverified |
| `frontier/profiling/step5/vllm_wrapper.py` |15| source/declared-image evidence done; actual image unverified |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` |2| source threading done |
| `frontier/scheduler/utils/ep_trace.py` |11| live/physical conservation done |
| `frontier/scheduler/utils/ep_wave.py` |5| actual DP source map done |
| `frontier/scheduler/utils/layer_workload.py` |22| strict actual source materialization done |
| `tests/integration/run_step5_measured_main.py` |0| new fixture/finite-plan runner pending |
| `tests/integration/step5_measured_compute_fixture.py` |0| new fixture pending; old fixtures preserved |
| `tests/integration/step5_profiling/run_on_worker.sh` |21| neutral installed-source preflight interface done |
| `tests/integration/step5_profiling/submit_profiling_job.py` |3| image propagation done |
| `tests/integration/step5_profiling/verify_fixed_probe.py` |41| sealed/source/image checks done; selected MoE dry-run pending |
| `tests/integration/test_step5_measured_main.py` |0| fresh selected actual-main checks pending |
| `tests/unit/test_moe_grouped_gemm_padded_features.py` |0| actual pre-loader/schema tests pending |
| `tests/unit/test_moe_position_routing.py` |159| count/source/constructor/cache/scheduler tests done |
| `tests/unit/test_moe_profiling_output_metadata.py` |50| actual linear attachment/source cell tests done; selected MoE metadata pending |
| `tests/unit/test_moe_routing_conservation.py` |0| existing checks pass; planned additions pending |
| `tests/unit/test_moe_routing_runtime.py` |19| constructor order test done |
| `tests/unit/test_step5_attention_profiling.py` |0| existing source checks pass; actual selected evidence pending |
| `tests/unit/test_step5_fixed_probe.py` |150| source/verifier/legacy checks done |
| `tests/unit/test_step5_measured_ffn.py` |0| selected physical exact-pair checks pending |
| `tests/unit/test_step5_moe_profiling.py` |203| real timer/router/norm/oracle source checks done; prepare/root native integration pending |
| `tests/unit/test_step5_request_free_profiling.py` |0| existing four torch-unavailable checks preserved |

## Concrete unresolved prerequisite boundary

1. Root must authorize importing accepted peer commits. PRICING `7b86614` has the sole `get_moe_profiling_operators(context)` and contextual Step binding; current branch lacks it. No peer fallback or alternate registry was added.
2. PRICING must complete/commit actual `NativeRoutedExperts.prepare_source/prepared_context/apply_prepared` scope contract, selected native alignment/capacity/ranges/schema/target facts and finite-plan numerical fixture/input/config/metric payload. An uncommitted preliminary callable is not the complete tested interface. Supply contributor facts for CORE sole-writer shared paths; do not edit them concurrently.
3. Then resume bound operator facts before both legacy pre-loader helpers/preflight, one shared source/gathered frame filter, paired exact duration/count models/cache/schema ranges and applicable empty physical-key rejection; finish selected wrapper/main one-cell/one-dict protocol.
4. Finally add fresh synthetic CPU actual-main fixture/runner while retaining old fixtures,6144/1024 semantic workload,D26 P6128+D16,D27 payload,D30 sync and unchanged2400s CLI limit. This was not executed; old fixed-probe dry-run is not a substitute.
5. Root owns final installed-image receipt/digest, full-unit/strict77/independent review, image build and actual native/GPU/runtime/E2E qualification. Native `set_routing_positions` candidate/factory contract is available and consumed by CPU contract tests; image publication/full receipt are absent. All GPU numerical/kernel fidelity remains UNVERIFIED.

This task stopped at the peer-dependency boundary, **not quota exhaustion**. Requirements were re-read before handoff. No native numerical approximation, fabricated image SHA, synthetic zero missing scope, extra agent or external operation was used.

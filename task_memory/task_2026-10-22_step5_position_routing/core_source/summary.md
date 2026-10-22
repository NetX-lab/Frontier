# CORE/SOURCE partial implementation handoff

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive verified interfaces/runtime/source slices and precise dependency boundary. |
| 2026-10-23 | Extend partial archive with authorized peer integration and verified selected consumers. |

TaskID: **D29-CORE-SOURCE-IMPLEMENTATION**. Branch `feat/step5-position-core`, worktree `/data/ycfeng/Frontier/.worktrees/step5-position-core`, baseline `1aee0af`. Authority: [requirements](requirements.md), central D29-SCOPE-107-10000 and sole-writer matrix. This is a **partial handoff**, not complete D29 acceptance.

## Completed deliverables

1. `c995fdc`: registry-owned interval position counts; frozen source/lane descriptors; physical/live workload separation; canonical `inputs.py` source tensors and one cell schema; full installed-source module/callable verifier; legacy identity omission.713 gross code/test lines.
2. `ac48363`: policy resolved before parent initialization; shared scheduler actual admitted DP source map; complete immutable-source cache; no per-wave Hamilton/ratio reset in position mode; legacy paths retained.230 additional gross lines.
3. `04294dc`: existing image/launcher/copy/seal propagation, GateLinear/router/lazy native norm callable evidence, per-row shared-native identity attachment, real singleton RECORD_FUNCTION replay/parser/restore/clear and preserved CUDA_EVENT duration fields, masked sentinel filtering.622 additional gross lines.
4. Authorized peer imports preserve7b86614/2c4cc9e history in3f04414/6011b26. Selected consumer slice adds shared filter/preflights/loader, actual shared+independent sklearn/cache identities and exact pairs, registry-context runtime dispatch, expected-cell validator and executor/native-fact/source-context integration. Production schema enumeration/wrapper/fixture remains pending.

Interface contracts and root size-note proposal: [interface_receipt.md](interface_receipt.md). Test evidence: [interface report](test_report_2026-10-23_interfaces.md), [runtime/source report](test_report_2026-10-23_runtime_source.md), [source/timer report](test_report_2026-10-23_source_timer.md).

## Observed validation

- frontier-py310 Python3.10.6 actual source/cache/constructor and selected legacy checks:215 PASS,19 SKIP,4 deselected.
- asyncopd-py312 Python3.12.13 real CPU timer/router/callable provenance checks:18 PASS,11 deselected.
- actual source parser/linear attachment/fixed-probe tests:35 PASS.
- Actual old stage/verifier:PASS,540 sealed files,18 attention candidate rows,16 linear rows, both methods dry-run PASS,0 platform operations,admitted=false. Retained result `/data/ycfeng/tmp/d29-core-actual-stage-verifier-01/verification.json`.
- Exact owned paths/per-file ceiling, AST and whitespace:PASS.29/44 paths changed,+1470/-95=**1565/4404 gross**,2839 remaining. No peer paths, root-only size-note, README, platform or unallocated code file changed.
- Preserved unavailable baseline checks separately: four unmodified request-free tests fail because frontier-py310 lacks torch; asyncopd-py312 lacks fasteners for full simulator imports. No dependency install or behavior/test fallback.
- Continuation:111 PASS/1 SKIP actual selected/legacy consumers;19 PASS source/timer;12 PASS/18 SKIP minimal legacy source. Peer union40 PASS/5 FAIL (unchanged inventory/fake-family assumptions after7b), recorded separately. See test_report_2026-10-23_selected_consumers.md for exact commands, evidence and corrected legacy invocation failure. Current37/44 paths,+2056/-133=2189/4404 gross,2215 remain; AST/whitespace/path/per-file ceilings PASS. Metadata test reaches85-line allowance.

## Exact 44-owned-path accounting

Gross is the baseline1aee0af added+deleted code/test/config diff; task records excluded. Zero does not mean a required responsibility is complete. H04/H17 are explicitly PRICING-owned and excluded.

| Owned path | Gross | Status |
| --- | ---: | --- |
| `frontier/config/replica_config.py` |16| implemented |
| `frontier/execution_time_predictor/layer_contract_resolution.py` |11| bound preflight done; schema enumeration pending |
| `frontier/execution_time_predictor/moe_dataset_training.py` |88| actual selected fit/cache/metadata done; production enumeration pending |
| `frontier/execution_time_predictor/moe_routing_workload.py` |32| source/cache done; native schema pending |
| `frontier/execution_time_predictor/prediction_family_trainers.py` |63| actual selected fit/cache/pairs done; schema enumeration pending |
| `frontier/execution_time_predictor/profiling_dataframe_loaders.py` |9| selected operator facts forwarded before legacy helpers |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` |32| admission/runtime done |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` |48| constructor/context dispatch/exact-pair checks done; pricing composition pending |
| `frontier/execution_time_predictor/step5.py` |26| selected row/source/typed context identity done; native numerical prerequisites pending |
| `frontier/moe_ep_workload.py` |48| implemented |
| `frontier/moe_load_imbalance.py` |0| native schema classifier pending |
| `frontier/moe_routing_policy.py` |196| implemented |
| `frontier/moe_routing_runtime.py` |101| registry runtime and selected frame/pair filter done |
| `frontier/profiling/common/layers/layernorm.py` |12| callable verification done |
| `frontier/profiling/common/runtime_identity.py` |133| verifier/identity implemented |
| `frontier/profiling/common/step5_job.py` |67| source interface done; full selected package unverified |
| `frontier/profiling/linear_op/main.py` |29| identity/provenance implemented |
| `frontier/profiling/moe/inputs.py` |123| implemented |
| `frontier/profiling/moe/load_distribution.py` |1| sentinel fix done |
| `frontier/profiling/moe/main.py` |49| executor/selected inventory/native facts done; census/row binding pending |
| `frontier/profiling/moe/moe_impl.py` |52| actual factory/source adapter done; reached prepare connector pending |
| `frontier/profiling/moe/moe_wrapper.py` |49| singleton timer/mask/source context done; selected cell/native integration pending |
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
| `tests/unit/test_moe_grouped_gemm_padded_features.py` |173| actual pre-loader/fit/cache/source tests done; native layout tests pending |
| `tests/unit/test_moe_position_routing.py` |159| count/source/constructor/cache/scheduler tests done |
| `tests/unit/test_moe_profiling_output_metadata.py` |85| actual linear/MoE attachment/source cell tests done; allowance full |
| `tests/unit/test_moe_routing_conservation.py` |0| existing checks pass; planned additions pending |
| `tests/unit/test_moe_routing_runtime.py` |19| constructor order test done |
| `tests/unit/test_step5_attention_profiling.py` |0| existing source checks pass; actual selected evidence pending |
| `tests/unit/test_step5_fixed_probe.py` |150| source/verifier/legacy checks done |
| `tests/unit/test_step5_measured_ffn.py` |32| selected physical exact-pair runtime checks done |
| `tests/unit/test_step5_moe_profiling.py` |217| real timer/router/norm/oracle/source-context checks done; prepare/root native integration pending |
| `tests/unit/test_step5_request_free_profiling.py` |0| existing four torch-unavailable checks preserved |

## Concrete unresolved prerequisite boundary

1. Completed early imports7b86614/2c4cc9e. Root authorizes67d5b5e after current owned commit, including independently reviewed native6827507 and authentic published image/source records.
2. PRICING409093c now commits NativeRoutedExperts prepare/context/inclusive and finite importer. Root authorization to import409093c remains required. Exact additional seam: canonical actual-context/layout/padding/workspace CSV encoding, feature schema/range classifier entries and admitted DP/TP-to-EP source association; no invented fields/classifier or duplicate adapter.
3. Bound consumer preflights/fit/cache/exact-pair interfaces are verified. Finish production selected enumeration from actual rows, schema/ranges, wrapper/main one-cell execution and pricing composition using the supplied native contract.
4. Finally add fresh synthetic CPU actual-main fixture/runner while retaining old fixtures,6144/1024 semantic workload,D26 P6128+D16,D27 payload,D30 sync and unchanged2400s CLI limit. This was not executed; old fixed-probe dry-run is not a substitute.
5. Root delivered authentic published Linux/amd64 image f037d958... and2907-file receipt/export at67d5b5e; these remove the image/source blocker and can be checked independently. Root still owns full-unit/strict77/independent review and native/GPU/runtime/E2E qualification. All GPU numerical/kernel fidelity remains UNVERIFIED.

This task stopped at the peer-dependency boundary, **not quota exhaustion**. Requirements were re-read before handoff. No native numerical approximation, fabricated image SHA, synthetic zero missing scope, extra agent or external operation was used.

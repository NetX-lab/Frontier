# D23 actual-main CPU diagnosis

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Archived measured diagnosis-only deliverables and parent follow-up. |
| 2026-10-22 | Archived authorized fixed-binding implementation and bounded verification; retained parent acceptance limits. |
| 2026-10-22 | Archived diagnostics-only complete actual06 public iteration RED and profile proposal. |

## Outcome

Completed a deterministic seconds-scale real-cache RED loop, CPU profile, real cold/warm collective replay, and enabled trace metadata/JSONL persistence checks. No production behavior changed. Original2400s timeout retained; no new full workload or native acceptance claimed.

Task `D23-ACTUAL-MAIN/PERF`; session `ses_e9f519775ffe3Cv0bhEzJSKNhH`; branch `feat/step5-mrv2-perf` from `ec5de154c19ceeae23a3652f511582858170f813`.

## Deliverables

- `tests/performance/step5_mrv2_cpu.py`: SHA256-verified actual05 typed cache replay; physical attention/FFN feature truth, duration/count/precision/source-proof assertions, latency RED gate, optional cProfile and enabled trace persistence.
- `tests/performance/step5_mrv2_collective.py`: real actual05 topology/configuration materialization using existing built collective-sim; cold versus exact warm-hit measurements without cloning/building or original cache mutation.
- `task_memory/task_2026-10-22_step5_mrv2_perf/test_report_2026-10-22_cpu_timeout.md`: exact commands, acceptance, observed metrics, hypothesis discrimination, original bound, and limitations.
- `task_memory/task_2026-10-22_step5_mrv2_perf/progress.md`: completed scope and parent-owned follow-up.

## Decisive measurements

Non-profiled64 EP lane queries:0.52744/0.54212s on repeat, source truth PASS, RED. Mean8.23ms/query versus unchanged-bound conservative0.968ms budget. Metadata5.83ms, operator work2.39ms. Profile:32 descriptions of only2 physical layers,192 precision resolutions,97 full Step validations and228 layer-contract resolutions. Metadata reconstruction is73.3% of profiled work.

Final current-script repeat completed in3.798875s,815716KiB maximum RSS, RED/source PASS. Both scripts pass Python AST syntax validation.

Real EP16 all-to-all cold2.545s; identical-key hit0.172ms with exactly equal output. Real ATTN_TP8 allreduce cold1.885s; hit0.110ms. Enabled trace metadata/JSONL persistence40.5ms for228 events with persisted truth PASS. Exact-key lookup, hash and output serialization are not the dominant measured recurring work.

## Smallest source proposal for parent approval

1. Add predictor-lifetime layer metadata reuse at the two current consumers: `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:93` and `frontier/execution_time_predictor/step5_cost.py:126`. Reuse the existing MetricsStore precedent (`frontier/metrics/metrics_store.py:562`) rather than globally caching mutable configurations. Key by actual physical layer within an instance already bound to model/TP/EP configuration. This retains all dense/sparse/FP8/BF16 identities. Approximately8.21s/round of measured metadata work is avoidable by inference from88 MoE layers×EP16.
2. Reuse fixed selected contract/model binding at `_predict_step5_ffn_operator` (`sklearn_moe_execution_time_predictor.py:113`) by physical layer/operator/measurement family within the same predictor and immutable loaded artifact set. This prevents repeated full-model layer-kind parsing before every already-cached numeric lookup. Metadata reuse alone leaves2.39ms/operator bundle versus0.968ms gate, so it is not expected to settle the original bound by itself. Measure the second seam after the first approved change; do not claim an unmeasured speedup.

Preserve per-query feature coercion/bounds/exact coverage, exact duration/count pairing, evidence/model hashes, source measurement identity, zero-routed handling, all16 EP lane projections and enabled trace/metrics. Do not cache numeric results by layer alone or substitute fake costs. Different model/quantization/TP/EP configurations must use distinct predictor instances; verify the configuration lifetime before implementing instance reuse. Measurement family must partition selected models. Add no per-query large fingerprint scan.

## Remaining work

The diagnosis-only scope initially left approval, production implementation, review and full acceptance to the parent. The follow-up below implements the explicitly authorized source correction; review/reverification/integration and original full offline/online acceptance still belong to the parent. Setup998/1004s attribution is incomplete. Extrapolated per-round costs are not full-round measurements.

## Authorized follow-up: D23-MRV2-CONTRACT-REUSE

Reused fixed physical-layer metadata within each predictor and fixed `(layer, operator)` precision/serialized contract identity. The parent specifically excluded model caching; every measurement-family registry selection remains dynamic. Source consumers share `frontier/execution_time_predictor/step5_cost.py`; `sklearn_moe_execution_time_predictor.py` preserves nested context/cost restoration and delegates only fixed construction. Model/TP/EP configuration remains the existing predictor lifetime boundary. No initializer, global cache, output/feature cache, fallback, factor, mode flag or original diagnostic-script change.

Deliverables:

- `frontier/execution_time_predictor/step5_cost.py`: shared predictor-instance layer metadata and fixed operator binding tables.
- `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`: minimal delegates; per-query registry, feature validation, paired count queries and proofs unchanged.
- `tests/unit/test_step5_contract_reuse.py`: seven actual fitted-estimator regressions, including source-sensitive fixed-work counts, dynamic features/errors, registry replacement/removal, count failures, nested contexts, independent predictors/TP, physical precisions and distinct fitted measurement families/proof hashes.
- `docs/general/module-size-exceptions.md`: inspected source boundary and cleanup; Step cost 220->263 lines, sklearn_moe1814->1791; no over-limit module touched.
- `task_memory/task_2026-10-22_step5_mrv2_perf/test_report_2026-10-22_contract_reuse.md`: exact bounded commands, measured deltas, observed test failures/timeouts and acceptance limits.

Regression RED occurred before implementation: twelve layer descriptions/seventy-two precision resolutions for six real queries, with all output assertions passing. GREEN warm queries have zero repeated descriptions/precision/contract resolution, retaining every registry selection. The unchanged actual05 microbench measured0.073477s/64 lanes/456 operator queries against parent baseline0.532956s (7.253x improvement). Every cost dictionary and loaded artifact descriptor is exactly equal; source truth PASS. The maximum-query latency gate remains RED because first fixed binding and the slowest warm query exceed the unchanged0.96824ms/query budget. There is no1024 completion claim.

Existing validation:91 measured/FFN/on-demand tests PASS,44 public/root/metrics tests PASS,75 MRv2 forward tests PASS,7 new regressions PASS. Structural tests:38 PASS/2 FAIL on an unchanged stale scheduler stub missing `take_forward_start_batch`, before cost queries. Optional broader PD tests did not finish within60s; no timeout increase or unrelated edits. Enabled trace preserves456 events/344788 bytes and persisted truth PASS. Remaining recurring profile cost is dynamic duration/count feature validation and source-proof JSON parsing; full initialization, scheduler/stage E2E and fullCLI remain unmeasured here.

Code/tests/module-boundary commit: `ee28bf7ea43d3c7c73b7176eeb10b6f07d51492a`. Final source-sensitive unchanged command `/data/ycfeng/tmp/d23_contract_committed01/receipt.json` identifies that commit:0.074077174s64 lanes/456 operator queries,7.194605x speedup,sourceTruthPASS,exact proof/artifact equality,RED max-query gate. Enabled trace456 events/344787bytes0.068545761s,persisted truthPASS. Final focused regressions7 PASS9.15s (`d23_contract_green_final01.log`). Parent receives this source commit separately from the completion-record commit.

## Diagnostics-only follow-up: D23-ACTUAL06-WALL

Fast-forwarded clean branch to accepted `7e8bea3`; no further production optimization. `tests/performance/step5_actual06_iteration.py` uses original Simulator public event/predictor/scheduler/metrics path, SHA-validated original bundles copied to scratch, and exact first-output state with all92 layers/EP16 lanes/P+D metadata/KV pool and identity counters. Original6144/1024 request and2400s bound remain unchanged; no fullCLI/native acceptance.

Complete earlyq1 intervals9.540253/9.676296s are RED versus optimistic2.012202s budget.1168 events,352 waves/5632 lanes. D contribution4.418710/4.876795s explains most original later5.1s cadence by inference; concurrent earlyP adds5.121543/4.799501s. Exact later-context state was not replayed. All41288 FFN+960 attention queries match source duration/count/features/proofs; repeated audit dictionary, JSON event trace, two stage-ledger rows and5632 lane ledger digest equal.12224 real collective queries all hits,zero misses.

Whole-path profile:34.1s/42.9million calls, public stages24.054s, FFN9.896s, dynamic duration/count lookup5.933s, fixed identity JSON3.313s, lane builds3.754s/fullRequest construction2.635s, config scans1.735s. Nested cumulative times cannot be added. One-variable unprofiled fixed identity parsing3.135s/19.146s loop confirms its real cost but does not make it the sole timeout cause. Original metrics are small; additional enabled op trace31.611s and51988 persisted events is separately measured, not originalsetting attribution.

Proposal only: reuse already validated estimator-bound canonical identity/evidence-kind while retaining current model registry selection, identity pairing/runtime checks and every dynamic query/count/proof. Expected upper saving~0.8s/D-only round, insufficient alone for budget. Parent owns source decision, later-context/online/cold/init/finalexport/fullCLI/native/strict77 acceptance.

Report: `task_memory/task_2026-10-22_step5_mrv2_perf/test_report_2026-10-22_actual06_wall.md`. Exact reusable command and snapshot paths are there. Final baseline/audit receipts: `/data/ycfeng/tmp/d23_actual06_verified_{red,audit}01/`; whole profile `/data/ycfeng/tmp/d23_actual06_complete_profile01/`; enabledtrace `/data/ycfeng/tmp/d23_actual06_final_trace01/`.

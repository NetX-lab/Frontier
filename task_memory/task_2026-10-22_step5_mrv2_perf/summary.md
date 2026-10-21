# D23 actual-main CPU diagnosis

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Archived measured diagnosis-only deliverables and parent follow-up. |

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

Parent approval, production implementation, semantic regression review, and original full offline/online acceptance remain pending by design. Setup998/1004s attribution is incomplete. Public stage/scheduler/complete MetricsStore execution was not replayed, so extrapolated11.58s/round is not a full-round measurement. Use the non-profiled FFN RED command in the test report for the next edit; profile once only when the dominant boundary changes.

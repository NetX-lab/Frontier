## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded two reproducible W03 comparison divergences and the D01 decision boundary. |

# W03 D01 evidence and concrete options

Status: decision pending; no tolerance or golden changes. Candidate is the uncommitted W03 migration following `2e8ab0e1`. Baseline is `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2` in `../pr33-r12-baseline-20260915`. Environment: `/usr/bin/python` 3.12.3, no conda activation.

## Reproduction

```bash
python tests/integration/run_scheduler_refactor_fidelity.py --baseline ../pr33-r12-baseline-20260915 --candidate . --output /data/ycfeng/tmp/pr33-w03-fidelity-confirm --workers 2 --case co-location_offline_moe_model_basic_short --case pd-af-disagg_offline_moe_model_basic_short
```

Both runs reproduce the same first differences in `/data/ycfeng/tmp/pr33-w03-fidelity1` and `/data/ycfeng/tmp/pr33-w03-fidelity-confirm`. Every individual full invocation/environment is preserved in each case/side `invocation.json`.

## Observed divergences and source cause

1. Co-location MoE: request_metrics.csv and system_metrics.json match exactly under the unchanged comparator. First differing stage ledger record is row 1: attention_all_reduce_time baseline 1 ms, candidate 32 ms. The decoder supplies a 32-layer attention stage to `_create_corrected_execution_time_for_metrics`; the old helper reconstructed a single-layer ExecutionTime and dropped stage scope. The migrated helper preserves the supplied stage. Restoring the old reporting value would require an explicit legacy output projection; it must not restore ambiguous layer/stage runtime semantics.
2. PD-AF MoE PREFILL: first row attention KV-save baseline 0.03125 ms, candidate 1 ms. The old disaggregation dummy branch constructed an ExecutionTime with stage count 32, then multiplied all per-layer components by requested_layers/stage_layers (1/32 for real single-layer scheduler calls). Migrating to one physical layer removes that count-dependent transformation. Observed TTFT baseline 17 ms, candidate 358 ms; absolute difference 341 ms, relative 2005.8823529411782%. Request E2E baseline 1657.5943718399844 ms, candidate 1998.594371839978 ms; absolute difference about 341 ms. The new stage owner charges CPU/PP once; no tolerance relaxation can reconcile the layer normalization difference.

The three sampled dense cases (co-location/PDD/PD-AF offline short) PASS full artifact comparison. Hybrid real production-constructor and stage/metrics tests: 25 PASS in 8.84 s before the later metadata ownership deduplication. This is scoped evidence, not W03 completion.

## Supported choices

A. Retain uniform one-layer runtime and stage-wide reporting semantics from the assigned plan; accept the specifically diagnosed baseline defects as intentional corrections. Keep unchanged comparator and report these baseline lanes as known semantic differences; validate them against independent per-layer/owner arithmetic and repeat full E2E. This is recommended because it matches the requested single-contract architecture. It changes historical dummy PD-AF timing and fixes underreported stage ledger values.
B. Preserve all historical numeric artifacts through explicitly named boundary adapters: normalize disaggregated dummy payloads according to historical call scope and project historical single-layer ledger outputs. This retains historical artifacts at the cost of documented legacy policies at dummy/reporting boundaries and would require a scope decision against the plan's stage-wide reporting acceptance.

Neither option alters native profiling values, numeric tolerance, golden fixtures, or hardware support. Do not silently choose either while D01 is pending. Continue independent work.

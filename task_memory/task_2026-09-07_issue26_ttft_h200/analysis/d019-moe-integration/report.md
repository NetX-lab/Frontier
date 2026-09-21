## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Actual fresh-generation training/first-stage query and heldout-query checks passed; dataset builder committed0199432c. |
| 2026-09-08 | Prepared per-op corrected expert data, fresh-cache first-forward query, and same-case missing-shape inventory. |

# Corrected MoE data integration

Status: actual sparse-table training and first-stage integration PASS; dataset builder committed0199432c. Full-case calibration remains pending. No production behavior or communication setting changes in this step. Preparation statements below are historical.

Root's current execution contract retains ideal communication and physical Frontier first-forward M4096. vLLM's observed physical M4097 is retained as a separate diagnostic comparison; no dummy population/interface change is introduced.

## Dataset and provenance

`tests/performance/issue26_moe_profile_dataset.py` builds `moe_first_forward.csv` from current-task `supplements/moe-uniform-01/moe.csv` and corrected `analysis/d019-moe-normalized/moe.csv`.

- 774 original current-task rows retain unchanged gating/shuffling measurements. Every old `time_stats.moe_grouped_gemm.*` value is disabled as NaN.
- 9 fresh M4096 rows (EP0/1/7 x seeds0/1/2) provide the corrected expert-path targets. Their other operation targets are disabled to isolate this repair's effect.
- No M4095 holdout or physical-M4097 reference rows enter corrected-expert training.
- Each row records its source file, original CSV line and target owner. There are zero retained old expert-path targets.

This is explicitly a sparse per-operator training table. It is not claimed to be a dense canonical profiler export. Existing per-target training filtering at `sklearn_execution_time_predictor.py:3048` and `training/moe_trainer.py:471` can drop NaN targets; the actual runtime path must still be checked. Both required standalone_legacy and prefill_hot gating populations remain available from the unchanged source.

`config.json` keeps the previous first-forward case settings and changes the MoE input and cache/output paths only. New cache `/data/ycfeng/tmp/issue26-d019-moe-integration-01/predictor-cache` was confirmed nonexistent before launch. The predictor tool owner is running the explicit `--allow-fit` diagnostic, which stops at first stage completion before any E2E completion claim.

## Matched corrected S versus new vLLM scopes

`matched_moe_sv.json` records the scope-aligned physical-M4097 comparison. vLLM complete expert compute is its grouped parent plus following sum, averaged across48 layers. S is the median of three seed medians from independent loops.

| DP/TP/EP | S context | S ms/layer | V ms/layer | Signed (S−V)/V |
| --- | --- | ---: | ---: | ---: |
| 0/0/0 | standalone | .458335996 | .410792000 | +11.574% |
| 0/0/0 | prefill_hot | .399632007 | .410792000 | −2.717% |
| 0/1/1 | standalone | .453167990 | .471597998 | −3.908% |
| 0/1/1 | prefill_hot | .401840001 | .471597998 | −14.792% |

The opposite rank/context shifts prevent choosing a context merely because one rank's error is small. V is an instrumented event-span measurement with unresolved host/rank effects; S hot prefix is synthetic. These are implementation/scope-aligned diagnostics, not a closed clean-TTFT budget or a demonstrated RF error. The production profiling context stays unchanged.

## Full-case corrected coverage remains incomplete

`coverage_inventory.json` lists the complete observed global-token grid and missing corrected sizes from the previous current-task100-request simulation. Grouping uses shared stage start/end timestamps and distinct replica_local_id within each group:3662 source batches join1844 forward boundaries. The initial scratch check incorrectly used Replica id as a DP-lane key; actual rows show shared Replica0 with distinct replica_local_id0/1. Corrected grouping passes common-start and lane-uniqueness checks.

There are188 observed global sizes:99 decode sizes (1–100 except58),80 single-prefill/mixed sizes, seven double-prefill sizes and two triple-prefill sizes. Existing corrected measurements cover only M4096/4097 in this observed grid, leaving186 sizes without measured corrected GG data. M4095 is an independent holdout and was not an observed old-run shape.

This inventory reuses request/token composition only, not old numerical timing. New costs can change admission and the future grid; it is a same-case supplementation plan, not an exhaustive future-state proof. EP2–6 map positions are also not directly sampled. The existing exact wrapper can accept the listed sizes with deterministic `routing_inputs`; only corrected expert-path measurements are needed to fill the changed operation, so a future GPU supplement should call `MoEWrapper.profile_grouped_gemm` directly rather than repeat unchanged gating/attention measurements. No GPU launch is performed here.

## Minimal later full-case supplement proposal

Root explicitly requests a minimal coverage plan, not188 exact measurements. `full_case_minimal_profile_plan.json` freezes22 proposed training sizes and10 disjoint holdouts, all verified present in the old same-case token-composition inventory. No GPU execution is started.

| Phase population | Training global M | Held-out global M |
| --- | --- | --- |
| Decode | 1,2,4,8,16,32,48,64,80,96,100 | 9,23,68,92 |
| One prefill plus inflight decode | 4096,4097,4098,4111,4114,4160,4195 | 4109,4141,4187 |
| Two prefills plus inflight decode | 8197,8233,8284 | 8208,8267 |
| Three prefills plus inflight decode | 12359 | 12371 |

The choices cover observed occupancy, the first4096 padding transition, and low/mid/high observed multi-prefill sizes. Frequent interior decode shapes are reserved for validation. Measure only the repaired expert operation through existing `MoEWrapper.profile_grouped_gemm(routing_inputs=...)`; retain EP0/1/7 receipts and three seeds. No unchanged attention/gating operation needs recollection for this repair.

Train on the corrected targets alone and evaluate actual P on untouched holdouts. Report absolute/relative errors, sample spread and frequency-weighted cost contribution. Add points only in failing regions. The final updated same-case simulation can reach new features because timing changes admission; inspect those before declaring full-case coverage. The task driver must apply this explicit split, since the first-stage exact script's current split is intentionally4096/4097 versus other token sizes.

## Actual first integration failure

The first fresh-cache initialization trained unchanged linear/attention predictors, then rejected the combined MoE input because its equivalent measurement labels mixed `CUDA_EVENT` and `cuda_event`. Preserved `moe_first_forward.csv` and query01 log. The dataset builder now canonicalizes with existing `MeasurementType.from_string(value).value`, producing `moe_first_forward_v2.csv` / `config_v2.json`; verified every other column is unchanged. No production or timing value change. The retry reuses only unchanged-model caches newly trained in this same generation; MoE had not yet produced a cache. The query owner is executing retry02.

## Actual integration and holdout results

Retry02 PASS. Existing trainer logs prove it drops774/783 rows before grouped-expert training and9/783 before shuffling training; gated-hot models drop9/396. There were6 successful RF fits in the successful process. Prior unchanged-model caches were produced from an initially nonexistent cache during query01; no earlier-version cache was imported.

Actual new grouped-expert P is **.4619448847240872ms/layer**, returned by measured exact lookup from exactly9 corrected M4096 rows. Other10 independent first-forward compute predictions are bit-identical to the previous query. Actual first-stage boundary is **72.32753521728748ms**, versus65.79007690374297ms before. The6.537458313545ms change equals48 times the sole grouped-expert prediction increase to floating-point precision. The run stops before first stage completion handling and establishes no E2E request metric.

`first_forward_validation.json` independently checks the query owner's raw receipt and source rows. `holdout_query.json` calls the actual `_get_on_demand_prediction` method on the freshly trained GG cache with untouched4095 features; no additional fitting or4095 E2E case occurs.

| Holdout M / EP | Actual query path | P ms | S mean of3 seed medians, ms | Absolute error, ms | Signed relative error |
| --- | --- | ---: | ---: | ---: | ---: |
| 4095 / 0 | measured exact, duplicate local feature | .461944885 | .445365330 | .016579555 | +3.723% |
| 4095 / 1 | measured exact, duplicate local feature | .461944885 | .457087994 | .004856891 | +1.063% |
| 4095 / 7 | actual RF, untrained local feature | .461809295 | .473600000 | .011790705 | −2.490% |

All three nearby heldout checks are below10%, but only EP7 exercises a distinct key. All9 training rows share one local-feature vector: the fitted RF has not learned a size-response relationship. Therefore these checks validate the bounded first-forward correction and a nearby holdout, not decode/full-case model generalization. Current physical-M4096 and ideal communication remain unchanged; paired V physical-M4097 differences remain explicitly separate.

# D23-MODEL-IDENTITY-REUSE verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded narrow identity reuse, real public regression RED/GREEN and frozen-parent iteration/audit equality. |

## Outcome

Source/tests commit `7e98e5c7039695b1eefb56b245e108c1f6a54c8b` reduces only fixed measurement-identity parsing. Committed complete public iteration measured8.061941/7.931088s versus frozen parent9.451313/9.549744s: savings1.389372/1.618656s (14.70%/16.95%). Loop15.999181s versus19.007184s,15.82% less wall. **Still RED** against unchanged2.012201733s optimistic budget. No1024 completion, native accuracy, strict77 or solved-timeout claim.

Task/session: `D23-MODEL-IDENTITY-REUSE` / `ses_e9f519775ffe3Cv0bhEzJSKNhH`; worktree `/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf`, branch `feat/step5-mrv2-perf`, base clean `a5e0879`. Parent confirmed production parity before change. No parent outputs, original fixtures/caches/snapshot, routing, metrics configuration or diagnostic script changed. No GPU/external/native/fullCLI work. Calibration order remains workflow→GPU/opgap→CPU overhead last; this is Python execution enablement, not CPU-gap modeling.

## Source boundary and lifetime

Changed files:

1. `frontier/execution_time_predictor/prediction_model_registry.py`: private estimator-bound `(full current marker, parsed identity)` reuse through `_measurement_identity`; warm at `_set_measurement_identity` and successful `_validate_cached_measurement_identity`. A changed marker reparses and is checked through existing canonical identity validation; malformed/nonstring/empty/nonfinite/null metadata rejects without installing a replacement cache entry. Original conflicting attachment rejection stays before any update.
2. `frontier/execution_time_predictor/step5_cost.py`: measured attention proof reads the registry-owned parsed identity, retaining every paired-marker/runtime model/quant/device/TP/block/statistic/kind/target check and dynamic duration/count query.
3. `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`: measured FFN proof evidence-kind reads selected estimator identity; per-query `_get_family_model` still runs. Added explicit paired-marker comparison, required by the user; original FFN lacked this runtime check. No selection/model/result/feature cache, proof output change, lane suppression or timing factor.
4. `tests/unit/test_step5_model_identity_reuse.py`:13 real fitted-estimator cases at attachment→public stage and bundle-load→public attention, repeated fixed-read count, dynamic features/count failures, marker mutation/restoration, replacement with unchanged hash, invalid markers and pair conflict.

Modules remain below2000 lines: registry1068→1083, Step cost263→264, sklearn_moe1791→1793. Existing obsolete local FFN `json` import is removed. No oversized module/exception document is touched; keeping parsing with the existing canonical registry avoids a second model owner or alternative API. Runtime consumers only read the private parsed dictionary; the current full marker, not hash, is authoritative.

### Initializer versus restored state

Production new training attaches/warm-parses before atomic model storage. Existing artifact load validates canonical identity then warms the derived tuple in memory; no rewrite of original pickle is required. Marker comparison remains on every access, so estimator replacement or mutation cannot reuse the old selected model's proof. Independent family registries and predictor nested state stay unchanged.

Trusted diagnostic snapshot persists model references, not estimator objects. Its unpickler resolves them to the current SHA-validated bundle-loaded estimators; original savedstate predates the new tuple. Current `load_model_bundle` warms the tuple before state restoration. Verified original copied artifact has no parsed tuple; access installs a marker-equal tuple. **No diagnostic restore adjustment needed; no old fixed cache is unknowingly resurrected.** The original savedstate file was not changed. Production full initialization337/339s was not rerun or optimized.

## Environment and commands

Python3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python`; workdir perf worktree. Every command bounded60s, logs/temp/basetemp under `/data/ycfeng/tmp`. No dependency installs or broad CSV inspection. Independent checks ran concurrently; host contention means per-role speed is not a controlled microbenchmark.

### Regression RED before source change

```bash
timeout 60s env PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_model_identity_reuse.py -q --basetemp=/data/ycfeng/tmp/d23_identity_red02
timeout 60s env PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_model_identity_reuse.py::test_bundle_load_to_public_attention_reuses_identity_and_checks_mutation -q --basetemp=/data/ycfeng/tmp/d23_identity_attention_red01
```

- `d23_identity_red02.log`:14 repeated fixed reads for two public stage calls; mutation/replacement2 instead of1; pair conflict/empty identity not rejected.8 failed/1passed8.48s; one public-attention setup field failure remained in this first run.
- Fixed test setup **before** production change by using the actual fittedFFN predictor and its measured input-layernorm. `d23_identity_attention_red01.log`: correct bundle-load→public-attention seam failed specifically on2 reads instead of0,7.20s. No synthetic cost substituted for the measured root.

### Focused and affected GREEN

All invocations use the same environment prefix and `python -B -m pytest <files> -q --basetemp=<new path>`, `timeout 60s`:

| File set | Basetemp/log stem | Actual | Result |
| --- | --- | --- | --- |
| `test_step5_model_identity_reuse.py test_step5_contract_reuse.py test_step5_model_bundle.py` | `d23_identity_green_final01` |96 tests37.51s|PASS|
| `test_step5_measured_training.py test_step5_measured_replay.py test_step5_measured_ffn_replay.py test_step5_mrv2_inputs.py` | `d23_identity_measured01` |92 tests25.55s|PASS|
| `test_step5_measured_ffn.py test_step5_measured_stage_boundaries.py test_measured_attention_root_metrics.py test_on_demand_prediction_contract.py` | `d23_identity_related04` |86 tests45.42s|PASS|
| `test_step5_attention_input_union.py` | `d23_identity_union01` |25 tests14.21s|PASS|

Total299 passing affected checks; no strict77 or broad CPU suite. Initial combined related/union invocation hit60s without final verdict, then split into the existing file sets above and passed without raising timeout. An earlier invocation named nonexistent `test_step5_root_only_eager_policy.py` and collected0 tests; corrected using repository file discovery. Final AST and `git diff --check` PASS. The final extra malformed-marker cases only add cold/mutation validation; successful warm/query behavior was unchanged from the preceding92-test pass.

## Unchanged complete public iteration and audit

Executed after source commit; use **new** output names when repeating:

```bash
timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_actual06_iteration.py --output /data/ycfeng/tmp/d23_identity_committed_wall01 --max-runtime 30 --resume-state /data/ycfeng/tmp/d23_actual06_exact_q1_state/state.pkl --start-emitted 1 --outputs 3
timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_actual06_iteration.py --output /data/ycfeng/tmp/d23_identity_committed_audit01 --max-runtime 30 --resume-state /data/ycfeng/tmp/d23_actual06_exact_q1_state/state.pkl --start-emitted 1 --outputs 3 --audit
```

Exit1 means expected latencyRED; outputs/invariants completed. Both receipts report source `7e98e5c`.

| Run | Setup | Loop | Complete intervals | Truth |
| --- | ---: | ---: | --- | --- |
| Frozen parent | original control |19.007184s|9.451313/9.549744s|frozen|
| Earlier unchanged new source repeat `wall02` |16.520511s|16.454303s|8.264371/8.182132s|parent event/stage/lane equal|
| Final source precommit `final_wall01` |15.391591s|16.174721s|8.007545/8.160508s|parent event/stage/lane equal|
| Committed wall |14.482305s|15.999181s|8.061941/7.931088s|PASS|
| Committed audit |15.076803s|21.946963s|11.142503/10.797263s|PASS; audit overhead separate|

Frozen parent `/data/ycfeng/tmp/d23_parent_actual06_wall01` and audit `/data/ycfeng/tmp/d23_parent_actual06_audit01` were read only. Comparison `/data/ycfeng/tmp/d23_identity_parent_comparison.json` asserts:

-1168 events and5632 lane ledger rows; all92 physical layers and original6144/1024 request intact.
-JSON event trace **bytes equal**; completed stage-ledger JSON **full equal**; lane digest **equal** `bdd74d1e1ec310fa94f8ad60af888d61bbc0e4279ba92349cdaed3fd33cb2526`.
-Complete artifact source-path/SHA/size descriptors equal; same four bundles and exact coverage.
-Independent audit full `source_truth.json` **dictionary equal** to frozen parent:41288 FFN+960 attention queries, currentCUDA_EVENT family, exact dynamic features/duration/count/model hashes/evidence/precision/proof unchanged.
-6016P+6208D collective queries, allhits/0miss, same dynamic population.
-Original flags unchanged: write_metrics=true,stage ledger=true,JSONeventtrace=true; originaloperationmetrics=false andoptrace=false. No flag disabling/new trace campaign.
-Normal committed loop has **zero** canonical-marker JSON loads (parent42248 loads3.128339s). Independent audit intentionally still reparses original identity42248 times3.240352s outside production proof construction; this is validation overhead, not a failed cache.

Committed role wall: D3.951603/3.657595s andP4.110338/4.273493s. Earlier repeated D4.865916/4.737160s shows host/run variation; do not claim a controlled0.8s/D savings or extrapolate exact long-run speed. Whole complete-iteration reduction and zero fixed-parse work are observed; complete iteration remains~4×budget.

## Residual measured blocker; no further project launched

A short affected-path cProfile was collected **before** the user's final narrowing; no new overhead modeling, sampling or trace campaign follows it. `d23_identity_profile02` covers one complete originalq1 interval (2816lanes/61952Request constructors),15.551940s under profiler. Diagnostic receipt labels fewer than two intervals `INCOMPLETE_PREFIX`; this profile is call-work evidence, not a full latency-gate or1024 result. The earlier two-interval30s profile stopped partway through the second interval and is not treated as completion.

| Remaining function, one complete q1 | Calls | Profile cumulative |
| --- | ---: | ---: |
| EP wave preparation |176|12.304s|
| Public stage prediction |3205|10.458s|
| FFN operator query |20644|3.866s|
| Dynamic duration/count lookup |42248|3.406s|
| Lane batch construction |2816|2.000s|
| FullRequest construction |61952|1.458s|
| Prediction bounds parsing |89800|1.064s|
| Attention config validation |499|0.836s|
| Feature-name validation |84496|0.452s|
| Per-query family/model selection |20664|0.271s|
| Reused identity access |21124|0.025s|

Inclusive profile times overlap and must not be added. No canonical-marker`loads`, no prior fixed layer/precision construction recurs. Remaining runtime blocker is still the complete lane/stage dynamic work, not a reason to weaken cost/proof or suppressPdummy. Parent chooses any next minimal enablement action after workflow/routing alignment; no speculative bundle is implemented here.

## Review and remaining acceptance

Review diff: `/data/ycfeng/tmp/d23_identity_review.diff`, generated by `git diff a5e0879 7e98e5c -- frontier tests/unit/test_step5_model_identity_reuse.py`. Four changed source/test files listed above;15line registry addition and smallStep delegates, no giant source edits. Source/tests committed separately from task records.

Parent owns independent review/merge,explicitbalanced routing,originalfull6144/1024 offline/onlineCLI,strict77 and native/workflow/GPU/opgap calibration acceptance. Later-contextPshutdown,full initialization/final metrics export and long-run memory behavior remain unverified. No bound increase or timeout-solved claim.

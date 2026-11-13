# Post-prepared actual CPU pricing report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archive source-bound wave, pricing-only caller census, fresh-value control and nine lifecycle records. |

## Source, setup and boundaries

- Source: `eece87b2378bf86c6673b74f1a769f566bc5792a`, worktree `/data/ycfeng/Frontier/.worktrees/step5-d29-post-prepared-pricing`; submodule `3dac53c`.
- Original harness: `/data/ycfeng/tmp/d29_main08_residual_repro/wave_loop.py`. The rebound script changes only source binding and instrumentation of the pricing boundary/caller census. Both `_step5_ffn_calibration` and `_step5_residual_calibration` inherited fixture overrides are removed as in the accepted full-wave harness.
- Actual DECODE position6147, layer60, sixteen EP lanes; current registry/models, immutable paired caches, current backend and prepared structural query. No diagnostic preparation adapter or numerical reuse.
- Real CLI config is parsed only after all output/cache paths are relocated to new scratch. No retained run config is rewritten.
- Interpreter: existing `/data/ycfeng/envs/frontier-py310/bin/python`; no dependencies installed. Every heavy command used the shown sudo systemd scope, MemoryMax30G and timeout60s.
- Full captured layer60 comparison uses only the existing bounded8MiB tail, never the1.4GB complete trace. Baseline and control each have208/208 captured pairs, zero differences in duration, compute/count proofs, native/source context and complete collective plan/cost.
- Script and source hashes, including original/rebound scripts, are retained in `provenance.json`. Original and rebound script bytes are kept in scratch with a unified diff. Existing accepted scripts/source/artifacts were not modified.

## Exact final baseline command

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_post_prepared_pricing/pricing_probe.py --output /data/ycfeng/tmp/d29_post_prepared_pricing/pricing-final01 --iterations 4 > /data/ycfeng/tmp/d29_post_prepared_pricing/pricing-final01.log 2>&1
```

The output directory is exclusive; use a fresh name for another run. Final sequential observations:

| Variant | Cold pricing seconds | Warm pricing seconds | Warm mean | Warm pricing JSON decodes | Current-file reads |
| --- | ---: | --- | ---: | ---: | ---: |
| Actual baseline | 1.861020 | 1.017791 / 0.995219 / 1.050349 | 1.021120 | 1520 | 80 |
| Fresh-binding normalization control | 1.821565 | 1.073023 / 1.038505 / 1.089010 | 1.066846 | 1232 | 80 |

Every iteration emits208 events. Full payload SHA256 is always `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`. One16464-row/168-key preparation occurs; accepted structural preparation is used throughout. Timings include current file-read overhead, which is not suppressed or controlled here.

## Strict pricing profile and parse decomposition

Same scope/environment, command:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_post_prepared_pricing/pricing_probe.py --output /data/ycfeng/tmp/d29_post_prepared_pricing/pricing-profile02 --iterations 2 --profile > /data/ycfeng/tmp/d29_post_prepared_pricing/pricing-profile02.log 2>&1
```

Warm profile is1.352193s; existing current-file reads consume0.777336s. The remaining measured interval is approximately0.574857s, including profiling/instrumentation overhead. This subtraction is descriptive, not a proposal to suppress reads. Pricing1520 parses plus trace448 equals the earlier1968 warm total. Trace and verification are excluded from the following pricing-only caller census:

| Source/caller | Warm calls | Unique texts | Input bytes |
| --- | ---: | ---: | ---: |
| `prediction_model_registry.py:410`, registered binding key | 272 | 2 | 353344 |
| `prediction_model_registry.py:672`, explicit binding fields | 192 | 4 | 116048 |
| `prediction_model_registry.py:684`, nested selected binding | 96 | 2 | 52784 |
| `prediction_model_registry.py:687`, fresh normalized result | 176 | 2 | 229712 |
| `collective_plan.py:87`, `to_dict` context | 192 | 1 | 411072 |
| `collective_plan.py:88`, `to_dict` buffer | 192 | 2 | 281880 |
| `moe_operator_times.py:99`, current plan context validation | 64 | 1 | 137024 |
| `moe_operator_times.py:100`, current buffer validation | 64 | 2 | 93960 |
| `collective_sim_cc_backend.py:818`, backend context validation | 64 | 1 | 137024 |
| `collective_sim_cc_backend.py:824`, backend buffer validation | 64 | 2 | 93960 |
| `collective_plan.py:125`, actual count receipt validation | 64 | 8 | 861128 |
| `collective_sim_cc_backend.py:825`, count receipt reader | 64 | 8 | 861128 |
| `moe_routing_workload.py:207`, fresh selected native point | 16 | 2 | 22642 |
| **Pricing total** | **1520** | not additive | **3651706** |

The byte total is repeated input bytes, not a memory-allocation measurement. No cache-key normalization or serialized identity changes were made. Collective file preparation/lifetime and trace owners remain separate; these calls are observed, not modified.

## Non-I/O hot paths and existing preparation

Profile02 cumulative costs are nested and must not be summed:

| Real path | Warm calls | Cumulative seconds | Interpretation |
| --- | ---: | ---: | --- |
| `prediction_model_registry.py:727 _measurement_identity` | 240 | 0.000516 | Existing marker-keyed parsed identity is used, not bypassed |
| `prediction_model_registry.py:666 _native_measurement_binding` | 176 | 0.026676 | Fresh normalized selector result; about2% of profiled wave |
| `prediction_model_registry.py:370 _get_family_model` | 544 | 0.027657 | Live current family/precision/contract/native selection |
| `prediction_model_registry.py:515 _models_view_for_family` | 16 | 0.082998 | Generic current registry projection; includes typed-contract work |
| `prediction_model_registry.py:458 _resolve_cluster_model_contract` | 384 | 0.071614 | Fixed geometry resolved repeatedly; cache safety not established here |
| `sklearn_moe_execution_time_predictor.py:108 _step5_ffn_model` | 160 | 0.031676 | Current layer/native binding selected on every call |
| `sklearn_moe_execution_time_predictor.py:135 _predict_step5_ffn_operator` | 144 | 0.071469 | Live model/source/paired-count/exact-feature/positive-value checks |
| `sklearn_execution_time_predictor.py:4729 _get_on_demand_prediction` | 288 | 0.024777 | Actual duration and kernel-count lookups; never cached by this task |
| `moe_routing_workload.py:187 _step5_position_queries` | 16 | 0.299718 | Includes current read and all fresh per-point checks |

`CollectivePlan.to_dict/cache_key`, count validation and proof object construction also have observed CPU costs, but no alternative serialization/cache/trace control is attempted here. Required fresh-context processing remains live.

## Hypotheses and one-variable evidence

The ranked falsifiable hypotheses were stated after the first real profile and before controls:

1. Repeated binding normalization can be prepared without changing fresh-return/mutation contracts. Expected maximum gain is small; test a faithful structural-only control.
2. Registry-view typed-contract resolution is a larger non-I/O repeated cost. Profiling establishes repetition but not a safe lifecycle/causal repair; no new view cache is attempted.
3. Raw parse count alone overstates fresh per-point CPU work. Pricing1520 parses total only0.046621s in profile02; accepted marker cache is already active.
4. Numerical/paired-count and layer-proof work is real and must stay live. The288 actual numerical calls remain in both variants.

Control command:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_post_prepared_pricing/binding_control.py --output /data/ycfeng/tmp/d29_post_prepared_pricing/binding-control-final01 --iterations 4 --binding-control > /data/ycfeng/tmp/d29_post_prepared_pricing/binding-control-final01.log 2>&1
```

The scratch-only control keys successful normalization by the complete current input serialization, retains a serialized normalized result, and JSON-decodes a **fresh mapping for every call**. Changed inputs rebuild; failed original normalization is never cached. It does not suppress any numerical/source/point/model/count check or reuse a duration/result.

It observes704 binding calls,7 successful normalization inputs,704 fresh result decodes,0.112906s total binding-control time over four waves. It reduces warm decodes by288 but the sequential warm mean is slightly slower (1.066846s vs1.021120s). This is **negative optimization evidence**, not a timing regression gate or a claim of universal slowdown. No new identity cache is justified by it.

## Identity/registry contracts and differential gates

`_measurement_identity` at727 reads the selected estimator's current marker and rebuilds its existing parsed value when the marker string changes. `_set_measurement_identity` at740 canonicalizes before persistence; the only other production marker assignment found is independent trainer setup `moe_dataset_training.py:424`. No serving in-place mutation path for the parsed mapping was found, but the mapping is mutable and no new immutability claim is invented.

`_native_measurement_binding` at666 accepts explicit mappings or JSON strings, validates nonempty mappings and strips point-layout fields through the existing normalizer. Line687 deliberately returns a fresh JSON tree. Returning a cached mutable dictionary would change that contract.

`_store_model_precision` at341 registers canonical native binding in the existing immutable tuple key. `_get_family_model` at399–410 normalizes the current explicit request and decodes those registered-key strings for comparison. They are not interchangeable with the estimator's current marker: supported estimator/marker replacement must remain visible to later selected-source and paired-count checks. The existing parsed estimator identity cannot blindly substitute for the registered native key.

Lifecycle command, same bounds/environment:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_post_prepared_pricing/identity_contracts.py > /data/ycfeng/tmp/d29_post_prepared_pricing/identity-contracts02.log 2>&1
```

Nine records pass: current cached estimator identity; explicit changed source input; mutation of a returned binding cannot contaminate a later return; empty native mapping; invalid JSON; valid marker replacement refreshes the existing reader; foreign marker rejected at actual numerical seam; actual registry estimator replacement reads its own marker; missing actual count key rejected. Restored normal pricing returns0.0102ms through the actual estimator, not a fabricated value.

## Actual comparison and verdict

Comparison command example, same scope/bounds:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_post_prepared_pricing/compare_actual.py --replay pricing-final01 > /data/ycfeng/tmp/d29_post_prepared_pricing/pricing-final01-comparison.log 2>&1
```

The same command with `--replay binding-control-final01` verifies the control. Both checked208/208 pairs with zero differences. These two results and the nine lifecycle records are different checks, not a summed test-suite pass count.

**Verdict: no new identity-cache repair allocation recommended (`NO_SAFE_SEAM` recommendation for a high-impact repair on this evidence).** This does not prove that every possible structural optimization is impossible. It means the accepted marker mechanism is already used, the tested fresh-value control is not faster, and a generic registry-view cache has no tested replacement/selection lifetime design. Any later view/cache proposal requires separate root scope and contract evidence; numerical/receipt gates cannot be weakened for it.

No whole CLI, startup, other layergroups/workload points, full1024-token throughput, native/GPU qualification or2400s success is established.

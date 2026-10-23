# D29 registry mixed-view repair test report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive exact coexistence RED/GREEN,217-check acceptance and revised budget. |

TaskID: D29-REGISTRY-VIEW-REPAIR. Source baseline8f96408c,
records baselineb6714757; worktree
`/data/ycfeng/Frontier/.worktrees/step5-registry-bindings`.
CPU-only on kun-workspace-vgen2. Python
`/data/ycfeng/envs/frontier-py310/bin/python`3.10.6;
pytest9.1.1, sklearn1.7.2, pandas2.3.3, numpy2.2.6.

## Acceptance results

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| RED real mixed views | Legacy/native or non-native typed/native same-name view fails before fix |4 failed,0 errors/skips,36.54s; exact ValueError in view resolver | PASS expected RED |
| GREEN both insertion orders/domains | Family and cluster views retain bare model and native hash label |4 passed,0 errors/skips,35.85s | PASS |
| Strict public get | Unselected same-name query remains ambiguous; explicit native selector resolves |Both asserted in all four cases | PASS |
| Existing semantics | Legacy/typed/precision defaults, hashes/cache and native bundle retention remain |Original213 acceptance tests plus4 new cases:217 passed,0 failures/errors/skips,213.85s | PASS |
| Source scope | Only registry and existing regression test change |3 registry additions,26 test additions; caller diff empty | PASS |
| Allocations |Incremental<=58, total<=360; all revised caps |29/58,331/360; registry96/120,test200/200,predictor113/118,routing136/145; callers35/60 | PASS |
| Whitespace |Clean diff | `git diff --check` has no output | PASS |

The existing actual trainer/binder fixture supplies the native FP32 gate. Public
`get_models()` and `get_models_for_cluster(MONOLITHIC)` are executed, including
the real Step contract resolver. A non-native typed marker is attached before
registration for the typed-domain case; the original legacy-domain registration
attaches its existing matching marker afterwards, exercising matching legacy
contract resolution without relaxing lookup. Both insertion orders are tested.

## Exact RED and tight GREEN commands

Before the three-line fix, the RED command was run with the final four-case test:

```bash
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_prediction_measurement_bindings.py -k mixed_native_non_native -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_registry_view_red02 --junitxml=/data/ycfeng/tmp/d29_registry_view_red02.xml > /data/ycfeng/tmp/d29_registry_view_red02.log 2>&1
```

RED exit1 with four exact mixed-domain view failures, no setup errors. Two-case
RED01 first confirmed the typed-domain pattern; RED02 is the final matrix.
After the fix:

```bash
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_prediction_measurement_bindings.py -k mixed_native_non_native -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_registry_view_green01 --junitxml=/data/ycfeng/tmp/d29_registry_view_green01.xml > /data/ycfeng/tmp/d29_registry_view_green01.log 2>&1
```

## Exact final affected-check command

```bash
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_prediction_measurement_bindings.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_contract_reuse.py tests/unit/test_step5_model_identity_reuse.py tests/unit/test_step5_model_bundle.py tests/unit/test_step5_measured_manager.py tests/unit/test_step5_measured_public_predictor.py tests/unit/test_predictor_cache_atomicity.py tests/unit/test_shared_prediction_model_manager_mixed_layer_moe.py tests/unit/test_step5_measured_training.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_registry_view_final01 --junitxml=/data/ycfeng/tmp/d29_registry_view_final01.xml > /data/ycfeng/tmp/d29_registry_view_final01.log 2>&1
```

Expected zero failures/errors/skips. Actual217 passed in213.85s, exit0.
Tests include actual shared fit/binder, independent fit/cache, dual layer3/88
native binding, all native bundle reloads, exact lookups and legacy/default/cache.
This is not an independent public-chain or native GPU qualification claim.

## Budget and artifacts

Source delta vsb6714757: registry3+0, test26+0=29gross; remaining allowance58.
Whole repair vsd96b3279: registry85+11=96, predictor17+2=19,
routing14+2=16,test200+0=200,total331/360. Caller delta remains35/60.
Revised path caps vs1aee0af: registry96/120, predictor113/118,
routing136/145,test200/200. No caller, global typed identity or cache-hash source
changed. Root-owned collective-sim gitlink is not part of the fix.

- Raw RED: `/data/ycfeng/tmp/d29_registry_view_red02.{log,xml}`.
- Raw tight GREEN: `/data/ycfeng/tmp/d29_registry_view_green01.{log,xml}`.
- Raw final GREEN: `/data/ycfeng/tmp/d29_registry_view_final01.{log,xml}`.
- Durable summaries and diff: local `evidence/` beside this report.
- Original binding receipts: parent `registry_binding/evidence/` archived byb6714757.

Root re-review and integration remain pending. Pre-existing independent consumer
binding collapse/public manager issue is explicitly outside this two-path repair.
No GPU/platform/network/dependency or peer operations were performed.

# CORE three-finding repair test report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record R1 actual inheritance/delegation RED/GREEN with scoped quota. |
| 2026-10-23 | Record R2 automatic preflight/trainer missing-descriptor RED and36PASSGREEN. |

TaskID:D29-CORE-REVIEW-REPAIR. Isolated worktree `/data/ycfeng/Frontier/.worktrees/step5-core-review-fixes`,baseline d96b3279. No full CLI or GPU/native qualification. Root submodule checkout3dac preserved unstaged. Commands use `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp`;py312 addsVLLM_TARGET_DEVICE=cpu. Python3.12.3 `/data/ycfeng/envs/asyncopd-py312/bin/python`,Python3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`.

## R1 actual supported routing callable chain

Observed source: supplied published export `/data/ycfeng/tmp/d29_published_sources01/source/vllm/model_executor/layers/fused_moe/router/{fused_moe_router,base_router,routing_simulator_router}.py`. Reached chain: inherited select_experts -> inherited _select_experts -> bound router._compute_routing -> RoutingSimulator.simulate_routing -> selected position_cyclic strategy.route_tokens. Existing PRICING C1 probe independently verifies these same callables. Production appends these three actual callables to the existing full-source verifier tuple; previous _select_experts check stays. No recursion/verifier changes/native strategy changes.

Test reuses real exported class definitions in the existing sealed full-source CPU fixture, with only unrelated import/environment hardware seams isolated. Valid foreign replacement tensors first execute through the real inheritance chain; class/public selector/inherited selector/module filename/spec/hash remain unchanged. Wrong-source replacement is in callable's code filename, not misleading changed module metadata. Four cases cover previous selector and each reached implementation. Absent supplied source export is explicitly SKIP; this is CPU provenance fault injection, not published-runtime/kernel execution qualification.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-r1-red01 tests/unit/test_step5_moe_profiling.py -k 'only_replaced_bound_router' > /data/ycfeng/tmp/d29-core-review-r1-red01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-r1-red02 tests/unit/test_step5_moe_profiling.py -k 'only_replaced_bound_router' > /data/ycfeng/tmp/d29-core-review-r1-red02.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-r1-green01 tests/unit/test_step5_moe_profiling.py -k 'only_replaced_bound_router or position_adapter_binds or real_device_timer or masked_expert or oracle_callable or lazy_norm or actual_source_context' > /data/ycfeng/tmp/d29-core-review-r1-green01.log 2>&1
```

- RED01:3FAIL/1PASS2.96s;bad injected string results fail actual inherited dtype conversion before verifier. Not counted as genuine reproduction.
- RED02:3FAIL/1PASS3.10s;three `DID NOT RAISE ValueError` for actual compute/simulator/selected strategy. Previous selector negativePASS. This proves incomplete bound callable verification.
- GREEN01:23PASS/11deselected9.21s;all four actual-chain negatives plus timer/restore/event preservation/source/Gate/oracle seam tests. Two SWIG deprecation warnings only.
- R1repairgross44. Baseline1aee totals:moe_impl56/62,test303/350 after both authorized overlays. All six exclusive path caps remain authoritative; final accounting follows remaining repairs.

## Retained earlier CLI terminal, not a new run

Original shell notification already arrived before this repair task. `/data/ycfeng/tmp/d29-core-position-main-seam03/command.json`, `execution.json`, `run.log`:exit1,wall1555.208038s,timed_out=false,internal2400s unchanged. Initialization passes after2a862c25 and reaches actual PREFILL sync/EP wave then missing old `data/profiling/network/h800/collective_local.json`. This vector predates complete CPU payload/path forwarding, so proves neither current complete fixture/main nor native importer rejection. No polling, original worktree edit or CLI rerun in this repair branch.

## R2 automatic supplied source admission

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_profiling_output_metadata.py -k 'automatic_position' > /data/ycfeng/tmp/d29-core-review-r2-red01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_moe_routing_runtime.py tests/integration/test_step5_measured_main.py -k 'not position_source_wrapper' > /data/ycfeng/tmp/d29-core-review-r2-green01.log 2>&1
```

- Existing actual producer creates canonical252-row FP8/BF16 source/gathered CSV; initial real manager preflight succeeds. Only routing_input removed (or replaced with empty descriptor), then actual automatic shared trainer with its preflight or independent trainer called. Estimator/metadata sentinel proves rejection order, not a shallow filter-only test.
- RED01:absent column shared/independent reach forbidden estimator/metadata (`Failed: Unvalidated source reached estimator or legacy helper`). Invalid descriptor already rejected;old test regex schema_version mismatched actualsource_frames and was corrected. No production relaxation.
- GREEN01:36PASS/3deselected68.81s;four new source negatives plus current automatic actual producercold/warmfit/cache,command/defaults/metadata contracts. Hardware wrapper excluded here andcoveredR1py312/currentintegrationseam separately.
- Production requires routing_input column and always calls existing validate_position_moe_source_row. No new source validator,estimator/default,unownedtrainer orregistryedit. R2delta3added/4deleted+40testadded=47;baseline runtime103/133,test125/135. R1+R2delta91/230.

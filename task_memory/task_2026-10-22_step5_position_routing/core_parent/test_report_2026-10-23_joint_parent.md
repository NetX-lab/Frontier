# Joint CORE, producer and registry parent verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Independently verify frozen combined code, retain five fixture failures and complete three wrapper checks with isolated dependencies. |

Source: `bfbc87ad6b99d5be357b3fd8b680c8baefc459c4`, submodule
`3dac53c544e546574cb6e8078cd14053d0916113`. This combines CORE e49b26f8,
PRICING a7e2a3ca and registry a5189e19. Exact component diffs and receipt are
retained beside this report. Worktree: `/data/ycfeng/Frontier/.worktrees/verify-step5-core-closure-01`.

## Results

| Check | PASS | FAIL | SKIP | Duration |
| --- | ---: | ---: | ---: | ---: |
| Combined registry, trainer, CPU consumer and artifact checks | 431 | 5 | 4 | 274.93s |
| Profiling provenance, real timers and diagnostics persistence | 24 | 0 | 0 | 9.44s |
| Native-source dispatch probe tests | 235 | 0 | 15 | 7.41s |
| Actual source/gathered wrapper with BF16, FP8 and deferred prepare | 3 | 0 | 0 | 9.96s |

Unique union: **717 cases,693 PASS,5 FAIL,19 SKIP**. There are no duplicate case
IDs across these four selections. All five failures are the previously identified
two runtime fixtures missing actual model/group/registry state and three training
fixtures missing `routing_input`. Their exact IDs and errors are retained in
`parent_results.json` and the full XML/logs. They remain failures; full regression
is not accepted. The original CORE owner has a separate two-file80-gross repair.

## Commands and environment

Python3.10.6: `/data/ycfeng/envs/frontier-py310/bin/python` (P10).
Python3.12.13: `/data/ycfeng/envs/asyncopd-py312/bin/python` (P12).
Working directory is the frozen worktree above. Common environment:
`PYTHONPATH=$PWD:$PWD/tests/unit`, `PYTHONDONTWRITEBYTECODE=1`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`TMPDIR=/data/ycfeng/tmp`. Both `FRONTIER_TEST_CPU_WRITER_FILE` and
`FRONTIER_TEST_CPU_WRITER_SHA256` are unset for the combined run.

All commands use `-B -m pytest -q -p no:cacheprovider`; logs and XML use the
same `/data/ycfeng/tmp/<stem>` as the `--basetemp` prefix.

```bash
timeout 360s "$P10" -B -m pytest \
  tests/unit/test_prediction_measurement_bindings.py \
  tests/unit/test_moe_grouped_gemm_padded_features.py \
  tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py \
  tests/unit/test_step5_contract_reuse.py tests/unit/test_step5_model_identity_reuse.py \
  tests/unit/test_step5_model_bundle.py tests/unit/test_step5_measured_manager.py \
  tests/unit/test_step5_measured_public_predictor.py tests/unit/test_predictor_cache_atomicity.py \
  tests/unit/test_shared_prediction_model_manager_mixed_layer_moe.py \
  tests/unit/test_step5_measured_training.py tests/unit/test_collective_sim_cc_backend.py \
  tests/unit/test_op_trace_utils.py tests/unit/test_ep_wave_trace_context.py \
  tests/unit/test_step5_moe_pricing.py tests/unit/test_moe_profiling_output_metadata.py \
  tests/unit/test_moe_routing_runtime.py tests/integration/test_step5_measured_main.py \
  -k 'not position_source_wrapper' -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/d29_core_closure_parent01 \
  --junitxml=/data/ycfeng/tmp/d29_core_closure_parent01.xml \
  > /data/ycfeng/tmp/d29_core_closure_parent01.log 2>&1
```

P12 runs additionally set `VLLM_TARGET_DEVICE=cpu` and
`STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm`:

```bash
timeout 180s "$P12" -B -m pytest tests/unit/test_step5_moe_profiling.py \
  -k 'position_row_preserves or only_replaced_bound_router or position_adapter_binds or real_device_timer or masked_expert or oracle_callable or lazy_norm or actual_source_context' \
  -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_core_profiling_parent01 \
  --junitxml=/data/ycfeng/tmp/d29_core_profiling_parent01.xml \
  > /data/ycfeng/tmp/d29_core_profiling_parent01.log 2>&1
timeout 180s "$P12" -B -m pytest tests/unit/test_step5_dispatch_probe.py \
  -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_core_producer_parent01 \
  --junitxml=/data/ycfeng/tmp/d29_core_producer_parent01.xml \
  > /data/ycfeng/tmp/d29_core_producer_parent01.log 2>&1
```

The three wrapper tests initially had a known missing-fasteners dependency.
Inspection confirms P12 also lacks sklearn, plotly and ddsketch. An isolated
`/data/ycfeng/tmp/d29_py312_cpu_extras01/site` supplies only the missing packages
and their required missing dependencies; neither existing environment is edited.
Versions match the installed P10 test environment, recorded with wheel hashes
in `wrapper_dependency_receipt.json`. Py312 wheels were downloaded using the
published image's python3, `--network=host`, cleared proxy variables,
`PIP_CONFIG_FILE=/dev/null`, `pip download --no-deps --only-binary=:all:` and
`http://mirrors.i.basemind.com/pypi/simple/`. Installation used the same image
with `--network=none`, `pip install --no-deps --no-index --find-links=/extras/wheels
--target=/extras/site -r /extras/requirements.txt`. Raw commands' inputs and
download/install logs remain in that temporary directory.

```bash
PYTHONPATH="$PWD:$PWD/tests/unit:/data/ycfeng/tmp/d29_py312_cpu_extras01/site" \
timeout 120s "$P12" -B -m pytest tests/integration/test_step5_measured_main.py \
  -k position_source_wrapper -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/d29_core_wrapper_parent01 \
  --junitxml=/data/ycfeng/tmp/d29_core_wrapper_parent01.xml \
  > /data/ycfeng/tmp/d29_core_wrapper_parent01.log 2>&1
```

## Acceptance boundaries

Independent joint review closes producer P1/P2 and CORE R1/R2/R3 at source level;
the separate registry review closes mixed-view P2. Tests cover segmented real
prepared-context/observer/verifier, wrapper/CSV persistence and CPU exact-cost
consumption. No single complete native producer-to-CSV-to-export-to-H04 execution,
GPU/backend/graph, G3 or four-metric numerical qualification is established here.
Both actual trainer fit/cache paths remain required; an independent public
consumer is outside the approved P/D product path. Full offline/online and
strict77 remain pending.

# Parent model-group reuse verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Separate shared-suite and actual two-trainer receipts;archive records-only P2 closure and component acceptance. |
| 2026-10-23 | Verify the delivered three-path code through82 collected tests and unchanged actual cold/warm pricing. |

## Source and environment

Code3763e5b87a845eac186cfd481f0b56cffb2dd045; execution HEAD
88fe6c65db675609ee48043ca63b7673c5f739c8 adds only records. Workdir:
/data/ycfeng/Frontier/.worktrees/step5-d29-group-reuse, clean after verification.
Parent compares the three edited files byte-for-byte with3763e5b8 and checks
frontier/tests/data/Dockerfile equality between the two revisions. Python executable
is /data/ycfeng/envs/frontier-py310/bin/python,version3.10.6;BLAS/OMP threads are1.
Final component records-only HEAD is37c7832ee5ee2e9588ba6fed411a9e9894949ab8.
Primary reports the isolated tree clean and all three source/test files still
byte-identical to3763e5b8. Existing tests/pricing stay bound to execution HEAD
88fe6c65;they are not relabelled as fresh executions on37c7832e.

## Affected regression

```bash
timeout 300s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD:$PWD/tests/unit" TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/unit/test_prediction_measurement_bindings.py tests/unit/test_step5_measured_ffn_replay.py \
  tests/unit/test_step5_measured_ffn.py tests/unit/test_moe_routing_runtime.py \
  tests/unit/test_moe_position_routing.py tests/unit/test_step5_measured_training.py \
  --basetemp=/data/ycfeng/tmp/d29_group_reuse_parent01 \
  --junitxml=/data/ycfeng/tmp/d29_group_reuse_parent01.xml \
  > /data/ycfeng/tmp/d29_group_reuse_parent01.log 2>&1
```

Expected:no failures and only the existing Torch-dependent skip. Observed:
82 cases,81PASS,1SKIP,0FAIL/ERROR;164.431s. The skipped case is
tests.unit.test_moe_position_routing.test_one_source_tensor_interface_preserves_masks_and_request_intervals,
because this environment has no torch. PASS for the selected regression.
XML SHA256:a37987c7b72f64a87be43623b3136d93e12c8b103b00856cf525b8918177cb4f.
Coverage:shared consumer/trainer/fit/cache only;this command does not include the
independent trainer and does not establish both-trainer coverage by itself.

## Separate actual trainer verification

Parent executed these two existing tests in the same isolated worktree on88fe6c65:

```bash
timeout 120s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD:$PWD/tests/unit" TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_grouped_gemm_padded_features.py::test_selected_sources_reach_real_preflights_loader_and_both_trainers_before_old_helpers tests/unit/test_moe_grouped_gemm_padded_features.py::test_independent_selected_actual_fit_cache_binds_native_context --basetemp=/data/ycfeng/tmp/d29_group_reuse_trainers_parent01 --junitxml=/data/ycfeng/tmp/d29_group_reuse_trainers_parent01.xml > /data/ycfeng/tmp/d29_group_reuse_trainers_parent01.log 2>&1
```

Expected:both existing actual trainer/preflight and independent fit/cache checks
pass,without mock writer substitution. Observed:2PASS,0FAIL/ERROR/SKIP,10.327s.
XML SHA256:6b411b2457883e46b96c136ffa5b43f420fcac465c8481bbe2ec798f3186e5d9.
Keep this selection separate;do not add2 to the shared-suite81PASS.

## Actual pricing, without diagnostic group reuse

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONPATH="$PWD:$PWD/tests/unit" TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_group_reuse_parent_loop.py \
  --output /data/ycfeng/tmp/d29_group_reuse_parent_pricing01 --iterations 2 --execution --no-profile \
  > /data/ycfeng/tmp/d29_group_reuse_parent_pricing01.log 2>&1
```

The launcher validates the audited owner probe hash, clean source equality and
exact records-only HEAD, then changes only its two source/receipt revision
literals. It does not alter a method, point, cache or input. Owner probe:
76cfd438baf402dc6d2b2e75deae76c3905b8bea0be1bfea87a30a13981f08cf; launcher:
75b4e6c9973936f229c8060da4ca77b4926db3c6cba9d2708b6dbb61af5a5151;
executed probe:dd6013ce5ea31bf0d88d048fc0be039b60ee006f09d97ec3e6f7a756b97099f6.
The original30GiB probe bound remains.

Expected:cold88 descriptions,warm0,original cost/proof/full-plan-key payload.
Observed:0.924303847/0.354695124s,88/0 descriptions,exit0 and reuse_groups=false.
Payload SHA256:d2c141f088b52063edb539094fe3e0967f35e53303d70ebe2e8fbdbe4c86c952,
equal to the parent pre-fix baseline. PASS for the original P6128+idle1 point.
Result JSON SHA256:ddbfba8b77c8ddc9e912d649ff58e518488aa921b577300e004af04f6811bdc8.

## Scope and limits

Parent git numstat confirms65 additions/6 deletions=71 gross,source17/test54.
Retained paths143/175,111/128,254/260; candidate CORE3932/4404 and global9175/10000.
Step105 and all path ceilings fit. Independent Standards and Spec are separately
closed No findings,0 findings/worst none. The Spec records P2 is corrected by
37c7832e and closed by the same reviewer;no source/test/script change or new test
run accompanies that correction. Primary explicitly accepts this CPU component.

Archive task D29-PERFORMANCE-COMPONENT-ARCHIVE reread/rehashed both existing XMLs;
counts,times and hashes match the values above. It ran no tests or pricing probes.
No combined integration/final source freeze,whole6144/1024 CLI,fresh strict77/
full-unit rerun or T2/native/GPU/G3/E2E qualification follows from these selected
tests or this one-point measurement. Combined-candidate accounting remains pending.

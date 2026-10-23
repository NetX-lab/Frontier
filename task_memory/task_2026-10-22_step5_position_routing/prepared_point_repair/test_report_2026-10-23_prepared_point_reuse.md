# D29 prepared point reuse verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual RED/GREEN,immutable candidate wave/matrix,complete captured comparison and exact scope. |

Task ID: **D29-PREPARED-POINT-REUSE**. Code commit `0be11762d0f83e9db77658a1447a672507166872`, base `caffccbb48011f0140359d391ad75b58906ff15b`, submodule `3dac53c544e546574cb6e8078cd14053d0916113`. Worktree `/data/ycfeng/Frontier/.worktrees/step5-d29-prepared-point`. Three source/test paths only; no donor,backend,trace,registry,fixture/cache,primary/frozen or network changes.

## Environment and exact commands

CPython3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python`. CPU only. `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; scratch/cache under `/data/ycfeng/tmp`. The new approved sudo scope works and is in every command below. Actual probes retain their internal RLIMIT_AS30GiB and outer60s bound. Tests use the specified60s focused or240s affected bound, not a complete CLI attempt.

RED (before source change; appended tests only):

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_prediction_measurement_bindings.py::test_prepared_points_follow_actual_model_rows_identity_and_fresh_queries tests/unit/test_moe_routing_runtime.py::test_position_runtime_binds_retained_source_and_full_plan_policy > /data/ycfeng/tmp/d29_prepared_point_red01.log 2>&1
```

RED:1 failed/2 passed in27.29s. Actual warm query decoded653 texts (`1310-657`), failing the preparation expectation, not a shallow memoization utility test. Final focused command is identical except log `d29_prepared_point_green05.log`:3 PASS. Initial GREEN setup failures are preserved in progress: unsupported total-parse threshold was replaced with actual source-text decoding assertion; paired identity mismatch was retained as a real negative and then restored before positive numerical lookup. No behavior was weakened.

Final six-file affected selection:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 240s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_prediction_measurement_bindings.py tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_moe_position_routing.py tests/unit/test_step5_measured_training.py tests/unit/test_step5_measured_ffn.py --junitxml=/data/ycfeng/tmp/d29_prepared_point_affected_final01.xml > /data/ycfeng/tmp/d29_prepared_point_affected_final01.log 2>&1
```

Observed82 PASS/1 existing torch SKIP/0 FAIL or ERROR,132.09s (XML132.087s). XML SHA `664d505c5451706695eba8e9326d4211bbaf6320fe6f500fdd38ce90d8ba4f52`. This selection demonstrates shared manager/binding/runtime coverage; it does not include the independent trainer.

Separate shared/independent trainer coverage, not summed with82:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_grouped_gemm_padded_features.py::test_selected_sources_reach_real_preflights_loader_and_both_trainers_before_old_helpers tests/unit/test_moe_grouped_gemm_padded_features.py::test_independent_selected_actual_fit_cache_binds_native_context --junitxml=/data/ycfeng/tmp/d29_prepared_point_trainers01.xml > /data/ycfeng/tmp/d29_prepared_point_trainers01.log 2>&1
```

Observed2 PASS/0 FAIL,ERROR,SKIP,6.55s (XML6.490s); SHA `a436f0eb905b621548434458f46cc891ec171dc35e81f2b8b3013e6b28de450e`. Existing current-repository CPU writer, no external writer override. This is both real trainer/preflight/fit/cache evidence, not a new independent public-consumer mode.

## Candidate-source full-wave and matrix commands

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_prepared_point_repair/candidate_wave.py --output /data/ycfeng/tmp/d29_prepared_point_repair/wave-final01 > /data/ycfeng/tmp/d29_prepared_point_repair/wave-final01.log 2>&1
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_prepared_point_repair/candidate_matrix.py --output /data/ycfeng/tmp/d29_prepared_point_repair/matrix-final01 > /data/ycfeng/tmp/d29_prepared_point_repair/matrix-final01.log 2>&1
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_prepared_point_repair/candidate_wave.py --output /data/ycfeng/tmp/d29_prepared_point_repair/wave-layer60-final01 --captured-source /data/ycfeng/tmp/d29_main08_residual_repro/parent-layer60-source.json > /data/ycfeng/tmp/d29_prepared_point_repair/wave-layer60-final01.log 2>&1
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_prepared_point_repair/candidate_wave.py --output /data/ycfeng/tmp/d29_prepared_point_repair/wave-profile-final01 --profile > /data/ycfeng/tmp/d29_prepared_point_repair/wave-profile-final01.log 2>&1
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_prepared_point_repair/candidate_compare.py --replay wave-final01 --layer 61 > /data/ycfeng/tmp/d29_prepared_point_repair/wave-final01-comparison.log 2>&1
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_prepared_point_repair/candidate_compare.py --replay wave-layer60-final01 --layer 60 > /data/ycfeng/tmp/d29_prepared_point_repair/wave-layer60-final01-comparison.log 2>&1
```

Fresh scratch outputs created exclusively; rerun with new names. Only source path/revision was rebound; optional layer60 uses parent’s already verified captured-source file. Counts are observational wrappers around actual private methods, not a preparation implementation. Exact source rebinding plus observational-count hunk is `probe_source_rebinding.diff`, SHA `10f0b5754230762eef782207022ac1d9be66f643224febfa4ecdb637d499becf`. Diagnostic variable is `none`, no scratch `PreparedSelection` is imported by the wave. Both calibration fixture overrides are removed by the unchanged actual-wave setup. All output/cache argv values relocate before CLI parse. Cache bytes/model hashes and actual source/native proofs are verified unchanged.

Matrix uses the original predictor fixture/paired-cache construction seam, not a complete public predictor constructor. It binds the frozen original method as differential truth and invokes the candidate production query separately; no scratch preparation implementation. Its measurement wrapper counts actual candidate state transitions. Selected numerical duration/count methods remain current candidate source for both sides. Matrix calibration fixtures are not relabeled as actual full-wave behavior. The actual wave independently covers the public stage/Cluster/backend pricing and removes those overrides.

## Results and passing conditions

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Actual binding/replacement/fresh-context tests | Warm structural work absent; replacements rebuild; numerical/count checks live |3 focused PASS; tuple/model/identity change redecodes frames; bad rows/text leave successful state unchanged |PASS |
| Current-text behavior | Original read/JSON errors; same content reused; changed text rebuilt |Both runtime cases verify malformed JSON,error/restore,buffer/context mismatches and whitespace-only successful rebuild |PASS |
| Candidate layer61 | Whole cost/proof/full plan key bytes equal;one row/key preparation |2.025761s cold/1.086332s warm;3258/1968 decodes,80 reads;16464 rows/168 keys once |PASS multiplicity/equality;timing observed |
| Candidate layer60 | Same actual complete208 pairs with current layer proofs |1.691007s cold/0.984837s warm;same multiplicities;208/208 captured pairs,0diff |PASS |
| Layer61 direct capture | Compared actual captured subset,not relabeled complete |165 captured/208 replay pairs,0diff |PASS subset |
| Multi-point selection |P6128/P16/D6147/D6148,layers3/88,EP0/7/8/15,current family/model |36 positive/24 negative records identical to original method |PASS;not36 pytest cases |
| Original payload61 |No cross-layer proof reuse |`b2593e723ee095488ff213e84ddcd6e85391c11f42e092a7a5ba8b77a38ffc3a` |PASS |
| Original payload60 |Retain own global-layer proof identity |`3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83` |PASS |
| Scope and source binding |Three paths only;159<=250;pin unchanged |Git+140/-19=159;AST,whitespace,197/323,299/299,75/78 all PASS |PASS |

No final owned failure. Timing varies: WIP candidate1.763/0.989s, committed layer612.026/1.086s, layer601.691/0.985s; parent prototype1.868/1.237s. The deterministic result is fixed preparation/decoding multiplicity and unchanged payload, not a fastest-run wall threshold.

## Residual profile and verification limits

Two-wave cProfile5.087s:query2.027s including first model-group preparation1.213s, backend1.380s,160 current-file reads1.383s,trace flush0.825s. Existing full proof/key and trace serialization still run. These cumulative categories overlap and cannot be summed as independent time. Read/trace investigation and repairs are other owners’ scope, untouched here.

Serving lifecycle: rows are tuples of mutable dicts, assigned after fitting before shared manager returns and P/D predictors exist. Current estimator/tuple/string replacement rebuilds; no supported serving path mutates nested row dicts in-place. This is not a universal Python frozen-object claim. No extra mutation fingerprint or result cache was added.

Local GREEN establishes the allocated structural seam, not complete1024-token or original2400s full-workflow acceptance. Startup,whole regressions,strict77,native/GPU and E2E gates remain parent-owned. No full CLI,reduced-workload acceptance,new dependency,network or platform operation was run. `source_receipt.json` contains complete source/script/result SHA bindings;diff and exact budget are retained beside this report.

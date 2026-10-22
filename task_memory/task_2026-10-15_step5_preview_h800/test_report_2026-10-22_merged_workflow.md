# Merged Step workflow verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Close combined D26/D27/identity/D28 checks and preserve initial failures. |

## Source and environment

Merged source: `fa5bae2980366da1d075125d6c26248a737fc419`. The subsequent
change only corrects the pre-D27 wire-size expectation in
`tests/unit/test_step5_measured_pd.py:159`. D29/D30 are separate pending work.
Working directory: `/data/ycfeng/Frontier/.worktrees/step5-preview`.
Simulator Python3.10.6, NumPy2.2.6, pandas2.3.3, sklearn1.7.2.
Native-source/HTTP Python3.12.13, Torch2.9.1+cu128; CPU execution only.

## First merged invocation: FAIL retained

```bash
timeout 600s env PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_prefill_target.py tests/integration/test_step5_measured_main.py tests/unit/test_step5_measured_pd.py tests/unit/test_nixl_read_cost.py tests/unit/test_step5_structural_pd.py tests/unit/test_glm53_structural_pd.py tests/unit/test_step5_model_identity_reuse.py tests/unit/test_step5_measured_cost.py tests/unit/test_measured_attention_root_metrics.py tests/unit/test_step5_measured_public_predictor.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pressure_observation.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d26_d27_d28_merged01 --junitxml=/data/ycfeng/tmp/d26_d27_d28_merged01.xml
```

Observed239 PASS,3 FAIL,5 SKIP in234.57s, exit1.
Log `/data/ycfeng/tmp/d26_d27_d28_merged01.log`; XML above.

1. Measured-PD expected103051776 bytes, observed102203904. Its4080-token
   prefix has255 pages. Independent arithmetic:
   `69*32*12288 + 23*255*(12288+512) = 102203904`.
   The old assertion counted33 SWA pages;847872 bytes is the approved D27
   reduction. Target allocation1020 and all lifetime assertions are preserved.
2. Two HTTP tests failed with `ModuleNotFoundError: No module named 'aiohttp'`.
   Reuse the existing complete HTTP environment below; no package installation.
3. Five native fault-injection checks skipped for absent
   `STEP5_NATIVE_SOURCE_ROOT`; supply the existing locked path below.

## Affected reruns: PASS

```bash
timeout 240s env PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_measured_pd.py tests/unit/test_step5_prefill_target.py tests/unit/test_step5_structural_pd.py tests/unit/test_nixl_read_cost.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d26_d27_affected02 --junitxml=/data/ycfeng/tmp/d26_d27_affected02.xml

timeout 60s env TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_pressure_observation.py tests/unit/test_step5_native_pd_r2_stage.py tests/unit/test_step5_canonical_g3.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d28_merged02 --junitxml=/data/ycfeng/tmp/d28_merged02.xml
```

Expected no failed/skipped affected cases. Observed95 PASS in170.57s and81
PASS in3.50s, both exit0. Logs share the XML basename with `.log`.
The native suite adds existing stage cases. Latest JUnit outcomes keyed by
`(classname,name)` across the three invocations: **258 distinct PASS,
0 FAIL,0 SKIP**. This reconciles separate environments, rather than one run.

## Legacy fidelity and limits

Cleanfa5bae2 vs previously proven7e8bea3:77/77 identical,519 equal cache names,
902/902 artifact size/SHA checks. Report:
`../task_2026-10-22_step5_workflow_fidelity/test_report_2026-10-22_merged_clean77.md`.
The expectation-only correction changes no production or matrix input.

Merged source/legacy verification is complete. Full6144/1024 CLI, new routing/
finish-sync inputs, native graph/routing/pressure/state evidence, profile
admission and calibration remain open. No GPU or native CPU-overhead
measurement was performed.

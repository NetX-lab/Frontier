# D29 Final Ledger Frozen Regression Report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 UTC / 2026-10-25 HKT | Record original final146 suites, exact baseline comparison and retained archival evidence. |

## Result and binding

PASS for the mandatory source-regression gate only. Final source1461573de388b72450189bc0c1b88b4ac32491f0 and tree447a285aa403af747f838a3bf404b2b17e2eacf7 are clean and unchanged before/after every suite. Native checkout is3dac53c544e546574cb6e8078cd14053d0916113; no native binary is built or projected.

| Suite | Expected condition | Observed result | Raw exit | Verdict |
| --- | --- | --- | --- | --- |
| strict77 | All77 cases identical | 77/77 identical | run0,compare0 | PASS |
| Original strict auditor | Source/cache/environment and all artifacts match | 451 artifacts,519 cache names on each side | 0 | PASS |
| Canonical full-unit | All8620 IDs/statuses andFAIL/ERROR/SKIP sets equal accepted native-key baseline | 8028PASS/279FAIL/32ERROR/281SKIP;exact equality | 1 | PASS for baseline preservation;raw failures retained |
| supplement101 | Exact IDs/statuses/environment | 101PASS | 0 | PASS |
| metadata196 recipe | Exact IDs/statuses/environment | actual197PASS | 0 | PASS |
| catalogue53 | Exact IDs/statuses/environment | 53PASS | 0 | PASS |
| governance52 | Exact IDs/statuses/environment | 52PASS | 0 | PASS |

Do not sum overlapping suite counts. Full-unit raw pytest elapsed1273.17s; supervised subprocess duration1275.156656945s. Strict run/compare durations473.568930037/3.092091786s. The four supplementary durations are11.316076026/9.419466728/3.725470888/28.749895599s.

## Commands and environment

Original suite launcher: `/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/run_suite.py <suite>`, with the six exact suite labels above. The retained per-suite `execution.json` stores the full systemd/bwrap command, immutable source mounts, generated paths, timeout, environment, Python executable, dependency versions and actual exit. Below are the exact workload commands inside those recorded namespaces; none are rerun by this report.

### strict77

```bash
timeout 1800s /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/d29-ledger-final-1f6b4314 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --label candidate --output-root /data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/strict77_output --jobs 4 --clean-cache --continue-on-failure
```

```bash
timeout 180s /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/strict77_output --baseline-label /data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/frozen-matrix/candidate --candidate-label /data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/strict77_output/candidate
```

### unit

```bash
timeout 1800s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit -q -p no:cacheprovider --continue-on-collection-errors --basetemp=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/unit/results --junitxml=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/unit/results.xml
```

### supplement101

```bash
timeout 180s /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_position_cyclic.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/supplement101/basetemp --junitxml=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/supplement101/results.xml
```

### metadata196

```bash
timeout 180s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_comm_operator_families.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_quantization_manager_registry.py tests/unit/test_quantization_module_precision.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_trace.py tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/metadata196/basetemp --junitxml=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/metadata196/results.xml
```

### catalogue53

```bash
timeout 120s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_moe_share_expert_operator_families.py tests/unit/test_dsv4_precision_metadata.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/catalogue53/basetemp --junitxml=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/catalogue53/results.xml
```

### governance52

```bash
timeout 120s /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_profiling_governance_minimal_red.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/governance52/basetemp --junitxml=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/governance52/results.xml
```

Canonical interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6. Supplement101/governance use the retained Python3.12 environment recorded by their receipts. OMP/BLAS/MKL/NumExpr threads are1; each suite is configured with MemoryMax30GiB. This is not a measured cgroup RSS peak. No Conda environment is selected by these commands.

Canonical package versions:
```json
{
  "python": "3.10.6",
  "executable": "/data/ycfeng/envs/frontier-py310/bin/python",
  "pytest": "9.1.1",
  "numpy": "2.2.6",
  "pandas": "2.3.3",
  "scipy": "1.15.3",
  "scikit-learn": "1.7.2",
  "plotly": "7.1.0",
  "fasteners": "0.20",
  "ddsketch": "3.0.1",
  "torch": null,
  "aiohttp": null
}
```

## Comparison, failures and archive

Canonical baseline XML: `/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/frozen-unit/results.xml`, source376983d2ecc441d1e0a8cb8b943e65a2dcb217b5,SHA256bb1703a9cb5ca0d28f4fbdb732b391c8f9d38717330e375d48e6e098fe8df344. Candidate XML SHA256968d068ff4635fa291473cd5a3b48d462066e2aadf38b9aca7576dfec95bee9c. Comparator returns0 in0.177777176s and preserves all8620 IDs/statuses and versions. Final comparator SHA2569dfc87e46c242ff64ab7705eb7b8fb87712c06823699ee91b51b6acfb8aade38.

The first comparator exits1 because baseline metadata uses python_versions while candidate metadata uses versions. The actual receipt is inspected and only the reader location changes; equality is unchanged. Initial script/error log remain retained; first failure timing was not independently captured. No suite is rerun. The previously approved old hot-update ID replacement still has raw presenceFAIL versus the historical79 baseline and is not waived.

Archive command:
```bash
timeout 60s sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/archive_frozen_regressions.py
```

Archive wrapper is `/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/regressions/run_archive.py`. It exits0 in0.202845174s. All64 selected files/7126759B have exact byte/hash copies; archive execution/log are also retained. `archive_receipt.json` lists every original/archive path and digest. Full generated caches and outputs remain at their cited scratch locations.

## Verification limits

The candidate is still UNADMITTED at this pre-merge record. Production integration and the parent scope/source/binding guard follow this archive. No fresh wholeCLI, cohesive timing, worker/cloud append throughput, measured full-workload RSS, native timing, finalT2/package,G3 or E2E<=10% result is asserted. Required-shape6/6 and all source/owner/global ceilings remain unchanged.

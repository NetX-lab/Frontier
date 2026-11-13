# Fresh performance-source short regression selections

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record separate101/197/53 results from fresh XML and metadata business artifacts. |

TaskID D29-PERFORMANCE-FINAL-REGRESSIONS. All results freshly run on
79f1a3fd35d42b751cd1520c764ddac717b923d9, tree
8c6db078e852d3b885a6f14bdf5e8007d9ecddb7, submodule
3dac53c544e546574cb6e8078cd14053d0916113. Source clean before/after each suite.
Cwd `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06`; immutable
entries mounted read-only, original paths retained and generated cache isolated
under `/data/ycfeng/tmp/d29_performance_regressions_final05/<selection>/generated_repo_root`.

## Common execution and environment

Exact orchestration command for each selection:
```bash
python3 /data/ycfeng/tmp/d29_performance_regressions_final05/run_suite.py supplement101
python3 /data/ycfeng/tmp/d29_performance_regressions_final05/run_suite.py metadata196
python3 /data/ycfeng/tmp/d29_performance_regressions_final05/run_suite.py catalogue53
```

Each expanded bubblewrap+pytest argv, versions, environment and raw exit is in
`/data/ycfeng/tmp/d29_performance_regressions_final05/<selection>/execution.json`.
Payload is `timeout <cap>s <python> -B -m pytest <files> -q -p no:cacheprovider
--basetemp=<scratch>/<selection>/basetemp --junitxml=<scratch>/<selection>/results.xml`.
No exclusion, dependency install or extra import root. Common environment:
PYTHONPATH=repo:repo/tests/unit, PYTHONDONTWRITEBYTECODE=1,
TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp, OMP/OPENBLAS/MKL threads1;
FRONTIER_TEST_CPU_WRITER_FILE/SHA256 unset.

## Independent results, not summed

| Selection | Expected | Actual PASS/FAIL/ERROR/SKIP | JUnit seconds | Raw exit | Gate |
| --- | --- | --- | ---: | ---: | --- |
| Required same-source supplement101 | Same101 IDs,all pass | 101/0/0/0 | 9.085 | 0 | PASS |
| Original metadata196 file selection | Old196 IDs plus new affected cases pass | 197/0/0/0 | 6.346 | 0 | PASS |
| Required catalogue53 | Same53 IDs,all pass | 53/0/0/0 | 3.258 | 0 | PASS |

Counts are parsed from actual fresh XML, not retained expectations.
No old ID loss or PASS-to-SKIP. The metadata selection contains the new passing
`tests.unit.test_collective_sim_cc_backend::test_collective_artifact_reuses_only_identical_current_text`;
its historical selection name is not a claim of196 current cases.

### supplement101

Python3.12.13 `/data/ycfeng/envs/asyncopd-py312/bin/python`, pytest9.0.2,
Torch2.9.1, aiohttp3.13.3, numpy2.2.6, pandas3.0.1, scipy1.17.1. Existing
environment only; no CPU extras path needed for these modules.
STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm;
VLLM_TARGET_DEVICE=cpu. Timeout180s. Files:
`tests/unit/test_step5_canonical_g3.py`,
`tests/unit/test_step5_native_position_cyclic.py`.

All101 IDs exactly match the old supplement. Later full-unit reconciliation must
join its actual dependency exceptions to the12 identical HTTP IDs and16 native
module cases; this report does not pre-accept unfinished full-unit results.

### metadata selection, actual197

Python3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`, pytest9.1.1,
numpy2.2.6, pandas2.3.3, scipy1.15.3, sklearn1.7.2, plotly7.1.0,
fasteners0.20, ddsketch3.0.1; Torch/aiohttp absent. Timeout180s. Files:
`test_comm_operator_families.py`, `test_ep_wave_trace_context.py`,
`test_quantization_manager_registry.py`, `test_quantization_module_precision.py`,
`test_op_trace_utils.py`, `test_ep_trace.py`, `test_collective_sim_cc_backend.py`,
all under `tests/unit/`.

Actual generated CSVs separately inspected against retained e41 hashes/rows:
legacy two files byte-equal original baseline, no prepare row; matching Step
prepare FP8/profiling/empty factor; mismatching Step prepare BF16/approximation2.0;
grouped-GEMM FP8/approximation0.5 remains. Every CSV equals prior e41 bytes.
Exact paths/hashes/rows: scratch `metadata_business_receipt.json`.

### catalogue53

Same Python3.10 environment,timeout120s. Files:
`tests/unit/test_moe_share_expert_operator_families.py`,
`tests/unit/test_dsv4_precision_metadata.py`. Same53 old IDs,all pass.

## Limits

Raw XML/logs remain in each new scratch selection directory. Source-stamped
execution receipts and artifact hashes will be retained in this documentation
directory when all required comparisons finish. No totals are combined with
full-unit or other overlapping selections. main08 and native/GPU/numerical
qualification remain primary-owned and separate.

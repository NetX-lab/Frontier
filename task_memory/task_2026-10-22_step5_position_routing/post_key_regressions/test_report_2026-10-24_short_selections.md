# Post-key frozen-source short regressions

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record independent101/197/53 raw results and exact retained79 IDs. |
| 2026-10-24 | Correct historical binary availability;none of the selected files depends on that binary. |

TaskID D29-POST-KEY-FROZEN-REGRESSIONS. Source
46fe83ac55876c10f6dd8ff792705e87efd1905f/tree
bbbca98802261662e02d32a59c1a4d6ab20a12e5/submodule
3dac53c544e546574cb6e8078cd14053d0916113;all clean before/after.
Cwd `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-07`.

## Exact commands and environments

```bash
python3 -B /data/ycfeng/tmp/d29_post_key_regressions_20261024/run_suite.py supplement101
python3 -B /data/ycfeng/tmp/d29_post_key_regressions_20261024/run_suite.py metadata196
python3 -B /data/ycfeng/tmp/d29_post_key_regressions_20261024/run_suite.py catalogue53
```

Payloads use the exact prior file selections and `timeout <180/180/120>s <python>
-B -m pytest <files> -q -p no:cacheprovider --basetemp=<scratch>/basetemp
--junitxml=<scratch>/results.xml`. Expanded argv/env/versions/raw exits are in each
new `<suite>/execution.json`. All tests run under `sudo -n systemd-run --scope
--quiet -p MemoryMax=30G runuser -u brainpp -- env ... bwrap ...`. Existing source
entries are read-only;cache/output roots are unique scratch. The first namespace
projected a currently available3dac htsim binary,without building or writing the
frozen source. Retained79 XML later proved this was not historical availability.
None of the selected files checks that binary;these short results remain valid
and are not rerun. Full-unit environment is corrected separately. No
production/test/config/helper source edits.

Common: PYTHONPATH=repo:repo/tests/unit,PYTHONDONTWRITEBYTECODE=1,
TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp,OMP/OPENBLAS/MKL threads1;
FRONTIER_TEST_CPU_WRITER_FILE/SHA256 unset.
Metadata/catalogue: frontier-py310 Python3.10.6,pytest9.1.1,numpy2.2.6,
pandas2.3.3,scipy1.15.3,sklearn1.7.2,plotly7.1.0,fasteners0.20,ddsketch3.0.1;
Torch/aiohttp absent. supplement: asyncopd-py312 Python3.12.13,pytest9.0.2,
Torch2.9.1,aiohttp3.13.3,numpy2.2.6,pandas3.0.1,scipy1.17.1;
STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm,
VLLM_TARGET_DEVICE=cpu. No extras path or dependency installation for these suites.

## Actual results, never summed

| Selection | Actual PASS/FAIL/ERROR/SKIP | Raw exit | Retained79 ID comparison |
| --- | --- | --- | --- |
| supplement101 | 101/0/0/0 | 0 | Same101 IDs/statuses |
| original metadata selection | 197/0/0/0 | 0 | 196 IDs retained;1 removed/1 new |
| catalogue53 | 53/0/0/0 | 0 | Same53 IDs/statuses |

All fresh cases pass. No retained79 status changes orPASS-to-SKIP. The metadata
selection's exact removed PASS ID is
`tests.unit.test_collective_sim_cc_backend::test_collective_artifact_reuses_only_identical_current_text`.
The new passing ID is
`tests.unit.test_collective_sim_cc_backend::test_collective_artifact_is_fixed_and_failed_initial_reads_are_not_published`.
The approved source replaces per-query hot-update behavior with run-fixed input
and retryable failed initial loading,as explicitly decided in root requirements
lines262–302. This is recorded as raw ID loss/replacement,not an alias,skip or
claim that the old ID was executed. Raw exact-ID preservation is not all-green.

Detailed old/new XML hashes,all ID deltas and JUnit attributes:
`/data/ycfeng/tmp/d29_post_key_regressions_20261024/short_selections_comparison.json`.
Raw XML/logs stay in that root under each selection. Full unit,strict77 and
possible governance supplement remain independent and unfinished at this entry.
These are observed regressions only,not full CLI/native/GPU qualification.

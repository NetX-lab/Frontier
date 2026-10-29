# D30 focused finish synchronization test report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record approved seven-file consumer/lifecycle RED-to-GREEN and complete affected validation. |
| 2026-10-22 | Record RED/GREEN, source checks and unclosed adjacent/native validation. |

## Environment

Worktree `/data/ycfeng/Frontier/.worktrees/step5-finish-sync`, base `fa5bae2`.
CPU `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6, pytest9.1.1.
R2 `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python3.12.13,
pytest9.0.2, aiohttp3.13.3. These are existing virtual environments, not a new
conda environment. No package installation or host configuration change.
All commands set `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp`.
`STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src`
is also set for the final focused and R2 source-applicability commands.

## Historical candidate final commands

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_finish_sync.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_finish_sync_final04 --junitxml=/data/ycfeng/tmp/d30_finish_sync_final04.xml
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_pd_r2_stage.py tests/unit/test_step5_native_forward_census.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_finish_observer_adjacent03 --junitxml=/data/ycfeng/tmp/d30_finish_observer_adjacent03.xml
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/integration/test_vllm_dp_dummy_pass_runtime.py -k 'after_32_iterations or attention_dp_one' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_finish_legacy_runtime01 --junitxml=/data/ycfeng/tmp/d30_finish_legacy_runtime01.xml
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_measured_pd.py tests/unit/test_step5_structural_pd.py tests/unit/test_step5_pd_placement.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_finish_pd_adjacent01 --junitxml=/data/ycfeng/tmp/d30_finish_pd_adjacent01.xml
git diff --check
sha256sum /data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm/config/parallel.py /data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm/v1/engine/core.py
```

Stdout/stderr is retained at each matching `/data/ycfeng/tmp/<name>.log`.

## Historical candidate expectations and observed results

| Check | Passing condition | Actual | Result |
| --- | --- | --- | --- |
| Scheduler RED | Old policy misses step1/16; reset/new-wave distinguishes step32 |9FAIL/7PASS,65.18s | Expected RED |
| Initial idle RED | P/D issues a second dummy instead of waiting at1 |2FAIL,9.25s | Expected RED |
| Observer RED | Missing counter records detected |5FAIL/1PASS | Expected RED |
| Final focused | P/D1/16/32 boundaries, active peer, dummy, reset, non-boundary, legacy32, DP1, transparent observer, pinned native method |27PASS,65.58s | PASS |
| Final R2 adjacent | Existing install/pin/error/transfer/runtime tests unchanged |128PASS,1.98s | PASS |
| Legacy runtime | Unaffected family release at32, no DP1 dummy; request/token/lane conservation |3PASS,16.10s | PASS |
| Extra P/D adjacent | Existing assertions and requests complete |68PASS/6FAIL,137.50s | FAIL; parent scope disposition required |
| Same P/D with baseline scheduler | Identify prior failures without overwriting disk source |73PASS/1FAIL,137.92s | Prior READ-byte failure; five new direct-lane fixtures |
| Whitespace/sizes | No diff whitespace error; <=5 source/test files, <=500 changed lines, modules<=2000 |4files/327lines; scheduler1811, observer1258, focused261, existing observer tests1364 | PASS |

The baseline command used the same pytest arguments with `baseline01` paths
and a `MetaPathFinder`/loader that reads only
`git show fa5bae2980366da1d075125d6c26248a737fc419:frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`
and executes those bytes for that module; all other code stays current. It
establishes source-test attribution, not a full baseline simulation.

Native parallel.py SHA256:
`4b5f3d5938cdb4644367a7c4c33cc657f8834d83a69f8a838ef4168627e12c21`.
Native core.py SHA256:
`5a0344942c5af99cd85eb68d9eded7f856df40ff0986bda8de6d2ff22f6d357e`.
Both match parent analysis; native files remain unchanged.

## Historical failures addressed or deferred

Initial full RED hit a60s test timeout; rerun with180s completed, no fix involved.
DP1 fixture TP8 failed native memory planning; direct legal DP1/TP16 fixture
passes without relaxing memory or expert-divisibility guards. R2 first used
Python3.10 without aiohttp; existing Python3.12 rerun passes128 tests.
At the original candidate handoff, five newly failing adjacent direct-lane
fixtures and one existing READ-byte mismatch remained open. The parent fixed
READ at d581625; the authorized follow-up resolves the six full lifecycle
instances revealed after that fix, as recorded below.

## D30-SCOPE-7-750 environment and commands

Baseline `b8be2871f5f8e2ea762b7471599240dcfff5fa97` was merged without reset or
rebase as `739f15c`. Only three task documents conflicted; approval, independent
dependency and child evidence were retained. Parent READ102203904/1020-block
assertions are unchanged. Existing Python/pytest/aiohttp versions above were
rechecked with imports; no environment modification or package install occurred.

Each pytest command below redirects stdout/stderr to the matching basetemp name
plus `.log`; JUnit evidence is the specified `.xml` under `/data/ycfeng/tmp`.

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp
export STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src
# Expected lifecycle RED on b8be287 + the original D30 candidate.
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_pd_placement.py tests/unit/test_step5_measured_pd.py -k 'p_client_finishes or selected_d_lane or cost_shape or covered_read_cost or capacity_retry_rejects' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_scope_lifecycle_red01 --junitxml=/data/ycfeng/tmp/d30_scope_lifecycle_red01.xml
# Actual observer emitter JSONL -> load/judge RED, followed by the identical GREEN selection.
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_finish_sync.py -k emitted_native_counters -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_scope_metadata_red01 --junitxml=/data/ycfeng/tmp/d30_scope_metadata_red01.xml
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_finish_sync.py -k emitted_native_counters -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_scope_metadata_green01 --junitxml=/data/ycfeng/tmp/d30_scope_metadata_green01.xml
# Complete affected PD modules, not strict77 or a performance suite.
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_pd_placement.py tests/unit/test_step5_measured_pd.py tests/unit/test_step5_structural_pd.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_scope_pd_final01 --junitxml=/data/ycfeng/tmp/d30_scope_pd_final01.xml
# Focused Step boundaries/emitter, legacy32 and DP1 runtime checks.
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_finish_sync.py tests/integration/test_vllm_dp_dummy_pass_runtime.py -k 'not test_vllm_dp_dummy_pass_runtime or after_32_iterations or attention_dp_one' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_scope_boundaries_final01 --junitxml=/data/ycfeng/tmp/d30_scope_boundaries_final01.xml
# Existing R2 install/pin/error, stage, census and verdict checks in the aiohttp environment.
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_pd_r2_stage.py tests/unit/test_step5_native_forward_census.py tests/unit/test_step5_native_pd_r2_verdict.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_scope_r2_final01 --junitxml=/data/ycfeng/tmp/d30_scope_r2_final01.xml
# Recheck the consumer after counter event classification is centralized.
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_pd_r2_verdict.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_scope_verdict_final02 --junitxml=/data/ycfeng/tmp/d30_scope_verdict_final02.xml
# Recheck exact peer-release identities after the final fixture-only refinement.
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_pd_placement.py tests/unit/test_step5_measured_pd.py -k 'p_client_finishes or selected_d_lane or cost_shape or covered_read_cost or capacity_retry_rejects' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_scope_lifecycle_final02 --junitxml=/data/ycfeng/tmp/d30_scope_lifecycle_final02.xml
git diff --check
git diff --numstat b8be287 -- frontier tests
git diff --unified=0 b8be287 -- frontier tests > task_memory/task_2026-10-22_step5_finish_sync/d30_source_test.diff
wc -l frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py tests/integration/step5_pd/r2_observation.py tests/unit/test_step5_finish_sync.py tests/unit/test_step5_native_pd_r2_observation.py tests/integration/step5_pd/r2_verdict.py tests/unit/test_step5_pd_placement.py tests/unit/test_step5_measured_pd.py
sha256sum /data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm/config/parallel.py /data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm/v1/engine/core.py
```

## D30-SCOPE-7-750 expectations and observations

| Check | Passing condition | Observed result | Status |
| --- | --- | --- | --- |
| Lifecycle RED | All six direct-lane fixtures fail, unchanged retry-fault branch passes |6FAIL/1PASS45.78s | Expected RED |
| Emitter/consumer RED | Both P/D counters rejected before consumer repair |2FAIL2.56s; Unknown event | Expected RED |
| Emitter/consumer GREEN | Raw+location retained outside request records, unchanged valid verdict, metadata-only UNVERIFIED, unknown/error rejection retained |2PASS2.74s; included in final32 | PASS |
| Placement lifecycle | Real events complete P response/source retention and D retry/readiness/output/reset |2PASS80.09s | PASS |
| Complete affected PD | All six lifecycle instances plus existing ownership/READ/retry faults pass |74PASS297.53s | PASS |
| Step/observer/legacy boundaries | Step1/16/reset/active-peer/dummy and scalar call/return/error transparency; legacy32/DP1 unchanged |32PASS83.14s | PASS |
| R2 affected | Observer/stage/census/verdict including newly admitted metadata parametrizations |388PASS10.06s | PASS |
| Verdict refinement | Existing error and qualification behavior unchanged |260PASS8.40s, overlapping388 | PASS |
| Exact release peers | Config-derived full peer set observed; all queued events dispatched and lane reset checked |7PASS203.78s; overlapping PD74 | PASS |

Distinct full-matrix passing cases are `74 + 32 + 388 = 494`; overlapping
iterations and verdict reruns are not additional coverage. `test_results.json`
contains observed JUnit counts and exact six repair IDs plus unchanged fault ID.
These timings are pytest wall durations only, not performance measurements.

Fixture iterations first exposed missing typed attention identity, a discarded
retry output, an incorrect import, and missing structured EP operator timing.
Corrections reuse registered `bind_layer_attention`, retain actual returned
events and supply explicit existing execution fields; no scheduler, routing,
READ price, native behavior or error suppression was introduced.

The changed-file set must equal the seven receipt paths, sum all additions and
deletions against b8be287 including cf1a53f, and keep every module<=2000 lines.
Machine-readable evidence is `source_test_budget.json`; the complete source/test
diff is `d30_source_test.diff` (all changed hunks, zero context so blank context
lines do not cause artifact whitespace errors). Native SHA256 values above are unchanged.

| Approved source/test file | Insertions | Deletions | Total changed | Module lines |
| --- | --- | --- | --- | --- |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` |17 |7 |24 |1811 |
| `tests/integration/step5_pd/r2_observation.py` |39 |0 |39 |1258 |
| `tests/integration/step5_pd/r2_verdict.py` |6 |1 |7 |1478 |
| `tests/unit/test_step5_finish_sync.py` |305 |0 |305 |305 |
| `tests/unit/test_step5_measured_pd.py` |42 |47 |89 |416 |
| `tests/unit/test_step5_native_pd_r2_observation.py` |2 |1 |3 |1364 |
| `tests/unit/test_step5_pd_placement.py` |117 |16 |133 |306 |
| **Total** |**528** |**72** |**600** |**All<=2000** |

PASS: exact receipt path set, `528 + 72 = 600 <= 750`, AST parse all seven files,
module-size and `git diff --check`. Original candidate327 + follow-up273 =600
gross lines; parent READ hunk is unchanged and excluded from this D30 delta.

## Verification limits

The AST-extracted pinned native method preserves its original branch/body and
observed bool return; a CPU substitute for its existing collective verifies
there are only the original calls at1/16/32. This is not a live collective or
GPU trajectory. New fields label `source_sync_boundary` as source-derived.
Live DP-rank/wave completeness, active peer, first idle boundary, reset and
observer overhead/perturbation remain UNVERIFIED. No full CLI, strict77, GPU run,
wall profiling or CPU-overhead suite was performed. Parent owns integration,
native stage/hash/package and all platform operations.

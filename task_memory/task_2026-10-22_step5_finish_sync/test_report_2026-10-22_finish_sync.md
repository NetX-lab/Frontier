# D30 focused finish synchronization test report

## Modification History
| Date | Change |
| --- | --- |
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

## Exact final commands

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_finish_sync.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_finish_sync_final04 --junitxml=/data/ycfeng/tmp/d30_finish_sync_final04.xml
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_pd_r2_stage.py tests/unit/test_step5_native_forward_census.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_finish_observer_adjacent03 --junitxml=/data/ycfeng/tmp/d30_finish_observer_adjacent03.xml
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/integration/test_vllm_dp_dummy_pass_runtime.py -k 'after_32_iterations or attention_dp_one' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_finish_legacy_runtime01 --junitxml=/data/ycfeng/tmp/d30_finish_legacy_runtime01.xml
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_measured_pd.py tests/unit/test_step5_structural_pd.py tests/unit/test_step5_pd_placement.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d30_finish_pd_adjacent01 --junitxml=/data/ycfeng/tmp/d30_finish_pd_adjacent01.xml
git diff --check
sha256sum /data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm/config/parallel.py /data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm/v1/engine/core.py
```

Stdout/stderr is retained at each matching `/data/ycfeng/tmp/<name>.log`.

## Expectations and observed results

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

## Failures addressed or deferred

Initial full RED hit a60s test timeout; rerun with180s completed, no fix involved.
DP1 fixture TP8 failed native memory planning; direct legal DP1/TP16 fixture
passes without relaxing memory or expert-divisibility guards. R2 first used
Python3.10 without aiohttp; existing Python3.12 rerun passes128 tests.
Five newly failing adjacent direct-lane fixtures and one existing READ-byte
mismatch remain explicitly open in `issues.md`; source edits stopped before
expanding the five-file bound.

## Verification limits

The AST-extracted pinned native method preserves its original branch/body and
observed bool return; a CPU substitute for its existing collective verifies
there are only the original calls at1/16/32. This is not a live collective or
GPU trajectory. New fields label `source_sync_boundary` as source-derived.
Live DP-rank/wave completeness, active peer, first idle boundary, reset and
observer overhead/perturbation remain UNVERIFIED. No full CLI, strict77, GPU run,
wall profiling or CPU-overhead suite was performed. Parent owns integration,
native stage/hash/package and all platform operations.

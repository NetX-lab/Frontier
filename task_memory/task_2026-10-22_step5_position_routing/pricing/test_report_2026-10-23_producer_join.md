# D29 producer join and portable CPU writer validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record real verifier two-producer handoff, references, quota and shared persistence boundary. |

TaskID D29-PRODUCER-JOIN-REPAIR; workdir `/data/ycfeng/Frontier/.worktrees/step5-position-pricing`.
Python frontier-py3103.10.6 and asyncopd-py3123.12.13; bytecode disabled/thread caps.
No GPU, source qualification, peer import/edit, platform operation or new code path.

## Test evidence and criteria

Actual NativeRoutedExperts.prepared_context and DispatchObserver.compute_context
run the unchanged real verify_step5_profile_runtime using installed_profile_receipt,
a sealed generated Python package with actual inspectable expert/padding callees.
One valid extra module is loaded only in the second producer. Stable point contexts
must equal, complete point_diagnostics must differ and retain this module, and
constructor provenance remains identical. Actual exporter.observed_collective_points
retains the point; canonical equality matches the independent prepared-context.
No mock verifier or dictionary-only handoff. Wrong callable source, expert bytes
and ambient bytes fail real verifier; wrong layout cannot match the point.
CPU-source tests require default current repo path or explicit file with SHA256.

RED join restores old full-import assignment temporarily, then restores fix:
5 JUnit=4 PASS/1 FAIL,0 SKIP/ERROR; legitimate two-producer identity mismatch.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_dispatch_probe.py::test_two_native_point_producers_join_with_different_valid_imports -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_join_red --junitxml=/data/ycfeng/tmp/d29_join_red.xml > /data/ycfeng/tmp/d29_join_red.log 2>&1
```

RED CPU reference temporarily restores old host path, then restores new resolver:
5 FAIL/0 PASS/SKIP/ERROR (default,missing/wrong hash,file,explicit exact).

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py::test_cpu_writer_reference_is_explicit_and_sealed -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_writer_reference_red --junitxml=/data/ycfeng/tmp/d29_writer_reference_red.xml > /data/ycfeng/tmp/d29_writer_reference_red.log 2>&1
```

Final consumer run:145=144 PASS/1 SKIP/0 FAIL/ERROR,4.830s. Explicit stage source
is CORE actual writer at d96b3279; file SHA256 verified before parsing/executing.
Current-repo default resolution is tested; execution without reference awaits
CORE's function integration (current branch has legacy writer, no silent skip).
Normal cold constructor/four phases/25 negatives/native injection/trace preserved.

```sh
env -u PYTHONPATH FRONTIER_TEST_CPU_WRITER_FILE=/data/ycfeng/Frontier/.worktrees/verify-step5-core-delivery-01/tests/integration/step5_measured_compute_fixture.py FRONTIER_TEST_CPU_WRITER_SHA256=87752f5816cc59baa8eb63733bfe9a4be7dde9b32d890ad5a0afbf3336f0d86b PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_step5_moe_pricing.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_join_consumer_final --junitxml=/data/ycfeng/tmp/d29_join_consumer_final.xml > /data/ycfeng/tmp/d29_join_consumer_final.log 2>&1
```

Final probe:250=171 PASS/79 SKIP/0 FAIL/ERROR,6.301s; includes5 real verifier cases.
Separate original published router execution64 PASS/0 SKIP/FAIL/ERROR,3.527s.
The64 are selected from probe skipped source cases, not an extra395 distinct cases.
Disjoint affected suites395=315 PASS/80 SKIP; no broad regression/GPU claim.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_dispatch_probe.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_join_probe_final --junitxml=/data/ycfeng/tmp/d29_join_probe_final.xml > /data/ycfeng/tmp/d29_join_probe_final.log 2>&1
env PYTHONPATH=$PWD STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_dispatch_probe.py -k actual_native_router_observer -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_join_native_final --junitxml=/data/ycfeng/tmp/d29_join_native_final.xml > /data/ycfeng/tmp/d29_join_native_final.log 2>&1
```

## Iterations and verification limits

Initial join5 PASS. Adding H04 into torch-only asyncopd env exposed missing
fasteners through eager execution_time_predictor imports (two errors); trying
entire py310 site-packages caused incompatible NumPy extension collection error.
This inspection expansion was removed, not masked by production fallback or env
edit. Final tests use real exporter handoff and independent existing H04 suite.
Attempt to collect both pricing/probe in asyncopd also failed missing fasteners;
final suites use their existing appropriate environments. Raw logs:
`d29_join_probe_green.log`, `d29_join_h04_green.log`, `d29_join_h04_green2.log` under
`/data/ycfeng/tmp`. They are failures, not passing evidence. After selector source
added, old stub expected only apply; fixed expected callable list, native source
execution remains a prior pricing test unless explicitly available in selected env.

Full ambient diagnostics are retained on owned native object/probe report; CORE
MoEWrapper._profile_position_cell must add exact3-gross hunk in interface_receipt
to retain per-point full inventory in row diagnostics/native_facts union. No claim
that shared writer is fixed here. No consumer context stripping. Wrong layout,
native origin/hash/callable and original routing/numerical criteria unchanged.

## Exact scope

Five existing owned source/test paths:+98/-8=106 revision churn. Baseline owned
43 paths:+3708/-97=3805/3956;remaining151. Isolated sub333 included/unchanged3dac53c;
root gitlink unstaged; no new paths/peer writes. AST/whitespace/sub clean PASS.

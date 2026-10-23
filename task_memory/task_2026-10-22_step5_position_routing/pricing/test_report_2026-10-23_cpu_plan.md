# D29 explicit CPU plan consumer validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual writer RED/GREEN, cold constructor, trace and exact quota. |

TaskID: D29-CPU-PLAN-CONSUMER. Parent implementation scope remains D29-PRICING-
IMPLEMENTATION. Workdir `/data/ycfeng/Frontier/.worktrees/step5-position-pricing`.
Python `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6,GCC11.3.0. Each command
below removes PYTHONPATH, disables bytecode, caps native threads and keeps test
temporary files/raw logs in `/data/ycfeng/tmp`. No submodule change, GPU or peer write.

## Actual writer and expectations

Tests compile/execute the retained `position_collective_costs` function from
`/data/ycfeng/Frontier/.worktrees/verify-step5-core-delivery-01/tests/integration/step5_measured_compute_fixture.py`
read-only (SHA25687752f5816cc59baa8eb63733bfe9a4be7dde9b32d890ad5a0afbf3336f0d86b).
Synthetic table rows provide CPU source/native tensors; the actual CORE writer
produces all four phases and existing artifact/count/cost keys. No writer-copy,
handwritten-only artifact or bootstrap mock. Constructor is real.

Expected CPU cost:0.03ms=0.01ms network+max(16x0.01ms local)+0.01ms remainder.
These are writer fixture values, not production defaults. Explicit CPU artifact
AND plan.shared, matching device/schema and unadmitted/candidate proof are required.
Exact entry/context/buffer/embedded count receipt and cost plan key must match.
Native cost injection999ms must be ignored by the unchanged native fit path.
Trace CPU label must be cpu_fixture_composition_not_native; native label unchanged.

## Exact commands and observed results

RED importer: four valid CPU phases fail at `artifact["qualification"]` as expected;
JUnit31 cases=27 PASS/4 FAIL/0 SKIP/ERROR. Existing negatives/injection pass.

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_cpu_plan_red --junitxml=/data/ycfeng/tmp/d29_cpu_plan_red.xml > /data/ycfeng/tmp/d29_cpu_plan_red.log 2>&1
```

Initial GREEN: same prefix, tests `tests/unit/test_collective_sim_cc_backend.py
tests/unit/test_op_trace_utils.py`, basetemp `/data/ycfeng/tmp/d29_cpu_plan_green`,
JUnit/log matching names:50 PASS/0 FAIL/SKIP/ERROR. Added explicit missing/relabelled
keys and existing real wave trace suite; final first run68 PASS in4.73s.

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py tests/unit/test_op_trace_utils.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_cpu_plan_green --junitxml=/data/ycfeng/tmp/d29_cpu_plan_green.xml > /data/ycfeng/tmp/d29_cpu_plan_green.log 2>&1
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_wave_trace_context.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_cpu_plan_final --junitxml=/data/ycfeng/tmp/d29_cpu_plan_final.xml > /data/ycfeng/tmp/d29_cpu_plan_final.log 2>&1
```

Trace RED: temporarily restore old hardcoded label in owned code only; CPU label
case FAIL/native label PASS (2 JUnit cases,0 SKIP/ERROR), then restore new expression.

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_op_trace_utils.py::test_collective_trace_source_uses_explicit_cost_provenance -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_cpu_trace_red --junitxml=/data/ycfeng/tmp/d29_cpu_trace_red.xml > /data/ycfeng/tmp/d29_cpu_trace_red.log 2>&1
```

Final GREEN:68 JUnit cases=68 PASS/0 FAIL/0 SKIP/0 ERROR (4.235s XML;4.23s console).
Includes25 CPU missing/admission/device/schema/key/context/buffer/receipt/cost
negative cases, four phases, original native/cold tests, injection and trace.

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_wave_trace_context.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_cpu_plan_final2 --junitxml=/data/ycfeng/tmp/d29_cpu_plan_final2.xml > /data/ycfeng/tmp/d29_cpu_plan_final2.log 2>&1
```

`git diff --check`:PASS. `git status --short` in isolated submodule:empty/PASS.
No broad regression rerun; historical producer350/80, native64 and HTSIM12 are
unchanged prior results, not new CPU-run counts or GPU qualification.

## Scope/budget and limits

Four existing owned paths delta+109/-2=111 versus120 reserve. Baseline owned
43 paths,+3617/-96=3713/3956;243 remain. Gross delta107 in final baseline stream
diff differs from revision-to-revision111 patch churn. Submodule333 included and
unchanged3dac53c. Full retained patch SHA256c96cace541b72f7627b5f429d55615a17be4cd6c86d1827ef53969d05d766a1d.
Root still owns gitlink/integrated actual-main, external review and native/GPU
qualification. Existing observed producer9dbca5b3 is complete; this isolated fix
does not introduce source observations, numerical calibration or native admission.

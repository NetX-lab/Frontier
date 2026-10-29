# Final-source affected metadata196

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record independent196PASS and actual CSV checks. |

TaskID: D29-E41-FINAL-REGRESSIONS. Frozen e41abc073500d61f224eda4fca1123e74e40544d,
submodule3dac53c544e546574cb6e8078cd14053d0916113; clean before/after.
Cwd: `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05`.
Python3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`, pytest9.1.1.
PYTHONPATH=repo:repo/tests/unit; PYTHONDONTWRITEBYTECODE=1;
TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp; OMP/OPENBLAS/MKL threads1;
FRONTIER_TEST_CPU_WRITER_FILE/SHA256 unset. Frozen files read-only through
bubblewrap; exact expanded mounts/environment/packages in evidence execution.json.

```bash
python3 /data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/run_suite.py metadata196
```

Executed payload:
```bash
timeout 180s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_comm_operator_families.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_quantization_manager_registry.py tests/unit/test_quantization_module_precision.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_trace.py tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/metadata196/basetemp --junitxml=/data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/metadata196/results.xml
```

Expected196PASS with no failures/errors/skips. Observed **196PASS/0FAIL/0ERROR/0SKIP**,
raw exit0, JUnit8.208s, outer8.792s: PASS.

Actual MetricsStore CSV rows separately inspected:

| Case | Expected and observed | Result |
| --- | --- | --- |
| Llama2, either signature | No prepare row; bytes equal retained baseline | PASS |
| Step, matching signature | FP8/profiling/empty approximation_factor | PASS |
| Step, mismatching signature | BF16/approximation/2.0 | PASS |
| Step grouped-GEMM | FP8/approximation/0.5 remains | PASS |

Retained baseline CSV:
`/data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate/cases/dp_dense_online_lanes2_replicas2/metrics/meta_llama_llama_2_7b_hf/online_serving/dp_dense_online_lanes2_replicas2/op_precision_metadata.csv`.
Raw suite/CSV receipt: scratch `/metadata196/`; retained evidence under
`evidence/metadata196/`. No numerical GPU fidelity claim. This suite stays separate;
its IDs will also be checked as passing subsets of full-unit parent04, not added
to that suite's totals.

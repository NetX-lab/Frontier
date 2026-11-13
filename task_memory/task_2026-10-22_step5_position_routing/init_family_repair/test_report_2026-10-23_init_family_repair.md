# D29 loader family correction test report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual-loader RED/GREEN, affected suites, environment limits and gross caps. |

TaskID: D29-INIT-FAMILY-RESOLUTION.
Worktree: `/data/ycfeng/Frontier/.worktrees/step5-d29-loader-family`.
Baseline: `24f970cbd6cae059467f68e0e1cf771b07aeecfe`.
Code: `ee7e66ee439054978897c91fa32598fc13804883`.
Scratch: `/data/ycfeng/tmp/d29_init_family_repair.ePNCE48J`.
Selected raw logs are retained in `evidence/` beside this report.

## Environment

Primary Python `/data/ycfeng/envs/frontier-py310/bin/python`, 3.10.6, pytest9.1.1,
pandas2.3.3, NumPy2.2.6, SciPy1.15.3, scikit-learn1.7.2, fasteners0.20. No install
or dependency/environment edits. All pytest processes have `timeout 60s` and
RLIMIT_AS30GiB. `systemd-run --user --scope -p MemoryMax=30G` could not move PIDs
into its cgroup (I/O error/permission denied); system scope needed interactive
authentication. Both failed before pytest. The existing parent RLIMIT_AS pattern
was used instead, without increasing time or memory limits. Bare `python` is absent.

The existing Python3.12.13 `asyncopd-py312` predictor supplement failed collection
for missing `fasteners`; it also lacks ddsketch/sklearn/plotly. The final governance
supplement appends existing same-ABI
`/data/ycfeng/envs/distillm-py310/lib/python3.10/site-packages` after the primary
environment in this test process only, supplying Torch2.1.2. One warning reports
Torch/NumPy `_ARRAY_API` initialization incompatibility; no conversion or GPU
qualification is claimed. No stub replaces Torch, validators or bindings.

## Deterministic RED/GREEN

One retained test calls the actual `_load_moe_df`, reuses `_position_source_rows`
and appends one valid row outside selected `moe_layer_ids`. Its actual wrapped
`validate_step5_config` must run exactly once per call. Entire selected frames must
equal the first two source rows, with the existing BF16->FP8 precision projection;
this checks typed JSON, row IDs, all metadata, duration and kernel_count columns.
Two successive calls must each derive once, not share a cache. An invalid
`tensor_parallel_mode` on `moe_gating_linear` in the third, unselected row must still
raise before filtering when `moe_input_quantize` is requested.

Exact RED command (new test, unchanged baseline source):

```bash
timeout 60s env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -c 'import resource, runpy; resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3)); runpy.run_module("pytest", run_name="__main__")' tests/unit/test_moe_grouped_gemm_padded_features.py::test_moe_loader_derives_family_once_without_skipping_unselected_rows -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/red-temp > /data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/red.log 2>&1
```

Observed exit1, `1 failed in 1.85s`, exactly `assert 9 == 1`. Complete selected
frame equality passed before this assertion. This is the intended RED, not an
environment/import failure. Three rows each declare three typed operators.

Exact GREEN command (four source lines added):

```bash
timeout 60s env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -c 'import resource, runpy; resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3)); runpy.run_module("pytest", run_name="__main__")' tests/unit/test_moe_grouped_gemm_padded_features.py::test_moe_loader_derives_family_once_without_skipping_unselected_rows -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/green-temp > /data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/green.log 2>&1
```

Observed exit0, `1 passed in 2.31s`: one derivation for each of two valid calls and
the corrupted-row call, entire selected frames equal, unselected row corruption
raises the expected enum error. PASS. Test timing includes instrumentation and is
not a performance comparison.

## Affected suites and baseline comparison

```bash
timeout 60s env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -c 'import resource, runpy; resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3)); runpy.run_module("pytest", run_name="__main__")' tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_moe_routing_runtime.py tests/unit/test_profiling_governance_minimal_red.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/affected-temp > /data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/affected.log 2>&1

timeout 60s env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -c 'import resource, runpy; resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3)); runpy.run_module("pytest", run_name="__main__")' tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_moe_routing_runtime.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/moe310-temp > /data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/moe310.log 2>&1

timeout 60s env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu /data/ycfeng/envs/asyncopd-py312/bin/python -B -c 'import resource, runpy; resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3)); runpy.run_module("pytest", run_name="__main__")' tests/unit/test_profiling_governance_minimal_red.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/governance312-temp > /data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/governance312.log 2>&1

timeout 60s env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu /data/ycfeng/envs/frontier-py310/bin/python -B -c 'import resource, runpy, sys; resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3)); sys.path.append("/data/ycfeng/envs/distillm-py310/lib/python3.10/site-packages"); runpy.run_module("pytest", run_name="__main__")' tests/unit/test_profiling_governance_minimal_red.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/governance-torch-temp > /data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/governance-torch.log 2>&1
```

Frozen baseline command uses cwd
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05`, `e41abc07`, and
the same Python3.10 configuration; it is read-only and creates only new scratch:

```bash
timeout 60s env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -c 'import resource, runpy; resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3)); runpy.run_module("pytest", run_name="__main__")' tests/unit/test_profiling_governance_minimal_red.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/governance-baseline-temp > /data/ycfeng/tmp/d29_init_family_repair.ePNCE48J/governance-baseline.log 2>&1
```

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Full MoE/routing files | All existing and new tests pass | 29PASS, exit0,33.18s | PASS |
| Combined primary environment | No new code failures | 76PASS/5FAIL,33.07s; five producer imports lack Torch | Raw FAIL; environment-limited |
| Frozen e41 governance primary environment | Same dependency failures | 47PASS/5FAIL,2.29s; exact same IDs and missing-Torch errors | PASS for failure-set comparison |
| Python3.12 governance | Dependencies available | Collection error: no fasteners,1.90s | FAIL; not a usable supplement |
| Full governance existing Torch supplement | All52 tests pass | 52PASS, exit0,14.44s; one ABI warning | PASS for CPU test coverage only |

The unchanged missing-Torch IDs are:
`test_linear_wrapper_result_carries_plan_typed_operator_contracts`,
`test_linear_wrapper_result_omits_unmeasured_typed_operator_contracts`,
`test_real_linear_producer_csv_round_trip_reaches_manager_loader`,
`test_moe_wrapper_typed_result_round_trips_through_manager`,
`test_moe_wrapper_initializes_runtime_with_routed_width`.
Prior negative tests remain unchanged; no deselection or skip marker was added.
Suite counts overlap and are not summed as independent evidence.

## AST, whitespace, gross and ownership

Checks executed `ast.parse` for both files, `git diff --check`,
`git diff --numstat 24f970cb -- <two owned files>`,
`git diff --numstat 1aee0af -- <two owned files>`, and
`git ls-tree HEAD frontier/cc_backend/backends/collective-sim`. Immutable diff:
`git diff 24f970cb ee7e66ee -- <two owned files>` in `code.diff`.

| Path | New gross / cap | Retained gross / cap | AST |
| --- | --- | --- | --- |
| frontier/execution_time_predictor/profiling_dataframe_loaders.py | 4/6 | 13/17 | PASS |
| tests/unit/test_moe_grouped_gemm_padded_features.py | 34/39 | 231/237 | PASS |

Total38/45 gross, all insertions. Whitespace PASS. Existing import, validator API,
all-row map, downstream row validation/filtering, datasets, peer code and gitlink
are unchanged. Root gitlink remains `3dac53c544e546574cb6e8078cd14053d0916113`.
Record markdown/JSON whitespace checks also pass. Raw pytest logs and `code.diff`
retain exact original bytes (including pytest traceback trailing spaces and diff
context prefixes); they are excluded from record whitespace checks, not rewritten.
`accounting.json` retains hashes and conservative aggregate bounds from the recorded
root delivery: CORE3873+38+105reserved=4016<=4404;
global9116+38+105reserved+50reserved=9309<=10000. No effective path addition (111)
or allocation amendment; final integrated aggregate reconciliation is root-owned.

## Evidence limits

The parent already measured the real3000-row typed-map control:
2.837165373s ->0.274640171s, one derivation and equal SHA256
`f460359008814440c80a40fd2e97b557e7bcbb802c72a4076c6ebe3c8a2cdd3f`.
This slice did not repeat input scans or timings. It supplies actual-loader
deterministic correctness/derivation evidence only. No complete CLI,
initialization, native/GPU, G3 or E2E qualification, and no claim to resolve the
unexplained initial startup interval. Final review/integration/freeze remain primary.

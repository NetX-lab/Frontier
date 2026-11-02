# G2 profiling driver CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded initial driver checks, expanded regression and the baseline-reproduced cold-start fixture repair; independent review remains open. |
| 2026-10-23 | Recorded three independent-review RED controls and repaired-source suites32/19/39/18PASS, separately. |
| 2026-10-23 | Completed final frozen-source verification at 05952396 and accepted the CPU component after both independent re-reviews. |

Current result: **ACCEPTED_CPU_PROFILE_DRIVER_COMPONENT**, source `05952396b0b8e5a794c53e10f9a21213beb5193d`; see `primary_acceptance.json`. The earlier iterations below retain their original outcomes.

## Scope and environment

Baseline: `313c8167`. Initial six-file source is bound by `source_manifest.json` and `change.diff`; the seventh test-only correction is described in `cold_start_scope.md`. No GPU/native runtime acceptance is claimed.

Interpreter: `/data/ycfeng/envs/asyncopd-py312/bin/python` (exact version in `cpu_checks.json`). Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

Explicit environment:

```text
PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site
PYTHONDONTWRITEBYTECODE=1
TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode
FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69
NATIVE_READER_BINDING_CHECKER=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py
```

Each suite was invoked separately with:

```text
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/<suite>.py -q -p no:cacheprovider --junitxml=<output>/<suite>.xml
```

Every actual argv is retained in `cpu_checks.json` and `regression_checks_initial.json`. Their original output roots are `/data/ycfeng/tmp/glm53_dsv4/g2_driver_cpu01` and `/data/ycfeng/tmp/glm53_dsv4/g2_driver_regression01`; each contains the corresponding stdout/stderr log and JUnit XML. Source bytes were hashed before and after execution.

## Results, reported separately

Passing condition: expected scenario assertions, return code0, no failures/errors/skips. Counts are not summed.

| Suite | Tests | Result |
| --- | ---: | --- |
| test_native_serve_profile | 32 | PASS |
| test_native_pd_worker | 17 | PASS |
| test_native_streaming_replay | 36 | PASS |
| test_native_serving_inventory | 28 | PASS |
| test_native_v2_serving_inventory | 45 | PASS |
| test_native_reader_binding | 135 | PASS |
| test_dp_placement_harness | 75 | PASS |
| test_calibration_pd_harness | 68 | PASS |
| test_native_pd_manifest | 14 | PASS |
| test_native_cold_start | 18 | Initial17pass/1fail; repaired18PASS |
| test_canonical_client_metrics | 24 | PASS |
| test_native_pd_canonical_http | 1 | PASS |

The real local-HTTP controls keep432 canonical requests, three repetitions and seven idle boundaries. Both role starts precede first-formal dispatch; both stops follow the observed count144 drain. HTTP500, timeout, invalid auxiliary configuration and an injected profile receipt write error preserve native client outcomes. A replay exception remains the identical exception after idle/stop cleanup, including failed stop RPCs. These are CPU/stand-in facts.

Collector controls prove a fresh successful-start budget survives prior startup drops, retains exactly three worker steps under the native-shaped limit and preserves startup/window snapshots. Archive-failure coverage initially proved business continuation only; the independent Standards review identified its missing budget-survival guarantee. That finding is open and this test run does not close it.

## Native iteration control

`native_iteration_control.json` records execution of the actual pinned `WorkerProfiler` class, extracted from its hash-verified source with CPU stand-ins for profiler start/stop. With limit3 the states before five executions are `[true,true,true,false,false]`, with one start and one stop. No CUDA import/initialization. Worker steps do not guarantee useful nonempty logical batches on every DP lane.

## Cold-start failure and correction

Original signal: isolated child `ImportError: cannot import name 'NATIVE_WORKER_EXTENSIONS' from 'vllm_replay'`. The fixture staged the current P/D reader with a historical command-builder peer. `cold_start_baseline_control.json` reproduces the same import failure using baseline313c8167 and passes after changing only the staged peer; all original fixture hashes are unchanged. This excludes a G2 default-replay regression and a missing Python dependency.

The correction stages and hash-binds both current repository modules in the existing fixture. All admission assertions remain. Exact rerun:

```text
env PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_cold_start.py -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/glm53_dsv4/g2_cold_start_repaired.xml > /data/ycfeng/tmp/glm53_dsv4/g2_cold_start_repaired.log 2>&1
```

Observed:18passed in34.65seconds; no errors/failures/skips. Original failure log remains under the initial regression root; isolated stderr is `/data/ycfeng/tmp/glm53_dsv4/g2_cold_start_child_stderr.log`.

## Default matrix and limits

`matrix_binding_initial.json` re-verifies the existing77-case comparison hash and unchanged `frontier/`, `data/`, `examples/`, fidelity-driver and canonical-input domains against capture4a1b53e6. This is an explicit binding, not a new capture. Final repaired source, independent review and commit binding remain pending. No KERNEL_ONLY serving total, GPU stream/PDL/capture correctness, complete decode coverage or native reference admission follows from these CPU results.

## Review correction iteration

Original production code with strengthened controls (`review_red.json`, logs/XML under `/data/ycfeng/tmp/glm53_dsv4/g2_review_red`): collector archive controls2tests/1failure; native CUDA/URI preservation2tests/2failures; active/persistent receipt and URI-window controls3tests/3failures. These reproduce all three distinct findings and the adjacent valid-URI preservation case.

Repairs: independently observe startup archive before unconditionally initializing a new window; derive per-node paths only for local torch output; retain native CUDA/URI configuration and reject unsupported auxiliary windows softly; route checkpoint errors carrying a profiling receipt into its own error list, including while replay is active and after stop.

Post-repair whole suites (not a summed count):

| Suite | Tests | Result |
| --- | ---: | --- |
| test_native_serve_profile | 32 | PASS |
| test_native_pd_worker | 19 | PASS |
| test_native_streaming_replay | 39 | PASS |
| test_native_cold_start | 18 | PASS |

All have0errors/0failures/0skips. `review_green.json` retains exact commands, Python version, explicit environment and seven source hashes; original logs/XML are `/data/ycfeng/tmp/glm53_dsv4/g2_review_green/`. `source_manifest_r1.json` and `change_r1.diff` bind433changed lines. Independent follow-up and final-source checks are still pending.

## Final frozen-source verification

Both independent R1 reviews closed all original findings with no new actionable findings (`review_final.md`). Root then froze commit `05952396`, checked every changed file against both the reviewed SHA-256 and `git show`, verified the unchanged frozen R4 reader and full native inventory, and ran the complete required set separately. Execution interval: 2026-10-23 08:25:32–08:28:42 UTC.

| Suite | Expected tests | Actual tests | Result |
| --- | ---: | ---: | --- |
| test_native_serve_profile | 32 | 32 | PASS |
| test_native_pd_worker | 19 | 19 | PASS |
| test_native_streaming_replay | 39 | 39 | PASS |
| test_native_serving_inventory | 28 | 28 | PASS |
| test_native_v2_serving_inventory | 45 | 45 | PASS |
| test_native_reader_binding | 135 | 135 | PASS |
| test_dp_placement_harness | 75 | 75 | PASS |
| test_calibration_pd_harness | 68 | 68 | PASS |
| test_native_pd_manifest | 14 | 14 | PASS |
| test_native_cold_start | 18 | 18 | PASS |
| test_canonical_client_metrics | 24 | 24 | PASS |
| test_native_pd_canonical_http | 1 | 1 | PASS |

Every row has return code 0 and zero failures, errors and skips. Counts are not combined. `final_checks.json` records every exact argv, Python version, environment, source hash, start/end timestamp and XML digest. Logs and XML remain at `/data/ycfeng/tmp/glm53_dsv4/g2_driver_final_05952396`. Root rehashed the source and all XML files before acceptance.

The existing matrix's 77 identical cases and empty failure/finding lists were verified against its recorded digest. All simulator/input/fidelity domains are unchanged from capture `4a1b53e6`; `matrix_binding_final.json` binds this evidence to `05952396`, explicitly with `new_capture=false`. The first binding helper assertion confused list-valued result fields with counts and failed before producing an artifact. The corrected assertion matches the actual report schema; no evidence or implementation changed.

The acceptance covers only the CPU driver/collector integration. Actual GPU hook execution, useful aligned batches, trace completeness, stream/event dependencies, PDL/graph attribution and numerical costs remain unverified; no GPU run was submitted during these checks.

## Archive integrity

Ordinary staged-file whitespace checking initially flagged the two raw unified-diff artifacts' blank context markers. All flagged contents were exactly one space, required by the patch format. Both artifacts then passed `git apply --cached --check --whitespace=error` against baseline313c8167 in separate temporary indexes; all other staged files passed ordinary whitespace checking. `patch_validation.json` records commands, digests and the original diagnostic. No source, frozen diff, active index contents or acceptance hashes were altered by these checks. `docs_check.json` records the final scope/source/receipt and parent-document consistency checks.

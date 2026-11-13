# F4a/F4b boundary repair verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record F4a/F4b RED/GREEN, actual native seams,552-case JUnit and budget. |

TaskID D29-PRICING-F4-BOUNDARIES. ReviewTaskID
D29-PRICING-FINITE-REPAIR-REVIEW, reviewer ses_e9abd8364ffee5sQKdpiQh3yH4.
Starting parent27b3499/sub3dac53c, original source baseline1aee0af and pinned
submodule d28fe917370c2d55230c3ea226e5660785b44aca. Re-review closes F3/F5/F6;
only F4a/F4b repaired. No new source paths/peers/imports/platform/GPU/agents.

## Environment and exact commands

Parent cwd `/data/ycfeng/Frontier/.worktrees/step5-position-pricing` unless stated.
Python3.10.6 in `/data/ycfeng/envs/frontier-py310` (existing conda environment).
All invocations cap OMP/OpenBLAS/MKL threads at1, disable bytecode/cacheprovider,
and retain logs/JUnit under `/data/ycfeng/tmp`. Optional torch/vLLM integration
checks remain skipped in this environment; those skips do not qualify GPU runtime.

RED command before production repairs, log `/data/ycfeng/tmp/d29_f4ab_red.log`:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_comm_cost_export.py tests/unit/test_step5_comm_transport.py -k 'branch_switch or lazy_fallback or stale_fallback' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f4ab_red --junitxml=/data/ycfeng/tmp/d29_f4ab_red.xml
```
Observed JUnit3 =2 FAIL/1 PASS/0 ERROR/SKIP,12.662s. Actual exporter branch_switch
fails with `Affine fit requires three distinct anchors and independent qualification`.
Actual published workspace-failure/PyNccl-fallback observer succeeds natively,
but validator rejects its changed enabled-state. Stale fallback negative passes.

Final affected command, log `/data/ycfeng/tmp/d29_f4_boundaries_final.log`:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_moe_pricing.py tests/unit/test_comm_operator_families.py tests/unit/test_step5_comm_cost_export.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_op_trace_utils.py tests/unit/test_step5_contract_reuse.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_comm_package.py tests/unit/test_step5_comm_transport.py tests/unit/test_step5_dispatch_probe.py tests/unit/test_step5_moe_profiling.py tests/unit/test_step5_fixed_probe.py tests/unit/test_execution_time_op_times.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f4_boundaries_final --junitxml=/data/ycfeng/tmp/d29_f4_boundaries_final.xml
```
Observed JUnit552 =463 PASS/89 SKIP/0 FAIL/0 ERROR,51.251s: PASS.

Independent cold importer/constructor command, log `/data/ycfeng/tmp/d29_f4_cold.log`:
```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f4_cold --junitxml=/data/ycfeng/tmp/d29_f4_cold.xml
```
Observed JUnit10 =10 PASS/0 FAIL/ERROR/SKIP,1.320s: PASS. No loader mock,
syspath injection or production fallback. Synthetic schema labels are explicit in
the test; this exercises the actual importer/local formula, not native qualification.

Targeted validator/observer command, log `/data/ycfeng/tmp/d29_f4_scopes.log`:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_comm_transport.py -k bound_selected -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f4_scopes --junitxml=/data/ycfeng/tmp/d29_f4_scopes.xml
```
Observed JUnit17 =17 PASS/0 FAIL/ERROR/SKIP,3.615s: PASS.

Submodule cwd `frontier/cc_backend/backends/collective-sim`, log
`/data/ycfeng/tmp/d29_f4_boundaries_sub.log`:
```sh
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-position-pricing:$PWD/python PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/test_grouped_plan_input.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f4_boundaries_sub --junitxml=/data/ycfeng/tmp/d29_f4_boundaries_sub.xml
```
Observed JUnit12 =12 PASS/0 FAIL/ERROR/SKIP,1.694s: PASS. Real HTSIM binary used;
completed-flow gates and TP-local no-network branch remain unchanged. No new sub
commit: tip3dac53c remains clean. Complete baseline-to-tip patch unchanged at
`collective_sim.patch`, SHA256
`c96cace541b72f7627b5f429d55615a17be4cd6c86d1827ef53969d05d766a1d`.

## Passing conditions and observations

F4a: one frozen TP8 BF16 runtime policy retains three calibration row anchors
16/256/8192 and held-out64 across custom/PyNccl switches; one affine model has
design-matrix rank2 and distinct_sizes3. All raw passes/count contexts preserve
actual branch/buffer/runtime; different PyNccl policy is not pooled and fails
missing-anchor admission. Candidate remains cpu_fixture/UNVERIFIED. DP path still
executes actual joint HTSIM and rejects partial/zero/nonfinite completed makespan.

F4b: execute published native FlashInfer `_ensure_workspace`/`should_use_fi_ar`
and CUDA communicator `all_reduce` methods from
`/data/ycfeng/tmp/d29_published_sources01/source/vllm/distributed/device_communicators`
through DispatchObserver. CPU stand-ins make workspace return None exactly once,
native method sets disabled=True and successful PyNccl fallback returns the original
tensor. Observer records actual fallback branch and post-native disabled state;
validator accepts matching collector/probe policy despite different startup state.
Stale fallback state, branch, buffer, frozen env/PyNccl version, missing policy,
selected scope/replay/source/probe key defects remain UNVERIFIED. Original native
exceptions retain identity; no retry/backend override/exception suppression added.

Importer: one frozen model is selectable across branch/buffer/current-enabled
changes, but plan.cache_key differs and missing complete count receipt is rejected.
After supplying exact paired synthetic count proof, real local predictor returns
the same affine duration and the new exact plan/count context. This establishes
binding/cache behavior only; schema labels do not constitute measured GPU evidence.

Frozen model context excludes only per-cell branches/buffers and mutable enabled
snapshot; image defaults/overrides, PyNccl versions, parallel flags and execution-
affecting NCCL settings remain identity. Full execution_policy and buffer_receipt
remain count/cache identity. Startup diagnostics use per_rank_startup_runtime;
probe runtime_policy is called after native operation, not reused from startup.
No qualification algorithm/criteria or approved finite model changes.

## Failed iterations and scope

- First d29_f4_green invocation referenced nonexistent
  tests/unit/test_collective_sim_prediction_modes.py: exit4,0 cases. Corrected
  invocation only; no new path or test fallback.
- Next d29_f4_green:2 FAIL/276 PASS/71 SKIP, both lazy tests missing explicit
  selected-manifest env and therefore not entering the selected observer seam.
  Test binds it with monkeypatch; d29_f4_lazy_green15 PASS and final17 PASS follow.
- d29_f4_green02 affected5:278 PASS/71 SKIP,37.89s. Superseded by final552 after
  exact importer/count/cache and frozen-env/buffer negatives were added.
- Budget-check first attempt counted pricing task records as source paths and
  rejected4 permitted documentation files. Source check now excludes task_memory,
  independently validates all10 changed source paths belong to44 authorized paths.

Actual gross source diff:43 changed paths,+2972/-95=3067/3956, remaining889,
including5 submodule files333. This repair is parent diff152 additions/29 deletions
relative27b3499 (181 gross), but approved original-baseline gross rises123 because
previously added lines are replaced; exact original-baseline budget is authoritative.
See summary.md for the complete per-path ledger. AST/JSON/whitespace and patch
byte equality are checked separately. Root-owned gitlink remains unstaged.

No real GPU parameters, device-selected policy, calibration, source/backend/T2/G3
or global D29 qualification. CPU original-method execution and real HTSIM prove
only recorded method/structural/numerical seams. Root owns independent re-review
and final gitlink/integrated acceptance.

Verified scoped parent source/record commit506b43ac. No new submodule commit;
tip3dac53c and complete patch unchanged. Root independent re-review remains open.

# Current D29 execution preparation verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 UTC / 2026-10-25 HKT | Verify recovered source/worktrees, prepared scripts, full-output corruption rejection and no-operation worker dry run. |

Production source remains `2ca5e57113b0bc99ab84e926e5bb492b3670072e`; records HEAD at preparation is `e08a133d419fa27008d9ccd5261f05ef065322f1`. Both comparisons against production2ca and frozen146 are empty outside `task_memory`. Four current satellite worktrees are clean and their HEADs are main-branch ancestors. The older89-attachment census is retained, not independently recomputed.

No new pricing, required-shape, local/image runtime probe, fullCLI or StepMind operation was executed. Existing strict77/full-unit/supplement results are revision-bound retained evidence; the raw279FAIL/32ERROR/281SKIP, unitexit1 and historical presenceFAIL remain unchanged.

## Exact static command and environment

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-preview`. Python executable `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6, `-B`, single BLAS/OpenMP/NumExpr threads, `TMPDIR=/data/ycfeng/tmp`.

```text
["timeout", "60s", "sudo", "-n", "systemd-run", "--scope", "-p", "MemoryMax=30G", "runuser", "-u", "brainpp", "--", "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1", "TMPDIR=/data/ycfeng/tmp", "/data/ycfeng/envs/frontier-py310/bin/python", "-B", "/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/verify_preparation.py", "/data/ycfeng/tmp/d29_execution_preparation_20261025/static02"]
```

The script is `evidence/execution_preparation_2026-10-25/verify_preparation.py`. Raw outputs are under `/data/ycfeng/tmp/d29_execution_preparation_20261025/static02`; selected receipts/stdout/stderr and the parsed report are archived beside the prepared scripts. The command exited0 in2.363082259s. The30GiB scope was started through the retained `sudo -n systemd-run ... runuser -u brainpp` recipe; measured cgroup peak is unavailable and no whole-workload RSS claim is made.

## Expectations and observations

| Check | Passing condition | Actual result | Status |
| --- | --- | --- | --- |
| Source/input binding | Current bytes equal admitted source, original inputs and retained fixture metadata | 25 source hashes, six inputs, 262 fixture metadata entries; pinned clean collective checkout | PASS |
| Prepared Python/shell scripts | Syntax valid without execution of simulator setup | Six Python files compile; bash syntax check passes | PASS |
| Wave expression/setup boundary | Original wave AST identical; old list-ledger/reporting/pricing tail absent | Exact AST equality; prefix has no store/phases/timing/plan assignments | PASS |
| New JSONL framing/full values | Complete reference values and new IDs serialize exactly and no extra rows |16 sorted ledger rows,208 exact operator rows,144 compute proofs | PASS |
| Corruption controls | Reject all injected corruption without pricing | Seven expected rejections: missing ledger row, order, compute proof, duration, extra operator, stale cold batch ID, source identity | PASS |
| Worker dry run | No StepMind/auth/source import or submission | JSON status `DRY_RUN_NO_PLATFORM_OPERATIONS`,4CPU/0GPU/64mem_gb, pinned image and mounts | PASS |
| Invocation preservation | No wave/shape/fullCLI/GPU execution | New pricing0, shapes0, platform0; required-shape6/6 unchanged | PASS |

The positive control reserializes the paid retained16/208 references with a supported new synthetic-ID offset. It validates the checker only; it is not a newly generated wave. Actual joint runtime, warmed plan reporting, worker image/mount connectivity, append throughput, lifecycle and full-workload memory still require their stated executions.

## Original scope startup errors

The initial static command omitted the retained `sudo/runuser` prefix. System `systemd-run` returnedexit1 with `Failed to start transient scope unit: Interactive authentication required.` before Python started; wall0.012073246s. A direct user-scope `/usr/bin/true` check also returnedexit1 without output. Reading the already committed runtime command establishes the correct root scope plus brainpp child recipe. Static02 uses that exact mechanism and passes; neither failed scope check invoked pricing or model setup. Preserve `preparation_static01` raw evidence and do not rewrite it as PASS.

## Final prepared bytes and cleanup

The staged whitespace check found one new blank EOF line in `joint_setup.py`. Remove only that line and refresh exact setup/joint/worker bindings. Finalstatic03 executes the same controls on the final prepared bytes:exit0,2.199947817s, all seven expected rejections and no-operation dry runPASS. Its exact command and outputs are retained as `preparation_static03.*`; static02 and the original scope failures are unchanged. No production source or wave is involved.

After recording full hashes, remove only the four generated reference-copy JSONL files from this preparation's `static02` and `static03` scratch directories. Original active fixture/ledger/operator references and selected control evidence remain. Cleanup details are `preparation_cleanup.json`; cloud files were not touched.

## Remaining limits

Two new ordinary D1/layer60 pricing calls are a concrete pending proposal, not granted by resume or a new label. The worker preparation uses existing H800/codesign task authority with zero requested GPUs; no native/profile collection or fullCLI is hidden in it. The1GiB/30s append control is serialized-reference storage evidence only. Full6144/1024 offline/online,2400s per main child, final T2/source/package, native/G3 and each E2E<=10% remain unverified.

# Current D29 execution preparation verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 UTC / 2026-10-25 HKT | Close preparation Standards/Spec follow-ups with No findings; freeze task-only scripts before one readiness worker. |
| 2026-10-24 UTC / 2026-10-25 HKT | Verify the final copied Git and task-scoped configuration in the fixed image; preserve prior failures and affected static06 controls. |
| 2026-10-24 UTC / 2026-10-25 HKT | Repair preparation review findings and the SDK field/mount checker; retain original failures and zero runtime calls. |
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

## Review correction and original SDK object checks

All commands use the same retained60s/sudo-n/systemd-run30G/runuser-brainpp recipe. SDK checks use /data/ycfeng/tmp/stepmind-env/bin/python, Python3.10.6, PYTHONPATH=/data/ycfeng/steptron and PYTHONNOUSERSITE=1; static checks use the canonical Frontier Python3.10.6. All use-B, one math thread and local TMPDIR. Exact argument arrays, UTC times and raw stdout/stderr are in worker_spec01/02/03.execution.json and preparation_static04/05.execution.json beside the scripts. SDK03 and static05 ran concurrently; these walls are observed controls, not isolated timing or full-workload memory.

| Check | Expected condition | Actual | Verdict / limit |
| --- | --- | --- | --- |
| worker_spec01 | Original Task fields readable |exit1,4.682937818s,AttributeError private_machine | FAIL preserved; original checkerbdc0676a and traceback retained. |
| worker_spec02 | Correct fields and resources |exit0,4.299797723s; original SDK adds tenant mount | Field assertions pass; mount equality is tainted by input-list mutation and does not qualify the frozen mount plan. |
| worker_spec03 | Plan unchanged; original SDK task/image/resources/NFS and documented default mount exact |exit0,4.515597923s,4CPU/0GPU/64000Mi/1replica/H800; NFS100.96.128.195:/data/ycfeng:/data/ycfeng | PASS_OBJECT_MATERIALIZATION_ONLY; no client/auth/query/submission. |
| static04 | Repaired header identity and original full-output predicates |exit0,2.704284248s,10 expected rejection controls | PASS_CHECKER_ONLY; no setup/pricing. |
| finalstatic05 | Final worker-plan/script hashes and unchanged reference predicates |exit0,2.865911072s,25source/sixinputs/262metadata,16ledger/208operator/144proofs,10 rejections and no-operation dry run | PASS_PREPARATION_ONLY; no pricing/shape/platform. |

Trace checks now compare expected evidence, cold/warm label and source_commit exactly. Frozen cache_model_name fields are required; model_bytes names the serialized bytes, and unused profile/iterations parameters are gone. Three new corruption controls independently change each trace identity field and are rejected. The earlier seven controls remain. Final executable/checker hashes are in the receipts; the exact intermediate checker was reconstructed and verified against its original executionSHA e6e7e788 before retention as verify_worker_spec_fields02.py.

ResourceConfig.__iter__ is independently inspected at base_exp.py4385–4460, SHA0fc88ea9fd70ad08f69d76637f4e05c65309f80ce81b4ff933c3b0d0320d4d37. It always adds its tenant mount and mutates the supplied list. The corrected constructor copies requested mounts and the checker separately verifies the unchanged plan and the mandatory original SDK default. The workload never reads/writes tenant; all task cloud reads/writes remain personal codesign. This is an observation of the original API, with no SDK override or new task-data source. Paramiko TripleDES warnings are retained and are not the original failure cause.

Configured MemoryMax30G and the requested64mem_gb do not establish effective full-workload RSS. Independent follow-up review is pending; new pricing0/shape0/fullCLI0/platform0, exhausted required-shape6/6 and old-control ceilings remain unchanged. Existing production source2ca/146 identity and budgets are retained without rerunning source regressions.

## Fixed-image Git failures and final affected verification

All Git controls use the pinnedf037d958 image, copied runtime Python3.10.6, --pull=never, --network none, read-only root/local /data/ycfeng mount,4CPUs,60s child limit and configured Docker memory30GiB. Exact argument arrays, UTC times, script/plan hashes and raw stdout/stderr are retained beside the scripts. The controls import only stdlib joint_timing binding code, with no model setup, runtime import probe, pricing, shape, fullCLI or StepMind operation.

| Control | Expected condition / hypothesis | Observed result | Status / limit |
| --- | --- | --- | --- |
| worker_git_preflight01 | Read source identities in the pinned image |exit1,1.818949030s,FileNotFoundError git | FAIL retained. |
| worker_git_preflight02 | Copied Git links; exact environment safe.directory entries permit owned source queries |exit1,1.043054268s; assertion omits child stderr | FAIL retained; no PASS inferred. |
| worker_git_diagnostic03 | Same queries expose child results without parameter changes |exit0,0.975511407s; imageUID0/sourceUID10250, both repositories report dubious ownership even with the attempted environment entries | DIAGNOSTIC_ONLY; no pass claim. |
| worker_git_preflight04 | Hash-bound Git and protected task-only file trust exactly the two repositories; real binding calls are equal |exit0,1.567929203s; Git2.34.1, both hashes/entries/origins and two real check_bindings calls pass | PASS_FIXED_IMAGE_GIT_SOURCE_BINDINGS_ONLY. |
| preparation_static06 | Final worker hashes and original complete-output predicates survive Git setup changes |exit0,3.099603861s;25source/sixinputs/262metadata,16lane/208operator/144proof,10 expected rejections, dry-run and bash-n | PASS_PREPARATION_ONLY. |

Git executable3665328B/SHA06c1cfb4848572479bd3bbf678c224432d262732049957884dd9be2457185ad7 and configurationSHA672995b50f25e01b216a981ac1f08eebee28b0713b4bb6c53aecfedad0bff6d4 are bound in worker_plan.json. The task configuration uses GIT_CONFIG_GLOBAL and only the current worktree/collective checkout. No system/global host config, original runtime or image was edited. The runtime copy/configuration operation receipts preserve exact paths and contents.

Static06 retains the canonical60s/sudo-n/systemd-run MemoryMax30G/runuser-brainpp, one math thread, -B and localTMPDIR recipe. Configured limits do not establish actual full-workload RSS. Follow-up reviews remain pending; joint pricing remains unapproved and actual worker/readiness has not run at this checkpoint.

Final independent follow-up is complete on the same prepared bytes: Standards0 documented findings/0 smells, Spec0 findings. The initial issues are resolved; neither reviewer executes runtime or platform work. The raw review identities, inspected phase and remediation evidence are in review_execution_preparation_2026-10-25.md.

Final metadata receipt preparation_review_followup.json verifies the five current worktrees, empty non-task diffs against2ca/146, final scripts/Git hashes and both passing affected receipts. preparation_review_cleanup.json records exact hashes/bytes and deletion of only six completed own static04/05/06 reference-copy files,15603024B. Original active inputs, selected full-control reports/stdout/stderr/receipts and other-session work remain. No cloud cleanup or platform operation occurs in this preparation.

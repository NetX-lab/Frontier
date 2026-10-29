# Complete CLI preparation review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 UTC / HKT | Collect both independent axes, reproduce two defects, verify repairs and close fixed8e3d5088 follow-up with No findings on both axes. |

Fixed point: `0dca7f83ea62f18fef3f6c400db27ccc45260d39`. Initial candidate: `0eded7cf380d68da9c647c775af0b89b16c2d478`.

Exact comparison: `git diff 0dca7f83ea62f18fef3f6c400db27ccc45260d39...0eded7cf380d68da9c647c775af0b89b16c2d478`.

Review scope is the complete CLI plan, launcher, worker, shell entry, preflight and changed current task records. Both reviewers inspect independently and remain read-only; no client/platform/workload operations occur.

## Standards

Reviewer: Tesla, `01a17149-b68d-7422-a9e3-28209ce1ea2f`. One documented finding, no separate heuristic smell.

**P2:** The `spawn_tasks()` failure and a later API/check exception both write `launcher.failure.log` (initial launcher lines 66 and 138). The second write replaces the first traceback; the final receipt also keeps only the later error. Workspace AGENTS.md Verification requires relevant logs/errors, and stepmind-gpu-worker/SKILL.md requires distinguishing query failures from terminal workload failures. Preserve each stage separately.

## Spec

Reviewer: Poincare, `01a17149-b789-7cf3-9310-0cdf739db657`. Two findings.

**P1:** A submitted worker can fail before creating `worker_started.json`. The original `spawn_tasks()` waits before returning, so its exception leaves no return-based job name and the startup-file fallback is unavailable. This skips the required creator/quota/resources/image/source and replica/container terminal evidence (`execution_plan_2026-10-25.md`, complete CLI preparation).

**P2:** A later API/check exception overwrites the initial traceback and error field. The execution plan requires every raw workload, validation, memory or platform failure to remain explicit evidence.

## Remediation and affected verification

The unchanged original SDK emits its exact `Job Create` event after successful submission and before its blocking wait. A synchronous, narrowly filtered log sink persists that identity immediately; the returned worker identity must match. The launcher removes the startup-file fallback. Its original submission/wait/atexit behavior is unchanged, and the exact SDK source is now an input binding.

The initial failure remains `launch_error` plus `launcher.failure.log`; a later failure remains `error` plus `launcher.followup.failure.log`. The original launcher bytes are retained as `evidence/complete_cli_2026-10-25/launch_complete_cli_initial.py`.

`review_fault_injection.py` executes the actual hash-bound SDK controller log statement with fake submission/query objects. It reproduces both original defects, then passes the pre-worker terminal-read, two-exception preservation and success cases. Raw exit 0, outer wall 0.422564383s. No real client/job/workload is created.

Affected original-SDK/static/dry-run check passes with raw exit 0, outer wall 4.461541410s; syntax, exact command, 25 source/six inputs/262 metadata bindings, 4CPU/0GPU/64000Mi, one replica, fixed image/current NFS and unchanged mounts all match. Original preflight evidence remains unchanged. Final plan SHA: `4e7df1aa584a8ed52ba1712b8ff729367c35b0895025be68222dfe4b7266567e`.

Independent follow-up on `8e3d5088ed82855b16aebdc868032a5624cc71c6` is complete with No findings on both axes. Initial counts stay separate: Standards 1 finding (P2); Spec 2 findings (P1/P2). The repairs and CPU checks do not declare fullCLI/RSS or D29 acceptance.

## Standards follow-up

Reviewer: Tesla, `01a17149-b68d-7422-a9e3-28209ce1ea2f`. Candidate: `8e3d5088ed82855b16aebdc868032a5624cc71c6`. P2 CLOSED; **No findings**. The original and follow-up traceback/error fields are independently retained. The original SDK log sink is narrow and removed in finally; source/plan hashes and archived fault/static controls match. No new documented breach or separate heuristic smell. The reviewer remains read-only and runs no test/client/platform/workload.

## Spec follow-up

Reviewer: Poincare, `01a17149-b789-7cf3-9310-0cdf739db657`. Candidate: `8e3d5088ed82855b16aebdc868032a5624cc71c6`. P1 and P2 CLOSED; **No findings**. Actual SDK creation identity is saved before wait and independently of worker startup; returned identity is checked, and the failed job/replica read remains reachable. Both exception stages survive. Original SDK/wait/atexit, complete workload, resources, outputs and invocation limits stay unchanged. The reviewer remains read-only and runs no test/client/platform/workload.

Final counts: Standards0 / Spec0; initial Standards1(P2) and Spec2(P1/P2) remain recorded separately with their closure. This closes preparation review only; actual complete CLI/RSS and D29 qualification remain open.

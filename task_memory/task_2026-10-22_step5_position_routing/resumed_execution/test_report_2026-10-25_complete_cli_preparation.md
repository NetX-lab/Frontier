# Complete CLI preparation and review repair verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 UTC / HKT | Archive the initial preflight and affected repair checks without platform/workload calls; independent follow-up on8e3d5088 closes both axes with No findings. |

The complete CPU-fixture worker is prepared, with two independently reported evidence defects repaired. The complete CLI has not been submitted. Production remains `2ca5e57113b0bc99ab84e926e5bb492b3670072e`, equal to frozen `1461573d` outside task_memory; existing source budgets, joint2/2 and required-shape6/6 remain unchanged.

## Commands and environment

Working directory for all checks: `/data/ycfeng/Frontier/.worktrees/step5-preview`. Python: `/data/ycfeng/tmp/stepmind-env/bin/python`, 3.10.6, `-B`, no user site, single OMP/OPENBLAS/MKL/NUMEXPR thread. Each small check uses `timeout 60s` and the existing sudo/systemd scope configured with `MemoryMax=30G` (32,212,254,720B). That configured limit is not a measured workload RSS.

Affected checks run once each in independent scopes:

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G \
  runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=/data/ycfeng/steptron \
  /data/ycfeng/tmp/stepmind-env/bin/python -B \
  task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/complete_cli_2026-10-25/review_fault_injection.py
```

The second command has the same environment and replaces only the final script with `preflight_review_followup.py`. Exact absolute-path argv, UTC start/end, script SHA, raw exit and captured stdout/stderr are recorded in each `*.execution.json` and its sibling logs.

The retained initial preflight command used the identical scope/environment with `/data/ycfeng/tmp/d29_main09_preflight_20261025_01.py`. Its exact original script is `preflight.py`, SHA `d09a769cc9bebc9921543b60a5e910bef793e22d5487d9f6bea4abe5cbda587a`; raw exit 0, wall 4.695963354s. Its original plan SHA remains `fafd6776f36e68db9e4a925564245ad1a34c5c0168a0206b7843eadb49793303`. Those files are retained unchanged and are not re-signed as final-plan evidence.

## Expected and observed results

| Check | Passing condition | Observed | Result |
| --- | --- | --- | --- |
| Initial syntax/dry-run/SDK | Both Python scripts compile, bash-n/dry-run pass, exact original SDK resource/mount command | Raw exit0, 4.695963354s; no client/job | PASS for original plan |
| Original defects | Demonstrate missing pre-start terminal reads and overwritten original traceback | Both fail their intended evidence predicates using original launcher bytes | RED reproduced |
| Repaired early worker failure | Persist original SDK job name before wait; collect failed job and replica despite absent startup file | Exact created name already in launcher.json; fake job and replica read; Failed/exit1 retained | GREEN |
| Repaired query failure | Initial and follow-up tracebacks/error fields survive independently | INITIAL_WAIT_FAILURE in original log/launch_error; FOLLOWUP_QUERY_FAILURE in separate log/error | GREEN |
| Successful completion path | Original wait returns and all exact terminal/business predicates pass | Full launcher terminal/business status PASS using fake objects | GREEN control only |
| Affected preflight | Final syntax/bash-n/dry-run, bindings and original SDK resources/mounts | Raw exit0, 4.461541410s;25source/sixinputs/262metadata,4CPU/0GPU/64000Mi,one replica,current NFS/fixed image | PASS |

Fault-check script wall is 0.267910046s; outer scope wall is 0.422564383s. Affected preflight script wall is 3.872275814s; outer scope wall is 4.461541410s. These are distinct timing scopes.

Final plan SHA is `4e7df1aa584a8ed52ba1712b8ff729367c35b0895025be68222dfe4b7266567e`. Independent Standards/Spec follow-up on8e3d5088 is complete: P2/P1-P2 CLOSED, No findings on both axes. The initial axes report Standards1 (P2), Spec2 (P1/P2); overlap is preserved.

## Evidence and limits

Exact evidence lives under `resumed_execution/evidence/complete_cli_2026-10-25/`: original launcher/preflight, both affected scripts, full JSON receipts/stdout/stderr, final plan and `review_initial.json`. `review_fault_injection.json` retains every fake case's full receipt and both tracebacks. SDK objects and extracted controller code are local-only; no real platform client, submission/query or production workload is called. Fake exp names are explicitly diagnostic and are not platform evidence.

All five current worktrees were directly inspected: the four satellite heads1461573d/1461573d/b061f884/6335f581 remain clean mainline ancestors. Collective-sim remains clean pinned `3dac53c544e546574cb6e8078cd14053d0916113`. No composed audit, source regression, readiness, pricing or shape check is rerun.

Existing source admission remains strict77=77/77 and canonical full-unit raw8028PASS/279FAIL/32ERROR/281SKIP,exit1, all8620IDs/statuses unchanged. Independent101/197/53/52 supplements overlap. Current preparation never qualifies full6144/1024CLI, effective full-workload memory, finalT2/package/native/G3/E2E or the complete D29 task.

The personal cloud output directory and four local output/capture paths were directly checked absent before launch. Existing credentials were checked only for presence/private permissions. This is path preparation, not a platform, quota, throughput, memory or workload measurement. Exact review and path evidence are review_followup.json and launch_path_readiness.json; the unique worker is still unsubmitted at this report checkpoint.

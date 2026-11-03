# Formal P2-64 generation03 preparation and execution

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-03 | Recorded lead-caused launcher interruption, independent terminal/event/audit acceptance and zero-container evidence; no reference was produced. |
| 2026-11-03 | Independently accepted the third-generation package after generation02 preemption, under the existing three-job allowance. |

## Scope and result

The existing D-P2-64-FORMAL-B preemption rule permits one remaining GPU submission after exp-1102-164536-160317 and exp-1102-233941-576872. The builder adds only the `p2_64_formal_retry03` CASES entry. It keeps native `a06a72a4326cba02234df6cfbdc21b78d3a15d26`, harness `f33306d40443cfc0c44842d6ee044f6703c736e7`, model, engine,3repetitions of128formal6144/1024requests,16pair warmups per repeat, promotion receipt,64H200/all-preemptible/step_main/3600s/64GPUh. Run tag: `dsv4-h200-p2-64-formal-20261103-03`.

Package: `h200_mainline/p2_64_formal_retry03/package_01`. Manifest SHA256 `9f9ec3c1f9d2b72b957d0678a0e8f06e05de8bd1fbd68b8addc3c092043bb474`. It binds promotion receipt SHA256 `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596`. The old generation02 package remains unchanged at manifest `9f9a1ab954ef3b7d61ff67f2f62c891e9da8547590a21b19db020c8557e8ed7d`.

## Verification

Repository cwd: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Environment `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp PYTHONPATH=$PWD`; no conda activation. Build/check-only Python3.12.13 `/data/ycfeng/envs/openmopd-py312/bin/python`; lead audit system Python3.10.6.

```bash
/data/ycfeng/envs/openmopd-py312/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/build_package.py p2_64_formal_retry03 --package package_01 --harness /data/ycfeng/Frontier/.worktrees/p2-64-startup-harness
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/tmp/glm53_dsv4/h200/audit_formal03_package.py
```

The audit invokes the new package's `launch.py --check-only` with the3.12interpreter and independently rehashes both generations.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Existing builder seal/preflight/R3 reconstruction/offline launch path | Accepted input identity and8TP/4DP/32EP/PP1 per role |47sealed files; all stages PASS | PASS |
| Lead seal hashes | Every input equals its manifest |47/47new and47/47old match | PASS |
| Generation comparison | Only identity/path/provenance changes |41/47 byte-identical;6files differ only by exact package-root/run-tag/cloud-path/build-time/purpose/overlay-digest replacements | PASS |
| Source/workload/resource preservation | Clean native/harness pins, unchanged code and workload, approved resource shape | Both commits clean; every code/trace/engine/Frontier input byte-identical;8x8H200/preemptible/3600s | PASS |
| Lead launch check-only | No submission and explicit hash success |exit0,INPUT_HASHES_PASS_NO_SUBMISSION | PASS |
| Fresh generation | No existing launch markers/case runs | No markers or case/runs;submitted=false | PASS, local scope |

The first lead comparison failed because its replacement order changed the run tag before replacing the containing cloud path. The second failed because the YAML generation omits the `dsv4-` prefix. The audit now applies longest identity substitutions first and includes the exact YAML generation string; all remaining bytes compare equal. Package files were never changed to satisfy the audit. No runtime tests were repeated for this registry-only addition; actual build/check-only settle its acceptance. Evidence: `package_01_build.json`, `lead_package_acceptance.json`, and `executor_build.log` in the new generation directory. Pi label p64-formal3-build-1103a, session68a759cf-88df-4d7e-8532-fa8c9266a23c, settled/exit0; lead reviewed full7-line diff.

## Execution and limits

Package/source commit9651750b is complete. Launcher907414 submitted exp-1103-092042-217677 at2026-11-03T01:20:42Z. Submission resources were verified through Pi p64-formal3-1103b. That Pi label was later stopped by the lead, unintentionally interrupting its detached launcher before any replica scheduled. Launch through `/ds4-exec`, retain the launcher, inspect the exact job, and collect terminal/copy evidence. It is the third GPU job; no fourth or automatic restart is authorized. The24.3GPUh historical estimate is not measured usage of this pair-warmed generation; the unchanged ceiling is64GPUh. Post-run coverage/R5/all-request/warm checks require a complete reference. Prior formal attempts remain incomplete, and missing end clocks keep their exact usage open. Do not repeat the completed Frontier replay.

Commit whitespace check: three inherited trailing-space lines in frozen native_config_parallel.py at228,271,274 remain byte-identical to the approved native source and generation02. The authored-file check passes; immutable source bytes are preserved.

Post-run exact argv is recorded in `h200_mainline/p2_64_formal_retry03/post_run_command.json` as PLANNED_NOT_RUN. It uses existing compare.py with explicit R5 and the completed formal replay. The existing copy wrapper was prepared as `/data/ycfeng/tmp/glm53_dsv4/h200/copy_p2_64_formal_retry03.sh`; bash-nPASS, not executed before GPU terminal. A transient Git index.lock interrupted the first commit command; the lock disappeared before inspection, and the scoped commit succeeded without deleting a lock or rewriting history.

## Terminal result and cause — independently accepted

At2026-11-03T01:26:58Z (09:26:58 Asia/Hong_Kong), the lead called `bash /data/ycfeng/codex-home/skills/ds4-exec/scripts/run_ds4.sh --stop p64-formal3-1103b` to replace a monitor that repeatedly called an unauthorized Pod endpoint. This was an operational mistake. The host stop implementation at `host_launch.sh:75` selects every process carrying that Pi session's `DS4_SESSION_ID`, regardless of its process group/session. `submit_once.py` copied its environment into `Popen(start_new_session=True)`, so launcher907414 was selected too. Its traceback is KeyboardInterrupt in `spawn_tasks` wait, followed by Job Stop at01:26:58.920Z. The platform audit has an explicit succeeded stop by i-fengyicheng at01:26:58Z, receipt01M6FG8GK3XYNR1DBAKDP3SZRA. The lead owns this interruption; it is not preemption, a native-engine failure, or a user-requested stop.

Pi `p64-formal3-terminal-1103c`, sessioneba262b5-3141-417a-a059-e1ca621c3e23, completed normally with exit0/settled. Read-only environment: `/data/ycfeng/tmp/stepmind-env/bin/python`3.10.6, personal credentials, runbook proxy and `STEPMIND_BACKEND=rjob`. Exact commands:

```bash
timeout 60s /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/ds4_runs/p64-formal3-terminal-1103c/scratch/query_job.py
timeout 60s /data/ycfeng/tmp/stepmind-env/bin/python -u task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/terminal_receipt.py exp-1103-092042-217677 /data/ycfeng/tmp/ds4_runs/p64-formal3-terminal-1103c/scratch/platform_terminal_receipt.json
timeout 60s /data/ycfeng/tmp/stepmind-env/bin/python -u /data/ycfeng/tmp/ds4_runs/p64-formal3-terminal-1103c/scratch/collect_evidence.py
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/tmp/glm53_dsv4/h200/audit_formal03_terminal.py
python3 /data/ycfeng/tmp/glm53_dsv4/h200/check_detached_launcher_lifecycle.py
```

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Terminal identity/cause | Exact job/UID/creator and initiating stop evidence | JobUID f8d2b600-f5c6-4e04-bf62-28123802235d; Stopped01:26:58Z; local interruption plus succeeded personal stop audit | PASS for cause, native run unsuccessful |
| Replica placement/history | Every replica owner and retained scheduling state inspected |8/8owner UIDs match; all PodScheduled=False/Unschedulable/assigned0; no node/IP/container state | PASS for never-scheduled observation |
| Query coverage | Exact objects, complete event/audit pagination |29/29PASS;82live events,66historical events;9RJob audit rows and2per replica | PASS |
| Business artifact | Complete3repeats needed for comparison | Only prelaunch run_manifest.json,12473bytes,SHAe7f337107d0e759ad32c1ba953a7ad32155d3b9d7d5f0a495dd4e335c0a75e04;0native outputs/repeats | INSUFFICIENT_EVIDENCE |
| Local lifecycle reproduction | Detached child survives normal parent exit yet matches stop marker | SeparatePID/SID/PGID;normal parent exit preserved child;marker retained;test-only SIGINT observed | PASS, CPU sleeper only |
| Cloud copy/coverage/R5/gate/warm | Require produced native artifacts | NOT RUN: no container started; no native-produced artifact to recover | NOT RUN |

The82live/66historical counts include job-projected replica events, not that many unique incidents. The platform created replicas at01:25:42Z but never scheduled them; `Start replica` denotes a controller action, not a started container. No Scheduled/Pulled/Created/Started/Preempted/Evicted event appears in the returned evidence. No deeper node or billing query was made.

Accounting: the generic terminal receipt retains exact `container_gpu_hours=null`, known subtotal0.0 and clocks_complete=false. Separately, the complete unscheduled replica state and events support an inferred0container GPUh for generation03. This inference is not a billing receipt and does not close missing generation01/02 end clocks. Historical35-job exact158.27166666666665GPUh and generation01 known4.053333333333334 remain unchanged.

Durable acceptance and original selected evidence: `h200_mainline/p2_64_formal_retry03/generation03_terminal_acceptance.json`, the `generation03_*` JSON/log files, `local_stop_mechanism.json` and `local_lifecycle_check.json`. The12selected evidence files were checked against known credentials and copied byte-for-byte; original raw queries remain in executor scratch. No fourth GPU submission, CPU copy, model/harness change, repeated R5 test, or Frontier replay occurred.

Execution correction: never call ds4 --stop on a label that owns a live StepMind launcher. Let finite submission turns finish normally, then use a distinct monitor label with exact replica queries and complete command output. Do not remove the tracking marker. The fourth candidate is locally prepared and independently checked in the separate generation04 report; its resource approval is pending.

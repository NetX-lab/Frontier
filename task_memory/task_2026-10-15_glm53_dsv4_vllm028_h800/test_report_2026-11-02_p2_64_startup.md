# Test report: formal P2-64 startup and current-state recovery

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | Retained exact-job scheduling evidence, verified its bindings locally and corrected stale current task pointers. Native execution and post-run acceptance remain pending. |

## Scope and outcome

Authority: requirements D-P2-64-FORMAL-B, D-P2-INPUT-QUALIFICATION and the current-branch continuation. Tickets 18/19. Repository baseline `0ee7f6c2`; package and standalone harness were accepted before this continuation ([Q5 report](test_report_2026-11-02_p2_64_formal_package.md)).

The formal job is `exp-1102-164536-160317`, created 2026-11-02T08:45:36Z. This continuation did not submit, stop or restart a GPU job. At the detailed snapshot, 2026-11-02T10:29:23Z, all eight replicas were unassigned and had `PodScheduled=False/Unschedulable`; no container had started. The job was past queue admission and repeatedly rejected by the co-scheduling prefilter. The original monitor's later local status file `status_p64f_36.json` (mtime 10:33:50Z) still reports Starting / eight pending replicas. Local launcher PID 3140124 remained alive.

This is a scheduling blocker. No native success, reference admission, warm coverage or numerical acceptance is claimed.

## Commands and environment

The platform investigation used the already authorized Pi executor, `custom/deepseek-v4.1-flash-sys`, thinking `high`, mode `read-only`, label `p64-startup-inspect-1102b`, session `0adb96f9-4ad5-48f2-80ca-42533e7676c2`. It completed with `exit 0` and `[settled] aborted=false`. The lead inspected the report and actual retained fields rather than inferring success from the process exit.

Launch command:

```bash
bash /data/ycfeng/codex-home/skills/ds4-exec/scripts/run_ds4.sh \
  p64-startup-inspect-1102b \
  /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 \
  /data/ycfeng/tmp/ds4_runs/prompts/p64-startup-inspect-1102b.md read-only
```

Pi used `/data/ycfeng/tmp/stepmind-env/bin/python` (the existing Python 3.10.6 / brainpp 2.8.6 environment) after the handbook's personal-auth/proxy initialization. Its bounded exact-job reads were `timeout 90s ... job_raw.py`, `timeout 120s ... replicas.py`, `timeout 180s ... events.py`, and `timeout 120s ... snapshot.py` from `/data/ycfeng/tmp/ds4_runs/p64-startup-inspect-1102b/scratch/`. Platform: `kun-workspace-vgen2`, namespace `shai-core`, personal creator `i-fengyicheng`. No credential or raw pod environment is retained in the task directory.

Independent local evidence command (Python version and full argv in the JSON receipt):

```bash
python3 /data/ycfeng/tmp/glm53_dsv4/h200/verify_p64_startup_20261102.py
```

Result: `PASS 16 local evidence checks; snapshot 2026-11-02T10:29:23Z`. Receipt: [verification.json](h200_mainline/p2_64_formal/startup_20261102/verification.json). This script reads Pi's captured data and local files; it makes no platform call.

## Expected versus observed

| Check | Expected / passing condition | Observed | Result |
| --- | --- | --- | --- |
| Receipt and package identity | Match Q5's issued receipt and manifest | Receipt SHA256 `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596`; package manifest SHA256 `c22085987055d018efb8abc104081cfe56d001b22026bb3aa367564d46754cd7` | PASS |
| Chronology | Receipt < package build < native submission | 07:38:49Z < 08:42:19Z < 08:45:36Z on November 2, 2026 | PASS |
| Resource and creator | 8 replicas × 8 H200, step_main, all preemptible, personal creator | All agree between config, pre-launch manifest and platform snapshot; platform label `yes` | PASS |
| Local source | Mount the sealed current-host package | `100.96.128.195:.../p2_64_formal/package_02` at the same container path | PASS |
| Replica census | Inspect all eight replicas and their scheduling conditions | 8/8 unassigned, no start time or container status, each Unschedulable | PASS for inspection; runtime BLOCKED |
| Scheduler cause | Actual event text distinguishes scheduling from image/mount/init | 27 RJob event records; all FailedScheduling records concern co-scheduling; member gaps 8, 2, 7, 8; no replica event-read errors | PASS for cause class |
| Retention | Retained bytes equal the Pi snapshot/report | Both SHA256 values match source and retention receipt | PASS |
| Launcher | Existing Python launcher stays alive | PID 3140124, correct package command, no signal sent | PASS |
| Native terminal, download, coverage, R4, metrics | Actual terminal and business evidence | No terminal receipt or business output yet | NOT RUN; depends on scheduling/execution |

The `member gap` values are deficits reported by the scheduler, not counts of available nodes. The returned resource-gap map is empty; the evidence does not identify GPU versus CPU, memory or RDMA as the specific constraint. No ETA follows from these records. The separate 10:18:35Z capacity snapshot (30 usable H200, three full eight-GPU nodes) is consistent context only.

## Task corrections and remaining work

`check-progress` found S5 stale current pointers. `requirements.md`, `plan.md`, `progress.md`, ticket 19 and `summary.md` now record the issued receipt, Q5 completion, existing job and recovered clocks. The previous 35 terminal jobs total 158.271667 container GPUh. The new job has no terminal container usage; queuing and the 64 GPUh ceiling are not added to the ledger.

The already accepted five-file harness change is being consolidated onto this final branch in parallel with the native wait. The frozen package and standalone harness remain fixed for the run and R4. Integration verification is recorded separately when complete.

After native termination: terminal receipt → hash-identical cloud copy → pair coverage soft gate → R4 → unchanged formal replay's all-request and warm/empty readings → actual GPU-hours and per-repeat case verdict. A tool's soft exit 0 never substitutes for inspecting its result JSON. H800 resources require the concrete package/estimate and subsequent approval specified in D-P2-64-FORMAL-B; GLM and the EP kernel-count correction remain paused/deferred.

## Evidence locations

- `h200_mainline/p2_64_formal/startup_20261102/report.json`: sanitized job, eight replicas and scheduling events.
- `h200_mainline/p2_64_formal/startup_20261102/snapshot.json`: exact-job/replica snapshot.
- `h200_mainline/p2_64_formal/startup_20261102/retention.json`: source paths and retained hashes.
- `h200_mainline/p2_64_formal/startup_20261102/verification.json`: independent 16-check receipt and launcher command.
- [Phase checkpoint](check_progress_2026-11-02_p2_64_formal.md): current route and S5 correction.

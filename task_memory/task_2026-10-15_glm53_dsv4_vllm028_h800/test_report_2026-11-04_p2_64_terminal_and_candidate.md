# P2-64 generation04 terminal and generation05 candidate verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-04 | Independently accepted the generation04 terminal cause and unsubmitted generation05 package; preserved incomplete accounting and native acceptance. |

## Result

PASS for terminal-cause recovery and CPU-only candidate verification.
Formal P2-64 native/numerical acceptance remains INSUFFICIENT_EVIDENCE.
No fifth allocation, platform control, new native run, copy worker or metrics
comparison was performed in this checkpoint.

## Environment and execution

Worktree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch
`feat/glm53-dsv4-vllm028`, starting HEAD `8f68960d`. Date November 4, 2026 HKT
(the evidence timestamps use November 3 UTC). Lead audit uses system Python
3.10.6; builder/check-only uses `/data/ycfeng/envs/openmopd-py312/bin/python`
3.12.13, with `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp`.

Pi monitoring label `p64-formal4-monitor-1103f`, turn5, session
`7bdd2659-fd5e-4a5f-9c58-e109958279a6`, ended exit0/settled. Its exact-job personal
StepMind reads and two SSH cloud reads are retained in
`h200_mainline/p2_64_formal_retry04/terminal_20261104/`. All raw queries came
through `/ds4-exec`; the lead independently read/recomputed saved evidence.
Pi preparation label `p64-formal5-prepare-1104b`, turn1, session
`16269a79-261d-435e-9dc2-232119ec186b`, model
`custom/deepseek-v4.1-flash-sys`, thinking high, ended exit0/settled.
Prompt: `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal5-prepare-1104b.md`.

Exact lead commands, both exit0:

```bash
python3 /data/ycfeng/tmp/glm53_dsv4/h200/audit_formal04_terminal.py
python3 /data/ycfeng/tmp/glm53_dsv4/h200/audit_formal05_package.py
```

Their source and stdout are retained as `terminal_20261104/lead_terminal_audit.{py,log}`
and `p2_64_formal_retry05/lead_package_audit.{py,log}`. JSON acceptance records
include the source SHA256 and exact interpreter version. The package audit also
independently executes the following command (exit0,
`INPUT_HASHES_PASS_NO_SUBMISSION`):

```bash
/data/ycfeng/envs/openmopd-py312/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/p2_64_formal_retry05/package_01/launch.py --check-only
```

The original build command, from `h200_mainline/`, was:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/openmopd-py312/bin/python -B build_package.py p2_64_formal_retry05 --package package_01 --harness /data/ycfeng/Frontier/.worktrees/p2-64-startup-harness
```

Build output: `p2_64_formal_retry05/package_01_build.json` and
`executor_build.log`; empty stderr. Existing builder preflight reconstructs both
H200 TP8/DP4/EP32/PP1 roles and checks the launch plan. The existing R5 153 reader
checks, 25 tool checks, legacy PASS_EXACT and completed Frontier replay were
reused, not rerun.

## Expected versus observed

| Check | Passing condition | Observed | Result |
| --- | --- | --- | --- |
| Terminal identity | Exact job/UID/creator and matching status | exp-1103-131952-259161 / 9d02faf7-621e-4368-ae73-4110312c6ae2 / i-fengyicheng; Stopped | PASS |
| Stop initiator | Matching annotation and successful audit timestamp | alimasiyilixiati, release_quota, 2026-11-03T10:00:43Z | PASS |
| Audit/event completeness | Every returned page counted, no missing audit receipt | 178/178 unique audit rows on 2 pages; 65/65 historical QueueWaiting events | PASS |
| Launcher chronology | Platform stop precedes launcher observation and cleanup | 10:00:43Z < 10:00:46.504Z observation < 10:00:47.086386Z exit0 < 10:00:47.218Z cleanup; PID81539 absent | PASS |
| Output availability | Review actual query results | Exact cloud path DIR_ABSENT; same-date glob absent; both SSH tool receipts retained, exit0 | PASS, no reference |
| Accounting | Do not convert missing clocks to exact zero | exact null, known subtotal 0.0, clocks incomplete; queue 16,851 s excluded | PASS within limits |
| Sealed candidate | All 47 file hashes correct | 47/47 PASS; manifest 23e26da6c02cc42e54134c12abfdfea298f64191e0f985a860505e374278227c | PASS |
| Candidate scope | Identical workload/code; exact identity replacements only | 41 files byte-identical; six equal after declared path/tag/purpose/build-time/overlay-digest replacements | PASS |
| Source/workload pins | Match accepted frozen inputs | Native a06a72a4 and harness f33306d4 clean; promotion d505eece unchanged; 3x128 at 6144/1024, QPS1, 16 pair warmups/repeat | PASS |
| Resources | 64H200/all-preemptible/step_main/3600s | 8x8 H200; all fields preserved; maximum 64 container GPUh | PASS |
| No submission | No approval/launcher/run marker; check-only succeeds | submitted=false; no approval_receipt.json, launcher_started.json, launcher_exit.json or case/runs | PASS |
| Pair coverage/R5/new metrics | Complete new reference required | No native output; copy, coverage, R5, metrics and warm reading NOT RUN | NOT RUN |

## Findings and limits

Null job startTime, no retained replica objects, complete available queue-event
history and the absent output jointly support the inference that generation04
never started a container. They do not create complete container clocks; the
exact receipt remains null. Historical 35-job exact accounting stays
158.27166666666665 GPU-hours. Generation01/02 missing end clocks remain open;
no further investigation is justified by this candidate preparation.

The native runtime did not get an execution opportunity in generation04.
No new native/code failure is established. An external release_quota stop is
established; a preemption classification is not asserted. Queue time is 4.680833
hours and is not GPU consumption. The 24.3-GPU-hour estimate for a future run is
historical and unvalidated for pair warmup. Fifth-submission approval is absent.

Lead small direct documentation adjustment: added the required modification
history to Pi's preparation report and corrected its description of the old
purpose text. No sealed package file changed. Initial local `python` lookup
failed (command not found), so the lead used python3. A directory-size print
briefly called integer st_size as a function and was corrected. Neither local
inspection error affected the verification scripts or package.

check-progress found only stale current-state text (S5). Updated plan, progress,
summary and tickets18/19; retained historical queue snapshots and changed no
acceptance criteria. Its next route is the reviewed fifth-candidate decision,
then the existing run/terminal/copy/coverage/R5/metric sequence if approved.

Final local record checks: all35JSON files in the new terminal/candidate trees parse;
three authored Python sources parse; retained post-run script/interpreters/case/Frontier
replay paths exist; all five current state pointers agree. `git diff --check` PASS
for tracked authored edits. No new package execution was needed for these record checks.

Full staged whitespace check reports three trailing-space lines in the unchanged
sealed upstream `package_01/native_config_parallel.py` (228,271,274). The source
is byte-identical to generation04 and remains frozen. Authored-file staged
whitespace validation passes when that preserved source copy is excluded.

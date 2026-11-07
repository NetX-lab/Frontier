# Test report: P2-64 generation05 terminal cause and generation06 submission (2026-11-07)

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-07 | Generation06 later failed at device admission; its terminal audit and generation07 are in [the next report](test_report_2026-11-07_p2_64_generation06_terminal_generation07_submission.md). |
| 2026-11-07 | Created: generation05 terminal evidence, cause classification and GPU-hour accounting; generation06 package audit, submission and monitor start. |

## Scope

Two steps under D-P2-64-FIFTH (requirements.md):

1. Establish why generation05 (`exp-1107-135759-122907`) ended, and count its GPU hours.
2. Because the cause is a platform cause, build and submit the identity-only generation06 package (platform-cause
   resubmission 1 of 2).

It covers no native result and no metric comparison: neither job has a complete native reference yet.

## Environment

- Pi on HOST, model `custom/deepseek-v4.1-flash-sys`, thinking `high`, through
  `bash /data/ycfeng/codex-home/skills/ds4-exec/scripts/run_ds4.sh <label> /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 <prompt>`.
- Platform access: personal i-fengyicheng StepMind Python RJobBackend, `/data/ycfeng/tmp/stepmind-env/bin/python`
  (Python 3.10), environment from `h200_mainline/stepmind_env.sh`.
- Package build and check-only: `/data/ycfeng/envs/openmopd-py312/bin/python`. Lead audits: system `python3` 3.10.6.

| Label | Turns | Purpose | Prompt |
| --- | --- | --- | --- |
| `p64-formal5-terminal-1107i` | 3 | terminal receipt, one CPU-only cloud copy, platform events and audit | `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal5-terminal-1107i{,.2,.3}.md` |
| `p64-formal5-logs-1107h` | 1 | per-replica platform logs | `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal5-logs-1107h.md` |
| `p64-formal6-submit-1107j` | 2 | reader smoke test, one submission, readback | `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal6-submit-1107j{,.2}.md` |
| `p64-formal6-monitor-1107k` | 3 | launcher check and monitor start | `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal6-monitor-1107k{,.2,.3}.md` |

All four labels hit the provider stream defect I-010 (`unfinished tool call`) after a tool call. Each was resumed with
the same label and a prompt that continued from the last completed step; the lead checked the state before every
resume.

## Part 1: generation05 terminal cause

Commands (Pi, one tool call each):

- `/data/ycfeng/tmp/stepmind-env/bin/python h200_mainline/cloud_copy.py submit h200_mainline/p2_64_formal_retry05/package_01/config.json /data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261104_05 <cloud run dir>`
  (CPU-only copy job `exp-1107-142442-714621`).
- `/data/ycfeng/tmp/stepmind-env/bin/python h200_mainline/job_events.py exp-1107-135759-122907 a2d12f21-f384-411e-9dc4-f596e0f3e86a <scratch>/events`
  (new read-only script; live and historical events, audit log without an operation filter).

Lead audit: `python3 h200_mainline/p2_64_formal_retry05/terminal_20261107/lead_terminal_audit.py` copies 68 evidence
files with sha256 (`archive_manifest.json`) and writes `lead_terminal_acceptance.json`.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Job identity and phase | creator i-fengyicheng, phase Failed | as expected, finish 06:16:02Z | PASS |
| User operations in the audit | only the create | 1 row: create at 05:57:59Z | PASS |
| Stop operations on 8f14e45f, a87ff679, c81e728d | none (stopped from outside) | none; phase Stopped | PASS |
| Stops of the other replicas | by rjob-controller | all by rjob-controller at 06:16:02-03Z | PASS |
| Failed containers | 2, exit 1 | e4da3b7f (0876) and eccbc87e (0790), exit 1 | PASS |
| Launcher exit after job end | later than 06:16:02Z | 06:16:04Z, RuntimeError, exit 1 | PASS |
| Cloud copy | `HASH_IDENTICAL_DOWNLOAD` | 37 files, 371,893 bytes | PASS |
| Event and audit queries | 0 failed | 0 failed (query_log.json) | PASS |

Observed sequence (node logs in the copy):

- 06:07:00Z: the server on gpu-h200-0924 (8f14e45f, decode lane 3) starts. Its `server.log` stays empty, and the
  replica is Stopped by 06:09:59Z.
- 06:10:34Z: gpu-h200-0196 (a87ff679, prefill lane 1) shuts down cleanly.
- 06:15:56Z: gpu-h200-0505 (c81e728d, decode DP head) shuts down cleanly and writes no `failed.json`.
- Effect on the remaining lanes:
  - Decode lanes DP1 and DP2 (0790, 0876) lose the TCPStore on the DP head. Their vLLM exits 0, the worker raises
    `RuntimeError: Native server exited early: 0`, and the containers exit 1.
  - The platform's any-replica fail policy then stops the job at 06:16:02Z.
- Queue-controller updates of the job object at 06:07:04Z, 06:07:19Z, 06:10:40Z and 06:15:59Z coincide with these
  stops. The job is preemptible (spot GPU), and its scheduling events show waits for resources to be preempted.

Classification (inference): platform-side external stops of three replicas, so a platform cause under the
D-P2-64-FIFTH resubmission rule. No event names preemption explicitly. The stops are not a native or code failure:
the three replicas show no native error, and the two failures follow the loss of their DP head.

GPU-hour accounting:

- The exact container clocks are incomplete. The two Failed containers give 2.587 GPUh known.
- The counted upper bound is 64 GPUs x 1042 s, from the first replica create (05:58:41Z) to the last stop
  (06:16:03Z): 18.52 GPUh.
- Cumulative under D-P2-64-FIFTH is 18.52 of 128.

Not run for generation05: pair coverage, R5 and the metric readings, because no native reference exists.

## Part 2: generation06 package and submission

Build: one entry added through the unchanged CASES registry of `h200_mainline/build_package.py`
(`p2_64_formal_retry06`, run tag `dsv4-h200-p2-64-formal-20261107-06`), then
`build_package.py p2_64_formal_retry06 --package package_01 --harness /data/ycfeng/Frontier/.worktrees/p2-64-startup-harness`
(exit 0).

Lead audit: `python3 h200_mainline/p2_64_formal_retry06/lead_package_audit.py` compares generation06 with generation05.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Sealed files | 47, hashes match | 47 | PASS |
| Byte-identical files | 41 | 41 | PASS |
| Changed files | the 6 identity files, exact identity replacements only | as expected | PASS |
| Pins | harness f33306d4, native a06a72a4, both clean | as expected | PASS |
| check-only | `INPUT_HASHES_PASS_NO_SUBMISSION`, rc 0 | as expected | PASS |
| Generation05 manifest | unchanged 23e26da6… | unchanged | PASS |

Generation06 manifest: `d70352ad05902eae58fcaea5f0f2709523675f16e3b2ad9830eb00a4bd4480fb`.

Supporting records:

- Approval receipt `h200_mainline/p2_64_formal_retry06/approval_receipt.json`: resubmission 1 of 2, same resources as
  generation05, 18.52 GPUh counted before this job.
- Wrapper `/data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry06.sh`: it differs from the generation05
  wrapper only in `AUTHORIZATION` and `PKG`, and passes `bash -n`.
- Post-run command `h200_mainline/p2_64_formal_retry06/post_run_command.json`, pointing at the generation06 download,
  controller run and output paths.

Submission commands (Pi `p64-formal6-submit-1107j`):

1. `bash h200_mainline/read_job.sh exp-1107-135759-122907 <scratch>/reader_smoke_generation05.json`
2. `bash h200_mainline/submit_package.sh h200_mainline/p2_64_formal_retry06/package_01 /data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry06.sh /data/ycfeng/tmp/ds4_runs/p64-formal6-submit-1107j/scratch`

Monitor (Pi `p64-formal6-monitor-1107k`, turn 3, one combined call):
`ps -p 3297045` followed by a detached `bash h200_mainline/monitor_job.sh exp-1107-144321-305543 3297045 /data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry06.sh /data/ycfeng/tmp/ds4_runs/p64-formal6-monitor-1107k/scratch/monitor 300 36`.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Reader smoke test | generation05 phase Failed, 0 errors | Failed, 0 errors, 8 replicas | PASS |
| Approval and seal | 9 checks true | 9/9, manifest d70352ad… | PASS |
| Launch | exactly one detached start | PID 3297045 at 06:43:17Z, one `Job Create` line | PASS |
| Job identity and spec | 18 checks true | 18/18 (`SUBMITTED_ALL_CHECKS_PASS`) | PASS |
| Launcher alive after the submitting Pi exited | present on HOST | PID 3297045, SID 3297045, elapsed 03:05 | PASS |
| Monitor first read | launcher_alive true, read_error null | as expected; phase Starting, 8/8 replicas Starting, queue approved | PASS |

Observed job values:

- Job `exp-1107-144321-305543`, UID `ceaa0653-8a7b-424b-b3e0-9c5bb52b7abd`, created 2026-11-07T06:43:21Z.
- Creator i-fengyicheng; step_main; 8 replicas x 8 H200; preemptible yes; command timeout 3600 s.

Evidence:

- Copies with sha256: `h200_mainline/p2_64_formal_retry06/generation06_*`.
- Acceptance record: `h200_mainline/p2_64_formal_retry06/lead_submission_acceptance.json`, status
  `PASS_SUBMISSION_IDENTITY_AND_LAUNCHER_SURVIVAL`.

## Limits

- The generation05 cause is an inference from phases, the audit and controller timing; no platform record names
  preemption.
- Not yet observed for generation06: worker `nvidia-smi`, API readiness, native completion, terminal status, container
  clocks.
- The lead's shell cannot see HOST processes; launcher and monitor liveness come from HOST-side reads by Pi and by the
  monitor's `/proc` check.
- I-010 is worked around, not fixed.

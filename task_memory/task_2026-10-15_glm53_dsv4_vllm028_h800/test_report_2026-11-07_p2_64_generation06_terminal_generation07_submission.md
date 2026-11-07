# Test report: P2-64 generation06 terminal cause and generation07 submission (2026-11-07)

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-07 | Created: generation06 terminal evidence, cause classification and GPU-hour accounting; generation07 package audit, submission and monitor start. |

## Scope

Two steps under D-P2-64-FIFTH (requirements.md), continuing
[the generation05/06 report](test_report_2026-11-07_p2_64_generation05_terminal_generation06_submission.md):

1. Establish why generation06 (`exp-1107-144321-305543`) ended, and count its GPU hours.
2. Because the cause is a platform cause, build and submit the identity-only generation07 package. This is
   platform-cause resubmission 2 of 2, the last one the decision allows.

It covers no native result and no metric comparison: no job has a complete native reference yet.

## Environment

- Pi on HOST, model `custom/deepseek-v4.1-flash-sys`, thinking `high`, through
  `bash /data/ycfeng/codex-home/skills/ds4-exec/scripts/run_ds4.sh <label> /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 <prompt> [read-only]`.
- Platform access: personal i-fengyicheng StepMind Python RJobBackend, `/data/ycfeng/tmp/stepmind-env/bin/python`
  (Python 3.10), environment from `h200_mainline/stepmind_env.sh`.
- Package build and check-only: `/data/ycfeng/envs/openmopd-py312/bin/python`. Lead audits: system `python3` 3.10.6.

| Label | Turns | Purpose | Prompt |
| --- | --- | --- | --- |
| `p64-formal6-terminal-1107l` (read-only) | 1 | terminal receipt, platform events and audit, replica logs | `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal6-terminal-1107l.md` |
| `p64-formal7-submit-1107m` | 1 | reader read of generation06, one submission, readback, monitor start | `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal7-submit-1107m.md` |

Both turns used a single tool call, so the provider stream defect I-010 did not interrupt them. The generation07
steps run from the lead-written script `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal7-submit-1107m.sh`.

## Part 1: generation06 terminal cause

Pi commands (one combined call): `h200_mainline/terminal_receipt.py`,
`h200_mainline/job_events.py exp-1107-144321-305543 ceaa0653-8a7b-424b-b3e0-9c5bb52b7abd <scratch>/events` and
`h200_mainline/replica_logs.py`.

Lead audit: `python3 h200_mainline/p2_64_formal_retry06/terminal_20261107/lead_terminal_audit.py` copies 27 evidence
files with sha256 (`archive_manifest.json`) and writes `lead_terminal_acceptance.json` =
`PASS_TERMINAL_PLATFORM_CAUSE_NO_REFERENCE`.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Job identity and phase | creator i-fengyicheng, Failed | Failed at 06:48:31Z, `startTime` null | PASS |
| System stop reason | a platform stop | `Some critical tasks failed.` | PASS |
| User operations in the audit | only the create | 1 row: create at 06:43:21Z | PASS |
| Admission error | on the failed replica only | `UnexpectedAdmissionError` on 8f14e45f (gpu-h200-0757) at 06:48:30Z | PASS |
| Failed replica | 8f14e45f only, without a container | as expected | PASS |
| Containers started | none | none running or terminated; 8/8 replica logs empty | PASS |
| Stops of the replicas | by rjob-controller, after the admission error | all by rjob-controller, 06:48:31-47Z | PASS |
| Launcher exit after job end | later than 06:48:31Z | 06:48:33Z, RuntimeError, exit 1 | PASS |
| Event and audit queries | all pass | all PASS (query_log.json) | PASS |

Observed sequence (platform events):

- 06:43:55-56Z: the queue approves the job and all 8 replicas start. The pods wait for the full group of 8 to be
  scheduled together, then wait on their assigned nodes for spot GPUs to be released by preemption.
- 06:48:30Z: the kubelet on gpu-h200-0757 rejects replica 8f14e45f:
  `Allocate failed due to requested number of devices unavailable for nvidia.com/spot-gpu. Requested: 8, Available: 7, which is unexpected`.
- 06:48:31-32Z: rjob-controller closes the other 7 replicas.
- Teardown on the closing pods follows (juicefs `FailedMount`, an image pull that finishes at 06:51:57Z, then
  `configmaps ... not found`).

Classification: platform cause. The platform admitted a pod to a node whose spot GPUs were not yet free. No container
started and no native code ran, so this is not a native or code failure.

GPU-hour accounting:

- Known container GPU hours: 0.0 (no container started); container clocks are incomplete.
- Counted upper bound: 64 GPUs x 482 s, from the first replica create (06:43:55Z) to the last replica event
  (06:51:57Z, the image pull on a pod being torn down). The bound is later than the last stop (06:48:47Z), so it
  over-counts rather than under-counts. Result: 8.57 GPUh.
- Cumulative under D-P2-64-FIFTH: 18.52 + 8.57 = 27.09 of 128.

Not run for generation06: cloud copy (no output exists), pair coverage, R5 and the metric readings.

## Part 2: generation07 package and submission

Build: one entry added through the unchanged CASES registry of `h200_mainline/build_package.py`
(`p2_64_formal_retry07`, run tag `dsv4-h200-p2-64-formal-20261107-07`), then
`build_package.py p2_64_formal_retry07 --package package_01 --harness /data/ycfeng/Frontier/.worktrees/p2-64-startup-harness`
(exit 0, check-only `INPUT_HASHES_PASS_NO_SUBMISSION`).

Lead audit: `python3 h200_mainline/p2_64_formal_retry07/lead_package_audit.py` compares generation07 with generation06.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Sealed files | 47, hashes match | 47 | PASS |
| Byte-identical files | 41 | 41 | PASS |
| Changed files | the 6 identity files, exact identity replacements only | as expected | PASS |
| Pins | harness f33306d4, native a06a72a4, both clean | as expected | PASS |
| check-only | `INPUT_HASHES_PASS_NO_SUBMISSION`, rc 0 | as expected | PASS |
| Generation06 manifest | unchanged d70352ad… | unchanged | PASS |

Generation07 manifest: `8fd5d9dc625bae451689775bfdcd578e3c7b60be6d0b7b08c0edbcc00703190a`.

Supporting records:

- Approval receipt `h200_mainline/p2_64_formal_retry07/approval_receipt.json`: resubmission 2 of 2, same resources
  as generation06, 27.09 GPUh counted before this job; any further GPU job needs a new user decision.
- Wrapper `/data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry07.sh`: it differs from the generation06
  wrapper only in `AUTHORIZATION` and `PKG`, and passes `bash -n`.
- Post-run command `h200_mainline/p2_64_formal_retry07/post_run_command.json`, pointing at the generation07
  download, controller run and output paths.

Submission (Pi `p64-formal7-submit-1107m`, one call):
`bash /data/ycfeng/tmp/ds4_runs/prompts/p64-formal7-submit-1107m.sh`. The script runs, in order:

1. `bash h200_mainline/read_job.sh exp-1107-144321-305543 <scratch>/reader_smoke_generation06.json`, and stops unless
   the phase is Failed or Stopped with 0 errors;
2. `bash h200_mainline/submit_package.sh h200_mainline/p2_64_formal_retry07/package_01 /data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry07.sh <scratch>`;
3. a detached `bash h200_mainline/monitor_job.sh <job> <launcher pid> <wrapper> <scratch>/monitor 300 36`.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Reader read of generation06 | phase Failed, 0 errors | Failed, 0 errors | PASS |
| Approval and seal | 9 checks true | 9/9, manifest 8fd5d9dc… | PASS |
| Launch | exactly one detached start | PID 3498815 at 07:01:28Z, one `Job Create` line | PASS |
| Job identity and spec | 18 checks true | 18/18 (`SUBMITTED_ALL_CHECKS_PASS`), 0 controls issued | PASS |
| Launcher alive after the Pi turn ended (07:02:58Z) | monitor reads `launcher_alive` true | true at 07:06:43Z | PASS |
| Monitor reads | `read_error` null | null at 07:01:42Z and 07:06:43Z | PASS |

Observed job values:

- Job `exp-1107-150132-396145`, UID `45849203-b2c0-4de5-9ef3-e3b4ad716168`, created 2026-11-07T07:01:32Z.
- Creator i-fengyicheng; step_main; 8 replicas x 8 H200; preemptible yes; command timeout 3600 s.
- Phase Pending, queue status `suspend`: `Insufficient GPU quota ... 64 GPU (H200)`. This is a queue wait, not a
  failure.

Evidence:

- Copies with sha256: `h200_mainline/p2_64_formal_retry07/generation07_*`.
- Acceptance record: `h200_mainline/p2_64_formal_retry07/lead_submission_acceptance.json`, status
  `PASS_SUBMISSION_IDENTITY_AND_LAUNCHER_SURVIVAL`.

## Limits

- Pi's own report on the terminal read said the events contain no failure message; the lead's reading of
  `historical_events.json` found the `UnexpectedAdmissionError` event. The classification rests on the lead's reading.
- Not yet observed for generation07: scheduling, worker `nvidia-smi`, API readiness, native completion, terminal
  status, container clocks.
- The lead's shell cannot see HOST processes; launcher and monitor liveness come from the monitor's `/proc` check on
  HOST.
- Generation07 is the last resubmission under D-P2-64-FIFTH. If it also ends without a reference, the chain stops and
  the next GPU job needs a new user decision.

# Test report: P2-64 generation05 submission and monitor start (2026-11-07)

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-07 | Created: generation05 submission checks, launcher survival and monitor start. |

## Scope

This report covers one approved submission under D-P2-64-FIFTH (requirements.md): package
`h200_mainline/p2_64_formal_retry05/package_01`, manifest
`23e26da6c02cc42e54134c12abfdfea298f64191e0f985a860505e374278227c`, wrapper
`/data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry05.sh`. It covers no native result, metric or GPU-hour
reading.

## Environment

- Pi on HOST, model `custom/deepseek-v4.1-flash-sys`, thinking `high`, through
  `bash /data/ycfeng/codex-home/skills/ds4-exec/scripts/run_ds4.sh <label> /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 <prompt>`.
- Submission label `p64-formal5-submit-1107e`, turn 1, session `ad4896d0-0bb9-4e11-98c9-920365799664`, prompt
  `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal5-submit-1107e.md`.
- Monitor label `p64-formal5-monitor-1107f`, turn 1, session `6dc31a05-df87-4149-8525-14fc951b7e13`, prompt
  `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal5-monitor-1107f.md`.
- Platform access: personal i-fengyicheng StepMind Python RJobBackend, `/data/ycfeng/tmp/stepmind-env/bin/python`
  (Python 3.10); check-only with `/data/ycfeng/envs/openmopd-py312/bin/python` without `PYTHONPATH`.
- Scripts: `h200_mainline/submit_package.py`, `h200_mainline/read_job.py`, `h200_mainline/monitor_job.py`, with
  wrappers `submit_package.sh`, `read_job.sh`, `monitor_job.sh` and the shared `stepmind_env.sh`. At submission time
  `submit_package.sh` carried the same environment lines inline; they moved into `stepmind_env.sh` at 05:59:28Z,
  after the launch.

## Commands (run by Pi on HOST, one tool call each)

1. `bash h200_mainline/read_job.sh exp-1103-131952-259161 <scratch>/reader_smoke_generation04.json` (reader smoke
   test on the stopped generation04 job).
2. `bash h200_mainline/submit_package.sh h200_mainline/p2_64_formal_retry05/package_01 /data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry05.sh /data/ycfeng/tmp/ds4_runs/p64-formal5-submit-1107e/scratch`
3. `ps -o pid,ppid,sid,etime,args -p 2753825` (monitor label), then a detached
   `bash h200_mainline/monitor_job.sh exp-1107-135759-122907 2753825 /data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry05.sh /data/ycfeng/tmp/ds4_runs/p64-formal5-monitor-1107f/scratch/monitor 300 36`.

## Results

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Reader smoke test | `phase Stopped errors 0` on exp-1103-131952-259161 | as expected | PASS |
| Approval and seal | 9 approval checks true; manifest equals approval | 9/9; `23e26da6…` | PASS |
| Launch-evidence guard (HOST /proc) | no markers, no live retry05 launcher or wrapper | none found | PASS |
| check-only | `INPUT_HASHES_PASS_NO_SUBMISSION`, rc 0 | as expected | PASS |
| Quota snapshot | rc 0 (not a veto) | rc 0 | PASS |
| Launch | exactly one detached start | PID 2753825 at 05:57:54.570258Z, one `Job Create` line | PASS |
| Job identity and spec | 18 checks true | 18/18 (`SUBMITTED_ALL_CHECKS_PASS`) | PASS |
| Launcher alive after the submitting Pi exited | PID 2753825 present on HOST | present, SID 2753825, elapsed 01:58, argv `.../p2_64_formal_retry05/package_01/launch.py` | PASS |
| Monitor running after its Pi exited | new monitor.jsonl lines every 300 s | lines at 05:59:55Z and 06:04:57Z | PASS |

Observed job values: `exp-1107-135759-122907`, UID `a2d12f21-f384-411e-9dc4-f596e0f3e86a`, created
2026-11-07T05:57:59Z; creator i-fengyicheng; 8 replicas, `nvidia.com/gpu` 8, cpu 64, memory 500Gi; positive tag H200;
preemptible yes; quota group step_main; queue quotagroup-global-default; restartable false, restartPolicy Never;
command timeout 3600 s; image `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e853…`; RDMA mlnx_rdma 1 and
mlnx_shared 8; juicefs includes `codesign-exp`; NFS source 100.96.128.195 (a current-host address), paths equal to the
package. Phase Pending at the submission read; Starting with 8/8 replicas Starting at 05:59:55Z and 06:04:57Z; queue
status approved; zero read errors.

Evidence (sha256 in `h200_mainline/p2_64_formal_retry05/lead_submission_acceptance.json`):
`generation05_submission_acceptance.json`, `generation05_submission_identity.json`, `generation05_launch_record.json`,
`generation05_quota_snapshot.json`, `generation05_check_only.txt`, `generation05_launcher_log_at_submission.txt`,
`generation05_reader_smoke_generation04.json`, `generation05_monitor_first_read.json`.

## Limits

- Observation: the job is admitted and its replicas are starting. Not yet observed: worker `nvidia-smi`, API
  readiness, native completion, terminal status and container clocks.
- The lead's own shell cannot see HOST processes; launcher and monitor liveness come from HOST-side reads by Pi and
  by the monitor's `/proc` check.
- The provider stream defect (issues.md I-010) is worked around, not fixed.

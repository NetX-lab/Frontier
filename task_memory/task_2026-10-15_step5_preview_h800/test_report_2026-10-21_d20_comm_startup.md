# H200 communication startup failure and bounded correction

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Accepted6a9cedc asb68ce4d after101 parent passes; verified189 pins and resubmitted the unchanged scenario under96/130 reservations. |
| 2026-10-21 | Preserved the failed platform/container receipts and reproduced startup SIGPIPE through the actual worker script on CPU. Correction/restaging assigned without interrupting compute/PD implementation. |

## Observed failure

Job `exp-1021-114449-837730` requested32 H200, TP8/DP4/EP32, first-prefill physical tokens `[6144,1,1,1]`. It waited from03:44UTC until about06:20UTC and finishedFailed at06:20:13UTC. This was not a timing-quality rejection.

Personal Python RJobClient inspection at06:21:50UTC returned one failed and three stopped replicas. The failed replica `eccbc87e`, platform NODE_RANK2 on gpu-h200-0819, ran06:20:06–06:20:08UTC and exited141 without restart. The other two not-yet-started containers still reportedContainerCreating; node0 was markedStopped while its last container state still reportedRunning. Only the failed container has verified terminal exit evidence. The platform records `Task fail policy [any] caused undesired running replicas.` The launcher's subsequent stop message is the SDK exit handler after failure, not a parent-issued stop.

The failed node's platform log reaches `Ready to go` and thenExitCode141, without `STEP5_COMM_WORKER_LOG`. Node0 reaches that worker marker and is subsequently terminated by the platform policy. Both started containers use the frozen image69308454…a7f20. No qualified complete30-row communication result is established.

## Exact red-capable CPU reproduction

The committed `tests/integration/step5_comm/run_on_worker.sh:6` runs this command under `set -euo pipefail`:

```bash
major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)
```

Parent executed the actual script with a paced eight-row `nvidia-smi` test double:

```bash
bash -c 'nvidia-smi() { printf "570.124.06\n"; sleep 0.05; for i in {2..8}; do printf "570.124.06\n"; done; }
export -f nvidia-smi
bash -x /data/ycfeng/Frontier/.worktrees/step5-comm/tests/integration/step5_comm/run_on_worker.sh /data/ycfeng/tmp/d20_comm_failure01/unused.json > /data/ycfeng/tmp/d20_comm_failure01/driver_pipe_repro01.log 2>&1
rc=$?
printf "Actual worker script with paced eight-GPU driver output: exit=%s (expected: not 141)\n" "$rc"
test "$rc" -ne 141'
```

Actual: `exit=141`, assertion exit1. The worker exits during the driver query before reaching Python or the fixture config path. This deterministically reproduces the observed exit class: `head` closes its input before all GPU rows are written; the writer receivesSIGPIPE and `pipefail` propagates141. No GPU, native imports or additional profiler is needed for this feedback loop. Bash runs on the local CPU host; the stub controls output scheduling only.

Broad bisection/instrumentation is unnecessary for this bounded correction: the actual caller reproduces the exact startup failure at its sole pre-worker pipe. The GPU log does not contain a shell trace, so the local reproduction and matching startup boundary are the causal evidence; it does not independently prove every healthy-node runtime path.

## Scope and next acceptance

The communication owner fixes the query to consume its complete output while preserving strict shell options, real nvidia-smi failures,570/580 driver policy and platform NCCL variables. Acceptance is the real-script paced-output regression, both supported driver branches, genuine producer-failure propagation and existing affected tests. Seal a new02 run/config/output path after committing; keep old01 and its receipts unchanged. Parent owns fresh reservation/quota verification and any subsequent submission.

The running/queued64-H200 qualification package was separately searched for `nvidia-smi`, `driver_version` and `head -n`: it uses Python subprocess capture and has no matching short-read driver pipeline. It is not changed or cancelled by this repair. Compute/bundle and measured PD remain the two main implementation streams.

## Evidence and limits

Sealed startup-script SHA256: `b77020c01333a42ac732d13de5f21412788fecb3ce253b7a7b69776f863574e3`.

Six filtered files are retained at `scratch_evidence/d20_comm_startup/`: platform.json, four replica console logs and driver_pipe_repro01.log. The raw launcher is `/data/ycfeng/tmp/d18_h200_first_prefill_20261021_01_launcher.log`; selected lines were inspected with personal credentials redacted. Platform reads used the runbook's personal Python API and `_get_replica_kapis`, selecting status fields only.

Initial result: CPU reproducerRED; correction/restaging were pending. The following acceptance supersedes this pending status; the original failure remains unchanged.

## Accepted correction and new submission

Commit `6a9cedc7e468fee396ca5c2a9c5987bab76dd9ac`, merged as `b68ce4d`, replaces only `head -n1` with full-consuming `sed -n '1p'`. Parent inspected both changed files. The existing actual-wrapper test now verifies eight paced driver rows,570/580/590 handling, unchanged NCCL values, unsupported550 rejection and genuine producer exit9 before/after a valid first row. No retry, error suppression or new CLI is introduced.

Parent ran from the integrated step5-preview tree, Python3.10.6:

```bash
PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_comm_h200.py tests/unit/test_step5_comm_package.py tests/unit/test_step5_comm_multinode.py -q -p no:cacheprovider
```

Actual: **101 passed in13.47s**, including all seven startup cases. Author's recorded actual-caller RED is6 failed/1 passed, GREEN7 passed. Simulator sources are unchanged; the next whole-simulator fidelity comparison remains the integrated D20 compute/PD acceptance.

Parent independently checked189/189 source pins, unchanged resources, and equality of every benchmark field except output_dir. The sole changed source pin is the wrapper: `25762505b389ca99bbe160ffca8e4b8204acca922f444089a2d79dcafb09c832`. Author verifies all six old01 files unchanged. Full old/new identities and test logs: `/data/ycfeng/Frontier/.worktrees/step5-comm/task_memory/step5_comm/d20_comm_startup_20261021/test_report_2026-10-21_comm_startup.md`.

Fresh personal quota read06:36:38–06:36:41UTC used the exact creator selector, handled continuation and completed in one page. Exact replicas×nvidia.com/gpu limits include queued reservations: only qualification `exp-1021-125354-923219`,64 GPUs; new32 gives96<=130. Cluster non-preemptible quota was506/512, so placement may queue. Receipt: `scratch_evidence/d20_comm_startup/d20_comm02_preflight.json`. An initial local quoting error prevented the first query from running; corrected stdin quoting succeeded before submission.

Submitted06:37:08UTC as **`exp-1021-143708-523626`** through the existing local personal Python launcher:

```bash
setsid bash /data/ycfeng/Frontier/.worktrees/step5-comm/task_memory/step5_comm/d20_comm_startup_20261021/h200_first_prefill_02/submit_with_env.sh > /data/ycfeng/tmp/d20_h200_first_prefill_20261021_02_launcher.log 2>&1
```

06:38:10UTC platform receipt: Pending, creator i-fengyicheng, non-preemptible,4×8 GPUs, both RDMA requests, pinned image69308454…a7f20 and local NFS `100.96.128.195:/data/ycfeng/Frontier/.worktrees/step5-comm`. See `scratch_evidence/d20_comm_startup/resubmission02.json`. Python launcher PID3800860 remained alive; the setsid wrapper's immediate exit is not workload completion. Blocking pidfd watchers now observe the existing communication and qualification launchers without polling or controlling them.

Output: `/mnt/codesign-exp/ycfeng/frontier/step5_comm/d20_h200_first_prefill_20261021_02`. Expected30 observations/10 kernel rows,15-minute/8-GPU-hour plan and30-minute native timeout remain unchanged. CPU startup repairPASS; native startup, complete32-rank execution, timing quality and formal admission remain pending.

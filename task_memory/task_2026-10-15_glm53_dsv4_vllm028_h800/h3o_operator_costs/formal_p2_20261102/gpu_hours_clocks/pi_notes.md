# Read-only platform clocks for the 19 finished H200 RJobs

Task: `gpuh-clocks-1102a`. Read-only query of the platform's own records; no job was
submitted, stopped, or changed, and no platform write of any kind was made.

## Environment (handbook section 2, same shell as every query)

```
source scratch/env_init.sh      # proxy env + personal i-fengyicheng keys + these:
export STEPMIND_BACKEND=rjob
export PYTHONPATH=/data/ycfeng/steptron
export KUBEBRAIN_CLUSTER_ENTRY=https://platform.shaipower.com
export KUBEBRAIN_NAMESPACE=shai-core
export KUBEBRAIN_WORKSPACE_NAME=kun-workspace-vgen2
/data/ycfeng/tmp/stepmind-env/bin/python   # 3.10.6, brainpp 2.8.6
```

The three `KUBEBRAIN_*` exports are required: `RJobClient._get_type()` accepts only the
kubebrain type, which needs both `KUBEBRAIN_CLUSTER_ENTRY` and `KUBEBRAIN_NAMESPACE`
(`brainpp/rjob/client.py:654-662`, raise at `:78`). The host shell has neither, so the
first step-1 attempt failed with `RuntimeError: brainpp only work in kubebrain environment`
for all 19 jobs; the retry with the three variables set is the recorded run.

## Step 1 — terminal receipts (tool unchanged)

```
bash run_receipts.sh             # timeout 120s per job, exit codes in receipt_exit_codes.txt
```

16/19 exit 0. 3/19 exit 1 because the parent RJob object itself returns HTTP 404:

```
exp-1028-024316-296035  exp-1028-031147-398843  exp-1028-132243-707076
```

A bounded listing of all 397 rjobs owned by `i-fengyicheng` in namespace `shai-core`
shows no job created before `exp-1028-153332-790276`; those three were purged from the
job collection (age/retention), not misfiled in another namespace. Their replica objects
still exist (`get_rjob_infos` returns one row each), so the container clocks were
recovered from the replica objects instead of the job object.

## Step 2 — fallbacks for the three failures

```
bash run_job_only.sh             # job object only; all three return 404 (job_only_exit_codes.txt)
source env_init.sh
"$STEPMIND_PYTHON" replica_only_receipt.py <job> receipts/<job>.replica_only.json
```

- `receipts/<job>.job_only.json`: job object only, `"basis": "job object only; no container clocks"`.
  All three record the 404 `read_error` (exception type plus API message, no credentials).
- `receipts/<job>.replica_only.json`: replica objects only, with per-container
  start/finish clocks, `gpu_limit` read from the replica pod spec, and `container_gpu_hours`.
  Basis line states the parent job object was not found. This is the value used in
  `summary.csv` for those three rows; `receipt_ok` stays `false` because no receipt exists.

## Findings

- Container clocks exist for all 19 jobs (1 container per replica, all terminated),
  total 16.861111 container GPU-hours.
- No job-level `startTime` survives on any of the 19 job objects: `status` has only
  `phase`, `reason`, `message`, `finishTime`, `taskStage`, `taskStatuses`, and
  `taskStatuses` holds only `succeeded`/`taskStage` (no timestamps). Therefore
  `job_envelope_gpu_hours` is undefined for every job and `job_start` is empty;
  container clocks are the only retained measure.
- `exp-1102-030025-971155`: the earlier HTTP 403 on a pod read no longer reproduces;
  both replicas were read and both containers carry clocks.
- `container_gpu_hours` uses the tool's contract: sum of container durations times the
  task's GPU limit. GPU limits observed: 8 for the 2- and 4-replica jobs, 1 for the
  three recovered `exp-1028-02/03/13` jobs.
- One replica of `exp-1028-024316-296035` ended `Failed` with exit code 1 (container
  ran 2026-10-27T18:46:16Z to 18:53:11Z); the job-level phase is unavailable (404).
  The other two recovered jobs have `Succeeded` replicas. Their job names use local
  time (UTC+8): e.g. `exp-1028-024316` started 2026-10-27T18:43:31Z.

## Files

- `receipts/<job>.json`: 16 step-1 receipts from `terminal_receipt.py`.
- `receipts/<job>.job_only.json`: 3 job-object-only records (404).
- `receipts/<job>.replica_only.json`: 3 replica-object records with recovered clocks.
- `receipts/<job>.step1.log`, `<job>.step2.log`: stdout/stderr of each attempt.
- `receipt_exit_codes.txt`, `job_only_exit_codes.txt`: exit codes per job.
- `summary.csv`: one row per job, columns as specified.
- Scripts: `env_init.sh`, `run_receipts.sh`, `run_job_only.sh`, `job_only_receipt.py`,
  `replica_only_receipt.py`, `build_summary.py`, `probe_missing_jobs.py`,
  `probe_replica.py`, `probe_status_keys.py`, `probe_task_status.py`.
- `sha256sums.txt`: digests of every file above (excluding itself).

## Verification

- `ls receipts | wc -l` -> 44: 16 step-1 receipts, 19 step-1 logs, 3 job-only records, 3 replica-only records, 3 step-2 logs.
- `summary.csv` -> exactly 19 data rows; every `container_gpu_hours` equals its receipt value.
- Credential scan: neither personal key value appears anywhere under `scratch/`.

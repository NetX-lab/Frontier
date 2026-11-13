# Local Capacity and Personal Cloud Access

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Add explicit cloud/cleanup approval, the root-owned parent diagnosis, actual write/read/delete PASS and retained duplicate cleanup evidence. |
| 2026-10-24 | Record timestamped local metadata, projected shortfalls, evidence limits and approval options. |

## Outcome

**BLOCKER:** neither inspected filesystem can hold the sampled261.4GB single-mode
projection. Two-mode522.8GB is an estimate,not a hard minimum or upper bound.
No capacity is reserved and no cleanup, external staging or full CLI was run.

## Commands and Actual Values

D29-STORAGE-READINESS,explore session`ses_e91ee5fefffee4jkdiQMREtWQh`,used
local memory-light metadata commands with<=60s command limits. No Python/conda,
GPU authentication, source imports, model/CSV/trace read or write was involved.
The shell path loop queries each named path independently;it does not scan data.

```bash
date -Is
for p in /data/ycfeng/tmp /data/ycfeng/Frontier /home/brainpp /mnt; do df -B1 --output=source,fstype,size,used,avail,pcent,target "$p"; done
for p in /data /data/ycfeng /data/ycfeng/tmp /data/ycfeng/Frontier /data/ycfeng/Frontier/.worktrees/step5-preview /home/brainpp /mnt /; do findmnt -T "$p" -o TARGET,SOURCE,FSTYPE,OPTIONS; done
findmnt -R /data
df -B1 -i /data /
```

| Timestamp(+08:00) | Pool | Available bytes | GB | GiB |
| --- | --- | ---: | ---: | ---: |
| 2026-10-24T21:20:14 | /data,/dev/rbd1,xfs | 230939615232 | 230.940 | 215.079 |
| 2026-10-24T21:20:14 | /,/dev/rbd0,xfs | 59512037376 | 59.512 | 55.425 |
| 2026-10-24T21:20:32 | /data,/dev/rbd1,xfs | 230937346048 | 230.937 | 215.077 |
| 2026-10-24T21:20:32 | /,/dev/rbd0,xfs | 59510960128 | 59.511 | 55.424 |

`/data/ycfeng/tmp` and`/data/ycfeng/Frontier` share/dev/rbd1;
`/home/brainpp` and`/mnt` share/dev/rbd0. Paths cannot be added as separate
capacity. Distinct pools also cannot transparently hold one output file.
No/data submount was observed. XFS mounts advertise`prjquota`;project limits
are not checked. Inode use is about6%(/data) and2%(root),not the observed blocker.
Available space changed by2,269,184B(/data) and1,077,248B(root) between readings;
the responsible activity and any long-term growth rate are unknown.

## Expectations and Comparison

Expected readiness: an approved output location has enough usable capacity and
headroom for complete outputs. **FAIL/BLOCKER** with current local metadata:

| Projection | /data available(B) | Deficit(B) | Deficit(GB) |
| --- | ---: | ---: | ---: |
| One mode261400000000B | 230937346048 | 30462653952 | 30.463 |
| Two modes522800000000B | 230937346048 | 291862653952 | 291.863 |

Projection source: parent`workload_multiplicity/report.md` and prior
`summary.md` workload/capacity section. BF16/P6128/D16 full bytes, non-wave rows
and other overhead are unmeasured. The projection is not a correctness or
impossibility proof and does not create a520GB acceptance criterion.

## Approval Options

1. Recommended: identify an approved additional output location with enough
   usable capacity/headroom. Actual external mount/upload requires permission.
2. Explicitly scoped cleanup of proven disposable targets might increase local
   space;no such targets or reclaimable count is established. Deletion requires
   permission and must preserve cited and other-session evidence.
3. Keep full CLI blocked while capacity is unresolved. Do not reduce output,
   delete one mode between runs, assume compression permission or pool filesystems.

These options describe the earlier unapproved state. YC subsequently selects
codesign OSS test storage and requires prompt cleanup of completed disposable
files. The original local capacity failure remains;the new cloud checks follow.

## Authorized Cloud Checks

The authoritative runbook prohibits directly mounting this cloud on the CPU
host and defines worker mounts through StepMind RJobBackend. Global storage
guidance provides the existing shai-ycfeng personal alias`/mnt/ws-jfs/ycfeng`.
No worker allocation or platform operation is performed for these storage checks.

Metadata command: `timeout 60s ssh -o BatchMode=yes -o ConnectTimeout=10 shai-ycfeng python3 -`
with stdin asserting`id -un == i-fengyicheng`,then`os.statvfs('/mnt/ws-jfs/ycfeng')`
and`findmnt -J -T /mnt/ws-jfs/ycfeng -o TARGET,SOURCE,FSTYPE,OPTIONS`.
Observed at2026-10-24T13:27:14.046996Z:38,994,847,006,720B available and
197,912,092,999,680B total. The alias is exposed as virtiofs,not a direct
client-side JuiceFS mount. Metadata alone does not independently bind the backend
bucket or verify a personal quota/reservation.

The first actual write/read/delete control uses the same SSH command without
sudo,with a unique directory under
`/mnt/ws-jfs/ycfeng/frontier/step5_position_routing/task_2026-10-22_step5_position_routing/`.
It returns rawexit1/PermissionError before any file is created. Parent metadata
shows personal root10250:10250/0775,writable;frontier0:0/0755,not writable by
i-fengyicheng. `sudo -n id -u` returns0. This isolates parent ownership rather
than an observed capacity/quota failure. No chmod/chown is applied.

The corrected command is
`timeout 75s python3 -B -` locally,constructing stdin for
`ssh -o BatchMode=yes -o ConnectTimeout=10 shai-ycfeng sudo -n python3 -`.
The script assertsSUDO_USER=i-fengyicheng andeuid0,confines the resolved test
directory to the personal root,opens a unique file with`xb`,writes the original
2406-byte local evidence JSON,flushes/fsyncs,compares the entire read payload,
then unlinks the file and removes only its unique empty directory. The full
control is one<=60s remote storage command,not a production shape invocation.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Ordinary write | Writable intended parent | Root-owned0755 parent deniesmkdir | FAIL,retained |
| Approved sudo write | Exact unique file,no overwrite | 2406B written | PASS |
| Readback | Full bytes equal original payload | True | PASS |
| Timely test cleanup | File and unique directory absent | Both absent | PASS |
| Existing permissions | No changes | Unchanged | PASS |

Receipt: `cloud_control.json`,timestamp2026-10-24T13:29:38.694000Z.
First failure log and successful output are retained under `evidence/`.

## Completed Duplicate Cleanup

`cleanup_receipt.json` records16 exact shape-failure scratch duplicates removed
after verifying each source and retained archive SHA256. Removed bytes298,219;
all16 canonical archived logs/receipts/profiles remain,as do active fixture,
scripts,source and independent-investigation inputs. The small cloud control
file was also deleted immediately. No large cited evidence or other-session
data was selected for cleanup,and no reclaimed gigabyte amount is claimed.

## Remaining Limits

The next launch must verify actual workload access/runtime,location quota,
capacity and shared-pool margin. This CPU host has no cloud mount and may not
mount it directly. Remote small-file PASS is not proof that the actual CLI
can stream hundreds of GB there or pass its2400s gate. Whole-workflow/native/
G3/E2E acceptance remains open;no new GPU or required-shape run was submitted.

## Record Verification

Command:`timeout 60s python3 -B -` in the primary worktree,with inline stdlib
verification of the two artifact manifests,failed/corrected driver SHA256,
five checker outcomes,cloud cleanup flags,the16 removed-path/retained-copy pairs,
capacity arithmetic and the existing frozen source/fixture bindings. Source
checks run`git diff --name-only 886608ea HEAD -- . ':(exclude)task_memory/**'`
and`git diff --name-only -- . ':(exclude)task_memory/**'`.

Actual`verification.json`:PASS;16 shape artifacts,5 storage hash bindings,
5 checker controls,23 frozen source hashes,262 fixture metadata bindings.
Both committed and working non-task tracked-source diffs are empty.16 duplicates
are absent and their archived bytes remain correct. No production shape method,
full CLI or GPU job ran during this verification.

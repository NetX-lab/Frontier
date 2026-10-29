# Fresh scope/lifetime prelaunch gates

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Recorded current storage/project quota, personal platform quota/capacity, NFS source and personal destination reads before the existing one-attempt GPU validation. |

Frozen package acceptance remains unchanged. These are new read-only gates; no source tests, image workload or GPU job were rerun. Exact command arrays, timestamps, stdout/stderr and source scripts are in prelaunch/*.receipt.json and prelaunch/receipt.json. Local Python3.10.6; StepMind virtualenv Python3.10.6/brainpp2.8.6; legacy project-quota reader compiled with gcc -O2 -Wall -Wextra -Werror against the installed Linux quota headers.

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Frozen executable and main archive |18inputs, same manifest/source bytes, original pins unchanged |18/18 inputs and67/67 original pins exact;184495B |PASS |
| Local storage |Current free bytes, actual project quota and NFS coverage |Shared /data XFS/prjquota counted once;228644413440 available bytes; project0 accounting/enforcement true, no hard/soft limit; /data NFS export covers the original source |PASS read |
| Personal platform reads |codesign/H800 quota and node snapshot; no duplicate package job |non-preemptible87/128;12 usable GPUs in returned nodes;129 personal jobs checked with full pagination;0 matching package jobs |PASS read, not reservation |
| Personal cloud |Exact destination absent, current filesystem and parent metadata |Destination absent;38517024747520 available bytes; parent root:root/0755 |PASS absence/storage read |
| Worker write prerequisite |Image's configured user consistent with root-owned parent |Locked image has no User field, Docker default UID0; no permission/mount changes |Static prerequisite only; verify actual worker write |
| Submission |No old or new attempt used before seal |0/1, launcher_started/result absent |PASS |

Preserved failures: xfs_quota was unavailable; quotactl_fd returned ENOSYS, and the legacy read-only quotactl succeeded. The first environment probe required a task AUTHORIZATION label before it was set; the ready launcher environment now uses the existing frozen config label. An SSH-user write assertion failed because the reader is not the image's root user. A Docker template addressed absent Config.User and failed; parsed metadata preserves the missing field. None is counted as a successful gate or hidden. No package input, cloud permission or destination was modified.

Snapshots do not reserve capacity or storage. NFS/worker UID/H800 identity, actual cloud write, runtime P1–P6 and platform/container terminal evidence remain pending. No TLS/native ownership, timing/non-perturbation, cost/reference admission.

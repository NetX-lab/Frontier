# D29 Storage Readiness Progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Correct current projection to include EP ledger; record static worker proposal and memory/readback blockers without a job or source allocation. |
| 2026-10-24 | Record current local capacity and mark storage BLOCKED without authorizing cleanup or external staging. |

## Status

- completed/cloud-write planning: D29-CLI-CLOUD-WRITE-PLAN,
  ses_e91b47e39ffeRn6lPWVUegoZQ6. Existing output-root flags work with a mounted
  worker volume; zero-GPU codesign is a candidate, not a qualified resource shape.
- blocked/current readiness: older261GB/mode accounting omitted EP lane ledger.
  Representative historical scaling is approximately469GB/mode and938GB/both,
  excluding other outputs and without any RSS/full-file/throughput measurement.
  Production retains all ledger rows;the validator materializes the entire file.
  Worker resources/image/mount,quota,append throughput and cohesive current timing
  remain unresolved. See execution_proposal.md;no worker or source repair allocated.
- in-progress/ledger design and scope: primary reviews the bounded write/read
  proposal; D29-LEDGER-SCOPE-BASIS,ses_e91a0552fffeM34Rv3FXfozhYu, verifies exact
  cumulative baseline costs before allocation. Standing repair-path authority
  permits necessary path amendments within unchanged owner/global budgets.

## Dated local capacity and access evidence

- completed/ownership: primary owns D29-STORAGE-READINESS.
- completed/local inspection: explore session ses_e91ee5fefffee4jkdiQMREtWQh,
 2026-10-24T21:20:14+08:00 and21:20:32+08:00, local metadata only.
- completed/arithmetic: latest/data available230,937,346,048B versus sampled
 261,400,000,000B/mode and522,800,000,000B/both. Projected deficits are
 30,462,653,952B and291,862,653,952B, before unmeasured overhead/headroom.
- blocked/eligible location: all inspected/data paths share/dev/rbd1;root
 /home/brainpp and/mnt share/dev/rbd0 with59,510,960,128B available. Neither can
 host the projected complete single-mode output. Separate pools are not one file.
- completed/approval: YC authorizes codesign OSS test storage and timely cleanup.
- completed/cloud metadata:personal alias/mnt/ws-jfs/ycfeng reports
 38,994,847,006,720B available at2026-10-24T13:27:14Z. Mount is exposed as virtiofs;
 the documented alias,not this metadata alone,identifies the intended cloud volume.
- completed/storage control: ordinary-user mkdir fails because frontier is
 root:root0755. Sudo is available;the same2406B full-byte write/read/delete passes
 at13:29:38Z without changing existing permissions. Unique test directory removed.
- completed/cleanup:16 exact completed scratch duplicates(298,219B) removed only
 after verifying their retained archive hashes. Active fixture/scripts,other
 investigations and selected failure evidence are unchanged.
- pending/CLI access/runtime/location quota/headroom/prelaunch check: XFS project limits are not
 measured;filesystem free space may exceed usable project capacity. Capacity is
 shared,variable and unreserved. No full CLI or GPU/platform operation was run.

The CPU host has no cloud mount;the handbook prohibits directly mounting it.
Actual workload access is not proved by the small remote storage control.
Worker mounts must use the approved local StepMind route;no new job was submitted.

Evidence: `evidence.json`, `test_report_2026-10-24_local_capacity.md`.

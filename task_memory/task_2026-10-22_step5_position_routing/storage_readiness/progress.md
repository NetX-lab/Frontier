# D29 Storage Readiness Progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 | Record USER_CONFIRMED_SUFFICIENT OSS quota; source admission and worker execution readiness remain separate handoff dependencies. |
| 2026-10-24 | Correct current projection to include EP ledger; record static worker proposal and memory/readback blockers without a job or source allocation. |
| 2026-10-24 | Record current local capacity and mark storage BLOCKED without authorizing cleanup or external staging. |

## Status

- completed/cloud-write planning: D29-CLI-CLOUD-WRITE-PLAN,
  ses_e91b47e39ffeRn6lPWVUegoZQ6. Existing output-root flags work with a mounted
  worker volume; zero-GPU codesign is a candidate, not a qualified resource shape.
- blocked/current readiness: older261GB/mode accounting omitted EP lane ledger.
  Representative historical scaling is approximately469GB/mode and938GB/both,
  excluding other outputs and without any RSS/full-file/throughput measurement.
   Accepted production still retains all ledger rows/materializes the file; a
   committed bounded write/read candidate exists but is not admitted. Worker
   resources/image/mount,append throughput and cohesive timing remain unresolved.
   No worker was launched. See execution_proposal.md and ledger_memory/.
- completed/quota decision: YC explicitly confirms sufficient codesign OSS quota
   on2026-10-25 and points to ssh shai-ycfeng. Capacity is USER_CONFIRMED, not an
   independent quota measurement; further quota discovery is not a blocker.
- completed/ledger focused candidate: allocation886945f3 is36/20/65/14,
   still135gross. Final source1f6b4314/recordsdd3b6a69 passes caller,66/6integration,
   EP30 andprimary39. Independent composed audit/reviews/final admission remain; no worker/shape
   allowance is extended. Final candidate records are carried into handoff.

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
- pending/execution environment/worker append/runtime/prelaunch check: local XFS
  project limits are not measured, and local shared capacity is unreserved. OSS
  quota sufficiency is now user-confirmed; this is distinct from proving worker
  write performance. No full CLI or GPU/platform operation was run for handoff.

The CPU host has no cloud mount;the handbook prohibits directly mounting it.
Actual workload access is not proved by the small remote storage control.
Worker mounts must use the approved local StepMind route;no new job was submitted.

Evidence: `evidence.json`, `test_report_2026-10-24_local_capacity.md`.

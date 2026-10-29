# P0 GC commands

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Retained exact command receipts and selected evidence; no additional experiment. |

Scratch root: `/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024/`.
Each execution uses `sudo -n systemd-run --scope -p MemoryMax=30G timeout60s`;
the actual v1memory-controller value is32212254720bytes, with its mount-root prefix removed.
Full argv, timestamps, source/output hashes and initial controller state:
`p0_observe_01.receipt.json`, `decode_dp1_freeze_control_01.receipt.json`,
`freeze_control_equivalence.json`, `p0_freeze_01.receipt.json`.
The original observer control's execution/rows are bound by `control_equivalence.json`.
Commands use `/usr/bin/python3`, `PYTHONDONTWRITEBYTECODE=1` and the same exact
P0trace path shown in the receipts; comparison imports the existing stage auditor.

Derived comparison: `freeze_comparison.json` SHA25636179e6ad2326bdb1cb5cd54811ed40c7506b5ee6f79716457e5b63ae269583a.
Selected local evidence is bound in `gc_evidence.json`; large raw input/partial indexes
remain in the existing scratch paths. No log is hashed before its writer terminates.
Two earlier pre-entry memory-path failures are diagnostic-harness history, not audit runs.

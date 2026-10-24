# D29 Storage Readiness Plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Define a bounded read-only capacity inspection and the remaining approval-dependent readiness steps. |

## Dependency Graph

`recorded storage blocker -> local metadata -> options/approval -> selected location/quota/headroom -> prelaunch recheck`

Copy and initialization investigations are independent. Actual full CLI waits
for storage readiness and an accepted frozen source, not for additional isolated
shape checks masquerading as whole-workflow validation.

## Steps and Done Conditions

1. Primary owns the storage prerequisite. Record full-output expectations from
   existing multiplicity evidence; preserve their estimate/measurement limits.
2. Inspect named local filesystem metadata only,<=60s per command. Done when
   timestamp, commands, pool identities and exact available bytes are recorded.
3. Compare capacity with one/two-mode projections; propose eligible strategies.
   Done with a BLOCKER or an approved location, not a barely-fitting estimate.
4. YC approves codesign OSS test storage and timely completed-file cleanup.
   Verify only the personal alias;use the existing sudo access for root-owned
   frontier rather than changing permissions. Remove the test immediately.
5. Verify actual CLI access/runtime,selected location/project quota/headroom and recheck before launch.
   Stop if capacity is unavailable; actual CLI keeps its2400s limit and full output.

## Verification

Use `df -B1` and `findmnt` scoped to approved local paths. Do not run directory,
CSV/model/trace scans, network/platform commands or cleanup to improve a verdict.

The explicit cloud decision permits scoped SSH storage reads/writes/cleanup on
the documented personal alias;it does not authorize remote RJob control or a
direct CPU-host mount. Worker mounts retain the StepMind RJobBackend route.

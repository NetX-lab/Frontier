## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Added fail-fast admission invariant and stale-owner wakeup lessons. |
| 2026-09-05 | Added Replica-local TP/DP/EP and cohort lifecycle lessons. |
| 2026-09-05 | Recorded reusable scheduler and metrics lane lessons. |

# Lessons

Scheduler lane identity and metrics indexing must share one explicit contract. PD-AF transfer provenance is a separate full-stage contract and must not inherit DP-lane semantics from co-location or PDD.

One Frontier `Replica` represents one complete GPU pod. In the supported
mapping, request ownership is `(replica_id, dp_id)` for shared-domain
co-location/PDD roles, while TP remains a logical workload abstraction and
does not create per-TP schedulers. MoE routing then forms one Replica-local EP
cohort over the configured `moe_expert_parallel_size`; outer Replica count is
scheduler capacity and never a collective dimension. For `TP4 x DP2 x EP8`,
the physical backend must therefore materialize `world=8`, attention `TP4 x
DP2`, and MoE `TP1 x DP1 x EP8`.

Replica-local cohort IDs have an explicit lifecycle. A same-time real lane may
replace an idle placeholder while the cohort is open; once the placeholder
wave closes, a late real lane receives a new cohort identity. Reusing a
lane-local counter after closure can hit `completed_keys` and silently drop a
request. Stage completion also retains the batch owner identity so stale
events release the correct lane and wake queued siblings.

Production cohort promotion and restoration must validate admission tickets for
every live lane before changing scope. Missing tickets indicate a broken stage
ownership invariant and must raise explicitly; compatibility behavior belongs
only to direct probes that have no stage context at all. The final regression
suite passed `720` tests with `19` skips.

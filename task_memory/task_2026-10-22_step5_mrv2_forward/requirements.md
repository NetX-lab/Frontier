# D23-MRV2-FORWARD Requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record the delegated approved CPU core scope. |

## [Original Request]
Implement approved pre-first-prediction agreement using existing stage group,
owners and stage events for all useful and dummy P/D lanes. Bind
FULL_AND_PIECEWISE; NONE preserves local widths; graphs share capture padding.
Preserve local query/context, serving ownership and zero dummy request credit.
Reuse the shared MRv2 descriptor, retaining FULL raw null and PIECEWISE null
request dimension. Validate first calls, lifecycle/release/wakeup/duplicates,
producer-fit-cache-count-lookup coverage, and legacy behavior. CPU only, no
further delegation or platform/SSH/push. Parent owns publication, G3 and full
integration; do not edit actual-main runner/fixtures. Base is immutable
5c2319cd; inputs worktree remains untouched.

Authoritative approval: primary worktree
`task_memory/task_2026-10-15_step5_preview_h800/requirements.md:106-118` and
`mrv2_consumer_20261022/design.md`.

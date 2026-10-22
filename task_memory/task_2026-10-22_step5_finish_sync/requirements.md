# D30 Step finish-sync requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record approved Step workflow correction and observation scope. |

## [Original Request / Decision] D30

Authoritative verbatim question, answer and option text:
`../task_2026-10-15_step5_preview_h800/requirements.md`, D30 Step finish sync.
YC selected **批准同步对齐 (Recommended)**.

Match current Step native finish synchronization on wave step1 and each16
steps, with counter reset at wave completion. Retain other models' existing
behavior and legitimate dummy participation. Use the existing native-family/
runtime contract and scheduler owners; no global32-to16 replacement or
unnecessary alternative configuration path. The approved scope includes
necessary configuration/scheduler changes, focused boundaries and native
counter observation. Native scheduling itself is unchanged.

Preserve request/READ/readiness/source-release semantics, legacy artifacts,
source-bound input identity, full CLI2400s limit and D24 resource bounds.
No native CPU overhead/gap analysis or measurement suite. Observation errors
are isolated and must not affect native calls, returns or exceptions.

Bound source/test changes to five files/500 lines unless a concrete larger
design is presented to YC. Native image/package/submission remain parent-owned.

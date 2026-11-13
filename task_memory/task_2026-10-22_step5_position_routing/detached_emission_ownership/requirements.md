# Requirements — D29-DETACHED-EMISSION-OWNERSHIP

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Captured the parent request and the decisions taken in this read-only assignment. |

## [Original Request]

Read-only scope `D29-DETACHED-EMISSION-OWNERSHIP`. Locate a materially useful
private emission ownership/preparation boundary after the required real
`ExecutionTime` getter deep snapshots, without changing public `to_dict`, snapshot
semantics, output bytes or the late-failure prefix. Rank H1 (redundant second
traversal of already-detached getter data), H2 (per-lane common metadata derived
per operator) and H3 (residual is required reconstruction). Inspect existing
profiles and inventories first; trace object ownership, callers and escaping
references through `ep_wave_metrics.py`, `op_trace_utils.py` and the getters.
Distinguish transferring an already-owned private subtree from borrowing mutable
source. Preserve within-lane and cross-lane alias boundaries. No new runtime
campaign, no production/test edits, no encoder/private-API/deepcopy-protocol/
caching proposal by fiat. Report under 500 words with line refs, alias evidence,
profile-derived cost bound, one concrete next control or a no-seam verdict, and
limits.

## Decisions and constraints

- The assigned key-only candidate (`root 087db1bf`) is another agent's; it is
  neither duplicated nor edited, and its 39.956 ms baseline stays diagnostic.
- The parent's statement that `ep_wave_metrics.py:30` is not two getter calls is
  confirmed: line 29 calls the getter once and line 30 reads a plain field.
- Only the records worktree
  `.worktrees/d29-trace-encoding/task_memory/task_2026-10-22_step5_position_routing/detached_emission_ownership/`
  is written. Scratch evidence is read from the existing
  `/data/ycfeng/tmp/d29_emission_immutable_work/` profiles and the retained
  208-event artifact.
- No claim of public-domain correctness: parity evidence remains the canonical
  208-event wave.

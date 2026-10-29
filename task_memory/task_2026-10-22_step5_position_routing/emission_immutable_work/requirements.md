# D29 emission immutable work requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record parent continuation and fixed scope after separate trace correction commit 01a1c58d. |

## [Original Request]

Continue as `D29-EMISSION-IMMUTABLE-WORK` (parent owns design/allocation). The trace
escape candidate stays unallocated pending larger emission gains; no production
edits. Trace corrections must be committed separately first.

Bounded emission diagnosis on frozen `75404143`, fresh own scratch only. Preserve
the retained real 208-event emission loop's probe body/assertions as the
RED-capable seam. Ranked hypotheses shown to YC:

1. H1 getter deepcopies traverse truly immutable children unnecessarily.
2. H2 emission repeats derived plan-key serialization.
3. H3 `to_dict` recursively converts immutable field structure.

Actual getter deepcopies are `frontier/entities/execution_time.py:709-786`;
`physical_frame` is a plain field, so `ep_wave_metrics.py:30` is not two getter
calls. Do not bypass a getter, borrow mutable components, cache whole proofs or
trace rows, move JSON failure timing, join writes or alter trace encoding.
Inspect actually traversed types. An immutable object's `__deepcopy__` can return
itself only for genuinely deeply immutable reached values (not `frozen=True`
alone). Preserve source→event, event→event and trace→ledger mutable isolation.

Owner `ses_e963...` proved complete plan preparation exists (8 plans, 0 warm
constructors). Its narrow cached-string diagnostic in
`/data/ycfeng/tmp/d29_immutable_plan_preparation/compare.py` eliminates 192 pricing
key encodes. It may be adapted to measure emission separately, but do not
investigate pricing or edit owner files. Public `to_dict` stays fresh. A
fields-based conversion must create fresh tensor/context/buffer dicts with
identical tuples/order/full keys.

At most one control per hypothesis; combined only if isolated gains justify it.
60 s/30 GiB each; both `python -B` and `PYTHONDONTWRITEBYTECODE=1`. No broad
campaign, full CLI, GPU, network, output reduction, source/test/peer edit or
allocation. Records in this directory; source probes only in own scratch
`/data/ycfeng/tmp/d29_emission_immutable_work/`.

## Acceptance

Real 208 events/JSONL/ledger byte equality; all current snapshot/ownership checks,
64 wrong-key and 3902 B prefix checks. Measured getter/deepcopy/asdict/key counts,
phase-isolated unprofiled timings, limits, exact minimal paths and Git-derived
gross/net estimate. No safe gain → no justified seam, not relaxed semantics.
Compact handoff with TaskID/sessionID, commits, decisive evidence, unresolved
points and exact paths.

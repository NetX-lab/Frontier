# D29 emission immutable work plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Define bounded serial diagnosis and acceptance. |

`trace correction commit -> baseline/type inventory -> {H1, H2, H3} -> conditional combined -> report`

1. Completed prerequisite: trace corrections committed as `01a1c58d`.
2. Run unchanged accepted probe body against frozen source; inventory actual
   getter children and measured getter counts without contaminating timings.
3. One control per ranked hypothesis, serial in separate processes. Each runs
   the unchanged body (15 unprofiled emission timings, outside setup/flush/ledger
   encoding); instrumentation is separate and untimed.
4. Combine only isolated candidates with measured gains and passing semantics.
5. Record exact output hashes, ownership negatives, type evidence, counts,
   timings, minimal patch artifact numstat and limitations; commit records only.

Mutable isolation is evaluated relative to the original emitted graph. If the
original already shares a per-lane native-context map among several events, do
not claim full event→event independence or silently expand the correction.
Document the actual boundary and do not introduce new aliases.

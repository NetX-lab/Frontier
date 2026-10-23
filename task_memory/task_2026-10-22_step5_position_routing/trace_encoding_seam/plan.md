# D29 trace encoding seam plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Initial plan; all steps executed the same day. |

## Scope

Read-only diagnosis of the trace flush encoding on the frozen `75404143` source.
Scratch controls only; records in this directory.

## Steps and dependencies

`a -> b -> {c, d} -> e -> f`

a. Byte/work composition of the retained 208-event body (stdlib only).
b. Phase-only split of the actual flush; rank hypotheses before any control.
c. One-variable control (only the stdlib string escaper replaced, per flush) on
   one fresh wave and on a real 5000-event buffer; bytes, counters, memory.
d. Contract negatives, call-count differential, production-shaped variant (P1),
   exact-type-guard variants, py312 stdlib replica.
e. Actual-source emitted wave: actual flush versus P1; leaf types and aliasing.
f. Scratch proposal patch (never applied), validated on the actual wave and with
   the accepted unchanged probe body; verdict and gross estimate.

## Acceptance

Byte-identical output on every timed run; identical error text, 3902 B prefix and
buffer restoration; RED-capable negative; one-variable timing versus the actual
unmodified flush; verdict with explicit preconditions.

# Workflow Gap Analysis

**When to use:** Use when Frontier and vLLM batches or timing paths diverge and action, event, or behavior control flow must be explained.

Read `../references/contract.md` and `../references/evidence-tiers.md`. Require
formal IDs, clean/instrumented separation, scheduler decisions, Frontier event
and TraceStore rows, and dispatch evidence when online. Compare arrival
visibility, initial wait, token budgets, admission, phase changes, batch
formation, KV availability, routing, transfers, and completion bookkeeping.

Choose the first valid non-empty formal scheduler batch, validate ordered IDs and
token vectors, and distinguish client dispatch skew from internal control-flow
differences. Record every row as `MATCH`, `MISMATCH`, `UNSET`, or `UNSUPPORTED`
with values, source anchors, confidence, affected phase, and Frontier owner.
Require the shared first-batch and three-aligned-batch predicates before saying
operator comparison is meaningful; otherwise route to `op-supplement`.

Write `analysis/workflow_gap_table.csv`, summary, and status. This entry is
read-only and does not permit a code change.

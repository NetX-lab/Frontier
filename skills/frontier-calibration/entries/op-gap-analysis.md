# Operator Gap Analysis

**When to use:** Use when aligned scheduler batches and CUDA-op timings need an auditable per-operator comparison.

Read `../references/contract.md`, `../references/evidence-tiers.md`, and
`../references/tool-boundaries.md`. Require formal request IDs, scheduler
decisions, Frontier batch/op rows, vLLM CUDA-op rows, lane/rank provenance, and
profiling coverage. Normalize scheduler, batch, phase, lane, and operator
identity internally; retain the resulting `analysis/op_trace_normalized.jsonl`
as an intermediate artifact, not as a separate route.

Select the first valid formal batch and require at least three distinct aligned
logical batches covering every active phase. Validate ordered IDs, token totals,
available blocks, measurement family, and operator-family registry membership.
For each matched key compute signed `gap_ms`, absolute gap, and relative gap with
a positive finite denominator. Preserve missing, conflicting, and excluded rows
and route missing coverage to `op-supplement`.

Write `analysis/op_gap_table.csv`, summary, normalized stream, and status. This
entry only reports evidence; it never changes predictors, profiling inputs, or
Frontier code.

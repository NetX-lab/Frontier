# Dispatch Align Trace

**When to use:** Use when online client dispatch timing causes a visible skew between vLLM and the Frontier arrival trace.

Read `../references/contract.md` and `../references/tool-boundaries.md`. Use only
for an online multi-request case whose evidence shows client visibility skew.
Run the existing clean observation with at least three warmups and capture
formal IDs, token counts, priorities, intended arrivals, finite non-negative
dispatch offsets, and dispatch lag.

Invoke the existing dispatch helpers by the exact paths in
`../references/tool-boundaries.md`; do not infer a relative `scripts/` path.
Rewrite only the
declared `arrived_at` field and write
`analysis/dispatch_aligned_trace.csv` plus
`analysis/dispatch_alignment_manifest.json`. Re-read and compare IDs, tokens,
priorities, counts, and payload fields before `simulator-run` consumes the trace.
Reject duplicate IDs, invalid offsets, missing priorities, uniform arrivals for
a staggered study, or changed request content. This entry changes a data trace,
not Frontier or vLLM code, and it never closes the E2E gate.

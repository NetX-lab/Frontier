# Error Analysis

**When to use:** Use when a completed Frontier-vLLM case needs a read-only explanation of its clean E2E, workflow, operator, residual, or CPU gap.

Read `../references/contract.md`, `../references/evidence-tiers.md`, and
`../references/call-graph.md`. Start with `e2e-metrics-gap`; stop with a clean
`PASS` unless diagnostics were explicitly requested. For a failing metric,
check semantic meaning and routing/prefix state, then compare workflow and
dispatch visibility, run `op-gap-analysis` (including its internal
normalization), classify residual lanes, and open CPU analysis only under its
strict post-correction predicate.

Route missing first-batch or coverage evidence to the matching supplement and
return to the owning analysis. Write one review-ready
`reports/error_analysis_summary.md` that links, rather than copies, the atomic
rows and source paths. This composite is read-only and never authorizes a code
change.

# Error Correction

**When to use:** Use when reviewed Frontier-vLLM analysis artifacts identify corrections that must be applied and revalidated.

Read `../references/contract.md`, `../references/repair-approval.md`, and
`../references/call-graph.md`. This is the only composite that coordinates code
changes. Select one reviewed domain at a time: semantic alignment, workflow,
operator, residual lane, or CPU. Require complete analysis, external human
review `PASS`, or the explicit automatic exception before each domain entry.

Record the owner-controlled repair plan, approved scope, and code-change marker;
call the domain entry; run focused and isolated checks; rerun the relevant
analysis; and obtain a fresh approval before the next domain. After workflow and
operator corrections, run a fresh clean E2E case before considering CPU. Finish
with `parity-run` and `e2e-metrics-gap`; close only when every applicable metric
is within `0.10` and all diagnostic evidence is valid.

This composite coordinates domain entries but does not use a result artifact as
its own permission to change code.

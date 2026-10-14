# CPU Overhead Repair

**When to use:** Use when reviewed CPU evidence proves a scoped Frontier host-side accounting correction.

Read `../references/contract.md` and `../references/repair-approval.md`. Require
the CPU entry predicate, complete CPU analysis with a reproducible dominant
component, pending correction state, and external human review `PASS`, or
`case_init.auto_recycle: true` with a complete auto receipt. Keep the change
within the named Frontier CPU
accounting owner and preserve the complete component schema.

Record approval and a code-change marker before editing. Apply one scoped
Frontier change, run focused CPU checks, an isolated CPU replay, and a fresh
clean E2E case, then write `repairs/cpu_repair_receipt.json` with exact files,
commands, before/after component values, approval path, and validation status.
Require fresh CPU and E2E analysis before another code change.

# Workflow Repair

**When to use:** Use when reviewed workflow evidence identifies a reproducible Frontier control-flow difference.

Read `../references/contract.md` and `../references/repair-approval.md`. Require
a complete workflow status with a concrete causal mismatch, pending correction
state, and external human review `PASS`, or `case_init.auto_recycle: true` with
a complete auto receipt.
Re-check the cited Frontier symbols and keep vLLM scheduling/inference behavior
as the reference.

Record the approved scope and code-change marker before editing. Apply one
smallest Frontier control-flow change, run focused checks, an isolated
diagnostic replay, and a clean E2E case, then write
`repairs/workflow_repair_receipt.json` with exact files, commands, before/after
evidence, approval path, and validation status. Require fresh workflow and E2E
analysis before another code change.

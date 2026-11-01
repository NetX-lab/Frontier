# Workflow Repair

**When to use:** Use when a decided workflow analysis identifies a reproducible Frontier control-flow difference.

Read `../references/contract.md` and `../references/repair-approval.md`.
Require a complete workflow status with a concrete causal `simulation_logic`
mismatch, `correction_state=pending`, and the user's recorded decision with its
decision ID; the human review record follows `repair-approval.md`. Re-check the
cited Frontier symbols against the selected engine's scheduling and inference
behavior, including intentional fork changes and their declared base. A scheduler or event change touches a
core module, so a design note goes to the user before the edit.

Write the code-change marker (`change_marker`) with the decision ID and scope
before editing. Apply one smallest Frontier control-flow change as one commit.
Run focused tests and the regression set, rerun the affected cells, and confirm
that cells the change does not apply to stay byte-identical. Write
`repairs/workflow_repair_receipt.json` with changed files, commit, commands,
before and after evidence, the decision ID, and validation status. Require fresh
workflow and E2E analysis before another code change.

# Operator Repair

**When to use:** Use when a decided operator analysis proves a scoped predictor, measurement-family, padded-width, or pricing correction in Frontier.

Read `../references/contract.md` and `../references/repair-approval.md`.
Require complete op analysis with valid aligned batches, a `simulation_logic`
class, `correction_state=pending`, and the user's recorded decision with its
decision ID; the human review record follows `repair-approval.md`. Keep the
change within the named operator family, phase, measurement family, and
Frontier owner, and leave every other family's prediction unchanged. A global
scale factor is outside the contract; a table problem is a supplement, not a
repair.

Write the code-change marker (`change_marker`) with the decision ID and scope
before editing. Apply one smallest Frontier change as one commit. Run focused
predictor tests and the regression set, rerun the affected cells with fresh
predictor caches, and write `repairs/op_repair_receipt.json` with changed
files, commit, commands, before and after values, the decision ID, and
validation status. Require fresh op and E2E analysis before another code
change.

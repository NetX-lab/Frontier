# Semantic Alignment Repair

**When to use:** Use when a decided semantic analysis names a concrete Frontier setting, mapping, or case-setting mismatch.

Read `../references/contract.md` and `../references/repair-approval.md`.
Require a complete semantic status (`analysis_state=COMPLETE`, `status=PASS`,
`correction_state=pending`), a concrete mismatch, and the user's recorded
decision with its decision ID; the human review record follows
`repair-approval.md`. A case-setting fix (engine file, settings file, trace)
starts a new run generation; a Frontier code fix follows the full approval
order.

Write the code-change marker (`change_marker`) with the decision ID and scope
before the first edit. Change only the named Frontier setting, mapping, or
module. Run the regression set, rerun the affected cells, and write fresh
semantic analysis, then `repairs/semantic_alignment_repair_receipt.json` with
changed files, commit, commands, before and after values, the decision ID, and
validation status. A failed or stale result stops the chain and authorizes no
further change.

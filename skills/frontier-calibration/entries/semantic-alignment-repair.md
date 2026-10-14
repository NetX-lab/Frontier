# Semantic Alignment Repair

**When to use:** Use when reviewed semantic evidence identifies a concrete Frontier or user-setting mismatch.

Read `../references/contract.md` and `../references/repair-approval.md`. Require
a complete semantic status with `analysis_state=COMPLETE`, top-level `status=PASS`,
`correction_state=pending`, a concrete mismatch, and a matching human review
`PASS`. Use the automatic path only when the immutable case initialization
records `case_init.auto_recycle: true` and its complete auto receipt exists
before the code change.

Record the approved scope and a plain code-change marker immediately before the
first edit. Change only the named Frontier setting or module, run an isolated
semantic check and a clean case, then write fresh semantic analysis before
`repairs/semantic_alignment_repair_receipt.json` with changed files, commands,
before/after values, authorization receipt, and validation status. A failed or
stale result stops the chain and cannot authorize another change.

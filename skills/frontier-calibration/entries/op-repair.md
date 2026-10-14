# Operator Repair

**When to use:** Use when reviewed operator evidence proves a scoped predictor, mapping, or profiling-input correction.

Read `../references/contract.md` and `../references/repair-approval.md`. Require
complete op analysis with valid aligned batches, a pending correction state, and
external human review `PASS`, or `case_init.auto_recycle: true` with a complete
auto receipt. Keep any
change scoped to the named operator family, phase, measurement family, and
Frontier owner; do not add a global scale.

Record approval and a code-change marker before editing. Apply one smallest
Frontier change, run focused and isolated operator diagnostics plus a fresh clean
E2E case, and write `repairs/op_repair_receipt.json` with exact files, commands,
before/after values, approval path, and validation status. Require fresh op and
E2E analysis before another code change.

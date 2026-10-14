# Residual Repair

**When to use:** Use when reviewed residual evidence identifies a PP, collective, or KV/M2N correction.

Read `../references/contract.md`, `../references/repair-approval.md`, and the
lane fields in `../references/registry.json`. Load the immutable case manifest
before selecting a lane. Require a complete residual status
with `primary_class` resolving to exactly one `lane_registry` entry, pending
correction state, and an external human review `PASS`, or
`case_init.auto_recycle: true` with a complete auto receipt.

Use the machine-readable lane registry to select required evidence, the
analysis-provided `frontier_owner`, and the lane-specific receipt path:

- `pp` resolves from `classification=pp_overhead`;
- `collective` resolves from `classification=collective_communication`;
- `kv_or_m2n_transfer` resolves from `classification=kv_or_m2n_transfer`.

Record approval and a code-change marker before the first edit. Apply one scoped
Frontier change, run isolated lane diagnostics and a fresh clean E2E case, and
write fresh residual analysis before another code change and the lane-specific
receipt under `repairs/`. Keep the three lane receipts
distinct in the artifact contract. A mixed or unclear class stops without a
code change.

# Residual Repair

**When to use:** Use when a decided residual classification identifies a PP, collective, or KV/M2N correction in Frontier.

Read `../references/contract.md`, `../references/repair-approval.md`, and the
lane fields in `../references/registry.json`. Load the case manifest before
selecting a lane. Require a complete residual status whose `primary_class`
resolves to exactly one `lane_registry` entry, `correction_state=pending`, and
the user's recorded decision with its decision ID; the human review record
follows `repair-approval.md`.

Use the machine-readable lane registry to select the required evidence, the
analysis-provided `frontier_owner`, and the lane-specific receipt path:

- `pp` resolves from `classification=pp_overhead`;
- `collective` resolves from `classification=collective_communication`;
- `kv_or_m2n_transfer` resolves from `classification=kv_or_m2n_transfer`.

When the user asks for a temporary correction instead of a model (for example,
a measured per-layer KV-send slowdown), make it a named setting whose default
leaves every existing run unchanged, and record the deferred model as open
work.

Write the code-change marker (`change_marker`) with the decision ID and scope
before the first edit. Apply one scoped Frontier change as one commit, run the
regression set and the lane diagnostics, rerun the affected cells, and write
fresh residual analysis and the lane-specific receipt under `repairs/` before
another code change. Keep the three lane receipts distinct. A mixed or unclear
class stops without a code change.

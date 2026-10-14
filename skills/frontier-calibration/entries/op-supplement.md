# Operator Supplement

**When to use:** Use when operator analysis lacks a formal batch, aligned batch, phase, shape, or profiling row.

Read `../references/contract.md` and `../references/tool-boundaries.md`. Require
an explicit missing tuple list from `op-gap-analysis`. Derive the first formal
batch, three-batch minimum, active phases, and observed in-flight endpoints;
label analytical rows rather than guessing a neighboring shape. Keep clean E2E
metrics separate from operator timing and use at least three warmup replays.

Invoke the existing profiling merge tool at the exact path in
`../references/tool-boundaries.md` with an explicit fresh output path; do not
infer a relative `scripts/` path.
Publish a complete merged CSV under the case `supplements/` directory: retain
every base row, append accepted rows deterministically, record every added or
deduplicated row with identity, source row, source path, fingerprint, and
coverage basis, and keep `dropped_rows=[]`. Reject same-context contradictions,
missing measurements, stale markers, or output paths that alias the base CSV.

Write `op_supplement_plan.yaml`, result CSV, summary receipt, and completion
marker, then return to `op-gap-analysis`. This entry changes data only and never
authorizes a code change.

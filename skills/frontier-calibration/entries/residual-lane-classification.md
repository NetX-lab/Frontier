# Residual Lane Classification

**When to use:** Use when a residual remains after metric, workflow, and operator checks and needs a source classification.

Read `../references/contract.md` and `../references/evidence-tiers.md`. Require
the latest clean E2E, workflow, and operator statuses plus native PP,
collective, and KV/M2N traces. Test each lane with its own identity, topology,
scope, overlap, and timing fields; remove PP rank copies, EP sub-batches, and
duplicate transfer events without inventing batches.

Assign one evidence-backed primary class: `pp_overhead`,
`collective_communication`, `kv_or_m2n_transfer`, `mixed`, or `still_unclear`.
For the first three labels, resolve the repair lane and receipt through the
`lane_registry` in `../references/registry.json`; `mixed` and `still_unclear`
have no repair lane and stop without a code change.
Record both timings, numeric gaps, source type, ruled-out lanes, and missing
coverage. Open the CPU action only when workflow and operator corrections are
complete and a fresh clean E2E generation has an applicable relative error over
`0.10`.

Write `analysis/residual_gap_table.csv`, summary, and status. This entry is
read-only; a classified lane never grants permission for a code change.

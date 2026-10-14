# CPU Overhead Supplement

**When to use:** Use when CPU analysis reports missing identity tuples, phases, or replay/profile rows.

Read `../references/contract.md` and `../references/tool-boundaries.md`. Require
an explicit missing tuple list from `cpu-overhead-analysis` and preserve its
post-workflow/op E2E entry condition. Plan every tuple as
`model,batch,TP,prefill tokens,decode tokens,scheduling mode,precision,phase`.
Use measured capacity, an explicitly provisioned multi-node run, or an explicitly
declared analytical TP model; never silently substitute a lower degree.

Run at least three warmup replays with CPU probes isolated from E2E and operator
instrumentation. Use the existing lossless merge tool at the exact path in
`../references/tool-boundaries.md` to publish a complete
`supplements/cpu_supplement_result.csv` retaining every base row. Record every
added or deduplicated row, canonical source, validation state, source path,
fingerprint, and coverage basis; reject contradictory identities, non-finite
components, partial bundles, and stale completion markers.

Write the plan, result CSV, summary receipt, and completion marker, then return
to `cpu-overhead-analysis`. This entry changes data only and never authorizes a
code change.

# Parity Run

**When to use:** Use when a user needs one reproducible Frontier and instrumented-vLLM case launched with aligned semantics.

Read `../references/contract.md`, `../references/evidence-tiers.md`, and
`../references/call-graph.md`. Create one manifest with user-approved settings,
formal and warmup IDs, mode paths, producer and ID provenance, the exact clean
`feature/frontier-comparison-instrumentation` checkout tuple, requesting-user
identity, and immutable `case_init.auto_recycle`. Invoke `$grill-me` for every
missing semantic setting or material design choice and store its question,
evidence-backed options, answer, decision id, and UTC time in the manifest.

Call `semantic-alignment-check`, then `groundtruth-run-check`,
`groundtruth-run`, and `simulator-run` in that order. A blocking semantic result
stops for the approved semantic repair path and a fresh check. Insert
`dispatch-align-trace` only for an online skew study. Keep clean, instrumented,
CPU, and Frontier outputs disjoint and retain command receipts.

This composite launches and records a case. It does not diagnose a gap or apply
a code change itself.

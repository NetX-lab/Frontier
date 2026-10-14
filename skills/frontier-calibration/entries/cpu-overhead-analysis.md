# CPU Overhead Analysis

**When to use:** Use when workflow and operator correction attempts are complete but a fresh clean E2E metric still exceeds ten percent.

Read `../references/contract.md` and `../references/evidence-tiers.md`. Do not
enter this entry for a passing E2E case. Require complete workflow and operator
analysis with `status=PASS` and `correction_state` equal to `complete` or
`not_applicable`, a new clean E2E
generation, and at least one applicable relative error greater than `0.10`.

Use the CPU identity tuple from the Frontier schema and separate measured,
vLLM-replay, and analytical sources. Attribute scheduler, preparation,
communication, sampler, process, and bookkeeping components without introducing
an independent CPU threshold. Keep clean E2E and operator instrumentation out
of the CPU run.

Write `analysis/cpu_overhead_table.csv`, summary, and status. Treat the current
CPU producer identity, row-count, freshness, and bundle-publication watch items
as non-normative until their own analysis receives a human `PASS`. This entry is
read-only and never changes code.

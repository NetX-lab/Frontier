# Ground-Truth Run Check

**When to use:** Use when a ground-truth case needs clean, warmup, instrumentation, or CPU preflight before execution.

Read `../references/contract.md` and `../references/registry.json`. Load the
case manifest and verify the selected checkout, mode, formal request ids,
run generation, and disjoint output directories before running anything. The
selected ground-truth checkout must be on the exact branch
`feature/frontier-comparison-instrumentation` at ref
`refs/heads/feature/frontier-comparison-instrumentation`; the expected remote is
`https://github.com/fwyc0573/vLLM-BS.git`.

Check the declared mode predicate:

- clean: at least three warmups, E2E runtime metrics on, operator and CPU probes off;
- instrumented: operator diagnostics on, E2E runtime metrics off;
- cpu: CPU probe on, operator diagnostics and E2E runtime metrics off;
- Frontier: clean-style E2E metrics.

Verify fresh output paths and the complete checkout tuple: path, branch, ref,
commit, tree status, remote URL/tip, diff artifact, dirty-patch SHA-256, and
overlay-patch state. Require a clean tree for parity. Verify request-ID encoding
and all command receipt fields. Write the mode-specific
`analysis/run_check_status.<mode>.json` and summary. A missing field, dirty or
unproven checkout, failed predicate, stale target, or abnormal preflight returns
`INSUFFICIENT_EVIDENCE` or `FAIL` and stops before `groundtruth-run`.

This entry never changes code or profiling data. It does not treat a status
`PASS` as numeric parity; it only permits the next declared run.

# Ground-Truth Run

**When to use:** Use when a passed run check permits an isolated instrumented-vLLM ground-truth execution.

Read `../references/contract.md` and `../references/tool-boundaries.md`. Require
the matching `groundtruth-run-check` result for the selected mode and the exact
manifest-bound checkout tuple. The selected checkout must remain on
`feature/frontier-comparison-instrumentation` with ref
`refs/heads/feature/frontier-comparison-instrumentation`; record the remote URL
and tip, commit, tree status, diff artifact, dirty-patch SHA-256, and overlay
patch state. Run clean ground truth first, then only the instrumented or CPU mode
explicitly declared by the case.

Before launch, write `runs/<mode>/run_manifest.json` with the exact command,
working directory, declared environment, revision, visible devices, case id,
generation, and expected artifacts. After launch, retain stdout/stderr, exit
code, source paths, fresh-file checks, and mode status. Reject a missing artifact,
non-finite value, stale target, wrong mode predicate, or abnormal exit.

Use clean output only for the E2E gate. Use instrumented output only for
workflow/operator/residual diagnosis. Use CPU output only after the CPU entry
predicate. This entry never changes Frontier or vLLM source.

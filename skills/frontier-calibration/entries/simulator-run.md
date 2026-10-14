# Simulator Run

**When to use:** Use when a frozen Frontier case is ready to produce simulator metrics and diagnostic traces.

Read `../references/contract.md` and `../references/tool-boundaries.md`. Require
the same manifest and a passed semantic check. For an online dispatch study,
consume exactly the aligned trace named by the manifest; for other cases use the
declared request trace without rewriting it.

Write `runs/frontier/run_manifest.json` before launch with command, working
directory, environment, Frontier revision, case id, generation, KV budget,
dispatch-trace path, and expected outputs. Run one command, retain compact
stdout/stderr, verify exit code and every declared metric/trace, and reject
missing, stale, negative, or non-finite values.

Keep request metrics separate from workflow, operator, residual, and CPU traces.
This entry executes a case only; it never changes Frontier or vLLM source.

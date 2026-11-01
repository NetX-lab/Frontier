# CPU Overhead Supplement

**When to use:** Use when CPU analysis reports missing or unadmitted host rows and the user approved the measurement work for the task.

Read `../references/contract.md`, `../references/tool-boundaries.md`, and
`../references/pitfalls.md` (Ground-Truth Collection, Admission and Tables).
Require an explicit missing tuple list from `cpu-overhead-analysis`, its
predicate in `../references/call-graph.md`, and the user's recorded approval.
Plan each tuple as `model, batch, TP, prefill tokens, decode tokens, scheduling
mode, precision, phase`.

Collect the bound engine's host observations on the same node group as paired
clean runs. Validate probe cost and host-speed comparability under a declared
admission method before creating tables. Keep diagnostic probes isolated and
preserve all valid clean runs for the E2E population. Missing native host spans
or an incompatible table producer require a scoped capability change; no rows
may be invented to satisfy a vLLM schema.

For the existing vLLM profile, use the `cpu` mode and admission rules in
`../references/tool-boundaries.md`, Existing vLLM Collection Profile. The
`run_gate.py admit` rules are `client_e2e`, plus `send_span` for the relevant
PDD observations or `isolated_prefill` for the recorded DP x PP cohort. Read
its `--help` and the producer before applying a rule. A rejected probe requires
new evidence or an explicitly recorded, justified deviation; it never enters
silently.

That profile produces tables with
`python -m frontier.profiling.cpu_overhead.vllm_cpu_probe`; complete pinned
recipes are in `tests/comparison/e2e_gate/artifacts.json`. Other engines need
a verified mapping of their spans to the actual Frontier table fields. Preserve
measurement families, step/stage ownership, overlap, and units in that mapping.

Publish into a new `inputs/cpu_overhead_<generation>/` with `provenance.json`:
engine/build identity, producer commit and arguments, input hashes and run
checks, admission evidence, and output hashes. The existing vLLM producer
rejects logs missing its request-phase and KV-load fields; this is its input
contract, not a requirement that another engine implement those exact probes.
Never edit a published table. The existing pinned rebuild path remains
`run_gate.py cpu-table --producer-tree` for compatible published recipes.

Return to `cpu-overhead-analysis` with a new settings generation and fresh
Frontier runs. This entry changes data only and never authorizes a code change.

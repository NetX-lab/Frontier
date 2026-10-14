# v2 Registry

Read `registry.json` as the fixed entry catalog and candidate dependency
inventory. It contains 17 atomics and 3 composites. Every entry has one owner, a
trigger, input paths, fixed output paths, a direct code-change permission, stop
statuses, and a validation command. The `calls` array on a composite lists
possible dependencies; it is not an unconditional execution order.

Use `call-graph.md` as the authoritative execution route for
predicates, conditional branches, stops, return edges, and one-domain-at-a-time
ordering. A `calls` entry never bypasses the analysis-before-code-change or
human review gate.

The v2 reductions are deliberate:

- `op-trace-normalize` is an internal phase of `op-gap-analysis`;
- PP, collective, and KV/M2N repairs share `residual-repair` with a lane
  registry;
- semantic, workflow, operator, and CPU repairs remain separate owners;
- operator and CPU supplements remain separate owners and share only the
  lossless merge mechanism;
- compatibility aliases remain outside this registry;
- composites retain separate start, read-only analysis, and code-change goals.

The top-level `lane_registry` is the only classification source for residual
repair. It maps the analysis label (`classification`) to the repair lane,
required evidence fields, the analysis-provided `frontier_owner`, and a
lane-specific receipt path. `mixed` and `still_unclear` are classifier outcomes
without a repair lane and therefore stop before any code change.

An entry may call another entry only through the dependency graph in
`call-graph.md`. A status of `PASS` describes evidence validity; it never grants
code-change permission by itself.

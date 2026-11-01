# v2 Registry

`registry.json` is the fixed entry catalog and candidate dependency inventory:
17 atomics and 3 composites. Every entry has one owner, a trigger, inputs, fixed
output paths, a direct code-change permission, stop statuses, and a validation
summary. The `calls` array of a composite lists possible dependencies; it is
not an execution order. Engine/build selection and capability bindings are
case inputs (`engine-integration.md`); they do not add per-engine entries or
change this registry mechanism.

`call-graph.md` is the authoritative route: predicates, branches, stops, return
edges, and one-domain-at-a-time ordering. A `calls` entry never bypasses the
analysis-before-code-change rule or the human review gate in
`repair-approval.md`. A status of `PASS` describes evidence validity; it never
grants code-change permission.

Deliberate reductions:

- `op-trace-normalize` is an internal phase of `op-gap-analysis`, and the error
  budget is an internal phase of `error-analysis`;
- PP, collective, and KV/M2N repairs share `residual-repair` through the lane
  registry;
- semantic, workflow, operator, and CPU repairs keep separate owners;
- operator and CPU supplements keep separate owners and separate producers;
- compatibility aliases stay outside this registry (`migration.md`).

The top-level `lane_registry` is the only classification source for residual
repair. It maps the analysis label (`classification`) to the repair lane, the
required evidence fields, the analysis-provided `frontier_owner`, and the
lane-specific receipt path. `mixed` and `still_unclear` have no repair lane and
stop before any code change.

Output paths are relative to the case directory, except Frontier runs, which go
under `$FRONTIER_TMP_ROOT`.

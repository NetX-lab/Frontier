# Semantic Alignment Check

**When to use:** Use when Frontier and the selected engine may give settings different topology, scheduling, routing, metric, or runtime meaning.

Read `../references/contract.md` (Case Binding, MoE routing gate),
`../references/engine-integration.md`, and `../references/evidence-tiers.md`.
Compare native configuration and effective runtime values with Frontier's
settings and command:

- model/tokenizer, dtype, weight identity, generation policy, request shapes,
  warmups, and formal window;
- native parallel groups, placement, and participant roles mapped to Frontier
  lanes and collective domains; use the repository `AGENTS.md` vLLM mapping
  only for a build to which it applies;
- scheduling policy and limits, chunking, prefix reuse, admission/preemption,
  and the capabilities of the selected Frontier scheduler;
- graph execution and capture shapes, attention/kernel backend, measurement
  families, and the profiling generation named by the settings;
- KV capacity and allocation/reuse/freeing semantics, transfer protocol and
  overlap, and communication parameters; equal memory bytes do not prove the
  same usable cache or lifecycle;
- actual metric endpoints, clocks, token denominators, and measurement windows;
  the current gate's TTFT means arrival to prefill completion. A different
  engine/client definition needs a proven conversion or an explicit gap;
- selected engine/build and intentional fork behavior. Compare against the
  declared fork base when it explains a difference; no fixed upstream version
  defines the behavior for all cases.

Record each row's native field, effective value, unit, source anchor, Frontier
mapping, and `MATCH`, `MISMATCH`, `UNSET`, or `UNSUPPORTED`, with impact and owner.
Missing or unsupported semantics block the affected accuracy claim. Resolve
unknown required values from inspected configuration and source; put remaining
material choices to the user under the existing decision rules. Do not replace
the reference engine or force a similar-looking Frontier backend to pass.

Write `analysis/semantic_alignment_table.csv`, summary, and status with
`analysis_state`, `status`, `correction_state`, and `next_action`. This entry is
read-only and never grants a code change.

## MoE Routing Audit

Apply the criterion from `contract.md`, MoE routing gate. Record router path,
scoring, `top_k`, weighting, expert count, EP scope, and the load distribution
separately. Use source and representative diagnostics to establish equivalence
and report unsupported native routing instead of replacing it automatically.

For an explicitly selected vLLM forced-routing case, `moe_uniform_routing`
selects the round-robin expert IDs used by the existing `balanced` comparison.
Join its diagnostic records by formal request, logical batch, layer, and phase;
that deterministic profile requires `count_abs_error = 0`. Other native routers
use their declared semantic/distribution criterion, not this synthetic setting.
Record the supported routing status in the manifest and registered case so the
existing gate can enforce it.

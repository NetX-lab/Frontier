# Semantic Alignment Check

**When to use:** Use when Frontier and vLLM settings may have different topology, scheduling, metric, or runtime meaning.

Read `../references/contract.md` and `../references/evidence-tiers.md`. Compare
user-approved settings and effective defaults row by row across model, dtype,
request shape, scheduler, chunked prefill, prefix cache, graph/eager mode,
parallel domains, KV budget, backend, warmup, metrics, routing, and selected
checkouts. Keep attention and MoE parallel meanings separate.

Record source field, effective value, unit, source anchor, and one of
`MATCH`, `MISMATCH`, `UNSET`, or `UNSUPPORTED`; assign impact and an owner.
Unknown defaults, routing differences, prefix-state contamination, checkout
differences, or metric-window differences block the E2E gate. Invoke `$grill-me`
for any missing setting that changes semantics; record its decision id, question,
evidence-backed options, answer, requesting user, reviewer identity, and UTC time
in the case manifest. Resolve the decision before continuing and do not guess.

Write `analysis/semantic_alignment_table.csv`, summary, and status with
`analysis_state`, `status`, `correction_state`, and `next_action`. This entry is
read-only and never grants a code change.

## MoE Routing-Distortion Audit

For every MoE case, audit routing semantics separately from the parallelism
audit. Treat the ground truth router as the reference. Record the model-native
router implementation, scoring function, `top_k`, weight renormalization,
expert count, EP participant scope, and any routing bias or grouped-top-k
settings for both sides. Distinguish a routing implementation path (for
example, `standard_fused_topk` or `uniform_topk`) from a load-distribution
label (for example, balanced, random, or runtime); never use one as a proxy
for the other.

Join instrumented routing records by formal request, logical batch, layer, and
phase. Compare at least these observed quantities when available: total routed
tokens, per-expert token counts, expert utilization, load-imbalance CV,
minimum and maximum load ratios, load entropy, and Gini coefficient. Compute
the per-expert signed and absolute count differences plus the relative
distortion against the ground-truth count vector. Preserve the full vectors and
the aggregation rule in the case artifacts.

Mark routing alignment `MATCH` only when the router semantics and participant
scope match and the observed distortion is either zero or explicitly covered
by an approved simulator abstraction with a recorded bound and rationale.
Mark it `MISMATCH` when routing paths, `top_k`, weighting, expert scope, or
load statistics differ without that approval. Mark it `UNSET` when either side
lacks routing records or profile coverage. Any `MISMATCH` or `UNSET` routing
result blocks the E2E gate and routes to routing analysis or supplementation;
do not compute a numeric TTFT pass over an unmeasured routing distortion.

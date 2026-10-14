# Semantic Alignment Check

**When to use:** Use when Frontier and vLLM settings may have different topology, scheduling, routing, metric, or runtime meaning.

Read `../references/contract.md` (Case Binding, MoE routing gate) and
`../references/evidence-tiers.md`. Compare the engine file with the Frontier
settings file and the effective Frontier command, row by row:

- model, dtype, weight mode, request shapes, warmups, and the formal window;
- parallel layout: map vLLM TP, DP, PP, and EP to Frontier lanes and
  collective domains per the repository `AGENTS.md` ("vLLM Parallel Semantics
  and Frontier Mapping"); keep attention and MoE parallel meanings separate;
- scheduler limits (`max_num_seqs`, `max_num_batched_tokens`, `max_model_len`,
  block size, long-prefill threshold), chunked prefill, prefix caching, and the
  scheduling policy;
- graph mode: an eager engine maps to `decode_cuda_graph_mode none`;
  `FULL_DECODE_ONLY` at compilation level 0 maps to `full_decode_only` with the
  same capture sizes;
- attention backend, measurement family of each step class, and the table
  generation named by the settings;
- KV budget (block count from the cell's clean run, watermark), KV transfer
  connector and parameters, collective parameters;
- metric endpoints: TTFT is request arrival to prefill completion on both
  sides, and the vLLM endpoint offsets the gate declares (API-server and proxy
  time) stay declared, not modeled;
- the selected checkouts; check the fork diff against upstream v0.10.2 before
  attributing a scheduling detail to vLLM.

Record each row's source field, effective value, unit, source anchor, and one
of `MATCH`, `MISMATCH`, `UNSET`, or `UNSUPPORTED`, with impact and owner. An
unknown default, a routing difference, prefix-state contamination, a checkout
difference, or a window difference blocks the E2E gate. A missing setting that
changes meaning goes to the user as a decision (contract.md, Decisions and
Approval); continue only after the answer is recorded.

Write `analysis/semantic_alignment_table.csv`, summary, and status with
`analysis_state`, `status`, `correction_state`, and `next_action`. This entry is
read-only and never grants a code change.

## MoE Routing Audit

For every MoE case, audit routing apart from the parallel layout. Record, for
both sides, the router implementation, scoring function, `top_k`, weight
renormalization, expert count, EP participant scope, any routing bias or
grouped-top-k setting, and the load distribution. Keep a routing
implementation path (for example, `standard_fused_topk` or `uniform_topk`)
distinct from a load-distribution label (`balanced`, `random`, runtime); one is
never a proxy for the other.

The reference runs with vLLM-BS round-robin routing (engine key
`moe_uniform_routing`), which swaps only the expert ids and keeps the router's
kernels; Frontier uses its `balanced` distribution. Join the `instrumented`
run's routing records with Frontier's per formal request, logical batch,
layer, and phase, and compute the per-expert count vectors and
`count_abs_error`. Record `MATCH` only for the same router semantics and scope
with `count_abs_error = 0`, `MISMATCH` for a path, `top_k`, weighting, scope,
or count difference, and `UNSET` when either side lacks records. The status
goes into the manifest and the scenario matrix's case `routing` entry, where
the gate reads it.

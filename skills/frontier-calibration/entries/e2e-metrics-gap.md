# E2E Metrics Gap

**When to use:** Use when clean Frontier and vLLM request artifacts need independent latency and throughput gate results.

Read `../references/contract.md`, `../references/evidence-tiers.md`, and
`../references/tool-boundaries.md`. Require passed run checks, at least three
clean warmups, current-generation normalized request rows, formal IDs, and
explicit first-arrival/last-completion bounds. Use the exact request normalizer
path recorded in `../references/tool-boundaries.md` with a fresh output path;
do not infer a relative `scripts/` path or use aggregate-only files.

The E2E gate compares runtime timing and throughput only. It does not compare
generated text, accuracy, quality, or model intelligence. The clean vLLM source
may use the contract's `groundtruth_weight_mode=dummy`; valid formal request rows
from that run remain eligible because the gate measures runtime behavior.

Filter the normalized inputs to the declared formal request IDs and formal replay
window. Exclude every warmup row and every setup row explicitly marked as a
dummy-shape or dummy-request row before joining or calculating metrics. A dummy
weight-loading mode alone does not mark a formal request row as excluded. Missing
or ambiguous row classification returns `INSUFFICIENT_EVIDENCE` rather than
silently including setup work in the E2E sample.

Join each formal request exactly once. Report TTFT, TPOT, request E2E, request
throughput, and token throughput independently. Latency uses the arithmetic mean
of eligible formal rows and every latency row records `metric_statistic=mean`;
median and p95 remain diagnostic fields. Use the shared TTFT semantic
`request_arrival_to_prefill_completion`.

Use `last_formal_completion_s - first_formal_arrival_s` as the formal throughput
window. Request-throughput numerator is the formal request count. Token-
throughput numerator is `sum(prefill_tokens + decode_tokens)` using the
canonical `request_num_prefill_tokens` and `request_num_decode_tokens` fields.
TPOT is eligible only when `request_num_decode_tokens > 1`; one-token rows enter
`excluded_count` and never create a zero denominator. Record
`sample_count`, `eligible_count`, `excluded_count`, `numerator`, `denominator`,
`missing_request_ids`, and `duplicate_request_ids` for every metric.

Require a positive finite denominator. A zero, negative, non-finite, or missing
denominator returns `FAIL` or `INSUFFICIENT_EVIDENCE`. Calculate relative error
as `abs(frontier - groundtruth) / abs(groundtruth)` and mark an applicable row
`PASS` only when it is at most `0.10`; one failing row keeps the case outside the
gate. Routing or prefix-state mismatch takes precedence over numeric values.

Write `analysis/e2e_metrics_table.csv`, summary, and status. This entry is
read-only and never proposes a scaling factor or code change.

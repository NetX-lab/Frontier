# Error Analysis

**When to use:** Use when a completed Frontier-vLLM case needs a read-only explanation of its E2E gap and its error sources.

Read `../references/contract.md` (Error-Source Classes),
`../references/evidence-tiers.md`, `../references/call-graph.md`, and
`../references/pitfalls.md` (Analysis and Correction).

1. Run `e2e-metrics-gap`. Stop on a cell `PASS` unless the user asked for
   diagnostics.
2. Run `semantic-alignment-check` on the failing cell, including routing,
   prefix state, and the metric windows. A mismatch stops for a decided
   semantic repair.
3. Build the error budget per case and failing metric: from the
   `schedule_timing` run (step membership and periods), the admitted `cpu` run
   (device and host split per step shape), the clean run (gate values), and the
   Frontier label's stage ledgers, split each request's Frontier-minus-vLLM
   gap into queue (TTFT only), step mix, device price, host price, and
   declared boundary terms. The budget closes when its parts add up to the gap
   within a stated residual; an open budget is `INSUFFICIENT_EVIDENCE`. Write it
   under `analysis/error_budget/`.
4. Run the analyses the budget's largest parts select (call-graph.md); route
   missing coverage to the owning supplement and return to a fresh analysis.
5. Assign every attributed part an error-source class, and rank the
   `simulation_logic` parts first by their share of the failing metric.

Write one review-ready `reports/error_analysis_summary.md`: per failing metric,
both sides' absolute values, the budget table, each part's class and owner,
proposed corrections with options and their expected effect, and the
`measurement_input` and `reference_instability` items stated as measurement
work for the user to approve or defer. Link the atomic rows and source paths
instead of copying them. This composite is read-only and never authorizes a
code change.

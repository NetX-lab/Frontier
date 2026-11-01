# E2E Metrics Gap

**When to use:** Use when clean Frontier and reference-engine request rows of a cell need the five independent E2E metric decisions.

Read `../references/contract.md` (Calibration Objective, E2E Gate, MoE routing gate),
`../references/evidence-tiers.md`, `../references/engine-integration.md`, and
`tests/comparison/e2e_gate/README.md`. Verify the selected collector/converter
and the gate's metric semantics before normalizing a new engine's rows.
Require passing post-run checks for every clean reference run, a Frontier label
run on the current settings generation, and the case routing status.

The gate compares runtime timing and throughput only, never generated text or
accuracy. Weight and routing substitutions follow the contract's declared-case
limits. Warmup rows and setup rows never enter a metric.

1. Normalize through the converter recorded in `engine_binding`, retaining raw
   records and a field/endpoint mapping receipt. For the existing compatible
   vLLM collector only: `run_gate.py normalize --cell <case_id>/<cell> --artifacts-root
   <bundle> --vllm-run-dir <run> ... --frontier-run <label>=<dir> --output-dir
   analysis/e2e_gap_<label>/`. Other engines need a verified converter to the
   same schema, as described in `engine-integration.md`; this command does not
   accept arbitrary native logs. Read `sample.json`: any missing, duplicate, or
   unclassified formal id, or a token-count mismatch, makes every metric
   `INSUFFICIENT_EVIDENCE`.
2. Gate: `run_gate.py gate --cell <case_id>/<cell> --rows
   analysis/e2e_gap_<label>/ --output analysis/e2e_gap_<label>/reading.json`.
   The cell's gate kind (`single`, `pd_s33`, `dp_pp_pairs`) pools every
   admissible reference run; list each run used. Apply `pd_s33` offsets or
   `dp_pp_pairs` conditioning only when the case validates their semantics.
   Keep the actual engine identity in the report despite legacy `vllm` keys.
3. For compatible vLLM logs only,
   `tests/comparison/calibration/e2e_metrics_gap.py` compares one reference run
   and writes `e2e_metrics_table.csv`, status, and summary; the cell verdict still
   comes from step 2.

`run_gate.py` knows only the claimed cells of
`tests/comparison/e2e_gate/cases/scenario_matrix.json`. A new case or cell
first gets its matrix entry (gate kind, trace, reference runs, recorded
Frontier command, coverage, pinned hashes) on the task branch; that entry is a
repository change reviewed together with its readings.

The metric definitions are fixed: TTFT, TPOT, and request E2E use the
arithmetic mean of eligible formal rows with `metric_statistic=mean`; TPOT
counts only rows with `request_num_decode_tokens > 1`, the others enter
`excluded_count`; request throughput uses the formal request count and token
throughput `sum(prefill_tokens + decode_tokens)` over
`last_formal_completion_s - first_formal_arrival_s`. Every metric records
`sample_count`, `eligible_count`, `excluded_count`, `numerator`,
`denominator`, `missing_request_ids`, and `duplicate_request_ids`. Require a
positive finite denominator. A metric passes when
`abs(frontier - groundtruth) / abs(groundtruth)` is at most `0.10`; one failing
metric keeps the cell outside the gate. A routing status of `UNSET` or
`MISMATCH`, or prefix-state contamination, decides the cell before any number.

Write `analysis/e2e_metrics_summary.md` and `analysis/e2e_metrics_status.json`.
The summary gives each metric's two absolute values with units, the relative
error, the gate kind, the independent clean-run count, between-run variability,
statistic uncertainty, the reference-sufficiency assessment, the coverage
class, and, when the selected gate uses mode pairs, the gated pools and
reference modes without gated pairs. Report the ordinary clean-run comparison alongside any
conditioned gate, with its stated boundary. Preserve `reading.json` as the
harness's arithmetic verdict; when reference evidence is insufficient, the
analysis status is `INSUFFICIENT_EVIDENCE` and calibration remains open even if
that reading says PASS. This entry is read-only and never proposes a scaling
factor or code change.

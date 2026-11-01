# E2E Metrics Gap

**When to use:** Use when clean Frontier and reference-engine request rows of a cell need the five independent E2E metric decisions.

Read `../references/contract.md` (Calibration Objective, Warm steady state,
E2E Gate, MoE routing gate),
`../references/evidence-tiers.md`, `../references/engine-integration.md`, and
`tests/comparison/e2e_gate/README.md`. Verify the selected collector/converter
and the gate's metric semantics before normalizing a new engine's rows.
Require passing post-run validity checks for every clean reference run, its
separate readiness result and event evidence when needed, a Frontier label run
on the current settings generation, and the case routing status.

The gate compares runtime timing and throughput only, never generated text or
accuracy. Weight and routing substitutions follow the contract's declared-case
limits. Read `contract.md`, Warm steady state, to select the comparison
population; normalization retains every formal row before that selection.

1. Normalize through the converter recorded in `engine_binding`, retaining raw
   records and a field/endpoint mapping receipt. For the existing compatible
   vLLM collector only: `run_gate.py normalize --cell <case_id>/<cell> --artifacts-root
   <bundle> --vllm-run-dir <run> ... --frontier-run <label>=<dir> --output-dir
   analysis/e2e_gap_<label>/`. Other engines need a verified converter to the
   same schema, as described in `engine-integration.md`; this command does not
   accept arbitrary native logs. Read `sample.json`: any missing, duplicate, or
   unclassified formal id, or a token-count mismatch, makes every metric
   `INSUFFICIENT_EVIDENCE`.
2. Preserve the all-request reading: `run_gate.py gate --cell <case_id>/<cell> --rows
   analysis/e2e_gap_<label>/ --output analysis/e2e_gap_<label>/reading.json`.
   Use a fresh output directory for a new generation; reuse an existing
   published reading unchanged.
   The cell's gate kind (`single`, `pd_s33`, `dp_pp_pairs`) pools every
   admissible reference run; list each run used. Apply `pd_s33` offsets or
   `dp_pp_pairs` conditioning only when the case validates their semantics.
   Keep the actual engine identity in the report despite legacy `vllm` keys.
   If every repeat is verified `WARM`, this is the deciding arithmetic reading.
   Otherwise calculate a separate `analysis/e2e_gap_<label>/warm_reading.json`:

   - For each valid clean repeat `r`, record the complete formal set `F`, the
     event-bound excluded IDs `E_r`, and retained IDs `W_r = F - E_r`. A verified
     warm repeat has `E_r = {}`. Retain the source/event anchors and the fixed
     exclusion rule for every `E_r`. Unknown readiness or unbound events leave
     the analysis `INSUFFICIENT_EVIDENCE`; step 1 admission failures cannot be
     hidden by exclusion. Keep every repeat, even when `W_r` is empty.
   - Select `W_r` from repeat `r` and from its matching Frontier label or each
     ensemble member. For each side recompute the window as
     `max(completion_s for W_r) - min(arrival_s for W_r)` on its own clock.
     Use only retained request/token counts in throughput numerators and the
     ordinary metric eligibility rules below for latency. Preserve original
     timestamps and execution context: internal elapsed time, event durations
     and queueing paid by retained requests are not subtracted, and the workload
     is not rerun as a smaller trace. Record both sides' per-repeat counts,
     window endpoints, numerators, denominators and values, with warm exclusions
     separate from TPOT ineligibility.
   - For `single` and `pd_s33`, sum numerators and denominators across repeats
     on **both** sides. Reuse the Frontier rows filtered by that repeat's `W_r`
     once per reference repeat; one unfiltered Frontier value is not a matched
     comparison when retained IDs differ. For `pd_s33`, compute and record each
     repeat's minimum endpoint offset on its retained reference rows as in
     `gate.py`. For `dp_pp_pairs`, pair every repeat with every ensemble member
     on `W_r`, then apply the existing mode-agreement pools, distinct-request
     threshold and arithmetic on both sides. Preserve original mode windows
     and labels. Report whole-cell reference means over repeats and Frontier
     means over the matching retained-row values of every (repeat, member)
     pair; these remain diagnostics, not gated metrics.
   - Retain a replayable calculator path/command, version, input hashes and the
     per-repeat exclusion receipt. Reuse `e2e_metrics_gap.py` metric arithmetic
     and `gate.py` pooling/decision rules. The current harness has one shared
     `sample.formal_request_ids` and one Frontier set for `single`/`pd_s33`;
     one `run_gate.py gate` call cannot express differing `W_r`. Use a verified
     case-local calculation for that derived reading, not a cell-wide union or
     intersection of exclusions, edited `sample.json`, or an overwritten native
     `reading.json`. Empty retained sets, invalid denominators or insufficient
     reference precision keep the analysis insufficient, without discarding a
     repeat. Clearly label the result as a derived warm reading, not a native
     harness verdict.

3. Write the cell analysis status from the deciding reading in step 2, subject
   to admission, readiness and reference sufficiency. A valid derived warm
   reading decides it even when the all-request harness verdict differs; keep
   both readings and their scope on record. For compatible vLLM logs only,
   `tests/comparison/calibration/e2e_metrics_gap.py` compares one reference run
   and writes `e2e_metrics_table.csv`, status, and summary as a quick diagnostic;
   it does not replace the repeat-pooled reading selected in step 2.

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
reference modes without gated pairs. For a warm reading, include the per-repeat
exclusion and matched-pooling receipt, its replay command and each excluded
request's event evidence, beside the unchanged all-request reading. Report the
ordinary clean-run comparison alongside any conditioned gate, with its stated
boundary. Preserve `reading.json` as the native harness arithmetic verdict and
identify which reading decides analysis status. When reference evidence is
insufficient, the analysis status is `INSUFFICIENT_EVIDENCE` and calibration
remains open even if an arithmetic reading says PASS. This entry is read-only
and never proposes a scaling factor or code change.

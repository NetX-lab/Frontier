# v2 Call Graph

This graph owns execution order, predicates, stops, and return edges. A
`calls` list in `registry.json` names candidates only.

## Start a Case

```text
parity-run
  -> semantic-alignment-check
  -> [blocking mismatch] semantic-alignment-repair (decided only)
  -> semantic-alignment-check (fresh)
  -> groundtruth-run-check (preflight)
  -> groundtruth-run
  -> groundtruth-run-check (post-run, every run tag)
  -> [DP x PP cell or online dispatch skew] dispatch-align-trace
  -> simulator-run
```

`groundtruth-run-check` runs twice: before launch it checks the manifest,
checkout, mode predicate, and fresh output paths; after the run it checks the
run's evidence, and only a passing run reaches any analysis. Clean,
diagnostic, CPU, and Frontier outputs stay in separate run directories.

## Optional Workflow Calibration

Select this branch for a new model or when timing-estimation error obscures a
workflow check. `layer-timing-calibration.md` owns its prerequisites, timing
boundaries, diagnostic acceptance, and return to finer timing estimates.

```text
workflow-gap-analysis (optional measured-layer path)
  -> semantic-alignment-check
  -> groundtruth-run-check (preflight)
  -> groundtruth-run (clean and required timing observations)
  -> groundtruth-run-check (post-run, every run tag)
  -> [DP x PP cell or online dispatch skew] dispatch-align-trace
  -> measure/admit full-model layer and stage timing coverage
  -> diagnostic simulator run with measured times and native workflow
  -> compare key metrics and workflow events
  -> [timings admitted, reproducible workflow mismatch] workflow-repair (decided only)
  -> [workflow check passes] operator/kernel and CPU timing refinement
  -> simulator-run (native prediction) -> e2e-metrics-gap
```

Reuse passing runs of the same case generation. This branch may precede the
native `simulator-run` and fine-grained profiling; missing operator tables do
not by themselves block it. An unavailable target allocation or timing adapter
leaves this optional check unperformed and returns to the ordinary path. A
coverage or measurement-admission failure leaves its conclusion
`INSUFFICIENT_EVIDENCE`; it is not evidence of a workflow defect.

## Find an Error Source

```text
error-analysis
  -> e2e-metrics-gap (arithmetic and clean-reference sufficiency)
  -> [insufficient repetitions] collect approved clean repeats; reassess
       before attributing modeling error
  -> [sufficient reference, every gated metric PASS] stop, unless the user asked for diagnostics
  -> [a metric FAIL or INSUFFICIENT_EVIDENCE]
       -> semantic-alignment-check (settings, routing, prefix state, windows)
       -> [mismatch] stop for a decided semantic-alignment-repair
       -> [semantic PASS, optional workflow check selected]
            workflow-gap-analysis (measured-layer path above)
       -> [semantic PASS] error budget (internal phase of error-analysis)
       -> analyses chosen by the budget's largest parts:
            queue or step mix   -> workflow-gap-analysis
            device price        -> op-gap-analysis
                                   -> [first batch or three-batch gate missing] op-supplement
                                   -> fresh op-gap-analysis
            host price          -> cpu-overhead-analysis
            PP, collective,
            KV or M2N transfer  -> residual-lane-classification
       -> classify every attributed part (contract.md, Error-Source Classes)
```

The error budget splits each failing metric's Frontier-minus-vLLM gap per
request into queue, step mix, device price, host price, and declared boundary
terms, and closes to the gap within a stated residual. The analyses it selects
are read-only and may run in any order. `op-gap-analysis` normalizes the
operator trace internally and keeps `analysis/op_trace_normalized.jsonl` as an
intermediate artifact.

## Apply Decided Corrections

```text
error-correction
  -> representative-behavior rationale and fresh-evidence validation plan
       (contract.md, Calibration Objective; repair-approval.md)
  -> one decided domain change (one ticket, one commit)
  -> focused tests and the regression set (repair-approval.md)
  -> rerun the affected cells
  -> fresh domain analysis and a fresh decision
  -> next domain
  -> [groundtruth_defect fixed] parity-run with the new ground-truth commit
  -> e2e-metrics-gap
  -> republish the readings
```

The domain choices are `semantic-alignment-repair`, `workflow-repair`,
`op-repair`, `residual-repair(lane=pp|collective|kv_or_m2n_transfer)`, and
`cpu-overhead-repair`. Every edge needs the approval in `repair-approval.md`.

The CPU edge opens last for a metric. `cpu-overhead-repair` needs workflow
and operator analysis of that metric complete, both with `status=PASS` and
`correction_state` in `{complete,not_applicable}`, and a fresh clean E2E
generation whose applicable relative error is still greater than `0.10`, so
host terms never absorb a logic error. `cpu-overhead-supplement` needs the same
predicate plus the user's approval, because new CPU-probe collection and table
tuning are measurement engineering.

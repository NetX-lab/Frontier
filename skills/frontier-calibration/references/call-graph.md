# v2 Call Graph

## Start a Case

```text
parity-run
  -> semantic-alignment-check
  -> [blocking mismatch] semantic-alignment-repair (approved only)
  -> semantic-alignment-check (fresh)
  -> groundtruth-run-check
  -> groundtruth-run
  -> simulator-run
```

`groundtruth-run-check` always precedes `groundtruth-run`. An online case may
insert `dispatch-align-trace` before `simulator-run` when observed dispatch skew
is causal. Clean, instrumented, CPU, and Frontier outputs stay in separate mode
directories.

## Find an Error Source

```text
error-analysis
  -> e2e-metrics-gap
  -> [any applicable metric > 10%]
       -> semantic-alignment-check
       -> [semantic mismatch] stop for approved semantic-alignment-repair
       -> [semantic PASS]
            -> [online skew] dispatch-align-trace
            -> workflow-gap-analysis
            -> op-gap-analysis
            -> [first batch or three-batch gate missing] op-supplement
            -> fresh op-gap-analysis
            -> [gate passes] continue to residual-lane-classification
            -> residual-lane-classification
            -> [workflow/op corrections complete and fresh E2E still > 10%]
                 -> cpu-overhead-analysis
```

`op-gap-analysis` performs operator trace normalization internally and keeps
`analysis/op_trace_normalized.jsonl` as an intermediate artifact. Workflow and
operator analysis remain separate and mutually informative. A passing E2E case
stops lower-level analysis unless the user explicitly requests diagnostics.

## Apply Reviewed Corrections

```text
error-correction
  -> one approved domain code change
  -> isolated rerun
  -> fresh domain analysis and approval
  -> next eligible domain
  -> parity-run
  -> e2e-metrics-gap
```

The domain choices are `semantic-alignment-repair`, `workflow-repair`,
`op-repair`, `residual-repair(lane=pp|collective|kv_or_m2n_transfer)`, and
`cpu-overhead-repair`. Every edge requires the shared code-change approval
reference. The CPU edge opens only when workflow and op analysis are complete,
both have `status=PASS` and `correction_state` in
`{complete,not_applicable}`, and a fresh clean E2E
artifact from a new generation contains an applicable relative error greater
than `0.10`.

`registry.json` supplies the candidate dependency inventory for each composite;
it does not encode unconditional execution. This graph is the authoritative
source for predicates, stops, return edges, and one-domain-at-a-time ordering.
No registry `calls` list bypasses the analysis or human review gate.

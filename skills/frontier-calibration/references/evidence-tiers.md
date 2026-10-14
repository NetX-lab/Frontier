# Evidence Tiers

Use the smallest evidence set that can answer the current question. Keep every
other file for audit or troubleshooting without treating it as a second metric
source.

## Normative Evidence

These files may decide a gate when their provenance envelope is complete:

| File | Use |
| --- | --- |
| `runs/vllm-clean/request_metrics.normalized.jsonl` | Clean ground-truth request rows for TTFT, TPOT, request E2E, and throughput. |
| `runs/frontier/request_metrics.normalized.jsonl` | Same-run Frontier request rows joined to its timing sidecar. |
| `analysis/e2e_metrics_table.csv` and status | The five independent E2E gate rows and their counts. |
| `analysis/workflow_gap_*` | Workflow control-flow diagnosis after clean metric failure. |
| `analysis/op_trace_normalized.jsonl` and `analysis/op_gap_*` | Operator diagnosis after the aligned-batch gate. |
| `analysis/residual_gap_*` | Residual source classification when workflow/op evidence is complete. |

The workflow, operator, and residual files are normative diagnostic evidence for
their own analysis decisions; they never replace clean E2E values.

## Conditional CPU Evidence

Use `runs/cpu/*` and `analysis/cpu_overhead_*` only when workflow and operator
correction attempts are complete, a fresh clean E2E generation still has an
applicable relative error greater than `0.10`, and the CPU producer has passed
its own identity, row-count, freshness, and bundle-publication review. Until a
new CPU producer analysis receives a human `PASS` (or an explicit immutable
`auto_recycle` receipt), CPU artifacts remain non-normative watch evidence.

## Reference and Audit Files

Keep these files for reproducibility, consistency checks, and debugging:

- `manifest.yaml`, `case_init`, `run_manifest.json`, `status.json`, and
  `producer.json`;
- command receipts, stdout/stderr, checkout and environment metadata;
- `request_id_mapping.json`, completion markers, digests, plots, Chrome traces,
  and `system_metrics.json`;
- raw scheduler, event, collective, transfer, and component logs not selected
  by the active analysis.

These files establish identity or explain a result. They do not become an
additional numeric gate merely because they contain a latency or throughput
aggregate.

## Frontier Metric Overlap

Frontier already emits per-request `request_metrics.csv` and the same-run
`metrics_ground_truth.jsonl` timing sidecar. The normalized Frontier JSONL is a
derived join view: it resolves request identity, formal-window membership, and
TTFT provenance; it is not a new measurement source. Frontier
`system_metrics.json` contains cluster aggregates and can corroborate a run,
but it cannot provide the one-row-per-formal-request join required by the E2E
gate. `TraceStore` and scheduler/event ledgers have different granularity and
remain the useful sources for workflow/operator/residual diagnosis.

## Minimal Selection Rules

1. E2E comparison consumes only clean normalized request rows plus their
   producer/status records.
2. Workflow analysis adds scheduler and event traces; online cases add the
   dispatch-aligned trace only when dispatch skew is causal.
3. Operator analysis adds the normalized op stream and profiling coverage only
   after first-batch and three-aligned-batch predicates pass.
4. Residual analysis adds only the lane traces named by its classifier.
5. CPU analysis adds CPU rows only after its explicit entry predicate.

The case report records which tier was used and lists retained files that were
not consulted for the decision.

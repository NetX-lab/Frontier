# Evidence Tiers

Use the smallest evidence set that answers the current question. Keep every
other file for audit without treating it as a second metric source. Paths are
relative to the case directory unless they name the repository or the scratch
root.

## Gate Evidence

These files decide the E2E gate when their run check passed and their
provenance is complete:

| File | Use |
| --- | --- |
| `runs/groundtruth_clean/<run_tag>/run/request_metrics.jsonl` | co-location clean vLLM request rows |
| `runs/groundtruth_clean/<run_tag>/run/pd_request_metrics.jsonl` | PDD clean rows joined by `tests/comparison/calibration/pd_metrics.py` (proxy arrival, prefill first token, decode completion; unjoined requests in `pd_join_report.json`) |
| `analysis/run_check_status.groundtruth_clean.json` | the run check that admits each clean run |
| `$FRONTIER_TMP_ROOT/.../<case_id>/<label>/metrics/.../request_metrics.csv` | Frontier request rows of the gated label |
| `run_gate.py normalize` output: `sample.json`, `vllm/<run_tag>.csv`, `frontier/<label>.csv` | normalized rows and their identity gaps |
| `run_gate.py gate --output` JSON | the cell reading: verdict, each gated metric or mode pool, routing, coverage |
| `tests/comparison/e2e_gate/cases/published_readings.json` | the published reading of every claimed cell |

`tests/comparison/calibration/e2e_metrics_gap.py` writes
`e2e_metrics_table.csv`, `e2e_metrics_status.json`, and
`e2e_metrics_summary.md` for one vLLM run; use it for a quick single-run check.
A cell's verdict comes from `run_gate.py gate`, which pools every admissible
reference run.

## Diagnostic Evidence

These files decide their own analysis and never replace clean E2E values:

| File | Analysis |
| --- | --- |
| `runs/groundtruth_schedule_timing/<run_tag>/` scheduler log | workflow: step membership, queueing, periods |
| `extract_vllm_placement.py` and `compare_placement.py` outputs | workflow: DP placement, engine iterations, dummy passes |
| `op_gap_analysis.py` outputs from one clean, one `op_timing`, and one `kernel_timing` run of the same trace | operator: scope times, uncovered window, host gaps, measurement cost |
| `run/device_timeline.json` from `forward_device_timeline.py` | operator and residual: device busy and idle, all-reduce wait |
| `instrumented` routing records and the routing audit | MoE routing status |
| the error budget per case and metric | which analyses to run |
| `analysis/workflow_gap_*`, `analysis/op_gap_*`, `analysis/residual_gap_*`, `analysis/cpu_overhead_*` | each entry's table, summary, and status |

Measuring changes behavior: an `op_timing` run serializes scopes and a
`kernel_timing` run syncs the device. Record the measurement cost
(`measurement_cost.csv`) separately from the behavior it changes.

## Measurement Inputs

Profiling table sets, `kernel_gap.csv`, CPU-overhead tables, and collective or
KV-transfer fits feed Frontier's pricing. They are neither gate evidence nor
diagnosis; each carries its own provenance (producer command, input hashes,
output hashes) and admission record. A CPU-probe run enters a CPU-overhead
table only through the admission rules of `cpu-overhead-supplement`.

## Reference and Audit Files

Keep for reproducibility and debugging: `manifest.yaml`, `case_init.md`,
`run_manifest.json`, launcher receipts, command receipts, stdout and stderr,
checkout and environment metadata, completion markers, digests, plots, Chrome
traces, `system_metrics.json`, and raw logs that no active analysis selected.
They establish identity or explain a result; an aggregate they contain is never
an additional gate.

## Repository Replay Depth

The gate harness reproduces a published number to the depth of the tier you
can run (`tests/comparison/e2e_gate/README.md`, "Evidence tiers"): gate replay
from normalized rows, row normalization from raw runs, CPU-table rebuild from
pinned probe logs, trained Frontier replay from pinned tables, and native
collection on a GPU node. State the tier a claim rests on.

## Frontier Metric Overlap

Frontier writes per-request `request_metrics.csv`; enable
`metrics_ground_truth.jsonl` when throughput needs an independent wall-clock
cross-check. `system_metrics.json` holds cluster aggregates and cannot provide
the one-row-per-formal-request join the gate needs. Event, stage, and op traces
(`--op-trace` runs, under their own label) serve diagnosis only.

## Minimal Selection Rules

1. The E2E gate reads only normalized clean rows plus their run checks.
2. Workflow analysis adds scheduler and placement evidence; DP x PP cells add
   the mode labels and the dispatch-aligned members.
3. Operator analysis adds the op/kernel run pair and profiling coverage after
   the first-batch and three-aligned-batch predicates pass.
4. Residual analysis adds only the lane evidence its classifier names.
5. CPU analysis adds CPU-probe rows only after its entry predicate.

The case report names the tier each decision used and lists retained files it
did not consult.

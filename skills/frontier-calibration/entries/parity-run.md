# Parity Run

**When to use:** Use when a user needs a reproducible Frontier and vLLM-BS case, or new cells of a case, collected with aligned semantics.

Read `../references/contract.md`, `../references/evidence-tiers.md`,
`../references/call-graph.md`, and `../references/pitfalls.md`.

## Set Up the Case

1. Put every missing setting to the user as a decision: model, architecture,
   parallel layout, scheduler limits, graph mode, routing, KV budget, workload,
   load levels, device generation, and GPU resources. Record each answer under
   its decision ID before writing files.
2. Write `manifest.yaml` and `case_init.md` (contract.md, Case Binding), the
   engine file per mode, the Frontier settings file of the first table
   generation, and `workload.json`. Generate the trace and request ids with
   `tests/comparison/dp_placement_pp/make_trace.py`: at least three warmups per
   prompt-length class, drained before the formal window.
3. Size load levels from measured capacity. A low, medium, and high cell per
   case is the published pattern; DP x PP cells take three clean runs each
   because their lane mode varies between runs. Choose a medium load that keeps
   a small TPOT error from compounding into TTFT queueing.

## Collect

Follow `../references/call-graph.md`, "Start a Case": `semantic-alignment-check`,
the preflight of `groundtruth-run-check`, `groundtruth-run` (warm pass, clean,
and the diagnostic modes in one job), the post-run check of every run tag,
`dispatch-align-trace` for DP x PP or dispatch-skewed cells, and
`simulator-run` with the cell's own KV block count. A blocking semantic result
stops for a decided semantic repair and a fresh check.

Keep clean, diagnostic, CPU, and Frontier outputs in separate run directories
and retain every command receipt. The case is ready for `e2e-metrics-gap` when
each cell has passing clean runs and a Frontier label on the current settings
generation.

This composite launches and records a case. It neither diagnoses a gap nor
applies a code change.

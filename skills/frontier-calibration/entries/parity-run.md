# Parity Run

**When to use:** Use when a user needs a reproducible Frontier and reference-engine case, or new cells of a case, collected with aligned semantics.

Read `../references/contract.md`, `../references/evidence-tiers.md`,
`../references/call-graph.md`, and `../references/pitfalls.md`.

## Set Up the Case

1. Bind the user-selected engine/build and check the required capabilities in
   `../references/engine-integration.md`. Resolve effective settings from the
   selected configuration; put remaining required choices to the user: model, architecture,
   parallel layout, scheduler limits, graph mode, routing, KV budget, workload,
   load levels, device generation, and GPU resources. Record each answer under
   its decision ID before writing files.
2. Write `manifest.yaml` and `case_init.md` (contract.md, Case Binding), the
   native engine configuration per mode, the Frontier settings file of the first table
   generation, and `workload.json`. Use the bound workload generator/replayer and preserve request IDs across
   both sides. `tests/comparison/dp_placement_pp/make_trace.py` is available for
   its supported trace format; warmup policy follows the selected collector
   and the shared contract.
3. Size load levels from measured capacity. A low, medium, and high cell per
   case is the published pattern. Plan independent clean repetitions for each
   cell using the contract's reference-sufficiency rule; recurring engine modes
   are part of the variability to capture. Cover the load levels users need,
   including queueing sensitivity, and retain all admissible repeats.

## Collect

Follow `../references/call-graph.md`, "Start a Case": `semantic-alignment-check`,
the preflight of `groundtruth-run-check`, `groundtruth-run` (warm pass, clean,
and the diagnostic modes in one job), the post-run check of every run tag,
`dispatch-align-trace` when dispatch skew affects the comparison, and
`simulator-run` with the cell's observed KV capacity and verified mapping. A blocking semantic result
stops for a decided semantic repair and a fresh check.

Keep clean, diagnostic, CPU, and Frontier outputs in separate run directories
and retain every command receipt. The case is ready for `e2e-metrics-gap` when
each cell has passing clean runs and a Frontier label on the current settings
generation. Assess the reference population's sufficiency separately from each
run's validity; insufficient repeats require more collection within the approved
budget before a calibration conclusion.

This composite launches and records a case. It neither diagnoses a gap nor
applies a code change.

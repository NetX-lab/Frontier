# Ground-Truth Run Check

**When to use:** Use when a ground-truth run needs its preflight before launch, or its evidence check after the run before any analysis reads it.

Read `../references/contract.md` (Ground-Truth Engine and Build, Run Modes),
`../references/engine-integration.md`, and `../references/pitfalls.md`
(Ground-Truth Collection). Check the mode switches and output contract of the
selected collector. This check is case-local (`../references/tool-boundaries.md`).

## Preflight

Before launch, record in `runs/groundtruth_<mode>/<run_tag>/run_manifest.json`:

- engine name/version and pinned source or installed-build identity, including
  any instrumentation overlay; verify the source/build rules in the contract;
- the mode, native configuration, required collector/converter capabilities,
  and the conditions separating clean from diagnostic observations;
- warmup coverage and readiness for the measured shapes, with warmups drained
  before the first formal arrival;
- formal/warmup request IDs, workload and trace hashes, and fresh output paths;
- authorized resources and topology, paired clean/probe placement, and port
  isolation for concurrent jobs.

A failed required preflight check stops the affected run. A missing optional
probe leaves that diagnostic unperformed and does not stop supported clean
collection.

## Post-Run Check

For each run tag, write `analysis/run_check_status.groundtruth_<mode>.json`
with a PASS or FAIL for every required check and the evidence:

- serving build, overlays, configuration, inputs, and collector unchanged;
  actual GPU type/count and topology match the case; successful terminal status
  and exit code plus the expected outputs;
- the mode predicate holds; every formal request appears once with the observed
  token counts and declared generation/termination policy; warmups are separate;
- metric clocks, units, endpoints, and request joins match the recorded mapping;
  preserve missing and duplicate records through any conversion;
- the observed KV capacity and allocation semantics map to Frontier's cache
  configuration; obtain a block count only when the engine uses that concept;
- required workflow/probe observations cover the requested phases, participants,
  and ranks, with measurement cost recorded separately.

Apply collector-specific checks only when relevant. The existing vLLM PDD
collector requires both connector checks and the `pd_metrics.py` request join;
its CPU producer needs step, request-phase, and stage records. Such formats are
not requirements for another engine's logs.

A run passes when every applicable required check passes. Mark inapplicable
checks with their reason. All valid clean runs belong to the declared reference
population; probe admission into timing tables remains separate. Missing a
required measurement is insufficient evidence, not proof of a simulator defect.

This entry changes no code or data. A run PASS establishes validity, not numeric
parity or sufficient independent repetitions. Assess case-level sufficiency
under `contract.md`, Calibration Objective, retaining ordinary valid slow runs.

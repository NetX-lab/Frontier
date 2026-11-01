# Ground-Truth Run Check

**When to use:** Use when a ground-truth run needs its preflight before launch, or its evidence check after the run before any analysis reads it.

Read `../references/contract.md` (Ground-Truth Engine and Build, Run Modes,
Warm steady state),
`../references/engine-integration.md`, and `../references/pitfalls.md`
(Ground-Truth Collection). Check the mode switches and output contract of the
selected collector. This check is case-local (`../references/tool-boundaries.md`).

## Preflight

Before launch, record in `runs/groundtruth_<mode>/<run_tag>/run_manifest.json`:

- engine name/version and pinned source or installed-build identity, including
  any instrumentation overlay; verify the source/build rules in the contract;
- the mode, native configuration, required collector/converter capabilities,
  and the conditions separating clean from diagnostic observations;
- the warmup policy, expected path/connection coverage, and the method and
  records used to verify readiness and drain at the formal boundary
  (`contract.md`, Warm steady state); preflight checks this plan, not coverage
  that can only be observed during the run;
- formal/warmup request IDs, workload and trace hashes, and fresh output paths;
- authorized resources and topology, paired clean/probe placement, and port
  isolation for concurrent jobs.

A failed required preflight check stops the affected run. A missing optional
probe leaves that diagnostic unperformed and does not stop supported clean
collection.

## Post-Run Check

For each run tag, write `analysis/run_check_status.groundtruth_<mode>.json`
with separate clean-run validity and warm-readiness results. For validity,
record PASS or FAIL for every applicable required check and its evidence:

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

A run's validity passes when every applicable required validity check passes.
Mark inapplicable checks with their reason. Separately record warm readiness as
`WARM`, `INCOMPLETE`, or `UNVERIFIED`, with the actual coverage identities,
completion records, warmup drain and first formal arrival. Audit formal-window
events against the preflight method. Compatibility success alone is not proof
that connection setup completed. For `INCOMPLETE`, retain source/event anchors
and the exhaustive affected-request IDs needed by `e2e-metrics-gap`; unverified
drain or incomplete event binding leaves warm evidence insufficient.

Every valid clean repeat reaches analysis, including an incompletely warmed
repeat with complete request rows and event-exclusion evidence. Its validity
PASS is not changed to FAIL solely by its readiness result. The deciding reading
follows `contract.md`, Warm steady state; all-request rows and valid repeats
remain intact. Probe admission into timing tables remains separate. Missing a
required measurement is insufficient evidence, not proof of a simulator defect.

This entry changes no code or data. A run PASS establishes validity, not numeric
parity or sufficient independent repetitions. Assess case-level sufficiency
under `contract.md`, Calibration Objective, retaining ordinary valid slow runs.

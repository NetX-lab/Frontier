---
name: frontier-calibration
description: Use when calibrating Frontier against a chosen inference engine or fork, checking simulator workflow with measured layer timings, collecting reproducible cases, gating E2E metrics at 10%, or diagnosing and correcting simulation errors.
---

# Frontier Calibration

Calibration compares Frontier with a user-selected inference engine or fork,
such as vLLM, SGLang, or TensorRT-LLM, on the same workload. It checks clean E2E
metrics and traces each gap to a source. First verify the engine's collectors,
metric conversion, and Frontier semantic support. Existing vLLM tools are
usable only for compatible builds and formats; this skill does not supply ready-made runners
for every engine.

Model representative behavior of the selected engine under the user's
configuration. Establish a sufficient repeated clean-run reference before
attributing error or claiming accuracy. Fine-grained traces explain causes; ordinary users must be able to
assess predictions with clean runs. Follow `references/contract.md`,
"Calibration Objective", before collecting evidence or proposing a correction.

If `site.local.md` exists beside this file, read it first. It binds the reference
engine checkout or installed build, artifacts root, `$FRONTIER_TMP_ROOT`, and
GPU platform to this host and is not tracked. The case may supply those values
instead. Resolve only missing required values; a local default never overrides the user's selected engine.

## Read Before a Case

1. `references/contract.md`: calibration objective and reference sufficiency,
   case binding, engine/build identity, run modes, the E2E gate, error-source
   classes, decisions.
2. `references/engine-integration.md`: select and pin the reference build,
   check required capabilities, and validate its comparison path. Reuse the
   binding for subsequent cases only while its assumptions remain valid.
3. `references/call-graph.md`: execution order, predicates, stops, and return
   edges. It overrides any order implied elsewhere.
4. `references/pitfalls.md`: failures that earlier calibration tasks hit.
5. `references/evidence-tiers.md`: which artifact may support which claim.

Read on demand:

- `references/layer-timing-calibration.md` for the optional workflow check
  using full-model layer timings, before operator/kernel calibration;
- `references/tool-boundaries.md` before launching or writing a tool;
- `references/repair-approval.md` before any code change;
- `references/registry.json` for an entry's inputs, outputs, and stop statuses
  (`references/registry.md` explains its fields);
- `references/migration.md` only for records written under older entry names.

## Resolving an Entry Name

Every entry is a document under `entries/`, not a separate skill and not a
slash command. A name such as `parity-run` maps to `entries/parity-run.md` with
no exception. Entry documents refer to shared material as
`../references/<file>`.

## Route by Goal

| Goal | Entry |
| --- | --- |
| Collect a reproducible case or new cells | `parity-run` |
| Gate a cell's clean E2E metrics | `e2e-metrics-gap` |
| Check that both sides mean the same settings | `semantic-alignment-check` |
| Check workflow with measured layer timings, or explain scheduling, batching, or lane differences | `workflow-gap-analysis` |
| Compare matched operators or kernels | `op-gap-analysis` |
| Fill profiling coverage | `op-supplement` |
| Classify PP, collective, KV, or M2N time | `residual-lane-classification` |
| Attribute host time | `cpu-overhead-analysis` |
| Find an error source without changing code | `error-analysis` |
| Apply decided corrections and republish | `error-correction` |

The call graph reaches the other ten entries: `groundtruth-run-check`,
`groundtruth-run`, `simulator-run`, `dispatch-align-trace`,
`semantic-alignment-repair`, `workflow-repair`, `op-repair`, `residual-repair`,
`cpu-overhead-supplement`, and `cpu-overhead-repair`.

## Rules for Every Entry

- Only clean-run rows of the formal requests decide the gate. Each metric
  passes on its own when its relative error is at most `0.10`, and
  `run_gate.py gate` gives the verdict after the engine's metric conversion and
  gate compatibility are verified. Instrumented and diagnostic runs never
  close it.
- An unresolved required case choice stops the affected work. Gather evidence,
  prepare options with costs, ask one focused question, and record the user's verbatim
  answer under a decision ID in the task's `requirements.md`. No artifact,
  status, or earlier receipt approves a code change by itself.
- Fix `simulation_logic` errors first. Re-collecting or tuning measurement
  tables runs only when the user approves it for the task, and a Frontier
  change never models a ground-truth defect.
- One correction is one ticket and one scoped commit, followed by the
  regression set and fresh evidence (`references/repair-approval.md`).
- Published tables, readings, and run directories are never edited; a new
  generation replaces them and the old one stays on record.
- GPU jobs follow the operator's platform rules. Pushing Frontier or
  reference-engine branches and any other external action needs the user's approval.
- Write "code change" and "repair change" in records, in plain words.

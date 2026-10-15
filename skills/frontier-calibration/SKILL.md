---
name: frontier-calibration
description: Use when Frontier simulator results must be compared with vLLM-BS ground truth, to collect a reproducible case, gate E2E metrics against the 10% threshold, find the source of an error, or apply and republish a decided Frontier correction.
---

# Frontier Calibration

Calibration runs Frontier and vLLM-BS, an instrumented fork of vLLM v0.10.2, on
the same request trace, gates the clean E2E metrics, and traces each gap to a
source. The repository harness `tests/comparison/e2e_gate/` decides the gate;
this skill routes the work around it and constrains what may change.

If `site.local.md` exists beside this file, read it first. It binds the
placeholders used here (`<vLLM-BS checkout>`, the artifacts root given as
`--artifacts-root`, `$FRONTIER_TMP_ROOT`, the GPU platform) to this host and is
not tracked. Without it, ask the user for those values.

## Read Before a Case

1. `references/contract.md`: case layout and binding, ground-truth checkout,
   run modes, the E2E gate, error-source classes, decisions.
2. `references/call-graph.md`: execution order, predicates, stops, and return
   edges. It overrides any order implied elsewhere.
3. `references/pitfalls.md`: failures that earlier calibration tasks hit.
4. `references/evidence-tiers.md`: which artifact may support which claim.

Read on demand:

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
| Explain scheduling, batching, or lane differences | `workflow-gap-analysis` |
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
  `run_gate.py gate` gives the verdict. Instrumented and diagnostic runs never
  close it.
- A field the user has not decided stops the work. Gather evidence, prepare
  options with costs, ask one focused question, and record the user's verbatim
  answer under a decision ID in the task's `requirements.md`. No artifact,
  status, or earlier receipt approves a code change by itself.
- Fix `simulation_logic` errors first. Re-collecting or tuning measurement
  tables runs only when the user approves it for the task, and a Frontier
  change never models a ground-truth defect.
- One correction is one ticket and one scoped commit, followed by the
  regression set and fresh evidence (`references/repair-approval.md`).
- Published tables, readings, and run directories are never edited; a new
  generation replaces them and the old one stays on record.
- GPU jobs follow the operator's platform rules. Pushing Frontier or vLLM-BS
  branches and any other external action needs the user's approval.
- Write "code change" and "repair change" in records, in plain words.

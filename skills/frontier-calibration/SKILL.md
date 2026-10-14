---
name: frontier-calibration
description: Use when a Frontier simulator result must be compared with instrumented vLLM and the active v2 calibration routes are required.
---

# Frontier Calibration

Read `references/registry.json` first for the fixed entry catalog and candidate
dependencies. Read `references/contract.md`, `references/evidence-tiers.md`, and
`references/call-graph.md` before handling a case; the call graph is authoritative
for conditional execution order and return edges. Read
`references/tool-boundaries.md` only when a deterministic helper is needed, and
read `references/repair-approval.md` before any approved code change.

## Resolving An Entry Name

Every entry in this skill is a document under `entries/`, not a separate skill
and not a slash command. Whenever this file, `references/call-graph.md`,
`references/registry.json`, or another entry names an entry such as
`parity-run`, read `entries/parity-run.md` — the name maps to
`entries/<name>.md` with no exception. Entry documents refer back to shared
material as `../references/<file>`, which resolves from inside `entries/`.

## Route By Goal

| Goal | Entry | Document |
| --- | --- | --- |
| Start one reproducible pair | `parity-run` | `entries/parity-run.md` |
| Compare clean latency/throughput | `e2e-metrics-gap` | `entries/e2e-metrics-gap.md` |
| Check parameter meaning | `semantic-alignment-check` | `entries/semantic-alignment-check.md` |
| Explain action/event/behavior differences | `workflow-gap-analysis` | `entries/workflow-gap-analysis.md` |
| Compare aligned operators | `op-gap-analysis` | `entries/op-gap-analysis.md` |
| Fill operator coverage | `op-supplement` | `entries/op-supplement.md` |
| Classify remaining PP/collective/KV-M2N time | `residual-lane-classification` | `entries/residual-lane-classification.md` |
| Attribute late host CPU time | `cpu-overhead-analysis` | `entries/cpu-overhead-analysis.md` |
| Locate the source without code changes | `error-analysis` | `entries/error-analysis.md` |
| Apply reviewed corrections | `error-correction` | `entries/error-correction.md` |

The table lists the ten goal entry points. `entries/` also holds the remaining
ten entries that routes reach through the call graph: `groundtruth-run-check`,
`groundtruth-run`, `simulator-run`, `dispatch-align-trace`,
`semantic-alignment-repair`, `workflow-repair`, `op-repair`, `residual-repair`,
`cpu-overhead-supplement`, and `cpu-overhead-repair`.

Use the exact registry name. Legacy names stay explicit-only routing aliases in
the archived v1 tree and are not copied into this active directory.

## Global Boundary

Keep clean E2E metrics separate from diagnostic traces. Require at least three
ground-truth warmups. Treat each applicable E2E metric independently and require
relative error at most `0.10`. Require complete analysis plus external human
review `PASS` before a code change, except for a case initialized with
`auto_recycle: true` and a complete automatic receipt. Use plain “code change”
and “repair change” language in case records. Invoke `grill-me` skill for any missing
semantic setting or material design choice and record its decision in the case
manifest before execution.
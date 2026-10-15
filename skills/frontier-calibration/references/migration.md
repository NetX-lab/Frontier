# v2 Entry Map and History

## Entry Map

| Entry | Role |
| --- | --- |
| `groundtruth-run-check` | ground-truth preflight and post-run check atomic |
| `groundtruth-run` | ground-truth execution atomic |
| `simulator-run` | simulator execution atomic |
| `semantic-alignment-check` | settings, mapping, and routing analysis atomic |
| `semantic-alignment-repair` | decided setting or mapping repair atomic |
| `e2e-metrics-gap` | request-level gate atomic |
| `dispatch-align-trace` | dispatch and route-delay trace atomic |
| `workflow-gap-analysis` | workflow analysis atomic |
| `workflow-repair` | decided workflow repair atomic |
| `op-gap-analysis` | operator analysis atomic; includes trace normalization |
| `op-supplement` | profiling table supplement atomic |
| `op-repair` | decided operator repair atomic |
| `residual-lane-classification` | read-only residual classifier atomic |
| `residual-repair` | residual repair atomic selected by lane |
| `cpu-overhead-analysis` | host-time analysis atomic |
| `cpu-overhead-supplement` | CPU-overhead table supplement atomic |
| `cpu-overhead-repair` | decided host-side pricing repair atomic |
| `parity-run` | case setup and collection composite |
| `error-analysis` | read-only analysis composite; includes the error budget |
| `error-correction` | decided repair and republication composite |

## Compatibility Aliases

These v1 names are explicit-only aliases. New work uses the v2 names:

| Alias | v2 route |
| --- | --- |
| `frontier-run` | `simulator-run` |
| `vllm-run` | `groundtruth-run` |
| `semantic-config-audit` | `semantic-alignment-check` |
| `e2e-metrics-analysis` | `e2e-metrics-gap` |
| `residual-gap-analysis` | `residual-lane-classification` |
| `pp-overhead-repair` | `residual-repair` with `pp` lane |
| `collective-overhead-repair` | `residual-repair` with `collective` lane |
| `transfer-overhead-repair` | `residual-repair` with `kv_or_m2n_transfer` lane |

The historical `op-trace-normalize` entry is an internal phase of
`op-gap-analysis`; its normalized stream stays an intermediate evidence file.

## History

| Date | Change |
| --- | --- |
| 2026-08-31 | v2 replaced the v1 skill family as the user-level default. |
| 2026-09-24 | The archived v1 helper scripts were retired; entries that bound them had no runnable helper. |
| 2026-09-26 | Profiling merge and operator gap moved to new repository tools under `tests/comparison/calibration/`. |
| 2026-10-14 | PR 44 merged the gate harness (`tests/comparison/e2e_gate/`), the ground-truth harnesses, the CPU-probe producer, and the kernel-gap profiler. |
| 2026-10-15 | The skill moved into the repository at `skills/frontier-calibration/` and was revised against the practice of the E2E accuracy gate: repository tools, run modes, gate kinds, error budget, error-source classes, and decision records in place of automatic receipts. |

## Default Path Switch and Rollback

Agents discover the skill through a link from their skill directory to
`skills/frontier-calibration/` in a Frontier checkout (`skills/README.md`). A
default path switch changes that link's target. A rollback restores an earlier
revision with `git checkout <commit> -- skills/frontier-calibration` or points
the link at a checkout of that commit. Both need the user's approval and a
rerun of `skills/frontier-calibration/tests/pressure_scenarios.py`.

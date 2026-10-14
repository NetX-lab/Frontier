# v2 Enablement and Restore Map

v2 is now the active Frontier calibration skill family. The current default
skills directory is `/data/ycfeng/codex-home/skills/frontier-calibration/`.
The previous skill family and its recovery copy are stored outside codex-home
under `/data/ycfeng/frontier-calibration-old-20260831/`.

## Current Default Entry Map

The directory `/data/ycfeng/codex-home/skills/frontier-calibration/` now
contains the active v2 entries:

| Entry | Role |
| --- | --- |
| `groundtruth-run-check` | ground-truth preflight atomic |
| `groundtruth-run` | ground-truth execution atomic |
| `simulator-run` | simulator execution atomic |
| `semantic-alignment-check` | parameter-semantics analysis atomic |
| `semantic-alignment-repair` | reviewed parameter repair atomic |
| `e2e-metrics-gap` | request-level metric comparison atomic |
| `dispatch-align-trace` | online dispatch trace atomic |
| `workflow-gap-analysis` | workflow analysis atomic |
| `workflow-repair` | reviewed workflow repair atomic |
| `op-gap-analysis` | operator analysis atomic; includes trace normalization |
| `op-supplement` | operator data supplement atomic |
| `op-repair` | reviewed operator repair atomic |
| `residual-lane-classification` | read-only residual classifier atomic |
| `residual-repair` | residual repair atomic selected by lane |
| `cpu-overhead-analysis` | conditional CPU analysis atomic |
| `cpu-overhead-supplement` | CPU data supplement atomic |
| `cpu-overhead-repair` | reviewed CPU repair atomic |
| `parity-run` | run orchestration composite |
| `error-analysis` | read-only analysis composite |
| `error-correction` | reviewed repair orchestration composite |

## Compatibility Paths

These compatibility aliases are archived with the old version and remain
explicit-only. New work uses the v2 entry names above:

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

The historical `op-trace-normalize` entry is folded into `op-gap-analysis` as
an internal phase. Its normalized stream remains an intermediate evidence file,
not a separately discoverable v2 entry.

The old active directory is archived at
`/data/ycfeng/frontier-calibration-old-20260831/current-default/frontier-calibration/`.
The seven old top-level entries are under
`/data/ycfeng/frontier-calibration-old-20260831/legacy-entries/`, and the dated
recovery copy is under
`/data/ycfeng/frontier-calibration-old-20260831/dated-backup/frontier-calibration/`
with its existing `manifest.tsv`. The seven historical names remain recoverable:
`frontier-vllm-v1-error-analysis`,
`colocation-ttft-per-op-schedule-vs-vllm-v1`,
`online-mode-analysis-comparison`, `vllm-v1-cpu-overhead-measurement`,
`frontier-profiling`, `parallel-config-alignment-verifier`, and
`dispatch-aligned-trace`.

The dated `manifest.tsv` is an immutable migration-time snapshot. Its recorded
backup paths describe the location before the dated backup directory moved and
are not active discovery paths. The current locations and the manifest
SHA-256 are recorded in
`/data/ycfeng/stepfun-performance-optimization/Frontier/task_memory/task_2026-08-29-frontier-calibration-skills/old-version-archive-path-map-2026-08-31.md`.
Use that map when checking the moved snapshot; keep the manifest bytes
unchanged.

## Enablement Record

The v2 enablement performed on 2026-08-31 used the following ordered actions:

1. Validate the v2 registry, all v2 entrypoints, pressure rules, and destination
   paths before changing discovery.
2. Move the old default directory, seven old top-level entries, and dated
   recovery copy to `/data/ycfeng/frontier-calibration-old-20260831/`.
3. Move the v2 directory to
   `/data/ycfeng/codex-home/skills/frontier-calibration/`.
4. Re-run the validator and pressure checks through the current default path and
   verify that no old Frontier entry remains directly under codex-home/skills.

The operation changed skill discovery only. It did not modify Frontier or vLLM
source and did not delete any old directory.

## Default-Path Switch and Rollback

The phrase `default path switch` in the validation contract means one concrete
action: changing the default discovery path from the old skill family to the
v2 directory. That switch was completed on 2026-08-31 and is recorded above.
The word `rollback` means restoring the archived old directory as the default
and moving the current v2 directory to a separate recovery path. Both
operations are path moves, require a new explicit approval, and must be
followed by the same validator and pressure checks.

## Restore the Old Version

Restoring the old version would move the current v2 directory to a dated
recovery path, move the archived `current-default/frontier-calibration` back to
`/data/ycfeng/codex-home/skills/frontier-calibration/`, and then re-run the old
validator and alias checks. That action changes the default discovery path and
requires a separate explicit approval. No restore action is currently planned.

# D29-WAVE-COLLECTIVE-SEAM requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive historical scope; later YC run-fixed approval supersedes hot-update constraints without changing the recorded probe scope. |
| 2026-10-23 | Record the observed root source move to 5e149dd8 (CORE prepared points) and the re-based measurement. |
| 2026-10-23 | Record the delegated read-only collective-seam analysis scope and constraints. |

## [Original Request]

The following is the historical delegated scope. The task-root requirements
record YC's later2026-10-24 run-fixed input approval and the new implementation
allocation. Historical budgets below are not current available allowances.

Delegated subagent task D29-WAVE-COLLECTIVE-SEAM (verbatim intent): read-only
analysis plus a bounded scratch probe that identifies the correct existing module
seam for avoiding repeated common collective preparation across Step5's 16 EP
lanes and 88-layer waves, without dropping live current-input admission.

- Frozen source `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06`
  at `79f1a3fd`; root `step5-preview` at `905d31a9` is source-equivalent.
- Real wave loop `/data/ycfeng/tmp/d29_main08_residual_repro/wave_loop.py`:
  layer61, position6147, 16 lanes, 2.65s warm, 80 same-8MB artifact reads,
  28003 JSON decodes, payload hash `b2593e72…3ffc3a`.
- CORE owns row/plan preparation separately; do not duplicate it.
- Assigned paths: `scheduler/utils/expert_parallel.py:90–157`,
  `scheduler/utils/ep_wave.py:109–151`,
  `execution_time_predictor/moe_operator_times.py:80–149` plus related
  predictor/lifetime interfaces, `cc_backend/backends/collective_sim_cc_backend.py:803–881`
  and tests.
- The existing backend same-size/restored-mtime rewrite test must keep passing;
  stat/path/global caching is invalid. All source/point/family/context/placement/
  qualification/count/layout checks stay live; no blanket numerical memo bypass.
- Do not assume authority to change input-update semantics; surface the trade-off.
- Source/tests/retained inputs read-only; no full CLI, large CSV scan, GPU,
  platform or network. Heavy probes: `timeout 60s` + `RLIMIT_AS 30GiB`.
- Writes only `/data/ycfeng/tmp/d29_wave_collective_seam` and this directory.
  No commits, no further agents. Remaining budgets CORE434, PRICING36,
  global746; no allocation granted.

## Observed scope change (not a new request)

During the task the root `step5-preview` advanced from 905d31a9 to 5e149dd8, where CORE's
prepared row/plan selection landed, and then to docs-only 48d34106. The final measurements
use source that is equivalent to 5e149dd8 under a checked diff. The frozen tree 79f1a3fd
remains the earlier baseline. Constraints and budgets are unchanged.

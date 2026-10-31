# B30b scoped completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Archived the verified B30b implementation, current fresh plans and scoped-commit handoff. |

## Overview

Completed the authorized CPU verification of the existing F1b/F2/F3 probe package in `/data/ycfeng/Frontier/.worktrees/b30b-ex-situ-link-probes`, starting at `b0e568f55ed5ade0c234949697bf22a84cd2f184`. No source repair or implementation expansion was needed. The scoped commit includes the original profiling changes, tests, shared G5h runtime-check extension, existing B30b tools/stages and durable local evidence.

Measured CC remains the formal input under decision 46. B30b supplies diagnostic measurements and CPU fit/analysis tools. Main requirements/spec/plan/progress/tickets, READMEs, old plans and measurements remain unchanged. No GPU or cloud operation was performed.

## Deliverables

The exact 53-path changed-file list, implementation line counts, stage resource bounds and launch entry points are in `commit_scope.json`. Profiling and test paths are:

- `frontier/profiling/collectives/cuda_launch_depth.py`
- `frontier/profiling/collectives/rank_placement.py`
- `frontier/profiling/collectives/nixl_kv_read_benchmark.py`
- `frontier/profiling/collectives/vllm_collective_benchmark.py`
- `frontier/profiling/collectives/vllm_dsv4_forward_collectives.py`
- `tests/analysis/v4_dummy_kernel_count.py`
- `tests/unit/test_b30b_ex_situ_probes.py`
- `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/runtime_check.py`

Within this B30b directory, the existing tools are `planner.py`, `phase_runner.py`, `fit_tools.py` and `launch_analysis.py`; stage inputs are `stage_b30b_links_kv.json` and `stage_b30b_launch_depth.json`. Native evidence remains `kv_layout_evidence.json` and `native_dummy_kernels.json`.

Current full plan and budget paths are `fresh_dry_run_20261031_01/b30b_links_kv.{json,txt}` and `fresh_dry_run_20261031_01/b30b_launch_depth.{json,txt}`. Exact commands/environment, expected/actual results, source-review findings and limitations are in `test_report_2026-10-31_b30b_ex_situ_probes.md`. Durable CPU evidence is in `validation_20261031/`; fresh historical-preservation evidence is in `preservation_20261031/`. The original snapshot is in `evidence_20261031_initial/`.

## Observed validation

Required suite: **163 passed in 6.02 seconds**, 0 failures/errors/skips. Independent generated-plan reconciliation: **74 PASS checks**. Historical preservation: **8 successful listings byte-identical**, **5 historical failures unchanged**, **28 recorded sources match**. Final retention: **114/114 initial paths unchanged**, **0 writable descriptors**, **0 source repairs**. The frontier patch remains **5 files / +235/-8 = 243 changed lines**; all 12 implementation/test paths including existing B30b-local tools contain **1,367 changed lines**.

Fresh two-node job: A=132/node ×2, B=142 per placement ×2, C=80; **628 case instances**, **1,884 planned pass rows**, **1976.8120210185284 s**, **8.785831204526794 GPUh** on 16 GPUs. F2: **84 trials**, **807.6709306640623 s**, **0.22435303629557288 GPUh** on one GPU. Combined estimate: **9.010184240822367 GPUh**; configured timeout exposure: **21 GPUh**. Old README/dry-run values B=496 / 29.648560 GPUh are preserved historical values.

## Remaining owner work and limits

No blocker remains for this sidecar's CPU/scoped-commit acceptance. The lead must integrate into the stage's main workspace, where K5/runtime helpers exist, before mounted-source preparation and hardware execution. Physical/NIC placement, H200 communication fits, CUDA launch capacity and diagnostic A10 errors remain unmeasured here. The retained native count is 2,240 kernels in 93 FULL dummy forwards; no new trace census was run. The existing ASTRA full-pod GPU-coordinate-first classification limit is recorded in the test report and remains unchanged. B29, F1a, F4, ingress and main records remain separately owned.

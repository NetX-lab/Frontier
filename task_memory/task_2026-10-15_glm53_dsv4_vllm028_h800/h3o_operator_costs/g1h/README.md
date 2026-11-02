G1h is ready for the lead's package preparation after review and commit. The amended grid contains 1,041 windows, all reached by CPU prediction, using 989 cases and 989 measured steps. Each case has zero warm-up steps and one measured step. Native engine warm-up, compilation and graph capture still run. The coverage check reports zero missing windows, zero predicted steps outside the grid and zero prediction mismatches.

| Category | Requested and predicted windows | Cases | Cases shared by several windows | Additional shared windows | Measured steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| FULL serving | 351 | 336 | 15 | 15 | 336 |
| PIECEWISE without tails | 351 | 336 | 15 | 15 | 336 |
| First tail | 318 | 296 | 17 | 22 | 296 |
| Multi-tail | 7 | 7 | 0 | 0 | 7 |
| Dummies | 13 | 13 | 0 | 0 | 13 |
| Prefill serving | 1 | 1 | 0 | 0 | 1 |
| Total | 1,041 | 989 | 47 | 52 | 989 |

Decision D-G1H-GRID-AMENDMENT (October 28, 2026) supplies both grid rules. A first tail has 6,143 cached tokens and a query of one token, so it always writes at ratios 4 and 128. With t tails, each requested mix (n4, n128) becomes (max(n4, t), max(n128, t)). Duplicate windows are removed in grid order, preserving the first. This removes 33 windows from B11's 1,074 targets, all in the first-tail category. The seven multi-tail windows already request (tails, tails). A coordinate no native step can produce adds no coverage.

The same decision defines “near” as within max(ratios) - 1, or 127 cached tokens, of a profile target. The shared C1 table in `frontier/profiling/attention/v4_attention_windows.py` defines this once. Continuing requests stay inside 6,144–7,166: low permits 6,144–6,271, high permits 7,039–7,166, and spread applies the same allowance to evenly spaced targets. Cached 6,143 is reserved for first tails. This is the smallest allowance that reaches a ratio-128 writer at both endpoint profiles. The builder and checker have no separate allowance option.

Placement uses the native FCFS prediction to invert construction. For a first-tail case, it first predicts the continuing requests' first pure decode step, then accounts for their advance while the tails are chunked. A guess that overshoots at the high profile can be corrected before requests stop scheduling during the tail construction. The complete first-tail prediction must still reach the selected contexts, and admission must assign its measured step to the requested window.

That correction resolves the W24/own18/high windows with mixes (1, 1), (7, 1) and (4, 4), preserving their original contexts. B11's update could change the construction duration by one step and bring early requests to 7,167 before measurement. The W8 and W16/own6/low/(5, 1) windows instead cycled between six and seven steps because of the 6,143-token chunk cap. When inversion cycles, placement tries the nearest single-request adjustment that preserves the write phase and profile. Moving the non-writer from 6,144 to 6,146 resolves both windows; the four continuing ratio-4 writers stay at 6,147, and the tail stays at 6,143. No grid window or engine limit changes.

Identical submissions share a case. A one-request spread profile degenerates to low, and profiles with no continuing requests use the same physical coordinate. FULL W1 dummy is measured once for both planner roles, leaving 13 physical dummy windows for 14 requested role uses. Every case measures all three attention layer variants and both planner roles. Only the measured NONE/W6144 prefill crosses the 1,024-row multi-stream threshold; construction chunks are unmeasured.

Multi-tail cases use engine 2 with a 24,576-token budget (4 × 6,144) and a 6,143-token chunk cap. Engines 0 and 1 retain their 8,192-token budgets. Continuing requests first reach decode, then all tail prompts are submitted together. The larger budget fits every 6,143-token tail chunk in the same preceding step, alongside continuing decodes. The next step has cached 6,143/query 1 for every tail. The seven observed coordinates are (2, 2), (3, 2), (3, 3), (4, 4), (5, 2), (7, 2) and (8, 3).

Native evidence is read from `/data/ycfeng/tmp/opencode/vllm-v0.28.0`, revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`. In `vllm/v1/core/sched/scheduler.py`, lines 525 and 751 schedule running requests before waiting requests; lines 563–577 and 952–966 apply the chunk threshold, budget and context limit; lines 1380–1397 advance cached tokens after scheduling. `vllm/v1/worker/gpu/cudagraph_utils.py:170–282` and `dp_utils.py:16–95` determine local and synchronized graph dispatch. Query-one attention branches and actual decode planning are in `vllm/v1/attention/backends/mla/sparse_swa.py:531,712–788` and `vllm/models/deepseek_v4/nvidia/flashmla.py:148–169`. Compressor and scratch work slice actual rows (`compressor.py:348` onward; `eager_scratch.py:98–109,130–136`). These rules preserve the measured attention shapes and branches.

CPU evidence does not establish GPU kernel identity for the larger-budget engine. The zeroing at `sparse_swa.py:551–552` can process a larger allocation; its exact generated CUDA symbol is not qualified here. Construction prefill workspace/chunking (`flashmla.py:317–323`), allocation and autotuning can also change. The lead retains later GPU qualification and timing.

| Engine | Budget | Cases and measured steps | Estimated engine steps | Estimated GPU seconds | GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 8,192 | 686 | 6,098 | 1,873.394 | 0.520387 |
| 1 | 8,192 | 296 | 3,016 | 1,163.584 | 0.323218 |
| 2 | 24,576 | 7 | 20 | 491.604 | 0.136557 |
| Total | | 989 | 9,134 | 3,528.581 | 0.980161 |

The timing model charges 486.104 seconds of cold process/start/recapture time to each engine, derived from P1h package 03 engine 0's 576.104 seconds total minus about 90 seconds of cases. Engine 0 spent about 431.78 seconds compiling/autotuning; engine 1 took 90.291 seconds total, with 25.23 seconds compiling/autotuning. Removing 163 × 0.2-second measured holds from the 90-second case interval and dividing by 280 estimated engine steps gives 0.205 seconds per unheld step, or 0.405 seconds per measured step including the hold. The admitted full-grid run is estimated at 0.980161 GPU-hours. Charging a separate case to every window gives an envelope of 1.078888 GPU-hours. Both are below C1's 1.5-hour G1h share, the three-hour planning limit and the seven-hour G1h–G4h cap, so a grid reduction is unnecessary. Per-case profiler overhead and larger-budget autotuning remain unqualified. One measured step per window provides no repeated-window variance.

Reproduce from the working root with Python 3.10.6. Use a fresh scratch directory under the executor's assigned scratch root; all outputs below stay there except the four regenerated G1h planning files.

```bash
ulimit -v 31457280
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
G1H_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python
G1H_H3O=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs
G1H_DIR="$G1H_H3O/g1h"
G1H_SCRATCH=/data/ycfeng/tmp/codex-implement/ebbfa1e0-dd55-490b-b1d1-4d453d7fc0a4/b11b
export TMPDIR="$G1H_SCRATCH"

timeout 120s "$G1H_PYTHON" "$G1H_DIR/build_cases.py"
timeout 120s "$G1H_PYTHON" tests/analysis/v4_attention_coverage.py cases \
  --package-plan "$G1H_DIR/package_plan.json" --output "$G1H_SCRATCH/g1h_predictions.jsonl"
timeout 120s "$G1H_PYTHON" tests/analysis/v4_attention_coverage.py check \
  --package-plan "$G1H_DIR/package_plan.json" --raw "$G1H_SCRATCH/g1h_predictions.jsonl" \
  --predicted --require-complete-grid \
  --c1-json "$G1H_H3O/evidence/coverage/p2_32.json" "$G1H_H3O/evidence/coverage/p2_64.json" \
            "$G1H_H3O/evidence/coverage/p0.json" --output "$G1H_SCRATCH/g1h_coverage.json"
timeout 120s "$G1H_PYTHON" tests/analysis/v4_attention_coverage.py check \
  --package-plan "$G1H_H3O/p1h_bridge/package_plan.json" \
  --raw /data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_03/download/P1H_BRIDGE/bridge/FULL_AND_PIECEWISE/raw \
  --output "$G1H_SCRATCH/p1h_coverage.json"
timeout 180s "$G1H_PYTHON" -m pytest tests/unit/test_v4_attention_coverage.py \
  -q -rs -p no:cacheprovider --basetemp="$G1H_SCRATCH/pytest-readme-rerun"
```

Generation, prediction and complete-grid checks exit 0. All 145 case-specific native serving classes are inside: P2-32 43/43, P2-64 54/54 and P0 48/48. All seven observed multi-tail coordinates are inside. The streaming package 03 check matches all 175 measured steps of its 25 cases; it covers 21 amended windows and reports 72 steps outside the grid, with no prediction mismatches. The B11 acceptance test group plus the updated coverage tests passes 617 checks, with one existing optional pinned-native Git-object check skipped because `FRONTIER_VLLM_BS_ROOT` is unset.

`package_plan.json` keeps destination `package_01`, name `G1H_ATTENTION`, output `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/g1h_01`, and B11's source revision `5083d0369a5f9b86c333b30fce78154a3364225f`. The manifest records the fixed allowance and D-G1H-GRID-AMENDMENT; provisional tolerance status and fixed execution blockers are removed. `prepare_package.py` reuses P1h's unchanged package builder and gates on complete coverage. After committing B11 and B11b, the lead replaces source_revision and builds the package. No package was built or submitted by this executor.

Package 01 (source `32d3d08b`) ran as `exp-1029-010007-750668` and failed before its first measured step: the worker image has no `fasteners`, and the replay-gap profiler loaded it through the old `frontier.execution_time_predictor` package `__init__`. Commit `a689ed05` makes that package import nothing. `package_plan.json` now names destination `package_02`, job tag `v4-h200-g1h-attention-20261028-02`, output `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/g1h_02` and source revision `a689ed05f462eb269b9dffe640da151e6614a5ef`; the grid and the cases are unchanged. Package 01 stays in place as the record of the failed run.

# F1b H200 link measurements and diagnostic replay inputs

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Recorded both terminal COVERAGE_MISS outcomes and reproduced the exact predictor rejections from unchanged measured inputs. |
| 2026-11-01 | Verified completed collectives03, fitted the unchanged link model, retained failed G5h validation, and started both diagnostic replays. |

Tickets 06/11/19; requirements decision 46. The complete small-topology measurements are available, but the fitted link model fails target-topology validation. The measured CC backend remains the formal input. No operation multiplier, parameter search, G5h fit input or additional GPU run was introduced.

## Run and retained evidence

RJob `exp-1101-005741-958298` completed at **2026-11-01 01:12:00 Asia/Hong_Kong** (2026-10-31 17:12:00 UTC). Creator `i-fengyicheng`; two nodes, eight H200 each; `step_main`, non-preemptible. Both replicas are Succeeded, containers exit 0/restart 0, and the local launcher exits 0. Each node passed 22/22 runtime checks with source `ac0f908a`. The company image and NFS source `100.96.128.195:/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` match the launch receipt. Physical GPU placement was checked against each node's inventory: A uses all eight GPUs, B_same_rail uses indices 0/0, B_ring_boundary uses 7/0. Nodes are `gpu-h200-0375.lgcm.sh.istep.fun` and `gpu-h200-0912.lgcm.sh.istep.fun`.

Observed container durations are 659 and 833 seconds, totaling **3.315556 container GPU-hours**. This excludes scheduling/image-pull overhead and is not a billing statement. The original estimate was 7.455951 GPU-hours, with a 3,900-second job timeout.

Cloud root: `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/b30b_links_03`. Completed profiling traces stay there. The selected local download at `/data/ycfeng/tmp/glm53_dsv4/b30b_links03_20261101` contains 62 files / 34,807,505 bytes, all SHA-256 values matching the cloud read. Durable CSVs, phase plans, download manifest, verification and fit evidence are under `h3o_operator_costs/b30b/f1b_links03_20261101/`; platform and launcher receipts are in `execution_links03_20261101/`.

## Verification commands and results

CWD is this worktree. CPU Python `/data/ycfeng/envs/frontier-py310/bin/python` is 3.10.6; worker Python is 3.12, torch 2.13.0+cu130 and vLLM 0.28.0. CPU commands use `PYTHONPATH=$PWD`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `TMPDIR=/data/ycfeng/tmp`.

1. `python3 /data/ycfeng/tmp/glm53_dsv4/download_links03_20261101.py`: PASS. Reads only the personal cloud directory through `shai-ycfeng`, requires both node result files and all phase returncodes 0, downloads selected files and verifies all hashes. Exact command and paths: `download.json`; file list and hashes: `download_manifest.json`.
2. `timeout 60s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/glm53_dsv4/verify_links03_20261101.py`: PASS for data integrity and reconstruction. The existing planner produces exactly 132 cases per A node and 142 per B placement. All three passes are present: **1,644 raw/CSV records, 548 kernel records and 104,760 critical-path timing trials**. All planned identities, per-rank repeat counts, payload bytes, graph flags, call counts and physical placements match. All timing values are finite and positive; no numerical-content correctness claim is made for the collective tensors.
3. `fit_tools.py astra --intra <raw>/A_intra/node0/summary.csv --intra <raw>/A_intra/node1/summary.csv --inter <raw>/B_same_rail/summary.csv --inter <raw>/B_ring_boundary/summary.csv --output <evidence>/fit`: execution PASS, qualified accuracy FAIL. Full absolute commands/source/timestamps are in `fit_command.json`. All six observations per case (two nodes/placements × three passes) are used through the existing median pooling rule. No G5h row enters the fit.
4. Existing `f_variants/f1_astra_per_call.py <evidence>/heldout <all fields from fit/fields.txt>` from `/data/ycfeng/Frontier/.worktrees/f1a-astra-diagnostic` (`9f5f8b61`, includes the diagnostic B30a correction): execution PASS, target-topology accuracy FAIL. `heldout_command.json` records the exact command. All 128 G5h cases are retained in `heldout/per_call.csv`, with predicted and measured microseconds and their ratio.

The first reconstruction assertion compared floating-point values as decimal strings and failed on standard deviation 21.810430022864175 versus 21.81043002286418. Rechecking numeric values at relative/absolute tolerance 1e-12 gives maximum absolute differences of 2.842171e-14; metadata remains exact. This auxiliary cross-interpreter rounding issue did not change any measurement or calibration threshold.

Not every repeat remained queued throughout the GPU sleep: A node0 has 595/39,600 affected critical trials and node1 748/39,600; B placements have 12/12,780 and 8/12,780. For diagnosis only, comparing each case median with its fully queued trials changes the median by at most 5.525241%, 2.708035%, 0.224970% and 0.336802%, respectively; the median change over rows is zero in each phase. The fit uses every original row. These checks quantify the imperfection; they do not establish perfectly aligned launch timing. Logs also retain shutdown warnings about process-group destruction and leaked shared memory; all phases and containers nevertheless terminated successfully.

## Fitted values and independent validation

| Link | Latency (us) | Bandwidth (Gbit/s) | Pooled cases | Median absolute relative residual | p90 absolute relative residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intra-server | 3.044684 | 2621.593395 | 132 | 19.758737% | 48.025842% |
| Inter-server | 115.378450 | 250.475326 | 142 | 20.760167% | 105.365096% |

The two inter placements separately fit latency 111.907684 / 117.861220 us and bandwidth 251.072877 / 249.187145 Gbit/s. These are coefficients of the prescribed collective model, not an independent direct measurement of physical switch latency. The existing fitter and F1b replay retain the backend's `FullyConnected` topology primitives; F1a's specification variant uses `Switch`. All assignments are explicit in the receipts. Native grouped dispatch remains one summed-byte prediction, with no per-operation correction.

The G5h graph-forward totals expose the failure:

| Scope | Measured communication / forward | Predicted communication / forward | Prediction / measurement |
| --- | ---: | ---: | ---: |
| EP16, graph_t1 | 9,005.1 us | 149,177.2 us | 16.5659 |
| EP32, graph_t1 | 16,082.7 us | 308,299.6 us | 19.1696 |
| TP8, graph_t1 | 728.4 us | 1,906.1 us | 2.6168 |

Across all graph sizes, EP16 ratios are 13.4226–17.1593, EP32 11.4490–19.3002, and TP8 2.3775–2.9059. `fit/fit.json` retains all absolute residuals; `heldout/per_call.csv` retains every target case. The current common latency/bandwidth model does not reproduce all collective paths and does not extrapolate accurately to the target pods. The exact source of that failure is not isolated by this comparison.

## Full diagnostic replays

Both cases started from the unchanged `f1a-astra-diagnostic` checkout at `9f5f8b61`, each with a private copy of its B29 predictor cache. Command: `python3 /data/ycfeng/tmp/glm53_dsv4/run_f1b_links03_20261101.py p2_32` and `p2_64`. Roots: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/runs/f1b_p2_{32,64}_links03_codex_20261101`. Exact launch commands and fitted-field hash: `h3o_operator_costs/b30b/f1b_replays_20261101/p2_{32,64}_command.json`.

The existing A10 driver keeps K19, the original fixed workload and native repeats, and the 10.3/11.4 ms ingress/handoff inputs. These runs report the F1b proposal's complete four-metric errors despite the poor held-out validation. Both runs subsequently failed coverage admission, so complete four-metric readings are unavailable. The watcher correctly skipped `a10_gate.py` for both failed launchers; outputs are fresh paths under `h3o_operator_costs/f1b_comparisons_20261101/`. Original F1a/F2 watchers and runs continue independently. No formal input is promoted.

Initialization checks: `python3 /data/ycfeng/tmp/glm53_dsv4/verify_f1b_inputs_20261101.py` PASS for both cases, initialization 71.686125 / 72.884928 seconds. Six compute-table consumer/hash mappings, trace, native input hash and KV admission equal B29. There are 28 expected serialized differences: the communication backend and its global/role fields plus output/cache paths. The initial auxiliary allowlist omitted the ten explicitly requested flat role overrides and rejected them; including those documented inputs resolved the checker failure without changing either run. Both roles report `astra_sim_analytical`, no dummy mode and no step-cost input. Exact differences are in `f1b_replays_20261101/p2_{32,64}_input_comparison.json`.

## Terminal replay outcomes and coverage diagnosis

Both original launchers exited 1: P2-32 at 2026-11-01 01:24:31+08:00, P2-64 at 01:24:20+08:00. The driver statuses are FAILED and both exceptions are `DeepSeekV4CoverageError`. The watcher is COMPLETE with two RUN_FAILED entries, not a metric PASS. Neither incomplete request population is gated.

| Case | Simulation wall time | Rejected measured coordinate | Four-metric status |
| --- | ---: | --- | --- |
| P2-32 | 357.929698 s | `dsv4_embedding`, TP8, descriptor rows 32; measured rows are 1, 2, 4, 8, 16, 24, 6144 | COVERAGE_MISS; no complete TTFT/TPOT/E2E/THPT |
| P2-64 | 336.971305 s | `dsv4_fused_q_kv_prepare`, uncompressed/PIECEWISE/decode, padded rows 24, query tokens 16; measured query range is [17,24] | COVERAGE_MISS; no complete TTFT/TPOT/E2E/THPT |

The P2-32 failed batch contains 25 one-token requests and a graph descriptor padded to 32. The P2-64 batch contains 16 one-token requests and a graph descriptor padded to 24. Existing `dp_dummy_forward._synchronize_dp_descriptors` applies a shared DP token width while preserving each lane's request count; a local query count need not equal that width. `deepseek_v4.predict_source_execution` prices embedding by the descriptor width. The observed pairs are therefore reachable through the existing path. This inspection is not a new claim of native batch equivalence or proof that no further coverage gaps exist.

Reproduction command, from this worktree:

```bash
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/f1a-astra-diagnostic PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b30b/f1b_replays_20261101/coverage_reproduction.py
```

PASS: `PASS_REPRODUCED_BOTH_COVERAGE_FAILURES`, using Python 3.10.6 and the original source checkout. All six compute hashes on each side still match the initialized receipt. The narrow reproduction uses measured embedding values and the production attention loader/trainer's admission bounds; it does not fit or predict replacement times. Both actual predictor error strings match their terminal receipts. The failed launches are retained as `p2_{32,64}_terminal_receipt.json`, `p2_{32,64}_terminal.log`, and `coverage_reproduction.json` in `h3o_operator_costs/b30b/f1b_replays_20261101/`.

Disposition: the current F1b variant has failed held-out accuracy and unavailable complete E2E metrics due to coverage. No clipping, extrapolation, synthetic rows, shorter workload, or reuse of another graph group is applied. Preserve this outcome beside F1a/F2/F4; any further measured supplement must identify the required joint coordinates and its value before another run. Existing F1a P2-64 and both F2 replays continue independently.

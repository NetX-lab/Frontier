# Test report: F1-F4 comparison variants of the A10 inputs (decision 46)

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Verified six completed readings/96 metric entries for the final comparison collector while both original F2 replays continue. |
| 2026-11-01 | Completed F1a P2-64 with all-repeat latency FAIL; retained exact references, six compute hashes and active F2 replays. |
| 2026-10-31 | Codex update: F4/B28 five-artifact identity on both scales, B29 per-repeat results, F1a P2-32 final FAIL and fresh P2-64 replay. |
| 2026-10-31 | Created: F1 per-call screen and astra defect, F4 planner terms; F4 replays, B29, B30a and B30b pending. |

## Latest comparison checkpoint — 2026-11-01

F1a is complete at both scales. P2-32 has pooled TPOT28.38101% and E2E27.82468% errors (FAIL). P2-64 completed128 formal requests and917,504 tokens: TTFT/TPOT/E2E FAIL in every reference repetition; total THPT PASS. Specification-link parameters substantially underpredict the observed latencies on these two cases. The original incomplete P2-64 attempt remains retained; these results come only from the fresh completed run.

B29 remains the formal measured-CC input: P2-32 numerical PASS in all three repetitions; P2-64 repeat1 TTFT12.254940665% FAIL (779.888727400ms native versus684.313826604ms Frontier). Pooled TTFT7.67691% does not override it. F4 five metric artifacts remain byte-identical to B28 at both scales; its zero ingress/handoff differs from B29. F1b's prescribed positive fit fails G5h held-out validation and both replays terminate COVERAGE_MISS; no complete four-metric F1b result exists. F3's completed NIXL data yields an invalid affine fit and no replacement/replay. Both F2K18 CPU replays are still running; watcher23683 owns their gates.

F1a P2-64 uses source `9f5f8b614f1e278b02ad39015f0230de562c82ba`, clean, in `f1a-astra-diagnostic`. Both roles use ASTRA Switch links3,600/400Gbit/s and1/1µs latency, with K19 and B29 ingress10.3ms/handoff11.4ms. No dummy or step-cost table is bound. The six compute-table hashes match B29; all three normalized native reference CSVs are byte-identical to B29's retained CSVs. There are128 unique complete formal request rows, no token/request/metric gaps,53,248 forward and stage-end records, and zero stale stage ends. `collectives.calls=0` is the measured-backend census field on an ASTRA run; it is not evidence that no communication was simulated. Memory admission remains DIFFERS (2,940 modeled versus2,944 native blocks per request), as in the existing qualification record.

Execution: initialization72.455524s, simulation12,215.365581s; launcher session50972 exit0 and workload log ends `EXIT 0`. Existing startup warnings are retained in the selected log tail. The watcher executed the existing gate once at2026-10-31T18:22:43.811894Z; exit0 means the gate ran successfully, its accuracy verdict is FAIL. Evidence: `h3o_operator_costs/f_comparisons_20261101/f1a_p2_64/`, `f1a_p2_64_receipt.json` and `f1a_p2_64_verification.json`.

Commands:

```bash
bash /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/run_f1a.sh p2_64 f1a_p2_64_codex_20261031a
/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_gate.py p2_64 /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/runs/f1a_p2_64_codex_20261031a /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_comparisons_20261101/f1a_p2_64
/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_comparisons_20261101/verify_f1a_p2_64.py
```

The wrapper and exact environment are retained as `f_comparisons_20261101/run_f1a.sh`: Python3.10.6, the isolated checkout's PYTHONPATH, one BLAS/OpenMP thread, `/data/ycfeng/tmp` scratch and60GiB RLIMIT_AS. Verification initially looked for old normalized CSVs beside the archived reading; `commands.json` points to the original scratch output instead. Correcting that lookup produced `PASS_EVIDENCE_FAIL_ACCURACY`:128 rows, three unchanged references and six unchanged compute hashes. No normalization or gate was rerun.

Pooled reference values are supplementary; each repeat below still decides its own gate. Relative errors use absolute reference denominators, all positive; zero would be undefined.

| Metric | Native pooled | B29 measured-CC prediction | F1a prediction | F1a absolute error | F1a relative error | F1a verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TTFT (ms) | 741.216351 | 684.313827 | 463.897052 | 277.319300 | 37.414083% | FAIL |
| TPOT (ms) | 25.870531 | 25.295337 | 15.565031 | 10.305500 | 39.834900% | FAIL |
| E2E (ms) | 27206.769532 | 26561.443186 | 16386.923646 | 10819.845886 | 39.768947% | FAIL |
| THPT (tok/s) | 5963.206318 | 5975.251570 | 6404.273989 | 441.067671 | 7.396485% | PASS |

| Reference repeat | Metric | Native | F1a | Absolute error | Relative error | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| repeat01 | TTFT (ms) | 779.888727 | 463.897052 | 315.991676 | 40.517533% | FAIL |
| repeat01 | TPOT (ms) | 25.429869 | 15.565031 | 9.864838 | 38.792327% | FAIL |
| repeat01 | E2E (ms) | 26794.644444 | 16386.923646 | 10407.720798 | 38.842541% | FAIL |
| repeat01 | THPT (tok/s) | 5948.029629 | 6404.273989 | 456.244359 | 7.670513% | PASS |
| repeat02 | TTFT (ms) | 725.492280 | 463.897052 | 261.595228 | 36.057617% | FAIL |
| repeat02 | TPOT (ms) | 26.255496 | 15.565031 | 10.690465 | 40.717055% | FAIL |
| repeat02 | E2E (ms) | 27584.864259 | 16386.923646 | 11197.940614 | 40.594510% | FAIL |
| repeat02 | THPT (tok/s) | 5976.592892 | 6404.273989 | 427.681097 | 7.155935% | PASS |
| repeat03 | TTFT (ms) | 718.268047 | 463.897052 | 254.370995 | 35.414494% | FAIL |
| repeat03 | TPOT (ms) | 25.926229 | 15.565031 | 10.361198 | 39.964153% | FAIL |
| repeat03 | E2E (ms) | 27240.799892 | 16386.923646 | 10853.876246 | 39.844191% | FAIL |
| repeat03 | THPT (tok/s) | 5965.065720 | 6404.273989 | 439.208269 | 7.363008% | PASS |

Decision under D-NO-INTERRUPT: this completes the F1a comparison requested by decision46. Preserve the failing readings; do not tune specification links against these native values or promote B30a/the diagnostic backend. Continue the two existing F2 replays, then assemble the F1–F4 comparison. A6 repeatability is independently complete as a diagnostic with no qualification change; see [report](test_report_2026-11-01_a6_repeatability.md). Formal tables, K19, metric endpoints, thresholds and three references remain unchanged.

The initial screen, predictions and then-pending statuses below are historical. The measured backend stays formal; no diagnostic variant or B30a default change is promoted.

## Comparison collector verification — 2026-11-01

`tests/analysis/v4_f_variant_comparison.py` reads retained canonical outputs and does not execute a normalizer, gate or simulator. It checks the eight expected B29/F1a/F2/F4 readings against the corresponding B29 reference CSV bytes, six compute-table hashes, fixed request/token population, all four metrics, every repeat and the unchanged10% threshold. Its output uses a nonnegative absolute error and retains a separate signed delta. F1b coverage failures and F3 invalid-fit results have null metrics. F4 retains its pre-B29 zero-delay configuration.

Before F2 termination, the existing six readings were exercised directly through the collector: **PASS, six readings,96 pooled/per-repeat metric entries, three identical reference files per reading, six identical compute inputs**. This is a collector check on existing evidence; F2 is still running and no complete comparison result is written. Evidence: `h3o_operator_costs/f_comparisons_20261101/comparison_precheck.json` and `precheck_command.py`. Initial checks exposed three assumptions about old records: the canonical E2E name is `request_e2e_time_ms`; F4 predates the delay fields; and F4 predates the receipt's `cc_backend_types` field. The collector now reads the inspected canonical name and actual configuration. No input, old result or gate changed.

Command, current worktree, Python3.10.6 with30GiB RLIMIT_AS:

```bash
env PYTHONDONTWRITEBYTECODE=1 timeout 60s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/glm53_dsv4/f_comparison_precheck_20261101.py
```

The exact command script is retained as `precheck_command.py`. Final exit0; AST parsing and `git diff --check` PASS. The complete collector command remains pending both F2 terminal receipts and the existing watcher's once-run readings:

```bash
env PYTHONDONTWRITEBYTECODE=1 timeout 60s /data/ycfeng/envs/frontier-py310/bin/python tests/analysis/v4_f_variant_comparison.py --task-root task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800 --stage-root /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4 --output task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_comparisons_20261101/complete
```

## Historical initial scope

YC ruled on the ex-situ audit (decision 46): measured communication data is an allowed input, so the measured backend stays the formal input. The F1-F4 repair proposals of the audit ([test_report_2026-10-31_ex_situ_input_audit.md](test_report_2026-10-31_ex_situ_input_audit.md), section 5) run as diagnostic A10 variants, and their errors are compared with the B28 baseline. A variant replaces a formal input only after a new decision by YC.

| Variant | Input replaced | Replacement | Status |
| --- | --- | --- | --- |
| F1a | measured collective table | `astra_sim_analytical` with H200 specification link parameters | CPU per-call screen done (section 2); A10 replay waits for B29 and B30a |
| F1b | measured collective table | `astra_sim_analytical` with link parameters fitted from one node and a two-node one-rank-per-node test | probes and fit tools in B30b; GPU job not submitted |
| F2 | K = 19 from the native P2-64 run | K predicted from a one-GPU launch-depth measurement | probe in B30b; GPU job not submitted |
| F3 | H800 NIXL fit (126.2 Gbit/s, 64.7 us) | H200 NIXL fit with the V4 KV layout | probe in B30b; GPU job not submitted |
| F4 | native explicit `num_blocks` | Frontier's memory planner (`memory_planner_profiled`) | planner terms done (section 3); replays running |

Baseline (B28, main `ee29d548`, warm, gate `stage4/a10_gate/p2_{32,64}_warm_k19b28_post/gate_stdout.txt`):

| Case | TTFT | TPOT | E2E | Throughput | Gate |
| --- | --- | --- | --- | --- | --- |
| P2-32 | −11.22% (323.18 against 364.03 ms) | +0.04% | +0.16% | +0.08% | FAIL (TTFT) |
| P2-64 | −10.60% | +2.22% | +2.45% | +0.22% | FAIL (TTFT) |

## 1. Method of the F1 per-call screen

`h3o_operator_costs/f_variants/f1_astra_per_call.py` (commit `b0e568f5`; wrapper `run_f1_per_call.sh`) builds the DECODE-cluster CC backend of each A10 case through `Cluster`, so the backend gets the same materialized layout as in a replay (P2-32: 2 servers x 8 GPUs, TP8 DP2 EP16; P2-64: 4 x 8, TP8 DP4 EP32). It prices every TP and EP row of the formal H200 table (`data/profiling/network/h200_dgx/summary.csv`, the table the A10 inputs resolve) at the row's payload, the largest rank's input bytes, which is the payload rule of the measured backend. Rows of one call are pooled over passes by their median. The output reports the predicted/measured ratio per curve and per CUDA Graph decode forward (calls per forward x per-call time).

This screen checks only the collective prices. It does not run the simulator; the effect on TPOT and TTFT below is an inference.

## 2. F1 per-call results

Per CUDA Graph decode forward, predicted/measured ratio (graph token counts 1-32 per lane):

| Backend parameters | EP16 (P2-32) | EP32 (P2-64) | TP8 |
| --- | --- | --- | --- |
| `astra_sim_analytical` defaults (FullyConnected, 600 / 100 Gbit/s, 1 / 1 us) | 0.159-0.183 | 0.141-0.198 | 0.846-1.096 |
| Specification links (Switch, 3,600 / 400 Gbit/s, 1 / 1 us) | 0.239-0.299 | 0.204-0.337 | 1.567-1.911 |
| FullyConnected, 3,600 / 400 Gbit/s, inter latency 5, 6, 7 or 8 us | 0.124-0.152 | 0.105-0.170 | 0.799-0.964 |

Measured decode forward (graph_t1-t32): EP16 8.69-11.19 ms, EP32 15.97-27.11 ms, TP8 0.66-0.82 ms.

Per-curve median ratios with the specification links: `all_gatherv` EP16 0.215, EP32 0.440; `reduce_scatterv` EP16 0.485, EP32 0.169; `all_reduce` TP8 2.75; `all_gather` TP8 0.559; SP `all_gather` TP8 1.75; SP `reduce_scatter` TP8 1.46.

Observations:

1. **astra defect: a flat ring never uses the inter-server link.** `_rank_to_physical_address` returns `(local index, server index)` and `_get_first_differing_dimension` returns the first component that differs (`frontier/cc_backend/backends/astra_sim_analytical_cc_backend.py:435-450`). Ring neighbours on different servers, such as 7 -> 8, differ first in the local index, so `_send_ms` prices them on the intra-server link. Direct check on the P2-32 and P2-64 backends with inter latency 50 us and intra latency 1 us: `send(0 -> 1)` = 0.001 ms, `send(7 -> 8)` = 0.001 ms, `send(0 -> 8)` = 0.05 ms. The inter latency sweep 5-8 us gave byte-identical `per_call.csv` files. No unit test covers multi-server rings. Fix: assignment B30a.
2. **Payload geometry of the dispatch.** Native V4 dispatch is three grouped `all_gatherv` calls (hidden states, top-k weights, top-k ids); Frontier's analytical path prices one all-gather over the summed 8,264 B, so it carries the per-step latency of one call. The measured per ring step cost is about 9.9 us for `all_gatherv` and 4.5 us for `reduce_scatterv` at EP16 (8.3 us dispatch per step at EP32), so one per-link latency cannot fit both operations. A single latency could fit their per-layer sum.
3. **Size of the gap.** With the specification links the EP decode collectives come out at 0.20-0.34 of measured, and TP8 at 1.6-1.9.

Inference (not a replay): EP collectives are about 8.9 of the 19.88 ms P2-32 decode forward and 16.05 of the 25.30 ms P2-64 forward (audit `comm_share.txt`). At the specification-link ratios, the decode forward would shrink to about 14.1 ms (P2-32) and 15.1 ms (P2-64), so TPOT would move by about −29% and −40%. F1a with specification parameters is therefore expected to fail TPOT clearly, even after the B30a fix moves the cross-server steps onto the inter-server link.

## 3. F4 planner terms

`h3o_operator_costs/f_variants/f4_planner_blocks.py` (commit `c5094806`) initializes the A10 configuration with `vllm_v1_scheduler_config_num_blocks_mode=memory_planner_profiled` instead of the native explicit count. Outputs `stage4/f_variants/f4/p2_{32,64}_init/planner_blocks.json`:

| Case | Native blocks | Planner blocks per decode lane | Difference |
| --- | --- | --- | --- |
| P2-32 | 109,934 | 116,034 | +5.55% |
| P2-64 | 112,914 | 120,623 | +6.83% |

Planner terms: requested memory 128,687,957,606 B (141 GiB x 0.85; torch reports 139.81 GiB for the device, about 1.01 GiB less at 0.85), parameter memory per device 12,393,714,140 B (P2-32) and 7,794,659,804 B (P2-64) from `ParamCounter` (about 1.13 and 2.63 GiB below the native weight memory), non-KV overhead 0 (native about 3.54 GiB), block size 1,002,240 B. A driver fix was needed first: `kv_admission` derived the native per-request block range from the simulated pool, so a planner pool failed the driver after initialization; `6a7ff2a4` reads both native values from the case.

Full replays `stage4/f_variants/runs/f4_p2_{32,64}_planner` (detached worktree `a10-f-variants` at `6a7ff2a4`, warm cache copied from B28) started at about 09:10 UTC. Results: pending.

## 4. Commands and environment

- Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6, `PYTHONPATH` the repository root, `ulimit -v 31457280` (screens) or `62914560` (replays).
- F1 screen: `bash task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_variants/run_f1_per_call.sh <name> [field=value ...]`; outputs `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/f1/<name>/{per_call.csv,summary.json}`. Runs: `astra_default`, `astra_spec`, `astra_fc_lat{5,6,7,8}`; `astra_default_r2` and `_r3` reproduce `astra_default` byte for byte after the script cleanup.
- Dimension check: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/scripts/check_astra_dims.py`.
- F4: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/run_f4_init.sh <case> <name>` and `run_f_variant.sh <case> <name> [field=value ...]`.

## Limits

- The F1 screen prices the measured rows only; it does not include the effect of overlap or the eager prefill paths on the request metrics.
- The TPOT effect of F1a is an inference from the forward composition, not a replay.

# Test report: F1-F4 comparison variants of the A10 inputs (decision 46)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Codex update: F4/B28 five-artifact identity on both scales, B29 per-repeat results, F1a P2-32 final FAIL and fresh P2-64 replay. |
| 2026-10-31 | Created: F1 per-call screen and astra defect, F4 planner terms; F4 replays, B29, B30a and B30b pending. |

## Latest comparison checkpoint — 2026-10-31 Codex resume

B29 is merged and its finished measured-CC P2-32 reading PASSes all three references. P2-64 remains FAIL: repeat 1 TTFT 779.888727400 vs 684.313826604 ms, 12.254940665%; pooled 7.67691% does not change the verdict. F1a P2-32 is FINISHED: TTFT 3.31698%, TPOT 28.38101%, E2E 27.82468%, THPT 3.79951%, FAIL. A fresh F1a P2-64 replay is running at `stage4/f_variants/runs/f1a_p2_64_codex_20261031a`; the original incomplete attempt remains retained.

F4 at both scales is FINISHED and its five metric artifacts are byte-identical to B28. It used zero ingress/handoff, so its TTFT difference from B29 is a serving-delay difference. Planner-only metric effect is zero on these traces. F1b/F2/F3 preparation remains with B30b. Complete values, all per-repeat verdicts, exact commands, clean source revisions and limits: [Codex test report](test_report_2026-10-31_codex_resume.md) and `h3o_operator_costs/codex_resume_20261031/`.

The initial screen, predictions and then-pending statuses below are historical. The measured backend stays formal; no diagnostic variant or B30a default change is promoted.

## Scope

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

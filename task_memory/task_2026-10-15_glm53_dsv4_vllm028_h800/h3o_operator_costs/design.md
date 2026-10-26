# H3o-(a): operator-level DSV4 costs — scope, files, GPU jobs and time (proposal for YC's review)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Created: CPU-only proposal under D-OPERATOR-LEVEL-ROUTE (requirements item 27). Nothing in it is implemented or submitted. |

## 1. Goal and acceptance

Goal: Frontier prices every DSV4 step of the P0 (H800) and P2-32 / P2-64 (H200) cases by composing operator costs for the batch being priced. Each operator cost comes from an operator profiling CSV through a trained predictor, or from the communication backend. The step-cost table (`dsv4_step_cost.json`) is used only to cross-check the composed step time (D-NO-STEP-COST-SHORTCUT, ADR 0002).

H3o-(b) is complete when all of the following hold:

1. **No dummy values.** A non-dummy V4 run computes no scope from `predictor._dummy_execution_time`, and reads no step-cost table unless a diagnostic driver selects one explicitly.
2. **Errors outside the data.** A batch outside the measured range of any operator stops with a typed `DeepSeekV4CoverageError` that names the operator and the missing coordinate. Nothing is extrapolated or substituted.
3. **Predictor accuracy on held-out rows.** On rows left out of training, every per-layer scope prediction has a median relative error ≤ 5% and a p90 relative error ≤ 10%. The report lists each scope with a larger error.
4. **End-to-end runs.** The P0, P2-32 and P2-64 replays run end to end on the operator path, on their own hardware's data only.
5. **Diagnostic cross-check reported.** The report compares the composed step time with the native step time for each step class: decode-only by batch size, first tail with decodes, first tail alone, and prefill. It also gives the per-phase difference. This is a diagnostic, not a gate.
6. **Other paths unchanged.** The GLM, co-location, PDD and PD-AF paths keep identical results: the 77-case fidelity regression, the dummy smoke matrix and the unit tests pass.

The formal accuracy gates (the four metrics against native vLLM) belong to H4/H6 and come after H3o.

## 2. What already exists (inspected 2026-10-26)

| Input | Location | Content | Usable for |
| --- | --- | --- | --- |
| W6 attention grid, H800 | `/data/ycfeng/tmp/glm53_dsv4/w6_dsv4_profiles/attention_kernel_only.csv` (source `3ea2c3f`) | 7116 rows; KERNEL_ONLY, eager, one layer sampled; reduced native engine (physical TP1, per-rank TP8 dimensions), dummy weights, FP8 block quantization, 3 auxiliary streams. Decode: batch sizes 1-1024, contexts powers of two (258-32902), uniform and non-uniform mixes. Prefill: batch 1, 255-8192 tokens. No mixed rows, no first-tail rows | Training rows for the attention predictor (shape and trend). It is sparse near the serving window: no decode context between 4105 and 8194 for batch 1, and no 6144-token prefill |
| q1 attention window, H800 | `/data/ycfeng/tmp/glm53_dsv4/v4_h800_q1_b36c2b9_results/Q1_6143_7166/.../attention_kernel_only.csv` | 6144 rows; decode batch 1, contexts 6143-7166, KERNEL_ONLY eager | Dense context coverage at batch 1 |
| P6144 prefill, H200 | `/data/ycfeng/tmp/glm53_dsv4/v4_attention_cwd02_results/P6144/compute/h200/...` | 6 rows; CUDA_EVENT (with the outer `dsv4_attention_layer` scope) and KERNEL_ONLY | H200 prefill at 6144 only |
| Routed MoE grouped GEMM, H800 | `task_2026-10-23_v4_remaining_measurements/moe/results/primary_acceptance.json` | MXFP4 MARLIN, EP16 rank 0, balanced routing, M = 16/32/6152/12288; KERNEL_ONLY means 0.102/0.102/0.756/1.299 ms | Routed expert compute at the four observed P0 populations |
| Collective curves, H800 | `data/profiling/network/h800_dgx/summary.csv` and `glm_p0_ep16_20261021b/` | GLM serving process groups. TP8 all_gather/all_reduce/reduce_scatter and EP all_gatherv/reduce_scatterv by input bytes; smallest point 12 KB (all_gather) and 384 KB (reduce_scatterv) | V4 collectives whose payload is inside the measured byte range (`MeasuredCCBackend`, no extrapolation) |
| Collective curves, H200 | none | — | Nothing |

A rough diagnostic sum from these rows shows the size of the gap (not a prediction). For an H800 decode step at batch 16 and context near 8192, attention for 21 CSA, 20 HCA and 2 uncompressed layers is about 5.6 ms, and routed MoE at M=32 for 43 layers is about 4.4 ms. That is about 10 ms of the 19.3 ms native decode step. The other ~9 ms consists of operators with no data yet: mHC, shared expert, gating and routing, embedding, final norm, LM head, the SP and EP collectives, and the device idle time between kernels.

## 3. Operator inventory and coverage

The phase order is the one in `deepseek_v4.py::timing_operator_times`.

| Phase | Operators (scope names) | Current source in Frontier | Data after H3o-(b) |
| --- | --- | --- | --- |
| attention | the nine `dsv4_*` attention scopes per layer variant (csa / hca / uncompressed) | dummy (`_predict_dsv4_dummy_attention_time`) | Trained V4 attention predictor (new). Data: W6 + q1, plus the serving-window job G1/G1h |
| attention | `dsv4_mhc_*_pre`, `dsv4_mhc_*_post` (hc_mult 4, Sinkhorn) | dummy | V4 operator rows from G2/G2h |
| attention | `dsv4_sp_attention_allgather`, `dsv4_sp_attention_reduce_scatter` (ATTN_TP8) | communication backend | Measured curves. Coverage is checked first; H200 needs G5h |
| pre_dispatch | `moe_gating_linear` (4096→256), `moe_gating_routing_topk` (score top-6; hash routing on the 3 hash layers) | dummy | V4 operator rows from G2/G2h |
| dispatch | `dsv4_ep_activation_allgather`, `dsv4_ep_weights_allgather`, `dsv4_ep_ids_allgather` (EP16 / EP32) | communication backend | Measured curves. The ids and weights payloads at decode are below the smallest measured point, so they need G5/G5h points |
| routed_compute | `moe_grouped_gemm` (MXFP4 MARLIN) | dummy | Existing MoE trainer on the accepted rows plus G3/G3h points |
| combine | `dsv4_ep_output_reduce_scatter` | communication backend | Measured curves (G5/G5h for small payloads) |
| post_combine | `share_expert_up_proj/act/down_proj` (FP8, width 2048, replicated) | dummy | V4 operator rows from G2/G2h |
| first / last layer | `dsv4_embedding` + `dsv4_embedding_allreduce`, `dsv4_mhc_final_post`, `dsv4_mhc_head`, `dsv4_final_norm`, `dsv4_lm_head` + `dsv4_logits_gather`, `dsv4_sp_final_mhc_allgather` | dummy, plus communication | V4 operator rows from G2/G2h, plus communication |
| between kernels | device idle before each kernel (eager and graph replay) | none | `kernel_gap.csv` from the existing `frontier.profiling.kernel_gap.main` (G4/G4h) |

## 4. Measurement types and overlap

- **Decode steps** run in vLLM's breakable CUDA Graph mode (FULL plus PIECEWISE; the attention core is an eager segment). They are priced from KERNEL_ONLY rows plus `kernel_gap.csv`, through the existing kernel-gap path (`kernel_gap.py`), which charges the gap once per kernel using the measured kernel counts. The W6 table already records `kernel_count.*` for every attention scope.
- **Eager steps** (prefill, and first-tail steps when vLLM runs them without a graph) are priced from CUDA_EVENT rows, which include the device idle time caused by host launches.
- **Overlap inside an attention layer.** Three auxiliary streams run the compressor and indexer while the main stream continues, so the sum of the scope times is larger than the layer's busy time. The earlier proposal measured this: the owned-kernel sum exceeds the union of kernel intervals. Proposal: the profiler also writes `dsv4_attention_layer` for KERNEL_ONLY as the union of the layer's kernel intervals (CUDA_EVENT already has this scope). The predictor prices a layer from `dsv4_attention_layer` and keeps the nine scopes only for attribution. With this choice, no overlap factor appears anywhere.
- **Overlap between the shared expert and the routed path.** The step stays serial as in `timing_operator_times`. The native trace of the P0 reference measures the difference, as a diagnostic. If it is material, a later change models the stream explicitly.

## 5. Frontier file list (H3o-(b))

| # | File | Change | Lines (est.) |
| --- | --- | --- | --- |
| 1 | `frontier/execution_time_predictor/dsv4_predictor.py` (new) | Load and validate V4 attention and V4 operator rows; train one model per (scope, layer variant, measurement type) through the caller's `train`; predict with exact lookup, then fitted interpolation inside the measured bounds, else `DeepSeekV4CoverageError`. Same structure as `dsa_predictor.py` | ~250 |
| 2 | `frontier/attention/dsv4_features.py` (new) | Batch features per layer variant: query tokens, numbers of decode and prefill sequences, sum and max of contexts, window key length, compressed key length, selected compressed key length (`min(index_topk, compressed)`), whether index scores execute. Same structure as `dsa_features.py` | ~90 |
| 3 | `frontier/execution_time_predictor/deepseek_v4.py` | `local_scope` and the structural path take trained predictions; remove the non-dummy raises that the new data covers. Delete `read_isolated_scope` (lines 665-925): no runtime path in `frontier/` calls it, only `tests/unit/test_dsv4_graph_measured.py` and `tests/integration/check_dsv4_measured_scopes.py`. Its exact-row lookup is superseded by file 1 | +80 / −270 |
| 4 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | Register the V4 loader, trainer and predictor next to the DSA entries (lines 1444, 3574, 4052); `_predict_dsv4_dummy_attention_time` (7583-7621) calls the predictor when not in dummy mode | ~40 |
| 5 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `_predict_dsv4_structural_execution` (95-115) and the V4 FFN path (1380-1395) price gating, routing, grouped GEMM and the shared expert from trained models | ~40 |
| 6 | `frontier/model_architectures.py`, `frontier/attention/deepseek_v4.py` | Text of `runtime_limitations`; `require_enabled_for_execution` admits the operator path when its inputs are present | ~15 |
| 7 | `frontier/profiling/attention/dsv4_attention_plan.py` | Serving-window decode cases (batch × non-uniform contexts in the window), the 6144 prefill, first-tail cases (one prompt tail with 1 token over a 6143-token context, alone and with decodes) | ~90 |
| 8 | `frontier/profiling/attention/dsv4_attention_rows.py` | KERNEL_ONLY `dsv4_attention_layer` as the union of kernel intervals | ~40 |
| 9 | `frontier/profiling/linear_op/` V4 operator producer (new module, or an extension of `dsv4_attention_runner.py`) | Time mHC pre/post/head, shared expert, gating linear, score and hash routing top-k, embedding, final norm and LM head in the same reduced native engine, with typed operator contracts. The current reduced engine forces a dense MLP, so this variant keeps the native MoE block's gate and shared expert | ~250 |
| 10 | Tests: `tests/unit/test_dsv4_predictor.py` and `test_dsv4_features.py` (new); updates to the existing V4 dispatch tests; the `read_isolated_scope` cases in `test_dsv4_graph_measured.py` and `check_dsv4_measured_scopes.py` move to the new predictor; V4 cases in the fidelity regression | ~300 |

Total: about 10 files, about 1,150 lines added and about 270 deleted. This is above the D-H4A-CHECKPOINT limit of 5 files and about 200 lines, so it needs YC's approval.

- **No new config field.** V4 attention rows use the existing attention input, V4 operator rows use `linear_op.csv` with typed operator contracts, routed MoE uses `moe.csv`, collectives use the measured backend's `summary.csv`, and the device idle time uses `kernel_gap.csv`.
- **Large module.** `sklearn_execution_time_predictor.py` (8666 lines) is already listed in `docs/general/module-size-exceptions.md`. The new logic goes into files 1 and 2; the large module gets only the registration lines.

## 6. GPU profiling jobs

All jobs go through the StepMind Python `RJobBackend` with `i-fengyicheng` auth, via `/ds4-exec`. H800 jobs use codesign and H200 jobs use step_main, non-preemptible. H800 and H200 evidence never qualify each other. Each single-GPU job reuses the existing V4 profiling image and the reduced native engine of the W6 producers.

| Job | GPUs | Content | Estimated GPU-hours |
| --- | --- | --- | --- |
| G1 (H800) | 1 | Attention serving window. Decode batch 1/2/4/8/12/16/20 × contexts 6143-7168 (uniform and mixed); prefill 6144; first-tail alone and with decodes. KERNEL_ONLY and CUDA_EVENT | ≤ 2 |
| G2 (H800) | 1 | V4 non-attention operators (§3) over local rows 1-64 (decode), 768 (SP prefill) and 6144. KERNEL_ONLY and CUDA_EVENT | ≤ 1 |
| G3 (H800) | 1 | Routed grouped GEMM: additional M = 48 and 64 (P0 maximum decode concurrency 18.6 gives up to 3 local rows per DP rank) | ≤ 0.5 |
| G4 (H800) | 1 | `kernel_gap.csv` for V4, eager and graph replay | ≤ 0.5 |
| G5 (H800) | 16 (2 nodes) | Only if the CPU coverage check (step 2 in §8) finds V4 payloads outside the existing curves: small-payload points for TP8 and EP16 all_gatherv / reduce_scatterv | ≤ 12 |
| G1h-G4h (H200) | 1 each | Same as G1-G4 with the H200 window: decode batch up to 40, since the maximum concurrency is 37.3 / 38.4; routed M for DP2/EP16 and DP4/EP32 | ≤ 4 |
| G5h (H200) | 16, then 32 | No H200 collective data exists. TP8, EP16 and EP32 curves over the V4 payload range, with the existing `vllm_collective_benchmark.py` | ≤ 12 + 24 |

Total: at most about 16 GPU-hours on H800 and 40 on H200, within the H200 limit of 64 non-preemptible GPUs. G5h-EP32 uses 32 GPUs, so its checks are soft gates only. If H800 quota stays short, the H200 jobs run first and I report the shortage at once (D-GPU-SHORTAGE-REPORT).

## 7. Time estimate (elapsed, working days)

| Step | Days |
| --- | --- |
| Producer changes (files 7-9) and CPU checks | 1.5 |
| GPU jobs G1-G4 / G1h-G4h, then G5h (queue-bound; run in parallel with the predictor work) | 1-2 |
| Predictor, features and wiring (files 1-6) | 2 |
| Tests, fidelity regression and smoke matrix | 1 |
| Training on the new rows, held-out check, P0 / P2 replays and the diagnostic cross-check | 1 |
| **Total** | **about 6-8** |

The main schedule risk is GPU queueing and image or runtime failures. Earlier V4 profiling jobs needed several attempts.

## 8. Order of work

`CPU coverage check of the existing collective curves against V4 payloads -> {producer changes (7-9) -> G1-G4, G1h-G4h, G5h; predictor and wiring (1-6)} -> training and held-out check -> P0 / P2 replays -> diagnostic cross-check -> H4`

- The producer changes and the predictor work run in parallel. The predictor can be built and unit-tested on the W6 and q1 rows before the new jobs finish.

## 9. Risks

1. **First-tail step.** The native first-tail step alone takes 61-69 ms, against 20-24 ms for decode-only. Its graph mode and phase identity are open (ticket 05). G1 measures the tail shapes, but if vLLM runs this step eagerly, the host launch time decides the cost. Then G1's CUDA_EVENT rows must cover it, or the CPU-overhead tables (`vllm_cpu_probe`) are needed; that would need a decision from YC.
2. **Single layer versus 43 layers.** The rows sample one layer. L2 cache and stream interactions across 43 layers are not proven additive. The cross-check shows the size of this effect; it is not corrected with a factor.
3. **Routing.** Rows use balanced routing and dummy weights. Real router imbalance (ticket 14) is not modeled, and the hash layers' load comes from the hash table.
4. **Graph replay.** KERNEL_ONLY plus kernel gap approximates graph replay. The breakable-graph eager attention segment can add host time that this approach does not capture.

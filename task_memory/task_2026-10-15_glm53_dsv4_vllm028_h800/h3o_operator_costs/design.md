# H3o-(a): operator-level DSV4 costs — scope, files, GPU jobs and time (proposal for YC's review)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Revision 2 after Codex review round 1 (`codex-review-9c8b2435-…/review.md`, verdict `changes_required`, P1 × 5, P2 × 3). Agreed changes: per-operator composition with a source-defined stream graph instead of a layer union; runtime-mode census and pilots before the grids; full once-only ledger with PDD admission, endpoint collectives, block zeroing and pre-forward preparation; unequal-combine binding; EP rank classes; `balanced_topk` scope; workload-owner features and joint coverage; structural acceptance gates; recomputed files, GPU jobs and time; decisions for YC. Per-finding positions are in `review.md`. |
| 2026-10-26 | Created: CPU-only proposal under D-OPERATOR-LEVEL-ROUTE (requirements item 27). Nothing in it is implemented or submitted. |

## 1. Goal, scope and acceptance

Goal: Frontier prices every DSV4 forward of the P0 (H800) and P2-32 / P2-64 (H200) cases by composing operator costs for the batch being priced. Device costs come from operator profiling CSVs through trained predictors and from the measured communication backend. The step-cost table (`dsv4_step_cost.json`) is only a diagnostic cross-check of the composed time (D-NO-STEP-COST-SHORTCUT, ADR 0002). No cost, factor or residual is derived from it, or from any other measured whole step.

Runtime scope, as the three cases are configured: PD disaggregation with `PREFILL` and unified `DECODE` roles; `balanced_topk` routing on all 43 layers (every router is `RoutingSimulatorRouter`, `native_config.json:42`); speculative decoding / MTP off; prefix caching off. Real score or hash routing (ticket 14) and PD-AF are out of scope. A run whose runtime path differs from this scope stops with a typed error.

H3o-(b) is complete when all of the following hold:

| # | Gate | Check |
| --- | --- | --- |
| A1 | No dummy values, no default step table | A non-dummy V4 run reads no `predictor._dummy_execution_time` (a test makes any read raise) and binds no step-cost table unless a diagnostic driver selects one explicitly |
| A2 | Kernel ownership | In every pilot and grid window, each kernel belongs to exactly one operator scope; unowned kernels are listed and fail the window; an outer scope and its child scopes are never both charged |
| A3 | Once-only ledger | With sentinel costs, each operator is charged the number of times in §4 per forward (for example `dsv4_mhc_first_pre` × 1, `dsv4_mhc_ffn_post_attn_pre` × 42, `dsv4_mhc_attn_post_ffn_pre` × 43) and in its owning phase |
| A4 | Composition check | Pilot P1: the composed time of one attention layer (critical path over the stream graph of §3.2) against the stream-joined elapsed of the same call. Median ≤ 5%, p90 ≤ 10%. A failure stops the route and is reported; no factor is introduced |
| A5 | Typed coverage | Coverage is admitted jointly per workload group (mode, planner role, phase, layer variant, rows), not by independent min/max bounds. Outside it, a typed `DeepSeekV4CoverageError` names the operator and the missing coordinate. Collision tests: planner versus reuse layer, equal aggregates with different compressed writes, swapped active DP owner, unmeasured ordered SP vector |
| A6 | Predictor accuracy | Held out by complete workload group: per scope, median relative error ≤ 5% and p90 ≤ 10%. Scopes that are legally zero or no-op are checked for exact zero instead (no relative error with a zero denominator) |
| A7 | EP rank admission | Every EP lane of the target replays binds to a measured rank class (§3.5); the per-rank costs and their critical-path maximum are checked, not only rank 0 |
| A8 | Collective coverage | Every collective of the replays matches a measured (operation, group, cardinality, ordered local rows, bytes) point or a uniform curve inside its range; an unequal combine never falls back to the scalar curve |
| A9 | Cache identity | Cold-cache and warm-cache runs give identical predictions and artifact identities |
| A10 | End-to-end runs | P0, P2-32 and P2-64 replay end to end on the operator path, each on its own hardware's data only |
| A11 | Diagnostic cross-check reported, not a gate | Composed versus native time per step class (decode-only by batch size, first tail alone, first tail with decodes, prefill) and per phase. It never feeds a cost |
| A12 | Other paths unchanged | GLM, co-location, PDD and PD-AF keep identical results: 77-case fidelity regression, dummy smoke matrix and unit tests pass |

The formal accuracy gates (the four metrics against native vLLM) belong to H4/H6 and come after H3o.

## 2. What already exists and what it qualifies (inspected 2026-10-26)

| Input | Location | Content | Qualifies |
| --- | --- | --- | --- |
| W6 attention grid, H800 | `/data/ycfeng/tmp/glm53_dsv4/w6_dsv4_profiles/attention_kernel_only.csv` (source `3ea2c3f`) | 7116 rows, KERNEL_ONLY, eager, reduced native engine (physical TP1, per-rank TP8 dimensions), dummy weights, 3 auxiliary streams, one layer sampled. Counts are named `kernel_count.<scope>`. Interval coordinates are not kept | Per-scope busy time and trends; training candidates only. Sparse in the serving window (no batch-1 context 4105-8194, no 6144 prefill, no mixed or first-tail rows) |
| q1 attention window, H800 | `/data/ycfeng/tmp/glm53_dsv4/v4_h800_q1_b36c2b9_results/Q1_6143_7166/…/attention_kernel_only.csv` | 6144 rows, decode batch 1, contexts 6143-7166 | Dense context coverage at batch 1 |
| P6144 prefill, H200 | `/data/ycfeng/tmp/glm53_dsv4/v4_attention_cwd02_results/P6144/compute/h200/…` | 6 rows, CUDA_EVENT and KERNEL_ONLY | H200 prefill at 6144 only |
| Routed MoE, H800 | `task_2026-10-23_v4_remaining_measurements/moe/results/primary_acceptance.json` | MXFP4 MARLIN, physical TP1, EP16 rank 0, 16 local experts, balanced routing, M = 16/32/6152/12288. `native_tuple_cost_admission`, `serving_admission` and `simulator_consumption` are all `false` | Isolated training candidates. Not serving costs: at M16, 10 of 16 EP ranks get no tokens; at M32, 4 do (`moe_grid_design.md:32-45`) |
| Collective curves, H800 | `data/profiling/network/h800_dgx/summary.csv` (EP32 GLM) and `glm_p0_ep16_20261021b/summary.csv` | Uniform curves by input bytes. EP16: smallest all_gatherv point 12,352 B, smallest reduce_scatterv point 196,608 B | Uniform V4 collectives inside the byte range only. The V4 decode payloads at L1 are below it: hidden all-gather 8,192 B, weights and ids 24 B each, M16 reduce-scatter 131,072 B. So G5 is required, not optional |
| Collective curves, H200 | none | — | Nothing |
| Native P0 traces, H800 | `/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04/torch_traces/` | Torch-profiler traces of all 32 ranks of a native P0 deployment (physical TP8 / DP2 / EP16, case `dsv4-h800-p0-profile-fixed128-qps1`), 512 recorded steps per rank, with kernels, streams, runtime calls, graph launches and Frontier annotations | Runtime-mode census of H800 (stage 0). One CPU look at decode dp1 rank 0: the 61 useful decode steps ran as FULL graph replays (150 `cudaGraphLaunch`, about 915 kernels per launch, annotation `model_dispatch_run_fullgraph`), and 5,315 eager kernels ran on a separate stream (input and metadata preparation, for example `_compressed_slot_mapping_kernel`). Profiler overhead inflates host times, so host magnitudes from these traces are upper bounds |

The existing CPU-overhead mechanism prices an eager step as two streams: host `forward_launch` against device KERNEL_ONLY operators (`sklearn_execution_time_predictor.py:992-1006, 7125-7140`; enabled by `measurement_input_paths.py:160` when `skip_cpu_overhead_modeling` is false and the CPU-overhead table has `forward_launch`). The unequal-combine producer `frontier/profiling/collectives/vllm_forward_collectives.py` and consumer `measured_cc_backend.py:132-151, 217-229` exist.

## 3. Cost contract

### 3.1 Units of cost

The unit is one operator scope, or one native fused scope that vLLM executes as a unit (for example `dsv4_compressor_fused_compress_store`). Each scope's device cost is the sum of its owned kernels (KERNEL_ONLY) for the batch being priced. No layer union, layer elapsed or other composite is a cost, and no layer total is spread over child scopes. CUDA_EVENT rows are used for validation and for scopes whose events enclose a single kernel stream, never as a residual assigned to CPU.

### 3.2 Device stream composition

- **Across phases**: serial, in the phase order of `timing_operator_times` (`deepseek_v4.py:17-63`), with the shared expert in `pre_dispatch` as the code has it.
- **Inside an attention layer**: the compressor and indexer scopes run on auxiliary streams. The layer time is the critical path through a fork/join graph of scopes. The graph is taken from the vLLM V4 attention source (stream switches and event waits), recorded once per layer variant, and checked against the retained kernel intervals of pilot P1 and the native traces. Edge weights are the predicted scope costs. This keeps approved Option A (`task_2026-10-23_v4_remaining_measurements/proposal.md:23-55`): per-operator costs plus an explicit dependency account.
- **Shared expert versus routed path**: serial. The native traces measure any overlap, as a diagnostic. If it is material, a later change models it explicitly.

### 3.3 Runtime modes and the time between kernels

Stage 0 classifies each step class by its native runtime mode: FULL graph, PIECEWISE (breakable, with eager segments), or eager. The first trace look suggests decode-only steps are FULL. Each mode is priced as follows:

| Segment | Price |
| --- | --- |
| Graph-internal | Device costs plus the graph-internal gap per kernel on the critical path, from a V4 `kernel_gap.csv` measured on graph replays. Kernel counts use the `time_stats.<op>.kernel_count` naming that `kernel_gap.py:33-38` reads |
| Eager segment and eager step | The existing two-stream model: host `forward_launch` against the device stream. Decision D2 in §9 |
| Seam between segments, and pre-forward preparation | Measured as named operators or seam terms only if stage 0 shows them as material; each is charged once, in one owner |

The V4 EP path (`scheduler/utils/expert_parallel.py:129-139`) is rewired through `_predict_layer_with_kernel_gap`, so FFN phases get the same gap treatment as the attention phase.

### 3.4 Host cost

The V4 stage-owned CPU fields (`deepseek_v4.py:360-375`, currently dummy) are bound to the existing CPU-overhead inputs. With `skip_cpu_overhead_modeling=True`, the policy is an explicit zero and the report says so. With it false, no dummy value remains. CPU cost is never derived as native step minus operator sum.

### 3.5 Collectives and MoE

- **SP and EP collectives**: `MeasuredCCBackend`. The V4 combine `dsv4_ep_output_reduce_scatter` is bound to the ordered-SP-vector consumer, by classifying the combine operators in the communication registry instead of comparing one name (`operators/sequence_parallel_moe.py:133-146` today matches only `expert_parallel_alltoall_combine`).
- **Endpoint collectives**: `dsv4_embedding_allreduce` and `dsv4_logits_gather` become communication operators with payload, domain and cardinality builders (`families.py:815-823` registers them as plain `OperatorSpec` today).
- **PDD payloads**: the V4 SP payload builder admits `PREFILL` and unified `DECODE` (`families.py:434` accepts `MONOLITHIC` only), and so does the V4 attention component (`attention/deepseek_v4.py:101`).
- **Routed MoE**: priced per EP lane from its rank-local expert-count class: zero, light or heavy, with the alignment the routing algebra gives (`N_d → L_d = ceil(N_d/8) → M = 8ΣL_d`). The classes are measured on one GPU by selecting the local expert population in the producer (`moe/main.py:150` fixes rank 0 today). EP16 has 16 local experts per rank and EP32 has 8.
- **Gating**: the gate projection uses the local `L` grid, under the MoE family's typed contract. `moe_shuffling`, alignment and `moe_sum` are inside the grouped-GEMM envelope and are not charged again.
- **Hash layers**: layers 0-2 keep their architecture identity. No hash or score selection is charged, because the target runtime does not execute it.

### 3.6 Features and coverage keys

Attention features come from the existing V4 workload owner (`attention/deepseek_v4.py:161-192, 215-250`): mode (FULL / PIECEWISE / eager), planner role (`plans_tile_schedule`), scheduled, effective, local and gathered rows, phase (decode, prefill tail, first tail), compressed writes per request at the compression boundaries, and window, compressed and selected key lengths, each defined before and after the query. The model key is (scope, layer variant, planner role, mode, measurement type). The validations of `read_isolated_scope` (`deepseek_v4.py:665-827`: input hash, runtime and backend identity, layout, ownership, typed contracts) move into the new loader before `read_isolated_scope` is deleted.

## 4. Operator inventory and once-only ledger (per forward of one role)

| Phase | Operators | Count per forward | Owner family | Data |
| --- | --- | --- | --- | --- |
| before forward | input and metadata preparation kernels (stage 0 lists them; for example `_compressed_slot_mapping_kernel`); `kv_cache_block_zero` when the batch zeroes cache blocks | 1; zeroing by number of zeroed blocks | V4 operator | G2 |
| attention, first layer | `dsv4_embedding`, `dsv4_embedding_allreduce`, `dsv4_mhc_first_pre` | 1 each | V4 operator; collective | G2; G5 |
| attention, each layer | the attention scopes of its variant (21 CSA, 20 HCA, 2 uncompressed), composed per §3.2 | 43 layers | V4 attention | W6 + q1 + G1 |
| attention, each layer | `dsv4_sp_attention_allgather`, `dsv4_sp_attention_reduce_scatter` | 43 each | collective | measured curves + G5 |
| attention, layers 1-42 | `dsv4_mhc_ffn_post_attn_pre` (fused across layers) | 42 | V4 operator | G2 |
| pre_dispatch | `dsv4_mhc_attn_post_ffn_pre` | 43 | V4 operator | G2 |
| pre_dispatch | shared expert `share_expert_up_proj`, `share_expert_act`, `share_expert_down_proj` | 43 each | linear op | G2 |
| pre_dispatch | `moe_gating_linear`, `moe_gating_routing_topk` (balanced path) | 43 each | MoE | G3 |
| dispatch | `dsv4_ep_activation_allgather`, `dsv4_ep_weights_allgather`, `dsv4_ep_ids_allgather` | 43 each | collective | G5 |
| routed_compute | `moe_grouped_gemm` (envelope includes `moe_shuffling`, alignment, `moe_sum`) | 43 | MoE | accepted rows as candidates + G3 |
| combine | `dsv4_ep_output_reduce_scatter` (unequal vectors by exact pattern) | 43 | collective | G5 |
| post_combine, last layer | `dsv4_mhc_final_post`, `dsv4_sp_final_mhc_allgather`, `dsv4_mhc_head`, `dsv4_final_norm`, `dsv4_lm_head`, `dsv4_logits_gather` | 1 each | V4 operator; collective | G2; G5 |
| between kernels | graph-internal gap; eager host stream; seams (§3.3) | by kernel count / segment | kernel gap; CPU overhead | G4; R2 if needed |

## 5. Stages and order

`stage 0 (CPU) -> stage 1 pilots -> {stage 2 producers and grids; stage 3 simulator work} -> stage 4 validation -> H4`

| Stage | Content | Stop condition |
| --- | --- | --- |
| 0. CPU only | **R1** runtime census of the retained H800 native traces: mode per step class, segments, eager kernels, graph launches, aux-stream fork/join, gaps by segment class, first-tail steps. **C1** coverage matrix: the actual workload vectors of the three cases (batch per DP lane, attention features, SP rows, EP gathered M and rank classes, collective payloads and ordered vectors) against existing data, giving the exact missing coordinates that size G1-G5. **S1** stream graph of each attention variant from the vLLM source | — |
| 1. Pilots | **P1** (H800, 1 GPU): reduced engine with retained kernel intervals; checks A2 and A4, and kernel identity (names, grids) of the reduced TP1 engine against the native TP8 traces. **R1h** (H200, 32 GPUs, P2-32 topology, short profiled window): the H200 runtime census | A4 fails, or kernel identity differs: stop and report to YC |
| 2. Producers and grids | Producer changes (§6, files 13-17), then G1-G5 / G1h-G5h sized by C1 | — |
| 3. Simulator | Files 1-12 (§6). Unit-tested on W6 and q1 rows while the grids run | — |
| 4. Validation | Training, group-held-out accuracy, A1-A12, P0 / P2 replays, diagnostic cross-check | A6 or A10 fails: report with the failing scopes |

Order of cases: H800 P0 first, because its native traces already exist and its GPU jobs are 1-GPU jobs plus one 16-GPU network job; then H200 P2-32 and P2-64 with the same code. If H800 1-GPU jobs cannot start within one day, the order switches and the shortage is reported at once (D-GPU-SHORTAGE-REPORT).

## 6. Frontier file list (H3o-(b))

| # | File | Change | Lines (est.) |
| --- | --- | --- | --- |
| 1 | `frontier/execution_time_predictor/dsv4_predictor.py` (new) | Load and validate V4 attention and V4 operator rows, including the migrated `read_isolated_scope` validations; train per §3.6 key through the caller's `train`; joint coverage admission; typed errors. Same structure as `dsa_predictor.py` | ~320 |
| 2 | `frontier/execution_time_predictor/dsv4_composition.py` (new) | Stream graph per layer variant (§3.2) and its critical path; once-only ledger table (§4) | ~150 |
| 3 | `frontier/execution_time_predictor/deepseek_v4.py` | `local_scope` and the structural path take trained predictions and the composition; block zeroing, endpoints and pre-forward operators priced; CPU stage fields bound (§3.4); delete `read_isolated_scope` after migration | +150 / −270 |
| 4 | `frontier/attention/deepseek_v4.py` | Admit `PREFILL` and unified `DECODE`; expose the features of §3.6 from the existing workload owner | ~60 |
| 5 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | Register the V4 loader, trainer and predictor next to the DSA entries; route V4 eager steps through the existing two-stream path | ~50 |
| 6 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | V4 FFN path (95-115, 1380-1395): shared expert from linear-op models, gating and grouped GEMM from MoE models by rank class | ~80 |
| 7 | `frontier/scheduler/utils/expert_parallel.py` | V4 EP source path through `_predict_layer_with_kernel_gap` | ~20 |
| 8 | `frontier/operators/families.py` | PDD admission for the V4 SP payload; endpoint collectives as communication operators with builders; combine classification | ~70 |
| 9 | `frontier/operators/sequence_parallel_moe.py` | Unequal-combine consumer selects by the combine classification | ~10 |
| 10 | `frontier/execution_time_predictor/kernel_gap.py` | Segment classes (graph-internal, eager, seam) and V4 count naming | ~50 |
| 11 | `frontier/model_architectures.py` | `runtime_limitations` text | ~10 |
| 12 | `docs/general/module-size-exceptions.md` | Record the scoped change to `sklearn_execution_time_predictor.py` (8666 lines) | ~10 |
| 13 | `frontier/profiling/attention/dsa_kernel_trace.py`, `dsa_vllm_runner.py`, `dsv4_attention_rows.py` | Keep kernel intervals (start, end, stream) in raw records; counts in `time_stats.<op>.kernel_count` | ~120 |
| 14 | `frontier/profiling/attention/dsv4_attention_plan.py` | Serving-window cases from C1: decode batches × mixed contexts, 6144 prefill and SP-local rows, first tail alone and with decodes, both planner roles | ~100 |
| 15 | V4 operator producer (new module beside `dsv4_attention_runner.py`) | mHC (first, fused, attn-post, final, head), shared expert, embedding, final norm, LM head, `kv_cache_block_zero`, pre-forward preparation, with typed operator contracts | ~300 |
| 16 | `frontier/profiling/moe/main.py` and wrapper | Rank-class selection of the local expert population; V4 gate on the `L` grid; balanced path signature | ~100 |
| 17 | `frontier/profiling/kernel_gap/main.py` | V4 graph-replay sampling on the reduced engine with segment classification; per-segment first kernels included at seams | ~100 |
| 18 | `tests/analysis/v4_runtime_census.py` (new) | Stage-0 R1 reducer over native traces, streaming and memory-limited | ~200 |
| 19 | Tests | `test_dsv4_predictor.py`, `test_dsv4_composition.py` (new); V4 dispatch, payload, unequal binding, EP gap, ledger sentinel and collision tests; migrated `read_isolated_scope` cases; V4 cases in the fidelity regression | ~600 |

Total: about 19 files, about 2,500 lines added and 270 deleted. This is far above the D-H4A-CHECKPOINT limit of 5 files and about 200 lines (decision D3).

- **No new config field.** V4 attention rows use the attention input; V4 operator rows use `linear_op.csv` with typed contracts; gating and routed MoE use `moe.csv`; collectives use the measured backend's `summary.csv`; gaps use `kernel_gap.csv`; host costs use the CPU-overhead inputs and `skip_cpu_overhead_modeling`. If implementation finds a binding that these cannot express, it stops and asks YC.
- **Module sizes.** All touched modules except `sklearn_execution_time_predictor.py` stay below 2,000 lines; that module gets only registration and routing lines.

## 7. GPU jobs

All jobs go through the StepMind Python `RJobBackend` with `i-fengyicheng` auth, via `/ds4-exec`. H800 uses codesign; H200 uses step_main, non-preemptible, with running plus queued at most 64 GPUs. Every run with 32 or more GPUs uses soft gates only: validators log and report, never stop training or profiling. H800 and H200 evidence never qualify each other. The figures are planning caps; C1 and the pilots replace them with exact grids before stage 2 starts.

| Job | GPUs | Content | Cap (GPU-hours) |
| --- | --- | --- | --- |
| P1 (H800) | 1 | Composition and identity pilot (stage 1) | 1 |
| G1 (H800) | 1 | Attention serving window from C1; KERNEL_ONLY with intervals, CUDA_EVENT for validation | 3 |
| G2 (H800) | 1 | V4 operators of §4 over decode rows 1-64, SP prefill rows, 6144 | 1.5 |
| G3 (H800) | 1 | MoE rank classes for EP16 and gate `L` grid | 2 |
| G4 (H800) | 1 | V4 graph-replay `kernel_gap.csv` | 0.5 |
| G5 (H800) | 16 | Small payload points (TP8, EP16) and the ordered unequal vectors from C1, with `vllm_forward_collectives.py` | 12 |
| R2 (H800, only if R1 shows eager host time is material) | 32 | Native P0 with the CPU probe, for `forward_launch` rows | 32 |
| R1h (H200) | 32 | Native P2-32 profiled window, runtime census | 48 |
| G1h-G4h (H200) | 1 each | As G1-G4; decode batch up to 40; rank classes for EP16 and EP32 (8 local experts); M grid includes M6168 and M24576 for DP4 | 7 |
| G5h (H200) | 16, then 32 | Uniform curves and unequal vectors for TP8, EP16 and EP32 | 36 |
| R2h (H200, only if needed) | 32 | As R2 | 32 |

Caps: H800 20 GPU-hours, plus 32 if R2 runs; H200 91, plus 32 if R2h runs.

## 8. Time estimate (elapsed working days)

| Stage | Days |
| --- | --- |
| 0. R1, C1, S1 (CPU) | 2 |
| 1. Pilots P1 and R1h (queue-bound) | 1-2 |
| 2. Producers (files 13-17), then G1-G5 / G1h-G5h | 3 (grids run in parallel with stage 3) |
| 3. Simulator files 1-12 and tests | 4 |
| 4. Training, A1-A12, replays, cross-check | 2 |
| **Total** | **about 12-15**; the H800 P0 milestone at about 9 |

Main schedule risks: GPU queues, image or runtime failures (earlier V4 profiling jobs needed several attempts), and a failed pilot.

## 9. Decisions for YC

| # | Decision | Options | Recommendation |
| --- | --- | --- | --- |
| D1 | Attention layer cost | (a) Option A: per-scope costs and a critical path over the source stream graph, checked by P1. (b) A measured layer composite (elapsed of the whole layer, shape-aware) with scopes for attribution only; this changes approved Option A | (a) |
| D2 | Host time of eager steps and eager segments | (a) The existing CPU-overhead `forward_launch` rows (measured host launch time per forward, keyed by batch features), used only for the host stream; the device stream stays operator-composed. (b) A per-kernel launch cost times the kernel count, which is new modeling. (c) Leave host time out and state the limit | Decide after R1: if eager host time is not material, (c) with an explicit zero; if it is, (a). (a) is a measured per-forward host time, so YC should confirm that it is compatible with D-NO-STEP-COST-SHORTCUT |
| D3 | Size | About 19 files and 2,500 lines, above D-H4A-CHECKPOINT | Approve, staged as §5 |
| D4 | GPU budget | Caps of §7, including the 32-GPU H200 run R1h, and R2 / R2h only if needed | Approve the caps; exact grids are reported after C1 |
| D5 | Reduced TP1 engine for per-operator device time | (a) Reduced TP1 with per-rank TP8 shapes, qualified by kernel identity against the native TP8 traces (P1). (b) Per-operator profiling inside full native serving | (a) |
| D6 | Order | H800 P0 first, or H200 first | H800 P0 first; switch if H800 jobs cannot start within one day |

## 10. Risks

1. **First-tail step.** The native first tail alone takes 61-69 ms against 20-24 ms for decode-only. Its mode and boundaries are unknown until R1; it is not assumed to be host-bound.
2. **Pilot failure.** If the critical-path composition misses the joined elapsed (A4), the route stops for a decision; no factor is added.
3. **Single layer versus 43 layers.** Rows sample one layer. L2 and stream effects across 43 layers are not proven additive; the cross-check shows their size and nothing corrects them.
4. **Profiler overhead.** R1 and R1h host times are upper bounds. Host costs come only from unprofiled measurements (R2 / R2h).
5. **Routing.** Balanced routing and dummy weights only; real router imbalance belongs to ticket 14.

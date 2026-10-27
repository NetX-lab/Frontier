# H3o-(a): operator-level DSV4 costs — scope, files, GPU jobs and time (proposal for YC's review)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | YC's decisions (requirements item 33): D1 (a), the growth since D3 / D4 confirmed, D7 (a); round 3 approved. The FULL descriptor bound of §3.7 is stated from the source. |
| 2026-10-26 | Revision 2c after Codex review round 2 (`codex-review-4678dfdc-…/review.md`, verdict `changes_required`, F009-F015). Agreed changes: native `FULL_AND_PIECEWISE` local graph-mode classifier and the full source synchronization rule (the first tail runs PIECEWISE natively; the census claims of revision 2b are corrected); V2-runner dummy workload; DP rendezvous before post-DP preparation and first-layer device work; composed attention duration consumed by the events, with the gap charged once; conditional stream graph; per-stream gap; a FULL / PIECEWISE replay route before pilot P1h; A4 per group; A13 (c) from Frontier batches and A13 (d) timing; files 24-28; time and caps updated. Per-finding positions are in `review.md`. |
| 2026-10-26 | Revision 2b after YC's decisions (requirements items 29-31): H200 first (D6); D2-D5 decided as recommended; D1 open with (a) as the working choice. Routing identity with native `balanced_topk` (D-V4-ROUTING-IDENTITY): the routing algorithm, expert counts and per-rank counts already match; the routed row count does not in 1.7-5.4% of decode forwards because Frontier does not DP-synchronize real V4 lanes. Added §3.7 (DP-synchronized forward descriptor for every V4 lane), gate A13, files 20-23 and the deferred non-V4 gap. |
| 2026-10-26 | Revision 2a: the retained H200 native profiled runs (P2-32, 32 traces; P2-64, 64 traces) show the same FULL-graph decode structure, so the runtime census R1 covers all three cases on CPU and the 32-GPU H200 job R1h is removed; H200 pilot P1h (1 GPU) added; caps and time updated. Codex round 2 (`55d630bb-…`) failed with HTTP 429 before any report. |
| 2026-10-26 | Revision 2 after Codex review round 1 (`codex-review-9c8b2435-…/review.md`, verdict `changes_required`, P1 × 5, P2 × 3). Agreed changes: per-operator composition with a source-defined stream graph instead of a layer union; runtime-mode census and pilots before the grids; full once-only ledger with PDD admission, endpoint collectives, block zeroing and pre-forward preparation; unequal-combine binding; EP rank classes; `balanced_topk` scope; workload-owner features and joint coverage; structural acceptance gates; recomputed files, GPU jobs and time; decisions for YC. Per-finding positions are in `review.md`. |
| 2026-10-26 | Created: CPU-only proposal under D-OPERATOR-LEVEL-ROUTE (requirements item 27). Nothing in it is implemented or submitted. |

## 1. Goal, scope and acceptance

Goal: Frontier prices every DSV4 forward of the P2-32 / P2-64 (H200, mainline) and P0 (H800, after H200) cases by composing operator costs for the batch being priced. Device costs come from operator profiling CSVs through trained predictors and from the measured communication backend. The step-cost table (`dsv4_step_cost.json`) is only a diagnostic cross-check of the composed time (D-NO-STEP-COST-SHORTCUT, ADR 0002). No cost, factor or residual is derived from it, or from any other measured whole step.

Runtime scope, as the three cases are configured: PD disaggregation with `PREFILL` and unified `DECODE` roles; `balanced_topk` routing on all 43 layers (every router is `RoutingSimulatorRouter`, `native_config.json:42`); speculative decoding / MTP off; prefix caching off. Real score or hash routing (ticket 14) and PD-AF are out of scope. A run whose runtime path differs from this scope stops with a typed error. Frontier's routing must be identical to the native one (D-V4-ROUTING-IDENTITY): `balanced` routing on every layer, the same per-expert and per-EP-rank counts, and the same graph mode and routed row count per forward after vLLM's DP synchronization (§3.7). The native runner of all three cases is the V2 `vllm.v1.worker.gpu.model_runner.GPUModelRunner` with configured graph mode `FULL_AND_PIECEWISE` (counter inventories), and pipeline parallelism is 1; V4 with PP > 1 stays outside the scope.

H3o-(b) is complete when all of the following hold:

| # | Gate | Check |
| --- | --- | --- |
| A1 | No dummy values, no default step table | A non-dummy V4 run reads no `predictor._dummy_execution_time` (a test makes any read raise) and binds no step-cost table unless a diagnostic driver selects one explicitly |
| A2 | Kernel ownership | In every pilot and grid window, each kernel belongs to exactly one operator scope; unowned kernels are listed and fail the window; an outer scope and its child scopes are never both charged |
| A3 | Once-only ledger | With sentinel costs, each operator is charged the number of times in §4 per forward (for example `dsv4_mhc_first_pre` × 1, `dsv4_mhc_ffn_post_attn_pre` × 42, `dsv4_mhc_attn_post_ffn_pre` × 43) and in its owning phase |
| A4 | Composition check | Pilot P1h (H200), and P1 on H800 later, on FULL, PIECEWISE and eager replays of the route in §5 stage 1a: the composed time of one attention layer (critical path over the conditional stream graph of §3.2, gaps of §3.3 included) against the stream-joined elapsed of the same call. First with the measured scope costs of that call, so the composition is checked apart from predictor error (A6). Evaluated per group (mode, topology, layer variant, planner role, workload group), never pooled: median ≤ 5%, p90 ≤ 10% in every group. An empirical stop gate, not a proof: a failure stops the route and is reported; no factor is introduced |
| A5 | Typed coverage | Coverage is admitted jointly per workload group (mode, planner role, phase, layer variant, rows), not by independent min/max bounds. Outside it, a typed `DeepSeekV4CoverageError` names the operator and the missing coordinate. Collision tests: planner versus reuse layer, equal aggregates with different compressed writes, swapped active DP owner, unmeasured ordered SP vector |
| A6 | Predictor accuracy | Held out by complete workload group: per scope, median relative error ≤ 5% and p90 ≤ 10%. Scopes that are legally zero or no-op are checked for exact zero instead (no relative error with a zero denominator) |
| A7 | EP rank admission | Every EP lane of the target replays binds to a measured rank class (§3.5); the per-rank costs and their critical-path maximum are checked, not only rank 0 |
| A8 | Collective coverage | Every collective of the replays matches a measured (operation, group, cardinality, ordered local rows, bytes) point or a uniform curve inside its range; an unequal combine never falls back to the scalar curve |
| A9 | Cache identity | Cold-cache and warm-cache runs give identical predictions and artifact identities |
| A10 | End-to-end runs | P0, P2-32 and P2-64 replay end to end on the operator path, each on its own hardware's data only |
| A11 | Diagnostic cross-check reported, not a gate | Composed versus native time per step class (decode-only by batch size, first tail alone, first tail with decodes, prefill) and per phase; and the census of graph mode, synchronized widths and dummy counts per step class of each replay against the native counters (entry 18). It never feeds a cost |
| A12 | Other paths unchanged | GLM, co-location, PDD and PD-AF keep identical results: 77-case fidelity regression, dummy smoke matrix and unit tests pass |
| A13 | Routing and forward identity with native vLLM | (a) Projection check: the source synchronization rule of §3.7, applied to the recorded local sizes, desired modes, uniform counts and max query lengths, reproduces the recorded descriptor and `num_tokens_across_dp` of every serving and dummy forward of every PREFILL and DECODE node of P2-32, P2-64 and P0. (b) Native `balanced_topk` and native `determine_expert_map`, executed from the pinned vLLM source on the synchronized SP-local sizes, give the same per-expert and per-EP-rank counts as Frontier for DP2 / EP16 and DP4 / EP32 widths. (c) From Frontier batches and config through the local classifier, the synchronization, the actual V4 workload builders and the pricing consumer: first tail alone, first tail with steady decodes, steady decodes only, dummy lanes beside each of them, all-dummy forwards and PREFILL forwards give the native mode, per-lane width and request padding, the routed row count `M`, and dummy workloads with no request metrics or KV credit. The tuple of counter row 428 (local `[12, 8]`, one first tail) gives PIECEWISE, `[16, 16]`, `M = 32`; steady decodes alone stay FULL. (d) Timing: with unequal lane readiness, preparation and attention durations, every lane's first-layer device work starts after the DP rendezvous (the sentinel of F009 gives at least 110 ms), and the stage components, launch accounting and completion agree, for real and dummy lanes, FULL and eager, PREFILL and DECODE |

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
| Native P2 traces, H200 | `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces/` (32 ranks, case `dsv4-h200-p2-profile-fixed128-qps1`) and `dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces/` (64 ranks, case `dsv4-h200-p2-64-profile-fixed128-qps1`) | Torch-profiler traces of native P2-32 and P2-64 deployments | Runtime-mode census of H200 (stage 0). Decode dp0 rank 0 of both runs: FULL graph replays (249 and 301 `cudaGraphLaunch`, annotation `model_dispatch_run_fullgraph`) and eager pre-forward kernels on stream 23 (9,937 and 10,853). Same host-time limit as the H800 traces |
| Native P0 traces, H800 | `/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04/torch_traces/` | Torch-profiler traces of all 32 ranks of a native P0 deployment (physical TP8 / DP2 / EP16, case `dsv4-h800-p0-profile-fixed128-qps1`), 512 recorded steps per rank, with kernels, streams, runtime calls, graph launches and Frontier annotations | Runtime-mode census of H800 (stage 0). One CPU look at decode dp1 rank 0: the 61 useful decode steps ran as FULL graph replays (150 `cudaGraphLaunch`, about 915 kernels per launch, annotation `model_dispatch_run_fullgraph`), and 5,315 eager kernels ran on a separate stream (input and metadata preparation, for example `_compressed_slot_mapping_kernel`). Profiler overhead inflates host times, so host magnitudes from these traces are upper bounds |
| Native serving counters, all three cases | `nodes/*/serving/rank*.json` under the three run directories (`evidence/routing/dp_sync_rule_check.txt`) | Per forward: local sizes, desired modes, uniform counts, max query lengths, final descriptor and synchronized widths. Runner V2 `GPUModelRunner`, configured mode `FULL_AND_PIECEWISE`, FULL descriptors for 1-128 tokens, PIECEWISE sizes up to 256 | Mode per step class (first node of each role, rank 0; `evidence/routing/mode_census.txt`): decode serving FULL 16,469-23,114 and PIECEWISE 293-383 (first tails 79-204, steady lanes beside a peer's first tail 167-214); decode dummies mostly FULL; prefill serving all NONE (109-217); prefill dummies FULL 8,408-9,455 |

The existing CPU-overhead mechanism prices an eager step as two streams: host `forward_launch` against device KERNEL_ONLY operators (`sklearn_execution_time_predictor.py:992-1006, 7125-7140`; enabled by `measurement_input_paths.py:160` when `skip_cpu_overhead_modeling` is false and the CPU-overhead table has `forward_launch`). The unequal-combine producer `frontier/profiling/collectives/vllm_forward_collectives.py` and consumer `measured_cc_backend.py:132-151, 217-229` exist.

## 3. Cost contract

### 3.1 Units of cost

The unit is one operator scope, or one native fused scope that vLLM executes as a unit (for example `dsv4_compressor_fused_compress_store`). Each scope's device cost is the sum of its owned kernels (KERNEL_ONLY) for the batch being priced. No layer union, layer elapsed or other composite is a cost, and no layer total is spread over child scopes. CUDA_EVENT rows are used for validation and for scopes whose events enclose a single kernel stream, never as a residual assigned to CPU.

### 3.2 Device stream composition

- **Across phases**: serial, in the phase order of `timing_operator_times` (`deepseek_v4.py:17-63`), with the shared expert in `pre_dispatch` as the code has it.
- **Inside an attention layer**: the compressor and indexer scopes can run on auxiliary streams. The layer time is the critical path through a fork/join graph of scopes. The graph is taken from the vLLM V4 attention source (stream switches and event waits) and is conditional, not one graph per variant: the outer input projections fork only when the layer's rows are at most `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` (`attention.py:515-573`; default 1024, `envs.py:2016-2017`) and run serially above it; nested branches run serially under a Breakable (PIECEWISE) capture (`multi_stream_utils.py:45-61`); the indexer branch depends on the short-context and capture state (`attention.py:906-930, 952-958`). The graph is selected per call from the rows, the mode and capture kind, and that state. The threshold is not a literal in Frontier: it comes from the runtime identity the producer records with the attention rows (`dsv4_attention_runner.py:356`), and the loader rejects rows whose value differs from the native case's. Node weights are the predicted scope costs plus the gaps of §3.3, added before the longest path is taken. S1 and pilot P1h (P1 on H800) cover both sides of the threshold and every reachable capture branch. This keeps approved Option A (`task_2026-10-23_v4_remaining_measurements/proposal.md:23-55`): per-operator costs plus an explicit dependency account.
- **Consumer**: the composition returns two things from the same predicted costs: the composed elapsed of the attention layer and each scope's start offset on that schedule. `op_times` keeps every canonical scope cost unchanged, for the ledger (A3) and attribution; no scope is dropped and no composite is spread over scopes. The V4 branch of `ExecutionTime._get_attention_layer_execution_time` (`execution_time.py:1116-1125`, today a serial sum without the attention gap) returns the composed elapsed, so `get_single_layer_attention_scope_time`, the event delays (`collective_timing.py:29-32`, `replica_stage_schedule_event.py:263, 558`) and the block time read it, and the gap is charged once, inside it. The trace (`metrics_store.py:1157-1165`) places each scope at its start offset.
- **Shared expert versus routed path**: serial. The native traces measure any overlap, as a diagnostic. If it is material, a later change models it explicitly.

### 3.3 Runtime modes and the time between kernels

The native counters already give the mode of each step class (§2): steady decodes FULL; a forward with a first tail on any lane PIECEWISE on every lane; prefill NONE (eager); dummies FULL, PIECEWISE or NONE after their peers. R1 adds the segment structure of each mode from the traces. Each mode is priced as follows:

| Segment | Price |
| --- | --- |
| Graph-internal | Device costs plus a per-kernel gap on every path of the conditional stream graph, added before the longest path is taken. The V4 gap is measured per stream on graph replays: the idle time before a kernel after the later of the previous kernel on the same stream and the kernels it waits for, so a gap hidden behind another branch still counts on its own branch, and a dependency wait is never counted as a gap. One value per (mode, TP) in a V4 `kernel_gap.csv` with the existing schema; the existing reducer's global-union idle time (`kernel_gap/main.py:107-115`), which other models use, is unchanged. This one value per stream is an approximation; A4 checks it per group, and a failing group stops the route. Kernel counts use the `time_stats.<op>.kernel_count` naming that `kernel_gap.py:33-38` reads |
| Eager segment and eager step | The existing two-stream model: host `forward_launch` against the device stream. Decision D2 in §9 |
| Seam between segments, and pre-forward preparation | Measured as named operators or seam terms only if stage 0 shows them as material; each is charged once, in one owner |

The V4 EP path (`scheduler/utils/expert_parallel.py:129-139`) is rewired through `_predict_layer_with_kernel_gap`, so FFN phases get the same gap treatment as the attention phase.

### 3.4 Host cost

The V4 stage-owned CPU fields (`deepseek_v4.py:360-375`, currently dummy) are bound to the existing CPU-overhead inputs. With `skip_cpu_overhead_modeling=True`, the policy is an explicit zero and the report says so. With it false, no dummy value remains. CPU cost is never derived as native step minus operator sum.

In the V2 runner the DP all-reduce sits inside the probe's `prepare_inputs_e2e` interval: request-state updates come before it and input and attention-metadata preparation after it (`model_runner.py:1417-1546`). If R2h / R2 run, the probe reducer splits that interval at the recorded all-reduce (`dp_allreduce_start`, `dp_allreduce_end`, already in the probe logs) into a pre-DP and a post-DP share; the DP wait stays excluded, as it is today (`vllm_cpu_probe.py:23-55, 319-345`). Before R2h, a CPU check confirms that the probe instrumentation binds to the V2 runner's methods; the probe text that places the all-reduce at forward-context entry for eager forwards describes the older runner.

### 3.5 Collectives and MoE

- **SP and EP collectives**: `MeasuredCCBackend`. The V4 combine `dsv4_ep_output_reduce_scatter` is bound to the ordered-SP-vector consumer, by classifying the combine operators in the communication registry instead of comparing one name (`operators/sequence_parallel_moe.py:133-146` today matches only `expert_parallel_alltoall_combine`).
- **Endpoint collectives**: `dsv4_embedding_allreduce` and `dsv4_logits_gather` become communication operators with payload, domain and cardinality builders (`families.py:815-823` registers them as plain `OperatorSpec` today).
- **PDD payloads**: the V4 SP payload builder admits `PREFILL` and unified `DECODE` (`families.py:434` accepts `MONOLITHIC` only), and so does the V4 attention component (`attention/deepseek_v4.py:101`).
- **Routed MoE**: priced per EP lane from its rank-local expert-count class: zero, light or heavy, with the alignment the routing algebra gives (`N_d → L_d = ceil(N_d/8) → M = 8ΣL_d`). The classes are measured on one GPU by selecting the local expert population in the producer (`moe/main.py:150` fixes rank 0 today). EP16 has 16 local experts per rank and EP32 has 8.
- **Gating**: the gate projection uses the local `L` grid, under the MoE family's typed contract. `moe_shuffling`, alignment and `moe_sum` are inside the grouped-GEMM envelope and are not charged again.
- **Hash layers**: layers 0-2 keep their architecture identity. No hash or score selection is charged, because the target runtime does not execute it.

### 3.6 Features and coverage keys

Attention features come from the existing V4 workload owner (`attention/deepseek_v4.py:161-192, 215-250`): mode (FULL / PIECEWISE / eager), planner role (`plans_tile_schedule`), scheduled, effective, local and gathered rows, phase (decode, prefill tail, first tail), compressed writes per request at the compression boundaries, and window, compressed and selected key lengths, each defined before and after the query. The model key is (scope, layer variant, planner role, mode, measurement type). The validations of `read_isolated_scope` (`deepseek_v4.py:665-827`: input hash, runtime and backend identity, layout, ownership, typed contracts) move into the new loader before `read_isolated_scope` is deleted.

### 3.7 Routing identity and the DP-synchronized forward descriptor

What already matches native vLLM (checked 2026-10-26, evidence in `evidence/routing/`):

- **Algorithm.** Native `balanced_topk` assigns row `i` of EP rank `r` to experts `((offset_r + i) × k + j) mod E`, with `offset_r` the sum of the SP-local sizes of the ranks before `r`, and weights `1/k`. Over all ranks this gives expert `e` the count `floor(M × k / E) + [e < (M × k) mod E]`. Frontier's `balanced` ratios with Hamilton apportionment and expert-ID tie order (`moe_ep_workload.py:773-808`) give the same counts.
- **Placement.** Native linear placement (`expert_map_manager.py:determine_expert_map`, no EPLB configured) and Frontier's `build_contiguous_expert_ownership` assign the same contiguous expert blocks to each EP rank.
- **Test.** `tests/unit/test_balanced_routing.py` executes native `balanced_topk`, `_compute_sp_num_tokens` / `DPMetadata` and `determine_expert_map` from the pinned source and compares per-expert and per-EP-rank counts with Frontier. With the DP2 / EP16 and DP4 / EP32 widths added: 102 passed.

What does not match: the graph mode and the routed row count `M` of some forwards. The V2 runner (`model_runner.py:1417-1546`) first updates request state, then picks each lane's desired descriptor and synchronizes it across the DP group (`dispatch_cg_and_sync_dp`; `v1/worker/gpu/dp_utils.py:16-95`, `sync_cudagraph_and_dp_padding`), and only then prepares inputs and attention metadata and runs the model. The source rule:

- **Local descriptor.** Under `FULL_AND_PIECEWISE`, a lane's batch dispatches to the first compatible captured descriptor (`cudagraph_utils.py:75-100, 382-410`). FULL descriptors exist for uniform decodes only: every capture size up to `max_num_seqs` × decode query length and the largest capture size, with uniform count 1 and request padding (`cudagraph_utils.py:240-262`; 1-128 tokens in the three cases, which do not use the varlen decode descriptors of lines 223-235); a batch with any prefill row, including a one-token first tail, has uniform count `None` (`v1/worker/utils.py:601-607`) and gets PIECEWISE (sizes up to 256); a larger batch runs eager (NONE).
- **Synchronization.** The all-reduce exchanges local tokens, desired mode, uniform count and max query length. If any lane wants NONE, every lane runs NONE at its own width. Otherwise the descriptor is dispatched again on the synchronized values: the widest lane's tokens, the uniform count if all lanes agree and `None` otherwise, and the maximum of the max query lengths. Every lane is padded to the result. PIECEWISE pads tokens, not requests (`dp_utils.py:81-93`).

Frontier differs in three places:

1. **Local mode.** `full_decode_only` gives a first tail NONE (`vllm_v1_iteration_policy.py:98-107`); `piecewise` gives steady decodes PIECEWISE. Neither is `FULL_AND_PIECEWISE`. For the PREFILL role no graph metadata is built at all (`vllm_v1_iteration_policy.py:47-48`), while native prefill dummies run FULL.
2. **Synchronization of real lanes.** Frontier applies the minimum-mode, widest-width rule only to dummy forwards (`dp_dummy_forward.py:105-136`). Real V4 lanes keep their own mode and capture size (`vllm_v1_engine_replica_scheduler.py:468-470`), and `source_physical_rows` sums those lane-local widths (`execution_time_predictor/deepseek_v4.py:535-547`).
3. **Dummy workload.** A native DP dummy is `_dummy_run(1, uniform_decode=True)` executed as a real `execute_model` on the synchronized descriptor (`gpu_worker.py:1210-1212`, `model_runner.py:647-701, 1500-1546`). Its input batch has the descriptor's rows and requests, each request's sequence length equals its query length, and every row is padding (`input_batch.py:116-206`). It has dummy attention metadata, and under FULL the capture metadata is staged again. Frontier's V4 dummy is one row, has no metadata and is eager-only (`attention/deepseek_v4.py:268-276`); a dummy with graph metadata raises `NotImplementedError`.

The retained native counters show the extent (first node of each role, rank 0, `evidence/routing/mode_census.txt`; all nodes, `evidence/routing/review_counter_checks.txt`):

| Case | Decode forwards whose `M` differs under lane-local widths (native local modes) | Decode counter rows with different desired modes (all reach the uniform-disagreement branch) | Decode serving forwards that run PIECEWISE natively (Frontier `full_decode_only`: NONE for first tails, FULL for their peers) | Prefill dummies, FULL |
| --- | --- | --- | --- | --- |
| P2-32 (H200) | 842 / 24,134 (3.5%) | 274 rows, 688 forward observations | 357 (first tails 190, peers 167) | 9,079 |
| P2-64 (H200) | 1,029 / 19,014 (5.4%) | 803 rows, 1,224 forward observations | 293 (79, 214) | 8,408 |
| P0 (H800) | 398 / 23,622 (1.7%) | 347 rows, 750 forward observations | 383 (204, 179) | 9,455 |

Revision 2b said the uniform-disagreement branch is not reached and no forward has a mode difference. Both statements were wrong: `dp_sync_census.py` labelled every difference not under NONE as a FULL width difference. Prefill nodes also have desired-mode differences (14, 14 and 53 rows, a serving prefill beside a FULL dummy, synchronized to NONE). Observation counts repeat across the DP nodes that record the same forward, so they are not unique global forwards.

The projection check is unchanged: applied to the recorded local sizes and desired modes, the rule reproduces the recorded descriptor mode and `num_tokens_across_dp` of all 3,588 serving and dummy counter rows (245,088 forward observations) on every PREFILL and DECODE node of the three cases, with no mismatch (`evidence/routing/dp_sync_rule_check.txt`). It shows the synchronization rule. It does not check the Frontier local classifier that produces the desired modes; A13 (c) does.

Change (V4 only; files 20-25):

1. **Local classifier.** A new value `full_and_piecewise` of the existing `decode_cuda_graph_mode` field (`config.py:172-174, 621`) gives the native local descriptor above. FULL goes to uniform decodes up to the FULL bound of the source (capture sizes up to `max_num_seqs` × decode query length and the largest capture size), PIECEWISE to other batches up to the largest capture size, and NONE above that. PIECEWISE pads tokens only. A first tail is a prefill row. Under this value the PREFILL role also builds graph metadata, so prefill dummies get FULL. In H3o the value is accepted for V4 only; other models raise a typed configuration error until qualified. The V4 replay configurations of P0, P2-32 and P2-64 select it.
2. **Synchronized descriptor.** One function over the lanes of a forward implements the source rule above, including the uniform-count and max-query redispatch. It replaces the body of `resolve_dummy_forward_inputs`, and applies to every lane, real and dummy, in PREFILL, unified DECODE and MONOLITHIC roles. `source_physical_rows` and the per-lane SP widths of `predict_source_execution` read the synchronized widths through `compute_rows`; their formulas do not change.
3. **DP rendezvous before the first layer.** The first pre_moe room of a stage forward becomes the DP rendezvous. Each lane, real or dummy, enters it at its pre-DP ready time: stage start plus its pre-DP host share (zero under `skip_cpu_overhead_modeling=True`). The rendezvous time is the latest ready time. Only then is each lane's descriptor fixed and its first layer priced, and the room places, in this order from the rendezvous: the lane's post-DP host preparation, its pre-forward preparation kernels (sized by the synchronized descriptor, §4), and its first-layer device work. Each lane's arrival is the end of its first-layer attention on that schedule. The room's collective time is the latest arrival, so the stage components, launch start and completion all derive from one timeline. For V4 forwards this replaces the launch-start alignment of `sync_entry.py:113-131`, which today moves only the host launch share and leaves an early lane's device work before the barrier (F009: an early lane with 10 ms attention and a late lane ready at 100 ms gives 101 ms instead of at least 110 ms). Preparation stays lane-local in cost and ownership.
4. **Dummy workload.** The V4 dummy builders follow the V2 dummy: the descriptor's rows and requests (FULL pads requests to the descriptor, PIECEWISE and NONE keep one request), sequence length equal to query length, padding rows, dummy attention metadata, FULL re-staging. They are priced with the scopes that the native dummy executes (qualified in G1h / G1 on dummy coordinates from C1). The scheduler's request-free identity stays: a dummy produces no request metrics and no KV-cache credit. A dummy coordinate without coverage raises the typed coverage error.
5. Non-V4 models keep their current path (A12). Their own gap is deferred (§10). How often Frontier issues dummies, for example on PREFILL engines waiting for NIXL reads, is the existing lifecycle model and does not change here.

## 4. Operator inventory and once-only ledger (per forward of one role)

| Phase | Operators | Count per forward | Owner family | Data |
| --- | --- | --- | --- | --- |
| before forward, after the DP rendezvous | input and metadata preparation kernels (stage 0 lists them; for example `_compressed_slot_mapping_kernel`), sized by the synchronized descriptor (§3.7); `kv_cache_block_zero` when the batch zeroes cache blocks | 1; zeroing by number of zeroed blocks | V4 operator | G2 |
| DP dummy forward | the same phases as a real forward on the V2 dummy workload of §3.7 (descriptor rows and requests, padding rows, dummy attention metadata; FULL re-stages capture metadata); no request metrics, no KV credit | 1 per dummy forward | as the real phases | G1h / G1, G2 on dummy coordinates from C1 |
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

`stage 0 (CPU) -> stage 1a replay route -> stage 1 pilots -> {stage 2 producers and grids; stage 3 simulator work} -> stage 4 validation -> H4`; stage 3's files 20-25, 27 and 28 can start in parallel with stage 0.

| Stage | Content | Stop condition |
| --- | --- | --- |
| 0. CPU only | **R0** routing identity, A13 (a) and (b): done 2026-10-26, no mismatch (§3.7). **R1** runtime census of the retained native traces of all three cases (H200 P2-32, H200 P2-64, H800 P0). The serving counters already answer the mode per step class (§2; the revision 2b census is corrected in §3.7); the traces add segments, eager kernels, graph launches, aux-stream fork/join, gaps by segment class and the first-tail steps. **C1** coverage matrix: the actual workload vectors of the three cases (batch per DP lane, attention features, SP rows, EP gathered M and rank classes, collective payloads and ordered vectors) against existing data, giving the exact missing coordinates that size G1h-G5h and G1-G5. **S1** stream graph of each attention variant from the vLLM source | — |
| 1a. Replay route (CPU, then within P1h) | Timing today forces eager (`dsa_vllm_runner.py:597-604`) and the plan rejects graph scope timing (`dsa_plan.py:787-794`). The route: an eager pass with the scope hooks gives each kernel signature (name, grid, block, stream) its owning scope; the same call is then captured and replayed as FULL or PIECEWISE, and its replay kernel intervals are matched one to one to those signatures, with the joined boundary (first kernel start to last kernel end over all streams) as the replay elapsed. A kernel without exactly one match fails the window (A2) | Route cannot match one to one: stop and report to YC |
| 1. Pilots | **P1h** (H200, 1 GPU): reduced engine with retained kernel intervals, FULL, PIECEWISE and eager replays through route 1a, both sides of the multi-stream threshold; checks A2 and A4 per group, and kernel identity (names, grids) of the reduced TP1 engine against the H200 native TP8 traces. **P1** (H800, 1 GPU, after the H200 milestone): the same checks against the H800 native traces | A4 fails in any group, or kernel identity differs: stop and report to YC |
| 2. Producers and grids | Producer changes (§6, files 13-17), then G1h-G5h sized by C1; G1-G5 after the H200 milestone | — |
| 3. Simulator | Files 20-25, 27 and 28 first (§3.7 and the consumer of §3.2; they need neither D1 nor GPU data), then files 1-12 (§6). Unit-tested on W6 and q1 rows while the grids run | A13 (c) or (d) fails: fix before the replays |
| 4. Validation | Training, group-held-out accuracy, A1-A13, P2-32 / P2-64 replays, then P0; diagnostic cross-check | A6 or A10 fails: report with the failing scopes |

Order of cases (D6, decided): H200 P2-32 and P2-64 first, as the mainline; H800 P0 testing and calibration follow with the same code. H200 has no collective curves, so G5h is on the critical path of the H200 milestone. If an H200 job cannot start within one day, the shortage is reported at once (D-GPU-SHORTAGE-REPORT); H800 evidence never substitutes for H200 evidence, and CPU and simulator work continues meanwhile.

## 6. Frontier file list (H3o-(b))

| # | File | Change | Lines (est.) |
| --- | --- | --- | --- |
| 1 | `frontier/execution_time_predictor/dsv4_predictor.py` (new) | Load and validate V4 attention and V4 operator rows, including the migrated `read_isolated_scope` validations; train per §3.6 key through the caller's `train`; joint coverage admission; typed errors. Same structure as `dsa_predictor.py` | ~320 |
| 2 | `frontier/execution_time_predictor/dsv4_composition.py` (new) | Conditional stream graph per layer variant (§3.2), selected per call by rows against the recorded threshold, capture kind and the indexer's short-context state; per-path gaps; critical path returning the composed elapsed and per-scope start offsets; once-only ledger table (§4) | ~200 |
| 3 | `frontier/execution_time_predictor/deepseek_v4.py` | `local_scope` and the structural path take trained predictions and the composition; block zeroing, endpoints and pre-forward operators priced at the synchronized descriptor; V2 dummy forwards priced on their dummy coordinates; CPU stage fields bound (§3.4); delete `read_isolated_scope` after migration | +170 / −270 |
| 4 | `frontier/attention/deepseek_v4.py` | Admit `PREFILL` and unified `DECODE`; expose the features of §3.6 from the existing workload owner; the V2 dummy workload of §3.7 item 4 replaces the one-row no-metadata contract (lines 268-276) | +100 / −10 |
| 5 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | Register the V4 loader, trainer and predictor next to the DSA entries; route V4 eager steps through the existing two-stream path | ~50 |
| 6 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | V4 FFN path (95-115, 1380-1395): shared expert from linear-op models, gating and grouped GEMM from MoE models by rank class | ~80 |
| 7 | `frontier/scheduler/utils/expert_parallel.py` | V4 EP source path through `_predict_layer_with_kernel_gap` | ~20 |
| 8 | `frontier/operators/families.py` | PDD admission for the V4 SP payload; endpoint collectives as communication operators with builders; combine classification | ~70 |
| 9 | `frontier/operators/sequence_parallel_moe.py` | Unequal-combine consumer selects by the combine classification | ~10 |
| 10 | `frontier/execution_time_predictor/kernel_gap.py` | Segment classes (graph-internal, eager, seam) and V4 count naming | ~50 |
| 11 | `frontier/model_architectures.py` | `runtime_limitations` text | ~10 |
| 12 | `docs/general/module-size-exceptions.md` | Record the scoped changes to `sklearn_execution_time_predictor.py` (8,666 lines) and `metrics_store.py` (5,744 lines) | ~15 |
| 13 | `frontier/profiling/attention/dsa_kernel_trace.py`, `dsa_vllm_runner.py`, `dsa_plan.py`, `dsv4_attention_rows.py` | Keep kernel intervals (start, end, stream) in raw records; counts in `time_stats.<op>.kernel_count`. Bootstrap part first (stage 1a): FULL and PIECEWISE capture and replay of the same call for V4 replay windows, kernel-signature ownership from the eager hook pass, one-to-one interval match, joined boundary | ~200 |
| 14 | `frontier/profiling/attention/dsv4_attention_plan.py` | Serving-window cases from C1: decode batches × mixed contexts, 6144 prefill and SP-local rows, first tail alone and with decodes, both planner roles | ~100 |
| 15 | V4 operator producer (new module beside `dsv4_attention_runner.py`) | mHC (first, fused, attn-post, final, head), shared expert, embedding, final norm, LM head, `kv_cache_block_zero`, pre-forward preparation, with typed operator contracts | ~300 |
| 16 | `frontier/profiling/moe/main.py` and wrapper | Rank-class selection of the local expert population; V4 gate on the `L` grid; balanced path signature | ~100 |
| 17 | `frontier/profiling/kernel_gap/main.py` | V4 graph-replay sampling on the reduced engine with segment classification; per-segment first kernels included at seams; V4 per-stream gap reducer (§3.3) beside the unchanged global reducer. Bootstrap part first (stage 1a) | ~130 |
| 18 | `tests/analysis/v4_runtime_census.py` (new) | Stage-0 R1 reducer over native traces, streaming and memory-limited; the serving-counter mode census by step class (from `evidence/routing/mode_census.py`); the A11 diagnostic comparing each Frontier replay's mode and synchronized widths per forward with the native counters | ~230 |
| 19 | Tests | `test_dsv4_predictor.py`, `test_dsv4_composition.py` (new; both sides of the threshold, Breakable serialization, composed elapsed and offsets, gap charged once); V4 dispatch, payload, unequal binding, EP gap, ledger sentinel and collision tests; V2 dummy workload; migrated `read_isolated_scope` cases; V4 cases in the fidelity regression | ~680 |
| 20 | `frontier/scheduler/utils/dp_dummy_forward.py` | The DP-synchronized descriptor of §3.7 item 2 as one function over all lanes of a forward (minimum mode, widest width, uniform-count and max-query redispatch, FULL request padding), replacing the dummy-only body of `resolve_dummy_forward_inputs`; the rendezvous placement of item 3 | +70 / −30 |
| 21 | `frontier/events/replica_stage_schedule_event.py` | V4 real lanes enter the first pre_moe room at their pre-DP ready time and defer post-DP preparation and first-layer pricing to the rendezvous, in the PREFILL path (lines 222-300) and the DECODE / MONOLITHIC path (lines 520-600) | ~60 |
| 22 | `frontier/scheduler/utils/sync_entry.py` | For V4 forwards, the rendezvous timeline replaces the launch-start alignment (lines 113-131); the descriptor is applied to every lane of the first room before pricing (lines 110-112) | +25 / −10 |
| 23 | Tests | `test_balanced_routing.py`: DP2 / EP16 and DP4 / EP32 widths and per-EP-rank counts against native `determine_expert_map` (done, 102 passed). `test_dsv4_dp_sync.py` (new): A13 (a) on the distinct (local sizes, desired modes, uniform counts, max query lengths, synchronized descriptor and widths) tuples of the three cases' native counters, kept as `tests/fixtures/dsv4_native_dp_sync_rows.json` (685 distinct tuples and about 33 KB before the uniform-count and max-query fields are added); A13 (c) from Frontier batches through the classifier, the synchronization, the workload builders and the pricing consumer (the row-428 tuple; first tail alone and with steady decodes; steady decodes only; dummy lanes beside each; all-dummy and PREFILL forwards; two-lane and four-lane forwards with unequal capture sizes and with one eager lane); A13 (d) timing with the 110 ms sentinel. `test_dp_dummy_forward_pricing.py` extended to real lanes | ~220 |
| 24 | `frontier/scheduler/replica_scheduler/vllm_v1_iteration_policy.py` | The `full_and_piecewise` local classifier of §3.7 item 1: FULL for uniform decodes within the FULL bound, PIECEWISE (token padding only) up to the largest capture size, NONE above; graph metadata for the PREFILL role under this value; a typed configuration error for non-V4 models | ~40 |
| 25 | `frontier/config/config.py` | Add `full_and_piecewise` to the choices and validation of the existing `decode_cuda_graph_mode` field (lines 172-174, 621) | ~10 |
| 26 | `frontier/profiling/cpu_overhead/vllm_cpu_probe.py` (only if R2h / R2 run) | Split `prepare_inputs_e2e` at the recorded DP all-reduce into pre-DP and post-DP shares (§3.4) | ~30 |
| 27 | `frontier/entities/execution_time.py` | The V4 branch of `_get_attention_layer_execution_time` (lines 1116-1125) returns the composed elapsed, with the gap charged once | +15 / −10 |
| 28 | `frontier/metrics/metrics_store.py` | The trace places V4 attention scopes at their composed start offsets (lines 1157-1165) | ~5 |

Total: 28 entries (some cover several files), about 3,310 lines added and 330 deleted; revision 2b had 23 entries and about 2,740 / 290. This is far above the D-H4A-CHECKPOINT limit of 5 files and about 200 lines (decision D3). Entry 26 is needed only if R2h / R2 run.

- **No new config field; one new value of an existing field.** `decode_cuda_graph_mode` gains `full_and_piecewise` (entries 24-25, decision D7). V4 attention rows use the attention input; V4 operator rows use `linear_op.csv` with typed contracts; gating and routed MoE use `moe.csv`; collectives use the measured backend's `summary.csv`; gaps use `kernel_gap.csv`; host costs use the CPU-overhead inputs and `skip_cpu_overhead_modeling`. If implementation finds a binding that these cannot express, it stops and asks YC.
- **Module sizes.** All touched modules except `sklearn_execution_time_predictor.py` (8,666 lines) and `metrics_store.py` (5,744 lines) stay below 2,000 lines; those two get only registration, routing and trace-offset lines, recorded in entry 12.

## 7. GPU jobs

All jobs go through the StepMind Python `RJobBackend` with `i-fengyicheng` auth, via `/ds4-exec`. H800 uses codesign; H200 uses step_main, non-preemptible, with running plus queued at most 64 GPUs. Every run with 32 or more GPUs uses soft gates only: validators log and report, never stop training or profiling. H800 and H200 evidence never qualify each other. The figures are planning caps; C1 and the pilots replace them with exact grids before stage 2 starts.

| Job | GPUs | Content | Cap (GPU-hours) |
| --- | --- | --- | --- |
| P1h (H200, first) | 1 | Replay-route bootstrap (stage 1a), then the composition and identity pilot against the H200 native traces: FULL, PIECEWISE and eager replays, both sides of the multi-stream threshold (stage 1) | 2 |
| G1h-G4h (H200) | 1 each | Same content as G1-G4 below, including the dummy coordinates and PIECEWISE first-tail windows from C1; decode batch up to 40; rank classes for EP16 and EP32 (8 local experts); M grid includes M6168 and M24576 for DP4 | 7 |
| G5h (H200) | 16, then 32 | Uniform curves and unequal vectors for TP8, EP16 and EP32 | 36 |
| R2h (H200, only if R1 shows eager host time is material) | 32 | Native P2-32 with the CPU probe, for `forward_launch` rows | 32 |
| P1 (H800, after H200) | 1 | The same replay route and pilot against the H800 traces, FULL, PIECEWISE and eager | 2 |
| G1 (H800) | 1 | Attention serving window from C1, including dummy coordinates and PIECEWISE first-tail windows; KERNEL_ONLY with intervals, CUDA_EVENT for validation | 3 |
| G2 (H800) | 1 | V4 operators of §4 over decode rows 1-64, SP prefill rows, 6144 | 1.5 |
| G3 (H800) | 1 | MoE rank classes for EP16 and gate `L` grid | 2 |
| G4 (H800) | 1 | V4 graph-replay `kernel_gap.csv` | 0.5 |
| G5 (H800) | 16 | Small payload points (TP8, EP16) and the ordered unequal vectors from C1, with `vllm_forward_collectives.py` | 12 |
| R2 (H800, only if R1 shows eager host time is material) | 32 | Native P0 with the CPU probe, for `forward_launch` rows | 32 |

Caps: H200 45 GPU-hours, plus 32 if R2h runs; H800 21, plus 32 if R2 runs (revision 2b: 44 and 20). No H200 runtime-census job is needed: the retained native traces cover it.

## 8. Time estimate (elapsed working days)

| Stage | Days |
| --- | --- |
| 0. R1, C1, S1 (CPU) | 2 |
| 1a, 1. Replay route and pilot P1h (1 GPU, queue-bound); P1 later | 1.5 |
| 2. Producers (files 13-17), then G1h-G5h | 3 (grids run in parallel with stage 3) |
| 3. Simulator files 20-28, then 1-12, and tests | 5.5 |
| 4. Training, A1-A13, replays, cross-check | 2.5 |
| **Total** | **about 15-16.5** (revision 2b: 13-14.5); the H200 P2-32 / P2-64 milestone at about 11.5, H800 P0 about 3.5 days later (P1, G1-G5, replay) |

Main schedule risks: GPU queues, image or runtime failures (earlier V4 profiling jobs needed several attempts), a replay route that cannot match one to one (stage 1a), and a failed pilot.

## 9. Decisions for YC

| # | Decision | Options | Recommendation |
| --- | --- | --- | --- |
| D1 | Attention layer cost | (a) Option A: per-scope costs and a critical path over the conditional source stream graph, checked per group by P1h. (b) A measured layer composite (elapsed of the whole layer, shape-aware) with scopes for attribution only. (b) is close to the measured shortcut that D-NO-STEP-COST-SHORTCUT forbids, changes approved Option A, needs whole-layer grids over every workload group (more GPU time), cannot be recomposed when one scope changes, and keeps per-scope costs only as attribution, so the gap and eager host time are hidden inside the composite and are at risk of being charged twice | **Decided (2026-10-27): (a)** |
| D2 | Host time of eager steps and eager segments | (a) The existing CPU-overhead `forward_launch` rows (measured host launch time per forward, keyed by batch features), used only for the host stream; the device stream stays operator-composed. (b) A per-kernel launch cost times the kernel count, which is new modeling. (c) Leave host time out and state the limit | **Decided (as recommended):** after R1, (c) with an explicit zero if eager host time is not material, otherwise (a) for the host stream only |
| D3 | Size | About 19 files and 2,500 lines at the decision; 23 entries and about 2,740 lines in revision 2b; 28 entries and about 3,310 lines in revision 2c (§6), above D-H4A-CHECKPOINT | **Decided:** approved, staged as §5; the growth since the decision (the routing and timing identity of §3.7, the composed-duration consumer, the replay route) confirmed on 2026-10-27 |
| D4 | GPU budget | Caps of §7; R2 / R2h (32 GPUs each) only if R1 shows eager host time is material | **Decided:** caps approved; the revision 2c caps (H200 45, H800 21) confirmed on 2026-10-27; exact grids are reported after C1 |
| D5 | Reduced TP1 engine for per-operator device time | (a) Reduced TP1 with per-rank TP8 shapes, qualified by kernel identity against the native TP8 traces (P1h, P1). (b) Per-operator profiling inside full native serving | **Decided:** (a) |
| D6 | Order | H800 P0 first, or H200 first | **Decided:** H200 first (P2-32 and P2-64 mainline); H800 P0 after H200 |
| D7 | Local graph mode of V4 forwards | (a) A new value `full_and_piecewise` of the existing `decode_cuda_graph_mode` field, V4 only in H3o, that reproduces the native `FULL_AND_PIECEWISE` dispatch (§3.7 item 1). (b) Change the meaning of `full_decode_only` or `piecewise`, which would change GLM and other models' results (A12). (c) A new config field | **Decided (2026-10-27): (a)** |

## 10. Risks

1. **First-tail step.** The native first tail alone takes 61-69 ms against 20-24 ms for decode-only. It runs PIECEWISE on every lane of its forward (§2, §3.7); its segment boundaries are unknown until R1, and it is not assumed to be host-bound.
2. **Pilot failure.** If the critical-path composition misses the joined elapsed (A4), the route stops for a decision; no factor is added.
3. **Single layer versus 43 layers.** Rows sample one layer. L2 and stream effects across 43 layers are not proven additive; the cross-check shows their size and nothing corrects them.
4. **Profiler overhead.** R1 host times are upper bounds. Host costs come only from unprofiled measurements (R2 / R2h).
5. **Routing.** Balanced routing and dummy weights only; real router imbalance belongs to ticket 14. Identity with native balanced routing is gate A13.
6. **Diagnostic step-cost replays.** After files 20-25, the narrower lane of a synchronized forward is looked up at the synchronized width, so the diagnostic step-cost replays (H4a, H4b, P2-32d, P2-64d) can change or meet coverage errors. They stay diagnostic and are rerun only for the cross-check (A11).
7. **Dummy frequency.** How often Frontier issues DP dummies, notably on PREFILL engines waiting for NIXL reads (natively 8,408-9,455 FULL prefill dummies on the first prefill node), follows the existing lifecycle model, which H3o does not change. The A11 mode and width census shows any difference; it is reported, not corrected in H3o.

Deferred (not in H3o): the non-V4 paths have the same kind of gap. Their real-lane attention keeps lane-local widths, and their EP width rule (`ep_wave_inputs.py:31-43`) pads every lane to the widest one whenever any lane has graph metadata, instead of using the minimum mode. Changing it would change GLM results, which A12 forbids without YC's approval.

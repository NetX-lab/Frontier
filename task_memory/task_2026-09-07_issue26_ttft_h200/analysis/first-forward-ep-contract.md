## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Inspected the first-forward dummy, DP padding, dispatch, and uniform-routing shape contract without modifying source or running GPUs. |
| 2026-09-08 | Checked the fresh formal-start running-state and empty-schedule evidence; separated observed host boundaries from unobserved dummy GPU overlap. |

# First-forward EP shape contract

The current eager configuration does not pad every DP lane to 4,096 tokens. Given the first forward with DP0 carrying request 0 and 4,096 real tokens and DP1 having no ready request, the inspected vLLM implementation executes one dummy token on DP1. Naive dispatch concatenates `[4096, 1]` into a 4,097-row MoE input. Frontier's current shared-wave input excludes idle sources and contains 4,096 real tokens. Matching the local request and its 4,096-token prefill therefore qualifies local attention shapes but does not establish identical post-dispatch MoE shapes.

## Frozen source and evidence limits

- vLLM diagnostic source: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`, commit `8453dd342c6aa2721aaf4b410998aab2f38bc2ec`. The route-only extension in that commit does not modify the inspected model/collective code.
- Frontier source: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`. The fresh full-run ledger is `analysis/cpu-shared-forward-01/prefill_stage_ledger.jsonl`, produced after shared-forward repair `8ba22b49`.
- Relevant configuration: TP4, DP2, EP8, eager, naive all2all, FLASHINFER attention, uniform top-k, 128 experts, top-k 8, hidden size 2,048, intermediate width 768, BF16, one H200 node, no sequence-parallel configuration enabled in this case.
- This review does not collect fresh GPU shapes or operator durations. The 4,097 global row count below is a conditional conclusion from the pinned current source and the stated first-forward membership. The fresh op run must supply its own runtime shape evidence.

## vLLM execution contract

| Boundary | Pinned implementation | Consequence for the stated first forward |
| --- | --- | --- |
| Idle engine participation | `vllm/v1/engine/core.py:1090` busy loop calls `execute_dummy_batch()` when no model execution occurred while the engines remain running. `vllm/v1/worker/gpu_worker.py:556` calls `_dummy_run(1)`. | DP1 contributes one model row without admitting a real request. |
| DP graph padding | `vllm/v1/worker/gpu_model_runner.py:1927` returns `(0, None)` when `enforce_eager` is true. `_dummy_run` invokes it at line 3110; the real input path invokes it at line 2024. | DP0 remains 4,096 rows; DP1 remains one row. There is no 8,192-row max-padded global tensor in this case. |
| Dummy attention | `_dummy_run` initializes `attn_metadata=None` at line 3164; neither default `force_attention=false` nor graph mode NONE builds attention metadata. `vllm/v1/attention/backends/flashinfer.py:793` returns the supplied output immediately for absent metadata. | The dummy model still executes surrounding model operations and MoE participation, but it does not represent a real one-token decode request with valid request KV metadata. |
| DP metadata | `vllm/forward_context.py:204` constructs DP metadata even with absent attention metadata when `num_tokens` is provided. `DPMetadata.make` at line 90 obtains actual counts when no padded vector is provided; `num_tokens_across_dp` at line 72 gathers them through a CPU-group all-reduce. | The counts are `[4096, 1]`, cumulative bounds `[4096, 4097]`. No CUDA graph padding is required for DP rendezvous or naive dispatch. |
| Naive dispatch | `vllm/distributed/device_communicators/all2all.py:28` allocates the concatenated buffer using the final cumulative count, copies the local segment, then broadcasts every DP segment. `dispatch` at line 46 performs this for hidden states and router logits. | Post-dispatch hidden states have shape `[4097, 2048]`; router logits have shape `[4097, 128]`. The all2all manager uses DP groups, as initialized by `base_device_communicator.py:30`; EP8 does not multiply this token population by eight. |
| Routing location | `FusedMoE.forward_impl`, `vllm/model_executor/layers/fused_moe/layer.py:1780`, performs naive dispatch before `quant_method.apply`; `forward_cuda` at line 447 then selects experts. | Top-k operates on the global 4,097 rows, not solely the local 4,096 real-request rows. |
| Uniform assignments | `uniform_topk`, `vllm/model_executor/layers/fused_moe/fused_moe.py:932`, assigns token row `i` to `(i*8+k) % 128` for slots `k=0..7`. | 4,096 rows assign each expert 256 times. Row 4,096 adds one assignment to experts 0 through 7. Global assignments total 32,776: experts 0–7 have 257 each, experts 8–127 have 256 each. |
| Local expert ownership | `determine_expert_map`, `vllm/model_executor/layers/fused_moe/layer.py:680`, maps contiguous sets of 16 experts to each EP rank for 128/8. | Under that map, EP rank 0 has 4,104 routed assignments, and each remaining EP rank has 4,096. The expert GEMM's kernel block padding must be checked separately from these logical assignment counts. |
| Combine | `NaiveAll2AllManager.combine`, `all2all.py:57`, all-reduces the global output and slices the local DP segment. | The collective operates on a global `[4097, 2048]` output; DP0 receives 4,096 rows and DP1 receives one. This source establishes the selected implementation; it does not establish that Frontier predicts the same collective algorithm or traffic. |

The 4,097-row tensor is a global dispatched population including a dummy row. Calling it a max-padded local batch would be incorrect. Kernel-level alignment to expert block sizes is a separate kind of padding, and must retain the actual runtime configuration when comparing grouped GEMM/shuffling work.

## Frontier evidence

**Fresh raw observation:** the first row of `analysis/cpu-shared-forward-01/prefill_stage_ledger.jsonl` records batch 0, request `["0"]`, `request_num_tokens=[4096]`, `request_num_prefill_tokens=[4096]`, schedule epoch 1, start 0, and end `0.06579007690374297` seconds. This proves the local request composition and boundary. It does not itself log the remote idle source or a padded global tensor.

**Current source consequence:** `frontier/scheduler/utils/ep_wave_inputs.py:45` selects only sources with `is_idle=false`. Lines 58–66 sum tokens from those sources and reuse the one non-idle batch as the aggregate when there is only one. Therefore the stated first wave's aggregate logical input has 4,096 real tokens. Idle-lane participation in the shared synchronization does not add a synthetic token to that aggregate. The balanced routing contract consequently gives 32,768 assignments, 256 per expert, 4,096 per EP lane.

**Same-task corroboration with a narrower validity:** `analysis/uniform-frontier-first-prefill.json` retains an earlier first-prefill workload audit with 48 layers, 128 experts, 256 assignments per expert, and 32,768 total assignments. Its timing values are not substituted for the fresh repaired-run result in this review. Separately, prior `runs/h200-diagnostics-03/runtime/routing` recorded a local 4,096-token formal batch and a routing row with `num_tokens=4097`, as documented in `analysis/routing_global_counts_proposal.md`. That older diagnostic used the earlier routing experiment and does not validate the new uniform op run's durations or expert counts.

**Fresh-profile inventory observation:** current-task `supplements/moe-uniform-01/moe.csv` has 18 rows with `num_tokens=4096` (six labeled `load_distribution=uniform`) and zero rows with `num_tokens=4097`. Those rows declare EP8, 16 local experts, `uniform_topk`, top-k 8, hidden size 2,048, expert width 768, and MoE TP1. Row labels and local routed-count features are not a substitute for selecting the exact predictor row or establishing kernel work; the operator RCA lane owns that analysis.

## Operator comparison constraints

| Operator group | Required alignment |
| --- | --- |
| Real-lane QKV/projection, norms/RoPE, prefill attention, KV save, and local router linear | Preserve DP0's 4,096 real tokens, request KV state, TP4 sharding, and the actual operator's position before dispatch. Local membership is necessary here; a global 4,097 replacement would misstate these inputs. |
| Uniform top-k and other post-dispatch routing work | Use actual global M=4,097 and 128 experts/top-k 8. Preserve `[4096,1]` source segments. Do not compare against a purely local 4,096-row operator as exact shape parity. |
| Token-to-expert alignment/shuffling, W1, gated activation, W2, routed reduction | Match each EP rank's actual expert map and counts, logical global rows, intermediate widths, top-k expansion, kernel block size, and padding. The extra assignments can cross a block boundary, so the 1/4,096 logical-token ratio is not a proven latency-error bound. |
| Dispatch/combine | Match global DP segments and actual collective algorithm/participant groups. The selected vLLM naive path broadcasts hidden and router buffers, then reduces the global output. An operator name such as `alltoall` alone does not establish equal traffic or implementation. |
| Shared forward latency | Account for the critical path across all EP participants; do not sum EP8 GPU durations as sequential work or treat missing dummy-worker batch rows as no work. |

Conclusion: the local first batch is sufficient to identify the real prefill case, but insufficient to declare exact EP operator alignment. This is an explicit shape qualification, not proof that dummy-token modeling explains the first-batch latency gap. No simulator source correction or new GPU run was performed in this bounded review.

## Inspection record

Read-only commands used `rg`, bounded `sed`, and a small standard-library `json`/`csv` inspection with `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python` (conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13). The ledger inspection read only its first line. The CSV inventory compared exact `num_tokens` values 4096 and 4097; observations are recorded above. The initial combined search used the Frontier working directory for two vLLM paths and returned file-not-found errors; those source reads were immediately repeated against the correct diagnostic checkout. No source files were changed.

## Fresh formal-start evidence: drained requests versus idle engine wave

Source directory: `analysis/h200-rca-01/runtime/batch`. This section inspects the new same-run route/snapshot/enqueue and scheduler records. It does not reuse latency values from prior experiments.

**Established result:** the frontend's `engines_running` state was still true at the first formal routing decision. The logs also show an empty scheduler result shortly before that route. They do not directly establish that a particular dummy GPU forward was in flight at the precise routing timestamp, or that it accounts for all of the route-to-enqueue interval.

The first formal request is `cmpl-pf4096_dc1024:0-0`. Its route timestamp is `3906951.752607512` in the host monotonic domain.

| Same-run event | Monotonic timestamp (s) | Relative to first route (ms) | Raw anchor |
| --- | ---: | ---: | --- |
| Latest prior applied snapshot: counts `[[0,0],[0,1]]`, wave 0, `engines_running=true` | 3906950.703675288 | -1048.932224046 | `server.dp_route.jsonl.pid430.jsonl:740` |
| Empty schedule: step 4235, no scheduled tokens, waiting/running both zero | 3906951.722403820 | -30.203691684 | `server.scheduler.log:9349` |
| Formal route: DP0, counts `[[0,0],[0,1]]`, wave 0 | 3906951.752607512 | 0 | `server.dp_route.jsonl.pid430.jsonl:741` |
| Next applied snapshot: counts `[[0,0],[0,0]]`, wave 0, `engines_running=true` | 3906951.773079966 | +20.472454373 | `server.dp_route.jsonl.pid430.jsonl:742` |
| DP0 engine QUEUED event | 3906951.791754934 | +39.147422183 | `server.dp_route.jsonl.pid701.jsonl:154` |
| First formal schedule: step 4213, exactly request 0 / 4096 tokens | 3906951.793351891 | +40.744379163 | `server.scheduler.log:9351` |

`server.scheduler.log:9347` records the final one-token scheduling of warmup request `cmpl-warmup:pf4096_dc1024:r2:99-0`; line 9348 records its completion; line 9349 records the empty schedule. The scheduler log's `timestamp_monotonic` is sampled at the start of `schedule()` (`scheduler.py:364`), although the JSON row is emitted near its end (`scheduler.py:987`). This is a scheduler host boundary, not GPU start time. The shared scheduler file does not carry PID or DP identity, so its anonymous empty result must not be assigned a DP rank or global collective round solely by adjacent line order or its local step value.

The client artifact provides separate same-wall-clock corroboration of the short phase transition: last warmup request completion was `1788860540746859144 ns`, and the first formal request's client arrival/dispatch wall timestamp was `1788860540771403538 ns`, separated by `24.544394 ms`. These wall timestamps are not subtracted from the monotonic records. Completion of all warmup client requests establishes request drainage, not engine-wave quiescence.

### Why frontend running=true is established

The route logger itself records the wave and load counts, not the running flag. However, the complete PID430 stream contains no `snapshot_receive` with `engines_running=false`. In the selected `DPAsyncMPClient`/`DPLBAsyncMPClient` async path, the relevant assignments after initialization are the stats task's update from `(counts, wave, running)` (`core_client.py:1077`) and the task's first-request notification setting the state true (`core_client.py:1058`). The synchronous client's `get_output()` false assignment at line 674 is not on this execution path; the async output handler does not clear this flag. The immediate prior snapshot is true and no intervening snapshot exists before the route. Thus the frontend used true running state at this decision, rather than merely receiving a true flag sometime after the request.

This proves frontend-visible state. An asynchronous snapshot is not a simultaneous observation of every EngineCore's local state or GPU activity. In particular, the pre-route load snapshot was approximately 1.049 seconds old; its remaining running request count is not proof that the warmup request remained live at the formal route timestamp.

### Why drained requests can coexist with a running wave

Pinned `vllm/v1/engine/core.py` implements:

1. `_process_input_queue()` at line 759 handles available request inputs between engine steps. It waits for input only when engines are not running and no request/batch work remains.
2. `step()` at line 290 returns `model_executed=false` for no requests, or after an empty scheduled result. A cleanup scheduler result can occur after the last request completes because `has_requests()` includes finished requests not yet removed from the batch.
3. The DP busy loop at line 1090 invokes `execute_dummy_batch()` when no real model ran while the loop remains in running state. This preserves participation with the other DP engine.
4. `_has_global_unfinished_reqs()` at line 1131 increments the core step counter and unconditionally returns true on 31 out of every 32 calls. Only a step divisible by 32 performs the global finish check. An observed false result ends the wave and resets the counter.

Therefore, all client requests can finish before the wave is declared idle, and dummy rounds can continue during that interval. The fresh empty schedule and subsequent all-zero-but-running snapshot are consistent with this mechanism. The logged scheduler `step` is a different local counter from the core finish-sync `step_counter`; using `4235 % 32` to infer the finish-sync position would be invalid.

### Causal limit and minimal missing evidence

The observed `39.147422183 ms` is **route-to-engine-QUEUED elapsed host time**. It can include sending/transport, input-thread work, EngineCore input-queue waiting, ongoing model/RPC/collective work that delays the next input processing, and host scheduling/logging overhead. It is neither a CUDA-event duration nor a measured CPU execution sum.

The current artifacts do not record an identified dummy invocation's entry and exit, its core step/wave, or GPU timeline correlation. Consequently:

- They support the running-wave/dummy-wait explanation as a concrete source-grounded candidate.
- They do not prove the exact route instant falls within a dummy GPU execution interval.
- They do not allocate the entire 39.147 ms to dummy work or to CPU overhead.

The minimum additional evidence to test the candidate would be same-host monotonic begin/end boundaries for `execute_dummy_batch()` with PID, DP rank, core wave, and core step identity. Those would establish whether the engine is blocked in that dummy invocation when the request is routed and when it becomes available for input processing. Distinguishing actual GPU execution from host/RPC/collective waiting inside that invocation additionally requires correlation with the existing GPU/CUDA timeline. Isolating transport from Core input waiting would require the intermediate input-receipt/queue-insertion boundary for the same request. These are missing evidence descriptions only: this review adds no logger, synchronization, source change, or GPU run.

Read-only verification used Python 3.13.13 standard-library JSON iteration to inspect all route rows, selected the scheduler records within 150 ms before and 100 ms after the first route, and checked the client rows for warmup request 99 and formal request 0. Observed flags, timestamps, raw line anchors, and source mutation sites were reconciled as reported above.

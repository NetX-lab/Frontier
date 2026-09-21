## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded fresh Frontier and CUDA-event operator comparisons, complete first-batch kernel evidence, profiler coverage failure, and validated reduced communication probe. |

# First-batch operator RCA verification

## Execution

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
CPU conda environment: `dev-vidur-v03-hopper-e2e`; Python 3.13.13 at `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`.

Selection check:

```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_selection.py --checkout /data/ycfeng/tmp/issue26-vllm-diagnostics-20260908 --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/first-batch-op-rca/selection_validation.json
```

Fresh simulator measurement:

```bash
FRONTIER_LOG_LEVEL=INFO PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_cpu_frontier_worker.py --config /data/ycfeng/tmp/issue26-first-batch-op-frontier-01/config.json --output /data/ycfeng/tmp/issue26-first-batch-op-frontier-01/runtime
```

The wrapper created new caches under the path preserved in `analysis/first-batch-op-rca/frontier_scratch.txt`, retained the current task's freshly measured profile inputs, and used the current active worktree. The one-request arrival remains 0.0, 4096 prefill / 1024 decode; only the trace cardinality and intended time-limit differ from the 100-request clean run. Full simulator command is preserved in `analysis/first-batch-op-rca/frontier_command.json` and repeated here:

```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m frontier.main --simulation_mode online --sys_arch co-location --no-enable_parallel_clusters --cc_backend_config_type collective_sim --collective_sim_cc_backend_config_intra_server_model nvlink_analytic --cluster_config_num_replicas 1 --replica_config_model_name qwen3-a3b-30b-moe --replica_config_device h200 --replica_config_attn_tensor_parallel_size 4 --replica_config_attn_dp 2 --replica_config_moe_tensor_parallel_size 1 --replica_config_moe_expert_parallel_size 8 --replica_config_num_pipeline_stages 1 --replica_config_total_expert_num 128 --replica_config_router_topk 8 --replica_scheduler_config_type vllm_v1 --decode_cuda_graph_mode none --no-vllm_v1_scheduler_config_enable_prefix_caching --no-vllm_v1_scheduler_config_enable_chunked_prefill --vllm_v1_scheduler_config_max_tokens_in_batch 16384 --vllm_v1_scheduler_config_batch_size_cap 1024 --vllm_v1_scheduler_config_block_size 16 --vllm_v1_scheduler_config_num_blocks_mode explicit --vllm_v1_scheduler_config_num_blocks 310809 --request_generator_config_type trace_replay --trace_request_generator_config_max_tokens 16384 --no-random_forrest_execution_time_predictor_config_enable_dummy_mode --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling --replica_config_network_device h200_dgx --replica_config_moe_routing_distribution_type balanced --replica_config_moe_routing_seed 42 --collective_sim_cc_backend_config_cluster_servers 1 --collective_sim_cc_backend_config_cluster_gpus_per_server 8 --random_forrest_execution_time_predictor_config_linear_op_input_file /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-fresh-profiles-02/runtime/profiles/compute/h200/qwen3-a3b-30b-moe/linear_op.csv --random_forrest_execution_time_predictor_config_atten_input_file /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-fresh-profiles-02/runtime/profiles/compute/h200/qwen3-a3b-30b-moe/attention_combined.csv --random_forrest_execution_time_predictor_config_moe_input_file /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/supplements/moe-uniform-01/moe.csv --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 4096 --random_forrest_execution_time_predictor_config_prediction_max_batch_size 128 --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384 --random_forrest_execution_time_predictor_config_num_training_job_threads 8 --trace_request_generator_config_trace_file /data/ycfeng/tmp/issue26-first-batch-op-frontier-01/arrivals.csv --metrics_config_run_id runtime --metrics_config_write_metrics --metrics_config_store_request_metrics --metrics_config_store_batch_metrics --metrics_config_store_token_completion_metrics --metrics_config_enable_metrics_ground_truth_trace --no-metrics_config_store_plots --no-metrics_config_enable_chrome_trace --no-metrics_config_write_json_trace --replica_config_moe_gating_routing_runtime_path uniform_topk --cluster_scheduler_config_type vllm_load_balancing --time_limit 1 --log_level info --metrics_config_output_dir /data/ycfeng/tmp/issue26-first-batch-op-frontier-01/runtime/metrics --metrics_config_cache_dir /data/ycfeng/tmp/issue26-cpu-frontier-p5n9slvi/predictor-cache --collective_sim_cc_backend_config_cache_dir /data/ycfeng/tmp/issue26-cpu-frontier-p5n9slvi/collective-cache --collective_sim_cc_backend_config_runner_out_dir /data/ycfeng/tmp/issue26-cpu-frontier-p5n9slvi/htsim
```

Extraction:

```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_frontier.py --log /data/ycfeng/tmp/issue26-first-batch-op-frontier-01/runtime/frontier.log --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/first-batch-op-rca
```

## Criteria and observed evidence

- **PASS, diagnostic selection:** seven boundary cases, warmup/formal sequence `[false,true,true,true,false]`, unchanged model/get_dp_padding/_dummy_run call sites. This is source verification; full H200 execution is the runtime check.
- **PASS, first-prefill extraction:** 48 layers and 384 EP-lane records; duplicated predictions agree; every EP lane predicts identical per-op values for this balanced first wave. Sum each phase once, not all EP participants. The extracted log stops at request 0's first-prefill completion marker.
- **PASS, accounting:** rounded operator sum 61.539408 ms + EP dispatch/combine 4.250688 ms = 65.790096 ms. Explicit final EP wave endpoint is 65.790076904 ms; difference 0.000019096 ms is below the 0.000504 ms maximum rounding budget derived from the number of six-decimal log fields. This validates internal accounting, not Frontier-vLLM numerical agreement.
- **PASS, independent clean endpoint reconstruction:** all 100 native clean rows reproduce the root analysis. First official TTFT 121.48427963256836 ms; same-row reconstructed scheduled-to-first-engine-output 81.1579111032188 ms; remaining 40.326368529349566 ms combines other host/wait/output boundaries and is not CPU-only.
- **FAIL, requested time limit:** effective config contains `time_limit=1`, but sequential execution continued beyond it. The sequential guard in `frontier/simulator.py:1482` is unreachable after `return True` in `_maybe_export_sequential_checkpoint`; the separate guard at line 1142 is in the parallel loop. After preserving complete first-prefill evidence, the diagnostic process was interrupted (exit 130, `KeyboardInterrupt` in `subprocess.wait`). No complete-request E2E claim is made. This unrelated production defect is recorded, not modified.

## Pending

The H200 batch and CUDA-event scope supplements completed. The record-function run failed on its third selected DP0 batch; complete first-batch evidence is qualified below. The reduced communication-scope supplement has all 24 selected rank/batch records, validated below; the parent owns full 400-request completion and identity verification. Operator-family shape differences, missing metadata and profiler perturbation must be carried into the final comparison. The full three-aligned-logical-batch operator gate has not passed.

## Fresh CUDA-event operator supplement

Diagnostic checkout `8453dd342c6aa2721aaf4b410998aab2f38bc2ec`, eight H200 GPUs, image/runtime and exact commands are preserved in `analysis/h200-rca-01/runtime/operators/mode_manifest.json`, `server_command.json`, and the root run records. Runtime metadata is explicitly disabled; op logger uses CUDA events, default scopes, per-scope aggregation and three formal local batches per DP worker. Model execution and collective counts are unchanged by selection.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_vllm.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-01/runtime/operators --frontier task_memory/task_2026-09-07_issue26_ttft_h200/analysis/first-batch-op-rca/frontier_summary.json --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/first-batch-op-rca/operators
```

**PASS, collection integrity:** all eight workers have exactly three selected formal real batches with indices 0, 1, 2; op rows join exactly those 24 rank/batch identities. The first DP0/TP0 batch contains only request `cmpl-pf4096_dc1024:0-0`, 4096 prefill tokens. Raw normalized rows retain all workers and scope identity.

**Observed failure and correction:** the initial analysis expected 48 attention AR rows under the generic `tensor_parallel_allreduce` name. It failed its count assertion: actual generic AR count is one (embedding). Pinned `linear.py` resolves attention AR to `attn_post_proj_tp_allreduce`, absent from default logger scopes. The corrected analysis compares the parent projection+TP interval and preserves this missing standalone op evidence. It also excludes duplicated nested EP scopes and fused `add` from additive totals.

First batch CUDA-event interval is 116.210144 ms under op instrumentation; the independent same-generation batch-only run measures 80.335617 ms. The 35.874527 ms difference establishes substantial instrumentation/run sensitivity; no production operator correction may be sized directly from these op spans. De-duplicated outer-scope sum is 107.531296 ms, leaving 8.678848 ms inside the instrumented batch outside selected scopes.

| Operator / scope | Frontier ms | Instrumented vLLM ms | Absolute gap ms | Relative gap |
| --- | ---: | ---: | ---: | ---: |
| attn_kv_cache_save | 0.807168 | 0.721312 | 0.085856 | 11.903% |
| attn_post_proj_with_tp_allreduce | 20.145072 | 17.231616 | 2.913456 | 16.908% |
| attn_pre_proj | 6.584064 | 3.927296 | 2.656768 | 67.649% |
| attn_prefill | 7.026432 | 6.047232 | 0.979200 | 16.193% |
| attn_rope | 1.552128 | 1.029024 | 0.523104 | 50.835% |
| embedding_tensor_parallel_allreduce | missing | 0.222336 | missing | missing |
| expert_parallel_allreduce | missing | 37.563168 | missing | missing |
| expert_parallel_alltoall_combine | 2.125344 | 4.196768 | 2.071424 | 49.358% |
| expert_parallel_alltoall_dispatch | 2.125344 | 6.981152 | 4.855808 | 69.556% |
| input_layernorm | 1.085184 | 1.266720 | 0.181536 | 14.331% |
| moe_gating_linear | 0.661440 | 0.723584 | 0.062144 | 8.588% |
| moe_gating_routing_topk | 4.439280 | 6.306880 | 1.867600 | 29.612% |
| moe_grouped_gemm | 15.635904 | 18.180160 | 2.544256 | 13.995% |
| moe_shuffling | 2.541360 | 1.889632 | 0.651728 | 34.490% |
| post_attention_layernorm | 1.061376 | 1.244416 | 0.183040 | 14.709% |

These diagnostic discrepancies are not an aligned kernel-time gate. `expert_parallel_allreduce` illustrates the problem: TP0/1/3 report 37.563/37.883/37.672 ms, while TP2 reports 5.653 ms and spends longer in multiple earlier operator scopes. All four whole batches finish at 116.2–116.4 ms. Collective elapsed time includes rank waiting; attributing the entire 37 ms to network cost or a missing predictor op would be unsupported. The record-function/CUDA-kernel supplement produced complete first-batch activity and then failed, as documented below.


## Record-function failure and complete first-batch evidence

Reproducible offline analysis, Python 3.13.13 in the same CPU conda environment:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_kernels.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-01/runtime/kernels --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/first-batch-op-rca/kernels
```

**FAIL, complete kernel run:** at 10:02:55 UTC, DP0 workers failed in `finish_batch -> _collect_record_function_scopes` with `RuntimeError: Non-positive CUDA time for op tensor_parallel_allreduce: 0.0 us`. Batch 4736 is the third selected local formal forward. All four TP traces contain 3,221 kernel launches but lack the device activities for their first five launches. In TP0, the embedding AR `cuLaunchKernelEx` correlation 96038 is present under its CPU scope but has no matching kernel activity in the raw Chrome trace. The failure is incomplete recorded device activity, not a measured zero-duration all-reduce or omission of the cuda_driver category. A profiler-start activity/buffer issue is plausible; its lower-level CUPTI cause is unproven. No permissive zero handling or frozen-source change was made.

The second batch 4735 also has two missing launch correlations on TP1/TP2 despite not triggering this scope assertion. The first batch 4734 has 3,077 recorded kernel launches and matching kernel activities on each of the four DP0 TP ranks. This establishes first-batch launch/device coverage on DP0, not full-run or all-eight-rank coverage. First-batch records join request 0, 4096 tokens, and the worker PID from server.log. Coverage details and missing raw API events are preserved in `kernels/all_dp0_collector_coverage.json`.

| DP0 rank | Batch CUDA event ms | Device active union ms | Idle inside device envelope ms | Next launch not yet started ms | Extra MoE TP AR kernel ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP0 | 153.403107 | 143.099283 | 8.461469 | 4.054321 | 65.823747 |
| TP1 | 150.240097 | 54.768490 | 94.431108 | 74.125946 | 4.619221 |
| TP2 | 151.365372 | 143.430268 | 6.970282 | 2.981514 | 66.154996 |
| TP3 | 150.277786 | 109.843979 | 39.424278 | 28.834601 | 38.273540 |

The profiler run visibly accumulates command-submission gaps on TP1 while peer collective kernels remain active. The largest individual TP1 idle gap is only 0.382145 ms; 74.125946 ms is a sum across gaps, not one host stall. Kernel active duration therefore includes GPU collective waiting; it cannot establish pure wire or compute time. Next-launch-not-started time proves absent submission during that portion of the recorded idle gap, but does not distinguish CPU computation, thread scheduling, synchronization or other host waits. The 150–153 ms instrumented batch cannot replace the 80.335617 ms batch-only baseline.

First-batch TP1 kernel names additionally separate 48 attention projection GEMMs (1.241378 ms), 48 attention AR kernels (4.726896 ms), 96 fused MoE GEMMs (10.201439 ms), 48 gated SiLU kernels (7.070356 ms), and 48 MoE sum reduction kernels (2.723883 ms). Full names and counts are preserved in `kernels/first_tp1_selected_kernel_families.json`. The grouped-GEMM parent includes the first two MoE GEMMs plus SiLU; MoE sum is outside its scope. Frontier's `_run_fused_moe_iteration` uses a contiguous slice as W2 input and omits actual gated SiLU and final MoE sum. This is a source-backed coverage mismatch, with measured diagnostic kernel costs; the user-deferred gated-SiLU production repair remains deferred. Its measured value is below 20 ms but is not zero. These kernel numbers are not additive corrections to the clean TTFT or to CUDA-event profile predictions, which use a different timing family and contain their own launch gaps.

**Status:** first-batch DP0 trace coverage PASS; full profiler run FAIL; three aligned logical batches and complete operator numerical gate remain INSUFFICIENT_EVIDENCE.


## Reduced communication-scope supplement

Frozen vLLM checkout and case are unchanged. The separate H200 worker records only existing scopes selected by `VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES=expert_parallel_allreduce,attn_post_proj_tp_allreduce,tensor_parallel_allreduce`, with formal prefix `cmpl-pf4096_dc1024:` and per-DP limit 3. This uses 97 event pairs per selected forward. Exact runtime settings are preserved in `analysis/h200-rca-comm-01/runtime/operators/mode_manifest.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_communication.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-comm-01/runtime/operators --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/first-batch-op-rca/communication/summary.json
```

**PASS, diagnostic collection:** all 8 ranks have selected indices 0, 1, 2; the 2,328 rows join their batch identities and match DP/TP/PP, batch size, prefill/decode/token totals and per-request tokens. Every row uses CUDA-event/default/per-scope timing, count 1 and a positive finite duration. `(op_name, scope_seq)` keys are unique and continuous for each batch: 48 extra MoE TP AR, 48 attention TP AR and one embedding AR. These checks establish the intended records, not clean E2E parity. All raw paths and selected batch compositions are retained in `communication/summary.json`.

| DP0 rank | First batch ms | Attention TP AR ms | Extra MoE TP AR ms | Embedding TP AR ms |
| --- | ---: | ---: | ---: | ---: |
| TP0 | 86.412033 | 5.537248 | 4.792544 | 0.218336 |
| TP1 | 86.437950 | 5.649952 | 25.384416 | 0.301152 |
| TP2 | 86.414658 | 5.708768 | 28.413280 | 0.275616 |
| TP3 | 86.443520 | 5.644864 | 27.393568 | 0.296320 |

TP0 first batch is 6.076416 ms / 7.5638% above the independent batch-only 80.335617 ms measurement, versus the default all-op supplement's 35.874527 ms difference. Reduced instrumentation improves whole-batch proximity but retains material perturbation and cross-run variability. Extra MoE TP AR remains highly rank-dependent; its 28.413280 ms maximum is not a pure wire cost or an additive missing Frontier term. The rank with shorter observed AR spans measures 4.792544 ms, similar in scale to the profiler run's shorter-wait TP1 4.619221 ms, but neither establishes a production baseline minimum.

The attention AR measurements are stable across the four ranks (5.537248–5.708768 ms) and substantially below Frontier's 17.899440 ms prediction. In parallel, Frontier lacks the naive protocol's post-combine TP AR, and its dispatch/combine model differs from vLLM's broadcast/DP-reduction path. These opposing discrepancies require a collective-protocol and message/group-shape correction with matching measurements; adding a single positive constant would conceal the error direction.

Later selected batches remain diagnostic: DP0 has one mixed batch followed by two decode requests; DP1's first selected batch contains two prefill requests. First-three-per-DP selection does not establish common global-forward identity or three aligned logical batches. The final operator analysis therefore remains INSUFFICIENT_EVIDENCE for a full numerical gate; first-batch source and timing RCA evidence is available, and no speculative production correction was applied.

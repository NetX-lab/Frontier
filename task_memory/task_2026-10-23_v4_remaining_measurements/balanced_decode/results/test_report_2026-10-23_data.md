# Balanced q1 decode CPU data verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Independent immutable CPU verification artifacts. |

Task `W128-BALANCED-DECODE-DATA`; job `exp-1023-213905-916183`. **PASS for the scoped CPU integrity, raw/table reproduction and constructed routing identity checks. No data/native/root/serving admission.** Trace-wide attention ownership and device dependencies remain unqualified.

## Inputs and source binding

Runtime source `97803f07f72454030a4d3fe538d7cdf473906b8a`; builder `64133239c4f4c7fdc50b1b0b85d57d2d72fe3097`; merge `56e9405dad76d30d4ada9d5bdab124f898bf5065`. Manifest SHA-256 `9714cdb749e8504df32955bb62e080467434e33e20da1d23ae653da2ce8e09cf`. Original package: `/data/ycfeng/Frontier/.worktrees/v4-balanced-decode-run/task_memory/task_2026-10-23_v4_remaining_measurements/balanced_decode/h800_decode_balanced_02`. Local phase: `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_results_6zkg5vz4/artifacts/Q1_WARM_6143_7166_BALANCED/Q1_WARM_6143_7166_BALANCED`. Receipt: `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_results_6zkg5vz4/download_receipt.json`, SHA-256 `64922e4489ec717625e6ca66a1a786ce4b2f71827c0a35cc9ef4bb27bb6b9a67`. Local eight-file total is 1,601,865,978 bytes; all hashes match the root download receipt and remain unchanged before/after/final. Cloud hashes were not independently queried.

| Original file | Bytes | SHA-256 |
| --- | ---: | --- |
| `Q1_WARM_6143_7166_BALANCED/compute/h800/deepseek-ai/DeepSeek-V4-Flash/attention.csv` | 38983593 | `d51ce6a312c3ce108d10377125b2f8d2ecfd9368fa77f3094f911a8d48b685be` |
| `Q1_WARM_6143_7166_BALANCED/compute/h800/deepseek-ai/DeepSeek-V4-Flash/attention_kernel_only.csv` | 37164812 | `e92981d3fd5d910c7b333af5290cc0a8ac636a8a02ad4be2ca556d841e838399` |
| `Q1_WARM_6143_7166_BALANCED/logs/accepted_c26_q1_balanced.log` | 22628 | `e2cc1bc6e07b8c4c292ee90fc57509381ce1f9f65d990b4bb1567f856e92bb38` |
| `Q1_WARM_6143_7166_BALANCED/logs/aggregate_q1_balanced.log` | 689 | `88718a7b1bc9f5e0e58bd5ce57efa0ba6d7947bf2b265be520fc9f74741064bb` |
| `Q1_WARM_6143_7166_BALANCED/profiler_traces/e0-decode-q1-c6143-7166.json` | 1500192771 | `d1207a5e3879202fc5c41cedf897495a62d6c2c09cd03935fb7a5a58ed105855` |
| `Q1_WARM_6143_7166_BALANCED/raw/dsv4_aggregate_summary.json` | 2396 | `8a44630c4bbd7da55837391236c606777159925c0bb03da36d6a6d4d3c19a55b` |
| `Q1_WARM_6143_7166_BALANCED/raw/dsv4_raw_e0.jsonl` | 25497874 | `df1cff6e31fee49f06f16c660c8d477b1da7c6739bfcb9fb56ff8b75ce68dfa2` |
| `status.json` | 1215 | `ba13c4dda4dcb9884eb820e32ae1b3af7b2a1348c16165e45a2cfc940084df0f` |

## Acceptance predicates

| Predicate | Expected | Observed / result |
| --- | --- | --- |
| Measured keys | q1, C6143..7166 inclusive, six layers, both methods | PASS; exact context/layer/scope Cartesian sets; 1024 steps/method |
| Warmups | two unmeasured decode steps | PASS source/raw-window check; prompt 6141, skip=2, contexts 6141/6142 absent; skipped-step IDs not retained |
| Call counts | CUDA_EVENT 51200; KERNEL_ONLY 45056 | PASS; enclosing layer scope only in CUDA_EVENT explains the 6144 difference |
| Rows | 6144 per CSV; one sample/context/layer/scope | PASS; no missing/duplicate/malformed row or scope keys, zero duplicate CSV columns |
| Raw JSON fields | no duplicate/malformed fields | PASS; 98305 raw records parse with duplicate-field rejection |
| Costs/counts | finite positive raw time; positive integer KERNEL_ONLY count | PASS; event kernel_count is null; std=0 and count=1 for singleton scope stats |
| Every scope statistic | exact raw join | PASS; 307200 CUDA_EVENT and 315392 KERNEL_ONLY numeric cells exact |
| Full table reconstruction | all original CSV bytes equal | PASS; pinned builder plus identity rebuild both byte-identical CSVs |
| Routing identity | balanced_topk strategy, exact origin/SHA and native origins in every row | PASS; six constructed routers/runners, nine source pins, 12288 rows / 36864 routing cells |
| Trace scope reconstruction | 45056 annotations, exact raw time/count | PASS; every key, time and kernel count exact |
| Trace-wide kernel ownership | all kernels assigned to measured attention scopes | FAIL / not established; 227901 of 352829 kernels unowned by measured scopes |
| Actual event dependency edges | event handles/generations and stream edges | UNVERIFIED; synchronization args only External id/cbid/correlation |
| Serving or physical TP8 admission | not in this assignment | NOT ATTEMPTED; physical TP1/eager/six-layer isolated evidence only |

## Scope timings and sample/count coverage

All values are milliseconds. Each listed method/scope has 1024 retained context samples; each original table row has one sample, not 1024 repeated measurements at one context. Layers 0/2/3 plan their variant tile schedule; 1/4/5 reuse it. No nested scopes are summed. Full min/median/p95/p99/max, maxima contexts, top five points and every KERNEL_ONLY count distribution are in `data_audit.json`; all raw outliers remain intact.

| Layer / variant | Scope | CUDA_EVENT median / p95 / max | KERNEL_ONLY median / p95 / max | Kernel count per call |
| --- | --- | ---: | ---: | --- |
| 0 / uncompressed | `dsv4_attention_layer` | 0.869136 / 0.906720 / 1.478592 | not annotated | — |
| 0 / uncompressed | `dsv4_fused_q_kv_prepare` | 0.042080 / 0.045691 / 0.069440 | 0.002592 / 0.002752 / 0.002976 | 1×1024 |
| 0 / uncompressed | `dsv4_fused_qkv_rmsnorm` | 0.084240 / 0.088384 / 0.116928 | 0.001664 / 0.001856 / 0.002016 | 1×1024 |
| 0 / uncompressed | `dsv4_o_proj` | 0.265968 / 0.283808 / 0.316256 | 0.014433 / 0.014624 / 0.015488 | 4×1024 |
| 0 / uncompressed | `dsv4_parallel_input_projections` | 0.148256 / 0.156786 / 0.440000 | 0.008896 / 0.009058 / 0.009536 | 2×1024 |
| 0 / uncompressed | `dsv4_q_projection` | 0.089216 / 0.093421 / 0.176416 | 0.005921 / 0.006112 / 0.006560 | 2×1024 |
| 0 / uncompressed | `dsv4_sparse_attention` | 0.107072 / 0.114093 / 0.162304 | 0.042816 / 0.043104 / 0.043552 | 3×1024 |
| 1 / uncompressed | `dsv4_attention_layer` | 0.791632 / 0.818546 / 1.549664 | not annotated | — |
| 1 / uncompressed | `dsv4_fused_q_kv_prepare` | 0.036032 / 0.038112 / 0.062816 | 0.002912 / 0.003104 / 0.003360 | 1×1024 |
| 1 / uncompressed | `dsv4_fused_qkv_rmsnorm` | 0.085760 / 0.088096 / 0.103680 | 0.001664 / 0.001888 / 0.001952 | 1×1024 |
| 1 / uncompressed | `dsv4_o_proj` | 0.251168 / 0.264616 / 1.009728 | 0.014816 / 0.015136 / 0.015552 | 4×1024 |
| 1 / uncompressed | `dsv4_parallel_input_projections` | 0.103824 / 0.113298 / 0.158176 | 0.008832 / 0.009088 / 0.009344 | 2×1024 |
| 1 / uncompressed | `dsv4_q_projection` | 0.079488 / 0.086331 / 0.187264 | 0.006080 / 0.006304 / 0.006720 | 2×1024 |
| 1 / uncompressed | `dsv4_sparse_attention` | 0.106176 / 0.109659 / 0.145568 | 0.043166 / 0.043488 / 0.043936 | 3×1024 |
| 2 / csa | `dsv4_attention_layer` | 1.852448 / 1.978826 / 2.194624 | not annotated | — |
| 2 / csa | `dsv4_compressor` | 0.092992 / 0.104352 / 0.121984 | 0.003296 / 0.011840 / 0.012256 | 2×1024 |
| 2 / csa | `dsv4_fused_q_kv_prepare` | 0.038304 / 0.040896 / 0.051680 | 0.002624 / 0.002874 / 0.003008 | 1×1024 |
| 2 / csa | `dsv4_fused_qkv_rmsnorm` | 0.088608 / 0.091414 / 0.121152 | 0.001664 / 0.001856 / 0.002016 | 1×1024 |
| 2 / csa | `dsv4_index_prepare` | 0.305824 / 0.335034 / 0.401248 | 0.012544 / 0.017248 / 0.018080 | 5×1024 |
| 2 / csa | `dsv4_index_scores_topk` | 0.175968 / 0.197677 / 0.296416 | 0.012256 / 0.012608 / 0.013184 | 3×1024 |
| 2 / csa | `dsv4_o_proj` | 0.254048 / 0.267958 / 0.308704 | 0.014847 / 0.015137 / 0.015840 | 4×1024 |
| 2 / csa | `dsv4_parallel_input_projections` | 0.389552 / 0.425411 / 0.523072 | 0.031777 / 0.032320 / 0.033088 | 8×1024 |
| 2 / csa | `dsv4_q_projection` | 0.088128 / 0.096288 / 0.115872 | 0.005984 / 0.006336 / 0.006656 | 2×1024 |
| 2 / csa | `dsv4_sparse_attention` | 0.166240 / 0.178358 / 0.236608 | 0.050880 / 0.051488 / 0.051872 | 4×1024 |
| 3 / hca | `dsv4_attention_layer` | 1.131168 / 1.179291 / 1.320736 | not annotated | — |
| 3 / hca | `dsv4_compressor` | 0.121184 / 0.134637 / 0.145632 | 0.004896 / 0.005248 / 0.008192 | 3×1024 |
| 3 / hca | `dsv4_fused_q_kv_prepare` | 0.037504 / 0.040475 / 0.050688 | 0.002912 / 0.003136 / 0.003392 | 1×1024 |
| 3 / hca | `dsv4_fused_qkv_rmsnorm` | 0.089056 / 0.097723 / 0.121952 | 0.001664 / 0.001856 / 0.001984 | 1×1024 |
| 3 / hca | `dsv4_o_proj` | 0.256384 / 0.268872 / 0.355520 | 0.014848 / 0.015169 / 0.015808 | 4×1024 |
| 3 / hca | `dsv4_parallel_input_projections` | 0.230144 / 0.244466 / 0.417664 | 0.016640 / 0.016992 / 0.017439 | 4×1024 |
| 3 / hca | `dsv4_q_projection` | 0.086064 / 0.092155 / 0.105632 | 0.006016 / 0.006272 / 0.006751 | 2×1024 |
| 3 / hca | `dsv4_sparse_attention` | 0.121632 / 0.126747 / 0.144512 | 0.049409 / 0.049795 / 0.050208 | 3×1024 |
| 4 / csa | `dsv4_attention_layer` | 1.834496 / 1.944104 / 2.107456 | not annotated | — |
| 4 / csa | `dsv4_compressor` | 0.092576 / 0.102998 / 0.115680 | 0.003264 / 0.011643 / 0.012064 | 2×1024 |
| 4 / csa | `dsv4_fused_q_kv_prepare` | 0.038112 / 0.040832 / 0.129088 | 0.002528 / 0.002688 / 0.002816 | 1×1024 |
| 4 / csa | `dsv4_fused_qkv_rmsnorm` | 0.089888 / 0.093234 / 0.129760 | 0.001664 / 0.001856 / 0.002016 | 1×1024 |
| 4 / csa | `dsv4_index_prepare` | 0.304672 / 0.329150 / 0.467648 | 0.012576 / 0.017121 / 0.017953 | 5×1024 |
| 4 / csa | `dsv4_index_scores_topk` | 0.176672 / 0.193530 / 0.301600 | 0.012289 / 0.012672 / 0.013056 | 3×1024 |
| 4 / csa | `dsv4_o_proj` | 0.252800 / 0.267291 / 0.312352 | 0.014848 / 0.015072 / 0.015519 | 4×1024 |
| 4 / csa | `dsv4_parallel_input_projections` | 0.372032 / 0.397315 / 0.535104 | 0.031969 / 0.032544 / 0.033281 | 8×1024 |
| 4 / csa | `dsv4_q_projection` | 0.089664 / 0.098016 / 0.150656 | 0.005952 / 0.006304 / 0.006720 | 2×1024 |
| 4 / csa | `dsv4_sparse_attention` | 0.164864 / 0.175739 / 0.222688 | 0.050911 / 0.051520 / 0.052130 | 4×1024 |
| 5 / hca | `dsv4_attention_layer` | 1.124432 / 1.169568 / 4.143904 | not annotated | — |
| 5 / hca | `dsv4_compressor` | 0.121248 / 0.132530 / 0.188480 | 0.004896 / 0.005248 / 0.008000 | 3×1024 |
| 5 / hca | `dsv4_fused_q_kv_prepare` | 0.037216 / 0.039840 / 0.070496 | 0.002976 / 0.003168 / 0.003328 | 1×1024 |
| 5 / hca | `dsv4_fused_qkv_rmsnorm` | 0.088160 / 0.091355 / 0.107616 | 0.001664 / 0.001856 / 0.002016 | 1×1024 |
| 5 / hca | `dsv4_o_proj` | 0.256864 / 0.268715 / 0.408352 | 0.014689 / 0.014976 / 0.015456 | 4×1024 |
| 5 / hca | `dsv4_parallel_input_projections` | 0.224176 / 0.237288 / 0.319040 | 0.016641 / 0.016992 / 0.017632 | 4×1024 |
| 5 / hca | `dsv4_q_projection` | 0.085216 / 0.091954 / 2.873088 | 0.006048 / 0.006240 / 0.006752 | 2×1024 |
| 5 / hca | `dsv4_sparse_attention` | 0.121664 / 0.127200 / 0.245536 | 0.049376 / 0.049728 / 0.050113 | 3×1024 |

## Trace census and precise limits

Supported retained parser `load_trace_events` + `aggregate_kernel_trace` read all 5,177,456 events. There are 45,056 unique nonoverlapping measured annotations, 124,928 uniquely scoped kernels and 227,901 kernels outside measured scopes. All 352,829 kernels have exactly one launch correlation. Unowned regions: 1,163 before first annotation, 226,705 between first/last, 33 after last. These positions do not classify a kernel as attention, FFN, warmup or priming. 73,463 kernels have no External-id CPU-op parent. Five kernel streams are observed (7/21/25/29/33); no critical path is inferred. Synchronization calls: 47,733 event records, 46,187 stream waits, 1,027 event synchronizations, three device synchronizations; event handles/generations and wait-stream identity are absent. Source synchronization semantics are not runtime dependency edges.

No full attention/forward annotation or retained request/warmup identity supplies a completeness denominator. CUDA_EVENT and KERNEL_ONLY are separate instrumented passes. No CPU gap, serving total, graph equivalence, full-layer/TP8/DP2/EP16 equivalence or comparison to other routing variants is inferred. Router tensors/expert-ID values are not retained: the balanced assignment formula is source-backed, while numerical dispatch balance is not independently observed. Root retains platform terminal, actual installed package/device/constructor checks and final admission.

## Commands, execution and failure handling

`command.md` retains full exact commands, both temporary scripts and execution logs. Raw/CSV: Python 3.10.6 at `/data/ycfeng/envs/frontier-py310/bin/python`, 17.675 s, peak RSS 932204 KiB. Trace: `/usr/bin/python3` 3.10.6, 49.260 s, peak RSS 6062156 KiB. Each used `sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s`, with user-owned output via `runuser -u brainpp`; no conda activation or dependency install. Initial commands exited127 because `/usr/bin/time` is absent, before analysis started. Removing only that wrapper resolved it; caps were unchanged. Final duplicate-field/hash checks use the same cap. No Torch/vLLM import in either analysis. The existing experimental-architecture warning concerns simulator support; no simulation/admission ran here.

## 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 原始数据、统计表、routing 身份 | PASS | 两表字节重建和全部 keys 精确一致 |
| 有界完整 trace 检查 | PASS | 49.260 秒内完整解析，明确未归属 kernel 和依赖证据缺口 |
| 三个指定产物交付 | PASS | 本次生成且仅占用指定 results 三文件 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 当前 CPU 分配开始至今｜**当前前沿：** 产物交给 root，保持 admission 边界

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只复用所需函数和临时适配器，无维护工具扩张 |
| D2 细节反复重试 | CLEAR | 缺少 time 二进制一次修正，caps 不变 |
| D3 支线问题修补 | CLEAR | 无 source/platform 修补 |
| D4 可并行任务停滞 | CLEAR | raw/table 与 trace 独立并行执行 |
| D5 超范围镀金 | CLEAR | 只交付指定三文件 |
| D6 分析空转 | CLEAR | 每项检查产出确定结果或明确证据限制 |
| D7 路线违背 | CLEAR | 无求和、CPU gap、serving 推断或 admission |

**文档检查：** 本次明确分配的 CPU 验证范围内，S1–S6 均未发现明显缺陷；父记录的平台状态和后续接受仍由 root 更新。
**归因：** 无

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动

**需停止的偏离行为：** 无

**接下去的路线：**
1. Root 核对三文件及实际 terminal/package/device/constructor 证据 → 完成标志：独立 root 接受记录，不自动提升 data/native/serving admission。

## 4. 最值得关注的 3 点

1. 94 个 scope series 的原始统计和两表完整字节重建一致。
2. 227901 个 kernel 不属于已测 scope；唯一 correlation 不等于完整 attention 覆盖。
3. 无 event-handle 依赖边、TP8/graph/serving 证据；后续判定由 root 负责。

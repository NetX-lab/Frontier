# W128-MOE01-DATA local data audit

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Audited both four-row tables, all retained traces, source/runtime provenance, and measurement limits using local CPU reads. |

## Outcome and ownership

`PASS_LOCAL_DATA_AUDIT_WITH_LIMITS`. No cost admission or simulator consumption. Only `moe/results/{data_audit.json,command.md,test_report_2026-10-24_data.md}` were delivered; adapters remain temporary under `/data/ycfeng/tmp/glm53_dsv4/W128-MOE01-DATA/`.

Task `W128-MOE01-DATA`; session `ses_e966f8226ffepbK7SMlWxJ24hG`. Primary owns platform/resource/terminal verification and final acceptance. The supplied RJob `exp-1023-235250-540507` / Succeeded `2026-10-23T15:54:35Z` was not independently queried here.

## Environment, commands, expected and actual results

Exact successful commands and archived producer argv are in `command.md`. Local audit: `/usr/bin/python3`, Python `3.10.6 (main, Nov 14 2022, 16:10:14) [GCC 11.3.0]`, NumPy `1.24.3`, Linux `5.10.25-nvidia-gpu`, glibc `2.35`; no conda environment activation. `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, 60-second timeout and system scope `MemoryMax=30G`. Final audit took `1.1817206754349172 s`, peak RSS `40724 KiB`; Torch/vLLM were not imported.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Local input integrity | Exact downloaded file set, size and SHA256 match inventory | 19/19 files; 3,261,347 bytes; all hashes match | PASS |
| Source/configuration binding | Sealed manifest; audited producer/config/native API bytes match pins | Manifest matches root seal; 12 producer/config sources and 4 native API sources match; producer/config copies equal current worktree | PASS |
| Grid and metadata | Four rows/method; M=12288,6152,32,16; physical TP1; EP16 rank0 adapter; local16/global256; top-k6 | Exact rows; balanced + balanced_topk + uniform routing weights; MXFP4/MARLIN; BF16 activation; eager | PASS |
| Finite statistics | All timing values finite/nonnegative; mean/median between min/max | Both CSVs pass; CUDA_EVENT diagnostic counts=20 | PASS |
| Routing-feature reconstruction | Rank0 assignment totals 4608/2320/16/16; block M64/64/8/8; padded rows5120/3072/128/128 | Exact existing-feature/Marlin-source reconstruction; not a runtime routing-array readback | PASS, source arithmetic only |
| Trace statistics | All 16 operator/M summaries, 112 mean/std/min/max/median/count/kernel-count fields reproduce CSV within 1e-12 ms | 12 traces; every compared field difference=0.0 | PASS |
| Sample/ownership retention | 80 grouped-GEMM scopes, 20/M; 7 kernels/scope; 12 diagnostic scopes; no missing launch correlation or duplicate kernel ownership | 92 scopes retained; 560 grouped-GEMM kernels; zero detected missing launch correlation or duplicate ownership | PASS |
| Runtime provenance | Retained worker facts and GPU kernel evidence, not CPU labels alone | H800/sm90; world_size1/rank0; MARLIN selector logs and actual Marlin/BF16/clamp kernels | PASS within retained evidence |
| CUDA_EVENT grouped-GEMM count/raw samples | Recompute count and aggregate statistics from retained individual samples if available | Count column absent; individual CUDA event samples absent; configured20 only | NOT VERIFIABLE from retained data |

Two unsuccessful scope starts ran no adapter and produced no audit. Their exact commands were:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 timeout 60s systemd-run --user --scope -p MemoryMax=30G /usr/bin/python3 /data/ycfeng/tmp/glm53_dsv4/W128-MOE01-DATA/audit.py /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/moe/results/data_audit.json
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 timeout 60s systemd-run --scope -p MemoryMax=30G /usr/bin/python3 /data/ycfeng/tmp/glm53_dsv4/W128-MOE01-DATA/audit.py /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/moe/results/data_audit.json
```

Both exited1: respectively `Job failed. See "journalctl -xe" for details.` and `Failed to start transient scope unit: Interactive authentication required.` The existing non-interactive sudo system-scope route then succeeded, retaining the memory limit and running the adapter as `brainpp`. No systemd configuration was changed. Initial bounded adapter was190 lines; final provenance/count/launch checks brought it to209 lines. No persistent checker or source/test fix was introduced.

## Grouped-GEMM measurements

All values are **ms**. JSON retains full float values and every original CSV column name (71 CUDA_EVENT,76 KERNEL_ONLY); this table is rounded for readability. These are separate measurement modes, not predictions versus a common ground truth.

| M | Method | Mean | Population std | Median | Min | Max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | CUDA_EVENT | 0.211558400095 | 0.012457358605 | 0.205967999995 | 0.203712001443 | 0.252032011747 |
| 16 | KERNEL_ONLY | 0.101793700000 | 0.000425058596 | 0.101855000000 | 0.100862000000 | 0.102495000000 |
| 32 | CUDA_EVENT | 0.212177599967 | 0.010374923299 | 0.208176001906 | 0.205824002624 | 0.251616001129 |
| 32 | KERNEL_ONLY | 0.102299700000 | 0.000688390086 | 0.102174500000 | 0.100992000000 | 0.103615000000 |
| 6152 | CUDA_EVENT | 0.849601599574 | 0.007721762620 | 0.846448004246 | 0.839519977570 | 0.866815984249 |
| 6152 | KERNEL_ONLY | 0.755743400000 | 0.001500780843 | 0.755544500000 | 0.753430000000 | 0.758712000000 |
| 12288 | CUDA_EVENT | 1.385188794136 | 0.014395258490 | 1.380128026009 | 1.368384003639 | 1.419551968575 |
| 12288 | KERNEL_ONLY | 1.298690450000 | 0.002736029870 | 1.298337000000 | 1.293871000000 | 1.303282000000 |

Every KERNEL_ONLY grouped-GEMM row has observed `count=20`, `kernel_count=7`; all20 samples, including extremes, contribute to unchanged parser population statistics. CUDA_EVENT grouped-GEMM source loops20 times but removes count when returning its aggregate; neither these raw event samples nor individual diagnostic CUDA_EVENT samples were exported. Thus their observed aggregates are finite, but raw-event recomputation/all-sample-retention cannot be independently proven. No filtering/relabeling of retained inputs was performed.

| M | Grouped-GEMM trace | Gating trace | Standalone shuffle trace |
| ---: | --- | --- | --- |
| 12288 | `profiler_trace_24f37cba.json` | `profiler_trace_f6942362.json` | `profiler_trace_fec80bd5.json` |
| 6152 | `profiler_trace_87536bfa.json` | `profiler_trace_a3d9161f.json` | `profiler_trace_8e6282a6.json` |
| 32 | `profiler_trace_f2cbcb61.json` | `profiler_trace_ef6c36d4.json` | `profiler_trace_334d2f50.json` |
| 16 | `profiler_trace_0c4d4e65.json` | `profiler_trace_d20cc751.json` | `profiler_trace_4687a04a.json` |

Mapping is a unique match of all parser statistics, supported by chronological four-point producer order. Traces do **not** carry independent tensor-shape/M labels. JSON binds each raw file by SHA256 and provides annotation event indices, ordered sample-summary hashes, kernel names/counts and grids, without copying full trace arrays.

## Repeats, scopes and actual provenance

Source references below are under `moe/h800_moe_local_01/source/56e9405dad76d30d4ada9d5bdab124f898bf5065/`; native references are under `moe/h800_moe_local_01/native_source/2cf0a6915ce544dc493a0990f2ea38d81601128a/`. Exact absolute paths/hashes are in JSON.

- `frontier/profiling/moe/moe_wrapper.py:46-47,433-575,689-726` specifies two CUDA_EVENT warmups/20 active calls per operator. KERNEL_ONLY gating and standalone shuffle each have one source-specified warmup and **one observed call**; their std=0 is a one-sample statistic, not evidence of zero variance. Grouped-GEMM has two source-specified warmups/20 observed active calls (`moe_vllm_kernel.py:682-732,1052-1073`). No additional warmup sample is inferred from the traces.
- `RecordFunctionTracer:28-54,95-175` primes four tiny kernels outside every measured range, flushes CUDA at session end, sums correlated `cat=kernel` durations, and uses NumPy population std/median. Grouped-GEMM traces each contain144 kernels:140 attributed +4 priming. All attributed kernels are on stream7 with graph id0. KERNEL_ONLY launches20 calls in one profiler session; CUDA_EVENT synchronizes each event-timed call. Their different timing/cadence does not establish a CPU-gap decomposition.
- Actual seven-kernel envelope: integer fill, count/sort, block alignment, two `marlin_moe_wna16::Marlin` kernels, `_swiglu_limit_pad_aware_kernel`, and `moe_sum_vec_kernel<c10::BFloat16,int,6,true>`. Native `marlin_moe.py:135-229,333-397` explains the same operations, including clamp and expert-map-aware sum. MXFP4/MARLIN is supported by selector logs, operator metadata and executed kernel lineage; the CPU source labels alone would not prove execution.
- `mxfp4_marlin_experts.py:86-111,155-222` uses the actual native MXFP4 oracle/class and global256/local16 expert map with `apply_router_weight_on_input=False`. Runtime logs report compatibility API `functional_fused_experts`; this checkpoint path directly calls selected `MarlinExperts.apply`, not a serving dispatch. Weight content is random packed FP4 with unit E8M0 scales; raw weights/output tensors are not retained, so numerical/timing weight parity is unknown.
- Retained `status.json`: Python3.12.3, Torch2.13.0+cu130, vLLM0.28.0, FlashInfer0.6.16.post3, pandas2.3.3, CUDA available, one NVIDIA H800. CSV additionally reports CUDA13.0/Triton3.7.1; fingerprint `1ef9d0c19c7ef69f` exactly recomputes using unchanged `RuntimeIdentity`. Native trace properties independently show sm90/H800/132SMs and distributed world_size1/rank0. This is a **single-GPU EP16 rank0 local-expert adapter**, not a running16-rank EP group.
- Source revision `56e9405d...`; model revision `0d992cbe...`; model JSON SHA256 `b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818` match sealed inputs and CSV. Its `expert_dtype=fp4`/`swiglu_limit=10.0` are separate from model-level FP8 `quant_signature`; the console BF16/FP8 summary is not the routed-weight runtime identity. Worker source sets snapshot PYTHONPATH but does not rehash imported source or binaries during the run. Installed-image/native-binary validation is root's prelaunch evidence, not independently repeated runtime evidence here.
- Logs retain default standalone MoE-config warning, experimental model-profile limitations and `destroy_process_group()` shutdown warning. Both producer phases report success and output files; these warnings are not suppressed and do not prove serving equivalence. Platform terminal qualification remains primary-owned.

## Qualification boundaries

Only the existing rank0 grouped-GEMM envelope is a **candidate**. Workspace/output allocation precedes its producer timer (`moe_vllm_kernel.py:971-982`); the native serving step may include allocation. Gate receives producer M, whereas native gate receives L=M/16 for equal shards, or768/1 for mixed M6152. Standalone shuffle times a separate alignment call with its own heuristic (`moe_impl.py:368-407`) and overlaps grouped-GEMM alignment. **Do not sum gate/shuffle/grouped-GEMM or label the CUDA_EVENT minus KERNEL_ONLY difference CPU overhead.**

Rank0 source-reconstructed local assignment totals are16,16,2320,4608 at M16,32,6152,12288; `total_routed_tokens` is this local count, not6M. M16/M32 have respectively6/12 active EP ranks; rank0 measurements do not price inactive or other ranks. M6152 is a mixed-prefill/dummy population, not established steady-state serving. Routing tensors were not retained: balanced index algebra and CSV features match source, but there is no independent runtime histogram readback.

Not established: TP8, other EP ranks, EP all-gather/reduce-scatter costs, graph or serving execution, complete model/weight equivalence, multi-layer additivity, or blanket cost admission. No consumer/admission/input changes were made.

## Input/output hashes

Local input root: `/data/ycfeng/tmp/glm53_dsv4/moe01_results_6uzgd6nx/artifacts`. All19 per-file SHA256/bytes and raw trace references are in `data_audit.json`; inventory SHA256 `267a4fa777d25983684b6968c6e74050b0e0e0a1e55f85ca847ab42bbc5b88e0` independently matches download receipt. CSV hashes:

| File | SHA256 |
| --- | --- |
| CUDA_EVENT `moe.csv` | `ad661c7cbedc7c857185eafc861785d0f449e3de1abb055d22aaaa9abb899447` |
| KERNEL_ONLY `moe_kernel_only.csv` | `3ede04c6fee2efd4a15acfc489633e68f45f7616caeb963e8c72e39da6f17f46` |
| Delivered `data_audit.json` (170,752 bytes, below1MiB) | `6fdf1b7f9285660b2a6140fa56a79bae353045ae9647efa8fccec5b1c378cba9` |
| Delivered `command.md` (3,192 bytes) | `8142c4e80bc1c967ccea821c9fa8adc80fde2a61e044d73e01f3b3db04ed2566` |

Final integrity receipt `/data/ycfeng/tmp/glm53_dsv4/W128-MOE01-DATA/output_hashes.json` records all three output hashes, including this report's hash (kept externally to avoid a self-hash cycle), and rehashes42 input/source/binding references. Expected and observed result: `PASS_FINAL_LOCAL_INTEGRITY`,42 references and3 delivered files. Requirements were re-read before closure. The same integrity command is rerun after this final report update to bind its final bytes.

## 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 0｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 输入与 provenance 核验 | PASS | 19文件哈希、两份四行CSV及源/config/runtime绑定已核验 |
| 原始 trace 统计核验 | PASS | 112字段差值0；80次grouped-GEMM与全部diagnostic scope保留 |
| 三文件交付与最终 integrity | PASS | 三文件已交付；42引用重验PASS，最终报告字节再次绑定 |
| Primary独立接受 | TODO | 平台/resource/terminal与最终qualification归primary |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** W128-MOE01-DATA启动至最终integrity完成｜**当前前沿：** Primary独立接受

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 临时209行adapter只核验选定数据，无持久化工具 |
| D2 细节反复重试 | CLEAR | 两次scope启动权限失败已通过既有system scope解决；未改数据/行为 |
| D3 支线问题修补 | CLEAR | 未修复源、测试或运行时问题 |
| D4 可并行任务停滞 | CLEAR | 此委派限定单一数据审计，禁止进一步委派 |
| D5 超范围镀金 | CLEAR | 只写指定三文件 |
| D6 分析空转 | CLEAR | 统计重建和边界结论已生成JSON |
| D7 路线违背 | CLEAR | 本地CPU读取；未执行平台/GPU/网络/Git/准入操作 |

**文档检查：** 当前委派S1–S6均未发现明显缺陷；包内CPU-preparation标记及旧progress是历史快照，不作为当前run verdict。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动。
**需停止的偏离行为：** 无。
**接下去的路线：**
1. 最终报告更新后的相同integrity命令只绑定最终字节 → receipt为PASS。
2. 交回primary → 完成独立接受；不展开新分析或消费改动。

## 4. 最值得关注的 3 点

1. KERNEL_ONLY统计全量精确重建；CUDA_EVENT原始样本/count证据仍缺。
2. Gate的M与native L不等，standalone shuffle重复alignment；不得相加。
3. 单卡rank0、M6152 dummy和随机权重均不构成TP8/EP通信/serving等价或成本准入。

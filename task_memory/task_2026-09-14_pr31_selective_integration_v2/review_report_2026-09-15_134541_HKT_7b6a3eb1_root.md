## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-15 | Created an independent, read-only review of implementation, verification, main-branch parity, performance evidence, and maintainability at 7b6a3eb1. No fixes or tests were executed. |

# PR31 Selective Integration v2 独立审阅报告

boss YC，**代码主体已提交，但现有实现和证据不足以支持“plan 已全部完成、CPU 验收已闭合、与 main 无额外性能变化”的结论。**本报告确认两个可由生产调用链确定的 CPU 问题、未接线的 GDN state lifecycle，以及矩阵比较和性能分析缺口。报告中的修复和补测均为后续建议，本次没有执行。

## 0. 审阅身份、范围与结论

### 0.1 固定版本与执行边界

| 项目 | 固定值 |
| --- | --- |
| Worktree | `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn` |
| Branch | `feature-amd-sglang-gdn` |
| Candidate HEAD | `7b6a3eb192e7d911219fd942fbc55121c4b0ea54` |
| 本地 main / origin/main | `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2` |
| Merge-base | 同上述 main |
| Diff | `git diff 0515589ac7f49ac5288a5f55b0ce38b0ede29bb2...7b6a3eb192e7d911219fd942fbc55121c4b0ea54` |
| 分支增量 | 22 commits；123 files；+13,073 / -714 lines |
| 审阅记录时间 | 2026-09-15，Asia/Hong_Kong；报告标识 `134541_HKT_7b6a3eb1_root` |
| 主审 | `/root`：任务完成度、证据、矩阵、性能、发现复核与报告 |
| 独立 Standards 审阅 | `/root/standards_review` |
| 独立 Spec 审阅 | `/root/spec_review` |
| 工作方式 | 按 code-review skill 分别审阅 Spec / Standards，主审再核对实际代码与产物 |

**本次只做了** Git/文件读取、文本搜索、现存 JSON/CSV/log 解析、已有测量值的算术汇总，以及创建本报告。没有执行 pytest、simulation、profiling、benchmark、compileall、GPU/Docker 作业或任何修复；没有 fetch/push/merge。main 指本次可读的本地 refs，未联网核实远端是否更新。

本报告是独立新文件；没有合并其他 session 的 review，也没有改写该 task 原有 requirements、plan、progress、review、summary 或 test reports。代码行号以固定 candidate 为准。少数未修改文件用于证明新增代码的生产调用者和配置默认值。

### 0.2 按用户四项要求给出结论

| 用户要求 | 审阅结论 | 关键依据 |
| --- | --- | --- |
| 1. 是否完成 plan 的代码修改与实现 | **主体已落地，不能判定全部完成。** | GDN 自动容量规划未传 concurrency；state-slot 类未接 lifecycle；MLA 被新 resolver 误分类；ExecutionTime 双语义仍存在。详见 SP-01～SP-04、SP-07。 |
| 2. 关键模块是否完整测试 | **已有大量 focused/unit evidence，但关键生产边界存在漏测。** | 最新 raw unit log 为 3276 passed / 19 failed / 25 skipped；两个 hybrid E2E 均用固定 num_blocks，实际 fixture 缺 shared expert，continuation/TP>1/lifecycle 未完成整条链验证。 |
| 3. 是否充分对比 main 的 E2E 矩阵并分析/修复变化 | **没有充分比较；已记录的 slowdown 没有完成归因和关闭。** | 唯一早期跨 worktree fidelity 仅 5 个 offline dummy cases；final fidelity 是 candidate self-parity；现有 harness 有 58 场景。旧小样本 event loop 中位数为 9.419745×，缺少后续分析和最终版本复测。 |
| 4. 是否冗余、过重、防御过度、hard-coding 或临时补丁 | **存在具体问题，不能用总行数替代判断。** | 可达 legacy wrapper、为 test sentinel 设置的生产旁路、重复 family 分支、三份 DEVICE_EVENT path 解析、通用且无淘汰的 query cache、缺少大模块拆分记录。详见 Standards。 |

### 0.3 证据与严重程度

- **源码确定**：由已阅读的分支条件和生产调用链推出；本次未运行复现。
- **产物观察**：从此前保留的 manifest/results/log/metrics 中读取；不是本次新增测试结果。
- **报告记载**：旧 Markdown 的声明；缺少 raw evidence 时不提升为独立验证。
- **风险推断**：代码结构可能造成的性能或维护后果；未用 profile 证明的因果关系明确标注。

P1 表示阻止核心功能/既有行为保持或关键验收关闭；P2 表示局部契约、覆盖或维护问题。没有按一个统一排序混合 Spec 与 Standards；两个轴分别列出。AMD hardware SKIP 本身符合原授权边界，不作为 CPU 实现失败。

审阅基准：

- [requirements.md](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/requirements.md)
- [本地执行 plan](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/plan.md)
- [原始只读 plan v2](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md)
- [AGENTS.md Development Gates](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/AGENTS.md:749)

## 1. Spec

### 1.1 逐 work unit 核对

“主体可定位”表示存在相应实现与针对性记录，不代表所有 GPU 行为已实测。测试数量是各次 suite 的历史结果，存在重叠，不能相加当作独立覆盖总数。

| Unit | 实际实现/提交 | 现存验证证据 | 本次判断 |
| --- | --- | --- | --- |
| 0 | baseline 0515589a、source 215a80a2、Review 1 已记录 | baseline report：3144 passed / 19 failed / 25 skipped；self-parity；dense/MoE CSV smokes | 基础记录存在；baseline report 的部分命令仍含占位符，未提供完整原始失败清单/log 路径。 |
| 1 | e8ac59ae；attention_linear_ops 等内部 rename | progress 记录 84 focused passed | 主体可定位；upstream layer-type literal 保留。 |
| 2 | 0989f0ed；MI355X SKU/platform/topology/discovery | accelerator 12 passed；CPU mismatch/error contracts | CPU 合同主体完成；AMD discovery/runtime 未执行。 |
| 3 | 49fcd9b8；DEVICE_EVENT、timer、measurement/path families | focused 43 passed；legacy 84 passed；后续 timer/path tests | 主体完成；path helper 重复与行为分歧见 ST-05；CUDA GPU regression 证据缺口见 SP-10。 |
| 4 | ddd9bbb3、57da3683；hybrid resolver、GDN config/features、fixture | combined 134 passed；semantic 12 passed；Qwen3.8/Qwen3-Next structural pins | 部分完成；homogeneous MLA 分类回归，见 SP-02。 |
| 5 | b37cb38b；StageExecutionTime 及 dense/MoE/disaggregation migration | 89 focused；5-case fidelity；3+3 wall-time observations；后续 metrics tests | 部分完成；旧 aggregate 表示可达；完整矩阵和 slowdown 关闭未完成。 |
| 6 | 0d1b87a4；fixed state bytes、参数估算、guard、slot helper | memory/state/guard units；preemption 8/8；3206 passed / 19 failed 全量记录 | **未完成生产接线**：自动容量规划失败、slot lifecycle 未接线，见 SP-01/SP-03。 |
| 12 | 5764daf7；共享 routing-ratio helper | 9 focused、62 predictor、94 conservation/disaggregation；记录 48/48 old-formula equivalence | 主体可定位；后续 actual Replica ID repair 有代码和测试。 |
| 8 | fd5f49b1；GDNTrainer/GDNPredictor/manager/artifact identity | 33 passed；6 个 estimator；fresh load、dataset fingerprint rejection | 主体及 CPU artifact 路径已实现；它不单独证明完整 scheduler 生命周期。 |
| 14A/B | 45e66170、31b11762、2bdd96f2、7b6a3eb1；layer dispatch/真实 constructor E2E | 76+116 suites；220 focused 最终报告；synthetic real constructor 完成 1 request | **存在真实 fit/load/Simulator 链，验收场景仍不完整**，见 SP-05。 |
| 7 | 0d4efa22；标准 vLLM GDN profiler/CLI/schema | 28 focused；input、schema、import；wrapper 内保留 decomposition comparison | 部分完成；phase fallback 与 retained runtime test 缺口见 SP-04/SP-06。 |
| 9 | 294bc1b4；VLLM_ROCM attention/generic compatibility | 51 focused；sequence plan、lazy import、metadata | source/CPU contracts 存在；AMD 未执行；无 grouped NVIDIA regression 或具体 access SKIP 证据。 |
| 10 | 3b20d702；标准 MoE/AITER/MXFP4 | combined 63 passed；packed layout/quant mode/metadata mocks | source/CPU contracts 存在；真实 MXFP4 packing/kernel/backend 未执行。 |
| 11 | a092f637；standalone NCCL/RCCL collective runner | 9 focused、29 combined；dtype byte accounting/runner mocks | source/CPU contracts 存在；真实 process group/collective 未执行。 |
| 13 | f51231d4、31b11762；experimental SGLang primitives/replay/trace | 18 combined、7 focused、CPU graph orchestration | experimental 边界基本清晰；旧 tuple concern 已修复；HIP/SGLang runtime 未执行。 |
| 14C/D | 31b11762、97c1a737、ffb9feea 等；Review 2、单位回归、文档 | 56/91/15 focused；3276 passed raw full-unit log | **不能关闭**：当前 main/HEAD 全矩阵、performance 归因与 CPU integration 缺口仍存在。 |
| Docs/PR | 7937e534；focused docs 和 attribution trailers | 本地记录声明 PR #33 已创建；当前 branch tracking 同步 | 本地交付记录存在；本次未查询 PR 在线状态。无需执行任何发布或 merge。 |

### 1.2 SP-01 · P1 · GDN 自动容量规划没有接入 max_num_seqs

**要求**：[原 plan Increment 6](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:1221) 要求将 actual stage GDN state bytes/request 乘以 max_num_seqs，从 KV budget 扣除；[容量约束](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:1291) 禁止为了跑通而跳过容量计算。

**代码链**：

1. [BaseReplicaScheduler 构造 MemoryPlanner](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/scheduler/replica_scheduler/base_replica_scheduler.py:61) 仅传 replica_config、replica、cluster_type。
2. [VllmV1SchedulerConfig 默认配置](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/config/config.py:602) 为 num_blocks=0、num_blocks_mode="memory_planner_profiled"。显式选择 "memory_planner" 同样属于自动规划模式。
3. [自动规划调用](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/scheduler/replica_scheduler/base_replica_scheduler.py:240) 进入 get_num_blocks()。
4. [MemoryPlanner](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/scheduler/utils/memory_planner.py:197) 对 GDN state>0 且 _max_num_seqs is None 抛出 `ValueError("MemoryPlanner requires max_num_seqs to reserve fixed GDN state")`；[get_num_blocks](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/scheduler/utils/memory_planner.py:226) 会调用它。

**触发条件**：合法 hybrid/GDN 模型，使用 VllmV1 自动 block 规划，并具备正常其他输入。这里的问题与 AMD hardware 是否存在无关，也不是 D57 下允许的真实 OOM。

**测试为何未发现**：[memory unit](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_gdn_increment6_memory.py:121) 手动传 max_num_seqs=4；两条 [prepared E2E](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_gdn_hybrid_e2e_increment14ab.py:555) / [production constructor E2E](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_gdn_hybrid_e2e_increment14ab.py:810) 都设 num_blocks=100，跳过 get_num_blocks()。固定 blocks 可以用于局部控制流测试，但不能验收自动容量计算。

**建议**：从真实 scheduler concurrency 配置传入 reservation capacity，并用 num_blocks=0 的小 fixture 验收预留量和剩余 KV blocks；超容量仍保留普通 OOM。不要新增 fallback blocks。结论置信度：高，源码确定；本次未复现执行。

### 1.3 SP-02 · P1 · 非 Qwen3.5 的 per-layer resolver 破坏现有 MLA 路由

**要求**：[原 plan](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:132) 保留既有行为；[最终定义](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:2257) 要求新 profile 原生建模并保持既有模型语义。

**代码链**：

1. [resolve_layer_attention_specs](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/attention/gdn/config.py:290) 对所有非 Qwen3.5 config 直接产生 `LayerAttentionSpec(..., "dense_attention", ...)`。
2. 实际 [BaseModelConfig](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/config/model_config.py:544) 全部具有 get_layer_attention_spec 方法。
3. [predict_attention_layer_time](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_execution_time_predictor.py:7491) 发现该方法后调用 [bind_layer_attention](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/attention/model_binding.py:249)，后者无条件调用上述 resolver。
4. 真实 MLA config 因此跳过 [latent_mla 分支](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_execution_time_predictor.py:7516)，进入 dense attention prediction。
5. 同时 model-wide binder/训练仍按 MLA 选择 schema。正常只有 MLA keys 的 profile/cache 无法提供 dense attn_prefill/attn_decode；相关错误边界见 [prefill cache 检查](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_execution_time_predictor.py:6525) 和 [decode cache 检查](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_execution_time_predictor.py:6461)。

**现有测试盲区**：[MLA runtime fixture](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_mla_predictor_runtime_operator_times.py:170) 是不含 get_layer_attention_spec 的 SimpleNamespace，所以回到旧 binder。这解释了对应 focused tests 可以通过，而真实 BaseModelConfig 调用链仍错误。

**影响界限**：已确定 family 选择错误；正常 MLA 数据将触发错误 lookup 或错误算子路径。没有实测 latency 差值。DSA/exotic 同样被 resolver 标成 dense，但其他初始化 guard 可能提前拦截，故不声称本次已证明 DSA 整体 guard 被绕过。

**建议**：homogeneous layer spec 从现有 authoritative binder 获取 family/variant；Qwen3.5 仅负责其显式 GDN/full-attention schedule。增加真实 MLA BaseModelConfig 到 predictor 的验收，不能继续仅依赖缺新 API 的 test double。置信度：高，源码确定。

### 1.4 SP-03 · P1 · GDN state-slot lifecycle 尚未接入 Simulator

**要求**：[原 plan state lifecycle](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:1240) 与 [CPU E2E 要求](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:1825)：admitted/running 有 slot，waiting retain，resume reuse，complete/cancel release。

**观察**：[GatedDeltaNetStateSlotManager](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/attention/gdn/state.py:17) 实现 allocate/retain/resume/release，但全 production tree 只找到定义及 [__init__ export](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/attention/gdn/__init__.py:20)，未找到 scheduler、Replica、Request 或 event handler 使用。唯一直接调用来自 [独立 lifecycle unit](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_gdn_state_lifecycle.py:8)。

**结论**：state bytes 估算已落地，request-to-slot ownership 的模拟尚未落地。完整 Simulator E2E 不可能仅凭 request completion 证明这个没有接入的对象生命周期。这里不是要求保存 tensor state，也不是要求实现已明确排除的 preemption recovery。

**建议**：在现有 admission/等待/完成/取消边界接入固定 pool 和 ownership，验证同一请求跨 continuation/waiting 的 slot 不变、释放后可复用、容量耗尽行为正确。置信度：高；源码调用搜索与读图确认。

### 1.5 SP-04 · P2 · GDNProfileInput 仍根据 query_len 推断 phase

**要求**：[原 plan](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:1312) 必须显式携带 phase/mask；[明确禁止](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:1322) “Never infer phase from query_len == 1.”

**位置**：[logical_phase 默认 None](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/profiling/gdn/inputs.py:52)、[num_decode_tokens](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/profiling/gdn/inputs.py:105)、[phase/prefill_mask 推断](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/profiling/gdn/inputs.py:133)、[mixed factory](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/profiling/gdn/inputs.py:204)。

**具体反例，未执行**：

```python
GDNProfileInput.mixed(
    decode_batch_size=1,
    decode_context_len=128,
    prefill_seq_len=1,
    prefill_context_len=128,
)
```

两个 query 都为 1，mixed() 又未传 logical_phase，结果按当前代码被归为 decode，绕过 require_supported_phase() 的 mixed rejection。使用正的 continuation context 也不会被 [_prepare_initial_state 的全正 context 检查](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/profiling/gdn/vllm_wrapper.py:388) 阻止。普通 prefill()/decode() factory 显式传 phase 的路径是正确的，不应把整个 profiler 宣称为均错误。

**建议**：要求显式 phase/per-request prefill mask；mixed API 必须保留混合语义并在执行/写行前拒绝，不再用长度猜测。置信度：高，局部源码确定。

### 1.6 SP-05 · P2 · Hybrid CPU E2E 未覆盖完整验收场景

**已完成的部分需要承认**：[production constructor E2E](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_gdn_hybrid_e2e_increment14ab.py:771) 确实训练/加载 GDN 和 synthetic 标准 profiles，实例化真实 manager/predictor/Simulator 并 run()，不是只有 mock 构造。raw artifacts 确认一条 18-token request 完成。

但它没有完成 [plan 的完整 fixture 和 CPU 验收要求](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:1805)：

| 缺口 | 具体位置/原因 |
| --- | --- |
| 自动 state reservation | 两个 E2E 都 num_blocks=100，见 SP-01。 |
| Shared expert accounting | [_qwen35_fixture_config](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_gdn_hybrid_e2e_increment14ab.py:67) 没有 share_expert_dim；与 [model.json](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/fixtures/pr31_hybrid/model.json:26) 明示 shared_expert_intermediate_size=64 不一致。[supports_share_expert](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/model_architectures.py:651) 因此不激活该路径。 |
| 真正的 continuation/chunked scheduler | 真实 case 为 prefill=16、decode=2、max_tokens_in_batch=64；one-token continuation 在 [415 行](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_gdn_hybrid_e2e_increment14ab.py:415) 只构造 GDNBatchFeatures 并断言 phase。未证明真实多次 prefill 调度的状态延续。 |
| Slot wait/resume/complete/cancel | 没有生产接线，也无 Simulator ownership 断言，见 SP-03。 |
| CPU TP>1 shape + communication 综合 case | 实际两条 Simulator case 的 attention/MoE TP 均为 1。TP>1 layout 单测存在，但不是 plan 要求的通信/accounting 集成证据。 |

**建议**：复用完整 fixture 的一个配置来源，保留现有真实训练/构造链，补足上述会改变执行分支的场景。无需等待生产 AMD profiling 数据才能验证这些 CPU 合同。

### 1.7 SP-06 · P2 · GDN 承诺保留的 GPU runtime tests 不完整

[Plan Increment 7](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:1391) 和 [11.6](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:2032) 要求保留 TP1/TP>1、actual outputs/state、state continuity 和 timing tests；hardware unavailable 时 SKIP。

目前 [GDN profiler tests](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_gdn_profiler_cpu_increment7.py:10) 使用 import、object.__new__、input/schema/source 检查。没有发现创建真实 wrapper、比较 full sequence 与 carried-prefix continuation/decode state 的可运行测试。全 tests 搜索 wrapper 名的结果仅落到这个 CPU 文件。

[Wrapper 的 _validate_decomposition](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/profiling/gdn/vllm_wrapper.py:443) 确实保留同一 initial state 下 e2e/decomposed 输出和 state 检查，这是有价值的 source implementation；它不等价于跨 chunk/full-sequence 的 continuity test。CLI 也可以留待 AMD 运行，但不能替代缺失的验收断言。

**建议**：保留明确的 GPU test entry points 和 assertion，CPU host 记录 SKIP；不要把未实现的 test case 与“已有 test 因硬件不可用未执行”混为一谈。这里不要求本次或无 AMD 环境执行它们。

### 1.8 SP-07 · P2 · ExecutionTime 尚未统一成一种 real-layer 表示

[Plan D5](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:173)、[Increment 5](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:1137) 和 [review disposition](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:2278) 明确拒绝永久保留 aggregate-vs-layer 双语义。

实际 [ExecutionTime](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/entities/execution_time.py:134) 新增 _legacy_aggregate = global_layer_id is None；[_aggregation_factor](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/entities/execution_time.py:545) 在旧模式仍按 N 放大。生产 [disaggregation public predictor](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py:1361) 先无条件进入 _predict_stage_execution_time_legacy，再扩展为 StageExecutionTime。

因此公开返回 StageExecutionTime 的迁移取得了进展，但“内部两种含义已经消除”不成立。保留的 owner record 还装有随后被忽略的 per-layer 数据；不能仅将 legacy 路径改名便视为统一完成。

**建议**：让源头按实际 layer 构造同一语义的 ExecutionTime，stage-owned CPU/transfer/terminal work 用明确 stage 接口承载；完成调用方迁移后移除旧 multiplier。等价的数值 prediction 可以复用，不必每层重新调用 estimator。维护问题另见 ST-02。

### 1.9 SP-08 · P1 · 两个 broad milestone 的 main/candidate 矩阵未完成

[Plan 11.3](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:2008) 要求 CPU core checkpoint 和 final checkpoint 都跑完整现有 dummy matrix，并纳入 available non-dummy golden/smoke；[AGENTS.md](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/AGENTS.md:761) 还要求至少 50 具体场景。

现存 manifest/results 只能证明较早 5-case 比较和两个 1-case self-parity；没有找到 main=0515589a、current candidate=7b6a3eb1 的完整矩阵。详细 case、SHA 和 artifact 类型见第 3 节。最新 unit tests 中包含名为 test_moe_ep_non_dummy_matrix 的测试也不等于启动真实 E2E matrix，该文件声明自己是 harness contract tests。

**建议**：以固定 main/候选 revision、独立 caches/output 执行现有 58-case harness 与 available non-dummy cases，保留现有 numeric/discrete comparator；外部 reference 不可用时记录具体缺项。这是后续工作，本次未运行。

### 1.10 SP-09 · P1 · 已观察的明显 event-loop slowdown 未完成分析关闭

[Plan 11.4](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:2020) 明确要求对 repeatable material slowdown 调查并修正 redundant prediction/allocation；不设置通用百分比 gate，并不免除调查责任。

现有 3 次 baseline 和 3 次 candidate measurements：

- event loop median：0.009217162 s → 0.086823318 s，9.419745×，+841.9745%。
- process median：2.203441358 s → 2.280853388 s，1.035132×，+3.5132%。
- 各次均 2/2 requests 完成、104 events；candidate 三次都约 86～87 ms，baseline 三次都约 9.1～9.3 ms。

[旧 issue ledger](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/issues.md) 将它记为 observation / not a blocker；未发现进一步热点证据、必要开销与冗余开销分解、修复前后结果或 final HEAD wall-time 复测。测量顺序也是 baseline×3 后 candidate×3，未按 plan 交替。

**结论边界**：这是早期小 dummy case 的持续差异，不能直接宣称当前 HEAD 或所有 workload 均慢 9.42 倍，也不能断言 deepcopy 是唯一根因；但足以说明 plan 要求的调查尚未完成。详见第 3.4 节。

### 1.11 SP-10 · P2 · 共用 CUDA profiling 修改缺少 NVIDIA regression 或具体 access SKIP

[Plan 11.5](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:2026) / [Increment 9](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:1568) 要求在 9 和 10 完成后，若 StepMind NVIDIA access 可用，以同一 stack 对 baseline/candidate 的真实 CUDA_EVENT linear/attention/MoE 做 grouped regression；不可用则记录具体 access/environment 原因。

分支修改了 CUDA 与 ROCm 共用的 DeviceTimer/CudaTimer、layernorm、RoPE、vLLM compatibility、linear/MoE launcher/kernel 等。task records 只有 CPU/mocks 和统一 AMD unavailable 表述；未找到 NVIDIA job、真实 producer logs 或具体“无法访问 StepMind”的记录。**AMD 不可用不能说明 NVIDIA access 不可用。**

**建议**：后续执行已授权环境调查并保留具体结果；能运行则完成一次 grouped NVIDIA regression，不能运行则按真实缺失条件 SKIP。没有证据时不宣称 CUDA GPU producer behavior 已实证不变。

## 2. 关键模块测试验证审计

### 2.1 已有 unit/focused 证据及其范围

| 证据 | 读取结果 | 可以支持的结论 | 不能支持的结论 |
| --- | --- | --- | --- |
| baseline report | 3144 passed / 19 failed / 25 skipped | 记录了仓库原有失败 | 本次未找到该 baseline 的完整 raw log，不能独立重验“19 个名字逐一相同”。 |
| [最终 full-unit raw log](/data/ycfeng/tmp/pr31-full-unit-20260915-routingfix.log) | **19 failed, 3276 passed, 25 skipped, 576 warnings in 74.32s** | 真实存在一次 post-routing-fix 全 unit 执行产物 | 不是全仓 PASS；也不是 main/candidate E2E parity。 |
| [routing report](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/test_report_2026-09-15_routing_identity.md) | 最终 focused 220 passed in 15.05s | timer/GDN/hybrid/budget/TP/MoE harness/measurement contracts 有集中回归记录 | 不保证未覆盖的真实 MLA、automatic memory、slot lifecycle。 |
| [Review 2 report](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/test_report_2026-09-14_final_review2.md) | 56 focused；91 homogeneous/metrics；15 profiling docs | 多条针对性回归存在 | 报告主要锚定较早 578785bb；不能作为当前 HEAD 全矩阵证据。 |
| [Stage metrics tests](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_metrics_stage_execution_time.py) | 有 325 行定制 mixed-family/ledger/trace 测试 | 结构性 owner/layer 输出被测试 | 不替代全架构、features、models 的真实请求矩阵。 |
| [MLA runtime tests](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_mla_predictor_runtime_operator_times.py:157) | 使用 SimpleNamespace 配置 | 旧接口的 MLA helper 行为 | 没有覆盖真实 BaseModelConfig 进入新 layer binder 的分支。 |

目前 task 将 19 个失败归为：1 个缺 config_optimizer、9 个缺 debug E2E shell assets、2 个 release docs、4 个 MHA/MLA/MQA contracts、3 个 PDD README contracts。本次 raw candidate log 支持具体 failure list 和总数；baseline 对应关系主要依赖旧记录。既不把这些失败全归咎本分支，也不以它们为由忽略本次发现的新源码问题。

### 2.2 当前关键路径覆盖映射

| 关键模块/行为 | 现有测试层级 | 仍缺少的验收 |
| --- | --- | --- |
| Platform / DEVICE_EVENT | CPU discovery、mock timer order、schema/path/cache family | 真实 CUDA grouped regression；AMD runtime 按硬件边界留待执行。 |
| Layer family resolver | Qwen3.5/Qwen3-Next/dense structural tests | 真实 MLA config 经 public prediction 的保持性，见 SP-02。 |
| StageExecutionTime / metrics | unit、手工 stage records、早期 5-case dummy parity | 最终版完整数值/离散矩阵；完整移除双语义后复核。 |
| GDN memory | 直接构造 planner 并显式给 concurrency | production scheduler 的自动 reservation 与普通 OOM。 |
| GDN slot lifecycle | 独立 manager unit | Simulator admission/wait/resume/completion/cancel 接线与 ownership。 |
| GDNTrainer / predictor | synthetic fit/save/load、six tasks、fingerprint、phase | 已覆盖主要 artifact seam；完整共享专家/TP/continuation 集成仍缺。 |
| Production hybrid constructor | 真实 manager/predictor/Simulator、1 request | 非固定 blocks、shared expert、多 chunk、TP>1。 |
| MoE routing helper | deterministic conservation/equivalence、actual Replica IDs | 最终 across-main 多 replica/DP/EP E2E，不是仅 helper 等价。 |
| GDN producer | input/import/schema；内部 decomposition validation source | retained GPU continuity/TP assertions；phase 反例修正。 |
| ROCm attention/MXFP4/RCCL | CPU planner/metadata/runner mocks | 原生 runtime 合同，按硬件 SKIP；不推定为通过。 |
| SGLang | CPU primitive/trace importer、simulated graph orchestration | HIP graph/runtime 数值与时间证据，按硬件 SKIP。 |

### 2.3 已确认存在的真实 CPU Simulator 产物

这些值属于模拟输出，按 request metrics 口径以 ms 表示；不是硬件 groundtruth，也没有本次跨版本比较。

| 已有产物 | Requests / tokens | request_e2e_time | ttft | tpot |
| --- | --- | ---: | ---: | ---: |
| dense CSV smoke | 1/1；8 prefill + 2 decode | 16.77818517187422 | 8.430025150867172 | 8.348160021007049 |
| MoE CSV smoke | 1/1；8 prefill + 2 decode | 13.560338986109947 | 7.306700001040526 | 6.25363898506942 |
| synthetic hybrid production constructor | 1/1；16 prefill + 2 decode | 9.780000000000005 | 6.450000000000001 | 3.330000000000003 |

精确路径：

- [Dense request metrics](/data/ycfeng/tmp/pr31-final-dense-20260914/qwen2_dense_test/offline_batch/final_dense_csv/request_metrics.csv)
- [MoE request metrics](/data/ycfeng/tmp/pr31-final-moe-20260914/qwen3_30b_a3b_tiny/offline_batch/final_moe_csv/request_metrics.csv)
- [Hybrid production request metrics](/data/ycfeng/tmp/pr31-wiring-production-20260915-routingfix2/test_hybrid_gdn_production_con0/sim_metrics/pr31_hybrid_production_fixture/offline_batch/hybrid_production_sim/request_metrics.csv)

三处 system_metrics.json 均记载 total_requests=completed_requests=1。Hybrid 最终测试还断言 GDN layer IDs={0,1,2,4,5,6}、dense={3,7}。这些证据应保留，避免把已经补上的生产 constructor 再误报为缺失。

## 3. 当前 branch 与 main 的矩阵及性能比较

### 3.1 原有 harness 有 58 个场景，实际审得 5 个早期跨 worktree场景

[run_scheduler_refactor_fidelity.scenarios](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/integration/run_scheduler_refactor_fidelity.py:17) 的静态组合为：

- 3 architectures × 2 modes × 2 model recipes × 3 variants = **36**。
- co-location/PDD × offline/online × spec/prefix/thinking = **12**。
- co-location/PDD × offline/online × DP2/EP8 = **4**。
- PD-AF × offline/online × dense graph/MoE graph/MoE EP = **6**。
- 合计 **58**，满足项目“至少 50 个 concrete scenarios”的现成基础。

该 harness 是 dummy-predictor parity。它比较 request/system、CSV/JSONL 和 stage ledger，采用 rel_tol=1e-12、abs_tol=1e-9，保留字段与行顺序，不属于 sklearn numerical parity。

### 3.2 现存 manifest 审计

| 产物目录 | Baseline revision/root | Candidate revision/root | 结果与实际含义 |
| --- | --- | --- | --- |
| [inc5 final manifest](/data/ycfeng/tmp/pr31-inc5-fidelity-20260914-final-1789374105/manifest.json) | 57da3683；独立 /data/ycfeng/tmp/pr31-inc5-baseline | 57da3683；当前 worktree 当时的 working tree | 5/5 PASS。与记录一致，应是提交前 working-tree 修改比较；manifest 只保存 HEAD，未保存当时 dirty diff。不是 main→当前 HEAD 的证明。 |
| [inc14ab manifest](/data/ycfeng/tmp/pr31-inc14ab-fidelity-20260914-run2/manifest.json) | fd5f49b1；当前 worktree | fd5f49b1；同一 worktree | 1/1 PASS；self-parity。 |
| [final fidelity manifest](/data/ycfeng/tmp/pr31-final-fidelity-20260914/manifest.json) | f51231d4；当前 worktree | f51231d4；同一 worktree | 1/1 PASS；self-parity，且在后续 metrics、binder、routing fixes 前。 |

不能因为 inc5 两侧 SHA 相同便否认当时发生过代码比较：不同 root、此前 1→32 的实际差异，以及 task 的提交前记录支持 candidate 当时存在修改。**问题在于该 candidate 状态没有被 manifest 固定，无法仅按 SHA 重建，而且之后还有多个功能提交。**

早期 5 个 cases：

| Case ID | Stage ledger rows | 状态 |
| --- | ---: | --- |
| co-location_offline_dense_model_basic_short | 4 | PASS |
| co-location_offline_moe_model_basic_short | 4 | PASS |
| pdd_offline_dense_model_basic_short | 5 | PASS |
| pd-af-disagg_offline_dense_model_basic_short | 193 | PASS |
| pd-af-disagg_offline_moe_model_basic_short | 193 | PASS |

它们只覆盖默认 short/offline、少量请求。5/58≈8.6% 只表示早期场景数量覆盖，不能当作最终版本覆盖率。本次没有找到 **main=0515589a 对 current HEAD=7b6a3eb1 的成套最终结果**。

明显缺少跨 main 的 online、chunked/unchunked、变化 QPS/request count、PDD MoE、feature toggles、DP2/EP8、PD-AF graph/EP、不同模型配置组合。真实 non-dummy smokes 完成了流程，但没有配套 baseline/candidate comparator 输出和误差表；available golden cases 也未见执行或具体缺失 prerequisite 的清单。

### 3.3 “已发现差异并修复”的实际证据

| 事项 | 差异/错误 | 分析与修复状态 |
| --- | --- | --- |
| Inc5 MoE ledger | baseline attention_all_reduce_time=1.0 ms；candidate=32.0 ms；绝对误差31.0 ms，相对 baseline +3100% | [旧失败 results](/data/ycfeng/tmp/pr31-inc5-fidelity-20260914c/results.json) 可读。分析指向 private per-layer fields 被 stage sum 后再按层放大；修为 first-layer compatibility，随后5/5通过。此问题有闭环，但该 compatibility 带来 ST-02 的长期复杂度。 |
| Hybrid family binding | hybrid 调用 whole-model binder 报错 | 新 per-layer dispatch 与 profiling binding seam 有针对性修复记录；本次发现 homogeneous MLA 的另一条回归，不能视为同一问题已全面关闭。 |
| GDN op schema | inactive core key 缺失导致 trace consumer 失败 | predictor 明确补零、focused hybrid E2E 已记录 PASS；不重报为当前缺陷。 |
| Monolithic routing ID | routing map key0，实际 Replica.id=1 等 | Simulator→predictor 传实际 IDs；latest tests 对比 routing keys 与 cluster keys。已有修复闭环。 |
| SGLang tuple contract | native builders 返回形状不同 | [_make_replay_call](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/profiling/experimental/sglang/graph_replay.py:205) 归一化；CPU graph orchestration 覆盖多种形状。旧问题关闭；不等于 HIP runtime 已验证。 |
| CPU event-loop slowdown | 9.419745× | 有重复测量，无热点归因、修复证据或最终复测，SP-09 未闭合。 |
| 最终分支其他数值/调度差异 | 未建立完整 baseline/candidate矩阵 | 无法判断哪些场景完全等价，也无法宣称所有额外变化已经分析修复。 |

### 3.4 Wall-time 原始测量与可归因范围

原 runner：[pr31_inc5_walltime_run.py](/data/ycfeng/tmp/pr31_inc5_walltime_run.py)，复用 tests/performance/sim_walltime_scaling/run_case.py。配置是 online sequential PDD、llama3.3-70b、16 modeled GPUs、PP2、attention TP4、2 requests、prefill8、decode4、QPS2、dummy1ms、104 events；大量 metrics/traces 开关关闭。

环境来自原 result/report：host ycfeng、/usr/bin/python、Python3.12.3；无 Conda。原始 result.json 两侧 git_sha 均为57da3683，candidate 当时修改状态同样未保存在版本字段。

| Side | Attempt | sim_wallclock_s | init_s | total_proc_s |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 0.009217161998 | 2.223078153998 | 2.235523351999 |
| baseline | 1 | 0.009274923003 | 2.187447409000 | 2.200224051005 |
| baseline | 2 | 0.009079137002 | 2.191065608000 | 2.203441358004 |
| candidate | 0 | 0.086433376004 | 2.166795703000 | 2.256504671001 |
| candidate | 1 | 0.086823318001 | 2.190165941000 | 2.280853387994 |
| candidate | 2 | 0.087078498000 | 2.222510623003 | 2.312716245004 |

| Metric | Baseline median | Candidate median | Candidate−baseline | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| event loop | 0.009217162 s | 0.086823318 s | +0.077606156 s | +841.9745% |
| initialization | 2.191065608 s | 2.190165941 s | −0.000899667 s | −0.0411% |
| process total | 2.203441358 s | 2.280853388 s | +0.077412030 s | +3.5132% |
| event count | 104 | 104 | 0 | 0% |

原始文件位于 `/data/ycfeng/tmp/pr31-inc5-walltime/{baseline,candidate}/attempt-{0,1,2}/result.json`。时间戳显示 08:08:06～14 UTC 先跑完 baseline三次，08:08:14～22 再跑 candidate三次，属于 AAA BBB；不是要求的交替配对。上述比率是两组中位数之比。

**候选源码中的调查起点，不是已证实的 RCA**：

1. [StageExecutionTime.from_execution_time](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/entities/stage_execution_time.py:225) 对每层调用 [ExecutionTime.as_single_layer](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/entities/execution_time.py:1118)，后者 deepcopy 完整对象。表示迁移允许每层独立 record，但未证明需要复制完整 legacy payload。
2. [model_time_ms](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/entities/stage_execution_time.py:404) 每次读取都遍历各 layer；op_times 也重复聚合。实际调用次数与占比需要 profile。
3. [MoE public predictor](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:3440) 递归构造单层 Stage 包装后再组合；query key 每层递归 freeze，cache 命中仍 deepcopy。
4. [bind_layer_attention](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/attention/model_binding.py:256) 每次重新 resolve 整个 schedule，虽然 BaseModelConfig 已有缓存 getter；存在重复解析机会。

该测量是 dense/PDD，不能拿 MoE-only query cache 直接解释其中9.42×。也不能把后续 metrics additions 当作当时已测原因。后续需要在固定最终版本、同机同环境下交替重复，并分离必要 per-layer成本、重复 prediction、对象分配和 metrics 开销。小 case 受启动占比影响很大，process +3.5% 不会抹去 event loop 的 +77.6ms。

## 4. Standards

Standards 轴独立列项；ST-01 与 SP-02、ST-02 与 SP-07 是同一事实在不同审阅轴上的影响，不能相加当作不同 runtime bugs。其余 smell 是有明确代码依据的维护判断。未发现足够依据的 Fowler smell 不作泛泛罗列。

### 4.1 ST-01 · P1 · Attention 分类绕过既有 authoritative binder

[新 resolver](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/attention/gdn/config.py:290) 的“非 Qwen3.5 一律 dense”在调用者中创建第二套 family 分类规则；与 [原 binder 的 MLA/DSA 分支](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/attention/model_binding.py:139) 不一致。违反集中分类和行为保持要求，实际后果见 SP-02。

**简洁建议**：ordinary homogeneous family/variant 只通过原 binder；hybrid schedule 使用同一 family definitions。增加下一 attention family 应改一处归属规则，而不是让所有调用者再次猜测。

### 4.2 ST-02 · P1 · Legacy aggregate、first-layer probes 与 stage sums 共存，Stage 接口过重

位置：

- [ExecutionTime compatibility multiplier](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/entities/execution_time.py:134)
- [disaggregation legacy wrapper](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py:1361)
- [Stage 三份属性名白名单](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/entities/stage_execution_time.py:30)
- [operator accessors 返回 first layer](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/entities/stage_execution_time.py:293)
- [__getattr__ 私有字段转发](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/entities/stage_execution_time.py:486)

`attention_time` 等 public属性求 stage sum，`attention_operator_times` 等返回第一层，private字段又根据白名单取第一层或 owner；`op_times` 是另一种 stage aggregate。一个调用者必须先知道某个名字属于哪张名单，才能判断拿到的是 layer 值还是 stage 值。对 mixed family 尤其难读。

这不是单纯“511行太多”；问题是同一个类维持多种历史解释，已发生过1→32的双计数。它也偏离原 plan 明确决定的单一 representation。

**简洁建议**：显式 layer API 与 stage-owned aggregate API，迁移消费端后删除泛化 private forwarding和旧 multiplier。可以保留确有调用者的单层查询，但名字和参数应明确对象范围。

### 4.3 ST-03 · P2 · 为旧 test doubles 添加了生产 sentinel 旁路

[MoE predictor](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:3500) 与 [disaggregation predictor](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py:1373) 的注释直接写明为了 lightweight test doubles，非 ExecutionTime 值原样返回。

正常 production helper 返回 ExecutionTime；这些分支为 test sentinel 放松了声明为 StageExecutionTime 的 public return contract。它们可能隐藏错误 wiring，而真实运行后续才因缺属性失败。没有证据表明真实受支持 backend 需要任意 sentinel。

**简洁建议**：让测试 double 返回最小合法 timing record，生产接口始终返回约定类型。这是具体的不必要兼容补丁，不意味着全部 getattr/validation 都应删除。

### 4.4 ST-04 · P2 · Family/owner 分类散落在三个 metrics 消费路径

重复位置：

- [trace emitter](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/metrics/metrics_store.py:1053)
- [operation metrics](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/metrics/metrics_store.py:3709)
- [stage component ledger](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/metrics/metrics_store.py:4148)
- [owner_only_names](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/metrics/metrics_store.py:4128) 与 [Stage owner名单](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/entities/stage_execution_time.py:74)

这些分支按 `"gated_delta_net"` 决定 projection/核心算子归属。已有 family registry，但调用者继续复制同一分类知识。下一个有自身 fused projections 的 attention family 会要求修改多个消费者；属于 Repeated Switches/Shotgun Surgery，并违反 growing category 集中分类原则。

**简洁建议**：由已有 family/operator metadata 生成明确顺序和归属的 op records，trace/metrics/ledger 各自消费；修改范围保持在本次接口周围，不建议无边界重构整个 metrics 系统。

### 4.5 ST-05 · P2 · DEVICE_EVENT 路径解析有三份实现，已经出现不同结果

- [manager training resolver](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/shared_prediction_model_manager.py:637)
- [manager public path API](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/shared_prediction_model_manager.py:4678)
- [predictor resolver](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_execution_time_predictor.py:851)

当 configured path 和 fallback 都为空时，public API 返回空，另外两处按当前表达式生成 `"_device_event"`。因此最终 path contract测试通过，不等价于加载路径也满足同一合同。是否在某个 profile组合造成实际加载失败尚未执行。

**简洁建议**：复用一个受支持的 measurement→input path解析函数，三个调用者共享 empty/configured/derived语义；沿用现有机制，不新增 manager层。

### 4.6 ST-06 · P2 · Query cache 为等价数值复用引入通用序列化、fallback 与无限保留

[MoE predictor 2165～2364 附近](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:2165) 新增约200行逻辑，涉及：

- 对任意 Mapping/list/set递归 freeze；
- [未知对象 repr fallback](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:2198)；
- [callable request属性 TypeError转None](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:2215)；
- [family取值异常后放弃cache](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:2249)；
- [predictor生命周期 dict cache，无上限/淘汰/阶段清理](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:2351)，命中还 deepcopy。

等价 attention query复用是原 plan要求，本身合理；疑点是为了这个需求引入宽泛数据类型和异常fallback。长 trace中不同request progress会产生许多新key，保留期覆盖整个predictor；本次没有测量其内存或耗时占比。

**简洁建议**：先复用已解析的 immutable spec和实际workload features，考虑把复用范围限定为一次stage/已知短生命周期；只有真实跨stage收益证据充分时再保留更广cache。查清实际可达metadata类型后删除无需求的通用repr/异常吞并。

### 4.7 ST-07 · P2 · 大模块 cleanup/split门禁缺少任务记录

[AGENTS.md](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/AGENTS.md:749) 与 [原 plan Section2](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md:128) 要求：对已超2000行的critical模块先审冗余；仍超限则记录原因、拆分边界/顺序，并按increment记录before/after行数。

本次对 task Markdown 搜索未找到对应before/after表、cleanup结果或split分析。review/progress列举了功能变化，但未完成该门禁记录。以下数字由固定diff与候选文件静态行数计算：

| 文件 | main行数 | candidate行数 | Diff + / − | 分析 |
| --- | ---: | ---: | ---: | --- |
| frontier/config/config.py | 5663 | 5725 | +62 / −0 | 既有超大配置模块；应说明本次schema新增边界。 |
| shared_prediction_model_manager.py | 4570 | 4760 | +211 / −21 | family/path注册散落，ST-05展示具体重复。 |
| sklearn_execution_time_predictor.py | 8315 | 8397 | +172 / −90 | 历史体量大，本次净增仅82；layer dispatch与原binder关系应明确。 |
| sklearn_moe_execution_time_predictor.py | 3523 | 3719 | +300 / −104 | query cache约200行，stage migration与routing修改叠加。 |
| sklearn_disaggregation_execution_time_predictor.py | 3058 | 3074 | +51 / −35 | 净增不大，但wrapper保留旧实现。 |
| metrics/metrics_store.py | 5270 | 5587 | +324 / −7 | 净增317，新trace/metrics/ledger并行维护同一分类。 |
| vllm_v1_engine_replica_scheduler.py | 5073 | 5082 | +9 / −0 | 本次仅preemption guard；不建议借此做无关大重构。 |

新增/明显增长模块：

| 文件 | main→candidate | 审阅意见 |
| --- | --- | --- |
| entities/stage_execution_time.py | 0→511 | 关注ST-02的混合接口；不能单凭511判错。 |
| profiling/gdn/vllm_wrapper.py | 0→639 | 集中构造、state metadata、priming、decomposition、TP样本聚合与row输出；复杂度有部分真实来源，应先补runtime验证，再按metadata/state/measurement边界考虑小拆分。 |
| profiling/experimental/sglang/graph_replay.py | 0→494 | primitive归一化、graph执行、artifact orchestration集中；已有CPU orchestration测试，不重报已修tuple问题。 |
| profiling/moe/moe_vllm_kernel.py | 674→990，+343/−27 | MXFP4/functional API新增316净行；重点验证公共CUDA兼容性。 |
| tests/unit/test_gdn_hybrid_e2e_increment14ab.py | 0→930 | 大量fixture与两套构造路径，但关键branch缺测；可提取共享fixture并补真正缺失的cases，避免只增加mock体量。 |

**简洁建议**：先处理本报告指出的重复/多语义，记录为什么必须留在父模块；必要拆分只围绕新增attention查询、stage accounting、profile path职责，保留现有包边界。历史上已经超大的代码量不全部归因于本分支。

### 4.8 不应误报为 hard-coding/过度防御的项目

- D57明确批准的 MLP/MoE **2 bytes / parameter** 近似、GDN A_log FP32字节数属于接受的模型合同；不要求引入完整MXFP4 storage calculator。
- GDN unsupported PP/EP/DP/prefix/speculation/preemption的 fail-fast属于明确需求。
- MI355X registry entry、Qwen3.8真实层数/shape config是声明式数据；不属于随意调参。
- 原routing公式中的固定参数若按old/new一致提取，属于保持原行为；未发现新增任意scaling factor用于掩盖latency误差。
- GDN synthetic CSV的diagnostic e2e=99.0用于证明该字段不被predictor消费，属于有说明的测试fixture。
- CPU lazy imports与明确dependency errors是必要边界；不能因存在validation就批评为“过度防御”。

## 5. 建议交给后续 agent 的工作包

以下仅为可审阅建议，本次没有执行，也没有修改原 plan 或扩大授权。实际实施仍应遵守工作区的具体改动边界。

| 顺序 | 有界工作包 | 验收目标 | 依赖 |
| --- | --- | --- | --- |
| 1 | 修复 homogeneous attention spec归属，保留Qwen3-Next原处理 | 真实BaseModelConfig的MLA prefill/decode走正确family；原dense/MLA/frozen合同保持 | 无 |
| 2 | 连接GDN reservation capacity和request slot lifecycle | 自动规划小case成功；固定state预算正确；wait/resume/completion/cancel ownership成立；OOM不被绕过 | 无 |
| 3 | 清除GDN phase推断并补retained producer test contracts | one-token continuation保留prefill；mixed始终拒绝；CPU可读、GPU按硬件SKIP | 无 |
| 4 | 完成real-layer表示迁移、明确stage accounting范围 | 所有public predict返回一致类型；layer identity完整；stage-owned work只一次；无legacy multiplier旁路 | 1；需与metrics同步 |
| 5 | 使用完整shared-expert fixture补CPU场景 | 非固定blocks、actual chunked continuation、TP>1通信、shared expert无重计、slot lifecycle | 1～4 |
| 6 | 固定main/候选commit跑已有58-case矩阵及available non-dummy cases | 原numeric/discrete字段满足既有tolerance；每个差异有可核对cause/disposition | 1～5 |
| 7 | 重做交替CPU wall-time subset并调查差异；补NVIDIA证据/具体SKIP | 区分startup/event loop；清除确认的重复allocation/prediction；剩余必要开销有证据解释 | 4～6；NVIDIA可独立 |
| 8 | 更新原task完成度和最终archive | 完成结论与实际版本、matrix、hardware边界一致；选定关键raw artifacts持久保存 | 1～7 |

依赖概览：`{1,2,3} -> 4 -> 5 -> 6 -> 7 -> 8`。1/2/3的调查可独立；若实现修改同一module，必须明确ownership协调。NVIDIA evidence调查可以与CPU工作独立推进。这里不是要求在本次review里开展这些工作。

**建议的后续矩阵命令形态（未执行）**：

```bash
python tests/integration/run_scheduler_refactor_fidelity.py \
  --baseline "$BASELINE_REPO" \
  --candidate "$CANDIDATE_REPO" \
  --output "$FRESH_PARITY_OUTPUT" \
  --workers 1
```

baseline/candidate必须分别固定为要比较的revision、output必须为新目录；不传--case才覆盖全部58个既有场景。available non-dummy和golden harness仍需先按其真实输入/CLI选择，不能用本dummy命令替代。

## 6. 完成声明与证据维护建议

[plan.md:42](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/plan.md:42) 的“All work units ... complete”、[review.md:134](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/review.md:134) 的CPU acceptance PASS，与本次发现的生产接线和矩阵缺口不一致。建议后续完成修复与验证后再更新这些记录；本次保留原文件不动，让不同review结论可独立审阅。

需要准确保留的三类状态：

1. **已实现且已有CPU证据**：GDN training/artifact路径、production constructor、routing ID修复、部分stage/metrics/experimental contracts。
2. **待补实现/验收**：SP-01～SP-10对应事项。不能统一归为AMD hardware unavailable。
3. **允许的硬件未执行**：AMD/MI355X真实ROCm、AITER、RCCL、SGLang/HIP等。即使CPU事项关闭，仍不能宣称硬件性能/groundtruth parity。

现有大量logs/result files位于临时产物根/data/ycfeng/tmp。文档链接可用于本次审阅，但若需要长期交接，应将最终manifest、精选request/system/stage evidence和test summary保存在规定的持久task目录，并绑定确切revision。无需为简单留档新增hash系统；当前缺的是candidate revision/dirty状态及完整case结果，不是额外指纹。

## 7. 附录：本次只读检查方法与限制

### 7.1 执行过的只读命令类型

```bash
git status --short --branch
git rev-parse HEAD main origin/main
git merge-base main HEAD
git log main..HEAD --oneline
git diff --stat main...HEAD
git diff --numstat 0515589ac7f49ac5288a5f55b0ce38b0ede29bb2...7b6a3eb1
git diff main...7b6a3eb1 -- <specific source files>
git log --format='%h %s%n%(trailers:key=Co-authored-by)' main..7b6a3eb1
rg --files task_memory/task_2026-09-14_pr31_selective_integration_v2
rg -n '<specific symbols/requirements>' <specific files/directories>
nl -ba <specific source file>
```

同时用Python标准库Path/json/csv/statistics读取既有result/manifest/metrics并计算中位数；没有import Frontier模块或调用测试/运行入口。这些解析不构成重新跑测试。

初始文件定位中 .github、一个假设的replica_scheduler_config.py以及可选notes.md不存在，rg返回exit2；随后改查实际AGENTS.md、config.py及已有task文件。这些是检索定位结果，不是产品测试失败，也没有据此修改代码。

### 7.2 Evidence index

| ID | 主要证据 |
| --- | --- |
| E01 | [requirements.md](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/requirements.md)、[原 plan v2](/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md)、[AGENTS.md](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/AGENTS.md:749) |
| E02 | [baseline report](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/test_report_2026-09-14_baseline.md) |
| E03 | [inc5 report](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/test_report_2026-09-14_increment5.md) |
| E04 | [inc5 final results](/data/ycfeng/tmp/pr31-inc5-fidelity-20260914-final-1789374105/results.json) |
| E05 | [inc5 failed results](/data/ycfeng/tmp/pr31-inc5-fidelity-20260914c/results.json) |
| E06 | [inc14ab manifest](/data/ycfeng/tmp/pr31-inc14ab-fidelity-20260914-run2/manifest.json) |
| E07 | [final self-parity manifest](/data/ycfeng/tmp/pr31-final-fidelity-20260914/manifest.json) |
| E08 | [baseline wall-time attempt0](/data/ycfeng/tmp/pr31-inc5-walltime/baseline/attempt-0/result.json)、[candidate attempt0](/data/ycfeng/tmp/pr31-inc5-walltime/candidate/attempt-0/result.json)，同目录attempt1/2 |
| E09 | [post-routing full-unit log](/data/ycfeng/tmp/pr31-full-unit-20260915-routingfix.log) |
| E10 | [routing identity report](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/test_report_2026-09-15_routing_identity.md) |
| E11 | [hybrid simulator report](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/test_report_2026-09-15_hybrid_simulator_e2e.md) |
| E12 | [final Review2 report](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/test_report_2026-09-14_final_review2.md)、[issues](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/issues.md)、[summary](/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/task_memory/task_2026-09-14_pr31_selective_integration_v2/summary.md) |

### 7.3 最终审阅状态

- 本次review交付：**完成**；独立报告已经形成。
- 本次review待办：**无**。
- 本次新发现的实现/验证问题：**有**，见Spec与Standards；修复/测试不属于本次执行范围。
- Spec轴：**10项 findings**；最高严重程度P1，主要是GDN自动容量规划/MLA回归/state lifecycle以及最终matrix/performance验收未闭合。
- Standards轴：**7项 findings**；最高严重程度P1，主要是分类旁路与可达的双重execution语义。与Spec重叠的事实已明确标注，未合并排序。
- 当前branch的plan完成度：**不能接受现有“全部完成”声明**。
- 当前branch相对main的数值/调度/性能保持：**证据不足，不能声明没有额外变化或变化都已修复**。

本报告的完成不代表被审任务已经完成；所有runtime影响中的未实测部分已按源码判断或风险推断标注。

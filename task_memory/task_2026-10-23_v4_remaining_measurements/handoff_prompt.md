# Continuation prompt

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Prepared the next-session prompt; final collector closure is recorded in progress.md and the acceptance receipts. |

```text
你接手 Frontier 的 GLM-5.3 / DeepSeek-V4 校准工作。先核对实际 Git 状态和下列记录，再执行下一步。

工作目录：/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
分支：feat/glm53-dsv4-vllm028
主任务：task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/
当前子任务：task_memory/task_2026-10-23_v4_remaining_measurements/

首先读取：
1. 全局和仓库 AGENTS.md，以及适用的 frontier-calibration skill。
2. 当前子任务 requirements.md、plan.md、progress.md、summary.md；progress.md 的交接段是未完成工作和下一步的权威记录。
3. input_primary/primary_acceptance.json；collector_primary/ 的最终复审、测试及 acceptance 记录。不要把旧 d94b821a 的 CPU_COMPLETE 报告当作最终准入。
4. driver_design.md、native_profile_seam.md、audit/test_report_2026-10-23_kernel_coverage.md。proposal.md 中的早期 GPU 数字只是历史草案。

已完成且应复用的成果：
- native06 exp-1023-043123-830898 是 ACCEPTED_NATIVE_REFERENCE_R4：原运行 b62c155，32×H800，12.002222 GPU·h；R4 冻结复验8/8，client432/432。原 R3 FAIL、native05 UNVERIFIED 均保留。主任务 native06_acceptance_20261023/ 保存最终凭据。它只是 native reference，不等于 Frontier 数值或算子成本通过。
- 私有 attention input 已合入732406aa：选择 CSV 后，read_isolated_scope 可发现 dsv4_attention.json，并要求独立的 .admission.json。七个 non-timing engine/step records 推导十二个布局。真实自动读取94项与显式入口完全一致，merged194测试通过，legacy102/72保留，strict77/77一致；合并时明确复用固定4a1b53e6 capture。准入仅 isolated_prefill_input_only。
- 已准入输入目录：/data/ycfeng/tmp/glm53_dsv4/v4_attention_input_20261023_w128_candidate02/。使用前核对 input_primary 中的 hashes；文件缺失或不匹配应报告，不得 fallback 或改写原始表。
- native collector 的最终源码提交、修复结果和复测由 collector_primary/ 最终记录给出。初审五个问题为 graph-manager 生命周期、slot-mapping ownership、实际 source-visible PDL 参数、最终 step byte admission、重复错误日志上限；只依据最终复审判定其关闭。

下一阶段是 G2：
先根据最终 collector 状态实现真实 pd_replay 调用链上的 profiling driver，选择 serving_profile_limits 与原生 profiler 的有界采样窗口，做 CPU API/错误隔离验证；然后形成具体 GPU 请求，等待 YC 批准。
native_pd_worker.py 实际启动 pd_replay.py。在选定 formal 阶段首个请求前，对两个 role head 调 /start_profile；完成并 drain 后调 /stop_profile。沿原有 RPC fanout 覆盖 headless DP 和 TP workers；不能只新增一个从未被调用的脚本。driver_design.md 记录完整边界。
保留原 workload、warmup、三轮规则、request IDs、返回值和原生错误处理。所有辅助失败只使诊断 UNVERIFIED，不阻断健康 workload。
注意 collector 与 profiler 的窗口不同：f8621a8 的 collector 从启动起记录，emit 不清空，未 hook Worker.profile；/start_profile 不会自动重置其 records/bytes 预算。G2 必须明确预算和窗口对应关系，保证实际采样时所需记录没有被启动/预热阶段耗尽。

持续约束：
- Frontier/vLLM routing 都用已绑定 balanced_topk。先 workflow/GPU/operator-gap，CPU overhead 最后。
- 逐算子 KERNEL_ONLY 是已批准路线，但唯一 kernel→scope 归属不证明完整覆盖或可相加；不能直接求和 scopes，不能把 CUDA_EVENT−KERNEL_ONLY 当 CPU cost，不能缩放消差。
- 已接受的 isolated 数据只覆盖 TP1/eager/六层6144/C0。TP8/DP2/EP16/PP1、balanced decode6143..7166、graph 成本和 serving total 仍需要自己的证据。family/PD/graph/TP8拒绝不得提前去掉。
- V4 native 使用自动开启的 breakable CUDA graph。native06 的 decode 实际为 FULL+PIECEWISE，descriptor1/2/4/8/16、每 lane1..11条请求；128是 capture 容量，不是实际批次。has_prefill 取原生状态，不能从 tokens==requests 推断。
- GPU stream→trace ID、实际 PDL correlation、compile/breakable annotation/event survival、captured-segment ownership、GPU CUPTI restart 等未验证能力必须明确标识。源文件符号存在或 Torch CPU 控制不等于 GPU hook 可执行。
- 不改 native/C26，不改冻结证据，不重标旧 q1 数据；GLM 继续暂停。此前 prefill/native06 GPU 配额已经消费，不能再次使用。任何新 GPU 运行都需具体方案和新的明确批准。
- 32卡及以上的辅助采集/验证只作 soft gate。GPU 操作前读 /data/ycfeng/stepfun-env-handbook/stepmind-python-rjob.md；按当前 provider 的直接操作/授权路由规则执行。
- .pi/、.cortexkit/ 是先存的无关未跟踪目录，保留；不删除其他 worktree，不改无关用户工作。

按 owner 分配实现与修复，独立复审后 root 验收。已有批准允许必要的相似 repair-scope 调整，由 root 记录和决策，不重复询问；这不授权新 GPU 预算、共享语义改变或放宽验收。
完成一步就提交对应源码与文档。所有验证用最终固定源码，suite 分别报告，不累计重复测试数量。
```

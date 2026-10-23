# Continuation prompt

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Advanced to accepted GPU stream/restart evidence and bounded diagnostic DP sequence implementation. |
| 2026-10-24 | Advanced the stream probe to completed execution and pending independent data acceptance; added parallel DP-counter CPU verification. |
| 2026-10-24 | Bound the continuation to CPU stream-loader repair0d745706 and the pending targeted probe; retained the historical G2/run references. |
| 2026-10-24 | W128-OCT24-DATA-DOCS: continuation starts from diagnostic-only profile02 and isolated MoE acceptance; bounded gap design/CPU repairs precede fresh H800 collection and tuple-gated cost consumption. |
| 2026-10-23 | W128-PROFILE02-STAGE-DOCS factual delta: job exp-1023-180821-373637 created10:08:21Z, verified SUBMITTED_PENDING/queuing at10:10:16Z; I3 monitoring then I4 verification, auxiliary watch403 not a job failure or stop. |
| 2026-10-23 | W128-PROFILE02-STAGE-DOCS: synchronized preserved profile01 entry FAIL and repaired/re-reviewed sealed diagnostic profile02; actual final tests PASS separately, root launch started without evidenced platform job identity. |
| 2026-10-23 | W128-H800-AUTHORITY-DOCS: added current approved H800 authority and packaging route above the immutable historical H prompt; handoff remains informational during same-session work. |
| 2026-10-23 | Prepared the next-session prompt; final collector closure is recorded in progress.md and the acceptance receipts. |
| 2026-10-23 | Bound the prompt to accepted H sourcef8621a8/integration64f3c950 and final merged checks. |
| 2026-10-23 | Marked the H handoff historical after YC resumed this session and G2 CPU source 05952396 was accepted. |

## Current informational continuation state — 2026-10-24

Post-audit delta: CUPTI repair0d745706 and probe5f7d0950 have accepted targeted GPU stream/restart evidence from exp-1024-020317-012085, after independent No findings and byte-identical root162-check audit. Read `profile01/stream_probe/results/primary_acceptance.json`:8eager/16priming joins,16stable snapshots; captured per-op graph ownership remains UNKNOWN. Exact-image WS2 gloo control passed (`profile01/dp_sequence_cpu.md`). Next is four-file/180-line optional diagnostic DP sequence recording under the current requirements allocation, preserving frozen base dispatch/hook schemas. No custom callback, CUDA13.1 tools-ID API, numeric graph-ID conversion or paused GLM work is included.

YC requested continuation in this same session; this handoff is informational, not a scheduling instruction or a new transfer of execution. Read current [requirements.md](requirements.md), [plan.md](plan.md), [progress.md](progress.md) and [proposal.md](proposal.md) before using any historical prompt. G2 CPU source05952396 is ACCEPTED_CPU_PROFILE_DRIVER_COMPONENT; accepted source05952396 and recordsb8fe6b2d remain unchanged. Its final separate suites and explicit existing77/77 matrix binding are recorded in `driver/primary_acceptance.json`; no new strict77 capture.

YC's exact resource reply is “批准；h800集群的的运行无需向我过问，允许你提交符合任务需求的任意规模的rjob（在h800上）”. D-V4-H800-RJOB-AUTHORITY approves the current32-H800/codesign/non-preemptible profiling run, at most60minutes/32GPUh, P16+D16/TP8/DP2/EP16/PP1 and first-formal64worker steps. Task-required H800 RJobs of any necessary scale need no repeated resource approval; each later run records its concrete purpose, resources and evidence and respects actual quota/capacity, personal StepMind submission, local mounts, storage boundaries and auxiliary soft gates.

Initial profile01 fresh entry FAIL (`ModuleNotFoundError: No module named 'vllm_replay'`) remains preserved, unchanged, unsealed and unsubmitted. Owner repair `d87f4c74`, records `a90c9dbb`, merged `cc0227e9`, uses cumulative152/300 changed lines in two maintained source/config files; both independent re-reviews have no open findings. Read `profile01/review.md` and `profile01/delivery.json`; the delivery's unsealed status is the historical owner snapshot, superseded for root readiness only.

Final root tests after source/package binding passed separately: actual StepMind-interpreter launcher; real locked-image bootstrap/fresh child with87native/23tool/20binary pins; four isolated hook/role-path probes, each PASS. No actual GPU constructor/model forward/replay was exercised. Root sealed45 inputs with manifest `51e39a42ec863e496f20be8a165230e0dee17c34906ad6bd4fade14bf898ee91`, status SEALED_READY_DIAGNOSTIC. Read `profile01/test_report_2026-10-23_root.md` and `profile01/profile02_root_readiness.json`. Candidate: `/data/ycfeng/Frontier/.worktrees/v4-native-profile-run/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_02`; run tag `dsv4-h800-p0-profile-20261023-02`.

Historical first-submission status: SUBMITTED_PENDING, job `exp-1023-180821-373637`, created2026-10-23T10:08:21Z. Snapshot2026-10-23T10:10:16Z is Pending/`RJob is queuing`, `replicas=[]`, not Running or success. Verified creator `i-fengyicheng`, `codesign`, preemptible=no, four replicas each8GPU/64CPU/500Gi (SDK mem_gb=512 converts to500Gi), locked image, exact local NFS `100.96.128.195:<profile02-package>:<profile02-package>` and RDMA1/8; launcher PID3236764 was alive then. Read `profile01/profile02_platform_launch.json`; `profile01/profile02_root_submission.json` retains the prelaunch-gates snapshot and exact launch command/log `/data/ycfeng/tmp/glm53_dsv4/profile02_launch_20261023.log`. The completed run is not to be duplicated unchanged.

Current profile02: Succeeded4/4/exit0/restart0; [primary acceptance](profile01/results/primary_acceptance.json) is ACCEPTED_DIAGNOSTIC_AUDIT_WITH_GAPS only. [Root32-trace rerun](profile01/results/root_verification.json) matches all non-clock fields and decompressed indices,326328annotations/7600008kernels; useful P0/P1/D0/D1=23/2/61/61.192 null stream mappings (`libcupti.so`),6802600 graph kernels' UNKNOWN ownership and UNKNOWN shared cross-DP identity/dependencies prevent cost/serving admission. ONLY decodeDP1 has the shutdown error; prefillDP1 true/errors empty. The former two-headless claim was erroneous; cross-host wall timestamps cannot prove actual order.
[Archive receipt](profile01/results/archive_receipt.json) pins46files/86211102bytes, full53441663-byte audit plus32indices outside immutable run inputs/outputs at `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_profile_20261023/case/analysis/profile02_cpu_audit_20261024`; Git retains compact receipts. The earlier watch403 stays an auxiliary failure, not a workload failure.
Current Q2b: [MoE acceptance](moe/results/primary_acceptance.json) is ACCEPTED_ISOLATED_KERNEL_DATA_WITH_LIMITS after No findings/[root exact rerun](moe/results/root_verification.json). Source737b6311/docfix941a4450 mergedbe067bb5,204/300lines; [seal](moe/root_seal.json) actual image440/20/1003 and SDK PASS. Jobexp-1023-235250-540507 Succeeded15:54:35Z,1H800/16CPU/mem_gb128/non-preemptible/1800s ceiling,92container seconds/exit0/restart0. [Terminal receipt](moe/root_run_check.json) preserves the first guessed-label false negative; codesign/no and normal launcher exit are verified.
MoE19outputs/3261347bytes, two four-row tables,12traces/112statistics/delta0;20grouped calls/M and7kernels/call. Rank0-only KERNEL_ONLY means(ms) M16/32/6152/12288=0.1017937/0.1022997/0.7557434/1.29869045. CUDA_EVENT has no raw durations/grouped-count evidence; M6152 includes a dummy peer. No other-rank/TP8/communication/graph/serving/weights equivalence or native-tuple cost consumption. Q2a ACCEPTED_ISOLATED_DATA/901ef960 remains accepted; decoder consumption pending.
Next: bounded source-backed collection-gap design and CPU repairs/review -> fresh package/image/SDK checks -> recorded H800 run under existing standing authority. Require three useful batches per claimed phase, shared iteration identity, captured ownership and real stream/dependency evidence; request sets/graph IDs are not substitutes. Native tuple/rank/dependency admission precedes cost consumption. Keep auxiliary errors soft; full05/09/19 OPEN. Old pending/current snapshots are historical; native06 R4 is not replaced by profile02.

Native06 remains ACCEPTED_NATIVE_REFERENCE_R4, not replaced by this instrumented run; old R3 FAIL/native05 UNVERIFIED, balanced_topk, layout/source/routing/phase/graph/provenance refusals, official mean/30%/10% gates, GLM pause and CPU-overhead last are unchanged. Resource approval does not authorize native/C26 edits, publication, unrelated broad CUPTI investigation or CPU-overhead collection. Preserve original packages/data/frozen evidence and unrelated user work; do not reuse the spent prefill/native06 allocations as this run's authority.

## Historical H close-out prompt — not executed

The original text below is retained byte-for-byte. Its then-pending G2 implementation, new-GPU-approval requirement and next-session steps are superseded by the accepted G2 component and D-V4-H800-RJOB-AUTHORITY above; its technical refusals and evidence limits remain binding.

```text
你接手 Frontier 的 GLM-5.3 / DeepSeek-V4 校准工作。先核对实际 Git 状态和下列记录，再执行下一步。

工作目录：/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
分支：feat/glm53-dsv4-vllm028
已验证代码集成点：64f3c9509b09279f321d237e3a95cbe904c6dd22；其后收尾提交仅更新文档和验收记录。先核对实际 HEAD 与该提交的关系。
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
- native collector 源码f8621a8已合入64f3c950，collector_primary/primary_acceptance.json为ACCEPTED_CPU_COLLECTOR_COMPONENT。初审五个问题（graph-manager生命周期、slot-mapping、实际PDL参数、完整step容量检查、重复日志上限）均经独立复审关闭，五文件880/987行。Root在最终源码和合并后分别复跑：collector28、inventory28、V2inventory45、reader135、worker16、DPharness75、P/Dharness68、manifest14，全部通过，不能相加。合并后自动94结果完全一致；H的77/77矩阵通过显式复用已有capture完成，没有新capture。代码/CPU阶段已收尾，profile输出仍UNVERIFIED。

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

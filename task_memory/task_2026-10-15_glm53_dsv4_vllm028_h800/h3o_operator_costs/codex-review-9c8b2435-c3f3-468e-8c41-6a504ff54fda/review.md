# H3o-(a) operator-level DeepSeek-V4-Flash costs 设计审查

## 1. Review context

### 请求与 resolved scope

本次是 **design review，不是 diff review**。主要目标是判断 H3o-(a) 是否足以支持随后实现 operator-level costs，并回答 caller 提出的 method、inventory、overlap/graph、eager、数据与 jobs、file plan、acceptance、sequencing 八类问题。

| 项目 | 本次解析结果 |
| --- | --- |
| Working root | `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` |
| HEAD | `0ce318515e125f83cc0b3acf85d076285176f89d`，与 caller 的 `0ce31851` 一致 |
| Base revision | 未指定；本次不建立 base→head diff |
| Primary target | [h3o_operator_costs/design.md](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/design.md)，当前 working-tree 全文，第 1–121 行 |
| Git 状态 | 无 tracked staged/unstaged changes；存在未跟踪的 `.cortexkit/`、`.pi/`、`remaining_measurements/` |
| Exclusions | 上述未跟踪目录、无关历史 worktree 的整体变更、其他模型的新建模，以及 H4b READ/lifecycle 的重新设计 |
| Reviewer | Execution context 指定的 `gpt-6.1-sol-sys-ptu`，`xhigh` |

下文 `design.md` 均指上述 primary target；其他相对路径均以 working root 为基准。外部 profiling/native artifacts 仅作为相关证据读取，不属于全面审查范围。

### 已读取的要求、规则与 artifacts

- Task `requirements.md` items 20–27；其中第 26 项（第 306–310 行）禁止 measured whole-step cost shortcut，第 27 项（第 312 行）要求先提交 CPU proposal，经 YC review 后才能 implementation/GPU profiling。
- Task `plan.md` 的 H3o row（第 160 行）及 Current execution route（第 166 行）；`progress.md` 的 H3o rows（第 243–244 行）和 2026-10-26 records（第 433–434 行）。
- Tickets `05-dsv4-flash-family.md`、`06-communication-measurement.md`、`09-predictor-bundle.md`、`14-decision-d-routing.md`。历史记录中的旧路线不覆盖 requirements 26/27。
- Applicable `AGENTS.md`：parallel ownership（第 559–594 行）、typed profiling ownership（第 596–612 行）、Development Gates/Module Size（第 781–801 行）；同时检查了 `docs/general/module-size-exceptions.md`。未发现需要追加执行的 applicable `CLAUDE.md`。
- ADR `docs/adr/0002-operator-costs-not-measured-step-tables.md`，尤其第 3、7–10 行。
- Caller 指定的 V4 execution/dispatch、DSA/GDN predictors、kernel-gap producer/consumer、V4 profiler、MoE profiler、measured CC、native case driver，以及可读 CSV、acceptance 和 kernel-coverage receipts。

### 重要假设与边界

目标是 P0/H800、P2-32/H200 的 TP8/DP2/EP16，以及 P2-64/H200 的 TP8/DP4/EP32；每个 P/D role 使用完整 pod，DP 是 request owner，EP ranks 不是独立 scheduler。

三个 case 的 inputs 均选择 `balanced_topk`；Frontier inputs 的 `speculative_decoding_config_enabled=false`，native engine inputs 未配置 speculative decoding。因此本次未把 MTP 缺席认定为 inventory 缺陷；proposal 应把这个排除条件写明并通过 runtime inventory 验证。

当前代码对未经 qualification 的 non-dummy/PD 路径主动拒绝。下列 findings 针对 **proposal 在解除这些 guards 前遗漏的必要工作或错误假设**，并非把现有保护性 guard 本身判作 bug。

## 2. Assessment

**Verdict：`changes_required`。** 方法方向和 whole-step diagnostic 隔离原则正确，但目前 proposal 不能据此批准为完整、可执行的测量与实现计划。最 consequential 的风险是：测量出一组局部准确的数值后，仍无法证明它们按实际 stream、phase、rank 和 PDD contracts 组成 native forward 的执行时间。

共确认 **P1 × 5、P2 × 3**。未根据现有资料推断最终 E2E error，也未把缺少 GPU 验证本身当成已发生的实现故障。

| Material requirement / caller question | Assessment | 结论 |
| --- | --- | --- |
| 1. Operator composition；step table diagnostic-only | 部分满足、部分违反既定路线 | §1 明确保留 diagnostic-only，未发现重新把 whole-step table 放回默认输入的提议。但 §4 改用未 qualified 的 attention layer union，并把各 scope 降为 attribution，与此前 approved Option A 不一致；见 F001。 |
| 2. 完整 inventory 与一次计费 | 不满足 | 缺少 `kv_cache_block_zero` 的测量计划；endpoint communication 尚无完整 backend contract；shared-expert phase 写错；目标 routing 与清单不符。见 F003、F008。MTP 可明确排除；`moe_shuffling` 应明确由 grouped envelope 包含而不另计。 |
| 3. Aux overlap、breakable graph | 不满足 | union 是 busy time，不能直接证明 critical-path latency；现有 graph-gap reducer 排除 eager segment 与 replay 接缝，V4 EP consumer 也绕过 gap wrapper。见 F001、F002。 |
| 4. Eager / first-tail | 不足，实际原因未验证 | CUDA_EVENT isolated scopes 不覆盖完整 serving host stream；61–69 ms 的原因、graph mode 和测量边界仍未确认。见 F006。 |
| 5. G1–G5 数据与预算 | 数据 qualification 不满足；预算未验证 | batch 上限方向合理，但有效 padded rows、planner identity、EP rank population 和 unequal SP patterns 未闭合。G5 在 H800 已确定需要，不只是可能需要。见 F004、F005、F007。 |
| 6. 最小完整 file plan / config / deletion | 当前不完整 | 漏掉必要 producer、scheduler/collective 接线与 contracts。现有 input paths 和 config flags 有望实现 no new config field，但文档尚未给出完整 binding。删除 `read_isolated_scope` 可以成立，前提是迁移其 qualification checks。 |
| 7. Acceptance | 部分合理，缺少结构性 gates | 5%/10% 可作为 interpolation gate；不能替代 ownership completeness、runtime/rank/phase coverage、joint-domain coverage、cold/warm identity 和 no-fallback tests。native E2E accuracy 留给 H4/H6、cross-check 不作 gate 的安排正确。 |
| 8. Sequencing / owner decisions | 需要修订 | 在大规模 producer/predictor 工作前，应先固定计费边界、目标 runtime、资格要求和最小 pilot；再重算 jobs、files 与时间。 |

requirements 27 的 **CPU-only proposal 形式及先 review 的顺序已满足**；proposal 的技术内容尚不完整。当前审查没有执行 implementation 或 GPU job。

## 3. Findings

### F001 — P1 / High confidence：attention layer kernel union 被用作 latency，绕过了尚未证明的 operator/stream composition

**位置与 contract：** `design.md` §4，第 58 行；§5，第 65、72 行。requirements 26 要求按当前 batch 的 operator costs 组合；此前 `task_memory/task_2026-10-23_v4_remaining_measurements/proposal.md:23–55` 批准的是 **有完整 ownership 和 GPU-dependency proof 的 Option A**，并明确区分 composite elapsed 的 Option B。

**触发与 expected/actual：** 在 CSA/HCA aux-stream overlap 或存在等待/idle 的 layer 上，proposal 直接以 `dsv4_attention_layer` 的 kernel-interval union 给 layer 定价，各 scope 仅用于 attribution。期望是从已 qualified 的 operator/fused-scope costs 和实际依赖组成完成时间；union 只统计至少一个 kernel 活跃的时间，排除所有 kernel 都不运行的区间，且不能单独证明 layer 完成边界。

**证据：** 已有 `audit/kernel_coverage_receipt_r2.json:4654–4655` 的 step3/layer0，union 为 **0.977821289 ms**，span 为 **1.835579102 ms**。这不是 native serving latency 的证明，但直接证明 union 与 elapsed envelope 不相等。该 receipt 的 `limits`（第 123151 行起）还说明没有 enclosing attention/forward boundary，缺少 event handles，实际 dependency edges 未证明；CUDA_EVENT 独立采集，其差值不能归入 CPU。

此外，当前 pipeline 已丢弃构造 union 所需的 interval coordinates：`dsa_kernel_trace.py:80–83,117–132` 仅累计 durations/counts；`dsa_vllm_runner.py:497–520` 输出 `time_ms` 和 `kernel_count`；`dsv4_attention_rows.py:216–267` 从这些 call records 聚合。只改 rows 文件不能重建 union。W6 和 q1 CSV 也没有 layer target，不能从 scope aggregates 求出真实 union。

**影响：** 可能拟合出非常准确的 busy-time predictor，却缺失真正的 layer latency；并失去 per-operator composition 的可审计性。shape-sensitive layer model 并非自动等同 whole-step shortcut，但不能仅因带 batch features 就视为满足已批准的 Option A。

**Focused correction：** 保留独立 operator 或明确 owning fused scopes，补齐 kernel ownership census、stream/event dependency、完成边界；同时记录 sum、union、span 和 stream-joined event elapsed。若选择 layer composite 路线，先向 YC 明确说明它改变了 Option A 的计费/attribution contract。不得把 layer total 分摊到 `dsv4_sparse_attention` 或任意 child operator。补列 trace reducer/runner/raw schema changes；新 KERNEL_ONLY target 必须来自 retained intervals，不能由 CSV medians 构造。

**验证：** 对 overlap、无 overlap、dependency wait、unowned kernel 各 case 检查完整 ownership 和 source-backed composition；证明最终完成时间与同一次调用的 joined boundary 一致，并确认 outer/child scopes 从不同时收费。

### F002 — P1 / High confidence：现有 kernel-gap producer/consumer 不能直接支撑 G4 的 breakable CUDA Graph 模型

**位置与 contract：** `design.md` §4，第 56 行；§3，第 52 行；§6，第 90、92 行。其承诺是 KERNEL_ONLY 加现有 `kernel_gap.csv` 后覆盖 decode replay。

**触发与 expected/actual：** 对 graph segment → eager attention → graph segment 的 forward，期望每个已声明 runtime 边界的 gap 计入一次。`frontier/profiling/kernel_gap/main.py:90–115` 只选择 `cudaGraphLaunch` correlation 下的 kernels，并在每次 graph launch 重新开始，跳过每段的第一个 kernel；eager kernels、eager 内的 gaps，以及 graph/eager 接缝不进入该统计。

**复现：** CPU 执行实际两个 reducer 的 AST，输入包含 6 个 kernels 的 serial breakable trace。完整 trace 的 gaps 合计 **72 μs**；现有 graph reducer 只选出 4 个 kernels、合计 **4 μs**。这是机制边界复现，不是 native error estimate。

还有两处未纳入 file plan 的接线：

- `frontier/scheduler/utils/expert_parallel.py:129–139` 的 V4 EP source path 直接调用 `_predict_dsv4_structural_execution` 并 assemble，绕过 `sklearn_moe_execution_time_predictor.py:1314–1323` 的 `_predict_layer_with_kernel_gap`。仅训练 models 不会自动给这些 FFN phases 加 gap。
- `kernel_gap.py:33–38,108–120` 要求 `time_stats.<op>.kernel_count`；W6 CSV 的 9 个 counts 是旧 `kernel_count.<scope>` 命名。proposal 所说“已有 counts”不等于已经兼容 count pairing/recording。

G4 的资源假设也不闭合：`kernel_gap/main.py:219–245,304–314` 使用 generic `LinearOpWrapper` eager chain，然后启动 native engine；第 285–289 行要求 profiler TP 与 engine physical TP 相同。它没有 V4 reduced-engine adaptation；generic TP sweep 还明确排除 V4（`linear_op/profiling_plan.py:323–353`）。因此“一 GPU、直接复用 existing profiler”不能证明测到目标 TP8 runtime。

**影响：** graph cost 可能漏掉 FFN gap 和 replay seams；或者把 graph-internal mean 套到 eager kernels。G4 的数据即使生成，也不必然是可消费的 TP8/V4 证据。

**Focused correction：** 在 file plan 加入 V4-specific runtime sampling、完整 segment classification、count normalization/recording 和 EP caller 接线。先用最小 pilot 验证实际 FULL/PIECEWISE/eager 边界；分别处理 graph-internal、eager-internal 和 seam costs，明确与 host terms 的 ownership。选择 physical TP8 测量，或证明 reduced TP1 对相关 runtime quantity 的适用边界，并据此重算 GPU jobs。

**验证：** 增加含 eager segment/inter-segment wait 的 trace test；用 sentinel gap/count 检查 attention、shared/gating、routed、endpoint phases 一次计费；缺 count 时 typed failure。native pilot 需证明各 segment 的计费边界和 kernel-count completeness。

### F003 — P1 / High confidence：file plan 漏掉完成 non-dummy PDD path 必需的 endpoint contracts 与 before-forward cost

**位置与 contract：** `design.md` §1，第 15、18 行；§3，第 43–51 行；§5，第 67–78 行。目标是 P0/P2 的完整 non-dummy PDD execution，不能留下 dummy scope。

**触发与 expected/actual：** 放开 V4 PDD admission 后，第一个有 cache-block assignments 的 forward、embedding 或 last-layer logits 会到达目前未定价的 scopes。期望这些工作由测量或 CC backend 定价；proposal 的 producer/file list 不能完成这些 contracts。

**证据：**

- `deepseek_v4.py:278–289` 为非空 `batch.cache_block_zeroing` 给 `kv_cache_block_zero` 赋 dummy。它未列入 inventory/G2，也没有相应测量坐标；不能用 token rows 代替 block-store 工作量。
- `deepseek_v4.py:291–293,328–334` 的 `dsv4_embedding_allreduce`、`dsv4_logits_gather` 仍赋 dummy。`frontier/operators/families.py:815–823` 把它们注册为普通 `OperatorSpec`，没有 collective alias/payload/device/domain builders；结构路径第 339–341 行只处理 `CommOperatorSpec`，所以不会自动接入 backend。
- V4 SP payload builder `families.py:424–435` 明确要求 `MONOLITHIC`。proposal file 6 的 attention admission 修改不能代替 communication contract 的 PDD admission。
- inventory 第 50 行把 shared expert 放在 `post_combine`；实际 `timing_operator_times`（`deepseek_v4.py:44–47`）将其放在 `pre_dispatch`。phase contract 会影响 EP scheduling，不能只校验 step sum。

**影响：** 按列出的 files 实现后，PDD 仍会拒绝 payload，或端点/zeroing 保留 synthetic durations，直接违反 no-dummy 和 E2E acceptance。

**Focused correction：** 补充 `families.py`/必要 payload helper 的最小修改，建立 endpoint 的物理布局、selected logits rows、payload、collective/domain/cardinality contracts；明确 `kv_cache_block_zero` 的 native invocation 和测量 ownership。把清单展开为 once-only ledger，包括 first-pre ×1、cross-layer fused mHC ×42、attn-post/ffn-pre ×43、final-post ×1 和 endpoints；shared expert 回到 `pre_dispatch`。若保留新的 layer composite，还需解决 `time_components.py:111–123` 和 `deepseek_v4.py:60–63` 的 canonical timing ownership，而非任意新增 `op_times` 名字。

**验证：** 在 non-dummy PREFILL/DECODE、DP2/DP4 上覆盖 first/last layer 与非空 block zeroing；让任何读取 `_dummy_execution_time` 的路径立即失败；检查 physical payload ledger、phase totals、once-only invocation 和其他路径回归。

### F004 — P1 / High confidence：rank0 isolated MoE acceptance 被当成 serving 输入，G3 没有补齐 EP population qualification

**位置与 contract：** `design.md` §2，第 31 行；§3，第 48 行；§6，第 89、92 行。完整 operator path 必须为每个执行的 EP population 提供合格 cost，不能仅覆盖 rank0 的 M 值。

**触发与 expected/actual：** P0 的 balanced decode 在 gathered `M=16/32` 时有 active 和 zero-assignment EP ranks；mixed prefill 也有不同 gate local rows。期望 predictor 按实际 rank-local expert counts、alignment 和 backend workload 定价。proposal 直接使用“accepted rows + M48/64”而没有 rank-equivalence/admission 工作。

**证据：** `moe/results/primary_acceptance.json:33–35,74–78` 限定为 physicalTP1/EP16/rank0/local-experts16，并明确 `native_tuple_cost_admission=false`、`simulator_consumption=false`。其 next action 要求先证明 native tuple/rank population 和 dependency comparability。`moe_grid_design.md:32–45,61–70` 还明确：

- `N_d → L_d=ceil(N_d/8) → M=8ΣL_d`；gate 输入是 `L_d`，grouped GEMM 输入是 `M`。
- M16 有 10 个 zero-assignment ranks，M32 有 4 个；rank0 数据不 qualify 所有 ranks。
- grouped envelope 已包含 alignment、两次 GEMM、`moe_sum`；standalone `moe_shuffling` 不能再加。

增加 M48/64 不消除上述 qualification 限制。现有 producer `moe/main.py:145–150` 仍固定 `rank=0`，多 GPU profiling 不自动变成不同 EP-rank 测量。H200 EP32 的 local expert width 为 8，与 EP16 的 16 不同。

**影响：** 完整 replay 会出现未测 rank coordinates，或通过误用 rank0 时延/放宽 bounds 才能完成。后者可能给出错误的 EP critical-path maximum。

**Focused correction：** 将 accepted rows 定位为 isolated training candidates，增加明确的 cost-admission gate。按实际 routing algebra 枚举 EP16/EP32 的 rank-local expert-count/alignment classes，覆盖 zero、light、heavy populations；可以在单 GPU 上测等价 rank classes，不必机械增加 GPU 数。列出 H200 的完整 M grid，而不是“same as G3”：DP4 合法 vectors 如 `[6144,1,1,1]` 得 M6168，四个 6144 得 M24576；应先确认哪些是目标 replay 的实际/承诺 coverage。明确 gating 只消费 L-grid，shuffle 为不另计的 included scope。

**验证：** 每个目标 EP lane 都能绑定 measured/equivalent workload，并验证关键 expert-count transitions 和 zero-load 行；所有未 qualified populations typed failure。独立检查 per-rank costs 与 critical-path maximum，不能仅检查 rank0 held-out error。

### F005 — P1 / High confidence：byte-range coverage 会放过未测的 unequal SP combine，且 V4 绕过已有 exact-pattern consumer

**位置与 contract：** `design.md` §2，第 32 行；§3，第 47、49 行；§6，第 91–93 行；§8，第 112 行。Ticket06 要求实际 groups/payloads、qualified reuse 和 explicit coverage，而不是只检查 byte min/max。

**触发与 expected/actual：** 当一个 DP owner 有 6144 prefill tokens、peer 为一行 idle/light forward，SP widths 为 `(768,1)`，EP16 gathered M 为 6152；V4 output RS input 为 **50,397,184 bytes**。即使 bytes 在 uniform curve 内，unequal combine 仍需要其真实 per-root Reduce pattern 的测量。

**证据：** `measured_cc_backend.py:132–151` 将 single-active samples 与 uniform curves 分开；第 217–229 行要求 exact ordered SP vector、bytes 和 group，缺失即 error。可是 `operators/sequence_parallel_moe.py:133–146` 仅对 `expert_parallel_alltoall_combine` 调用该 consumer，V4 名 `dsv4_ep_output_reduce_scatter` 被跳过；`sklearn_execution_time_predictor.py:6302–6306` 随后回到 scalar pricing。AST 检查也复现 V4 返回 `None`、generic name 才调用 unequal backend。

H800 的 small-payload 缺口已确定：EP16 uniform all_gatherv 最小 **12,352 B**、RSv 最小 **196,608 B**；V4 L1 的 hidden AG 为 **8,192 B**、weights/ids 各 **24 B**、M16 RS 为 **131,072 B**。design 第 32 行的 384 KB 是 EP32 RSv 最小点，不是 EP16。G5 已是必要补测。

**影响：** CPU byte-only check 会把未 qualified 的 ragged shape 判为 covered，随后选中不同 collective kernel family 的 uniform cost；small-only G5 也不足以补齐 large unequal patterns。

**Focused correction：** 给 V4 combine 接上 ordered SP-vector consumer，在 file plan 加入该 binding；coverage matrix 包含 operation、group/cardinality、library/runtime、ordered local rows、bytes 和 per-root kernel identity。G5/G5h 明列需要的 small points及实际 large ragged patterns/排列；未测 pattern 不得沿 scalar curve 插值。复用 H800 curves 前按同一标准 qualification。

**验证：** 同 bytes/different SP vector、交换 active DP owner、缺 pattern 均 typed failure；已测 pattern 走 exact lookup，uniform predictions 不变。检查 DP2/DP4 prefill/idle 和 unequal decode 的所有实际 vectors。

### F006 — P2 / High confidence：isolated CUDA_EVENT scopes 不能保证覆盖 eager first-tail 的 serving host cost

**位置与 contract：** `design.md` §4，第 57 行；§9，第 118、121 行；§1 的 no-dummy/完整 step 目标。

**触发与 expected/actual：** eager prefill/first-tail 在 scope 之间、forward 前后，或 breakable replay seam 上等待 host launch 时，proposal 期望 CUDA_EVENT scope rows 已包含其成本，并把 CPU tables 推迟到发现大差距以后。

**证据：** `dsa_vllm_runner.py:215–247` 在 `function()` 紧邻前后记录 events；scope 外的 Python/launch 工作不在区间内。V4 runner 的说明（`dsv4_attention_runner.py:24–26`）还明确 aux-scope events 排除 join wait。isolated/reduced engine 的 event idle 也不自动等同 43-layer async serving 的 host stream。

现有 V4 stage-owned CPU fields仍由 dummy base 设置（`deepseek_v4.py:360–375`）。仓库已经有独立 device/host two-stream 机制（`sklearn_execution_time_predictor.py:992–1006,7125–7140` 的 `forward_launch`/`forward_drain`），但 proposal 未列绑定工作。

**影响：** 即使每个 CUDA_EVENT scope 的拟合通过 5%/10%，composed eager step 仍可能缺 host/runtime 边界。**尚不能断言 61–69 ms 已证明 host-bound**；相关 native scheduler-record step 与 profiled trace interval 还不是统一测量边界，见 `test_report_2026-10-26_h4b_k1_reprice.md:159–161`。

**Focused correction：** 现在把 runtime/host-boundary measurement 纳入 scope，先辨明 first-tail 的 mode、forward 起止及 host/device overlap；据此决定是否使用既有 CPU-overhead inputs/flags，或明确缩小本阶段的 step-cost 承诺。不能通过 native whole-step 减 operator sum 的 residual 来制造 CPU cost。使用 CUDA_EVENT 和 CPU terms 时需说明互斥 ownership，防止重复收费。

**验证：** 同 workload 的 isolated scopes、native forward GPU boundary、host forward launch/drain 和 serving step boundary分别对齐；覆盖 eager、graph seam、async scheduling。CPU modeling 关闭时用明确零值政策，开启时不能残留 dummy。

### F007 — P2 / High confidence：model key 和 features 丢失已有的 workload distinctions，bounds 不能证明 coverage

**位置与 contract：** `design.md` §5，第 65–66 行；§1，第 16–17 行。typed coverage 必须区分真实执行工作，而非仅检查若干 scalar 的独立范围。

**触发与 expected/actual：** 同一 variant 的 tile-schedule planning/reuse layers，或相同 aggregate query/context features 但 compression boundaries 不同的 batches，进入同一个 model/exact-lookup key。期望它们按执行分支、物理 rows 和 kernel metadata 区分；proposal 的 key 只有 `(scope, variant, measurement type)`。

**证据：** `dsv4_attention_plan.py:16–19,187–192` 专门保留 planning/reuse 两类 layer；`dsv4_attention_rows.py:234–240` 也把 `plans_tile_schedule` 放入 group key。proposal 删除了这个维度。`frontier/attention/deepseek_v4.py:161–192,215–250` 已区分 effective compute rows、actual rows、每 request 的 compressed writes，以及 FULL/PIECEWISE 的不同 branches。

CPU arithmetic 检查还得到：q=`[1,1,1]`，contexts A=`[6147,6145,6150]`、B=`[6144,6148,6150]`，query/sequence counts、context sum18442/max6150，以及 post-query compressed history sum4610/max1537 均相同，但 CSA new compressed writes 为 **1 与 0**。这是在“compressed length 使用 post-query history”解释下的碰撞；若同时加入 prior/post sums，该示例可区分。它说明 features 的精确定义必须写清，不能只写“compressed key length”。未据此声称实际 timing error 必然超过阈值。

**影响：** exact lookup 合并不同工作；随机 row holdout 可能掩盖 categorical 合并和邻近-context leakage。独立 min/max bounds 也会接受未测的 joint phase/geometry combinations。

**Focused correction：** 优先从现有 V4 workload/metadata owner 派生 features，明确 mode、planner identity、scheduled/effective/local/gathered rows、phase semantics、compression-write boundaries 和 scope-specific features。增加 joint coverage/admission，而非只有 marginal bounds。迁移 `read_isolated_scope` 时保留适用的 input hash、runtime/backend、layout、ownership 和 typed contract validations（`deepseek_v4.py:665–827`），不能只保留 lookup。

**验证：** 增加 planner/reuse、padded/unpadded、same aggregates/different writes、first-tail/ongoing decode 的 collision tests；按完整 workload group 划分 held-out sets。即使所有 scalar 在 bounds 内，未测 phase/mode combination 仍需 typed error；cold/warm artifact identity 一致。

### F008 — P2 / High confidence：routing inventory 与目标 balanced_topk runtime 不符，gating producer 也放错 owning family

**位置与 contract：** `design.md` §3，第 46 行；§5，第 73、78 行；§9，第 120 行。profile 必须测实际执行的 operator path；`AGENTS.md:610–612` 明确 shared expert 属 linear-op，routed gating/shuffling/grouped GEMM 属 MoE。

**触发与 expected/actual：** P0/P2 在 `balanced_topk` 下执行所有 43 个 routers。proposal 却安排 score top-6/hash top-k，并说前三个 hash layers 的 load 来自 hash table；同时把 gating/routing 写入 `linear_op.csv`。

**证据：** 三个 case 的 `case/inputs/native_config.json:42` 均为 `balanced_topk`。H800 四个 role-head rank0 inventories 的 `inventory.moe_routing.routers` 全部 43 项都是 `RoutingSimulatorRouter`，包括 layers0–2。native source `vllm/model_executor/layers/fused_moe/router/router_factory.py:114–120` 在 hash/score router 选择之前返回 simulator router。`moe_grid_design.md:62–64` 已说明 balanced assignment identity 和 native gate L-grid 的区别。

**影响：** G2 会测当前 case 未执行的 selection kernels，cost path 可能错误加入 hash/score selection；把 routed gating 放到 linear-op family 会破坏 typed single-owner contract。真实 hash/score routing 也不能因此被视为已 covered。

**Focused correction：** 在 proposal 明确目标是当前 `balanced_topk` cases，按 source/inventory 确认仍执行的 gate projection，并测实际 balanced routing path；保留 layers0–2 的 architecture identity，但不要给未执行的 hash selection 计费。真实 routing 留在 ticket14 的独立 qualification 范围。gating/routing 使用 MoE producer/typed contracts 和 local L-grid，shared expert 继续用 linear-op producer。

**验证：** 相同 geometry 下 runtime-path mismatch 必须拒绝；确认 43-layer inventory、producer routing signature 和 consumer binding 一致；typed family conflicts 失败；若将来开启 real routing，则不能消费 balanced rows。

## 4. Verification

### 已执行的 read-only checks

环境：working root 如上；标准库检查使用 `/usr/bin/python3` 3.10.6。该解释器未安装 pytest，随后发现既有 `/data/ycfeng/envs/frontier-py310/bin/python` 具备 pytest，使用它完成测试；没有安装或升级软件。

| Check / exact command | Expected | Observed / status |
| --- | --- | --- |
| `git rev-parse HEAD` | 与 caller HEAD 一致 | `0ce318515e125f83cc0b3acf85d076285176f89d`；PASS |
| `git diff --name-only`；`git diff --cached --name-only` | design review scope 中没有 tracked local diff | 两者为空；PASS |
| `git status --short`，review 前后 | Git 状态不变 | 都只有 `.cortexkit/`、`.pi/`、`remaining_measurements/` 三个 untracked directories；PASS |
| `nl -ba task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/design.md`，及相关 sources/records 的 `nl`/`rg` inspection | 检查实际 sources、要求和 artifacts | 已读取；关键位置列于 findings。不是 test execution |
| `/usr/bin/python3 -B /data/ycfeng/tmp/codex-review/9c8b2435-c3f3-468e-8c41-6a504ff54fda/review_checks.py` | 验证 reducer 边界、features arithmetic、artifact admission 和 CSV coverage | exit0；PASS，复现 F002 omission/F007 conditional collision，确认 MoE admission=false、router identity、count schema 和 CC minimums |
| `/usr/bin/python3 -B /data/ycfeng/tmp/codex-review/9c8b2435-c3f3-468e-8c41-6a504ff54fda/unequal_binding_check.py` | 验证实际 V4 name 是否调用 unequal consumer | exit0；PASS，复现 F005 的 name-guard bypass |

Scratch scripts 只在指定 private scratch directory 内写入。`review_checks.py` SHA256 为 `30599f2454329f1d0acaddacd909b7d8aa3d12d54dcc342971a4884b298082de`。AST checks 使用实际函数和受控 stubs；不等于完整 simulator/native execution。

### 已执行的 pytest

Exact command：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/codex-review/9c8b2435-c3f3-468e-8c41-6a504ff54fda/review_pytest tests/unit/test_kernel_gap_profiler.py tests/unit/test_dsv4_attention_workload.py tests/unit/test_measured_cc_backend.py::test_unequal_combine_exact_match_pools_samples_without_changing_uniform_curve tests/unit/test_measured_cc_backend.py::test_unequal_combine_requires_exact_ordered_rows_bytes_and_group tests/unit/test_measured_cc_backend.py::test_unequal_combine_does_not_interpolate_between_nonmonotone_points tests/unit/test_dsv4_attention_profiling.py::test_rows_split_variant_and_tile_schedule_role > /data/ycfeng/tmp/codex-review/9c8b2435-c3f3-468e-8c41-6a504ff54fda/review_pytest.log 2>&1
```

Expected：既有 parser、workload、ordered-vector backend 和 row-grouping contracts 通过。Observed：**81 passed in 1.74s，exit0，无失败/skip**。日志位于上述 scratch path。PASS 证明当前 contracts，而非 proposal 的新模型正确。

### NOT_RUN 与 coverage limits

- 新 V4 predictor implementation/training、held-out accuracy、cold/warm cache、P0/P2 operator-path E2E：**NOT_RUN**，proposal 尚未实现。
- GPU profiling、physical TP8 breakable replay、host/native timing parity、per-rank timing equivalence：**NOT_RUN**，本次 review 不授权 GPU jobs。
- 77-case fidelity matrix、dummy smoke matrix、完整 unit suite：**NOT_RUN**；当前没有 source change，本次运行的是与 claims 直接相关的 81 个 CPU tests。
- Existing CSV/acceptance 可读不代表 serving qualification。kernel audit 的 union/span 是 retained-data evidence；没有重跑其 GPU campaign。
- 未验证 16 H800 / 40 H200 GPU-hours 或 6–8 working days。现有小型 campaigns 只支持部分 job 的量级参考，不能覆盖修订后的 runtime/rank/ragged measurements 和 retry 成本。

## 5. Open questions and next actions

### 尚未确认的问题

| 问题 | 需要的证据 / owner decision |
| --- | --- |
| 61–69 ms first-tail-alone 的主要原因是什么？ | 每个 native step 的 semantic phase、actual graph mode、capture/effective width；对齐 host forward、GPU forward 和 scheduler-update boundaries。现有数字不能单独证明 host-bound。 |
| 哪些 measured composites 符合 operator-level route？ | YC 明确选择保留 approved Option A，或批准具体、shape-aware、single-owner fused/composite contract；不能笼统把 layer elapsed/union 当每个 operator 的替代。对应 F001。 |
| CPU/runtime overhead 是否本阶段必须定价？ | 先做最小 boundary pilot。若 material，则纳入已有 CPU table/two-stream mechanism；否则记录适用范围和证据。对应 F006，不宜等全部模型完成后才决定。 |
| Reduced TP1 对哪些 quantities 能 qualify physical TP8？ | 各 scope 的 kernel/backend/layout identity、graph/eager equivalence 和 dependency/host 边界；physical dimensions 一致并不证明 runtime timing 一致。 |
| G1–G5 的最终数量与预算是什么？ | 明列 workload vectors、categorical identities、每格 repeats/warmups、held-out groups、startup/pilot 时间和 bounded retry allowance；用 pilot timings 重算 GPU-hours/elapsed days。 |

### 修订依赖顺序

1. **先固定 cost/measurement contracts：F001、F006。** 定义 operator/fused-scope owners、stream completion、host/device boundaries，以及哪些 native numbers仅作 diagnostic。把 qualification completeness 写入 §1 acceptance。
2. **固定目标 runtime 和 workload keys：F007、F008。** 明确 balanced routing、MTP-off、planner/reuse、actual/effective/SP rows、phase/mode；从实际 target vectors生成 coverage matrix。
3. **补完整 file plan：F002、F003、F005。** 纳入 raw interval/segment producer、kernel counts、EP gap caller、PDD payload/endpoints、zeroing 和 unequal-vector binding。`read_isolated_scope` 只有在适用 validations 迁移后才删除；为超过 2,000 行 modules 更新本次 scoped cleanup/split justification。
4. **补数据 admission 与 exact grids：F004、F005。** 区分 isolated acceptance 与 serving cost admission；覆盖 rank-local equivalence classes、EP16/EP32 widths、local gate L、gathered M、小 payload及大 ragged patterns。减少未执行 hash/score selection 的 jobs。
5. **最后重写 §6–§9 并提交 YC review。** CPU dry-run/coverage check → 最小 runtime pilot → qualification → producer/predictor implementation → grouped held-out/identity/no-fallback checks → operator-path replays → diagnostic cross-check → H4/H6。新增工作必须反映在 files、GPU-hours 和时间估算中。

建议保留 median5%/p90 10% 作为按 workload group 划分的 per-scope interpolation gate，并追加：完整 kernel ownership、once-only reconciliation、phase/mode/rank admission、joint coverage、typed identity、cold/warm parity、no-dummy/no-step-table default-dispatch checks。对合法 no-op/zero scopes 单独定义验证，避免 relative-error 分母为零。native cross-check仍为 diagnostic，不因它接近 native 就提前宣称 Frontier predicts。

`no new config field` 目前有可行空间：已有 attention/linear/MoE/CC/kernel-gap/CPU input paths 与 runtime flags 足以承载多项修订；应先给出明确 binding，再决定是否存在不可表达的最小缺口。H200 resource 限制继续按 requirements 21 计算 running+queued 合计最多64、non-preemptible，并将所有 ≥32-GPU runs 的 soft-gate 规则写清，不能仅在 G5h-EP32 一行提及。

本次未修改 source、Git state、configuration 或 task records；仅创建 private scratch verification artifacts。

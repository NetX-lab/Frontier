# H3o-(a) revision 2c 独立设计审查，round 3

## 1. 审查上下文

本次按 caller 指定进行 **design review**，审查 revision 2c 的完整设计及其 source/evidence，而非整个分支的 diff。重点是 F009–F015、F001/F002/F006 的 closure，以及 DP rendezvous、local classifier、V2 dummy workload、per-stream gap、replay ownership、文件范围和资源规则。

| 项目 | Resolved scope |
| --- | --- |
| Working root，以下记作 `W` | `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` |
| Branch / HEAD | `feat/glm53-dsv4-vllm028` / `4b4b3ab054a4ae44c36880a161486ce511791935`，与 caller 一致 |
| Task root，以下记作 `T` | `W/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800` |
| Primary target，以下记作 `D` | [h3o_operator_costs/design.md](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/design.md:1)，revision 2c，第 1–266 行；收尾检查确认 working-tree bytes 与 `4b4b3ab0` 的 blob 相同 |
| 历史参照 | revision 2b：`40299c0a9062c475bddf8261df2a4744fac04a11`；指定设计路径的 `40299c0a..4b4b3ab0` diff 为 +84 / −57 行 |
| Native source root，以下记作 `N` | `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm`；其 worktree HEAD 为 `2cf0a6915ce544dc493a0990f2ea38d81601128a`，Git status clean |
| Staged / unstaged tracked changes | 初始及收尾检查均无 |
| Untracked exclusions | `.cortexkit/`、`.pi/`、`remaining_measurements/`；审查期间另出现 `tests/analysis/v4_runtime_census.py`，不属于 target HEAD，本轮只读取了其开头作为并发工作上下文，未将其视为 revision 2c 的已验证实现 |
| 其他 exclusions | 整个分支的无关变更、H4b READ/lifecycle 的重新设计、real score/hash routing、PD-AF 新支持、non-V4 DP gap 修复、PP > 1 的 V4 qualification |
| Reviewer model / effort | caller execution context：`gpt-6.1-sol-sys-paygo` / `xhigh` |
| Scratch，以下记作 `S` | `/data/ycfeng/tmp/codex-review/d8d732a2-5682-4e9c-a111-9418dcfebc85` |

上述 `W`、`T`、`N`、`D`、`S` 在全文均按表中的绝对路径展开。其他 `frontier/`、`tests/`、`docs/` 路径以 `W` 为根。

已读取适用的 ancestor/root `AGENTS.md`，未发现其他适用的 `CLAUDE.md` 或 nested instructions。相关要求包括 `AGENTS.md:559–609` 的 DP/TP/EP ownership 与 typed operator contracts、`698–715` 的 TTFT contract、`781–801` 的 cleanup/split、模块规模和 fidelity gates，以及 `docs/general/module-size-exceptions.md` 的现有边界说明。

已读取以下 governing records：

- `T/requirements.md:309–352`，items 26–33：operator-composed costs、diagnostic-only step table、balanced routing identity、revision 2c 的工作决定和 YC 的 D1(a)、D7(a)、growth/budget、round-3 approval。
- `T/requirements.md:251–253`，item 21：H200 mainline 的 session running + queued ≤ 64、`step_main`、non-preemptible 和 ≥32 GPU soft gates。
- `T/plan.md:156–166`，H3o/H4 的范围与依赖；tickets `05-dsv4-flash-family.md:399–470`、`09-predictor-bundle.md:380–409` 及 `06-communication-measurement.md` 的相关 contracts。
- `docs/adr/0002-operator-costs-not-measured-step-tables.md:3,7–10`。
- round-2 report `T/h3o_operator_costs/codex-review-4678dfdc-a4d3-47a2-8c00-de2e7803a915/review.md`，以及作者的 spot checks / positions `T/h3o_operator_costs/review.md:39–71`。

物质假设是：target 三个 cases 均使用 pinned V2 `GPUModelRunner`、PP1、`FULL_AND_PIECEWISE`、MTP/prefix caching off、all-43-layer `balanced_topk`。本报告判断 proposal 是否足以指导实现；不会把当前尚未 qualified 的 serving guards 本身当成应直接删除的 bug。设计层面的 resolved 不代表 proposed implementation 已完成。

本轮仅向 private scratch 写入检查脚本和测试输出，没有修改 source、Git、configuration 或 task records，没有安装软件或提交 GPU jobs。任务记录对作者后续实现的授权不改变本轮 review-only 范围。

## 2. Assessment

**Verdict：`changes_required`。确认 P1 × 1、P2 × 2。** revision 2c 的主要方法修订成立，但 replay bootstrap 仍依赖不成立的跨模式 signature identity；native pre-DP device work 和 serving dummy 的 terminal ownership 也尚未完整表达。

### Closure：F009–F015、F001/F002/F006

以下均为设计层面的 closure。

| 原 ID | 状态 | 已闭合内容和剩余项 |
| --- | --- | --- |
| F009 | **Partly resolved** | `D:135` 已明确先计算共同 rendezvous，再执行 descriptor-dependent preparation 和第一层 attention；原 101 ms / 110 ms 问题的 correction 已写入 A13(d)。但 `D:143` 将 `kv_cache_block_zero` 一并移到 barrier 后，不符合 V2 runner。见 F017。 |
| F010 | **Resolved** | `D:65–66` 明确 canonical scope costs、composed elapsed 和 scope offsets 的不同用途，指定真实 getter、event/block consumers 与 trace wiring；entries 2、27、28 对应接线。实现 acceptance 尚未运行。 |
| F011 | **Partly resolved** | `D:136` 已对齐 descriptor rows/request counts、padding、dummy metadata、FULL re-staging，保留 request-free scheduler identity。原 eager/no-metadata cause 已有明确 correction；§4 的 dummy once-only ledger 仍与 real terminal scopes 冲突。见 F018。 |
| F012 | **Resolved** | `D:108–136` 描述 native local dispatch、uniform/max-query redispatch、token-only PIECEWISE padding 和 PREFILL metadata；A13(c) 从 Frontier batches/config 开始。corrected census 可复现。 |
| F013 | **Resolved，作为 qualified approximation** | `D:75` 改为沿各 stream/dependency path 计算 residual idle，并明确单一 scalar 是 approximation；`D:27` 的 A4 按 mode/topology/variant/planner/workload group 检查并 fail closed。其 GPU qualification 依赖尚未闭合的 replay route，见 F016。 |
| F014 | **Resolved** | `D:65` 明确 threshold、Breakable serialization、indexer short-context/capture state，threshold 从 runtime identity 读取并拒绝不匹配 rows；S1/P1h 覆盖条件分支。 |
| F015 | **Partly resolved** | stage 1a 和 entries 13/17 已将 bootstrap 放到 pilot 前，解决依赖顺序；但 eager ownership pass → replay signature matching 的方法仍不能覆盖 native capture 改变 branches/streams 的情形。见 F016。 |
| F001 | **Partly resolved** | per-operator Option A、conditional composition、elapsed consumer 和 offsets 已补齐；qualified graph replay ownership/pilot 仍受 F016 阻塞。 |
| F002 | **Partly resolved** | segment census、per-stream approximation、gap-before-maximum、once-only consumer 和 EP wrapper 已明确；可靠的 replay intervals/ownership 和 A4 尚依赖 F016。 |
| F006 | **Partly resolved** | host-only D2、explicit zero、conditional R2/R2h、probe split 和 V2 binding precheck 均已明确。完整 preparation timeline 仍需 F017；实际 probe binding 和 split accounting 尚未验证。 |

### 对 caller 八个问题的回答

| 问题 | 判断 |
| --- | --- |
| 1. Closure | 如上表。不能给出全部 resolved。 |
| 2. New defects / source claims / counts | 确认 F016–F018。corrected census 的关键 counts、28 entries、+3,310 / −330 的 estimate 和 GPU caps 算术一致；未发现另一个需要单独立项的 material count defect。 |
| 3. DP rendezvous / downstream behavior | barrier-before-post-DP preparation 和 model 的方向正确；real lanes 的 pre-DP zeroing 时间位置错误。dummy lanes 跳过 real request updates，因此不能统一为全体 lanes 都只有 pre-DP host work。PP1 限制合理。尚不能宣称 stage final timing、非零 host launch shares、KV/NIXL 和 dummy lifecycle 已经不受损，见 verification limits 和 O2。 |
| 4. Local classifier / smallest field change | 对三个 cases 正确。FULL bound 是 captured size 同时受 `max_num_seqs × decode_query_len` 和 largest capture size 约束；这里为 128。first tail 的 uniform count 为 `None`，得到 PIECEWISE；PIECEWISE 只 pad tokens，最大 capture size 以上为 NONE。PREFILL 在新值下创建 metadata 使其 idle dummy 可用 FULL。V4-only 新 value 是最小且合理的兼容性边界，符合已采纳 D7(a)。 |
| 5. V2 dummy workload | physical rows、request count 和 attention metadata 规则正确；FULL width W 使用 W 个 query=seq=1 dummy slots，PIECEWISE width W 使用一个 query=seq=W slot，NONE 保持本地一个 token/slot；FULL metadata 还按 capture path re-stage。terminal operator ledger 不完整，见 F018。 |
| 6. Per-stream gap | 在获得真实 mode-qualified intervals 和 source/event dependencies 后可测量，也可作为明确受 A4 限制的 approximation。单靠 timestamps 不能识别全部 dependency waits；S1 的 source graph 必须参与 reducer。A4 per-group 的设计足以作为经验 stop gate，不是 correctness proof，且本轮未运行。 |
| 7. Replay route / ownership | reduced engine 可采用 native FULL/Breakable capture 机制；当前 eager-signature transfer recipe 不足以建立 ownership。one-to-one matching 只能验证所选 mapping 的一致性，不能证明跨 capture state 的 owner 相同。见 F016。 |
| 8. Smallest complete change / resources | 未发现可据 source 删除的主要 work unit。entries 13/17 的 ownership bootstrap、entries 2/3/20–22 的 pre-DP device accounting、dummy ledger/tests 尚需修正。资源规则和 cap 算术正确；实际 grids、queue/runtime、pilot 是否能在 cap 内完成仍未 verified。 |

### 关键正确性判断的依据

local classifier 的 source 是 `N/v1/worker/gpu/cudagraph_utils.py:75–100,240–272,382–410`。FULL candidates 要满足 uniform count、request bound 和 token bound；PIECEWISE candidates 不 pad requests。DP redispatch 使用 local request count、synchronized token count、common uniform count 和 max query length，见 `N/v1/worker/gpu/dp_utils.py:52–95`。这支持 revision 2c 的 rule，而不支持将两个历史 Frontier graph values 改义后用于其他模型。

V2 dummy 的 request-count rule 来自 `N/v1/worker/gpu/model_runner.py:652–675,1500–1546` 和 `N/v1/worker/gpu/input_batch.py:116–206`。`N/v1/worker/gpu/model_states/default.py:135–155` 还说明 FULL 的 capture metadata 使用 worst-case `max_model_len`，不能把 physical dummy seq=1 自动解释为 ordinary eager short-context metadata。

composition consumer 的接线具有真实 downstream 意义：`frontier/entities/execution_time.py:910–916,1138–1147` 将 attention getter 传给 event/block duration。`D:66` 指定替换 V4 serial sum，因此原 F010 不是仅增加 predictor 内部 diagnostics 的修订。trace 实现还必须改变真实 timestamp/cursor 行为，而不能只把 offset 写入 metadata。

## 3. Findings

新 finding 从 F016 起编号，分别指出 revision 2c 的 remediation 中仍需修正的原因。

### F016 — P1 high — eager kernel signatures 不能直接建立 FULL / PIECEWISE replay ownership

**Confidence：高，针对 source control-flow 不一致；具体 GPU matcher failure 尚未运行。**

**位置：** `D` §5 stage 1a，**165 行**；§6 entry 13，**189 行**。相关 source：`N/models/deepseek_v4/attention.py:302–307,376–399,466–493,906–958`、`N/utils/multi_stream_utils.py:45–61`、`N/compilation/breakable_cudagraph.py:143–148,160–171,195–214`。

**违反的 contract：** A2 要求 replay window 中每个 kernel 有唯一且正确的 scope owner；A4 必须比较同一 qualified replay 的 per-scope costs 和 joined elapsed。required pilot route 不能将普通 eager pass 的 branches/streams 当作 capture 的恒等副本。

`D:165` 的 recipe 是：eager hook pass 得到 `(name, grid, block, stream) → scope`，随后 capture/replay 同一 call，再一对一匹配。native source 明确改变其中的 stream 和 branch。

1. V2 使用 narrow `_prepare_and_attn` path，input preparation 留在 captured region，见 `attention.py:302–307,376–399`。HCA compressor 使用 `maybe_execute_in_parallel`，CSA indexer 内也使用它，见 `466–493,952–958`。
2. helper 在 Breakable capture active 时执行：

```python
if BreakableCUDAGraphCapture.is_active():
    aux_stream = None
```

普通 eager pass 的 compressor 在 auxiliary stream 上；对应 PIECEWISE captured work 则在 current stream 上。相同 name/grid/block 也不能匹配原来的完整 signature。
3. CSA indexer 的 short-context branch 还要求 `not torch.cuda.is_current_stream_capturing()`，见 `attention.py:906–930`。普通 seq=1 eager dummy ownership pass 会走 short-context fill，而 FULL capture 会走 projection/quantization path。FULL 的实际 re-staging 又使用 capture metadata，见 `model_states/default.py:151–155`。若 eager reference 特意使用 mode-qualified capture metadata，可以避免其中部分差异，但 `D:165` 未定义这样的 reference，且仍不能消除上述 PIECEWISE stream 差异。

**Reachable trigger：** target first-tail forwards 使用 PIECEWISE；其 43-layer model 包含 HCA/CSA。required dummy/short-context coordinates 和 capture branches 也进入 C1/G1h/P1h 的覆盖范围。此问题不需要假设未知模型或 PP > 1。

**Expected versus actual design：** expected 是 replay kernels 的 ownership map 对其实际 capture kind、metadata 和 stream roles 有依据；当前 recipe 从另一个 execution state 直接转移 ownership，并在合法差异出现时停止 window。fail-closed 能防止错误 rows 被准入，但不能让 required pilot 完成。

**本轮 reproduction：** scratch `S/round3_source_counterexamples.py` 执行了 exact native helper 的 AST-extracted function，使用 CPU stream/event stubs。观察到：

- `Breakable_capture=False`：`query@current, compressor@aux`。
- `Breakable_capture=True`：`query@current, compressor@current`。
- exact native short-context predicate，在 seq=1 时 ordinary eager 为 True，capture 为 False。

这证明 source routing 假设不成立；没有把该 CPU check 宣称为真实 CUDA kernel matching 或 GPU elapsed reproduction。

**Practical impact：** entries 13/17 的 bootstrap 可能在合法 PIECEWISE windows 上无法通过，阻塞 P1h、A4 和后续 grids；若实现通过忽略 stream/branch 差异来强行匹配，则可能把 replay kernels 归给错误 owners，损坏所有后续 per-operator costs。

**Focused correction：** 保留 stage 1a，但先建立 target-mode/capture-state 的 ownership reference。记录 capture-time scope、graph node/occurrence identity 或等价的可审计 mapping；使用 source-defined stream roles，区分 captured graph intervals 和 PIECEWISE eager-break intervals。eager pass 可以辅助检查，不能单独作为跨模式 ownership authority。不能仅删除 signature 中的 stream 或按 name 猜 owner。

**Correction check：** 在 reduced V2 engine 上覆盖 HCA/CSA、FULL/PIECEWISE/NONE、short/long contexts、dummy metadata、planner/reuse layers 和 threshold 两侧。每个 replay kernel occurrence 必须匹配实际 owner；重复 signatures 必须按 occurrence/identity 处理或 fail closed。随后才运行同一 call 的 A2/A4，并保留 native TP8 kernel identity comparison。

### F017 — P2 medium — cache zeroing 被错误串行到 DP rendezvous 之后

**Confidence：高。**

**位置：** `D` §3.7 item 3，**135 行**；§4，**143 行**；相关 entries 2/3/20–22，`D:178–179,196–198`。

**违反的 contract：** faithful V2 timeline、A3 once-only accounting、A13(d) event timing。descriptor-dependent input preparation 和 descriptor-independent request/cache updates 必须按 native 的先后与重叠关系建模。

**Native evidence：**

- `N/v1/worker/gpu/model_runner.py:1417–1424` 在 DP dispatch 前执行 real request updates 和 block-table staged writes。
- `update_requests` 在 **1034–1038 行**执行 `kv_block_zeroer.zero_block_ids`。
- `dispatch_cg_and_sync_dp` 在 **1456–1466 行**；input/attention preparation 在 **1473–1546 行**。
- `N/v1/worker/utils.py:211–232` 的 zeroer 使用 async H2D 和 `_zero_kv_blocks_kernel`，这是 pre-DP device work，不只是 host share。
- `N/v1/core/sched/scheduler.py:312,1286,1322–1334` 在需要 zeroing 且分配新 blocks 时将这些 IDs 传给 runner。

此路径对 target V4 可达。P0 native config `T/v4_p0_native_20261021/candidate_h800_p0_profile_04/case/inputs/native_config.json:33` 选择 `fp8`；V4 main attention spec 设置其 quant mode，indexer cache 使用 uint8/default quant mode，见 `N/models/deepseek_v4/attention.py:716–774,856–862`。`N/v1/kv_cache_interface.py:979–1005` 按 `(dtype, kv_quant_mode)` 识别 mixed-precision cache 并启用 zeroing。这里并非仅引用其他模型的 optional Mamba path。

**Reachable trigger：** real PREFILL 或 DECODE lane 本 step 分配新 cache blocks，DP peer 较晚到达。dummy lanes 则跳过 `1417–1424` 的 real update block。

**Expected versus actual design：** expected 是 zeroing 在该 lane 的 DP call 前被 queued，其 device span 可以与 CPU DP wait 重叠；post-DP input/metadata kernels 仍需等待正确的 stream dependencies。`D:135` 只允许 pre-DP host share，`D:143` 则明确将 zeroing 放在 “after the DP rendezvous”，使它与 DP wait 串行。

**Source-backed timing counterexample：** 假设早到 lane 的 zeroing 在 0–30 ms 执行，peer 在 100 ms ready，post-DP preparation 为 0，attention 为 10 ms。native dependency schedule 的 attention completion 是 `max(100,30)+10 = 110 ms`；按当前 §4 串行安排则是 `100+30+10 = 140 ms`。这是说明因果的 CPU arithmetic sentinel，30 ms 不是 native measured magnitude。本轮 AST check 同时确认了 `update_requests → dispatch_cg_and_sync_dp → prepare_inputs` 的 source order。

**Practical impact：** 即使 mode/width 全部正确，stage/EP completion 和 request latency 仍可被高估；zeroing 较短时也会产生系统性的重复等待。原 110 ms sentinel 只覆盖 barrier 后的 attention，无法检测此类 pre-DP overlap。

**Focused correction：** 将 pre-DP state/device work 与 post-DP descriptor-dependent work分开。保留 zeroing 的 lane-local block-count cost 和唯一 owner，记录其 device-ready endpoint；DP rendezvous 使用 host readiness，post-DP device schedule 按 stream dependencies 与 pre-DP endpoint 合并。不要把 zeroing cost 塞进纯 host share，也不要因为同步 width 改变而按 query rows 重算其 block workload。

**Correction check：** 扩展 A13(d)/ledger sentinels：unequal readiness、real+dummy、new-block zeroing 有/无、zeroing 被 DP wait 完全或部分隐藏，覆盖 PREFILL/DECODE 和 FULL/PIECEWISE/NONE。上述示例应为 110 ms；无 peer wait 时则仍须支付 zeroing。检查一次收费、stage components、trace spans 和 completion timestamps 一致。

### F018 — P2 medium — dummy once-only ledger 未排除 native 不执行的 LM head / logits gather

**Confidence：高，针对 ledger 的明确 counts 与 native call chain 冲突。**

**位置：** `D` §3.7 item 4，**136 行**；§4 dummy row，**144 行**；last-layer row，**155 行**；A3，**26 行**。

**违反的 contract：** dummy 只能支付 native dummy 执行的 scopes；A3 必须按实际执行次数计费。physical dummy request slots 不等于真实 sampled requests。

`D:136` 正确说 dummy 按 native executed scopes 定价，但 `D:144` 将其定义为与 real forward “the same phases”，`D:155` 又给 `dsv4_lm_head` 和 `dsv4_logits_gather` 明确的 “1 each”，没有 dummy exception。这使 implementation 和 sentinel oracle 存在直接冲突。

**Native evidence / causal chain：**

1. serving idle lane 的入口是 `N/v1/worker/gpu_worker.py:1210–1212`，只调用 `_dummy_run(..., uniform_decode=True)`。
2. `_dummy_run` 在 `N/v1/worker/gpu/model_runner.py:695–703` 执行 model，然后在 `709–717` 提取并清除 execution state，在 `766–768` 选择 sample hidden states 并返回。它不调用 `compute_logits`、`sample` 或 `sample_tokens`。
3. `_dummy_sampler_run` 在 `770–782` 是另一个入口，不能作为 serving DP dummy 的证据。
4. V4 model forward 包含 final mHC、SP gather、mHC head 和 norm，见 `N/models/deepseek_v4/nvidia/model.py:1189–1214`；LM head/logits 是独立 `compute_logits`，见 `1514–1519`，属于 real sampling chain，见 runner `1744–1751`。
5. 现有 Frontier contract 已区分此事：`frontier/execution_time_predictor/deepseek_v4.py:123–124` 禁止 dummy 选择 LM rows，`214–217` 只在有 selected logits rows 时收费，`519–520` 从真实 `batch.requests` 得到 rows。

**Reachable trigger：** 任一 target serving DP dummy，FULL/PIECEWISE/NONE 都适用，PREFILL 和 DECODE roles 都可达。physical FULL dummy 有 W 个 native slots 也不会因此进入 logits path。

**Expected versus actual design：** expected 是 dummy 的 final mHC/SP/head/norm 仍执行，LM head 和 logits gather count 为 0。若按 §4 的 explicit counts 构造 composition ledger 或 A3 oracle，就会给这些 dummy terminal scopes 各收费一次。若作者本意是让 `D:136` 覆盖表格，那么表格仍须将这个条件写出，否则 A3 没有唯一的 expected result。

**Practical impact：** prefill nodes 有数千次 FULL dummies，错误 terminal cost 会影响其 engine occupancy、后续 rendezvous 和 serving latency；还可能把修正后的 native request count 错用成 selected LM rows。

**Focused correction：** §4 给出 real/dummy 的 conditional counts：dummy LM head/logits gather=0，保留 native model-forward terminal scopes；real selected logits rows 和 physical dummy slots 分开。R1/G2 同时审计 `_dummy_run` 的实际 hidden-state selection work，若 material 则给其真实 owner，不以 LM head 代替。

**Correction check：** A3 和 A13(c) 通过真实 V4 pricing consumer 对 FULL/PIECEWISE/NONE dummy 运行 sentinel。dummy final mHC/head/norm 按 source 执行，LM/logits scopes 缺省或 exact zero；real forward 在实际 selected rows 上收费。保持无 request metrics、无 KV credit，并检查 warm/cold cache predictions 相同。

## 4. Verification

所有 native counter checks 都使用 `ulimit -v 31457280` 和 timeout。仅使用已有 Python/dependencies；测试 cache 和 temporary artifacts 在 `S` 下。

### 实际运行的检查

| Check | Expected | Observed | Status / limits |
| --- | --- | --- | --- |
| scope / revision / Git checks | 指定 branch/HEAD；target 与 committed blob 相同；无 review-induced tracked changes | HEAD、branch 一致；target bytes 相同；tracked staged/unstaged 均空；native worktree clean。另有并发出现的 untracked census 文件，已排除 | **PASS**，不等于整个分支审查 |
| `mode_census.py`，三个 retained runs | corrected mode census 可复现 | exit 0；stdout 与 checked-in `mode_census.txt` byte-identical | **PASS**，first node of each role、rank 0 |
| `review_counter_checks.py`，三个 retained runs | round-2 counter conclusions 可复现 | exit 0；stdout 与 checked-in `.txt` byte-identical | **PASS**，recorded counter evidence |
| `dp_sync_rule_check.py` | recorded desired modes/local sizes 的投影与 final descriptors/widths 一致 | 3,588 rows、245,088 forward observations、0 mismatches；stdout byte-identical | **PASS**，脚本是简化 min-mode/capture-width projection，未独立验证完整 uniform/max-query redispatch 或 Frontier classifier |
| `test_balanced_routing.py` + `test_dp_dummy_forward_pricing.py` | existing CPU contracts pass | **116 passed in 13.98s**，无 skips | **PASS**，未测试 proposed A13(c)/(d) implementation |
| `S/round3_source_counterexamples.py` | exact native helper/predicate/order 支持 source 分析 | exit 0；Breakable compressor stream 改变；short-context predicate 改变；request-update/DP/preparation order 和 dummy call set 符合上述 findings | **PASS**，CPU AST/stubs/arithmetic，不是 GPU reproduction |
| FULL/PIECEWISE reduced-engine route、A2/A4、G1h–G5h | mode-qualified ownership、per-group composition/accuracy | 未执行 | **NOT_RUN**，本轮不提交 GPU jobs |
| proposed A13 timing、actual MetricsStore/KV/NIXL lifecycle、77-case fidelity/dummy smoke matrix | 新时间线正确、其他 paths unchanged | 仅审查设计和 relevant callers，未运行新实现或全回归 | **NOT_RUN** |
| V2 CPU-probe binding / split | 实际 runner instrumentation 和 wait exclusion 正确 | proposal 有 precheck，未提供本轮可验证的 binding/split result | **NOT_RUN** |

重跑 census 的关键 counts：

| Case | Serving decode PIECEWISE | First tails / steady peers | PREFILL FULL dummies |
| --- | ---: | ---: | ---: |
| P2-32 | 357 | 190 / 167 | 9,079 |
| P2-64 | 293 | 79 / 214 | 8,408 |
| P0 | 383 | 204 / 179 | 9,455 |

decode desired-mode disagreement rows / observations 分别为 P2-32 `274 / 688`、P2-64 `803 / 1,224`、P0 `347 / 750`；prefill disagreement rows 分别为 `14 / 53 / 14`。这些 observation counts 会在 DP nodes 上重复，不能当作 unique global forwards。corrected evidence 也不意味着每个 Frontier replay 的 dummy frequency 已与 native 相同。

### Commands

scope commands 在 `W`，native commands 在 native worktree parent：

```bash
git rev-parse HEAD
git branch --show-current
git status --porcelain=v1 --untracked-files=normal
git diff --name-only
git diff --cached --name-only
git diff --numstat 40299c0a 4b4b3ab0 -- task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/design.md
```

另通过 `git show 4b4b3ab0:<design path>` 的 stdout bytes 与 working file 做相等比较。以下变量完整展开本轮 counter commands 的 paths：

```bash
W=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
S=/data/ycfeng/tmp/codex-review/d8d732a2-5682-4e9c-a111-9418dcfebc85
E="$W/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/evidence/routing"
P0=/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04
P232=/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts
P264=/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts

bash -c 'ulimit -v 31457280; exec timeout 120 /usr/bin/python3 -B "$@"' review "$E/mode_census.py" "$P0" "$P232" "$P264"
bash -c 'ulimit -v 31457280; exec timeout 120 /usr/bin/python3 -B "$@"' review "$E/review_counter_checks.py" "$P0" "$P232" "$P264"
bash -c 'ulimit -v 31457280; exec timeout 120 /usr/bin/python3 -B "$@"' review "$E/dp_sync_rule_check.py" "$P0" "$P232" "$P264"
```

这些 checks 的 stdout 保存为 `S/mode_census.round3.txt`、`S/review_counter_checks.round3.txt`、`S/dp_sync_rule_check.round3.txt`，随后通过 Python `read_bytes()` 与对应 checked-in `.txt` 比较，三者均为 True。

实际 pytest command，在 `W` 执行：

```bash
ulimit -v 31457280
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 \
PYTHONPATH=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028:/usr/local/lib/python3.10/dist-packages \
TMPDIR=/data/ycfeng/tmp/codex-review/d8d732a2-5682-4e9c-a111-9418dcfebc85 \
timeout 180 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest \
tests/unit/test_balanced_routing.py tests/unit/test_dp_dummy_forward_pricing.py \
-q -p no:cacheprovider \
--basetemp=/data/ycfeng/tmp/codex-review/d8d732a2-5682-4e9c-a111-9418dcfebc85/pytest_round3
```

source counterexample command：

```bash
ulimit -v 31457280
timeout 30 /usr/bin/python3 -B /data/ycfeng/tmp/codex-review/d8d732a2-5682-4e9c-a111-9418dcfebc85/round3_source_counterexamples.py
```

这些正式执行的 checks 均 exit 0。没有把 inspected tests、原 round-2 sentinels 或 proposed tests 计为本轮已运行。

### 文件与资源检查

`D` §6 的 28 是 work entries，若干行包含多个 files；+3,310 / −330 是 estimate。两个明确超 2,000 行的 touched modules 当前分别为 main predictor 8,666 行、metrics store 5,744 行，与设计一致。entry 12 应按现有 exception record 的模式补充 scoped cleanup inspection、boundary reason 和既有 split sequencing。

`D` §7 的 caps 算术正确：H200 `2 + 7 + 36 = 45` GPU-hours；H800 `2 + 3 + 1.5 + 2 + 0.5 + 12 = 21`；R2h/R2 各增加 optional 32。H200-first、硬件数据互不 qualification、session running+queued ≤64、`step_main` / non-preemptible、≥32 GPU soft gates 均与 governing records 一致。旧 ticket 中的 personal ceiling 不覆盖 item 21 的明确 session exception。

§8 的 H200 critical-path estimate 可由 `2 + 1.5 + max(3,5.5) + 2.5 = 11.5` working days 得到，随后 H800 约 3.5 天，即约 15 天，额外 queue/runtime 风险可解释 15–16.5 的区间。此为规划一致性检查，不是实际工期或 cap feasibility 的验证。F016 的 bootstrap 修正应在 exact grid/effort 更新时反映。

## 5. Open questions 和 next actions

### 尚未确认的问题

| ID | 未验证事项 | 需要的证据 |
| --- | --- | --- |
| O1 | V2 CPU probe 的实际 binding 与 split | 在实际 instrumentation/overlay 上的 CPU binding check 和 fixture logs，证明 pre-/post-DP shares 相加保留 all-reduce own latency，仅排除 rendezvous wait，并继续排除 KV-load/device waits。现有 reducer `frontier/profiling/cpu_overhead/vllm_cpu_probe.py:319–345` 排除的是 all-reduce duration 超过其 own-latency median 的部分；split 不能把全部 all-reduce latency丢掉。设计已要求 precheck，本轮未验证完成。 |
| O2 | 替换 launch alignment 后的 host/stage/KV behavior | `sync_entry.py:123–133` 还包含 `_launched_share` 的逐层 host-launch lower bound，不只是 latest launch-start assignment。需要非零 `forward_launch`、PIECEWISE eager segments、stage-final timing 和 request/dummy lifecycle 的实际 tests。`collective_timing.py:100–105,128–137` 消费 preparation/launch-start 以避免重复 overhead，必须与新 timeline 一致。原 host-launch=0 的 110 ms sentinel 单独不足以证明这些行为。PP1 guard 应保留；KV/NIXL tests 应验证事件的正确推迟和 ownership，而不是要求所有 timestamps 不变。 |
| O3 | native FULL/PIECEWISE gap 的 storage/selection bridge | 现有 `frontier/profiling/kernel_gap/schema.py:7–13` 使用 `eager` / `cuda_graph` 两个 values；loader `kernel_gap.py:41–56` 和 consumer `185–196` 也按它们读取。entry 10 已计划 V4 mode/segment selection，但实施前应明确 FULL/PW scalars 如何被存储和选择，保留旧 rows/消费者。若需要扩展 value vocabulary 或 schema owner，应在 file plan 中明列。现阶段不足以把这个尚未实现的接线单独判为 confirmed defect。 |

### 按依赖顺序的 corrections

1. **先修正 F016 的 stage-1a ownership route。** CPU source analysis明确 reference、capture state、stream roles 和 graph/eager segment occurrence mapping；之后在 approved 1-GPU P1h 内 qualification，再运行 A2/A4 和 grids。
2. **修正 F017 的 timeline 和 F018 的 conditional ledger。** 两者可在 simulator work 中并行推进，但 A13(d) 的 oracle 必须建立在正确的 pre-DP device dependencies 上，A3/A13(c) 必须使用 real/dummy 的真实 terminal counts。
3. **完成 O1/O2/O3 的接线证据。** 尤其检查 host launch 非零时的 per-layer lower bounds、final CPU subtraction、trace cursor/offsets 和现有 dummy/KV lifecycle；避免只用 mocked width getters 或零 host-cost sentinel。
4. **再执行 planned acceptance。** GPU A4/A6/A7/A8、operator-path end-to-end replays、A11 diagnostic census，以及 A12 的 fidelity/smoke/unit regressions。没有这些结果前，保持 typed coverage/unsupported-path failures，whole-step tables继续只作 diagnostic。

本轮没有要求增加大规模 native census job，也没有发现需要撤销 H200-first 或现有 caps 的证据。未确认的 GPU timing、probe binding、implementation regressions 和 resource availability 均保留为上述限制；它们不抵消已经由 pinned source 支持的 F016–F018。

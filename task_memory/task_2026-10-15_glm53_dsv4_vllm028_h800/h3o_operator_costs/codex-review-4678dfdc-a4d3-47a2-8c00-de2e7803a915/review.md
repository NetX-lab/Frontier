# H3o-(a) revisions 2 / 2a / 2b 独立设计审查

## 1. 审查上下文

本次按 caller 指定进行 **design review**，主目标是 revision 2b 的全文设计，核查 revisions 2、2a、2b 是否解决 round-1 findings，以及 operator composition、FULL graph、pilot、host cost、routing identity、file plan 和资源规则是否完整。历史 diff 用于解释修订；本次没有建立整个分支的 base→head 变更审查。

| 项目 | Resolved scope |
| --- | --- |
| Working root | `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` |
| Branch / HEAD | `feat/glm53-dsv4-vllm028` / `40299c0a9062c475bddf8261df2a4744fac04a11`，与 caller 一致 |
| Primary target | [h3o_operator_costs/design.md](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/design.md:1)，revision 2b，第 1–239 行；working-tree 内容与 HEAD 一致 |
| 历史参照 | `0ce31851` revision 1；`886290a1` revision 2；`e936fe70` revision 2a；检查了 `e936fe70..40299c0a` 的设计、routing tests 和支持记录 |
| Staged / unstaged | 无 staged changes。唯一 tracked unstaged change 是 `h3o_operator_costs/review.md` 的 round-2 submission metadata；作为上下文读取，不视为本次设计修订 |
| Untracked exclusions | `.cortexkit/`、`.pi/`、`remaining_measurements/` |
| 其他 exclusions | 整个分支的无关变更、H4b READ/lifecycle 的重新设计、real score/hash routing、PD-AF 新支持、non-V4 DP gap 的修复 |
| Reviewer model / effort | Execution context 指定的 `gpt-6.1-sol-sys-paygo` / `xhigh` |
| Scratch | `/data/ycfeng/tmp/codex-review/4678dfdc-a4d3-47a2-8c00-de2e7803a915` |

下文使用以下缩写，所有位置均可由这些绝对路径定位：

- `T`：`/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800`。
- `D`：`T/h3o_operator_costs/design.md`。
- `N`：`/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`。已确认 native HEAD 为 `2cf0a6915ce544dc493a0990f2ea38d81601128a`。
- 其他 `frontier/`、`tests/`、`docs/` 路径以 working root 为基准。

已读取 `T/requirements.md` items 20–31（245–333 行）、`T/plan.md` H3o row（160 行）和依赖说明（166 行），以及 tickets `05-dsv4-flash-family.md`、`06-communication-measurement.md`、`09-predictor-bundle.md` 的相关任务和 acceptance。已读取 round-1 report 与作者逐项 positions：`T/h3o_operator_costs/codex-review-9c8b2435-c3f3-468e-8c41-6a504ff54fda/review.md`、`T/h3o_operator_costs/review.md`。

适用规则包括 `AGENTS.md` 的 parallel ownership（559–594 行）、typed profiling ownership（596–612 行）、module-size/cleanup/split requirements（783–802 行），以及 `docs/general/module-size-exceptions.md`。未发现其他适用的 ancestor/nested `AGENTS.md` 或 `CLAUDE.md`。ADR `docs/adr/0002-operator-costs-not-measured-step-tables.md:3,7–10` 与 requirements item 26 要求 operator-composed device costs，whole-step table 仅用于 diagnostic cross-check。

目标是 P2-32 / P2-64 的 H200 mainline 和其后的 P0/H800；当前目标 cases 为 PP1。D1 保持 open，以 Option (a) 为 working choice；D2–D5 已按推荐采纳，D6 已决定 H200 first。本次没有扩大 V4 PP 支持。三个 retained counter inventories 都标识实际 runner 为 **`vllm.v1.worker.gpu.model_runner.GPUModelRunner`（V2）**，因此以下顺序判断以该 runner 为准。

本次只写 private scratch scripts。source、Git state、configuration、task records 均未修改；没有安装软件、提交 GPU jobs 或执行外部发布。本报告中的缺陷针对 proposal 的错误假设或遗漏工作；当前代码拒绝未 qualified V4 serving 的 guards 本身是有效保护，不是要求直接删除的 bug。

## 2. Assessment

**Verdict：`changes_required`。确认 P1 × 4、P2 × 3。** revision 2 已修正方法方向并补齐多项重要 contracts，但 revision 2b 尚不足以作为完整实现和测量计划。最 consequential 的风险是：获得准确的 isolated operator predictions 后，Frontier 仍可能选择错误的 native mode/width，拒绝 graph dummy，或把正确的 costs 转换成错误的 event timestamps。

现有 tests 和 counter checks 支持 routing algebra 及有限 recorded-tuples 投影；它们没有证明从 Frontier batch/config 到 local dispatch、DP synchronization、实际 V4 pricing 和 event timing 的完整 identity。

### 原 F001–F008 的 closure

以下是 **设计层面的 closure**，不代表 proposed implementation 已完成或通过 acceptance。

| 原 ID | 状态 | 已解决内容与剩余项 |
| --- | --- | --- |
| F001 | **Partly resolved** | §3.1–3.2 恢复 per-operator Option A，禁止 layer union 作为 cost 或分摊到 child scopes；raw intervals、audit union/span 和 A2/A4 已加入。仍缺 composed duration consumer、条件 stream topology 和 qualified FULL pilot：F010、F014、F015。 |
| F002 | **Partly resolved** | runtime census、segment classification、count naming、V4 EP gap wrapper 和 reduced-engine sampler 已列入。现有 scalar/global-idle gap 仍不能直接支持 critical-path gap；attention consumer 还丢失 gap，pilot route 未闭合：F010、F013、F015。 |
| F003 | **Resolved** | endpoint collective contracts、block zeroing、PDD payload 与 attention admission、pre-forward inventory、shared-expert phase 和 once-only ledger 已补入。新的 V2 dummy workload 问题见 F011。 |
| F004 | **Resolved** | rank0 数据已降为 candidates，增加 zero/light/heavy rank classes、EP16/EP32 producer selection 和 A7。其核心仍是每个 EP rank 的 cost qualification；balanced routing identity 不自动证明 rank timing equivalence。 |
| F005 | **Resolved** | exact ordered unequal-vector binding、combine classification、禁止 scalar fallback、small-payload G5 mandatory 和 A8 已明确。 |
| F006 | **Partly resolved** | D2、R1、conditional R2/R2h、CPU field binding 和 explicit-zero policy 已加入。host/input preparation 的 native 时间边界仍写错：F009。CPU probe 对实际 V2 runner 的绑定尚需验证。 |
| F007 | **Resolved** | workload-owner features、mode/planner key、joint coverage、group-held-out accuracy、collision tests 和 loader validation migration 已补入。 |
| F008 | **Resolved** | all-43-layer balanced routing scope、MoE-owned gating/local L grid、未执行 hash/score selection 不计费已明确。 |

### 对本轮问题 2–6、8–9 的判断

| Caller 问题 | 判断 |
| --- | --- |
| 2. 新 defects / source references | 存在 F009–F015。尤其 D:117 的 input preparation 顺序不符合实际 V2 runner；D:102、110 的 uniform-count/mode census 结论错误；`dp_utils.py:16–75` 引用截断了 84–90 行 final redispatch。 |
| 3. FULL graph composition | **原理 sound，当前说明不 sufficient。** FULL capture 保留 source-defined branches 和 event dependencies；不会使所有 scopes 串行。critical path 必须按该次 capture 的条件 topology 计算，并在选择最大路径前计入该路径的 gaps。完整 canonical operator ledger 与 composed elapsed 还需分别保存和消费。见 F010、F013、F014。 |
| 4. P1h / A4 | qualified 的同一次 FULL replay pilot 可以 falsify composition。median 5% / p90 10% 适合作为 empirical stop gate，但不是结构证明或 formal E2E gate。应按 mode、topology、variant、planner 和 workload group 分组，先用同次调用的 measured operator costs 检查 composition，再检查 predictor error，避免 pooled median 隐藏失败。当前 pilot 尚无可执行的 FULL measurement route：F015。 |
| 5. D2 / ADR | **接受 host-only `forward_launch`。** 它是直接测量的 host interval，device 仍由 operators 和 communication backend 组成，符合 ADR / D-NO-STEP-COST-SHORTCUT。必须排除 DP/KV/device waits，避免与 preparation、drain、gap 重复计费；禁止 whole-step residual。`vllm_cpu_probe.py:23–55,319–345` 提供了这些 interval 的既有定义。实际 V2 binding 和 PIECEWISE segment boundaries 尚未 qualified。 |
| 6. training CSV union/span | **接受不增加这些 training columns。** consumer 不读 union/span 时，保留 raw intervals、audit sum/union/span 和同次 stream-joined elapsed 已满足审计目的。应能追溯到同一 call/capture，不能只保留独立测量的 medians。 |
| 8. 最小完整变化 | 未发现可据证据删除的主要 work unit；仍缺 native-compatible local dispatch、V2 dummy workload、duration consumer integration，以及 stage-1 FULL pilot bootstrap。23 是表格 work units，若干行包含多个实际路径；2,740 lines 是 estimate，不是完整性证明。补齐后应重算 file/line plan，并检查新增 consumer 对 module-size records 的影响。 |
| 9. Order / resources | **H200-first 与资源规则正确。** retained H200 traces 支持在 CPU 做 R1并删除 R1h。§7 的 H200 44 GPU-hours、H800 20 GPU-hours，加各自 optional 32 GPU-hours，算术一致。按 requirements item 21，64-GPU cap 指本 session 的 running + queued，`step_main`、non-preemptible；YC 其他 jobs 不计入且不触碰。≥32 GPU runs 的 gates 为 soft，记录/报告，不终止 training/profiling。当前 queue availability、实际 runtime 和预算可行性未验证。stage order 需修正 F015 的 pilot bootstrap 依赖。 |

### 问题 7：routing identity 的四个方面

| 方面 | 判断 |
| --- | --- |
| §3.7 rule，包括所谓未到达分支 | recorded native desired modes 输入后的简化投影检查通过；完整 source rule 尚未被正确描述。uniform disagreement 分支确实到达，因为 one-token first-tail 的 semantic `has_prefill=True` 会得到 uniform `None`，steady decode 得到 1。native 随后按 synchronized uniform/max-query redispatch。见 F012。 |
| 第一层 deferral / host preparation | first pre_moe room 可以作为 descriptor rendezvous。必须先确定共同 barrier 时间，再执行 post-DP preparation 和第一层 device work。保持 lane-local ownership 是合理的，但把整段 preparation 放在 DP barrier 前不符合 native。当前 reused resolver 与 launch-start alignment 不能完成这个时间转换：F009。 |
| 其他 lane-local workload | real workload 中通过 `compute_rows` 读取 graph width 的部分可继续复用；local mode generation 仍不等价于 native，dummy 仍使用 one-row/no-metadata eager contract。descriptor-dependent input/metadata preparation 也不能被假定完全不受 synchronized width 影响。见 F009、F011、F012。PP、PREFILL 和 stage accounting 需保留明确 regression/guards；当前证据不能宣称 deferral 对它们已经验证。 |
| A13 是否充分 | (a)/(b) 有价值；(a) 验证的是 native-input projection，(b) 验证 routing counts。还需要从实际 Frontier batch/config 开始，经过 local dispatch、DP sync 和真实 V4 cost consumer 的测试，并检查 timestamps、dummy physical workload 和 semantic phase。当前 (c) 的宽度公式及 mock getters 不足以覆盖这些问题。 |

## 3. Findings

新 finding 使用 F009 起编号，与 round-1 IDs 分开。

### F009 — P1 / Confidence：高 — DP barrier 与第一层计时顺序错误，lane-local attention 可以被提前执行

**位置与 contract：** D §3.7:116–117、§6:176–178；`frontier/scheduler/utils/dp_dummy_forward.py:130–134`；`frontier/scheduler/utils/sync_entry.py:112–133`。要求 faithful native timing、A3 once-only ownership 和 A13 real/dummy descriptor identity。

**Native evidence：** `N/vllm/v1/worker/gpu/model_runner.py:1417–1424` 执行 request/state updates；1432–1466 完成 dispatch 和 DP synchronization；**1473–1546 才执行 `prepare_inputs`、`prepare_attn` 和 attention metadata preparation**；model execution 在其后。D:117 的“input preparation runs before the DP all-reduce”不适用于实际 V2 runner。

**触发与 actual/expected：** 两个 DP lanes 的 readiness 不同，早到 lane 有较长 attention。现有 deferral 将其 arrival 写为：

```python
room['arrival_times'][lane_id] = dummy.stage_start_time + attention_delay_seconds(execution_time)
```

之后 `sync_entry` 才统一 `_forward_launch_start_time`，并只对 host launch share 设置下界。它没有把早到 lane 的 device interval 移到共同 barrier 后。

使用实际 resolver 的 CPU sentinel：lane 0 ready=0、attention=10 ms；lane 1 launch-ready=100 ms、attention=1 ms；host launch time=0。Observed arrivals 为 `{0: 0.010, 1: 0.101}`，room completion=101 ms。native barrier-before-model-work 至少要求 `100 + max(10,1) = 110 ms`。这是当前 helper 的机制复现；将其同样扩展到 deferred real lanes 会继承该问题。

**影响与 correction：** descriptor 可以一致而 execution timeline 仍低估，继而影响 EP room、stage completion 和 request metrics。区分 pre-DP request/state work 与 post-DP input/metadata preparation；先计算共同 rendezvous，再更新各 lane 的 device arrival、launch start、stage model components 和 accounting。post-DP preparation 可保持 lane-local cost/ownership，但其时间位置与 descriptor-dependent workload 必须正确。

**验证 correction：** 在 actual V4 path 上覆盖 unequal readiness、preparation、attention durations、real+dummy、FULL/eager、PREFILL/DECODE；上述 sentinel 必须得到至少 110 ms。检查 launch alignment、每 stage 的 components 与 completion 一致；保留 target PP1 和其他 PP/non-V4 regression/guards。

### F010 — P1 / Confidence：高 — composition 没有明确的 duration consumer，现有 V4 getter 串行求和并丢失 attention gap

**位置与 contract：** D §3.1–3.2:57–62、§6:157–166；`frontier/entities/execution_time.py:910–916,1116–1125`；`frontier/execution_time_predictor/deepseek_v4.py:17–67`；`frontier/scheduler/utils/collective_timing.py:29–32`。要求 per-operator ownership 与 source critical-path elapsed 同时成立。

现有 V4 attention getter 的实际返回值是：

```python
return sum(value for _, _, value in timing_operator_times(self, 'attention'))
```

`timing_operator_times` 返回完整 canonical scopes。V4 分支也没有计入 `_kernel_gap_times['attention']`，而其他 attention 分支在 `execution_time.py:1130–1135` 包含它。events 直接消费该 getter。

**触发与 reproduction：** CPU 构造实际 `ExecutionTime`，四个 parallel input projections 分别为 10、7、3、2 ms。source fork 的 busy critical path 为 10 ms；getter 返回 22 ms。再加入 5 ms attention gap，getter 仍返回 22 ms，event delay 为 0.022 s。

**设计遗漏：** §6 只写 structural path“take predictions and composition”，未定义完整 operator timings 与 composed duration 的分离接口或消费者接线。把 composite 塞入任意新 `op_times` 名称会碰到 `time_components.py:111–134` 的 canonical-name contract；删掉非 critical-path scopes 或改写它们的 cost 又会损坏 ledger/attribution。

**影响与 correction：** predictor 的 per-scope accuracy 可以通过，DES 仍使用 serial sum；gap wrapper 接通后也可能不影响 attention duration。定义独立的 composed elapsed/phase duration，或让现有 V4 duration getter 明确委托 composition helper；完整 canonical costs 保留用于 ledger/trace。补齐实际 consumer/helper、stage aggregation 和 copy/cache integration 的 file plan。不得把 composite total 分摊成 child costs。

**验证 correction：** fork/join、serial、gap sentinel、first/last layer 和 single-layer copies 的测试须同时验证全部 operator costs 保留、event delay 使用 composed duration、gap 一次计入、stage/model totals 一致。

### F011 — P1 / Confidence：高 — 同步后的 V4 dummy 仍沿用 eager/no-metadata workload，不能代表实际 V2 dummy forward

**位置与 contract：** D:16、A13:34、§3.7:117–118、§6:159–160,176–179；`frontier/attention/deepseek_v4.py:268–337`；`frontier/execution_time_predictor/deepseek_v4.py:98–101,511,540–546`；`sklearn_execution_time_predictor.py:7583–7621`。要求 every real/dummy lane 使用 native descriptor 对应的 operator workload，A1 不留 dummy cost。

**触发与 evidence：** idle DP lane 与 graph real lane 同步后，dummy 得到 FULL metadata。实际 `compute_rows` 可返回 padded width=4，但 `validate_deepseek_v4_dummy_forward` 随即抛出：`NotImplementedError: V4 idle forward requires ordinary eager co-location`。

即使只放开该 guard，dummy builder 仍要求 absent attention metadata，使用 `tokens = batch.total_num_tokens`（一行），并构造 no-metadata projection/output-zero workload；这不能通过改 metadata 自动变成 graph dummy workload。

**Native expected：** `N/vllm/v1/worker/gpu_worker.py:1210–1212` 调用 `_dummy_run(..., uniform_decode=True)`。V2 `_dummy_run` 在 `model_runner.py:647–650` 将 skip-attention 限于 initial memory profiling，695–701 进入实际 `execute_model`。1500–1546 按 synchronized descriptor 创建 dummy InputBatch、准备 dummy attention metadata；FULL 还 re-stage capture metadata。`input_batch.py:116–206` 定义 dummy slots、token distribution 与 padding；`model_states/default.py:135–184` 定义 FULL padded metadata。不能套用 legacy runner 的 eager/no-metadata 假设。

**影响与 correction：** graph cohorts 会 fail fast；仅删除 guards 则会漏价 native kernels、错误 physical rows，甚至改变 dummy/request accounting。显式表示 V2 的 mode/descriptor-sized dummy workload，区分 scheduler 的 request-free identity 与 native physical dummy slots、invalid padding、attention metadata。按 native 分支 qualified operator scopes；不能给 dummy 真实 request metrics 或 KV credits。缺少 dummy coverage 时 typed failure。

**验证 correction：** A13(c) 必须运行 actual V4 workload builder 和 pricing consumer，覆盖 FULL、PIECEWISE、eager、不同 synchronized widths 与 PREFILL/DECODE；确认 native kernel/workload ownership、正确 rows、无真实 request credit，不能只断言 mocked width getter。

### F012 — P1 / Confidence：高 — A13 使用 native desired modes，未验证 Frontier local dispatch；first-tail 会同步到错误 mode/width

**位置与 contract：** D §3.7:102,110,112,116；A13:34；§6:176–179。相关 sources：`frontier/scheduler/replica_scheduler/vllm_v1_iteration_policy.py:44–123`，`N/vllm/v1/worker/gpu/dp_utils.py:52–95`。要求 D-V4-ROUTING-IDENTITY 对实际 Frontier forward 成立。

**确认的错误 claim：** one-token first-tail 仍是 semantic prefill。native `model_runner.py:1049–1099` 将 `has_prefill` 传入 classifier；`N/vllm/v1/worker/utils.py:601–607` 对 first-tail 返回 uniform `None`，对 steady decode 返回 1。因此 D:102 的“uniform disagreement branch not reached”错误。native graph branch 还在 `dp_utils.py:84–90` 按 synchronized uniform count / max query length final redispatch；引用到 75 行不足以描述完整规则。

**实际 counter：** H200 P2-32 artifact：

`/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/nodes/gpu-h200-0577.lgcm.sh.istep.fun/serving/rank0.json:1`

JSON path `counters.rows[428].dispatch_evidence` 为：

```text
dp_post_all_reduce = [[12,8], [1,2], [0,1], [1,1]]
classifier.has_prefill = true
native configured/dispatcher mode = FULL_AND_PIECEWISE
native final mode = PIECEWISE
native widths = [16,16]
```

四行分别是 local tokens、desired modes、uniform counts、max query lengths。全 decode-node census 还得到以下到达 graph uniform-disagreement 分支的记录；它们同时存在 lane desired-mode differences：

| Case | Counter rows | Forward observations |
| --- | ---: | ---: |
| P0 | 347 | 750 |
| P2-32 | 274 | 688 |
| P2-64 | 803 | 1,224 |

这些 observation 会在不同 DP-node recordings 中重复，不能当作 unique global forwards。D:110 的“no forward with a mode difference was recorded”也不成立。原 `dp_sync_census.py:32–35` 只在 routed M 不同时分类，并把 graph 情况归为 width difference，不能支持该 mode claim。

**Frontier trigger / expected versus actual：** 当前 `full_decode_only` policy 在 `vllm_v1_iteration_policy.py:98–107` 给 first-tail `NONE`、steady decode `FULL`；CPU microcheck 已复现。将该输入喂给 proposed min rule 会得到 NONE。上述 native tuple 的 first-tail+decodes lane 为 12 rows，peer 为 8；按当前 Frontier local policy 同步后 widths 为 `[12,8]`，routed M 为 24，native 是 32。这是 source/counter 支持的推导，尚非完整 simulator replay 的 error estimate。切换全局 `piecewise` 又会让 steady-only steps 失去 native FULL。

**检查的正确边界：** existing rule check 确实在输入 **native desired modes** 时复现全部 3,588 rows / 245,088 forward observations，0 mismatches。本 finding 不否定该结果；它指出缺失了产生这些 desired modes 的 Frontier 前半段。

**Correction / verification：** 将 native-compatible V4 local semantic classifier、capture/descriptor selection 纳入 scope 和 file plan；保留 synchronized uniform/max-query/capture identity，再按 source redispatch。修正 census claims。A13 从实际 Frontier batches/config 出发，覆盖 first-tail alone、first-tail+steady、steady-only、dummy 和 PREFILL，经 local dispatch→DP sync→actual pricing 与 native 比较。685 recorded-tuples check 保留为独立 projection check。上述 tuple 必须最终得到 PIECEWISE、`[16,16]`、M32，steady-only 仍为 FULL；non-V4 维持 A12。

### F013 — P2 / Confidence：高 — global idle-gap statistic 不能直接作为 critical-path per-kernel gap

**位置与 contract：** D §3.3:71、§6:166,173；`frontier/profiling/kernel_gap/main.py:90–115`；`frontier/execution_time_predictor/kernel_gap.py:41–55,185–196`。要求 graph-internal gaps 与 source critical path 相符。

现有 reducer 将同一 graph launch 的所有 streams 按开始时间合并，并以全局 `covered_end` 求 idle。consumer 每 mode/TP 读取一个 scalar，再乘 kernel count。它测量的是 global-union idle，不能识别被其他 branch 覆盖的局部 gap。

**Exact reducer counterexample：** 合法 fork 中，stream A kernel 为 0–100 μs；stream B 的 B1 为 0–20 μs、B2 为 50–110 μs。实际 `chain_gaps_us` 返回 `[0,0]`。busy-only critical path 为 `max(100,20+60)=100 μs`；joined elapsed 为 110 μs，B branch 有 30 μs gap 被 A 隐藏。先选 busy critical path再乘 global scalar 无法得到正确 completion。

这是机制 counterexample，不是 measured native error estimate。

**影响与 correction：** A4 在一个 topology 上通过不证明 global scalar 对其他 workload 正确。定义 stream/dependency-edge gap ownership，并在加入 costs/gaps 后求最大路径；或明确把 scalar 定位为 approximation，按相关 topology/workload 分组 qualified，失败即停止。不能把 dependency wait 再作为 gap 重复计入。raw scope kernel counts 也不足以识别 critical-path kernel count。

**验证 correction：** 加入上述 hidden-gap fork、critical branch 切换、serial chain 和 explicit dependency wait tests；pilot 用同次 intervals 检查 gap ownership 与 joined completion，按组执行 A4。

### F014 — P2 / Confidence：高 — 每 variant 固定一张 stream graph，遗漏 source 中可达的 topology 条件

**位置与 contract：** D §3.2:62、stage 0:145、file 2:158。要求 graph 来自选定 source 的真实 stream switches/event waits。

`N/vllm/models/deepseek_v4/attention.py:515–573` 的 outer input projections 仅在 `hidden_states.shape[0] <= VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` 时并行；`N/vllm/envs.py:2016–2017` 默认 threshold=1024。`multi_stream_utils.py:105–108` 在禁用时按序串行。设计 grid 同时包含小 decode 与 6144 prefill，因此同一 CSA/HCA variant 存在不同 topology。

此外，`multi_stream_utils.py:45–61` 在 active Breakable capture 下串行 nested branch；`attention.py:906–930,952–958` 的 short-context/capture 状态也改变 indexer 分支。§3.6 为 cost model 加 mode/planner features，并不自动解决 §3.2 固定 topology 的问题。

**触发与 impact：** 在 threshold 两侧或不同 capture/short-context 状态使用同一 fork graph，会对串行 native work 求 max，或对并行 work 求 sum。scope costs 即使准确仍会产生错误 elapsed。

**Correction / verification：** 定义 source-backed conditional graph，由实际 rows、mode/capture、stream configuration、short-context/planner state 选择。记录相关 env/runtime identity。CPU sentinel 与 pilot 覆盖 threshold 两侧和可达 capture branches，检查正确 topology、完整 ownership 及 joined elapsed；不需要改变 operator cost 的单一 ownership。

### F015 — P2 / Confidence：高 — P1h/A4 缺少在 grids 前可用的 qualified FULL replay 测量路径

**位置与 contract：** D A4:25、stage 1/2:146–147、files 13/17:169,173、P1h:192。要求先用最小 pilot falsify composition，再投入 grids。

**现有 producer evidence：** `frontier/profiling/attention/dsa_vllm_runner.py:597–604` 的 timing engine policy 强制 eager；`measure_engine:645` 调用 `require_scope_timing`。`dsa_plan.py:787–794` 明确拒绝未 qualified 的 native compiled/graph per-scope instrumentation。`dsv4_attention_runner.py:24–27` 在 engine startup 后安装 hooks；FULL replay 不重新执行 `dsa_vllm_runner.py:224–247` 的 Python per-scope callbacks。

**遗漏与 reachable trigger：** 若按 §5 顺序启动 P1h，现有 runner只能测 eager，或拒绝 graph scope timing。file 13 只列 interval retention/count naming；V4 graph-replay sampler file 17 在 stage 2，位于 pilot之后。retained native traces 可用于 mode/kernel census，但不能代替 reduced-engine 同一次 FULL call 的 joined boundary 与 scope ownership。

**影响与 correction：** eager pilot 的通过不能回答 FULL composition 是否正确，或 pilot 会在采集前停止。明确并提前实现最小 FULL replay pilot route：capture-time kernel ownership 标识、replay interval extraction、同一次 layer call 的 joined completion boundary，以及 instrumentation qualification。将必要 producer bootstrap 移到 stage 1 前，完整 grids 仍留在之后。按 R1 选择的 PIECEWISE/first-tail branches 也需相应 pilot。无需推翻已批准的 reduced TP1 D5。

**验证 correction：** 在 stage-1 H200 pilot 中确认实际 graph replay、scope ownership 完整、joined boundary 有效，随后执行分组 A4。不能用 eager-only hooks 的结果替代 FULL gate。

## 4. Verification

环境使用已存在的 `/data/ycfeng/envs/frontier-py310/bin/python` 和 `/usr/bin/python3`；没有安装/升级 dependencies。Python bytecode、pytest cache 已关闭；pytest temporary files 位于 scratch。所有 native trace/counter scripts 都使用 `ulimit -v 31457280`（30 GiB virtual-memory cap）和 timeout。

| Check | Expected | Observed / status |
| --- | --- | --- |
| `git rev-parse HEAD`、`git status --short`、`git diff --name-only`、`git diff --cached --name-only` | 确认 caller scope 和 local changes | HEAD 一致；无 staged；仅上述 review metadata unstaged 和三个 untracked directories。审查未改变状态。**PASS** |
| 相关 sources/records 的 `rg`、`nl -ba` 和历史 `git show` / `git diff` inspection | 核对设计 claim 与实际 contracts | 关键位置见 findings。属于 source inspection，不是 test execution |
| 两个相关 pytest modules | 现有 balanced-routing/native-source 和 dummy-pricing contracts 通过 | **116 passed in 17.03s**，无 skip；其中 balanced-routing 为 102 cases。**PASS** |
| existing native rule check，三个 retained inventories | recorded local sizes + native desired modes 投影等于 recorded descriptor | **3,588 rows / 245,088 forward observations，0 mismatches。PASS** |
| scratch semantic counter census | 检查实际 runner、uniform disagreement 与 desired-mode differences | 全部标识 V2 runner；复现 F012 的 counter evidence。**PASS** |
| actual ExecutionTime、exact reducer AST、timing algebra | 复现现有 duration/gap/timing 边界 | 22 ms serial sum、attention gap 未改变 duration；hidden-gap reducer 为 `[0,0]`；101 ms versus 110 ms。**PASS（缺陷复现成功）** |
| actual dummy resolver/guard、native uniform helper、Frontier policy | 检查 descriptor 后的真实 workload admission 和 local dispatch | FULL dummy guard 拒绝；first-tail uniform=None；Frontier `full_decode_only` modes 为 NONE/FULL。**PASS（缺陷复现成功）** |
| H200 P2-32 selected native trace census | 确认 retained decode FULL 与 eager pre-forward structure | 249 `cudaGraphLaunch`、9,937 stream-23 eager kernels、FULL annotations。**PASS** |
| H200 row 428 whitelist read | 定位具体 counter tuple 和 runtime identity | V2 / FULL_AND_PIECEWISE；PIECEWISE `[16,16]`；uniform `[0,1]`。**PASS** |

以下为主要执行命令。pytest 的先前同范围重复运行不累计为额外 cases；round-1 的 81-test report 本轮只作为历史证据读取。

```bash
ulimit -v 31457280
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 \
PYTHONPATH=$PWD:/usr/local/lib/python3.10/dist-packages \
TMPDIR=/data/ycfeng/tmp/codex-review/4678dfdc-a4d3-47a2-8c00-de2e7803a915 \
timeout 180 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest \
tests/unit/test_balanced_routing.py tests/unit/test_dp_dummy_forward_pricing.py \
-q -p no:cacheprovider \
--basetemp=/data/ycfeng/tmp/codex-review/4678dfdc-a4d3-47a2-8c00-de2e7803a915/pytest_dp_final
```

```bash
ulimit -v 31457280
PYTHONDONTWRITEBYTECODE=1 timeout 120 /usr/bin/python3 -B \
task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/evidence/routing/dp_sync_rule_check.py \
/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04 \
/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts \
/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts
```

```bash
ulimit -v 31457280
PYTHONDONTWRITEBYTECODE=1 timeout 120 /usr/bin/python3 -B \
/data/ycfeng/tmp/codex-review/4678dfdc-a4d3-47a2-8c00-de2e7803a915/review_counter_checks.py \
/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04 \
/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts \
/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts
```

```bash
ulimit -v 31457280
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:/usr/local/lib/python3.10/dist-packages \
timeout 120 /data/ycfeng/envs/frontier-py310/bin/python -B \
/data/ycfeng/tmp/codex-review/4678dfdc-a4d3-47a2-8c00-de2e7803a915/review_composition_checks.py
```

```bash
ulimit -v 31457280
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:/usr/local/lib/python3.10/dist-packages \
timeout 120 /data/ycfeng/envs/frontier-py310/bin/python -B \
/data/ycfeng/tmp/codex-review/4678dfdc-a4d3-47a2-8c00-de2e7803a915/review_dp_checks.py
```

```bash
ulimit -v 31457280
PYTHONDONTWRITEBYTECODE=1 timeout 120 /usr/bin/python3 -B \
task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/evidence/census_peek2.py \
/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792942701901162313.pt.trace.json.gz
```

具体 row 的补充读取命令如下；只输出选定 inventory/dispatch fields：

```bash
ulimit -v 31457280
PYTHONDONTWRITEBYTECODE=1 timeout 60 /usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path
p = Path('/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/nodes/gpu-h200-0577.lgcm.sh.istep.fun/serving/rank0.json')
d = json.loads(p.read_text())
print('file', p)
print('inventory.model_runner', d['inventory'].get('model_runner'))
print('inventory.effective_graph', d['inventory']['effective_graph'])
r = d['counters']['rows'][428]
print('row428 kind/count', r.get('kind'), r.get('successful_forwards'))
print('row428 dispatch_evidence', json.dumps(r['dispatch_evidence'], sort_keys=True))
PY
```

一次 scratch AST harness 初次缺少 `lineno`，经 `ast.fix_missing_locations` 修正后通过；这不是 repository test failure。一次只读 source command 误在 native cwd 执行，产生 `No such file` / unknown Frontier revision，已在 working root 重试成功。

**NOT_RUN / coverage limits：** proposed predictor、composition、A13(c) implementation 尚未完成，因此未执行 GPU P1h/P1、G1–G5、训练、A1–A13 implementation acceptance、operator-path P0/P2 E2E、77-case fidelity matrix 或完整 smoke/unit suite。未重读全部 32/64-rank trace files；H200 P2-64 的 301 launches / 10,853 eager kernels 本轮作为作者 retained evidence 读取，未独立重新运行其 trace census。counter census 覆盖了三个 inventories，但 observation counts 不是 deduplicated global forwards。AST/sentinel checks 验证机制，不量化 native E2E error。

## 5. Open questions 与 next actions

### 尚未确认的问题

| 问题 | 所需证据 |
| --- | --- |
| eager host time 是否 material，CPU probe 是否绑定实际 V2 runner？ | unprofiled R2h/R2 的实际 instrumentation binding，以及 pre/post-DP、KV wait、PIECEWISE seam、launch/drain interval 对齐。profiler host magnitudes 仅为 upper bounds，不能直接用于 host cost。 |
| reduced TP1 对 native TP8 的适用边界有多大？ | P1h/P1 的 runtime/capture identity、kernel names/grids/layout、同次 composition 与 joined boundary。kernel identity 是必要证据，不能单独证明 contention/timing 等价。 |
| 单 layer costs 在 43 layers 中的 cache/stream effects 是否 material？ | native diagnostic phase/step cross-check 与最终 operator-path E2E。D §10 已列为 limitation；本轮没有把它认定为确认 defect 或导入校正 factor。 |
| caps 与 elapsed days 是否可执行？ | C1 的 exact coordinates、pilot 实际 startup/measurement runtime、queue availability 和 bounded retries。当前 figures 可作为 planning caps，尚非执行可行性验证。 |
| D1 最终选择 | 仍为 YC open decision。本次按 Option (a) 审查；不能自动将 pilot failure 转为 layer-composite cost 或 step residual。 |

### 修正依赖顺序

1. **先完整化 native workload identity：F012、F011。** 将 actual V4 local classifier/descriptor selection 和 V2 dummy workload 纳入 file plan；修正 uniform/mode census；让 A13 从 Frontier batch/config 到实际 pricing 验证。
2. **修正 barrier 与时间戳：F009。** 明确 pre/post-DP preparation，先共同 rendezvous，再推进第一层 device work，更新 stage components 和 launch accounting。
3. **固定 composition contract：F014、F013。** 定义条件 stream graph、gap ownership 和 qualified approximation 的范围；costs/gaps 进入图后再求 critical path。
4. **接通 duration consumer：F010。** 保留完整 canonical operator ledger，同时让 DES 和 stage totals 使用 composed elapsed，验证 copies、gap 和 once-only accounting。
5. **提前 minimal FULL pilot bootstrap：F015。** 先使 P1h 能测 actual FULL replay 的同次 scopes/intervals/joined boundary，再投入 complete grids。A4 按 workload groups 执行，失败停止路线并报告。
6. 随后完成 C1 exact grids、rank/collective qualification、group-held-out training、A1–A13 和 operator-path replays；D2 host-only rows、audit-only union/span、H200-first 与既定 GPU rules 可以保留。

本次没有发现重新引入 whole-step cost shortcut 的提议。剩余 actionable defects 集中于 native descriptor、workload/timing identity 和可执行的 composition qualification；现有 PASS 结果不能替代这些修正后的 acceptance。

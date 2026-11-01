# Forward-attention continuation checkpoint

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Checked the approved A6 continuation after supplement/table and forward-code verification; original qualification/replay runs remain active. |

## 1. 规划完成情况

范围：`requirements.md` 的 D-A6-FORWARD-ATTENTION / D-A6-PASS-REPLAYS 和 `plan.md` 的 Approved A6 continuation。不是 tickets 05/09/11/19 的整体完成检查。

| 规划子任务 | 状态 | 证据 |
| --- | --- | --- |
| 432 个补测 workload 再测两轮并完成取数/业务核对 | PASS | `a6_three_run_20261101/{platform_final.json,download.json,repeatability_result.json,business_verification.json}`；864 步身份/形状/计数一致，容器耗时 0.9302778 GPUh。 |
| Forward 级判定实现及回归验证 | PASS | 原生 layer window reference、配置层数权重、固定 folds/门槛；主代理 31 tests PASS，B-FORWARD-REVIEW 无 actionable finding。 |
| 当前完整输入的 forward 判定和全 scope diagnostics | PASS | `a6_forward_attention_20261101/current_phase_heldout.json`，exit 0；1,373 complete / 48 omitted，三个完整类别 PASS，整体仍 NOT_EVALUABLE。 |
| 完整三轮表、两张独立轮次表和四个输入绑定 | PASS | `a6_pass_replays_20261101/{rows_manifest.json,tables_manifest.json,preflight.json}`；7,105 行均有三轮结果；kernel/gap/window 同轮替换，非 attention 输入不变。 |
| Median3 表的固定-fold A6/forward 复核 | TODO | 原 supervisor 236919 / timeout 237030 已启动，`median3_a6_command.json`；等待最终结果，不重复启动。 |
| 四次完整 P2 replay、逐 repeat 四项指标和跨表 spread | TODO | 原 supervisor 223761、四个 timeout PID 已记录；各 run 后只执行一次现有 gate，终态/比较尚未完成。 |

汇总：4 PASS / 0 BLOCKED / 0 FAILED / 0 SKIPPED / 2 TODO。历史失败的辅助检查均已查明并恢复，不计为未完成 workload。整体验收依然 OPEN。

## 2. 偏离检测

| 偏离模式 | 判定 | 证据与例外 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 864 步业务、完整表和 current forward output 已交付，CPU 比较已启动；所需最小 preflight 已结束，不继续扩建 validator。 |
| D2 细节反复重试 | CLEAR | running 读数、缺失 prerequisite、Boolean key 和 config-only 目录都有具体原因及最小修正；无 GPU 重跑或反复变参。 |
| D3 支线问题修补 | CLEAR | 修改限于 approved forward 判定、独立表和实际执行 prerequisite；未修补无关 source/GLM/prefill 问题。 |
| D4 可并行任务停滞 | CLEAR | 代码、A 取数/表、输入探索和 review 并行推进并完成；原 median3 与四次 replay 也并行运行。 |
| D5 超范围镀金 | CLEAR | 四个代码文件均由 approved code/table 接受条件覆盖；没有新 simulator feature、GPU campaign、prefill 数据或 formal table。 |
| D6 分析空转 | CLEAR | current main exit 0、完整表绑定通过、原四个 replay 已启动；调查产物已被实现/执行消费，没有重开原因分析。 |
| D7 路线违背 | CLEAR | requirements 两项批准、plan 真实 prerequisite 均执行；scope/phase 只诊断，所有 clean repeats 和门槛保持，未先运行未获准内容。 |

**文档检查：** S5 stale doc 曾存在于旧 current-route H3o 行和 Next actions：仍要求提交 A、等待 B 审批，与成功作业及批准决定矛盾。已经修正，重启 snapshot 标为 historical；未发现 S1/S2/S3/S4/S6 的新缺陷。完整 workload 限制与 omitted coverage 保留。

**归因：** 无当前 HIT。执行此前已遵循新决定；S5 是未来执行容易读错的旧文档状态，不是重复提交或绕过接受条件的实际行为。

## 3. 纠正建议

**文档更新（Branch B，均已落实，无需新增用户确认）：**

- `plan.md` 的 Current execution route / Approved A6 continuation：改为成功补测、已验代码/表和原 CPU 流程运行中；替换已被批准 supersede 的 scope-level verdict 与旧提交指令。
- `progress.md` 的 current evidence / Next actions：记录原 PID/command receipts，删除再次提交 A 和再次请求 B 选择的旧下一步，将重启 snapshot 标为 historical。
- `summary.md` 与 tickets 05/09 的最新 continuation：同步 observed PASS/NOT_EVALUABLE 和 pending CPU outcomes，保留整体验收 OPEN。没有改变已商定 scope/数字门槛/决定。

**需停止的偏离行为：无。**

**下一路线：**

1. 并行完成原 median3 和四个 P2 流程；重启后先读 command receipts 和原 PID。完成条件：两份 same-fold qualification 输出和四份 terminal run/gate evidence；通知丢失不构成重跑理由。
2. 汇总所有 retained readings 的 B29/pass1/pass2 TTFT/TPOT/E2E/THPT 与三次 clean reference；完成条件：每项 predicted/actual、绝对/相对误差及跨表 spread，无挑选最好一轮或 pooled PASS 覆盖 repeat FAIL。
3. 保留 median3 方法限制（仅 kernel-time median targets 聚合，gap/window reference 仍为原 generation），记录接受范围。新 formal table promotion 仍待进一步决定，不属于此次文档修正的权限。

## 4. 最值得关注的 3 点

1. 完整 forward 类别 p90 为 1.667818% / 1.646428% / 1.886652%，但 48 个遗漏 workload 不合格，整体不是全面 PASS。
2. B29 P2-64 repeat01 TTFT 已 FAIL 12.254940665%；最终结果必须区分测量噪声传到用户指标的 spread 和既有系统偏差。
3. 当前只推进已批准的 A6/CPU sensitivity；NIXL、memory、formal P2 和 H800 验收仍未完成，不能扩大本次任务或提前关闭 tickets。

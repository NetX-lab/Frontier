# Forward-attention continuation checkpoint

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Continued original CPU runs; prepared the final retained-reading collector with focused/schema checks. Still 5 PASS / 1 TODO; no new workload or scope. |

## 1. 规划完成情况

范围：`requirements.md` 的 D-A6-FORWARD-ATTENTION / D-A6-PASS-REPLAYS 和 `plan.md` 的 Approved A6 continuation。不是 tickets 05/09/11/19 的整体完成检查。

| 规划子任务 | 状态 | 证据 |
| --- | --- | --- |
| 432 个补测 workload 再测两轮并完成取数/业务核对 | PASS | `a6_three_run_20261101/{platform_final.json,download.json,repeatability_result.json,business_verification.json}`；864 步身份/形状/计数一致，容器耗时 0.9302778 GPUh。 |
| Forward 级判定实现及回归验证 | PASS | 原生 layer window reference、配置层数权重、固定 folds/门槛；主代理 31 tests PASS，B-FORWARD-REVIEW 无 actionable finding。 |
| 当前完整输入的 forward 判定和全 scope diagnostics | PASS | `a6_forward_attention_20261101/current_phase_heldout.json`，exit 0；1,373 complete / 48 omitted，三个完整类别 PASS，整体仍 NOT_EVALUABLE。 |
| 完整三轮表、两张独立轮次表和四个输入绑定 | PASS | `a6_pass_replays_20261101/{rows_manifest.json,tables_manifest.json,preflight.json}`；7,105 行均有三轮结果；kernel/gap/window 同轮替换，非 attention 输入不变。 |
| Median3 表的固定-fold A6/forward 复核 | PASS | 两阶段 exit 0，终态 08:37:48 UTC；`median3_verification.json` 验证所有 folds/reference/weights/omissions 不变，三个完整类别 p90 1.628628%/1.932193%/2.215912% PASS。不能算整体 PASS，不重复启动。 |
| 四次完整 P2 replay、逐 repeat 四项指标和跨表 spread | TODO | 原 supervisor 223761 与四个 driver 在 09:21 UTC 正常运行；collector 有 17 checks、四个实际输入绑定、nested-gate schema 和 4 个终态绑定检查 PASS，review 唯一发现已修复；终态/96-row 比较仍未完成。 |

汇总：5 PASS / 0 BLOCKED / 0 FAILED / 0 SKIPPED / 1 TODO。剩余一项包括四个原 replay 的完整终态、gate 和跨表比较。历史失败的辅助检查均已查明并恢复，不计为未完成 workload。整体验收依然 OPEN。

## 2. 偏离检测

**结论：** 未偏移，正常推进。**检测窗口：** median3 验证完成至当前 collector preparation；**当前前沿：** 四个原 P2 replay 的终态与比较。

| 偏离模式 | 判定 | 证据与例外 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 原 F 比较脚本限定 F variants，不能读取此次 attention 变化；新增只读 collector 仅补齐剩余 96-row 比较，复用原指标/哈希机制；到实际输出可核对为止，不扩建。 |
| D2 细节反复重试 | CLEAR | running 读数、缺失 prerequisite、Boolean key、config-only 目录及内联 exec 的 receipt 变量冲突均有具体原因及修正；命令记录已恢复，数值 SHA 不变，无 GPU/资格/replay 重跑或反复变参。 |
| D3 支线问题修补 | CLEAR | 修改限于 approved forward 判定、独立表和实际执行 prerequisite；未修补无关 source/GLM/prefill 问题。 |
| D4 可并行任务停滞 | CLEAR | 代码、A 取数/表、输入探索和 review 并行完成；median3 在原四次 replay 运行期间完成，没有等待它们才复核。 |
| D5 超范围镀金 | CLEAR | 四个代码文件均由 approved code/table 接受条件覆盖；没有新 simulator feature、GPU campaign、prefill 数据或 formal table。 |
| D6 分析空转 | CLEAR | current main exit 0、完整表绑定通过、原四个 replay 已启动；调查产物已被实现/执行消费，没有重开原因分析。 |
| D7 路线违背 | CLEAR | requirements 两项批准、plan 真实 prerequisite 均执行；scope/phase 只诊断，所有 clean repeats 和门槛保持，未先运行未获准内容。 |

**文档检查：** S1–S6 均未发现新的明显缺陷。上轮 S5 已修正；requirements、plan、tickets 05/09 的最新 continuation 与原进程和已验产物一致，历史记录已标明时间或 superseded。

**归因：** 无当前 HIT。执行此前已遵循新决定；S5 是未来执行容易读错的旧文档状态，不是重复提交或绕过接受条件的实际行为。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：无。**

**下一路线：**

1. 完成原四个 P2 流程；已有阻塞式退出通知，只等待事件，不轮询、不重启。完成条件是四份 terminal run 和 nested gate evidence；median3 已完成。
2. 用已准备的 collector 汇总 B29/pass1/pass2 的 TTFT/TPOT/E2E/THPT 与三次 clean reference；完成条件：96 行 predicted/actual、绝对/相对误差，8 组跨表 spread 和 8 组三轮 native 变异描述，无挑选最好一轮或 pooled PASS 覆盖 repeat FAIL。
3. 保留 median3 方法限制（仅 kernel-time median targets 聚合，gap/window reference 仍为原 generation），记录接受范围。新 formal table promotion 仍待进一步决定，不属于此次文档修正的权限。

## 4. 最值得关注的 3 点

1. Median3 完整 forward 类别 p90 为 1.628628% / 1.932193% / 2.215912%，两个 PIECEWISE 类别比原输入略升，不能宣称补测普遍提高准确度；48 个遗漏仍不合格，整体不是全面 PASS。
2. B29 P2-64 repeat01 TTFT 已 FAIL 12.254940665%；最终结果必须区分测量噪声传到用户指标的 spread 和既有系统偏差。
3. 当前只推进已批准的 A6/CPU sensitivity；NIXL、memory、formal P2 和 H800 验收仍未完成，不能扩大本次任务或提前关闭 tickets。

# Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Parent accepted CPU implementation at e2be122 after183/148 independent passes and actual shipped-matrix rejection; reviewP3 removed at15a3958. |
| 2026-10-21 | Start scoped executable implementation. |

- completed: inspected primary requirements/spec/plan, D19 command/input inventory, shared five-file merge delta and native Step canonical replay.
- completed: shared-code reuse and executable Step source adapter; case/source/image declaration and actual artifact pins are checked through the existing admission owner.
- completed: 106 focused/native-consumer passes (Python 3.12), 148 internal/legacy passes (Python 3.10), 77 native P/D harness passes (Python 3.12); these sets overlap in the legacy gate tests.
- completed: manually invoked both existing normalize+gate CLIs on Step and GLM raw-schema fixtures, plus the shipped Step matrix with missing evidence. Synthetic oracle values match; shipped matrix returns INSUFFICIENT_EVIDENCE.
- completed: source commits `ac92396` (shared reuse) and `4d62590` (Step executable adapter); task evidence archived with selected CLI JSON readings.
- pending outside this CPU task: actual clean native repeats, actual runtime/source/image/qualification receipts, measured Frontier run and routing/coverage admission.

The shared merge has two parents. Its first-parent delta contains exactly five owned files; the existing three shared file preimages match this branch. `canonical_workload` is the only missing dependency. Step's consumer currently duplicates that reader while additionally enforcing D18's immutable pins.

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 需求与格式检查 | PASS | 已核对 primary 文档和实际 native writer。 |
| 共享实现复用 | PASS | `ac92396`，62 PASS、1 SKIP。首次缺少 csv import 已修正。 |
| Step 格式与准入 | PASS | source-format registry、原生字段映射和真实证据准入已实现。 |
| 验证与提交 | PASS | 三组验证及六次手动 CLI 均完成，代码已提交为 4d62590。 |
| 完成记录 | PASS | 精确命令、限制、真实输入清单与选定 JSON 已归档。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** ac92396 至 4d62590｜**当前前沿：** 向 primary 交付，无未完成 CPU 实现。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 工具是交付本身；要求现可验证，不再扩展功能。 |
| D2 细节反复重试 | CLEAR | 格式变量遮蔽一次修正；环境缺包改用已有 3.12，无行为 fallback。 |
| D3 支线问题修补 | CLEAR | 所有检查均服务格式、身份和指标验收。 |
| D4 可并行任务停滞 | CLEAR | 同一共享文件，当前是顺序实现。 |
| D5 超范围镀金 | CLEAR | 没有新增 comparator 或训练功能。 |
| D6 分析空转 | CLEAR | 已有可执行 Step/GLM CLI 与明确输出。 |
| D7 路线违背 | CLEAR | D20 明确授权 CPU 与共享接口扩展。 |

**文档检查：** S1–S6 均未发现明显缺陷。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 向 primary 交付两个实现 commit、任务证据 archive、验证命令与仍未具备的真实证据。
2. 后续 primary 做 phase review 与集成，不将该 CPU 交付替代 native 验收，不扩展到 GPU 或训练。

## 4. 最值得关注的 3 点

1. Step 没有 GLM clean Git manifest，不能伪造。
2. 复用已有 run-check owner，缺失原始 runtime/source/image 证据不得 PASS。
3. 每轮独立比较与四项 10% 不变，pooled 仅诊断。

## Verification iterations

- Shared reuse first exposed a missing `csv` import; adding the required import passed 62 tests with one Python-version skip.
- First Python 3.12 extension run found the new source-format variable shadowed by the preexisting metric-field loop. Renaming that loop variable fixed the observed cause. It also exposed a shared test's unavailable `test_native_streaming_replay` helper and its unmerged GLM-only replay implementation. The test now checks the same compressed raw-schedule rejection directly; the existing real local-HTTP P/D harness remains independently tested. No whole GLM branch or unrelated replay implementation was imported.
- Python 3.10 lacks `aiohttp`; the Step consumer suite cannot collect there and the P/D HTTP tests fail. Python 3.12's already installed environment has aiohttp 3.13.3; those suites pass there without installing packages or changing behavior. A speculative test filename also returned pytest exit 4 before collection; the inspected actual `test_calibration_pd_harness.py` was then used.
- Final focused run: 106 PASS/47.19s, `/data/ycfeng/tmp/step5_client_metrics_verify_20261021_02`. Internal/legacy: 148 PASS/6.24s. P/D harness/worker: 77 PASS/11.38s.
- No external network, authentication, GPU operation, push, scheduler/predictor/profiler/R2 edit or frozen workload/stage rewrite occurred. HTTP harness checks use local loopback fixtures only.

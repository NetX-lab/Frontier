# D21-NATIVE-REPLAY progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Completed native multi-API diagnosis, caller fix, offline stage and historical64 judgment. |

- completed: clean worktree at42dda4c; read primary spec §0, D21, ticket09 and local parent requirements/progress.
- completed: local `python3 -B` assertion over four complete P0 server logs fails1 with `complete_node_logs 4 EngineNNN_rows 0`; original symptom reproduced without GPU.
- completed: ranked hypotheses were logger suppression/aggregation, MRv2 transport loss, log truncation. Actual head logs prove omitted API count defaults toDP2 (P0) /DP4 (old64), then each API prints `disabling stats logging to avoid incomplete stats`. Locked `StatLoggerManager` disables LoggingStatLogger for client_count>1 but retains Prometheus. Capture is not replay proof.
- completed: source/caller regression executes actual native CLI defaulting and StatLoggerManager construction. DP2/DP4 both failed with zero text logger adapters before the fix; both pass after explicit api_server_count=1 on heads, without passing the incompatible flag to headless nodes. Existing native source tests still cover MRv2 dispatch/transport/runtime tables. Added the three frontend source dependencies to pinned runtime bindings.
- completed:125 affected CPU checks PASS in86.13s, including the four priority topologies, actual-source selection, full exact HTTP populations, idle/error/soft-gate behavior and worker lifecycle. `git diff --check` PASS. No native kernel, request/accounting/threshold/soft-gate or simulator code changed.
- completed: code committed6838bc10135faf5bea1bc48321837caa938f85a1; fresh `/data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/stage` built from that committed source. Standalone verify, exact submission dry-run and shell syntaxPASS;37 staged files/50 native pins, APIcount1 on heads and omitted on headless nodes. No submit; native replay remainsUNVERIFIED.
- completed: old64 business judgment confirms24 exact requests,64 unique H200 UUIDs, allDP4 lanes,32 model-loaded EP ranks per role and both required idle boundaries. Original REFERENCE_UNSTABLE is retained. Initial missing gauges are diagnostic only; replay remains absent for the demonstrated same logger cause. Shutdown EngineDeadError records occur after requested stop and remain recorded, not relabelled clean.
- completed: `report.md` lists exact case-aligned G3 witness/reference gaps, basic reuse and fresh-stage recommendation; `evidence_receipt.json` hashes90 used evidence/source files and8 package pin-check records. All6 preexisting D21 sealed01/02 tool manifests still match stored pins; old64 original27 tool/47 source/3 metadata pins match. Requirements reread before closing.
- pending outside child scope: parent-authorized actual replay observation, G3/native readiness, admitted inputs and complete formal clean/reference comparison. P0 and H20064 formal-readyNO. No GPU/auth/network/SSH or further delegation performed; no failed/unfinished child implementation remains.

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 1 根因与源码链 | PASS | 实际API警告与锁定源码一致 |
| 2 最小修正、CPU验证、commit和fresh-stage | PASS |6838bc1；125 checks；offline stage验证通过 |
| 3 old64复用与G3缺口 | PASS |PARTIAL reuse、原始结局、独立G3缺口已归档 |
| 4 最终交付 | PASS |report、summary、receipt及可复现commands完成 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：**125 CPU checks通过至今｜**当前前沿：** 交回父任务；child scope已完成

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 最小caller correction与一个fresh stage直接解决委派问题；未继续扩展工具 |
| D2 细节反复重试 | CLEAR | 一个fixture缺少标准化engine字段、一个已知Python环境缺包，各定位一次 |
| D3 支线问题修补 | CLEAR | 未编辑simulator或其他问题 |
| D4 可并行任务停滞 | CLEAR | user禁止delegation；old64读取同步推进 |
| D5 超范围镀金 | CLEAR | 两行argv、一项config、验证/源码pins；无新运行模式 |
| D6 分析空转 | CLEAR | 已交付root cause、代码commit、stage和old64判定 |
| D7 路线违背 | CLEAR | 无平台操作、历史改写或G3替代 |

**文档检查：** S1–S6均未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. Parent接收6838bc1与qualification03 → 完成标志：明确offline准备不是actual replay；如授权运行，保留两条D owner lane的EngineNNN/FULL证据。
2. Parent按已有计划完成独立G3与准入 → 完成标志：实际prerequisites满足才formal，不为initial gauges盲目重跑64卡。

## 4. 最值得关注的 3 点

1. graph指标路径存在，实际native logger因多API被关闭，不是凭capture推断replay。
2. API count=1不减少任何TP/DP/EP rank，但新旧frontend配置的timing不可当成相同clean reference。
3. old64已有基本运行证明，不能因initial gauge诊断盲目重跑64卡；实际replay和G3仍缺。

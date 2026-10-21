# D21-NATIVE-REPLAY progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Reproduced absent graph rows and identified native multi-API logger suppression. |

- completed: clean worktree at42dda4c; read primary spec §0, D21, ticket09 and local parent requirements/progress.
- completed: local `python3 -B` assertion over four complete P0 server logs fails1 with `complete_node_logs 4 EngineNNN_rows 0`; original symptom reproduced without GPU.
- completed: ranked hypotheses were logger suppression/aggregation, MRv2 transport loss, log truncation. Actual head logs prove omitted API count defaults toDP2 (P0) /DP4 (old64), then each API prints `disabling stats logging to avoid incomplete stats`. Locked `StatLoggerManager` disables LoggingStatLogger for client_count>1 but retains Prometheus. Capture is not replay proof.
- completed: source/caller regression executes actual native CLI defaulting and StatLoggerManager construction. DP2/DP4 both failed with zero text logger adapters before the fix; both pass after explicit api_server_count=1 on heads, without passing the incompatible flag to headless nodes. Existing native source tests still cover MRv2 dispatch/transport/runtime tables. Added the three frontend source dependencies to pinned runtime bindings.
- completed:125 affected CPU checks PASS in86.13s, including the four priority topologies, actual-source selection, full exact HTTP populations, idle/error/soft-gate behavior and worker lifecycle. `git diff --check` PASS. No native kernel, request/accounting/threshold/soft-gate or simulator code changed.
- in-progress: old64 business judgment finds24 exact requests,64 unique H200 UUIDs, allDP4 lanes and both required idle boundaries; original REFERENCE_UNSTABLE is retained. Its initial missing-gauge failure is diagnostic only, but actual replay remains absent for the same multi-API reason.
- pending: verified fresh offline stage, final hash receipt/commit and handoff. No GPU/auth/network/SSH or further delegation performed.

## 1. 规划完成情况

**汇总：** PASS 1｜BLOCKED/FAILED/SKIPPED 0｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 1 根因与源码链 | PASS | 实际API警告与锁定源码一致 |
| 2 最小修正、CPU验证、commit和fresh-stage | TODO |125 checks通过；commit/stage待完成 |
| 3 old64复用与G3缺口 | TODO | 已有实际请求/idle/topology，待归档 |
| 4 最终交付 | TODO | 等待stage与证据hash |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 本委派开始至今｜**当前前沿：** commit/stage与old64判定

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 最小caller correction直接解决委派问题；验证已停止扩展 |
| D2 细节反复重试 | CLEAR | 一个fixture缺少标准化engine字段、一个已知Python环境缺包，各定位一次 |
| D3 支线问题修补 | CLEAR | 未编辑simulator或其他问题 |
| D4 可并行任务停滞 | CLEAR | user禁止delegation；old64读取同步推进 |
| D5 超范围镀金 | CLEAR | 两行argv、一项config、验证/源码pins；无新运行模式 |
| D6 分析空转 | CLEAR | 已有确定根因、修正与125 PASS |
| D7 路线违背 | CLEAR | 无平台操作、历史改写或G3替代 |

**文档检查：** S1–S6均未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. commit已验证caller修正并构建一个fresh P0 offline stage → 完成标志：原生单API参数、全部DP lanes、standalone verify和dry-run通过；不submit。
2. 归档old64可复用证明与G3缺口 → 完成标志：保留原始outcome，formal-ready NO，parent决定下一次native执行。

## 4. 最值得关注的 3 点

1. graph指标路径存在，实际native logger因多API被关闭，不是凭capture推断replay。
2. API count=1不减少任何TP/DP/EP rank，但新旧frontend配置的timing不可当成相同clean reference。
3. old64已有基本运行证明，不能因initial gauge诊断盲目重跑64卡；实际replay和G3仍缺。

# W128-V4-NATIVE-SERVE-PROFILE progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Inspected the mandatory sources, reproduced launch-seam blockers, verified baseline suites, and stopped before implementation under the file-ownership limit. |

Status: **BLOCKED**. Baseline: `993f05f59ef22bfab38433dd93e2fcf486fc113a`, branch `feat/v4-native-serve-profile`.

- completed: Read parent requirements, step H, design, proposal, native seam, lifecycle control, retained-trace minimum additions, repository instructions, worker/inventory/replay implementations, existing tests, and pinned native hook callsites.
- completed: CPU probes execute the real launch utilities in this worktree and establish three missing integration paths; see `launch_seam_receipt.json` and the test report.
- completed: Existing native serving, V2 serving, and R4 reader suites passed separately (28, 45, and 135 tests).
- pending: Collector implementation, collector stand-in tests, installed-image hook resolution, and code acceptance. No collector source or existing source was edited.
- blocked: The assigned file list excludes two files required for the selected extension to launch with balanced routing and profiler configuration. The explicit stop-on-increment instruction applies.

## Required scope increment

The proposed, **unapplied** `scope_increment.diff` adds exactly **12 changed lines across two additional existing files**:

1. `tests/comparison/dp_placement_pp/vllm_replay.py`: 7 added + 1 deleted = 8 changed. A four-line extension-name table keeps existing balanced-routing registration and new extension selection explicit; the existing refusal tests remain authoritative. Forward the existing native `profiler_config` as JSON, not Python `str(dict)`.
2. `tests/comparison/calibration/pd_replay.py`: 2 added + 2 deleted = 4 changed. Reuse that table for the existing worker `PYTHONPATH` path. No driver or phase-bracketing behavior is added.

`native_pd_worker.py` can then reuse the same table within its already authorized 25-line allowance. This is required integration, not a new shared TP parameter or a profiler lifecycle workaround. The root must allocate these extra files or approve the increment before this owner resumes. Parent records remain untouched.

## 1. 规划完成情况

**汇总：** PASS 2｜BLOCKED/FAILED/SKIPPED 1｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| H mandatory source inspection | PASS | Three launch omissions reproduced in `launch_seam_receipt.json`. |
| H existing-suite verification | PASS | Test report: separate 28/45/135 PASS results. |
| H extension integration | BLOCKED | Two necessary files are outside the explicit allocation. |
| H collector implementation and tests | TODO | No code started after the scope blocker was verified. |
| H installed-image resolution and code commit | TODO | Only local image identity was read; no collector hook list exists yet. |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** Task start 至今｜**当前前沿：** Root scope allocation, then H implementation.

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | Only required source reads, one launch probe, and requested suites. |
| D2 细节反复重试 | CLEAR | No failed fix attempts or repeated parameter changes. |
| D3 支线问题修补 | CLEAR | No unrelated edits or investigations. |
| D4 可并行任务停滞 | CLEAR | Explicit single-owner/no-delegation constraint; independent suites ran concurrently. |
| D5 超范围镀金 | CLEAR | Required extra integration recorded as an unapplied proposal, not implemented. |
| D6 分析空转 | CLEAR | Source inspection produced a verified scope decision; stopped immediately. |
| D7 路线违背 | CLEAR | No GPU, platform, SSH, cloud, native/C26 edits, scaling, or CPU-cost inference. |

**文档检查：** S3: the H assignment requires usable launch integration but excludes the actual balanced-routing and profiler-config command builder. S1/S2/S4/S5/S6: no additional defect found in this assigned scope.
**归因：** 无执行偏离；S3 is a file-allocation blocker, not evidence that the collector itself is infeasible.

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. Root-owned step-H allocation: add the two files and the exact 12-line launch increment in `scope_increment.diff`, or assign that increment to root（需用户确认：explicit owner allocation adjustment by root）.

**需停止的偏离行为：** 无

**接下去的路线：**
1. Root allocates the launch integration → 完成标志：the two missing files have an authorized owner.
2. H collector and stand-in tests → 完成标志：all assigned annotation, event, failure-isolation and inventory-preservation criteria verified within budget.
3. Root driver and later GPU proposal → 完成标志：phase bracket, topology, runtime proof gates and resource request are explicit; no launch before approval.

## 4. 最值得关注的 3 点

1. Balanced routing rejects the new extension before native startup; subclassing does not change this string selector.
2. The command builder currently drops `profiler_config`; existing native CLI fields alone do not enable a profiler through this harness.
3. All baseline suites pass, but there is no collector deliverable or new stream/PDL/capture-survival evidence yet.

# W128-V4-NATIVE-SERVE-PROFILE progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Inspected the mandatory sources, reproduced launch-seam blockers, verified baseline suites, and stopped before implementation under the file-ownership limit. |
| 2026-10-23 | Root approved the two-path increment; completed the composed collector, GREEN launch controls and CPU/image verification. |

Status: **completed (CPU collector); GPU qualification pending**. Baseline: `993f05f59ef22bfab38433dd93e2fcf486fc113a`, branch `feat/v4-native-serve-profile`.

- completed: Read parent requirements, step H, design, proposal, native seam, lifecycle control, retained-trace minimum additions, repository instructions, worker/inventory/replay implementations, existing tests, and pinned native hook callsites.
- completed: CPU probes execute the real launch utilities in this worktree and establish three missing integration paths; see `launch_seam_receipt.json` and the test report.
- completed: Existing native serving, V2 serving, and R4 reader suites passed separately (28, 45, and 135 tests).
- completed: Composed collector preserves the inherited RPC and original inventory observer; writes beside serving/.
- completed: 22 new collector tests, executable pinned native fan-out/join controls, separate existing suites, and real GREEN launch probes.
- completed: Locked-image installed-source24-symbol check and real torch2.13 CPU annotation control. Direct native imports timed out; runtime execution is not qualified.
- pending: Root review/driver; GPU stream mapping, capture/replay survival, PDL per-kernel facts and CUPTI restart; explicit GPU request.

## Required scope increment

Root explicitly approved the 12-line increment in the continuation request and main requirements/plan. Historical `scope_increment.diff` remains the original proposal; corresponding code is now applied:

1. `tests/comparison/dp_placement_pp/vllm_replay.py`: 7 added + 1 deleted = 8 changed. A four-line extension-name table keeps existing balanced-routing registration and new extension selection explicit; the existing refusal tests remain authoritative. Forward the existing native `profiler_config` as JSON, not Python `str(dict)`.
2. `tests/comparison/calibration/pd_replay.py`: 2 added + 2 deleted = 4 changed. Reuse that table for the existing worker `PYTHONPATH` path. No driver or phase-bracketing behavior is added.

`native_pd_worker.py` reuses the same mapping/module hash and supplies optional explicit `serving_profile_limits`. No TP parameter/default capacity is invented; root selects limits. Parent files/commits remain untouched/unmerged.

## Implementation evidence

- Gain: actual torch profiler ownership parents, separate nested outer prefixes, no kernel-name guesses/timing sums.
- Reuse existing execution-local `dispatch_evidence` before inventory clears it; retain local requests/tokens, native classifier has_prefill, synchronized descriptor/mode and exact dispatch join. No collective/readback added.
- Only actual Event.record/wait calls inside pinned parallel functions create handle/generation/stream rows. Disabled slots and sequential execution create no invented edges. CUPTI mapping failures retain raw handles/null IDs and traceback.
- Saved V2 `_prepare_and_attn_fn` rebound to the wrapped method. Sparse eager decorator stays outside annotation so the saved callback contains it; direct capture/Python-replay control emits two ranges and preserves the marker.
- First collector run:12 PASS/3 FAIL caused by stand-in context/signature and wrong dummy expectation; corrected from native shapes. Final:22 PASS.
- Extra worker suite initially failed on fake vllm_replay missing registry; module-level sibling import preserves one actual mapping without fallback. Final:16 PASS.
- Native import/JIT image control120-second timeout isolated by stopping only that CPU container. Source24-symbol check PASS. Read-only torch scratch failure fixed with approved local scratch; torch2.13 CPU ranges PASS, CUDA uninitialized.

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| H mandatory source inspection | PASS | Three launch omissions reproduced in `launch_seam_receipt.json`. |
| H existing-suite verification | PASS | Test report: separate 28/45/135 PASS results. |
| H extension integration | PASS | Real launch functions GREEN; one sibling mapping. |
| H collector implementation and tests | PASS | 22 collector tests and local1..11/padded16/callback controls. |
| H installed-image runtime qualification | TODO | Source symbols and torch CPU ranges PASS; native imports/runtime CUDA excluded or incomplete. |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** Root approval continuation 至今｜**当前前沿：** Commit CPU deliverable, root review/driver.

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | Controls/affected suites settle assigned criteria, no new framework. |
| D2 细节反复重试 | CLEAR | Stand-in causes diagnosed and corrected once; bounded image alternative. |
| D3 支线问题修补 | CLEAR | No unrelated edits or investigations. |
| D4 可并行任务停滞 | CLEAR | Explicit single-owner/no-delegation constraint; independent suites ran concurrently. |
| D5 超范围镀金 | CLEAR | Five allocated source files; no driver/timing model. |
| D6 分析空转 | CLEAR | Inspection produced alias/decorator controls and completed source. |
| D7 路线违背 | CLEAR | No GPU, platform, SSH, cloud, native/C26 edits, scaling, or CPU-cost inference. |

**文档检查：** S1–S6 均未发现明显缺陷；earlier S3 resolved by root allocation.
**归因：** 无

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动

**需停止的偏离行为：** 无

**接下去的路线：**
1. Archive and commit H → 完成标志：five source files/records committed, worktree clean.
2. Root review/driver and G2 proposal → 完成标志：phase bracket/capacity/topology/runtime gates and explicit GPU approval.

## 4. 最值得关注的 3 点

1. Collector/launch source CPU-verified; original inventory preserved.
2. Image24-symbol and real torch CPU checks pass; native runtime imports/CUDA not qualified.
3. Sidecars stay UNVERIFIED pending GPU mapping/completeness/PDL/capture/restart evidence.

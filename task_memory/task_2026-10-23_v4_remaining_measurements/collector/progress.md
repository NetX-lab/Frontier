# W128-V4-NATIVE-SERVE-PROFILE progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Inspected the mandatory sources, reproduced launch-seam blockers, verified baseline suites, and stopped before implementation under the file-ownership limit. |
| 2026-10-23 | Root approved the two-path increment; completed the composed collector, GREEN launch controls and CPU/image verification. |
| 2026-10-23 | Repaired five unique review findings, refreshed fixed-source tests/hashes, and prepared CPU-only handoff. |
| 2026-10-23 | Root accepted and integratedf8621a8 at64f3c950; owner-delivery pending statuses below are historical. |

Status: **completed (CPU collector); GPU qualification pending**. Baseline: `993f05f59ef22bfab38433dd93e2fcf486fc113a`, branch `feat/v4-native-serve-profile`.

## Root acceptance after delivery

Current status: **ACCEPTED_CPU_COLLECTOR_COMPONENT**, sourcef8621a8 integrated64f3c950. Both independent re-reviews closed all five findings; root final-source and merged suites independently reproduced collector28/inventory28/V2inventory45/reader135/worker16/DP75/P-D68/manifest14, separately. Joint automatic94 is unchanged; default77matrix is explicitly bound to the existing capture, with no new capture. See `../collector_primary/primary_acceptance.json` and `../progress.md` for current pending G2 work. GPU qualification remains UNVERIFIED. The owner-delivery/root-review TODO statements below describe their historical time and are superseded by this acceptance.

## Review repair and handoff (owner delivery)

Root continuation supersedes the historical allocation below: collector<=550, one test<=400, worker<=25, command builder<=8, P/D env<=4; five non-core files<=987 changed lines. Actual final scope:457+400+11+8+4=880. Relative to d94b821a:collector92 added/40 deleted; test112 added/12 deleted; other three unchanged. Net scope increment152 lines, patch edit volume256; no native/C26/inventory/root edits or merges.

- completed P1 lifecycle: ModelCudaGraphManager class hooks independent of the load-time None instance; other installers/facts separately observed. CPU None-at-load -> initialize_kv_cache -> FULL dispatch control passes; independent instance-install failure does not skip replay/facts.
- completed P2 metadata: GPUModelRunner.prepare_attn owns early GPU metadata work; BlockTables.compute_slot_mappings is a nested owner. Pinned native prepare_attn body executes this boundary before DefaultModelState metadata control.
- completed P2 source-visible PDL: indexed Triton launch wrappers record actual internal RMSNorm launch_pdl and inverseRoPE USE_GDC/launch_pdl; importing-module CuTeDSL callable owns the CUDA head512 local-import path. Pinned native function-body controls pass. Actual per-kernel GPU PDL remains unknown.
- completed P2 bytes: context updates execution-local pending state only; final detached enriched step uses the same atomic append admission. A95-byte limit previously retained391 serialized bytes/accounted204; now drops the whole step, accounts0 and increments dropped1. Shared dispatch mutation cannot change retained bytes.
- completed P2 log flood: retain/print at most64 detailed errors; later failures count in suppressed_errors without formatting/printing traceback. Repeated-failure RED501 detailed messages -> GREEN64 +437 suppressed;100 native results and original exception unchanged.

RED evidence: `repair_red.txt` (d94b821a source,5FAIL+1ERROR across six controls); `repair_head512_receipt.json`, `repair_retention_receipt.json`, `repair_log_receipt.json`. GREEN final source:collector28, inventory28, V2 inventory45, R4 reader135, worker16, DP75, P/D68, manifest14, each separate. Locked image31/31 installed source symbols pass; no native/JIT/CUPTI runtime retry.

Handoff: fixed source hashes, exact commands and evidence hashes in `repair_receipt.json`; repair rationale/limits in the maintained test report. CPU source delivery complete. Root owns independent re-review and integration; review closure is not self-granted. This session does not implement driver or begin GPU work. Profiles remain UNVERIFIED; GPU stream mapping, per-kernel PDL, compile/breakable annotation/event survival, captured-segment ownership and CUPTI restart remain UNKNOWN. Next session must read this section/repair receipt and root's current requirements/plan, verify fixed-source hashes, then follow root-assigned scope; no run authority is carried by this handoff.

### Repair phase self-check (check-progress)

PASS:five finding repairs/source controls/affected suites. TODO:root independent acceptance. SKIPPED:driver/GPU/native-runtime imports by explicit scope. Mainline is repairs -> fixed-source CPU evidence -> commit/handoff. D1–D7 CLEAR:every edit/control is assigned; no additional framework, new route, unexplained retry loop, delegation, or GPU. S5:parent allocation text still787, while continuation explicitly authorizes987; root owns updating its records, owner records latest authority here without editing root. BranchB correction:root synchronizes allocation when persisting review; already-explicit approval permits repair completion. Three key facts:five closures CPU-verified; profile UNVERIFIED/runtime unknowns unchanged; next action is commit/clean handoff, not driver/GPU.

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

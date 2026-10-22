# Native implementation progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Read approved allocation/design and begin interface milestone. |
| 2026-10-23 | Complete native19 integration, CPU418 PASS, D3048 PASS and static T2; retain root-only prerequisites. |

- completed: clean branch feat/step5-position-native at1aee0af; all approved native
  design sections and local instructions inspected; no peer writes/platform calls.
- completed: exclusive copied candidate
  /data/ycfeng/tmp/d29-native-derived-20261023-candidate01; immutable source unchanged.
- completed: N1-N4 scope adds actual tensor reference, deterministic sigmoid
  strategy and configured factory inputs without host readback/collectives.
- completed: actual-source CPU9 PASS; neighboring baseline251 PASS; native99
  gross lines with reviewable patch/hash/snapshot and interface_receipt.json.
- completed: packaging/clients/provenance and T2 script; CPU418 PASS, D3048 PASS,
  actual-source11 PASS, sparse build-context PASS, owned19/AST/whitespace PASS.
- completed: early e48cdc0 andintegration d12941c; summary/handoff records ready.
- pending (root): image build/publication/actual digest and all GPU acceptance.

## 1. 规划完成情况
**汇总：** PASS4｜BLOCKED/FAILED/SKIPPED0｜TODO0
| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| native接口及CPU | PASS | actual-source9 PASS，receipt/patch已生成 |
| packaging/clients/provenance | PASS | CPU418 PASS，sparse context PASS |
| T2实际graph script | PASS | AST及source contract PASS，GPU归root |
| 最终quota/handoff | PASS | committed19-path receipt matches1017 gross；summary及needed root inputs完整 |

## 2. 偏离检测
**结论：** 未偏移，正常推进。
**检测窗口：** integration d12941c至archive完成｜**当前前沿：** root image/export/live验收（本流外）
| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 仅approved clients/provenance/T2及验收，没有host overhead |
| D2 细节反复重试 | CLEAR | pytest路径、缺plotly、root层级分别定位；用已存在环境和正确root，不修改native行为 |
| D3 支线问题修补 | CLEAR | 仅native19 |
| D4 可并行任务停滞 | CLEAR | 用户要求early interface先交付，禁止subagent |
| D5 超范围镀金 | CLEAR | 四个native文件及15仓库文件，未扩界 |
| D6 分析空转 | CLEAR | 产出actual source和CPU证据 |
| D7 路线违背 | CLEAR | 无GPU/Docker/peer写入 |
**文档检查：** S5：summary旧design-only状态已在current implementation outcome中更新并保留历史。其余S1–S6未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议
**分支：** B：archive状态已过时，先更新本流记录。
**文档更新（先执行）：**
1. summary.md和progress.md：记录已完成implementation及1017/1582和root-only prerequisites（可直接执行，已完成）。
**需停止的偏离行为：** 无。
**接下去的路线：**
1. 文档archive commit -> 完成标志：requirements重读，记录与actual committed source一致。
2. root实际image build/source export/GPU -> 完成标志：真实immutable identity及各项live证据，不由本流执行。

## 4. 最值得关注的 3 点
1. GPU/graph/G3仍UNVERIFIED。
2. root提供真实derived digest，不冻结假值。
3. 四native文件99 gross，完整stream1017/1582；最终receipt绑定actual HEAD diff。

## Verification iterations
- Baseline251 PASS; interface9 PASS then addedinvalid K/logits controls11 PASS.
- Native/HTTP/canonical/reference/provenance/capacity/source suites418 PASS/39.78s
  (asyncopd-py312); D30finish sync/native census48 PASS/67.40s(frontier-py310).
- Earlier asyncopdD30 collection lackedplotly; existingfrontier-py310 used with
  no dependency install. Old D30 root convention expects candidate parent, while
  T1 expects candidate/vllm; supplying correct independent roots closes failure.
- T2 source-contract+AST/Dockerfile payload AST PASS, no GPU or Docker execution.
- Sparse context contains only four edited native files plus pins/manifest/Dockerfile.
  Base53 build pins are50 sealed pins plus3 byte-validated local base exports,
  never claimed as installed derived pins. Capture02/03manifest checks PASS,
  capture01not present at inspected root and no writes occurred.

## Preserved pre-allocation record (historical)

# D29 native routing progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Integrate c9e9274 and independent follow-up resolving all four native design findings. |
| 2026-10-22 | Record YC policy approval and integrate native design; numerical implementation scope remains pending. |
|2026-10-22|Recorded read-only native path findings and proposal investigation.|

## Current status

- completed: YC approved `position_cyclic`, necessary native/Frontier/profiling
  changes and internal derived-image publication; exact authority is recorded
  in `requirements.md` and the parent D29 decision.
- completed: `native_design.md`, commit `c9e9274`, is integrated on the parent
  branch. Estimated native-facing scope: 19 files, 1003–1582 added/deleted lines;
  the planned binding uses existing device positions at the uncompiled Step outer forward.
- completed: independent delta review resolves the four initial design findings
  with no new actionable issue. Parent recomputed 19 unique paths and the exact
  1003–1582 line range; the corrected proxy and reference-test paths exist.
- in-progress: Frontier/profile design review and complete pricing scope,
  followed by one combined numerical scope proposal. D30 proceeds in parallel.
- pending: approved bounded implementation, source/CPU/GPU graph verification,
  derived image, new source pins/package, and actual native/G3/profile evidence.
  Capture03 is a historical sealed pressure candidate, not a position-policy run.

## Completed investigation (historical)

- completed: frozen capture03 config gives default auto backend, no mega/SP override, default allgather_reducescatter, no EPLB. FP8 dynamic128 layers3–87; BF16 routed88–90; swiglustep7, E352/K8, expert-localTP1/EP16,22 full experts per rank.
- completed: retained actual H800 S2 TP8/DP2/EP16 module run uses auto backend and observes Fp8MoEMethod/TritonOrDeepGemmExperts→DeepGemmExperts and BF16/UnquantizedFusedMoEMethod→TritonExperts, both NaiveDPEPModular and no SP. E8M0 OR shape predicate determines DeepGEMM; small M does not alone imply Triton (the retained M9 case used DeepGEMM). This is native module evidence, not full serving/graph qualification.
- completed: exact oracle source removes H800 monolithic FP8 TRTLLM by SM100-only device predicate and rejects preferred CUTLASS for swiglustep; DeepGEMM modular supports it. BF16 H800 priority starts Triton modular. Both call `MoERunner._apply_quant_method`→router.select_experts before modular prepare's DP all_gatherv. Existing model gate remains upstream.
- completed: normal factory creates biased-sigmoid router; nonempty existing simulation env creates RoutingSimulatorRouter instead. Only random-uniform/normal built-ins exist; round-robin is expert placement and other load_balance matches belong to vision/attention. No balanced setting found.
- completed: `contract_probe.py` actual frozen factory/runner/DP method seams and seven hypothetical global-prefix count cases PASS; exact Frontier Hamilton counts, distinctness and conservation. DP reset fails6/7 cases. All16 retained native module witnesses agree; exact source/witness hashes retained in `cpu_receipt.json`.
- completed: `semantic_probe.py` counterexamples PASS. Forward reset differs on16 tail rows for P6128+D16/full6144 and4096 rows for chunks[2048,2048,2032,16]; absolute-position IDs preserve every real row. q1 starts224/232/240/248, not0. Stable-position and current ascending-Hamilton cannot both hold for arbitrary waves. Actual native dummy method CPU execution yields positions0 and padding true.
- completed: `proposal.md` records actual reachability, native settings absence, MRv2 exact device positions vs optimistic CPU mirror, FULL graph capture implications, weight/mask contract and two materially different pending decisions. No native strategy/image/package change or backend flip performed. Capture03 remains not balanced.
- completed: preservation check verified all57 capture03 manifest entries and50 native exported-source pins unchanged;3 probe sources compile. Exact manifest digest unchanged and no attempt/submitted marker. `test_report_2026-10-22_native_routing_contract.md` and `check_progress_2026-10-22.md` complete; no route drift.
- completed: task-only records/probes/receipts prepared for commit after `git diff --cached --check` PASS; no files outside this task are staged. Clean branch/commit identity is returned in the final handoff.
- completed: the original proposal-only handoff preceded YC's policy approval;
  remaining implementation work now follows the current status above.

Commands (owned repo, Python3.12.13/Torch2.9.1+cu128, CPU-only):

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_step5_d29_native_balanced/contract_probe.py
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_step5_d29_native_balanced/semantic_probe.py
```

Semantic probe first attempt failed only on the CPU seam object lacking `.device`; no receipt written. Inspected actual `make_dummy` dependencies, supplied CPU device and local deterministic UUID seam; one rerun passed. This did not change native behavior or suppress a native error. Receipts are exclusive writes; rerunning requires a new filename/directory.

Preservation probe first attempt misread image-installed absolute pin paths as local paths (`FileNotFoundError` for `/usr/local/lib/python3.12/dist-packages/vllm/config/kv_transfer.py`). Inspected sealed pin format and mapped its exact fixed prefix to the retained local export; one rerun passed50/50. Manifest/pins untouched. Task directory follows existing ignored-task convention; explicit force-add only this task's files, never production/native or another task's files.

Local calibration site GPU defaults are overridden by explicit task H800/read-only/provider rules. Source-only/module receipts are not formal timing/GPU route attestation. No CPU overhead measurements or suits.

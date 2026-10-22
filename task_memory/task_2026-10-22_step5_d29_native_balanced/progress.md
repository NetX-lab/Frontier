# Native implementation progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Read approved allocation/design and begin interface milestone. |
| 2026-10-23 | Complete native19 integration, CPU418 PASS, D3048 PASS and static T2; retain root-only prerequisites. |
| 2026-10-23 | Close four reviewer findings;485 affected CPU PASS and1269/1582 gross. |
| 2026-10-23 | Import accepted9090407 and prepare controlled T2 metadata/inputs/resource proposal; static checks only. |

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
- completed: four-finding repair, source/AST/hash/ownership/quota checks and
  final affected CPU485 PASS/36.16s; scoped code+evidence commit74d5c48.
- completed: root accepts native6827507 at9495cfb, publishes67d5b5e and archives
 9090407; this worktree fast-forwards cleanly to9090407 with root authorization.
- completed: D29-T2-INPUT-PREPARATION P1 retained metadata/publication/source identity;
  P2 controlled config/cases; P3 exact existing commands and resource proposal.
- completed: P4 schema/types/bounds/static topology/hash review and bash-n-only
  commands; all6 artifacts in hashes.sha256 verify. Input-only commit handoff
  is being finalized; no production/test delta against9090407.
- pending (root): independent sealing/readiness, resource/run approval, actual
  mounts/rendezvous/output and GPU evidence. Exact engine7168 remains unsupported
  by existing T2 interface if root requires more than the prepared input bound.

## 1. 规划完成情况
**汇总：** PASS3｜BLOCKED/FAILED/SKIPPED0｜TODO1（当前input-only任务）
| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| P1接受源码/metadata定位 | PASS | clean fast-forward9090407；actual retained config及revision/hash已绑定 |
| P2controlled inputs | PASS | graph2×17；module3个DP frames；原metadata字节未改 |
| P3commands/resources | PASS | existing torchrun/script，最低16H800，提案30min/8GPUh，非授权 |
| P4static/records/commit | TODO | schema/hash/topology/bash-n/sha256sum PASS；owned交接commit待完成 |

## 2. 偏离检测
**结论：** 未偏移，正常推进。
**检测窗口：** D29-T2-INPUT-PREPARATION dispatch至static输入检查｜**当前前沿：** P4记录及commit
| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只做用户要求static input/hash/command检查，未重跑source/CPU验收 |
| D2 细节反复重试 | CLEAR | 缺metadata路径后定位actual root retained input；没有行为修改/运行重试 |
| D3 支线问题修补 | CLEAR | 输入来源/拓扑/compat及7168限制均服务P1–P3 |
| D4 可并行任务停滞 | CLEAR | 明确禁止subagent；独立reads/checks并行，root持有GPU决策 |
| D5 超范围镀金 | CLEAR | 零production/test edits，无新mode/harness/tolerance，3frames来自已批准design |
| D6 分析空转 | CLEAR | 本任务交付即input/resource提案，已生成config/cases/provenance及commands |
| D7 路线违背 | CLEAR | no launch/platform，保持metadata1048576，不伪造engine7168或observed traces |
**文档检查：** S5 notes/progress旧publication pending与9090407矛盾已纠正；S3如7168意味着actual engine bound，现T2/helper无参数，已作为root依赖明确记录。S1/S2/S4/S6无明显缺陷。
**归因：** 无。

## 3. 纠正建议
**分支：** B：先纠正过时publication状态并显式记录input7168与engine7168的区别。
**文档更新（先执行）：**
1. notes/progress/summary：published image/export已完成；本轮inputs未运行（可直接执行）。
2. plan/requirements及execution_proposal：所有positions<7168，实际engine context未绑定；若root要求强绑定，需root决定supported seam（需用户确认；本任务不改代码）。
**需停止的偏离行为：** 无。
**接下去的路线：**
1. P4static输入交接：hash/source/paths/commands与现有schema一致并commit → 完成标志：只有owned records/input artifacts，无code changes。
2. root独立accept/seal/资源决策及7168依赖 → 完成标志：finite-unit source/input身份、真实资源/路径、明确actual run approval。

## 4. 最值得关注的 3 点
1. T2 inputs为controlled/unobserved，GPU/graph/backend/G3均UNVERIFIED。
2. inputs<7168，但现有EngineArgs未设max_model_len；不可把metadata改小来伪造绑定。
3. 最低16H800/2nodes；30min/8GPUh只是root资源提案，非launch approval。

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

## Four-finding repair iterations
- Reviewed source2c4cc9e; root independently verified418 PASS,19/1017 and source
  hashes, and reported context02 local build only.
- Red CPU regression14 FAIL/20 PASS/157 deselected/2.76s: keyword original called
  once but capture empty, stale dummy assertion and forward/kernel/monolithic
  provenance false positives reproduced. See d29-native-repair-red.log.
- Fixed explicit router signature/named logits, dummy replay-before-baseline,
  actual bound forward, and checkpoint/oracle class/source/modular relationships;
  soft observer boundary and native outputs/exceptions unchanged.
- Focused green01:275 PASS/11.45s; green02:307 PASS/8.35s; affected01:484 PASS/38.12s;
  final affected02:485 PASS/36.16s after actual PW dispatch/replay CPU seam,
  modular-property fixtures and missing-backend-schema negative.
- native4/base/sparse snapshot/patch hashes unchanged; candidate2907 Python files
  has exactly4 approved differences;12 existing backend sources unchanged.
  Owned19/AST/whitespace PASS;1269/1582 gross. No native rebuild required.
- Stage adds backend schema_source_hashes only; D30 source_hashes remains13 with
  engine.core, native counters untouched. Graph status narrowed to
  PASS_THIN_GRAPH_AND_EAGER_MODULES after future execution; MRv2/backend/G3 excluded.
- completed:74d5c48 verified scoped code+evidence commit; code diff7 paths,
  267 added+23 deleted (290 repair gross), baseline gross increase252. Commit's
  retained report/review/receipt identify4 findings and native UNVERIFIED items.

## T2 input-only preparation iterations
- Root explicit authorized import: `git status --short` clean at6827507;
  `git merge --ff-only 9090407` succeeds. No owned edits lost; central/peer files
  only arrive through the authorized merge, not this stream's edits.
- Read accepted graph/helper, designc9e9274, retained root reference_lock and
  metadata, published source receipts and existing compatibility recipe.
- Config78447 bytes matches e535c5c9...; original source revision364b3b03...;
  unchanged config-only copy, no tokenizer/checkpoint download or metadata rewrite.
- Controlled cases SHA7383a897...: graph2×17 rows/live16,4; modules3 frames
  physical[6129,1]/[1,17]/[4,4], live[6128,0]/[0,16]/[1,0]. No observed trace claim.
- Static stdlib review PASS schema/position bounds/hash/cycle-conservation,
  existing probe_plan/expected_topology/validate_launcher bodies:16ranks/2nodes,
  layers3/88, timing_rows0. No torch/vllm imports or source/CPU suite rerun.
- proposal16H800/codesign/nonpreemptible,32CPU/128GiB per node,1800s/8GPUh,
  single finite unit/no automatic retry. Raw weight/payload calculations are
  lower bounds, not memory fit/actual kernel/collective evidence.
- Actual engine7168 is not supplied by existing T2 CLI or EngineArgs calls;
  input bound7168 recorded independently, original metadata1048576 remains.
  Root exact-engine requirement, sealing/readiness/resources/runtime remain open.
- `sha256sum --check hashes.sha256` PASS6/6; hash indexSHA256
  67f1a13307dc5e19353fea4b0f8a50fddc73f04ecb53a2fa533e5cc78a939a29.
  `git diff --check` PASS; selected script Git bytes at9090407 equal recorded
  hashes; current image reference equals accepted publication receipt. No CPU/
  source suite rerun, native import or actual runtime receipt.

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

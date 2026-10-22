# Native implementation progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Read approved allocation/design and begin interface milestone. |
| 2026-10-23 | Complete native19 integration, CPU418 PASS, D3048 PASS and static T2; retain root-only prerequisites. |
| 2026-10-23 | Close four reviewer findings;485 affected CPU PASS and1269/1582 gross. |
| 2026-10-23 | Import accepted9090407 and prepare controlled T2 metadata/inputs/resource proposal; static checks only. |
| 2026-10-23 | Complete root-directed two-file engine-limit correction, verified helper-body flow and85 gross delta; root integration pending. |

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
  commands; all6 artifacts in hashes.sha256 verify. Input-only commit aa872fa;
  no production/test delta against9090407.
- pending (root): helper integration, independent sealing/readiness and actual-image
  config check; resource/run approval, mounts/rendezvous/output and GPU evidence.
  The unsupported engine-limit interface gap is closed below, not runtime accepted.
- completed: D29-T2-ENGINE-LIMITS L1 actual engine7168/8192/128 triple/hash and L2
  required cases mapping before both constructors; final limits observed/checks
  are diagnostic-only, no configuration assignment afterward.
- completed: L3 PRICING delivers verified3f54bc6 optional keyword. Read-only actual
  helper body used with host dependencies; mixed commit/central records not imported
  and no peer file/builder copied. Root owns the remaining dependency integration.
- completed: source/test commit a5e5189, only2 authorized paths,85 incremental gross
  versus100. Final native19 baseline1350/1582 (+81 from1269),232 remaining.
- completed: L4 focused16 PASS/1.96s,affected175 PASS/15 SKIP/5.12s;AST2/whitespace
  ownership/budget PASS. Current/peer controlled probe_plan equal with real metadata;
  routing_input_file=None preserves this path without invented observed traces.
- completed: input/hash/provenance/proposal and English summary refresh after
  verified flow. cases.json adds only retained limits; graph/modules unchanged,
  full config bytes/native4 unchanged; root actual-image/GPU claims remain pending.
- completed: final static engine/source/hash/case-value/AST/native4 checks,
  bash-n2/checksums6/6 PASS in d29-t2-limits-static-final02.log. First ad-hoc
  snapshot check stripped vllm/ twice and failed FileNotFoundError; corrected
  inspected snapshot path only, no production/evidence bytes changed.
- completed: preserved13 old actual-source CPU controls PASS/1.97s with new
  entrypoint cases excluded, against published native vllm source. AST2 and
  whitespace PASS. Draft two-file incremental87 gross (17add/5del script,
  65add test), within100; final85 gross/green evidence supersedes that draft.

## 1. 规划完成情况
**汇总：** PASS4｜BLOCKED/FAILED/SKIPPED0｜TODO0（D29-T2-ENGINE-LIMITS；不是GPU验收）
| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| L1 actual serving triple | PASS | retained engine SHA7947f58b…；7168/8192/128，metadata字节未变 |
| L2两路径constructor flow | PASS | a5e5189；同mapping在creation前传入，final值diagnostic-only |
| L3 verified helper | PASS | PRICING3f54bc6真实body host flow PASS；mixed import留root |
| L4checks/inputs/records | PASS | affected175 PASS/15 SKIP，85gross；static/hash/records已更新，交接commit封存 |

## 2. 偏离检测
**结论：** 未偏移，正常推进。
**检测窗口：** D29-T2-ENGINE-LIMITS dispatch至a5e5189和records封存｜**当前前沿：** root dependency整合/seal（本流外）
| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | focused/affected现有2suites验证明确flow，未重跑418/485或host gap |
| D2 细节反复重试 | CLEAR | red定位旧helper遗漏；verified signature后green，不做参数/运行workaround |
| D3 支线问题修补 | CLEAR | limits、peer plan接口及hash均服务L1–L4，无其它producer修复 |
| D4 可并行任务停滞 | CLEAR | L1/L2在真实helper dependency前完成；no subagents，root持有整合与GPU |
| D5 超范围镀金 | CLEAR | 限定2files/85gross；无new CLI/builder/native source/tolerance |
| D6 分析空转 | CLEAR | a5e5189已交付；只剩用户要求input/records hash收尾 |
| D7 路线违背 | CLEAR | no platform/GPU/peer写入；不导入mixed central记录；保持controlled和soft handling |
**文档检查：** S1–S6均未发现未记录缺陷；历史input-only gap已明确被root授权L1–L4替代，真实helper整合仍由root持有。
**归因：** 无。

## 3. 纠正建议
**分支：** A：当前spec/plan合理，无需扩展scope或新增seam。
**需停止的偏离行为：** 无。
**接下去的路线：**
1. root独立helper整合/actual-image/seal → 完成标志：新helper和native source身份、两配置真实值及input hash。
2. root readiness后请求run approval → 完成标志：实际资源/路径/finite unit授权；本流不运行。

## 4. 最值得关注的 3 点
1. 两个final config值已在host flow观测；不等于实际image/GPU配置验收。
2. PRICING mixed commit未导入，root必须整合optional keyword；native19为1350/1582。
3. controlled inputs及16H800/30min8GPUh提案均非observed/launch，G3/tolerance未变。

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
- completed: input/records commit aa872fa. All14 changed paths are within the
  exclusive existing native task; no code/test or central/peer changes. Root
  must independently accept/seal before later resource/run authorization.
- Exact-engine7168 seam if required: existing EngineArgs constructors in
  frontier/profiling/collectives/step5_dispatch_probe.py:create_native_config:334–341
  and tests/integration/step5_environment/test_position_cyclic_graph.py:276–279.
  Neither receives max_model_len; T2 argparse:244–248 has no such input. These
  facts identify the dependency, not authorization for edits or a workaround.

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

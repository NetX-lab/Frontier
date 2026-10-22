# CPU timeout diagnosis progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Created isolated worktree and located read-only prediction seam. |
| 2026-10-22 | Completed deterministic FFN RED loop, CPU/collective/trace discrimination, and diagnosis-only proposal. |
| 2026-10-22 | Started parent-authorized D23-MRV2-CONTRACT-REUSE. |
| 2026-10-22 | Completed narrow fixed-binding implementation and bounded regression/latency verification; recorded unrelated test limitations. |
| 2026-10-22 | Began diagnostics-only D23-ACTUAL06-WALL on accepted7e8bea3 source. |
| 2026-10-22 | Completed exact public q1 RED loop, whole-path profile, source/ledger equivalence and diagnostics-only proposal. |

## Status

- completed: Created `feat/step5-mrv2-perf` from `ec5de15`; moved this session to the new worktree. Existing preview worktree untracked `.cortexkit/` and `.pi/` were preserved.
- completed: Read repository AGENTS, GLOSSARY, ADR 0001 and `diagnosing-bugs`. Located actual command/environment and cached attention root/count estimators. CUDA_EVENT exact lookup has 94392 keys per estimator (~71 MB artifact); KERNEL_ONLY has 184 keys (~200 KB).
- completed: Short real-cache attention loop `/data/ycfeng/tmp/d23_perf_attention02`: RED, source truth PASS, loading 1.3583s, profiled work 0.0995s. Individual actual attention query approximately 0.0167s versus conservative 0.007409s per-layer budget for 1024 decode steps under the unchanged 2400s bound.
- completed: Actual EP16 FFN replay, exact source truth assertions, two repeated non-profiled RED runs, CPU profile, cold/warm real collective discrimination, and trace metadata/persistence measurement.
- completed: Recorded measured cause, bounded proposal, original-bound latency gate, and limitations. No production files changed.
- pending at diagnosis close: Parent approval/implementation/review and full workload acceptance. The authorized implementation follow-up is recorded below; parent review/integration/full acceptance still remain outside this subtask.

## Evidence and decisions

- Original Python: 3.10.6; executable `/data/ycfeng/envs/frontier-py310/bin/python`.
- Worktree creation: `git worktree add -b feat/step5-mrv2-perf /data/ycfeng/Frontier/.worktrees/step5-mrv2-perf ec5de15`.
- Original input scale is retained in stored estimators, not minimized to a smaller coverage table. Loading pickles is authorized and avoids initialization/fitting. Inputs remain explicitly `cpu_fixture`, not native evidence.
- Systemd memory-control scopes are unavailable per delegation; all inspections are scoped, bounded and avoid dumping large identities.
- Initial harness failure: duration pickle does not contain its paired kernel-count estimator, because production attaches it after caching. Corrected harness by reading the existing bundle link and both trusted artifacts, verifying their SHA256, and attaching the link only in memory. No cache/fixture changes.
- Phase 3 ranked hypotheses: (1) repeated full model validation/metadata/precision construction per layer and EP lane; predicts high calls into `validate_step5_config`, `build_operator_manifest`, `resolve_module_precision`. (2) coverage/registry/hash rescans; predicts high lookup/fingerprint time, scaling with exact-key count. (3) collective subprocess work; predicts real repeated collective cost and subprocess calls. (4) proof/metrics serialization; predicts recorded output boundary dominates compute query.
- Baseline attention cProfile: `describe_step5_layer` 0.083s for six calls; `validate_step5_config` 25 calls/0.041s; `resolve_module_precision` 34 calls/0.038s; no estimator prediction or dataset hashing in measured work. Setup independently re-derives features and is reported separately from per-query latency.
- FFN harness minimization exposed older bundle selection: actual03 scalar profiles lacked the actual05 aggregate `num_tokens=2` rows. Checked bundle registrations and exact keys, then bound replay to the actual05 offline READ profile path; actual03 and actual05 share attention artifacts but not FFN artifacts. This is a harness setup correction, not a production behavior issue. The first FFN attempt also included non-feature `load_distribution`/`seed` fields; corrected projection to the existing `MOE_LOAD_IMBALANCE_FEATURES` registry, matching production. No training/cache modification occurred.
- Second checkpoint after FFN harness attempts: real feedback loop remains complete; repeated attempts narrowed two setup problems, not alternate production fixes. Stop broad bundle selection and use exact actual05 bound bundle. Remaining frontier is the same scoped FFN profile and communication/proof discrimination. No scope expansion.
- Final repeat baseline `/data/ycfeng/tmp/d23_perf_red_final01` and `final02`: RED / source truth PASS. 64 EP lane queries took 0.52744/0.54212s, cache loading 1.47872/1.51473s. Average actual query 8.22655ms, metadata 5.83032ms, operator queries 2.38895ms. The unchanged 2400s bound after 1004s setup implies at most 0.96824ms per EP lane/layer over 1024 steps and 88 routed layers, even giving FFN the entire remaining budget.
- `d23_perf_ffn06/prediction.prof`: 32 `describe_step5_layer` calls took 0.45234s (73.3% of 0.61686s measured work); 192 precision resolutions, 97 full Step validations, 228 layer-contract resolutions. Large exact-key lookup and source hashes were not dominant. Fixture SHA256 runs only during setup.
- Hypothesis discrimination: actual ATTN_TP8 allreduce (8192 bytes) cold 1.88543s / hit 0.10995ms, actual EP16 all-to-all (16384 bytes) cold 2.54529s / hit 0.17243ms; predictions exactly repeat. Correct real collective replay is `d23_perf_cc03`, not the earlier EP allreduce exploratory run. No full CLI, cloning, build, original runner/cache writes or native admission.
- Actual source-bound metadata and enabled TraceStore persistence: 228 real operator events, 172466 bytes, 0.04050s unprofiled; persisted duration/name/layer tuple truth PASS. Profile identifies dataclass `asdict`/deepcopy as the largest trace-only cost, but trace work is well below the measured FFN query work. Full MetricsStore EP scheduling and final acceptance were not replayed.
- Attention-only unprofiled query is 6.59–7.36ms and standalone budget is 7.409ms: GREEN. Earlier standalone attention RED was profiler overhead; only FFN non-profiled RED is the authoritative fast feedback signal.
- Minimal proposal: predictor-bound cached metadata for each physical layer plus cached immutable selected operator contract/model binding per `(layer, operator, measurement family)`. Preserve feature validation, exact coverage, paired count queries, source proof identity, all EP lanes, physical layer/precision differences and enabled metrics. Metadata-only savings are measured at approximately 8.21s per modeled full round; output remains unmodified. Source approval belongs to parent.
- Final checkpoint: all four scoped diagnosis steps completed; D1–D7 CLEAR, S1–S6 no detected defects. Stop extending the harness now; return measured evidence/proposal. Parent's production/final-workload tasks intentionally remain pending.
- Final current-script verification: `d23_perf_red_final03` RED/source PASS, complete process3.798875s, max RSS815716KiB, loading1.483458s, measured0.543830s. Python syntax PASS for both scripts. Re-read requirements before close. `/usr/bin/time` is absent; used stdlib timing/resource wrapper without installing anything.

## D23-MRV2-CONTRACT-REUSE status

- completed: Inspected supported lifetime. `BaseExecutionTimePredictor.__init__` binds model/replica config once and validates Step role/PP/TP/EP groups before normal execution. Production execution methods do not reassign the model or parallel sizes. Profiling-only quantization replacement happens before predictor construction. Each predictor has its own state; model manager trains/loads and publishes family registries during construction. Keep per-query registry selection even if a registered model is replaced in memory.
- completed: Added fitted-estimator regression before the fix at the parent-authorized `_step5_measured_layer`, `_predict_step5_ffn_operator` and `step5_batch_features` seams. Initial RED: 12 layer descriptions, 72 precision resolutions, 30 contract resolutions for six real operator queries. All duration/count/precision/layer feature assertions passed before the work-count failure. The final warm-phase regression asserts zero repeated descriptions/precision/contract resolution and retains four family/model selections for four real queries.
- completed: Implemented shared predictor-instance metadata by physical layer and fixed `(layer, operator)` precision/serialized contract identity in existing `step5_cost.py`. The former repeated sklearn_moe binding code is replaced by delegates. No estimator/dynamic feature/output cache or initializer changes; family/model selection remains per query. Existing structural dummy metadata is untouched.
- completed: Seven focused regressions pass: changing token features/errors, current registry replacement/removal, missing/invalid paired counts, nested same/different-layer restoration after exception, isolated predictor instances with different attention TP, FP8/BF16 layers, distinct actual fitted CUDA_EVENT/KERNEL_ONLY model/proof hashes, and missing-family rejection.
- completed: Unchanged actual05 cache microbench takes 0.073477s for 64 lanes/456 operator queries versus parent baseline 0.532956s (7.253x); source truth PASS, all cost dictionaries and loaded artifact descriptors exactly equal. Gate remains RED because cold fixed binding and the slowest warm query exceed the unchanged 0.96824ms/query budget. No benchmark edits.
- completed: Enabled trace writes 456 events/344788 bytes in 0.072227s, persisted truth PASS. Profile shows 2 descriptions, 12 precision resolutions, 18 contract resolutions, 7 full validations, but 456 model selections and 912 dynamic duration/count lookups remain.
- completed: Existing measured/FFN/on-demand checks 91 PASS; public/root/metrics 44 PASS; MRv2 forward 75 PASS; structural 38 PASS / 2 FAIL before compute on a stale SimpleNamespace scheduler missing `take_forward_start_batch`. This failure occurs in the unchanged event module and an unchanged test stub. Unsupported role/parallel layout tests passed.
- skipped: Further optional PD suite retries. The first combined forward/structural/PD run reached the 60s bound; split MRv2 forward completed. Standalone PD and a reduced PD selection also hit 60s without a complete verdict. These scheduler/transfer tests do not exercise the fixed-binding path; no timeout increase, full CLI retry or parent test/source edits. Parent owns reverify/full acceptance.
- completed: Source/tests/module boundary committed as `ee28bf7ea43d3c7c73b7176eeb10b6f07d51492a`. Final focused regression `/data/ycfeng/tmp/d23_contract_green_final01.log`:7 PASS9.15s. Committed-source unchanged trace command `/data/ycfeng/tmp/d23_contract_committed01`:64 lanes/456 operators0.074077174s,loading1.490313328s,7.195x speedup,exact cost dictionaries/artifact descriptors equal,sourceTruthPASS,RED. Trace456 events/344787bytes0.068545761s,persisted truthPASS. Original diagnostics unchanged.
- completed: Final records and exact committed-source receipts prepared for the records-only commit; parent review/integration/full acceptance remains outside this subtask.
- Scope decision: Reuse fixed bindings only in each predictor lifetime; no cross-instance/global cache, model/result cache, configuration fingerprint/invalidation path or initializer change. Dummy timing's pre-existing metadata dictionary is distinct from the measured active-layer context and will not be rewritten in this task.

## Follow-up checkpoint

- Mainline: completed lifetime inspection -> regression RED -> fixed-binding correction -> regression GREEN; current frontier is final report/module boundary and commit. Parallel unchanged benchmark/profile and relevant existing test groups advanced.
- D1 CLEAR: the two original diagnostic scripts remain unmodified; stop extra PD reruns now that required cost/forward tests are verifiable. D2 CLEAR for implementation: no repeated source-fix attempts; PD bounded reruns changed test selection rather than production behavior and are explicitly stopped. D3 CLEAR: no unrelated defect was fixed. D4 CLEAR: independent verification ran concurrently. D5 CLEAR: no model/result cache, invalidation, mode flag, fake costs or broad refactor. D6 CLEAR: measured result produced a minimal implementation. D7 CLEAR: explicit parent authorization owns the two cost modules and preserves dynamic selection.
- S1–S4 and S6: no recorded defect. S5: earlier diagnosis-only summary and old pending follow-up status needed an authorized-follow-up section; update these records before close, preserving historical diagnosis evidence. This is a status correction, not an agreed-scope change.
- Route: finish report and module boundary -> scoped source/tests commit -> source-sensitive unchanged command on committed code -> records receipt commit -> return parent. Parent owns review/reverification/fullCLI/strict77. No pending latency gate verdict authorizes extra optimization.

## Final checkpoint

### 1. 规划完成情况

**汇总：** PASS4｜BLOCKED/FAILED/SKIPPED0｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Lifecycle inspection | PASS | Supported immutable predictor bindings inspected. |
| Real repeated-query RED | PASS | Fixed-work regression failed before source correction. |
| Fixed metadata/contract correction | PASS | Source/tests commit `ee28bf7`; no model/output cache. |
| Bounded relevant verification | PASS | New7 and existing210 tests pass; unrelated38/2 structural result and optionalPD timeouts explicitly recorded. |
| Record/return receipts | TODO | Final record commit remains; parent acceptance excluded. |

### 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** Verified source deliverable to committed-source receipt｜**当前前沿：** record commit and return

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | Diagnostic scripts unchanged; optional test extension stopped. |
| D2 细节反复重试 | CLEAR | No repeated source fixes; fixed-work regression GREEN. |
| D3 支线问题修补 | CLEAR | Stale scheduler stub recorded, not edited. |
| D4 可并行任务停滞 | CLEAR | Independent test groups and measurements ran concurrently. |
| D5 超范围镀金 | CLEAR | Only authorized fixed bindings retained. |
| D6 分析空转 | CLEAR | Source deliverable committed with measured gain. |
| D7 路线违背 | CLEAR | Scope and original bound retained. |

**文档检查：** S1–S6均未发现未解决缺陷；earlier status correction applied.
**归因：** 无

### 3. 纠正建议

**分支：** A：spec/plan合理，无需改动
**需停止的偏离行为：** 无
**接下去的路线：** Commit exact receipts -> return source/record commits; done when parent receives reproducible commands and explicit limitations. No extra latency optimization or fullCLI run.

### 4. 最值得关注的3点

1. Fixed repeated work eliminated, while dynamic selection/proof lookup remains.
2. Unchanged max-query gate remains RED despite7.195x measured improvement.
3. Parent owns structural stub issue, optionalPD reverify, strict77 and full acceptance.

## Return state

Scope implementation and verification are complete; source commit `ee28bf7` is ready for parent integration. The records-only commit follows it and preserves all baseline and follow-up evidence. No further source change or test retry is planned. Parent-owned open work: review/reverification, unrelated stale structural stub, optionalPD incomplete tests, initialization attribution and original1024 offline/online/strict77 acceptance. The unchanged latency verdict remains RED and is not represented as full performance acceptance.

## D23-ACTUAL06-WALL status

- completed: Worktree clean; `git merge --no-edit 7e8bea3` fast-forwarded to accepted source. Read diagnosing-bugs,GLOSSARY,ADR0001 and relevant instructions. Parent original receipt inspected only for delegated cadence/budget, no duplicate census.
- completed: Constructed complete original Simulator event-prefix diagnosis using trusted role bundles, real collective backend and original metrics/ledger/JSON trace. Writable cache APIs confined to copies/scratch; no initializer or runtime source modifications. Original op-level trace disabled in actual06 CLI; additional enabled trace measurement is a separate diagnostic configuration, not a silent original-setting claim.
- completed: Exact completion-boundary two-q1 RED, ranked hypotheses, whole-path profile, repeated source/ledger equivalence, real cold collective and warm q1 discrimination. Final records/syntax/diff verification complete; diagnostic/record commit prepared. FullCLI6144/1024 and source approval remain parent-owned.

### Feedback-loop checkpoint

- PASS: accepted source/read-only bundle setup and actual P6144 public events; original P completion simulation85.512575168 matches. TODO: complete tail/q1 RED and profiling.
- D2 HIT (execution): snapshot serialization survived three unrecorded workaround retries. Stop parameter changes; scoped component serialization identified only KV manager as the blocker. Its actual103675-block free queue is a doubly linked list, so generic pickle follows103k nested links. Store exact block identities/scalars and link IDs, reconstruct unchanged links; do not reduce pool size or alter runtime methods.
- D1,D3–D7 CLEAR; S1–S6 no scope/edge defect found. BranchA: keep spec/criteria, change only diagnostic serialization. Next: verified event-boundary snapshot -> completed original q1 loop RED -> ranked hypotheses/whole-path profile. No production theory or fix before RED.

### Measured feedback and discrimination

- First complete RED (`d23_actual06_red01`): two q1 intervals9.74065/9.40578s, original grouped-cache start/end event/role/iteration/simulation-time/token tuples equal. Hypotheses were disclosed before cProfile: (1) public attention/FFN or complete-stage rebuilding, (2) P dummy/EP coordination, (3) metrics/ledger/trace, (4) cold backend subprocess, (5) host load or state restoration mismatch. Initial mismatch with parent median was investigated against only the first36 scheduler starts, not a duplicate census: actual06 early q1 also9.1–9.9s; iteration24 onward about5.1s.
- State-fidelity correction: initial diagnostic snapshots omitted class ID counters and used old object IDs for ledger maps. Early measurements were source-cost/simulation-equivalent but not full entity/ledger-ID acceptance. Final setup was rebuilt from unmodified original event handlers, recording exact counters and pending object references, at the first complete D GlobalBatchEnd boundary. All earlier heuristic ID recovery was removed from the shipped script. The snapshot retains original request6144/1024, full103675-block pool per lane, real P/D state, and real backend cache; new output/cache paths alone are rebound to scratch.
- Final feedback (`d23_actual06_verified_red01`): setup13.26084s; two complete q1 intervals9.54025/9.67630s; full event loop19.22282s,1168 events. D293 events/iteration4.41871/4.87679s, P291 events/iteration5.12154/4.79950s. RED versus optimistic `(2400-339.5054249763489)/1024=2.012201733s`. This covers the complete early q1 path, not just FFN. The D component closely explains the original later5.1s cadence; exact later iteration24 state was not replayed and this remains an inference.
- Source audit (`d23_actual06_verified_audit01`): all41288 FFN and960 attention queries independently equal source artifact duration/count/model/evidence/exact-feature values; all92 layers, BF16/FP32/FP8,34 model hashes retained. Current q1 selects CUDA_EVENT for every audited operator. Audit JSON dictionary equals both earlier exact-state audits, SHA256 `c2dfc11132106d5ac0cdb3eef8b29f08dd54b490f311b47d009751b86aed648a`. Audit overhead is separate:24.45003s loop, not baseline latency.
- Dynamic population/metrics:352 waves×EP16=5632 lane ledger rows; baseline/audit ledger hash `bdd74d1e1ec310fa94f8ad60af888d61bbc0e4279ba92349cdaed3fd33cb2526` identical. Original JSON event trace bytes and two complete stage-ledger rows equal between baseline/audit. write_metrics, request/batch/utilization metrics, stage ledger, metrics-ground-truth and JSON event trace retained; actual06 store_operation_metrics=false and op-level trace=false unchanged.
- Warm backend:6016 PREFILL+6208 DECODE requests, all hits,0 misses. Cold real ATTN_TP8 allreduce1.88352s/hit0.11110ms; EP16 all-to-all2.48633s/hit0.16237ms; repeated costs exactly equal (`d23_actual06_collective_check`). Separate warm-backend full prefix: public P-to-transfer7.89735s, P+tail60.78812s across three <=60s process segments; all simulation boundaries exact. It is not cold-P/tail time or original init verification.
- Whole-path cProfile (`d23_actual06_complete_profile01`):42885839 function calls in34.100s, same1168 events/two complete q1 outputs. EP-wave preparation352 calls27.820s; public stage6410 calls24.054s; FFN41288 calls9.896s; on-demand duration/count84496 calls5.933s; identity JSON42248 calls3.313s; lane builders5632 calls3.754s with123904 full Request constructions2.635s; original EP metrics0.653s and stage metrics1.007s. Inclusive times overlap and must not be added. Static previous metadata/precision construction does not recur. Attention workloads still trigger998 full Step validations1.735s; not changed here.
- One-variable wall identity parser observation (`d23_actual06_final_red01`):42248 canonical marker parses3.13526s in19.14586s loop; per-role public stage D6.85726/P6.98384s, FFN D2.90862/P2.85948s, attention D1.01070/P0.69774s. No return value substituted; full source audit repeat still identical. Fixed proof parsing matters but is only~16% total loop.
- Separate enabled op trace (`d23_actual06_final_trace01`):31.61076s loop versus19.14586s baseline, same1168 events/5632 lanes and0 backend misses.51988 persisted events/76067468 bytes, all92 layers/P+D roles, finite nonnegative durations. It is deliberately an additional setting, not actual06 original configuration or a main acceptance artifact.
- Smallest supported proposal (not implemented): reuse the parsed canonical estimator measurement identity/evidence_kind at its existing attach/load boundary; retain current per-query model registry selection, canonical duration/count identity comparison, physical target/kind/runtime checks and every dynamic query/count/proof. Registry canonical identity is attached once and rejects conflicts; a changed marker must cause reparse/error, and registry replacement must use the replacement estimator's identity. Prefer a private attached immutable view, not a second model/result cache. Measured upper saving is~3.135s/two early P+D rounds (~0.8s/D-only round by query population), insufficient alone for2.012s budget. Dynamic schema/bounds parsing, lane Request construction, public layer assembly and attention config validation remain measured costs requiring separate parent decisions.
- Final syntax AST PASS and `git diff --check` PASS. No production source file, old diagnostic script, immutable fixture/cache or other worktree edited. Initialization337/339s, exact later-context q1 state, online full replay, final export/fullCLI/native accuracy/strict77 remain uncovered here.

### Whole-path final checkpoint

## 1. 规划完成情况

**汇总：** PASS4｜BLOCKED/FAILED/SKIPPED0｜TODO0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Accepted-source and scratch bundle setup | PASS | Clean fast-forward7e8bea3;68 artifacts/4 bundles SHA-validated. |
| Complete public iteration RED | PASS | Exact completion state; two full q1 rounds9.54/9.68s. |
| Profile and source/metrics/cold discrimination | PASS | Source/proof/event/ledger equality; full-path profile and trace receipts. |
| Diagnostic records and handoff | PASS | Exact command/report/evidence prepared and requirements re-read; commit follows. |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** Initial complete RED to final verified receipts｜**当前前沿：** diagnostic records and return

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | Snapshot corrections settle required identity/ledger fidelity; stop extension now. |
| D2 细节反复重试 | CLEAR | Earlier serialization drift was corrected; current exact-state measurements advance equivalence/profile. |
| D3 支线问题修补 | CLEAR | No unrelated production defect edited. |
| D4 可并行任务停滞 | CLEAR | Audit/baseline and profile/trace ran independently in parallel. |
| D5 超范围镀金 | CLEAR | Only delegated diagnostic options/setup; no production cache or behavior. |
| D6 分析空转 | CLEAR | Whole-path analysis is the requested deliverable; measured proposal ready. |
| D7 路线违背 | CLEAR | Original bound, topology, population and source costs preserved. |

**文档检查：** S5 progress was stale pending; updated above from actual receipts. S1–S4/S6 no defect.
**归因：** 无

## 3. 纠正建议

**分支：** B：stale execution status corrected directly; no agreed-scope change.

**文档更新（先执行）：**
1. `progress.md` D23-ACTUAL06-WALL status: record complete RED/profile/equivalence and explicit limitations（可直接执行）.

**需停止的偏离行为：** 无

**接下去的路线：**
1. Finish exact report and scoped diagnostic commit -> parent receives reusable command and supported proposal.
2. Stop diagnostic extension/fullCLI; parent owns next production decision and final acceptance.

## 4. 最值得关注的3点

1. Complete D q1 workload costs4.4–4.9s even without concurrent P work; original budget2.012s is still missed.
2. Proof parsing is measured but cannot alone close the gap; dynamic validation and lane/entity work remain.
3. No production optimization/fullCLI/native acceptance was performed in this phase.

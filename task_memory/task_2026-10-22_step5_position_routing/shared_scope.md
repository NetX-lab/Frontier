# D29 shared-hunk scope reconciliation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Parent accepts sole writers/H11 deduction, incorporates reviewed producer closure and reconciles107 paths/9942 gross lines. |
| 2026-10-22 | Reconcile all17 cross-owner shared paths and3 same-owner Common/Model increments; assign one file writer, remove one wholly duplicate allowance, retain conservative bounds for inseparable overlap. Design only. |

TaskID: **D29-SHARED-HUNK-UNION**. This document is the only new write.
It recommends hunk ownership and accounting; it does not edit, replace or approve
the four owner designs, any implementation or the finite numerical model.

## 1. Current parent scope and decision request

Parent has accepted the sole-writer allocation and the H11 whole-row deduction.
The final producer review D29-FINITE-PRODUCER-DELTA has No findings. The numerical
model and implementation scope still require YC's decision. Authoritative
per-path allowances are retained in `scope.json`; this is a conservative ceiling,
not a measured implementation diff or a claim that every mixed allowance is disjoint.

Complete pricing is60 paths/4549, including14 closure increments/737 lines.
The union is **107 code/test/config paths,9942 gross lines**. Five paths are
inside the existing collective-sim submodule; the four native source files are
counted in the native19 scope. Additionally allow the existing gitlink update
(two diff lines) and `docs/general/module-size-exceptions.md` (at most12 added
lines, already specified by CORE). The requested total ceiling is **10000**
added/deleted lines for these107 paths plus those two explicitly named updates.
Task records and generated evidence/measurement/image artifacts are recorded
separately and are not additional handwritten implementation paths.

The model decision is the fixed proposal in pricing3.2: ordered DP2 tensor/root
stages, configured element striping, all eight TP rails in one HTSIM network
run, measured copy/add proxies, and an affine effective remainder fitted after
fixing network/local inputs. Compose network+slowest-rank-local+remainder with
no overlap. This is a modeling assumption, not isolated protocol measurement;
retain signed residuals, independent qualification and each E2E error<=10%.
Existing numerical inputs must be frozen/qualified; no invented coefficients.
The finite collection shape is specified, but its platform resources/run approval
are decided at package readiness. Existing D24/D28 resource limits are unchanged.

Parent computation reads all Common/Model/Closure rows, retains the other three
reviewed owner ledgers, removes only H11's34, and checks exact path identities.
The five globally new paths are the actual dispatch probe, worker, validator,
and their existing transport/probe unit-test modules. CORE inputs.py was already
in the union. CORE writes that shared materializer once; its8-line allowance is
retained conservatively with the existing130-line mixed input scope. H09 gains
the8-line adapter connector. Neither creates a second tensor/count algorithm.

### Quantitative ledger

Evidence: `/data/ycfeng/tmp/d29_complete_scope_union_candidate01.json`, compared
with `design.md` sections5/6.2/6.3/9, `profile_source_design.md` sections4/5 and
`pricing_design.md` sections4/5/6.Common/6.Model. The pricing snapshot matches
the candidate's recorded source hash. Native19 is taken from the parent's
reviewed candidate rows; its native-image paths have no cross-owner overlap.
No native architecture or D30 inspection was needed.

| Quantity | Recorded / checked value |
| --- | ---: |
| Owner submissions | CORE33/2966; SOURCE16/879; NATIVE19/1582; PRICING60/4549 |
| Candidate allowance rows / distinct source-test-config paths | 139 /107 |
| Cross-owner shared paths / gross allowance after deduction | 18 /2976 |
| Same-owner pricing shared paths / gross allowance | 8 /1007 |
| Other paths / gross allowance | 81 /5959 |
| Candidate gross ceiling before behavioral reconciliation | 9976 |
| Wholly duplicate allowance withdrawn below (H11, pricing) | +30/-4 =34 |
| Accepted parent planning ceiling, awaiting YC | **107 paths /9942 gross lines** |

**9942 is a defensible planning upper bound, not a nonoverlapping implementation
diff or a proved minimum.** Each required behavior below has one implementation
owner; remaining mixed allowances may cover the same changed lines. The table
retains their sums where the source tables do not separate those portions. Do
not deduct guessed percentages, take the larger owner's allowance, or call
remaining overlaps independent numerical work. Actual approved diff accounting
may reduce the bound; needing more than it requires a scope correction.

The only numerical deduction recommended now is H11's entire pricing consumer
row, whose complete behavior is already expressly included in CORE B4. No path
is deleted. S16's transfer to the pricing file writer is **not** a deduction;
its23-line provenance allowance remains required once. No retired CORE expert
adapter or job.py allowance is restored or subtracted again.

This snapshot excludes the submodule gitlink pointer, module-size documentation,
task records and generated evidence. Parent must list those separately in its
scope request; this reconciliation does not invent their ceilings or hide them
in9942. This `shared_scope.md` is a task record, not an implementation path.

## 2. Cross-owner ownership matrix

Paths are repository-relative. C=CORE, P=PRICING, S=SOURCE. “Writer” is the one
agent responsible for editing the whole file during an approved step; contributing
owners supply their already scoped behavior/patch specification and accept it.
This is not permission for concurrent edits to the same file.

Status: **D** = required responsibilities are distinguishable; keep additive
allowances. **I** = identifiable duplicate sub-behavior in a mixed, inseparable
allowance; implement it once, retain the sum as an upper bound. **W** = whole
duplicate row withdrawn, with all required behavior retained by the named owner.
Combined +/- columns are allowance bounds, not predicted actual hunks.

| ID / exact path | Writer | Nonduplicated required responsibilities and owner | Allowances; conservative merged + / - =gross | Status |
| --- | --- | --- | --- | --- |
| H01 `frontier/moe_routing_policy.py` | C | C: count registry/callable, interval math, immutable lane/source shape and strict source checks. P: resolve/freeze actual source gating predicate and validate admitted DP/TP-to-EP association; C includes that field in identity, not a second predicate/mapper. | C200 + P12; **+212/-0 =212** | D |
| H02 `frontier/execution_time_predictor/moe_routing_workload.py` | C | C: supplied-source threading, entire lane-aware descriptor cache, typed model/schema selection and classifier integration. P: construct the selected native padding/alignment/capacity features using gathered M and live counts. One selected feature-construction branch calls the one classifier. | C112 + P33; **+115/-30 =145** | I |
| H03 `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | C | C: constructor519-525 count-registry admission before parent init; applicable exact duration/count dispatch and in-bound-miss rejection. P: source-local shared/routed TP costs, prepare/root phase composition and plan/proof return. Applicable prepare membership/dispatch is installed by C once, not added again by P. | C105 + P83; **+151/-37 =188** | I |
| H04 `frontier/execution_time_predictor/moe_operator_times.py` | P | P: actual source widths/context, gathered expert numeric queries and plan-backed DP dispatch/combine. C: position-policy removal of zero-live shortcut and requirement to route applicable physical scopes through paired exact lookup. P integrates that C rule into its numeric calls; neither owner adds a separate shuffle charge. | C45 + P90; **+100/-35 =135** | I |
| H05 `frontier/execution_time_predictor/moe_dataset_training.py` | C | C: selected-spec enumeration/metadata, preflight/pre-loader operator_facts before legacy helpers, frame/precision/context cache, schema selection/ranges and duration/count pairing. P: actual native prepare/capacity feature definitions supplied to that one selector/fit. C owns all row selection/paired-fit plumbing; no second prepare trainer or duplicate pairing. | C135 + P29; **+134/-30 =164** | I |
| H06 `frontier/execution_time_predictor/prediction_family_trainers.py` | C | C: bound TP/EP/typed precision/context before loader/helpers, actual contract/model/cache identities, selected schemas/ranges and paired fit. P: native numerical schema/target facts consumed by that existing fit. C performs the fit integration once; P does not repeat selected-spec discovery or pair creation. | C129 + P29; **+126/-32 =158** | I |
| H07 `frontier/moe_load_imbalance.py` | C | C: one exact shared schema classifier recognizing preserved legacy and new schemas. P: selected native backend capacity/alignment/schema entries and their meaning. Native entries extend C's classifier; no second classifier or universal Triton padding branch. | C26 + P30; **+50/-6 =56** | I |
| H08 `frontier/execution_time_predictor/step5.py` | C | C: scoped rows, typed duration/count/source/native identity propagation and applicable-target admission. P: native branch/plan capability prerequisites and selected physical-scope acceptance. C installs the common target/scope checks once; P supplies the additional plan/native prerequisite conditions. | C58 + P23; **+70/-11 =81** | I |
| H09 `frontier/profiling/moe/moe_impl.py` | C | C: native router/context and semantic validation; adapter calls the single inputs.py materializer. S14: actual callable provenance outside timers. P: reached source prepare timer and the materializer connector; no duplicate tensor/count algorithm. | C113 + S18 + P41; **+142/-30 =172** | I |
| H10 `frontier/profiling/moe/moe_wrapper.py` | C | C: one cell/one dict, local/gathered widths/context, selected-spec row targets, sentinel counting, singleton RECORD_FUNCTION replay/restore/clear and immutable CUDA_EVENT stats. P: actual prepare/gathered-sample/native inclusive-root contract consumed through its adapter and returned scope/layout identity. C owns cell dispatch/target enumeration once; P owns native scope semantics, not a second wrapper/row protocol. | C221 + P63; **+239/-45 =284** | I |
| H11 `frontier/profiling/moe/main.py` | C | C B4: census/cell work, three executor kwargs, one-dict flattening and registry-selected precision/reached-prepare targets, exact required columns/typed metadata/no structural zeros. S8: manifest/source verification and actual per-result evidence attachment. P's whole30/4 row restates C's selected-target/metadata admission; P reviews its spec facts but adds no second consumer hunk. | C145 + S26; P34 withdrawn; **+140/-31 =171** | W |
| H12 `tests/unit/test_step5_moe_profiling.py` | C | C C4: native router/context/mask integration and real DeviceTimer annotations/tracer parser/count replay/event preservation/error cleanup. S12: actual gate/norm/oracle/kernel callable wrong-root/spec/hash/missing-receipt negatives. P: original prepare/inclusive invocation/timer partition and selected padding/scales. Share fixtures, not another routing/count/provenance suite. | C175 + S50 + P65; **+290/-0 =290** | D |
| H13 `tests/unit/test_moe_grouped_gemm_padded_features.py` | C | C C5: actual two pre-loader/preflight/loader paths, old-helper traps, TP/EP/precision/context preservation, schema/ranges/cache separation. P: actual TRITON-vs-DEEPGEMM capacity/alignment/empty-lane numeric assertions. Test one production classifier; parser/cache coverage is C, backend formula correctness P. | C140 + P40; **+180/-0 =180** | D |
| H14 `tests/integration/step5_measured_compute_fixture.py` | C | C: canonical source/count producer, all source/EP/empty rows and one selected scoped target schema, fresh CPU fixture generation/identity. P: selected native numerical scope/layout and finite-variant plan/model CPU evidence consumed by main. C emits common targets/rows once; P supplies variant/numerical payload, never native qualification from synthetic values. | C170 + P60; **+190/-40 =230** | I |
| H15 `tests/integration/run_step5_measured_main.py` | C | C: manifest policy/input/schema forwarding and original offline/online run/timeout/capacity/READ acceptance. P: finite-variant config/input binding, real operation metrics/trace/ledger enabled and numerical plan/scope proofs. One invocation/manifest validator, with P's additional flags/checks; no second runner. | C35 + P56; **+75/-16 =91** | I |
| H16 `tests/integration/test_step5_measured_main.py` | C | C: paired actual-main source/position/empty exact coverage and retained workflow invariants. P: native-sized numerical plan/phase/metrics/trace assertions for that same run. Shared run fixture/assertions executed once; keep P's independently required reporting assertions. | C98 + P55; **+145/-8 =153** | I |
| H17 `frontier/profiling/moe/fp8_block_experts.py` | P | P: original inclusive expert invocation on prepared gathered tensors and selected backend/layout/config/callable identity. S16 specification: actual config/oracle/construction/returned kernel bound-callable receipt checks using S5. P writes that provenance hunk here once; S owns verifier/interface acceptance, not native invocation or backend choice. | S23 + P105; **+112/-16 =128** | D |
| H18 `frontier/profiling/moe/inputs.py` | C | C: input schema, source association, frame parsing and tensor materialization. P: expose that one materializer for the existing probe; no second interval/count implementation. | C130 + P8; **+138/-0 =138** | I |

Matrix sum: **18 paths, +2609/-367 =2976 gross**. There are4 D,13 I and1 W
rows. “D” separates acceptance responsibilities, not a promise that imports/test
setup lines never overlap. No deduction is taken for shared imports or fixtures.
I-row sums remain conservative because their allowance rows do not isolate the
shared calls/branches, imports, deleted statements and added numerical fields.

### Whole versus partial duplicate allowances

- **H11 whole duplicate, subtract34:** pricing6.Common's `main.py` row at869 is solely
  precision/reached-prepare CSV targets/metadata and no positive absent scope.
  CORE B4 at1394 and6.2.1 at838-851 explicitly own exactly that policy/contract-aware validator,
  selected column union, typed metadata and executor/CSV output. The pricing
  numerical metadata producer remains in native adapters/descriptors; S8 source
  attachment remains separately counted. Nothing in this row identifies an
  additional numerical producer or collector. Assign the whole consumer to C.
- **Partial duplicate, assign once without estimating savings:** H02/H07's
  common classifier integration; H03's applicable prepare dispatch; H04's no-zero/
  no-shuffle call routing; H05/H06's feature selection and paired-fit plumbing;
  H08's common target/scope admission; H09's actual callable source-verifier
  invocation; H10's selected-spec frame/row dispatch; H14's selected row emission;
  H15/H16's common manifest/run/check plumbing. Each also has additional native,
  numerical, provenance or workflow work. The owners' rows give no separable
  +/- sub-ceiling for the repeated behavior, so keep the sums shown, not a
  speculative reduction. H10's63 and H14's60 are specifically **not** treated
  as wholly redundant: their scope/variant language includes native numerical
  sample or finite-variant evidence responsibilities beyond generic row emission.
- **No behavioral duplication assumed:** H01's source-field storage vs actual
  predicate/placement checks; H12/H13's integration vs provenance/numeric
  assertions; H17's native invocation vs origin proof. Creating the same tensor,
  model field or test fixture twice is still prohibited; shared setup has no
  isolated allowance to subtract at design time.

## 3. Same-owner Common/Model additions (not cross-owner overlap)

All three remain single-writer **P**. Model rows are explicit additional work,
not complete replacements for Common; count their amounts once, not their
combined explanatory note a second time.

| Exact path | Common responsibility | Additional Model responsibility | Merged + / - =gross |
| --- | --- | --- | ---: |
| `frontier/profiling/collectives/vllm_collective_benchmark.py` | +60/-14: native ordinary preallocated into-output branch, alias/registration/no-copy/per-rank scope receipt | +140/-0: separately scoped local copy/add producer, queued timing/graph/per-rank local evidence | **+200/-14 =214** |
| `tests/unit/test_step5_comm_cost_export.py` | +40/-0: candidate/native scope cannot be admitted by old labels | +65/-0: bounded affine/TP fits, constraints/residuals/held-out errors/range rejection | **+105/-0 =105** |
| `tests/unit/test_collective_benchmark_helpers.py` | +60/-0: real into-output/alias/no-staging and per-rank count normalization | +55/-0: local-work construction/config/rows/preallocation and local-vs-native evidence | **+115/-0 =115** |

The original three rows above retain the260-line Model increment once. Closure
adds18 to benchmark and15 to benchmark helpers. Five more pricing-only paths
have explicit additive closure increments: forward_collectives34 combined,
prepare81, definition78, exporter247 and package tests100. The final same-owner
shared group has **8 paths,+981/-26=1007 gross**. Pricing42+15+14, after repeated
paths, gives60 unique paths; no same-owner increment is silently dropped.

## 4. Parent integration recommendation and remaining decisions

1. **Accept or correct this ownership recommendation, not the model by proxy.**
   Apply H11's34-line withdrawal in the parent's aggregate ledger only; the
   four owner documents remain untouched by this task. Current adjustment is
    C2966 + S879 + N1582 + (P4549 -34) =**9942**. Cross/shared2976 + same/shared1007
    + other5959 =9942. Keep107 paths. The16-source origin ledger still includes
   S16; its assigned implementation writer is P, not an extra pricing allowance.
2. **Freeze sole writer for each shared file before an implementation step.**
    C writes16 matrix paths, P writes H04/H17 and the8 pricing-only shared paths.
   Source supplies S8/S12/S14/S16 hunks/interfaces to the named writer and accepts
   provenance behavior. P supplies numeric/native schema/plan facts to C-writer
   paths. Neither contributors nor reviewer simultaneously edit those paths.
   Parent may be the integration executor without altering responsibility owners.
3. **Sequence approved work by real interfaces.** First N provides the router/
   context interface and S provides the existing manifest verifier/receipt format;
   C and P establish the single count descriptor and P's
   `get_moe_profiling_operators(context)`/native precision-layout facts. Then the
   named writers integrate adapters/provenance and per-cell rows/replay, runtime
   queries and pre-loader/exact paired admission. Finish one fixture/runner/report
   integration, not independent duplicate fixtures followed by reconciliation.
   Independent nonshared slices can proceed in parallel after their own approval.
   D30 is not a dependency and must not wait for this task; it was not inspected.
4. **Account actual hunks once during implementation.** Each writer annotates its
   changed responsibilities against H01-H17 and the3 model increments in parent
   progress. Compare the complete added/deleted diff with the merged allowance;
   do not count the same deletion under two contributors. Preserve all closed
   constructor/pre-loader/timer corrections. If a proposed H11 hunk actually has
   new required behavior not described in its row, parent must name it and revisit
   that deduction rather than silently charge34 again. I-row reduction requires
   actual scoped diff or an owner-provided separable allowance, not guesswork.

Required inputs/decisions still open:

- Finite pricing60/4549 and both producer seams have completed independent design review. Its model assumptions,
  exact numerical inputs and collection approval remain parent/YC decisions;
  this ledger does not approve simulated stage ordering/overlap/calibration or
  measured lookup. If that review changes shared responsibilities/ceilings, parent
  must update this snapshot's ledger before a final union decision.
- Parent must accept the proposed file writers and H11 whole-row deduction,
  reconcile remaining mixed allowance upper bounds, list excluded documentation/
  gitlink work, and obtain the complete D29 scope approval. No D29 code work is
  authorized by this task or by D30's separate approval.
- Fresh selected native quantization/layout/mask/source receipts, measured
  duration/count keys (including empty/all-padding applicable physical scopes),
  numerical parameters and subsequent native/E2E qualification remain required
  by the owner designs. Source equality, this table arithmetic and CPU fixture
  plumbing prove neither full-runtime correctness nor kernel parity.

## 5. Design validation and progress checkpoint

Read-only document arithmetic **PASS**, not code/tests: parse the candidate and check125
rows/102 paths/9239,17 cross-owner paths,3 same-owner paths and current pricing
snapshot agreement; then check all20 matrix paths against that candidate, merged
+/- arithmetic, the34-line withdrawal and9205 total. The first subtotal assertion
exposed an incorrect added/deleted split with the gross total unchanged; the
matrix sum was corrected to+2467/-363 and the full check passed. New-file
whitespace was checked with `git diff --no-index --check /dev/null
task_memory/task_2026-10-22_step5_position_routing/shared_scope.md`: no diagnostics
(exit1 denotes a new-file diff). `git check-ignore -v` confirms `task_memory/*`
ignores this new record; parent needs a targeted force-add if committing it.
No code/test/platform/agent/commit operation was performed. Only `shared_scope.md`
is written; owner plans/progress remain parent-owned.

## 1. 规划完成情况

**汇总：** PASS3｜BLOCKED/FAILED/SKIPPED1｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 17个跨owner路径责任与writer | PASS | H01-H17覆盖候选全部路径；闭合接口保留 |
| 重复行为与保守ceilings | PASS | H11全行去重34；12个混合allowance不猜行数 |
| 3个同owner增量/union算术 | PASS | 增量260只计一次；102路径/9205保守上限 |
| 实施/测试/platform/commit | SKIPPED | 本任务仅新增设计记录 |
| 模型review/联合批准 | TODO | parent独立执行，不由本矩阵宣称批准 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** D29-SHARED-HUNK-UNION请求至本次交付｜**当前前沿：** parent整合矩阵及有限模型review/决策

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只读表格/decisive seams和文档算术 |
| D2 细节反复重试 | CLEAR | 无实施fix或测试循环 |
| D3 支线问题修补 | CLEAR | 未重查架构、D30或平台 |
| D4 可并行任务停滞 | CLEAR | 有限模型review另行推进，本任务不阻塞D30 |
| D5 超范围镀金 | CLEAR | 无新增实施路径/模型/compatibility |
| D6 分析空转 | CLEAR | 分析落实为17行责任与20行budget |
| D7 路线违背 | CLEAR | 单selector、core闭合seams、native scope与source证明保持原约束 |

**文档检查：** S1-S6均未发现明显缺陷；candidate和progress87-92明确9239是待hunk reconciliation的初步总数。本任务填充计划24要求的责任核对，不把未完成的计划事项判成陈旧记录。
**归因：** 无执行偏离。

## 3. 纠正建议

**分支：** A：spec/plan合理，无需改动；parent接收本矩阵后完成原计划的联合scope和模型决策。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. parent核对矩阵；并行完成有限模型review → 完成标志：行为无双owner、模型及完整scope可决策。
2. 获批后按单writer/接口顺序实施与验收 → 完成标志：实际diff与必要行为都入账；D30独立继续。

## 4. 最值得关注的 3 点

1. 只有H11整行34可当前去重；其余混合行没有证据支持按比例或文件名削减。
2. 同一文件单writer；SOURCE证明、CORE路由/fit、PRICING真实native/数值职责均保留一次。
3. 102/9205是保守计划上限，不是实施diff/最小值/模型批准；有限模型review和scope仍待parent。

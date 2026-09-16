## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Added final-source c9f8f904 homogeneous assembly review and latest reconciled CPU/non-dummy/hunk evidence; preserved prior checkpoints. |
| 2026-09-16 | Reassessed all 38 original findings, R01–R14 and N01–N09 against implemented packages and observed evidence; independently reviewed final EP reporting and expansion ownership. |

# PR33 implementation reassessment

## Current status — final-source checkpoint c9f8f904

This section supersedes pending statements in the preserved earlier checkpoint below. Source reviewed: **`c9f8f904e3550c11aad3cc5d851d75d648cef6e1`**. The narrow independent homogeneous-assembly review found **no concrete correctness issue**: all model layer specs/ranges resolve first; optimization requires the same numerical object and identical family/variant; existing finalized expansion preserves real identities, once-only owners and mutable-source isolation. No tests or benchmarks were run by this lane during the final measurement reservation.

- **Full CPU baseline-relative gate closed, attributed:** `test_report_2026-09-16_w10_cpu.md` now records **3587 PASS / 18 retained baseline FAIL / 25 SKIP in 95.29 s**, 3630 collected, no candidate-only failure or new skip. The independent owner reconciled exact nodes, causes, source, collection and all skip identities. This is not an entirely green suite.
- **Non-dummy final-source gate passed within synthetic scope, attributed:** `test_report_2026-09-16_w10_nondummy_acceptance.md` records **8 PASS in 35.43 s**, **106/106** previous-final artifacts matching the established comparator, and all **7/7 request + 7/7 system** baseline comparisons matching. These checks include independent physical-layer/lane/capacity oracles and do not imply native or production-data parity.
- **Production hunk reconciliation complete, attributed to root ledger:** `changed_hunk_review.md` accounts for **all 471 original IDs** and **675 current zero-context hunks in 106 production files**, with owners and dispositions. The runtime review and this final narrow optimization check support their assigned portions; no unreviewed entry remains in that ledger.
- **Raw 58-case fidelity remains explicitly qualified:** the completed classified campaigns record **24 raw PASS / 34 raw FAIL**, including **22 request/system numerical differences** with named arithmetic/ownership causes. Root subsequently confirmed the completed `c9f8f904` repeat: identical24/34 raw classification,116 successful simulation subprocesses and574/574 artifacts byte-identical to the preceding campaign. The final fidelity report binds those results to the committed source. Historical raw failures remain visible, and no comparator tolerance/golden was changed.
- **Final performance measured; D02 resolved by explicit user acceptance:** root completed18 unprofiled executions on `c9f8f904`. Small/long dense median paired run ratios are1.775045x/1.679023x (14.149→25.225ms and29.309→49.268ms median durations); MoE is0.978816x. Concrete residual/representation alternatives are in `performance_rca.md`, and the user explicitly selected acceptance of the measured residual with the existing contract. Final local records are synchronized; native hardware evidence remains SKIP/UNVERIFIED. D01/D03 remain resolved only in their documented scope; no new acceptance or merge authorization is inferred.

## Preserved earlier implementation checkpoint

Reviewer: `/root/w03_contract_review`. Active specification: `Frontier_PR33_New_Execution_Plan_2026-09-16_EN.md`. Fixed main baseline: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`. Final reporter inspection checkpoint: `5a1cc2af280e5ae2d51caae55fe016c80bc3e2d2` plus working changes. Root continues validation and commits concurrently; this is a source/evidence checkpoint, not a claim that every later commit has been tested.

The requested runtime, boundary and reporting repairs have substantial direct CPU evidence. **Overall task acceptance remains pending W10 final fidelity/control adjudication, the current full CPU baseline/candidate comparison, final unprofiled paired performance and D02, and completed cross-package hunk reconciliation.** Native GPU lanes are collected but remain explicitly unverified on this CPU host. Historical 58-case dummy fidelity, historical full-unit counts, and the 9.42× historical regression are not current acceptance results. No merge or publication is authorized by this reassessment.

## Evidence index and confidence

The reviewer directly inspected runtime entities/scheduler/metrics hunks, the actual EP event-to-wave-to-export call chain, normal test constructors, the final phase projection table and source-batch expansion logic. Numerical/reporting experiments performed by this lane are marked **direct**. Results from other lanes are marked **attributed** to the inspected reports; they were not independently rerun for this document. Source inspection alone does not establish native execution, a full-suite pass, or production-data parity.

| Evidence | Observed result and practical limit |
| --- | --- |
| W01 `test_report_2026-09-16_w01_topology.md` | Attributed 51 PASS; model-owned homogeneous/hybrid identity, actual config binding and invalid layer checks. Does not alone close public non-dummy MLA. |
| W02 `test_report_2026-09-16_w02_lifecycle.md` | Attributed 26 PASS and 185 PASS in separate runs; automatic capacity arithmetic/OOM, transactional rollback and three-request normal Simulator lifecycle. Later owner cleanup uses initialized state directly. |
| W03 `test_report_2026-09-16_w03_fixture_migration.md`, `w03_contract_review.md` | Attributed 203 PASS and 45 PASS in separate fixture-migration runs; direct finalized-stage arithmetic and source/sibling/returned-map isolation tests. D01 differences are accepted only in their named scope. |
| W04 `w04_cache_review.md` | Attributed 52 PASS; stage-local two-key reuse over eight real layers, six hits/two misses, distinct MoE routing, constructor-owned bounded immutable EP workload cache. |
| W04 `w04_performance_review.md` | Attributed 21 successful subprocesses; unchanged hybrid outputs and median paired +22.34% run time when attention reuse is bypassed. EP-cache controls have zero calls and prove no EP-cache speedup. Finalized-container microtiming excludes mutable snapshots. |
| W05 `test_report_2026-09-16_w05_profiling.md` | Attributed 111 PASS / 9 hardware SKIP; early campaign rejection, timer ownership, actual sample requirements and cleanup boundaries. GPU output/state/timing remain unverified. |
| W06 `test_report_2026-09-16_w06_paths.md` | Attributed 51 PASS; all public path projections, aliases, explicit empties, partial overrides, PP paths and invalid-device rejection. No fabricated pre-fix RED. |
| W07 `test_report_2026-09-16_w07_replay_events.md`, `w07_sglang_review.md` | Attributed 48 PASS; independently established fused DEVICE_EVENT/statistics defect repaired, sorting/replay checks repaired. Ten added native cases collect without GPU packages and all ten SKIP on this host; no native success implied. |
| W08 `w08_reporting_review.md`, `hunk_review_runtime.md` | Direct 55 PASS transport/reporting combination; direct latest 14 PASS / 3.21 s includes both expansion orders. Attributed root combined reporting/hybrid 18 PASS / 12.06 s predates the two new quota cases. Actual lane JSONL export is checked. |
| W09 `test_report_2026-09-16_w09_artifacts.md` | Attributed 108 PASS; task/identity/schema completeness, numeric validity and atomic file publication. No whole-artifact-group transactional publication or native parity claim. |
| W10 `w10_nondummy_report.md` | Attributed final **8 PASS / 36.78 s** after predictor freeze, no skips; all seven homogeneous request/system artifacts MATCH unchanged baseline tolerances. Dense/MLA ledger MATCH; operation/lane/trace artifacts have enumerated ownership/schema additions validated by independent oracles. Full CPU and 58-case control/final adjudication remain open. |
| W00/W11 `performance_rca.md` | Fresh initial candidate ratios 1.462255/1.436879/1.007479 for small dense/longer dense/MoE, 18 successful samples. These precede repairs; final-source baseline pairing and D02 remain open. |

Environment of this lane's tests: `/usr/bin/python`, Python 3.12.3, local CPU, no active conda environment. Exact commands and failure iterations remain in the evidence reports. Later documentation commits do not rebind prior tests to new source.

## R01–R14 adjudication

| ID | Advice disposition | Current implementation and acceptance status |
| --- | --- | --- |
| R01 | ACCEPT, QUALIFY causal breadth | Homogeneous classification and model-owned layer identities share the established binder; real stages preserve global IDs/family/variant. Reject the earlier blanket claim that all MLA numerical prediction used dense attention. Focused and final synthetic non-dummy MLA acceptance pass; broader W10 control/final audit remains. |
| R02 | ACCEPT missing acceptance, RETAIN existing plumbing | One admitted request cap reaches planner and slot owner; normal automatic planning, floor-aware block counts, reservation, OOM and homogeneous cases pass. Do not treat explicit num_blocks as proof of automatic capacity. |
| R03 | ACCEPT within resolved D03 scope | Allocation rollback precedes queue/counter commit, continuations retain slots, completion releases/reuses them. Three-request real Simulator evidence closes supported lifecycle. User retained current lifecycle; no request-cancellation API is required. |
| R04 | ACCEPT and implement at campaign boundary | All workload descriptors and writer/method restrictions preflight before native construction or file mutation; inactive-only zero fill. CPU negative and continuation checks pass; native state equivalence remains hardware-limited. |
| R05 | ACCEPT scope migration, REJECT incorrect N02 alias cause | ExecutionTime is one physical layer; StageExecutionTime requires complete identity and snapshots finalized numerics. Generic stage getters aggregate; explicit singleton probes reject multiple layers. Real producers/consumers migrated. Old component getter already copied, so no source-mutation claim survives. |
| R06 | ACCEPT bounded existing ownership seam | Existing component/operator maps/family metadata drive stage and lane projections. Full-stage owner charges once; EP ledger records physical lane work at barrier phase starts. No second execution IR or new scheduler. Final W10 control/failure audit still required. |
| R07 | ACCEPT with per-field precision | One canonical resolver backs manager/predictor/training APIs and preserves keys, aliases, PP paths and explicit empty values. N03 repaired at source; duplication did not prove every old DEVICE_EVENT path wrong. |
| R08 | ACCEPT measured numerical reuse, REJECT blanket routing-cache ban | Cache lifetime is one synchronous stage query, with model-owned family/variant keys; exact immutable layer workload LRU remains separate. Controlled hybrid ablation proves scoped numerical-cache benefit; zero-call EP controls cannot prove EP benefit. |
| R09 | ACCEPT targeted expansion | Existing constructor E2E retained and extended. Actual replica ID repair remains closed. New W10 cases cover supported dense/MLA/MoE architectures and EP/PP; all eight synthetic cases pass, while full CPU/control comparison remains pending, so no branch-completeness claim. |
| R10 | ACCEPT evidence gap, REJECT absent-entrypoint claim | Existing wrappers and native correctness checks reused; collectable native timer/GDN/MoE/collective/SGLang lanes exist. CPU boundary checks pass. Native CUDA/ROCm execution remains SKIP/UNVERIFIED. |
| R11 | ACCEPT scoped artifact campaign | Eight-case non-dummy synthetic constructor/export campaign exists, distinct from historical dummy fidelity. Request/system/ledger and physical operator oracles are explicit. All eight candidate cases pass; request/system metrics match all seven controls. Raw operation/ledger/trace differences are explicitly enumerated; final W10/control adjudication remains open. |
| R12 | ACCEPT RCA and decision gate | Initial paired baseline plus controlled cache ablation now exist. The historical 9.42× is not the current result. Neither a cache benefit nor near-1× process time accepts remaining run-phase cost; final paired source and D02 remain pending. |
| R13 | ACCEPT continuous semantic review, REJECT count-driven deletion | Runtime hunk review maps all assigned original/current hunks; EP projection extracted into coherent `ep_wave_metrics.py`; ambiguous private stage facade and mock state fallback removed. Full 471-entry reconciliation and other-lane final review remain root-owned. |
| R14 | ACCEPT current evidence index | This reassessment preserves historical evidence, corrects withdrawn findings and lists pending gates. Requirements contain active D01/D03 decisions. Final synchronized checklist/summary/status and explicit handoff are still required; no merge. |

## N01–N09 confidence and disposition

| ID | Reassessment |
| --- | --- |
| N01 | Source-confirmed split was repaired at the model binding/topology owner. Family KV capability, actual stage identities and cache keys now agree. Focused tests and final synthetic MLA public non-dummy case pass; no production-data parity is inferred. |
| N02 | **Original source-alias accusation rejected.** Initial task HEAD's `communication_time_component` already deep-copied; Stage changed the detached copy. Finalized publication/isolation remains justified by the broader contract and direct tests, not that nonexistent getter mutation. |
| N03 | Source-confirmed partial override overwrite repaired through per-field canonical precedence; explicit attention/MoE values and empty strings survive missing compute. Post-fix tests pass; no valid pre-fix RED was obtained by this lane. |
| N04 | Reflection/repr/exception-heavy cache compatibility removed in favor of one stage-local concrete dictionary and constructor-owned bounded EP LRU. Real constructor fixtures and separate call lifetimes pass. Routing state is immutable after materialization; arbitrary live mutation is not newly supported. |
| N05 | Reproduced CPU recorder failures before repair established native construction happened too early. Full campaign validation now precedes construction/writing; sentinel outputs and negative cases pass. |
| N06 | Timer ownership now uses shared infrastructure rather than reading private Singleton state; standalone unnamed timing does not poison later named stores. All supported methods and owner precedence have CPU tests. Native event timing remains hardware-unverified. |
| N07 | Reproduced artifact-boundary negatives repaired: duplicate/missing/swapped tasks and metadata mismatches reject at load, using canonical family phase tasks. Attributed 108-pass W09 run includes integration. |
| N08 | Selected identity uniqueness/null checks, exact and estimator finite/nonnegative validation, shared fingerprint and existing atomic persistence implemented. One-row policy is explicitly single-grid-config; per-file atomicity is not a whole-group transaction. |
| N09 | Missing active/E2E samples now fail instead of emitting plausible zero data. Only inactive operators get schema zeros. CPU missing/count/finite checks pass; actual device samples remain hardware-unverified. |

## All 38 original finding IDs

“Implemented” below is scoped to the cited package evidence; it does not silently close pending W10, native or final performance gates.

| Original ID | Disposition and evidence owner |
| --- | --- |
| A SP-01 | ACCEPT; automatic capacity acceptance implemented and observed in W02. |
| A SP-02 | ACCEPT/QUALIFY; identity repaired W01/W03, blanket MLA routing accusation rejected; final synthetic MLA case passes, control audit pending. |
| A SP-03 | ACCEPT/QUALIFY; existing E2E extended to real three-request lifecycle W02, D03 resolved within existing API. |
| A SP-04 | ACCEPT; complete preflight and no-side-effect tests W05 pass. |
| A SP-05 | ACCEPT; meaningful capacity/lifecycle coverage added W02; eight-case public synthetic non-dummy matrix passes; full CPU/control audit pending. |
| A SP-06 | ACCEPT; native acceptance nodes collected and existing wrapper checks reused W05/W07; hardware remains SKIP. |
| A SP-07 | ACCEPT; uniform one-layer versus finalized-stage scope implemented W03. |
| A SP-08 | ACCEPT; new eight-case non-dummy artifact campaign W10 passes, separate from still-pending final dummy control. |
| A SP-09 | ACCEPT; controlled RCA W04 observed; final baseline pairing and D02 W11 pending. |
| A SP-10 | ACCEPT; NVIDIA/AMD claims remain separate and no native result inferred from CPU. |
| A ST-01 | MERGE with A SP-02; one authoritative model binder/topology owner W01. |
| A ST-02 | ACCEPT; producers and scheduler/metrics consumers migrated together W03/W08. |
| A ST-03 | ACCEPT; stage sentinel pass-through removed, constructor-valid fixtures migrated W03. |
| A ST-04 | ACCEPT; existing scalar/component/operator/family projections reused W03/W08, no replacement framework. |
| A ST-05 | ACCEPT; canonical resolver/public APIs and actual partial override tests W06 pass. |
| A ST-06 | ACCEPT; exact stage-local reuse W04 passes and trained synthetic ablation has observed benefit; W11 final pending. |
| A ST-07 | ACCEPT; implementation/hunk review performed throughout; final whole-inventory reconciliation still pending. |
| B D1 | MERGE with timing ownership; W03 code replaces dual behavior, not merely debt documentation. |
| B D2 | ACCEPT; attribution and optimization do not prove residual necessity; final W11/D02 remains open. |
| B D3 | ACCEPT/QUALIFY; request-owned slots retained, no new cancellation API under explicit D03 decision. |
| B D4 | ACCEPT; concrete native nodes exist, hardware absence remains truthful SKIP. |
| B T1 | ACCEPT; direct independent mixed-layer/owner/lane/barrier/expansion regressions W03/W08 pass. |
| B T2 | ACCEPT; supported homogeneous EP/PP W10 coverage added, no forbidden GDN topology enabled; full CPU/control audit pending. |
| B T3 | REJECT proposed flat-sum invariant; physical phase/barrier timing retained and lane work remains separate from critical-path duration. |
| B T4 | MERGE duplicate native/platform acceptance into W05/W07, not a second framework. |
| B T5 | ACCEPT; precision-aware parameter/state/KV cases W02, eight-case non-dummy matrix passes; final W10 control audit remains. |
| B T6 | RETAIN resolved actual-replica-ID fix; normal Simulator passes actual cluster keys and regressions remain. |
| B T7 | RETAIN accepted hardware boundary; final handoff must retain native SKIPs W11. |
| B Q1 | ACCEPT; generic stage views aggregate, explicit singleton probes guard scope W03/W08. |
| B Q2 | MERGE timing dual-semantics repair W03; no separate compatibility facade. |
| B Q3 | MERGE measured optimization and causal validation W04/W11; final paired source pending. |
| B Q4 | ACCEPT bounded owner seam; EP reporter is narrow projection using actual records W08. |
| B Q5 | REJECT count-based deletion; semantic responsibility/hunk review W00–W11 instead. |
| B Q6 | QUALIFY/MERGE true owners: identity W01, timing W03, path precedence W06. |
| B Q7 | ACCEPT; preserve timer methods, repair ownership W05, native evidence still pending hardware. |
| B Q8 | SPLIT by responsibility; identity W01 and supported lifecycle W02 pass, cleanup/handoff W11 reconciles. |
| B Q9 | ACCEPT semantic cleanup; arbitrary formatting/cosmetic work deferred, private scope debt removed W03. |
| B Q10 | RETAIN resolved contracts; no unrelated constants/tolerance changes. Explicit D01 accepts only its isolated differences. |

## Final EP reporting review and corrected runtime findings

`ep_wave_metrics.record_ep_wave` consumes actual lane snapshots and actual phase boundaries. Pre-dispatch starts at wave entry, dispatch after maximum pre-dispatch, routed work at dispatch barrier completion, combine after maximum routed duration, and post-combine at combine completion. Individual lane operator durations remain distinct; barrier idle time is a gap. Routed metadata uses lane routed tokens; full-stage CPU/PP/draft/terminal owners are absent from lane records. Separate JSONL rows retain phase/operator work and source batch identity; generic full-stage arithmetic is not forced to equal summed lane work.

The source-batch quota repair was independently tested in both call orders. Source batch 73 emits actual attention IDs and EP layer 7 even with request quota one; unrelated next batch 74 emits three aggregates with `layer_id=-1`. Latest direct command:

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 python -m pytest tests/unit/test_stage_reporting_contract.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w08-quota-review --tb=short
```

Observed **14 PASS in 3.21 s**. This includes real MetricsStore construction, independent numerical oracles, event output and lane ledger file serialization; no predictor/native parity inference follows.

New phase-table versus old getter comparison found no legacy gating semantic difference across six supported legacy/split/structured/override forms. One hundred mixed finite scalar cases differed by at most `1.4210854715202004e-14 ms` from floating addition association. No bit-for-bit equivalence claim is made.

Runtime review F1 (ledger completion accidentally gated by utilization) is repaired and tested. F2 (actual EP payload discarded before requested reporting) is repaired by conditional transport and the narrow reporter; eight-case architecture matrix passes with actual lane artifacts; full CPU/control audit remains W10. F3 (mock-justified missing slot-manager fallback) is repaired with direct constructor-owned reads and legitimate fixture initialization. F4 (all-GDN zero-KV admission) is withdrawn as a reachable production bug: normal runtime family resolution rejects a model without any full-attention layer. The initial schedule-helper-only inference did not establish production reachability.

`hunk_review_runtime.md` accounts for 72 original entities/scheduler/metrics H IDs plus H214–H215 KV transfer and H451–H455 Simulator, **79 original IDs total**, with exact current checkpoints and final reporter addendum. Other lanes/root own the remaining global inventory; this reviewer does not claim all 471 solely from this local review.

## Remaining acceptance gates

1. Finish W10 control/fidelity adjudication after the eight-case non-dummy matrix passed. Seven homogeneous request/system comparisons match, while operation/trace/ledger differences are explicitly enumerated. Diagnose any remaining control differences; D01's existing approval is limited to the two named original corrections and is not a blanket tolerance waiver.
2. Complete current full CPU baseline/candidate comparison with candidate-only failure analysis. Historical 19 baseline failures or old pass counts are insufficient for this source.
3. Complete final unprofiled alternating baseline/candidate performance with unchanged scenarios, actual event/request equality, initialization/run/total/RSS reporting, and causal interpretation. Present D02 only from final residual evidence; no budget acceptance is inferred here.
4. Reconcile all 471 original production hunks plus later repairs across review lanes, then synchronize the current status/checklist/summary and preserve original authorship/history. This reassessment is an evidence index, not the final overall PASS.
5. Retain native NVIDIA/AMD checks as explicit SKIP/UNVERIFIED until authorized hardware execution produces the required operator/state/timer/layout evidence. CPU synthetic training and mocks do not satisfy native parity.

D01 and D03 are explicitly resolved in `requirements.md` for their stated scope. No additional user design decision is requested by this document; D02 awaits the evidence above. No merge, force-push, rebase or remote publication was performed by this review lane.

## Full CPU intermediate failure follow-up

Root's current full CPU run (`/data/ycfeng/tmp/pr33-w10-final-unit.log`) observed **45 FAIL / 3551 PASS / 25 SKIP in 99.82 s**. This is a visible failing intermediate run, not the historical known-failure set or a current no-regression pass. Other lanes/root are diagnosing the complete difference set.

This lane owned two specific failures. `test_mla_core_native_op_tracing` inherited global QuantizationManager BF16 state from prior tests while asserting its unchanged FP16 metadata oracle. A narrowly requested fixture now resets and configures the real manager from an explicit float16 model, then resets on teardown. `test_transfer_metrics_contract` uses a ledger-only fake config that omitted `store_operation_metrics`; the stage reporting migration correctly consults that independent flag even when utilization is off. The fixture now explicitly sets it false. No production fallback, arithmetic tolerance, or assertion weakening was introduced.

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 python -m pytest tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_transfer_metrics_contract.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w10-metrics-fixture-review --tb=short
```

Observed **84 PASS in 3.05 s**. `git diff --check` for these two files and the appended quota tests passes. This resolves the assigned two failures only; a rerun of the full suite is still required after all relevant fixes. The newer W10 report independently records **8 PASS in 36.78 s**, all seven homogeneous request/system comparisons matching baseline, and explicit operation/trace/ledger differences. These newer scoped results supersede earlier W10 pending-candidate statements without closing the remaining gates.

A final fixture consistency edit explicitly sets the MLA model's own `torch_dtype="float16"` as well as configuring the manager from that same model, keeping trace context and quantization owner aligned. The same two-file command with `--basetemp /data/ycfeng/tmp/pr33-w10-metrics-fixture-consistent --tb=short` observed **84 PASS in 2.93 s**.

## Independent final homogeneous assembly review

Inspected exact commit `c9f8f904` changes in `BaseExecutionTimePredictor._assemble_stage` and `tests/unit/test_dense_execution_time_layer_scaling.py`, plus the existing `StageExecutionTime.from_execution_time`, `ExecutionTime.as_single_layer`/`finalized_copy`, model-owned spec accessor and all four production call sites. This is a read-only correctness review, not another test execution.

1. **Identity and range:** `_assemble_stage` first calls `get_layer_attention_spec(first_layer_id + offset)` for every requested layer. The normal model accessor checks bounds and returns the immutable indexed spec; topology construction assigns that same physical index. Therefore the existing homogeneous constructor's contiguous IDs match the already validated model specs. Negative/overshooting ranges fail before homogeneous total computation. Family and variant equality are both required; a shared object crossing a GDN/full-attention boundary takes the ordinary per-spec assembly path.
2. **Numerical admissibility:** object identity, not scalar equality or family label alone, selects the optimization. Independently predicted MoE/routed layers cannot collapse merely because their durations happen to be equal. All four actual callers pass lists, so the now-explicit length/index operations do not break a generator caller. Empty input still reaches the existing nonempty-stage rejection; singleton input retains ordinary assembly.
3. **Owner arithmetic:** `from_execution_time` caches `N * source.get_single_layer_block_time()` plus PP/proposer/terminal once. This is the existing homogeneous-stage contract; `ExecutionTime` remains one-layer scope. CPU and diagnostic totals continue to use the single owner record. Other public component/operator views still aggregate all physical layers and exclude/reinsert PP according to the existing owner rule.
4. **Mutable safety and IDs:** `as_single_layer(copy_components=False)` first finalizes a mutable source by copying its component objects and operator dictionaries. Already finalized payloads may share those immutable numerical objects, while each layer gets its own identity record. No new BaseEntity ID is consumed. Public component/map getters remain detached/read-only, and supported mutation methods reject finalized layers. No new mutable alias is introduced by this optimization.
5. **Test adequacy:** new tests observe one block calculation for five-layer ordinary/PDD predictions, repeated cached total access, unchanged complete range rejection, and mixed-family identities even with a shared source. Existing tests also assert one component snapshot, separate identity objects, and exact four-role arithmetic/operator/communication equality against independent per-layer construction. Existing finalized-stage tests cover source/returned-map/sibling isolation and owner fields. The reported focused run is **132 PASS in 5.59 s** in `test_report_2026-09-16_predictor_owner_once.md`; it is attributed, not rerun by this reviewer.

**Disposition: RETAIN / REFACTOR, no blocking finding.** Required O(N) range/identity materialization is preserved; only repeated arithmetic over the same finalized numerical payload is removed. Performance impact is left to the final exclusive paired measurement and must not be inferred from this source review.

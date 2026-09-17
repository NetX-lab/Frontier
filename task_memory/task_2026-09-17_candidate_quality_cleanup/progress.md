## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Started independent diff-driven quality cleanup. |

# Progress

## P0 — in-progress

- Read all 817 lines of AGENTS.md, prior requirements/plan/progress/current status, and relevant validation records. No nested AGENTS.md found in source/test/data/docs search.
- Pinned `main=0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`, `candidate=c288a19f59bec09529ee18d782fa57218da2c781`, identical merge-base/main. Initial tracked worktree clean; 266 changed files including substantial historical evidence.
- Local clock confirms 2026-09-17 Asia/Hong_Kong. No new remote access or publication performed.
- Planning-with-files applies using this task directory; code-review supplies independent Standards/Spec reviews; codebase-design informs invariants/ownership, subordinate to user scope.
- Read-only Standards worker owns profiling/training/config review; Spec worker owns prior behavior/verification boundary check. Main owns runtime inspection and all implementation.
- Previous task reported 18 known main-equivalent CPU failures and explicit native hardware skips. These are leads for fresh verification, not current PASS evidence.

## Pending work

1. Build ranked module inventory and preserve executable pre-refactor reference.
2. Establish fresh baseline evidence and inspect runtime changed hunks.
3. Execute P1–P5 in dependency order.

New unresolved issues: none established.

## P1a — completed: construction and finalized snapshot contracts

- Preserved pre-refactor candidate in `../quality-baseline-c288a19f`; pinned main reference remains `../pr33-r12-baseline-20260915`.
- Default Python 3.12.3 lacks pytest/dependencies. Consulted environment/package-mirror handbooks, then installed isolated CPU/test requirements with uv in `/data/ycfeng/tmp/quality-review-env`. No shared environment was modified.
- Fresh full-unit collection failed on eight missing-torch imports and one missing-matplotlib import. Public CPU Torch index timed out after three retries; logs: `/data/ycfeng/tmp/quality-pre-unit-env.log`, `/data/ycfeng/tmp/quality-env-torch.log`. Matplotlib installation through the internal mirror succeeded. Torch resolution remains pending.
- Motivation: Simulator duplicated predictor construction; ExecutionTime treated constructor-owned components as absent; StageExecutionTime repeated filtering after rejecting missing IDs.
- Method: one predictor construction loop, direct required component access with isolated operator-map copies, uniqueness check on validated IDs. Ordinary monolithic manager/path remain None; disaggregated/hybrid sharing and physical-layer identity remain unchanged.
- Checked CC backend lazy construction: backend factory reads cluster configuration/runtime topology; manager owns training/cache state, not backend configuration. Shared initialization remains before predictor creation.
- Verification: frozen candidate eight non-dummy cases passed (65.34 s). Cleaned candidate 78 focused/non-dummy tests passed (71.00 s). Existing comparator found 90/90 stable artifacts equal across eight cases. Bidirectional artifact inventory and acceptance/stage-summary comparison follow the independent Spec review recommendation.
- Editing command `apply_patch` was absent from PATH; no patch was applied on the first attempt. Resolved by invoking the installed Codex binary in its apply_patch mode, without changing system configuration.

## P1b — completed: canonical memory/model contract

- P1a committed as `48a4093b`; supplemental symmetric metrics inventory and acceptance/stage-summary comparisons also PASS for all eight cases (106 metric files total).
- Removed planner-local hybrid layer scanning and used `BaseModelConfig.get_attention_family`, the same runtime KV owner as Replica/head metadata. All-GDN Qwen is already rejected by construction, so the removed zero-KV fallback represented no admitted model.
- ParamCounter now calls the required per-layer and GDN methods directly; genuine `get_gdn_config() is None` remains supported for homogeneous models.
- Focused before/after suites: 47 PASS / 47 PASS. Four TP/PP memory snapshots match exactly, covering weight bytes, recurrent state bytes, page size and available block count.
- Found usable same-interpreter system Torch 2.12.0+cu132. Enabled standard system-site-packages inheritance only in the dedicated venv; no target overlay or different Python package tree was injected. Fresh full frozen-candidate unit run is in progress.

## P1c — completed: GDN batch and scheduler invariants

- Batch/Request expose all required feature properties; GDN feature extraction no longer fabricates missing counts/state or copies the request list. Positive query lengths establish a nonzero mean.
- Runtime guards consume the required BaseModelConfig method. Preemption reads constructor-owned replica configuration; the incomplete PD-AF preemption fixture now provides a real non-GDN model.
- Focused verification: 46 PASS, including feature vectors, mixed-batch rejection, admission/continuation/completion/rollback and preemption progress. Transactional exception cleanup remains unchanged.
- Full frozen-candidate baseline initially finished 3584 PASS / 21 FAIL / 25 SKIP. Three added environment failures came from example shells resolving `/usr/bin/python`, which lacks plotly. Re-running with the dedicated venv on PATH; no production workaround added.

## P1d — completed: MoE routing construction and layer state

- Removed the unreferenced private routing-generator alias (only test caller updated), None-then-dict initialization, repeated validation of internally generated maps and unsupported `_cluster_num_replicas` alternate state.
- Reused subclass `_get_cluster_replica_config` through normal override dispatch instead of reflective capability discovery. Preserved optional actual replica IDs, standalone local IDs and per-replica map isolation.
- Removed unused ExecutionTime layer-count attributes: neither production nor tests read them; constructor argument validation remains compatible.
- Verification: 116 focused/non-dummy PASS; 90 stable artifact comparisons PASS, symmetric 106-file metrics inventory and supplemental summaries match. No expected value changed.
- Corrected baseline environment: 3587 PASS / 18 FAIL / 25 SKIP. The same 18 failure nodes also fail on pinned main. Source causes are missing legacy debug/analysis assets and outdated release-document expectations. README remains untouched.

## P1e — completed: lowest-free-slot ownership

- Replaced list front removal and full release-time sorting with the standard heap priority queue. The observable rule remains allocation of the lowest available slot ID.
- 16 lifecycle/scheduler tests PASS; added one out-of-order release regression. A deterministic 1,000-request trace compared directly against the frozen state manager produces identical allocation/release records.

## P1 full regression follow-up — completed

- Full candidate run returned 3583 PASS / 22 FAIL / 25 SKIP. Four newly exposed failures are incomplete fixtures: the shared routing model omits required `is_moe`; two transfer model doubles omit `get_num_gdn_layers`. The production constructor/accessor contracts were verified before adding those members to the doubles. Expected routing and transfer values remain unchanged.
- The remaining 18 nodes match the freshly established frozen-candidate/main failures. Detailed log: `/data/ycfeng/tmp/quality-p1-unit.log` (132.10 s).
- Experimental routing integerization characterization completed 3,456 generated cases with zero admitted histogram mismatches. This is sampled evidence, not a proof for arbitrary ratios; unification remains under investigation.
- Focused fixture/transfer/GDN checks now PASS: 25 tests in 5.62 s, with unchanged expectations. No production accommodation was reintroduced.

## P1f — completed: centralized raw architecture identity

- Consolidated Qwen type/exact-alias recognition in `model_architectures.py` without entering profile/topology resolution. Kept registry, structural and topology precedence/normalization differences.
- Characterization: 100 tests PASS before (9.88 s) and after (10.78 s). Initial direct re-export triggered a verified import cycle; the existing exported GDN function now delegates with a function-local import. This adapter is required by actual import ordering, not an incomplete fixture.
- Binding follow-up: 110 PASS; canonical GDN family identity replaces consumer literals, homogeneous runtime binding avoids repeating its already-proven guard, and topology cache regression patches the actual owner lookup.

## Phase-boundary evidence — in-progress

- Running the source-stable 58-case fidelity campaign at `61595325` against frozen `c288a19f`, with two workers. Output: `/data/ycfeng/tmp/quality-p1-fidelity`; no production edits during execution.
- P2 pre-change reporting baseline: 129 PASS in 7.01 s; `/data/ycfeng/tmp/quality-p2-before.log`. The suite covers timing copies, full-stage scope, mixed layers, operator metadata and EP lane reporting.
- Corrected the supplemental artifact inspection to require one actual `frontier_stage_batch_ledger_summary.json` and one `acceptance_evidence.json` per case. Both P1a and P1d compare PASS for all 16 pairs, with an explicit count assertion. Earlier prose claiming stage-summary equality is now supported by a nonempty check; the earlier filename probe alone was insufficient.
- Reviewed finalized-copy paths: repeated calls return the existing immutable snapshot, so they are not repeated deep copies. Stage scalar projections use finite declared property sets, not broad `__getattr__`. Single-layer helpers reject multi-layer stages; keep these supported interfaces.
- P1 fidelity finished **58 PASS / 0 FAIL** at `61595325`; every requested stable artifact and ledger/request value matches the frozen candidate.
- Subsequent L2 private-layer cleanup removes an uncalled aggregate degree of freedom from two private interfaces. Real MTP stage width is retained. 105 focused tests PASS in 7.92 s; final E2E will include this small subsequent change.

## P1 timing / P2 S1 — completed

- Paired run completed 18/18 successful measurements at production HEAD `b0b9683b` against `c288a19f`; only the unrelated red-regression test was dirty. No concurrent test load or production edits during timing. Median paired Simulator.run changes: small dense -17.32%, longer dense -10.12%, representative MoE -2.96%. Events/completions match in every pair. Reporting is disabled by this existing harness; no metrics-enabled or native-device performance claim follows.
- S1 RCA reproduced six frozen-candidate failures and three native-main scalar successes: the added physical-layer metadata path conflated complete event metadata with additive identity. Restored the distinction using a keyword-only local emitter argument; residual/wait metadata bypass tensor resolution, layer identity still supplements derived tensor metadata. No resolver exception suppression or operator-name special case.
- Verification after S1: 139 PASS in 7.30 s, including all six positive residual cases and existing reporting contracts. This is a separately documented correctness restoration, not asserted as equality with the broken candidate path.
- Sidecar bounded scopes: Meitner owns scheduler reporting M1b/M2; Nash owns trace context M4/M5. Main owns metrics_store M1a/M3 and integrated verification. Preserve all workers' changes; stage each logical result separately.

## P2 M1a / M3 inspection — completed

- Removed orphaned `_trace_dense_*` and `_trace_execution_time_override` readers. A full production/test search finds no remaining writer except neutral annotation-isolation tests; actual mixed layers already arrive in StageExecutionTime. Removed an unused family lookup in stage tracing. Scalar timing/reporting remains supported.
- Focused reporting checks: 69 PASS in 6.57 s, log `/data/ycfeng/tmp/quality-p2-readers.log`; the S1 suite supplied the pre-change baseline.
- Corrected M3 reviewer proposal before implementation: FFN trace name `mlp_act` is NOT the historical metric label `mlp_activation`; `OperationMetrics(op_name)` would fail. No existing generic alias mapping exists. Retained three explicit scalar-to-metric projections rather than introducing an alias abstraction solely to shorten them. This is schema adaptation, not duplicated family ownership policy.
- P3 pre-change CPU baseline: 253 PASS in 17.65 s, `/data/ycfeng/tmp/quality-p3-before.log`, covering GDN artifacts/profiling, architecture/MLA config, native-adapter admission and experimental replay. These are not native GPU executions. Read-only preparation found no need to alter the standard/experimental dependency boundary.
- Discovery-only path misses: GDN shared schema lives in `frontier/attention/gdn/features.py` and `config.py`, not `frontier/gdn/`. No file edits or failed tests resulted.

## P2 scheduler reporting — completed

- Reconciled Meitner's eight-file patch with production callers: deleted only the unused dense-reference/correction helpers, migrated their tests to live helpers, used canonical mixed-layer count and direct configured model, removed unused correction arguments, and avoided an immediately discarded Stage copy. Actual dense layers remain separately predicted.
- 110 before / 114 after focused PASS. Independently confirmed five before/after timing snapshots are byte-identical with `cmp`; scalar, dense, zero-MoE, all-MoE and mixed stages preserve numerical ownership and call identities. Detailed commands and values are in `test_report_2026-09-17_p2_scheduler.md`.
- A broad temporary-root filename search encountered unrelated permission-denied directories; abandoned broad discovery and read only explicit task log paths. No data or permissions changed.

## P2 trace context — completed

- Reconciled Nash's patch and committed `8e67fc25`: canonical GDN family membership, phase-local trace context/parallel values, one token sum per traced lane and per-event flat dictionary copies. Positive-work and reporting-off gates remain; no persistent cache.
- Final focused result: 80 PASS in 7.12 s. Direct before/after comparison covers 64 reporting cases, 80 events per revision and all four GDN operators, with exact event/metric/ledger equality. A five-operator lane constructs 3 contexts rather than 5 and sums tokens once rather than 5 times.
- DP RCA corrected the initial fixture finding: the only production `predict_dp_moe_allreduce_times` implementation explicitly returns `(0, 0)` for the retired scope, with no override. Removed two candidate-added unreachable COMM shape labels; did not expand the registry or change the public zero-return seam. Positive synthetic DP input remains rejected identically.
- The sidecar's exact final command wrote evidence in its report/tool output, not a guessed `.log` path. A failed attempt to tail that nonexistent log skipped a documentation patch, not code verification; this entry repairs that documentation omission. Integrated P2 tests now provide a dedicated retained log.

## P2 integrated / P3 schema — completed

- Source-stable integrated P2 at `8e67fc25`: 159 PASS in 69.62 s, including eight real non-dummy simulations. Compared with frozen candidate: 90 stable artifacts PASS, 106-file symmetric inventory equal, 16 explicitly nonempty summary/acceptance JSON pairs equal. Logs: `/data/ycfeng/tmp/quality-p2-integrated.log`, `/data/ycfeng/tmp/quality-p2-artifact-comparison.log`.
- P3 S4 uses family-owned profiling schema/operation names and existing GDN_TASKS phase order; removes unreachable zero-mean handling after positive-query validation. Required columns remain the exact ordered 27-tuple, both phase task orders and the four quantization names match frozen declarations.
- S4 focused: 184 PASS in 16.34 s; pre-change broad P3 baseline was 253 PASS. Real optional mask/physical-batch normalization and inactive-phase output zeros remain unchanged.

## P3 S5 — completed

- Profiling ModelConfig now uses the canonical runtime family resolver, removing the production branch that existed for a module-local monkeypatch. The test patches the actual owner and still asserts one call.
- Removed only redundant BaseModelConfig-owned JSON overlays; preserved raw model_type and linear-shape overlays whose null/string serialization can differ.
- Verification: 129 PASS in 10.37 s. The full 22-model snapshot is identical: 21 valid configurations and one unchanged deepseek-v3 quantization rejection. Initial raw cmp differed only in logged warning timestamps; compared complete JSON payloads explicitly, without excluding config fields.
- Continuation reconciled HEAD and disjoint worker edits. S6 architecture and J1 ROCm workers remain active; main retains S5 and subsequent integration ownership.

## P3 J1 — completed

- Reviewed the ROCm sidecar patch against the mixed-batch guard: successful nonempty inputs contain only one phase, so metadata reuses the first sequence plan. Required lifecycle None states and both timer scopes remain.
- Verified stored snapshots with cmp: exact metadata/slot/output/timer equality, plan builder calls 5 -> 3 across prefill/decode/empty begins. Focused 5 before / 5 after PASS. CPU stand-ins exercise real wrapper construction, not native attention numerics.
- Main stages only J1 production/test/report files; J2 remains disjoint and starts after this commit.

## P3 small contract reuse — completed

- GDN topology uses the family-owned singleton variant; collective Ray runner reuses the existing environment setup owner with original ordering. No new registry/helper or standard-to-experimental dependency.
- 39 tests PASS before and after; direct normally constructed BenchmarkRunner setup against frozen candidate preserves all four environment values and device-selection call.
- S7 investigation expanded to 3,407,872 histogram comparisons across distributions, expert counts 4–256, sizes 1–4096, top-k up to 8 and four seeds. No admitted histogram mismatch. Plan: extract the existing integerization implementation within moe_ep_workload for both consumers, avoiding irrelevant replica/EP placeholder state; compare actual replay counts/assignments and runtime workload tests before commit.

## P3 S6 — completed

- Reconciled exact native model-type declarations and queries in the existing architecture registry. Norm/MXFP4 consumers no longer own Qwen classifications; supplied ModelConfig fields are read directly.
- Preserved exact matching, optional model-type value, norm/platform/API guard order and error strings. Do not broaden admission through profile aliases/normalization. Adding another native type requires one profile declaration.
- 183 focused PASS; 240 Gemma and 90 MXFP4 gate outcomes exactly match pre-change code. An initial early registry import exposed a native import-cycle regression; function-local import after normal bootstrap fixes that new path, verified in fresh interpreters. Existing registry-first import debt remains out of scope and recorded, not hidden.

## P3 S7 — completed

- Shared the existing Hamilton implementation within moe_ep_workload; replay consumes the histogram contract directly, with no placeholder replica/EP state. Alias retains the existing ratio-generator entry point.
- Immutable frozen baseline 52 PASS / cleaned 60 PASS. Discarded initial asynchronous local pre-check as baseline because its source window overlapped editing; reran on frozen checkout.
- Exact frozen/current comparison: 893 full replay dictionaries including assignments, 67 infeasible rejections and 2,679 EP workload values match. Larger histogram-only characterization also found no difference. Runtime's normalization/arithmetic/tie order is unchanged; final fidelity/timing will include its extracted function call.

## P3 J2 — completed

- Reviewed the replay sidecar's explicit profile_graph parameters and internal named call contract. Removed seven-position metadata overloading and GDN spec re-derivation; native builder signatures and output row schemas remain intact. Mutable snapshots, reset/check ownership and optional trace-probe lifecycle remain necessary.
- Focused 39 before / 40 after PASS (one added named-argument test). Main independently compared all 12 JSON snapshots: exact equality including every row/output and 1,802 ordered events. Simulated timer values do not establish GPU speed.
- Beginning source-stable P3 integrated CPU regression; workers perform read-only P4 preparation until release.

## P3 integrated — completed; P4 — in-progress

- Source-stable `4dd63482`: 721 PASS / 5 SKIP / 2 FAIL across 45 CPU modules. Both failures are the previously reproduced main/candidate MHA/MQA stage2 CLI nodes. Direct subprocess RCA reports missing flashinfer-python; scripts are unchanged from main. No new failure observed; do not describe this as an all-green/native-GPU suite.
- Released P3 source freeze. P4 independent scopes: Nash GDN predictor/trainer and valid fixtures; Meitner config guards/typed registry access/default ownership in bounded substeps; main manager registries, GDN paths and family binding. Existing manager measurement/precision identities and dataset-only optionals remain fixed.

## P4 M1 — completed

- Manager MLA classification delegates to runtime family binding; whole-model None remains supported. Measurement selector requires the replica already supplied by production and four formerly incomplete test calls.
- 127 frozen / 127 cleaned PASS. Exact method comparison across 120 architecture/device/role/graph selections and four model families also PASS. No path precedence or estimator output changed.

# D29 CPU performance component review archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Append separate prepared-point Standards/Spec closure and explicit primary acceptance at eece87b2; preserve earlier review sections and provenance. |
| 2026-10-23 | Add independent trace Standards/Spec closure and parent component acceptance at64881373. |
| 2026-10-23 | Archive separate Standards/Spec closure,group records P2 disposition and explicit primary component acceptance. |

Task ID: D29-PERFORMANCE-COMPONENT-ARCHIVE.
Component acceptance is complete for the original three CPU repairs and the
later trace repair described below. This file archives
the primary's supplied independent review results; it is not a fresh review.

## Reviewed artifacts

| Component | Code-only commit | Records delivery | Owner diff and records directory |
| --- | --- | --- | --- |
| D29-MODEL-GROUP-REUSE | 3763e5b87a845eac186cfd481f0b56cffb2dd045 | 88fe6c65db675609ee48043ca63b7673c5f739c8,then37c7832ee5ee2e9588ba6fed411a9e9894949ab8 | `/data/ycfeng/Frontier/.worktrees/step5-d29-group-reuse/task_memory/task_2026-10-22_step5_position_routing/cli_group_reuse/`;`group_reuse.diff` |
| D29-INIT-FAMILY-RESOLUTION | ee7e66ee439054978897c91fa32598fc13804883 | 35d2f74ed60855e48e5af4ea97c49721b9aba7ea | `/data/ycfeng/Frontier/.worktrees/step5-d29-loader-family/task_memory/task_2026-10-22_step5_position_routing/init_family_repair/`;`code.diff` |
| D29-COLLECTIVE-ARTIFACT-PARSE | 51e1f7205d07b269db2291c5f8076431a86d29e5 | 0fa22e49417b80ee315518f9ea6843b77663fe09 | `/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-parse/task_memory/task_2026-10-22_step5_position_routing/artifact_parse_repair/`;`source.patch` |

Review requirements: primary `requirements.md`, `plan.md`, and
`repairs/model_group_reuse.md`, `repairs/input_validation_and_artifact_parsing.md`
with their three ordered scope overlays; owner requirements/plan/progress,
complete source diffs, accounting and verification records.

## Standards axis

Reviewer session: `ses_e97df8ed7ffe6u75S4UBsS75hI`.
Result for each of the three components: **No findings;0 findings,worst none**.

- Group reuse stays at the fixed predictor's existing consumer-local boundary.
  `frontier/execution_time_predictor/moe_routing_workload.py:127-142` owns the
  complete immutable group tuple and selects the current layer each call;
  `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:121-125`
  removes duplicate derivation through that same method. The real regression at
  `tests/unit/test_prediction_measurement_bindings.py:166-217` covers layer3/88,
  independent predictors/models,dynamic bindings and negatives.
- Loader reuses the existing family resolver and `architecture_family_id`
  keyword; parser adds only backend-local latest-text/parsed-result state.
  Neither introduces another public API,global cache,flag or numerical rule.
- Applicable standards include `AGENTS.md:798-800,838`: existing boundaries,
  backward compatibility and unchanged results/intermediates. Reviewed loader
  and backend modules are992/934 lines; no2000-line exception is required.

## Spec axis

Reviewer session: `ses_e97df4d82ffetkqpyEbPP2iCA1`.
Final result for each component: **No findings;0 findings,worst none**.

- Group's current layer,source/family/native context,estimator,point admission,
  exact duration/count and plan checks remain live. No source defect was found.
  Initial records P2 overclaimed the six-file81PASS/1SKIP selection as
  both-trainer coverage. Records-only37c7832e corrects it to shared-only and
  records a separate parent2PASS actual trainer/preflight and independent
  selected native-context fit/cache selection. The same Spec reviewer recheck
  closes P2. Those counts stay separate. Owner records' earlier pending-review
  statements are historical,not the current review result.
- Loader at
  `frontier/execution_time_predictor/profiling_dataframe_loaders.py:720-729`
  derives once per call but preserves all-row typed `.map` before filtering,
  through the existing `architecture_family_id`. Regression
  `tests/unit/test_moe_grouped_gemm_padded_features.py:300-331` checks per-call
  derivation,full selected-frame equality and corrupt unselected operator
  rejection. No validation is skipped or cached.
- Parser at
  `frontier/cc_backend/backends/collective_sim_cc_backend.py:73-74,813-817`
  retains every current file read and every live gate. Reached
  `Scenario.from_dict`, `predict_collective_time`, `estimate_plan_local_ms`
  and runner code do not mutate the retained artifact. Regression
  `tests/unit/test_collective_sim_cc_backend.py:200-233` checks same-size
  restored-mtime rewrites,current parse/read errors and live count/key checks.
  Parse state changes only after successful parsing;native cost/cache hashes
  and qualification rules remain unchanged.

## Review execution and acceptance limits

Both independent reviewers inspected full diffs and records. Neither ran
commands/tests or independently reran hashes. Their axes remain separate;
parent verification is separate execution evidence,not reviewer execution.

Primary explicitly accepts all three completed CPU components after closure.
Primary asserts all three isolated trees clean and source/test bytes identical
to their code-only commits. The archive task does not rescan that assertion;
it reads owner records and reparses/re-hashes the five existing parent XMLs.

Group one-point pricing is parent-verified. Parser timings and complete payload
equality remain **owner evidence only** until the integrated parent probe.
See `test_report_2026-10-23_components.md` for provenance and exact receipt fields.

Integration,final freeze,actual merged accounting,full CLI,fresh strict77/
full-unit,T2/native/GPU/G3/E2E remain pending. No extra early-init fix is approved;
partial startup diagnosis cannot justify changing existing cache hashes.

## Trace serialization component

Fixed point `767da543`, code `390bb5a2`, reviewed records/tree `9ae60ebc`.
Diff: `git diff 767da543...9ae60ebc -- frontier/metrics/trace_store.py`.
Spec sources: parent requirements/plan, trace scope amendment and
`trace_serialization_repair/` requirements, plan and evidence. The independent
reviewers read the complete source and bounded producer/test evidence; neither
reran tests or hashes.

### Standards

Reviewer `ses_e972c468cffeZKEx6278S1oCde`: **No findings; 0, worst none**.
The declared-field serializer stays at the existing flush boundary without a
second helper or public alias-return interface. Naming, boundary reuse and
behavior preservation follow `AGENTS.md:791–800,838`; no actionable heuristic
smell was found. Error propagation and pending-before-concurrent restoration
remain unchanged in the reviewed `trace_store.py:122–126`.

### Spec

Reviewer `ses_e972c45a3ffenZuSi1vyddEDKM`: **No findings; 0, worst none**.
The one-path14-gross change preserves declared order, top-level None omission,
complete metadata and failure behavior. Unsupported nested dataclass metadata
is the explicit producer-domain boundary. The report preserves the raw98PASS/
1baselineFAIL result and does not present local timing as a complete CLI result.

### Parent acceptance

Parent independently reran the six-file status comparison,598-event byte replay,
real-disk outputs, field/error/AST checks and two complete layer60 waves. All208
actual operator duration/proof pairs match in each wave, with baseline event bytes
unchanged. Source was merged at `64881373`, byte-identical to reviewed `9ae60ebc`.
`trace_parent_acceptance.json` records commands' artifacts, hashes, source binding
and Git-derived14/20 scope. Final union regression/CLI/native gates remain open.

## Prepared point selection component

Archive task: **D29-PREPARED-PARENT-ARCHIVE**. Component:
**D29-PREPARED-POINT-REUSE**. Base `caffccbb48011f0140359d391ad75b58906ff15b`,
code `0be11762d0f83e9db77658a1447a672507166872`, owner records
`b476907e9afe865476e0a3a6720f8eb4e3290c40`, integrated at
`eece87b2378bf86c6673b74f1a769f566bc5792a`.

Both independent reviewers inspected actual complete source and the three-path
diff, parent/owner requirements and plan, source/receipt lifetime and the actual
contract `prepared_point_repair/interface_receipt.md`. The component directory
has no `design.md`. Diff artifact:
`task_memory/task_2026-10-22_step5_position_routing/prepared_point_repair/prepared_point_repair.diff`.
Neither reviewer reran tests or hashes. The following axes archive the primary's
supplied results; this records-only task does not repeat or reinterpret the
earlier source reviews above.

### Standards

Reviewer `ses_e97039e03ffeu3ZRIC116k0GwS`: **No findings;0 findings,worst none**.

The private preparation in
`frontier/execution_time_predictor/moe_routing_workload.py:136–246` stays local
to the predictor, keyed by actual estimator, attached physical-row tuple,
measurement identity and exact current artifact text. It reuses the existing
query interface and replaces the old scan rather than adding a public API or
alternate implementation. No numerical/global/whole-proof cache is introduced.
Supported receipt lifetime/replacement follows the inspected construction and
fit paths; nested in-place row mutation is not a supported serving path or a
newly promised invalidation mechanism.

### Spec

Reviewer `ses_e97039de3ffe8fnhZlDO38w5lY`: **No findings;0 findings,worst none**.

Current group/family/registry/source/model/point selection stays live, as do
per-query file reads, fresh native contexts, exact duration/count lookup and
complete source proofs. Full plan keys, shared/compute domains, ambiguity,
buffer/context/policy and error checks remain. Replacement/error assertions at
`tests/unit/test_prediction_measurement_bindings.py:254–272` and
`tests/unit/test_moe_routing_runtime.py:79–104` cover the actual seam without a
second production preparation path. Failed preparation does not publish partial
retained state. Existing error and numerical rules are not weakened.

### Primary acceptance and evidence limits

Primary explicitly **ACCEPTED** this component after both independent axes and
its own verification, then merged owner record deliveryb476 at eece. Parent raw
artifacts show82 PASS/1 existing Torch SKIP/0 FAIL/ERROR and a separate2 PASS
both-trainer/preflight/independent-fit-cache selection. Keep those selections and
their provenance separate; neither is reviewer execution.

The parent b476 layer60 replay has one16464-row/168-key preparation,
cold3258/warm1968 decodes and80 current reads per wave. Both complete payloads
retain `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`.
The retained comparison checks208 captured/208 replay/all208 duration and five
complete proof-field pairs, with zero mismatches. Raw wave/comparison scope
literals still say layer61; the executed filter and current layer60 source,
records/counts/proof hashes establish the accurate complete captured scope.
Both raw artifacts remain unchanged, with this limitation disclosed in
`test_report_2026-10-23_prepared_point_repair.md` and accurate derived scope in
`prepared_parent_acceptance.json`. This is source-replay parity, not GPU truth.

Owner committed36-positive/24-negative matrix remains owner evidence, not a root
rerun or the parent's earlier prototype matrix. The executed prepared worktree
does not include trace390, so its pricing/approximately0.131s flush measurements
cannot establish integrated-union timing or be added to separate trace speedups.

This archive independently checks source bytes/Git accounting and reparses
existing receipts only: the three prepared files equal0be; integrated trace
equals390; gitlink3dac and all18 donor paths remain unchanged. Step159/250 and
cumulative197/323,299/299,75/78 pass their caps. Accepted79 audit plus the actual
four-path retained increase135 gives CORE4105/source9375/global9389; this is not
the pending final112-path union audit. Main plan/progress/summary remain root-owned.

Final union review/accounting/freeze, fresh strict77/full-unit, complete6144/1024
CLI within2400s, T2 final-source seal, native runtime/GPU/G3 and four-metric E2E
acceptance remain open for the new union. No successful check was rerun, source
edited, new agent spawned or network/platform/GPU operation performed here.

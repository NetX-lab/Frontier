# D29 CPU performance component review archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive separate Standards/Spec closure,group records P2 disposition and explicit primary component acceptance. |

Task ID: D29-PERFORMANCE-COMPONENT-ARCHIVE.
Component acceptance is complete for all three CPU repairs. This file archives
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

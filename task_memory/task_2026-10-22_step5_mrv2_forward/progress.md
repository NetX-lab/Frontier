# D23-MRV2-FORWARD Progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record initial implementation and focused green check. |
| 2026-10-22 | Record real lifecycle/cache validation and broadened legacy baseline. |
| 2026-10-22 | Complete CPU verification and code commit; archive handoff contract. |
| 2026-10-22 | Start reviewer P2 correction; record real first-event RED before code. |
| 2026-10-22 | Finish P2 verification:221 stage +268 measured PASS; prepare scoped commit. |
| 2026-10-22 | Record verified correction commit and completed scoped handoff. |
| 2026-10-22 | Inspect fixed-capacity follow-up; classify it as invariant improvement. |
| 2026-10-22 | Primary completed fixed-capacity verification after provider interruption:242 checks pass. |

- completed: new worktree from 5c2319cd; primary approval/source inspected.
- completed: pre-prediction hold in existing stage scheduler; generic readiness
  on existing stage context; same-time existing events resume once.
- completed: one graph policy table and capture resolver; FULL_AND_PIECEWISE,
  native min-mode and local NONE/shared graph widths. No timing constants.
- completed: metadata-bearing MRv2 dummy uses shared validated descriptor and
  kind identity; actual producer features match runtime dummy.
- completed: real four-layer P/D event lifecycle (stage sync, EP owner promotion,
  restore, terminal release); dummy has no serving IDs or layer credit.
- completed: empty lane wake through real V1 engine -> dummy arrival -> stage
  event; duplicate schedule/resume and group reset; dispatched shape survives
  request preemption. FULL prompt extension changes physical phase only.
- completed: admitted real/dummy batch -> actual build_row/rows_to_dataframe ->
  shared fit/cold/warm cache/bundle/root+count exact lookup. Bidirectional missing
  kind coverage rejects. FULL/PIECEWISE/NONE exercised.
- completed: original 5c2319cd stage event substituted in memory: 12 first-call
  regressions RED, missing required descriptor before prediction.
- completed: final focused 30 PASS, affected stage 176 PASS, measured 268 PASS,
  producer 467 PASS and DP-runtime/EP/placement 55 PASS.
- completed: code commit 3cad8dbe398304d798a323389f9e3d18500a42de.
- completed: test report and parent fresh-input contract in summary.md.

First focused run: test harness lacked its workload delegate (6 failures);
second exposed missing runtime MRv2 provenance before feature validation
(6 failures). Corrected the delegate and used shared MRV2_DUMMY_ROW_IDENTITY
in runtime feature construction. Third run: 84 passed in 0.23s.
Commands/logs: `/data/ycfeng/tmp/d23_forward_first{,02,03}.log`.

Shared manager/public fixtures originally supplied no P CUDA_EVENT attention
despite FULL/piecewise policy (7 failures). Corrected required source fixtures
with actual producer MRv2 dummy kinds; did not suppress file/source validation.
One fixture write initially reused attention.csv and replaced kernel-only rows;
isolated event source directories fixed that fixture defect. Updated contracts:
46 passed in 30.98s, `/data/ycfeng/tmp/d23_forward_updated_contracts02.log`.

Final producer: 467 passed. DP-runtime/EP/placement: 55 passed in 124.09s.
Broadened stage/selector: 204 passed, 7 inherited failures. Exact same seven
cases/error messages reproduced on immutable 5c2319cd inputs worktree, no edits:
3 spec fixtures lack num_processed_tokens, 4 manager fixtures lack attention
geometry. Original and candidate XML paths:
`/data/ycfeng/tmp/d23_forward_inherited_selector_baseline.xml`,
`/data/ycfeng/tmp/d23_forward_final_stage.xml`. Comparison printed
`baseline failures 7 candidate failures 7 same cases/errors True`.

Checkpoint after first green: mainline unchanged, D1-D7 CLEAR; S1-S6 no
defect found. Remaining frontier: real completion/wakeup and fit/cache/cost,
then affected regression and parent handoff. Parent-owned actual main/native
work is not pulled into this scope.

Checkpoint after executable lifecycle/cache milestone: plan items 1-2 PASS,
item 3 TODO. D1-D7 CLEAR, S1-S6 no defect. Only remaining frontier is final
affected verification, scoped commit and fresh input handoff. Inherited selector
fixture repairs are outside this scope and remain deferred to parent.

Final checkpoint: plan items 1-3 PASS, parent full CLI/strict77/native work
explicitly outside this handoff. D1-D7 CLEAR, S1-S6 no defect. All approved core
behavior has executable CPU evidence; no inherited selector test was patched.
Read requirements.md before closure. No scheduler approximation, second queue,
new event or synthetic timing constant was introduced. Input worktree status
remains clean. Final task record commit follows the verified code commit.

## P2 correction

- completed P2.1: initial worktree is clean on `feat/step5-mrv2-forward`, HEAD
  `d8e3e941`; no delegation or main/CLI fixture access. Native
  `gpu/cudagraph_utils.py:158` takes scheduler `max_num_seqs`; lines321-326 omit
  uniform FULL descriptors exceeding that capacity; lines346-365 allow wider
  PIECEWISE descriptors with `num_reqs=None`. V1
  `vllm_v1_engine_replica_scheduler.py:181-183` maps `batch_size_cap` to
  `_max_num_running_reqs`. All DP owners of the Replica share that configuration.
- completed P2.1: extend the existing actual V1 dummy wakeup -> batch arrival ->
  stage agreement -> first prediction test for cap3/captures[1,2,4,8], both roles
  and both affected policies. Before any behavior edit:4 failed,2 passed,
  29 deselected in12.54s. Exact assertion failure at test line273:
  `assert ['full', 'full'] == ['piecewise', 'piecewise']` (P and D), and
  `assert ['full', 'full'] == ['none', 'none']` (P and D). Passing controls use
  FULL1/1 within cap3. Raw log and XML:
  `/data/ycfeng/tmp/d23_forward_p2_red.{log,xml}`.
- completed P2.2: ranked diagnosis: (1) capacity absent from agreement,
  (2) unsupported FULL fails to try policy.other PIECEWISE, (3) final redispatch
  must use the same supported candidate selection. The graph policy now filters
  uniform FULL token/request captures using the existing V1 request capacity,
  then tries the existing policy.other family. Agreement receives that capacity
  from the group's V1 scheduler and reuses the resolver after DP min/max.
  The same real engine loop is GREEN:6 passed,29 deselected in11.98s,
  `/data/ycfeng/tmp/d23_forward_p2_green.{log,xml}`. Expanded real first-event
  matrix plus all earlier focused contracts:75 passed in14.85s,
  `/data/ycfeng/tmp/d23_forward_p2_focused.{log,xml}`. Both roles, both arrival
  orders, useful/useful and useful/dummy, capacity3/4, FULL<=cap, NONE local
  width, miss and legal prefill7->PIECEWISE8 are observed. Query/context arrays,
  request lists, owner lane/ticket, raw null fields and zero dummy IDs remain
  unchanged. Direct public predictor tests pass their existing scheduler config
  capacity explicitly; no CLI fixture is accessed or edited.
- completed P2.3 verification:221 stage PASS in18.75s and268 measured PASS
  in202.83s, run concurrently. Same affected suites as prior176+268 check with
  45 added real first-event cases. All GREEN XML reports zero errors/failures/
  skips; `git diff --check` passes. Logs/XML:
  `/data/ycfeng/tmp/d23_forward_p2_{stage,measured}.{log,xml}`. Exact commands,
  RED errors and environment are archived in the existing test report. No
  producer or inherited selector suite was rerun.
- completed P2.3 closure: requirements re-read; records updated; correction,
  regression tests and records committed as
  `c9e9388f117c625c6d546d1d083f8aaf3f824f51`. Post-commit
  `git status --short --branch` prints only `## feat/step5-mrv2-forward`.
  No unrelated file is committed. The task directory is ignored by default;
  explicit `git add -f` was used only for the approved task records after normal
  `git add` reported that ignore rule. Parent-owned re-review/merge/full CLI/
  strict77 remain outstanding. This completion receipt is a docs-only commit.

RED command (CWD is this worktree):

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_mrv2_forward.py -k empty_lane_wakeup_runs_real_v1_engine_dummy_and_stage_arrival -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_forward_p2_red.xml > /data/ycfeng/tmp/d23_forward_p2_red.log 2>&1
```

### P2 checkpoint after focused verification

1. Plan: P2.1 PASS (source mapping and exact RED), P2.2 PASS (capacity-aware
   resolution,75 focused PASS), P2.3 TODO (affected suites, records and commit).
2. Drift: D1-D7 CLEAR. Tests serve explicit P2 cases only; one diagnosis/fix,
   no side repair, no unapproved delegation, no producer/CLI/native work. S1-S6
   have no observed defect; dependencies match RED -> fix -> verification.
3. Route branch A: no scope correction. Run affected stage and measured suites
   concurrently, inspect XML/logs, complete records and commit. Parent re-review,
   merge, full CLI and strict77 remain outside this slice.
4. Attention: preserve scheduler cap rather than max capture width; leave legal
   PIECEWISE token captures unrestricted; do not claim CPU tests establish
   native qualification or full CLI acceptance.

Final pre-commit checkpoint: P2.1/P2.2 PASS; P2.3 verification PASS and commit
TODO. D1-D7 CLEAR, S1-S6 no defect; no scope drift or side work. Exact full
phase checkpoint is in `review.md` (planning only, not parent code re-review).
Next: commit the2 core modules,2 focused tests and this task's records, inspect
clean branch/commit, and return precise parent handoff. No expanded testing
or unrelated edits are needed after489 affected PASS.

Post-commit checkpoint: P2.1/P2.2/P2.3 PASS, no local blocker. D1-D7 CLEAR and
S1-S6 no defect. Only handoff response and this completion-receipt commit remain;
parent independent acceptance is not included in the completed local scope.

## Fixed capacity follow-up

- completed F1: native locked `vllm/v1/worker/gpu/cudagraph_utils.py:158`
  initializes graph request capacity from fixed `scheduler_config.max_num_seqs`;
  Frontier V1 constructor `vllm_v1_engine_replica_scheduler.py:181-183` maps the
  existing `_config.batch_size_cap` to `_max_num_running_reqs` initially. This
  config field is the native-matched graph creation capacity.
- completed F1 reachability: the only runtime reassignment found in scheduler
  source is `vllm_v1_iteration_policy.py:362-367`. Its profile defaults to
  config cap at335-346; hidden/final cap override at348-359 requires a non-null
  round classification. Classification at319-333 requires phase-aware policy
  and a waiting/preempted request with `is_thinking_mode_enabled=True`.
  Step thinking is rejected by `config.py:388,591-597` before simulation;
  `Request.is_thinking_mode_enabled` at402-403 requires depth>1 and the ordinary
  request generator sets depth1 when thinking is off (`base_request_generator.py:67-68,143`).
  Existing rejection test `test_step5_structural_pd.py:211-225` is GREEN for
  co-location and PDD:2 passed,18 deselected in1.90s, log/XML
  `/data/ycfeng/tmp/d23_forward_fixed_capacity_reachability.{log,xml}`.
- disposition: accept fixed config source as the graph creation invariant.
  No supported current Step workload counterexample was established; previous
  mutable field equals config under reachable ordinary requests. Do not claim
  a reproduced arrival-order bug or inject unsupported Step thinking/overrides.
- completed F2: preserved interrupted work and inspected actual diff/status.
  The supported-state baseline before the source substitution passed46 cases,
  with29 deselected in12.90s, log/XML
  `/data/ycfeng/tmp/d23_forward_fixed_capacity_supported_baseline.{log,xml}`.
  This is GREEN baseline evidence, not RED or a reachable bug reproduction.
  Substituted only agreement line109 with the existing `_config.batch_size_cap`.
  The focused harness now uses an actual shared `VllmV1SchedulerConfig`; both
  arrival orders assert configured and admission caps match in supported state.
  Real engine tests assert the ordinary profile round is null and its cap is3.
  No artificial thinking request, lane override, branch, cost or policy change.
- completed F3 verification: primary resumed the saved change after provider
  HTTP502, inspected the source/config/phase-classification path and ran the
  affected stage/engine, direct public predictor and complete Step structural-PD
  suites:242 passed in33.92s, zero failures/errors/skips. Log/XML:
  `/data/ycfeng/tmp/d23_fixed_parent_takeover01.{log,xml}`.
- completed F3 disposition: keep the one-line fixed config binding because it
  directly expresses native graph creation capacity. Supported Step requests
  cannot activate the thinking override; no reachable arrival-order defect was
  demonstrated. This is an invariant clarification, not a claimed bug reproduction.
- completed source integration: primary committed `ef476fe`, obtained final
  independent Standards/Spec No findings and merged at `32fb934`. All16 changed
  core/test files in the merged tree exactly match the reviewed source.
- completed parent legacy acceptance: clean strict77 baseline9dcd876 versus
  candidate32fb934 gives77/77 identical, zero failures/missing evidence/cache-name
  differences/provenance findings. Full comparison receipt is in the main task's
  `d23_forward_fidelity.json`.
- in-progress parent actual-main acceptance: `step5-mrv2-main` reuses existing
  fixture02. Full offline/online MRv2 completion is not yet claimed.

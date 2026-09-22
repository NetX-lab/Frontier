# Stage admission ordering under pipeline parallelism — Plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | Applied the round-1 plan review (`review.md`). Changes: C1 now targets confirmed admission-deadlock witnesses from a phase-controlled group; C3 uses the stage ledger and overlap duration; C4 takes the reviewer's wording; P0 lists its artifacts and outcome classes; P2 covers both sides of the EP boundary, the `DECODE_FFN` dense-group control, a second admission round and per-fixture base expectations, and the dense fixture asserts a same-start condition that discriminates on the base; P3 has three acceptance paths. The matrix now publishes a concrete case list on the analytical backend and keeps `attn_dp=2, PP=3`. Added D-6 and D-7. Not executed. |
| 2026-09-23 | D-1..D-5 adopted by the user; D-3 executed now (push + draft PR for remote review); D-5 adjusted so the records travel with the branch. Work packages P0–P4 unblocked. |
| 2026-09-22 | Created for user review. Scope, acceptance criteria, work packages P0–P4 with dependencies, verification matrix, decisions D-1..D-4. No source change yet. |

Diagnosis, options and the recommended rule are in `design.md`. Review
dispositions are in `review.md`. This file is the executable plan.

## 1. Scope

Fix the pre-existing stage admission deadlock (W9-01 in the parent task) on its
own branch:

- Branch `fix/stage-admission-ordering`, base `origin/main` `1f694f7`, worktree
  `/data/ycfeng/Frontier/.worktrees/stage-admission-ordering`.
- Source change confined to
  `frontier/scheduler/replica_stage_scheduler/stage_execution_context.py`:
  the full-stage admission predicate in `try_acquire`, removal of the admitted
  ticket with `remove(ticket)`, and the two docstrings that describe admission
  as "FIFO-head".
- Out of scope: `full_stage_capacity`, the forward-group seal, EP wave
  protocol, sync rooms, wake-up helper, any configuration field, the Step 9
  report key, and mixed-phase forward failures on `main` (PR 35 W3; see
  `design.md` "Scope boundary").

## 2. Acceptance criteria

| Id | Criterion | Settled by |
| --- | --- | --- |
| C1 | Repaired liveness (path L). Every G3a case that P0 classifies as `admission_deadlock` completes after P1, with request count, prefill tokens and decode tokens conserved. G3a is the phase-controlled, prefill-only group. P0 must find at least one such case for each of `(attn_dp, PP)` ∈ {2, 4} × {2, 3}; if a pair has none, stop and report before P1, because the case list does not exercise the defect there. | P0 classification; P2(c); P3 path L. |
| C2 | Unchanged controls (path U). Every run-to-run-stable metrics file is byte-identical before and after for G1 (all 30 release recipes, including the 10 PD-AF recipes), every `PP = 1` cell of G3a, G3b and G4, and every G5 cell. | P0 vs P3 `sha256sums.txt`. |
| C3 | Timing change (path T). Base-successful cases with `attn_dp > 1` and `PP > 1` (all G4 `PP > 1` cells, and G3a/G3b cells P0 classifies as `success`) are either byte-identical, or their difference is explained with the stage-ledger metric of §4.5 plus batch membership and component durations. The designated contention witnesses (§4.2, marked W) show a strictly larger `multi_lane_busy_time` after P1. In every case: no lane overlaps itself, and `peak_lanes ≤ attn_dp`. | P3, §4.5 metric from `frontier_stage_batch_ledger.jsonl`. |
| C4 | Existing passing tests must remain passing without assertion changes. Existing failures, collection errors, and skips must be compared against a fresh run of the exact base revision in the same environment. Any new failure or required change to an ordering assertion stops implementation for review. | G2, §4.6. |
| C5 | The predicate is one readable condition, and the module and method docstrings state the ordering contract as implemented. The change adds no flag, config field, `getattr` fallback, lane field on tickets, acquisition wake-up, PP-specific branch, second queue or capacity-1 special case. | Review of the diff against the quality gates. |
| C6 | Informational. The Step 9 boundary probe on MoE `attn_dp=2, moe_ep=2, PP=2` runs to completion on this branch, or its remaining failure is classified. Composition with PR 35 (W3 mixed-phase forward) is validated in the parent task after merge-forward, before Step 9 is declared unblocked. | P3 probe rerun; parent-task follow-up (§6). |

## 3. Work packages

```text
P0 evidence and baseline (no source change)
    -> P1 rule change
    -> {P2 tests, P3 rerun and comparison}
    -> P4 records, commit, push
```

| Package | Content | Acceptance |
| --- | --- | --- |
| P0 Evidence and baseline | (1) Move the reproduction into `tests/e2e/stage_admission_matrix.py`, following the `tests/e2e/moe_ep_non_dummy_matrix.py` precedent. The module holds: the §4.1 fixture builder; the §4.2 case table; a runner with one child process per case, because `IS_MOE` is process-global; the outcome classifier and state-report writer of §4.3, which replace the session scripts `drain_state.py`/`drain_lanes.py`; and the §4.5 ledger metric. Outputs go to `resolve_scratch_root()/stage_admission_ordering/base/<case_id>/` (`tests/scratch_root.py`). (2) Confirm that the branch source equals `1f694f7` (`git diff --stat 1f694f7 -- . ':!task_memory' ':!.gitignore'` is empty). (3) Run R0, G1, G3a, G3b, G4, G5 and the G2 suites. (4) Write the §4.3 artifacts for every case. (5) Rerun two success cases (one G1 recipe, one G4 `PP=2` cell); an unstable file is named and excluded from C2, with the reason. (6) Record the classification table in the test report. | Every case has `case.json`, `run.json` and its class artifact. R0 reproduces the author-reported table, or each difference is explained. No `other_failure`. C1's per-pair witness condition holds. Every successful `attn_dp>1` case has `ATTN_DP_LANE` ledger rows for each of its lanes, otherwise §4.5 cannot be computed and P0 stops. |
| P1 Rule | Implement the `design.md` rule. Full-stage tickets are refused only by an EP wave queued ahead; EP waves keep the strict head rule; the admitted ticket leaves the FIFO by `remove(ticket)`. Update the `StageExecutionContext` class docstring ("A complete operation first enters the ready FIFO, then the owner admits it atomically") and the `try_acquire` docstring ("Acquire the FIFO-head ticket if this stage is currently idle") to state the implemented contract. Queued full-stage work may pass other full-stage work but not an earlier queued EP wave. Queued EP waves keep FIFO admission. Active layer-to-layer scope transitions remain a separate mechanism. | The diff touches one source file. `python -m pytest tests/unit/test_stage_execution_context.py tests/unit/test_shared_forward_group_admission.py -q` passes with no assertion change. |
| P2 Tests | **(a)** Contract tests in `tests/unit/test_stage_execution_context.py`. *Bypass* and *EP boundary* use a capacity-2 context (`ep_size=2`) with FIFO `full0, full1, wave0, full2`. *Bypass*: `full1` acquires before `full0`, and afterwards `queued_tickets == (full0, wave0, full2)`; `full2` is then refused although capacity remains, because `wave0` is ahead; `full0` acquires. *EP boundary* (acquisitions in head order, so it also runs on the base): `full0` and `full1` acquire; after `full1` releases, `full2` is still refused, because `wave0` is ahead; `wave0` is refused while `full0` is active and acquires once it releases; `full2` is refused while `wave0` is active and acquires after `wave0` releases. *Capacity 1*: on an idle context with FIFO `[full0, full1]`, `try_acquire(full1)` succeeds, pinning the API-level change stated in `design.md`. The two existing EP-order tests stay unchanged. **(a′)** A `DECODE_FFN` control in `tests/unit/test_mixed_layer_decode_ffn_scheduling.py` with its mixed-layer fixture. It materializes two successive `DenseFFNBatchGroup`s and a neighbouring EP group on one target replica and stage, through `_schedule_dense_ffn_from_m2n_group` and the real full-stage `ReplicaStageScheduler`. It asserts that FIFO order and heap order both follow the group counter, that the dense groups are admitted in counter order, and that neither dense group crosses an EP wave queued ahead of it. **(b)** A scheduler-level test in `tests/unit/test_shared_forward_group_admission.py` using its `make_stage`/`make_batch` helpers, parametrized over which lane enqueues first. It rebuilds the drain state: the first lane is active with a second ticket queued, and the other lane has two queued tickets. It asserts that the other lane's `pop_batch_if_not_busy` returns its heap head and binds the same forward group. It then continues through promotion to an EP wave and restoration to full-stage owners (`replace_full_stage_owners_with_ep_wave`, `replace_ep_wave_with_full_stage_owners`), release of both owners and `on_stage_end` of both lanes, and it asserts that both lanes admit their next queued batch into a later forward group, leaving the FIFO empty. **(c)** Simulator-level tests in `tests/integration/test_stage_admission_pipeline_lanes.py`, importing the §4.1 builder from `tests.e2e.stage_admission_matrix`, one child process per case. MoE witnesses `G3a-moe-dp2-pp2-n4` and `G3a-moe-dp4-pp2-n8` assert completion and conservation. The dense fixture `G4-dense-dp2-pp2-n8` asserts completion, and that the first stage-0 ledger rows of both lanes start at the same simulated time, because every request arrives at `t=0` and capacity admits both lanes into the first forward. A bare `multi_lane_busy_time > 0` would not discriminate: from source, the base already overlaps the lanes after the first release. Expected values are written from the scenario, not copied from a run. | Expected on `1f694f7`: (a) *bypass* fails at its first assertion, *capacity 1* fails, and *EP boundary* passes; (a′) passes; (b) fails at the other lane's first admission; (c) each MoE witness fails through the documented `admission_deadlock` signature, and the dense fixture completes but fails only its same-start assertion, because the second lane's first row starts at the first lane's first stage-0 end. After P1 all of them pass. The base failures are recorded as negative controls. |
| P3 Rerun | Rerun every P0 case on the P1 revision into `.../after/<case_id>/`, then apply the acceptance path of each case (§4.2). **U**: hashes identical. **L**: the case completes with conservation; there is no base metrics hash to compare. **T**: hashes identical, or the difference is explained by the §4.5 metric (before and after), batch membership and component durations; W cases must show a strict increase. Stop and report, adjusting nothing, on any of these: a U difference (including `attn_dp=4, PP=1`); an L case that fails in any other way, such as a mixed-phase failure in G3b; a T difference that the ledger does not explain; a class change outside these paths; a failure of the self-overlap or `peak_lanes ≤ attn_dp` checks. Rerun the Step 9 boundary probe for C6. | C1–C4 and C6 tables in `test_report_<date>_stage_admission_ordering.md`. |
| P4 Records | Test report, `progress.md`, `summary.md`. Commit P0's harness, P1 and P2 as code commits (harness separately from the rule, so the rule commit stays one file plus its tests), and the records as a docs commit. Push the branch and update the draft PR body with the C1–C3 tables. Note in the parent task (`issues.md` W9-01) the branch and commits. | Pushed and verified. |

## 4. Verification matrix

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (version recorded
in `run.json`), `PYTHONPATH` = the worktree, `WANDB_DISABLED=true`,
`VIDUR_DISABLE_WANDB=1`. Each Simulator run is a fresh process. Before P1 the
source tree is the base: P0 step (2) confirms that it equals `1f694f7`.

### 4.1 Common fixture for the synthetic groups (R0, G3a, G3b, G4, G5)

| Field | Value |
| --- | --- |
| Model | `num_layers=6` (divisible by PP 1, 2, 3), `num_q_heads=4`, `num_kv_heads=2`, `embedding_dim=256`, `mlp_hidden_dim=64`, `max_position_embeddings=4096`, `use_gated_mlp=True`, `use_bias=False`, `use_qkv_bias=False`, SiLU, RMSNorm, `post_attn_norm=True`, `vocab_size=1024`, `torch_dtype="bfloat16"`. MoE: `is_moe=True`, `num_experts=8`, `num_experts_per_tok=2`. Dense: `is_moe=False`. Injected by monkeypatching `BaseModelConfig.create_from_name`, as `tests/integration/test_pr33_nondummy_acceptance.py:170` does. |
| Replica | `device="a100"`, `network_device="a100_pairwise_nvlink"` (4 devices per node), `attn_tensor_parallel_size=1`, `attn_dp` and `num_pipeline_stages` per case, `memory_margin_fraction=0.1`. MoE: `moe_tensor_parallel_size=1`, `moe_expert_parallel_size=attn_dp`, `total_expert_num=8`, `router_topk=2`. |
| Replica scheduler | `VllmV1SchedulerConfig(num_blocks=128, block_size=16, batch_size_cap=4, max_tokens_in_batch=16, enable_chunked_prefill=True)`. With 16-token prompts each prefill batch carries one request. |
| Cluster scheduler, predictor | `RoundRobinClusterSchedulerConfig()`; `RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True)`. |
| Simulation | `simulation_mode="offline"`, `sys_arch="co-location"`, `enable_parallel_clusters=False`, `decode_cuda_graph_mode="none"`. |
| CC backend | `ClusterConfig.cc_backend_config = AnalyticalCCBackendConfig()` for G3a, G3b, G4 and G5, the public examples' choice (D-6). `analytical` applies no Replica-pod node-size rule, so `attn_dp=2, PP=3` (6 devices) is constructible; P0 confirms this, and a rejection is classified, not substituted. R0 leaves the default (`astra_sim_analytical`, `config.py:2494`) to reproduce the recorded evidence. |
| Arrivals | G3a, G3b, G4, G5: `StaticRequestIntervalGeneratorConfig()` (every request at `t=0`). R0: `PoissonRequestIntervalGeneratorConfig(qps=1e6)`. |
| Lengths | `FixedRequestLengthGeneratorConfig`. Profile **PF** (prefill-only): `prefill_tokens=16, decode_tokens=1`. Profile **PD**: `prefill_tokens=16, decode_tokens=3`. |
| Metrics | `write_metrics=True` (required: the stage ledger is written by `plot()`, which runs only with `write_metrics`), `store_request_metrics=True`, `store_plots=False`, `enable_chrome_trace=False`, `write_json_trace=False`; `store_frontier_stage_batch_ledger` left at its default `True`; `output_dir` = the case directory. |

G1 recipes run unchanged with `PYTHON_BIN`, `METRICS_OUTPUT_DIR` and `RUN_ID`
overridden per case (each recipe reads these variables).

### 4.2 Case list

Case id: `<group>-<moe|dense>-dp<attn_dp>-pp<PP>-n<requests>`. "Base
hypothesis" is the expectation before P0: `PP=1` succeeds; at `PP>1`,
`n ≥ 2·attn_dp` (every lane holds two or more batches at stage 0)
deadlocks for MoE; otherwise the case succeeds. **P0's classification, not the
hypothesis, assigns each case its path.** Class `admission_deadlock` → path
L; class `success` at `attn_dp>1, PP>1` → path T; `PP=1` or `attn_dp=1` →
path U; any other class in P0 is handled by §4.3.

| Group | Cases | Profile | Count | Base hypothesis | Path |
| --- | --- | --- | --- | --- | --- |
| R0 record | The 16 shapes of the `design.md` table: 15 author-run logs plus the MoE `dp2-pp3` rejection | PD, Poisson | 16 | As recorded in `design.md` | Evidence only. After P1: informational, classified the same way. |
| G1 release | The 30 `examples/architecture/{co-location,pdd,pd-af-disagg}/{offline,online}/*.sh` recipes (all `PP=1`) | recipe | 30 | success | U |
| G3a MoE, phase-controlled | `attn_dp ∈ {2,4}` × `PP ∈ {1,2,3}` × `n ∈ {4, 8, 12}` | PF | 18 | `PP=1`: success. `PP>1`: deadlock, except `dp4-n4`, which is success (one batch per lane). | U / L / T |
| G3b MoE, standard lengths | `attn_dp ∈ {2,4}` × `PP ∈ {1,2,3}` × `n ∈ {4, 8}` | PD | 12 | as G3a | U / L / T. A mixed-phase failure after P1 stops and is reported (scope boundary). |
| G4 dense | `attn_dp ∈ {2,4}` × `PP ∈ {1,2,3}` × `n ∈ {4, 8}` | PD | 12 | success; at `PP>1` the second lane's first admission waits for the first lane's first stage-0 release | `PP=1`: U. `PP>1`: T. Contention witnesses **W**: `n=8` at `PP ∈ {2,3}`, `attn_dp ∈ {2,4}` (4 cases). |
| G5 single lane | `attn_dp=1`, `PP ∈ {1,2,3}`, MoE (`moe_ep=1`) and dense, `n=6` | PD | 6 | success | U |
| G6 PD-AF | The 10 PD-AF recipes inside G1, plus `tests/unit/test_mixed_layer_decode_ffn_scheduling.py`, `test_decode_ep_wave_materialization.py` and `test_prefill_ep_wave_materialization.py` | — | (in G1) | success; tests pass | U, C4 |
| G2 suites | `tests/unit` and `tests/integration` | — | 2 | base identities from P0 | C4 |

Simulator runs: 30 + 18 + 12 + 12 + 6 = 78, plus 16 R0 record runs and the two
pytest suites. This satisfies the AGENTS.md gate of at least 50 concrete
settings.

### 4.3 Outcome classes and artifacts

| Class | Definition |
| --- | --- |
| `success` | `Simulator.run()` returns and every request is completed. |
| `admission_deadlock` | The run ends with "Sequential simulation ended with non-empty scheduler state", and the state report read from the live simulator objects shows this signature: on some stage context, active full-stage owners are fewer than capacity, no EP wave is active, and the bound forward group is unsealed; the FIFO head is a full-stage ticket whose lane is busy; another lane is idle, with a non-empty heap whose head ticket is queued behind that head; and a sync room of the bound group lists the busy lane and waits for the idle one. |
| `configuration_rejection` | A `ValueError` from configuration or topology validation, such as the Replica-pod node-size rule, wherever it surfaces. |
| `other_failure` | Anything else, including a drain without the signature. In P0 this stops the work before P1. |

Every case directory holds:

- `case.json`: the case id, group, every fixture field of §4.1, and the
  resolved `SimulationConfig` as JSON.
- `run.json`: the command line; interpreter path and `python -VV`; a digest of
  `pip freeze`; `git rev-parse HEAD`; whether `git status --porcelain` is clean
  outside `task_memory/`; start and end wall time; outcome class; exception
  type and message.
- One class artifact:
  - `success`: `sha256sums.txt` over every file in the metrics directory.
  - `admission_deadlock`: `state_report.json`, with per context the capacity,
    sealed flag, bound group, active owners, and the FIFO mapped ticket → lane
    → batch id → `global_id`; per lane the busy flag and heap; the sync rooms
    with the lanes present; and the simulated time. No metrics hash.
  - The two failure classes: `error.txt` with the traceback.

`cases.jsonl` indexes the cases, one line each. Large outputs stay under the
scratch root; the test report keeps the classification table and the metrics.

### 4.4 Acceptance paths

| Path | Applies to | Pass condition |
| --- | --- | --- |
| U unchanged | G1, G5, every `PP=1` cell, G6 recipes | Identical `sha256sums.txt` (run-to-run-unstable files excluded by P0 with a reason). |
| L repaired liveness | Cases P0 classifies as `admission_deadlock` | `success` after P1; completed requests = generated requests; the sums of prefill and decode tokens over `request_metrics.csv` equal the generated lengths. |
| T timing | Cases P0 classifies as `success` with `attn_dp>1, PP>1` | Identical hashes, or a ledger-explained difference (§4.5). W cases: strictly larger `multi_lane_busy_time`. |

### 4.5 Lane-overlap metric (C3)

Source: `frontier_stage_batch_ledger.jsonl` in the case's metrics directory.
Its rows carry `cluster_type`, `replica_id`, `stage_id`, `execution_scope`,
`replica_local_id`, `stage_start_ts` and `stage_end_ts`
(`metrics_store.py:4401-4422`). No production metrics change.

For each physical stage `(cluster_type, replica_id, stage_id)`, take the rows
with `execution_scope == "ATTN_DP_LANE"` as half-open intervals
`[stage_start_ts, stage_end_ts)`, keyed by `replica_local_id`.

| Output | Definition |
| --- | --- |
| `multi_lane_busy_time` | Total simulated time during which at least two distinct lanes have an open interval. Touching endpoints overlap for zero time; zero-length rows contribute nothing. |
| `peak_lanes` | The largest number of distinct lanes open at one instant. |
| `makespan` | The largest `stage_end_ts` in the ledger. |
| Checks | No lane's intervals overlap one another. `peak_lanes ≤ attn_dp`. |

A count of overlapping intervals is not used, because it depends on how
intervals are partitioned. For a T difference, the report pairs the metric
before and after with the batch membership (`request_ids` per row) and the
component durations (`execution_time`) of the rows that moved; a changed
aggregate latency alone does not establish the cause.

### 4.6 Test-identity comparison (G2, C4)

For `tests/unit` and `tests/integration` separately, run
`python -m pytest <suite> -q -p no:cacheprovider --continue-on-collection-errors --junitxml=<out>/<suite>.xml`
on the base source (P0) and on P1, in the same environment. Compare the node
id → outcome maps (passed, failed, error, skipped) and the collection errors
per module. Pass: every base-passed node id still passes; no node id newly
fails or errors; collection errors and skips are unchanged. A base failure that
now passes is reported, not treated as a stop. No failure count from another
checkpoint is used.

## 5. Decisions

Adopted by the user on 2026-09-23 ("采纳你d1-d5的推荐决策"):

| Id | Question | Recommendation | Outcome |
| --- | --- | --- | --- |
| D-1 | Adopt option B from `design.md` (full-stage tickets are ordered only behind EP waves) rather than option A (lane-aware skip) or C/D. | B. A adds lane identity and a dependency on peer acquisition that no wake covers; C and D are rejected on the working gates. | Adopted. |
| D-2 | Accept that dense `attn_dp>1, PP>1` timelines change (lanes overlap instead of serializing). | Accept as a fidelity fix; the serialization is the same defect. | Adopted. |
| D-3 | Authorize pushing `fix/stage-admission-ordering` and opening a draft PR against `main`. | Grant at P4; until then everything stays local. | Adopted and brought forward: the user reviews on the remote, so the branch is pushed and a draft PR opened with the plan itself (2026-09-23). Code commits follow per package. |
| D-4 | Baseline for byte comparison is `origin/main` `1f694f7`. PR 35 will merge this branch later instead of carrying the fix itself. | Confirm. | Adopted. |
| D-5 | `task_memory/` is ignored by `.gitignore` on `main` (line 171), so these records are local to the worktree unless force-added. Keep them local and archive the outcome in the parent task, or track them on this branch as PR 35 does? | Keep local; copy `summary.md` and the test report into the parent task at P4. | Adopted with one adjustment required by D-3: remote review needs the records on the branch, so `.gitignore` gets the same narrow exception PR 34/35 use (`task_memory/*` plus `!task_memory/task_2026-09-22_stage_admission_ordering/`). The copy into the parent task at P4 stands. |

Adopted from the round-1 review on 2026-09-23, under the user's instruction
"采纳高价值和必要决策" (`review.md`):

| Id | Decision | Reason |
| --- | --- | --- |
| D-6 | The synthetic groups select `AnalyticalCCBackendConfig` explicitly and keep `attn_dp=2, PP=3`, instead of substituting `attn_dp=4, PP=3`. | The change is admission-only. The node-size rule belongs to the `collective_sim`/`astra_sim_analytical` backends, and substituting the shape would leave part of C1 untested. |
| D-7 | C1 witnesses come from the phase-controlled prefill-only group G3a. Mixed-phase failures are out of scope: stop, report, diagnose separately. Composition with PR 35 is checked in the parent task. | `main` lacks PR 35 W3; this keeps the admission repair separable from the mixed-phase lifecycle. |

## 6. Dependencies and risks

- The reproduction scripts still live in the session scratchpad
  (`w10/probe_main.py`, `w10/repro_main.py`, `w10/drain_state.py`,
  `w10/drain_lanes.py`); P0 replaces them with `tests/e2e/stage_admission_matrix.py`.
- Risk: the `PP=1, attn_dp=4` cells could differ through the wake-order
  inversion described in `design.md`. That is a stop-and-report condition,
  not an automatic acceptance.
- Risk: a `DECODE_FFN` or `DECODE_ATTN` path that queues full-stage tickets from
  two schedulers on one context would see admission order change. `design.md`
  gives the caller-level reason this is not expected; P2(a′) and G6 measure it.
  If either differs, stop and report before adjusting anything.
- Risk: after P1, a G3b case may reach a mixed-phase cohort and fail (scope
  boundary). Stop and report; it is not repaired on this branch.
- The parent task's Step 9 resumes only after this branch is merged into `main`
  and merged forward into `fix/issue26-correctness-pr`. The parent task then
  reruns G3b on that branch, where W3 is present, as the composition check
  (C6).

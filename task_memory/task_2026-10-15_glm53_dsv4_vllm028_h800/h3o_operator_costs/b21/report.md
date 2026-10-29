# B21: A11 forward census and time comparison

Date: October 29, 2026. Outcome: **implemented**. The task-local observer, comparison, and Acceptance 1–6 checks are complete. Final review and acceptance remain with Claude. These are observations of the eight-request P2-32 scenario; the full P2-32/P2-64 A10 replays remain Claude's subsequent work.

## 1. Assignment, scope, and decisions

Working root: `/data/ycfeng/Frontier/.worktrees/b21-v4-census`.
Task prefix used below: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/`.
Allowed writes: that prefix's `a10/` and `b21/`, `tests/analysis/v4_replay_census.py`, `tests/unit/test_v4_replay_census.py`, and `tests/unit/test_dsv4_a10_tools.py`, plus private scratch storage.

The implementation follows the supplied B21 assignment and H3o design §1 (A11/A13), §2, §3.7, and entry 18. The worktree and parent AGENTS.md instructions apply. Existing A10 configuration/input ownership, R1's native classifier and trace reducer, and B19's measured collective admission are reused. The read-only reference evidence includes `evidence/routing/mode_census.py`, `review_counter_checks.py`, `mode_census.txt`, `evidence/runtime_census/p2_32.txt`, `p2_64.txt` and their logs, the A10 tools, and the named event/batch/DP synchronization sources.

Settled behavior: A11 is diagnostic, never a cost or accuracy gate; the observer is attached inside the A10 driver process; no source under `frontier/` is edited; failed collective requests preserve their exception; all exposed scope costs remain visible. No predictor, profiling table, scheduler, simulator configuration field, dependency, native input, GPU operation, commit, deployment, or lead task record was changed. No delegation was used. The existing A10 driver still performs its read-only source-identity queries for the receipt.

Both completed runtime receipts identify the B19-inclusive source HEAD `a27d3adeaf5ee929be48b645c10b99f29c054ac7` and have empty dirty-path lists under `frontier/` and `data/`. Existing base work and the task-local implementation inherited on resumption were preserved. `a10_common.py`, `compare_cache_runs.py`, and `v4_runtime_census.py` are reused without changes. B20's concurrent predictor changes were not present in these runs.

A material evidence limitation was found: native dummy counters have no phase, some first-tail counters lack the context/generation split, and R1 has no serving-phase label. The comparison preserves explicit unknowns rather than inventing those attributes. Section 4 gives the evidence and recommended next action.

## 2. Changes and observation contract

Every changed repository file is listed here. Paths beginning `a10/` or `b21/` are relative to the task prefix above.

| File | Status | Relevant implementation and reason |
| --- | --- | --- |
| [a10/a10_driver.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py:115) | Modified | `main`, `build_config` and CLI: attach observer by default, optional `--without-observer`/`--trace`, existing-field overrides, trace/A11 receipts on success and failure. |
| [a10/a10_observer.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_observer.py:98) | New | `ForwardObserver`, `batch_shape`, `frontier_step_class`, `collective_request`: per-lane forward snapshots, scope attribution and measured collective request inventory. |
| [tests/analysis/v4_replay_census.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/tests/analysis/v4_replay_census.py:226) | New | `read_counters`, `runtime_records`, `compare_run`, `summary`: all-node weighted census, native R1 time comparison, unknowns, exact/interpolated shares and JSON/text CLI. |
| [tests/unit/test_v4_replay_census.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/tests/unit/test_v4_replay_census.py:30) | New | Step-shape, counter/phase weighting, arithmetic, observer identity/exception, scope mapping and actual MoE dispatch tests. |
| [tests/unit/test_dsv4_a10_tools.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/tests/unit/test_dsv4_a10_tools.py:176) | Modified | Parameterize driver receipt checks for observer on/off and successful/failed runs. |
| [b21/generate_trace.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/generate_trace.py:8) | New | Generate the specified eight-request CSV into each run root and record its SHA-256. |
| [b21/run_small.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/run_small.py:15) | New | P2-32 cold-cache replay; independent successful collective and stage-end call counters in both runs. |
| [b21/check_identity.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/check_identity.py:14) | New | Exact parsed-artifact/cache-name checks, row uniqueness/content, scope presence, hashes and independent counters; durable result. |
| [b21/run_identity.sh](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/run_identity.sh:1) | New | Serial cold-cache pair reproducer with required Python/root, memory cap and timeouts. |
| [b21/run_comparison.sh](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/run_comparison.sh:1) | New | Produce one diagnostic JSON/text pair from a completed observed replay and matching native artifacts. |
| [b21/check_comparison.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/check_comparison.py:8) | New | Independently sum source counter rows and validate all comparison counts/window/output presence. |
| [b21/identity.json](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/identity.json:1) | New | Final corrected run: all 17 checks, wall times, cache counts and forward census. |
| [b21/comparison_validation.json](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/comparison_validation.json:1) | New | Final corrected small-run comparison: all six checks and independently summed native totals. |
| [b21/report.md](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/report.md:1) | New | Commands, environment, actual results, observation/segment contract, limitations and review handoff. |

The observer wraps these existing methods only in the driver process and restores the originals when its context exits:

1. `BatchStageEndEvent.__init__`: snapshots final graph descriptor, local shape/class, synchronization state, predicted stage time, and timestamps before request mutation. `BatchStageEndEvent.handle_event`: writes the snapshot after the original handler completes; stale events are counted separately.
2. `dp_dummy_forward._synchronize_dp_descriptors`: reads local classifier inputs/desire before the original synchronization, then retains the final width vector, peer modes, and peer classes.
3. `SklearnMoEExecutionTimePredictor.predict_stage_execution_time`: retains the existing single-layer source-attention result when `include_ffn=False`, once per physical layer. It does not request another prediction.
4. `BaseClusterScheduler._prepare_moe_ep_wave_plan`: keeps the owning source-batch context while the existing plan is prepared. Static `_get_shared_ep_phase_times_ms`: reads its already computed TP-rank-zero EP scopes, mapped to each DP source lane.
5. `MeasuredCCBackend.predict_operator_collective`: aggregates only after the original call succeeds. A failing original call creates no row and re-raises unchanged.

No hook is inside B20's edited modules. Observer dictionaries and output files are its own state; it does not modify requests, batches, descriptors, predictor inputs, or predictor caches.

Each completed lane-forward row includes role, replica, DP lane, stage/batch/epoch identity, start/end simulated seconds, real/dummy, local tokens/requests, V2 classifier inputs (`has_prefill`, maximum query length, uniform token count), desired and final mode, descriptor, synchronized width/vector, peer modes/classes, native step class, composed milliseconds, all exposed scope milliseconds, and segment milliseconds. Step classes call R1's `step_class`; the local shape uses `dsv4_request_is_decoding`, which correctly treats the first DECODE tail as V2-prefilling.

`collectives.json` aggregates successful API requests by role, curve key, requested bytes, and unequal ordered rows (or null). Each group records call count, predicted milliseconds/total, exact versus interpolated status, measured lower/upper bytes and times, and `log(bytes/lower)/log(upper/lower)` for interpolation. Exact requests use identical lower/upper points and a null ratio; admitted unequal combine coordinates are exact only. These are prediction API calls, including repeated estimator/metrics calls, rather than a count of executed collectives.

The A10 receipt stores both observer paths and SHA-256 values, forward rows/stage ends/stale ends, collective calls, trace identity, and scope mapping. The comparison rejects unfinished receipts and verifies observer hashes and row count.

### Scope-to-segment mapping

`composed_ms` is the stage's existing predicted elapsed duration. Scope values include their exposed operator gaps and are kept in `scope_ms` without distributing a composite over its children.

| Segment/column | Mapping |
| --- | --- |
| `prepare` | `dsv4_input_preparation` |
| `post` | `dsv4_lm_head`, `dsv4_logits_gather`, `dsv4_sampling` |
| `dispatch` | Registered DeepSeek V4 attention and mHC scopes, MoE/shared-expert scopes, and V4 communication scopes, excluding the three post scopes. This includes embedding, mHC head, final norm and final mHC gather. The exact scope-name list is in the receipt and both comparison outputs. |
| `kv_cache_block_zero`, `dsv4_block_table_write` | Each is its own column; native R1 assigns the following forward's pre-DP work to the previous forward's post. No cross-forward shift is inferred. |
| Unknown scope | Retained as a column with its own scope name. |
| `coordination_and_overlap` | Signed composed stage time minus mapped segments and own columns, preserving DP waiting, EP rank/barrier differences and overlap. This is an attribution remainder, never a new cost. |

Within attention, the existing `_dsv4_attention_schedule.elapsed_ms` critical path replaces serial scope attribution for the dispatch subtotal; individual scope values stay intact. EP attribution is TP rank zero per DP lane and is not an exact reconstruction of the EP critical path. Every row's mapped segments plus own columns reconcile to its composed duration.

Whole native forward time uses R1 `device.span_us`; native segment time uses R1 `device_union_us`, converted to milliseconds. Segment unions need not sum to the whole span. Every statistic carries its actual contributing sample count. Relative error is `abs(Frontier-native)/abs(native)` and is null at native zero or when either statistic is unavailable.

## 3. Verification, commands, and actual results

Python: `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python`, **3.10.6**, GCC 11.3.0. Linux x86_64, 64 logical CPUs. NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3, pytest 9.1.1. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` were unset. No dependencies were installed.

Private scratch: `/data/ycfeng/tmp/codex-implement/fcf9eca4-0142-4d72-a3ea-443401fae6fe`. All simulations ran with `ulimit -v 62914560` and `timeout 7200`; checks used `timeout 600`. Bytecode, pytest temp storage, predictor caches, run roots, native reductions, logs, and comparison outputs stayed in private scratch. The commands below use these aliases from the working root:

```bash
PYTHON=/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python
TASK=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs
SCRATCH=/data/ycfeng/tmp/codex-implement/fcf9eca4-0142-4d72-a3ea-443401fae6fe
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export TMPDIR="$SCRATCH/tmp"
ulimit -v 62914560
```

| Acceptance | Expected | Observed | Result |
| --- | --- | --- | --- |
| 1. Unit checks | Five requested shapes match R1; weighted all-node counter reading and phase separation; comparison arithmetic with undefined zero relative error | These tests plus observer snapshots, composition, interpolation/unequal coordinates, exception/restoration, unfinished receipts and real MoE dispatch coverage passed in the final required regression suite | **PASS** |
| 2. Identity | Eight-request trace, two cold caches; zero parsed-artifact differences; zero cache-name differences; observer run within 2% or both values reported | Zero artifact differences; zero name differences; 1,758 cache files each. Plain 1,140.308637 s; observed 1,137.052128 s; change −0.285581% | **PASS**, with timing caveat below |
| 3. Record content | Both roles and both lanes; PIECEWISE tail, FULL multi-request decode, dummy, NONE prefill; one row per stage end; independent collective counts agree | 320 rows = 320 unique forwards = 320 independently counted stage ends; 0 stale ends. All requested mode/class examples present. 267,170 successful collective calls agree per role in plain, observed and independent counters | **PASS** |
| 4. Small comparison | JSON/text exist; native counts equal source `successful_forwards` sums from all nodes | Four nodes, 67,600 observations; formal 36,108, phase-unassigned 20,750, excluded 10,742. R1 window 408. All six independent validation checks passed | **PASS**, explicit native unknowns retained |
| 5. Regression | Four requested unit files pass; dummy smoke matrix 12/12 with `PYTHON_BIN` set | **74 passed in 4.20 s**; **12/12** architecture smokes completed | **PASS** |
| 6. Records/size | Trace generator, run/comparison scripts, identity JSON, exact-command report, touched Python files ≤2,000 lines | All present in `b21/`; nine touched Python files parse successfully; largest is 302 lines; both shell scripts pass `bash -n` | **PASS** |

### Executed regression and smoke commands

```bash
timeout 600 "$PYTHON" -m pytest \
 tests/unit/test_dsv4_a10_tools.py tests/unit/test_dsv4_graph_dispatch.py \
 tests/unit/test_dsv4_dp_sync.py tests/unit/test_v4_replay_census.py \
 -q -p no:cacheprovider --basetemp="$SCRATCH/pytest_source_scope_fixed"

PYTHON_BIN="$PYTHON" METRICS_OUTPUT_DIR="$SCRATCH/dummy/metrics" timeout 600 \
 bash examples/architecture/run_dummy_smoke_matrix.sh -- \
 --metrics_config_cache_dir "$SCRATCH/dummy/cache"
```

Observed logs: `$SCRATCH/unit_source_scope_fixed.log` (74 passed in 4.20 s) and `$SCRATCH/dummy.log` (`Completed 12 profiling-independent dummy smoke case(s).`). The smoke covers dense/MoE, offline/online, co-location/PDD/PD-AF.

### Executed cold-cache runs and identity check

```bash
timeout 7200 "$PYTHON" "$TASK/b21/run_small.py" "$SCRATCH/small_plain" --without-observer
timeout 7200 "$PYTHON" "$TASK/b21/run_small.py" "$SCRATCH/small_observed_complete"
timeout 600 "$PYTHON" "$TASK/b21/check_identity.py" \
 "$SCRATCH/small_plain" "$SCRATCH/small_observed_complete" "$TASK/b21/identity.json"
```

Run logs: `$SCRATCH/plain.log`, `$SCRATCH/observed_complete.log`; identity log: `$SCRATCH/identity_complete.log`. The trace generator was invoked by each `run_small.py`: eight arrivals at 0.00, 0.05, …, 0.35 s, each 6,144 prompt tokens and 48 output tokens. It writes `trace.csv` into the run root. SHA-256 in both receipts and independently checked files:

`5418058ebec3922a08eed2e083ff863329effc86d7432b8e658cf0477f8dfcbc`.

| Run | Initialization seconds | `Simulator.run` seconds | Status |
| --- | ---: | ---: | --- |
| Plain, cold cache | 495.152905 | 1,140.308637 | FINISHED |
| Final corrected observer, cold cache | 492.573880 | 1,137.052128 | FINISHED |

The observed run ratio is −0.285581%, numerically within 2%. The plain run overlapped the preliminary observer run; the corrected observer ran later. This is a descriptive timing comparison, not an isolated overhead measurement. The supplied `run_identity.sh` runs the pair serially with a new `B21_RUN_ROOT`; that serial script was syntax-checked but not rerun end-to-end after the completed individual runs.

`b21/identity.json` has **17/17 PASS** checks, including exact `compare_artifact_directories` with `substitutions_for` through the existing A10 helper, `classify_cache_differences`, trace/observer hashes, both roles/lanes, unique stage-end identity, independent counts and scope presence. No tolerance was added to parsed-artifact comparison.

| Role/lane | Total rows | Dummies | PIECEWISE first tails | NONE real prefills |
| --- | ---: | ---: | ---: | ---: |
| PREFILL dp0 | 32 | 28 | 0 | 4 |
| PREFILL dp1 | 32 | 28 | 0 | 4 |
| DECODE dp0 | 128 | 40 | 4 | 0 |
| DECODE dp1 | 128 | 42 | 4 | 0 |

Totals: **138 dummies**, **8 PIECEWISE tails** (2 alone, 6 with decodes), **114 FULL decode rows with more than one request**, **8 NONE real prefills**. Source attention/preparation is present in all 320 rows. Block-zero scope is retained on 81 rows and block-table-write scope on 87 rows; an uninvoked pre-DP scope is not fabricated as a charge.

Final `forwards.jsonl.gz` SHA-256:
`844be98462a6b25cad4ebc68020c8f0227869a1b3cdc65838314d1750657f7e3`.
Final `collectives.json` SHA-256:
`686bace6c0d195e03456ea4eaacea03434d9ecfade936d252cf33c22c81440d2`.

Independent successful collective counts: PREFILL **50,192**, DECODE **216,978**, total **267,170**. Both the independently instrumented plain run and the observer run have the same per-role counts.

### Executed native reduction and comparison

```bash
timeout 600 "$PYTHON" tests/analysis/v4_runtime_census.py \
 --trace-dir /data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces \
 --output "$SCRATCH/native_runtime.json" --summary "$SCRATCH/native_runtime.txt"

PYTHON_BIN="$PYTHON" A10_RUN_ROOT="$SCRATCH/small_observed_complete" \
 NATIVE_ARTIFACTS=/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts \
 B21_OUTPUT_ROOT="$SCRATCH/comparison_complete" bash "$TASK/b21/run_comparison.sh"

timeout 600 "$PYTHON" "$TASK/b21/check_comparison.py" \
 "$SCRATCH/comparison_complete/comparison.json" "$TASK/b21/comparison_validation.json"
```

R1 execution passed: four TP-rank-zero traces and **408** forwards (DECODE dp0 120/dp1 64; PREFILL dp0 96/dp1 128). The final comparison independently imports and invokes R1. Final JSON and text are `$SCRATCH/comparison_complete/comparison.json` and `comparison.txt`; the command log is `$SCRATCH/comparison_complete.log` (exit 0, no stdout). `b21/comparison_validation.json` records **6/6 PASS**: all nodes read, every phase/kind count, formal lane/real/mode counts, unassigned lane/real/mode counts, profiler-window counts, both outputs exist.

The small-run composed-time observations below are diagnostic. Frontier covers its entire serving replay; native values cover only the profiler window, with different request/context distributions. Decode classes include steady lanes synchronized to their peers' PIECEWISE tails. Values are milliseconds; absolute/relative errors for both quantiles and all segments are in JSON/text.

| Role/class | Frontier n | Native window n | Frontier median / p90 | Native median / p90 | Median signed difference | Median absolute relative error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode decode batch 1 | 51 | 69 | 19.429212 / 19.707921 | 18.327267 / 18.572705 | 1.101945 | 6.013% |
| decode decode batch 2 | 48 | 48 | 19.603487 / 19.691029 | 18.381844 / 18.650757 | 1.221643 | 6.646% |
| decode decode batch 3 | 51 | 0 | 19.603487 / 19.627395 | undefined / undefined | undefined | undefined |
| decode decode batch 4 | 16 | 0 | 19.604319 / 19.618418 | undefined / undefined | undefined | undefined |
| decode dummy | 82 | 64 | 18.724965 / 19.426461 | 18.322917 / 18.540742 | 0.402049 | 2.194% |
| decode first_tail_alone | 2 | 2 | 20.389033 / 20.389033 | 25.222273 / 29.582595 | -4.833239 | 19.163% |
| decode first_tail_with_decodes | 6 | 1 | 20.272969 / 20.349388 | 34.185530 / 34.185530 | -13.912561 | 40.697% |
| prefill dummy | 56 | 198 | 18.724965 / 18.724965 | 19.338276 / 19.903310 | -0.613311 | 3.171% |
| prefill prefill | 8 | 26 | 245.233954 / 245.233954 | 307.438003 / 393.199666 | -62.204049 | 20.233% |

Role-wide phase distributions include real and dummy forwards; the native phase remains the untagged profiler window.

| Role | Frontier serving n | Native window n | Frontier composed median / p90 (ms) | Native span median / p90 (ms) |
| --- | ---: | ---: | ---: | ---: |
| decode | 256 | 184 | 19.426596 / 19.691576 | 18.350388 / 18.637093 |
| prefill | 64 | 224 | 18.724965 / 223.064147 | 19.422502 / 304.408176 |

### Small-run collective request shares

For every role/curve key below, **100% of calls and 100% of predicted collective time are exact; 0% of calls and time are interpolated**. Requested-coordinate/bracket details remain in `collectives.json`. This small scenario does not establish which coordinates the full replays will request.

| Role | Curve key | Calls | Exact call share | Interpolated call share | Predicted API-call time (ms) | Exact time share | Interpolated time share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | all_gatherv/ep/16 | 88,064 | 100% | 0% | 12399.974878 | 100% | 0% |
| decode | all_reduce/tp/8 | 430 | 100% | 0% | 4.054008 | 100% | 0% |
| decode | logits_all_gather/tp/8 | 1,392 | 100% | 0% | 32.167065 | 100% | 0% |
| decode | reduce_scatterv/ep/16 | 88,064 | 100% | 0% | 5391.066736 | 100% | 0% |
| decode | sp_all_gather/tp/8 | 20,538 | 100% | 0% | 152.104788 | 100% | 0% |
| decode | sp_reduce_scatter/tp/8 | 18,490 | 100% | 0% | 190.033306 | 100% | 0% |
| prefill | all_gatherv/ep/16 | 22,016 | 100% | 0% | 5373.332771 | 100% | 0% |
| prefill | all_reduce/tp/8 | 64 | 100% | 0% | 2.564717 | 100% | 0% |
| prefill | logits_all_gather/tp/8 | 64 | 100% | 0% | 1.228186 | 100% | 0% |
| prefill | reduce_scatterv/ep/16 | 22,016 | 100% | 0% | 3798.456823 | 100% | 0% |
| prefill | sp_all_gather/tp/8 | 3,272 | 100% | 0% | 113.714151 | 100% | 0% |
| prefill | sp_reduce_scatter/tp/8 | 2,760 | 100% | 0% | 83.904359 | 100% | 0% |

### Static checks and intermediate failures

Executed AST/line validation with the stated Python over the nine touched Python files, using `ast.parse(path.read_text(), filename=str(path))` and `assert len(text.splitlines()) <= 2000`; the results are `$SCRATCH/module_validation.json`. Current line counts: `a10_driver.py` 196, `a10_observer.py` 287, `v4_replay_census.py` 302, `test_v4_replay_census.py` 248, `test_dsv4_a10_tools.py` 239, `check_comparison.py` 58, `check_identity.py` 84, `generate_trace.py` 21, `run_small.py` 49. The exact shell syntax command was:

```bash
bash -n "$TASK/b21/run_identity.sh" "$TASK/b21/run_comparison.sh"
```

Both checks passed. No changed Python module approaches the 2,000-line limit.

The preliminary observer replay at `$SCRATCH/small_observed` was superseded for final forward-scope evidence. Manual inspection found that patching the dense `SklearnExecutionTimePredictor` base missed A10's `SklearnMoEExecutionTimePredictor` override: EP scopes appeared, but source attention/preparation did not. The fix changed only the task-local hook to the MoE override, added an actual disaggregation-to-MoE dispatch test, and completed a fresh cold-cache observer run. The first version of the new test fixture failed (`1 failed, 73 passed`, `$SCRATCH/unit_source_scope.log`) because it attempted to instantiate an abstract predictor; a concrete fixture subclass corrected it. Final regression passed 74 tests.

A separate source-scope probe's assertion also failed because it expected block-zeroing on every real forward. Actual first tails can have no new blocks to zero. The check now requires exposed pre-DP scopes to be retained exactly and both scope names to occur in the run, without fabricating an invocation. These were observer/test attribution issues; no simulator differences were fixed. Final identity and comparison use **only `small_observed_complete`**.

## 4. Remaining work and decisions for Claude

There is no remaining implementation or check failure within the assigned tooling scope. The full A10 P2-32/P2-64 replays and final acceptance were explicitly reserved for Claude and were **NOT_RUN** here. B20's speed changes and their integration are also **NOT_RUN** in this base.

Observed native evidence:

- All P2-32 and P2-64 dummy counter rows have `phase: null`. Real `kind: serving` rows use `warmup` and `formal`; capture/profile kinds are excluded regardless of phase. Thus formal dummy counts/differences cannot be reconstructed. They are null in the formal comparison, and all unassigned observations remain in a separate census.
- A multi-request DECODE counter with `has_prefill=True` gives no count of first-tail versus steady-decode requests. The native counter census labels it `first_tail_unresolved`; it does not assert that a multi-request first tail contains a steady decode. R1's range supplies the exact split only inside its profiler window.
- R1 does not identify warmup/formal phase. `time_by_phase` therefore labels the distributions `serving` versus `profiler_window`; it is not a matched per-serving-phase analysis.

| Native case | Nodes | Total weighted observations | Formal real | Null-phase dummies | Excluded warmup/capture/profile | Ambiguous formal first tails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2-32 | 4 | 67,600 | 36,108 | 20,750 | 10,742 | 328 |
| P2-64 | 8 | 111,904 | 52,494 | 41,449 | 17,961 | 267 |

P2-32 formal counts: DECODE 35,724, PREFILL 384; null-phase dummies: DECODE 2,290, PREFILL 18,460. P2-64 formal counts: DECODE 52,110, PREFILL 384; null-phase dummies: DECODE 6,913, PREFILL 34,536. Both inventory partitions were independently checked in `$SCRATCH/native_counter_validation.json`; final P2-32 comparison counts were additionally validated by the durable B21 checker.

Inference: the supplied aggregated counter/trace schemas cannot provide exact phase-matched dummy/class/time comparisons. Recommended decision: retain the explicit unknowns as diagnostic limitations, or obtain native exports with dummy phase, context/generation request counts, and profiler phase attribution before requiring that stronger comparison. No predictor/scheduler correction follows from the observed differences in this assignment.

For each future full replay, the default A10 driver writes `forwards.jsonl.gz` and `collectives.json` before returning its final receipt. Invoke the supplied comparison script with the matching native artifacts (`dsv4-h200-p2-profile-20261025-01/artifacts` or `dsv4-h200-p2-64-profile-20261025-01/artifacts`) and separate private output roots. Keep the full-run collective shares separate from the small scenario's all-exact result.

## 5. Review handoff

Review the process-local hook boundaries/restoration, MoE source-layer deduplication and EP source ownership, snapshot timing before request mutation, scope mapping/critical-path replacement and signed attribution remainder, receipt completion/hash checks, and unknown native phase/tail handling. The comparison reports counts and differences without changing serving behavior.

Rerun the four-file regression suite after merging with B20. Use `run_identity.sh` with a fresh private `B21_RUN_ROOT` if an isolated serial observer-overhead measurement is desired. Run the full A10 cases with the observer enabled, then generate JSON/text summaries against their matching native inputs. Preserve the `serving`/`profiler_window` labels and native sample counts when interpreting those summaries.

Durable handoff records: this `b21/report.md`, `b21/identity.json`, and `b21/comparison_validation.json`. Large run roots, cache/metrics files, observer streams, native reductions, and comparison JSON/text stay at the private scratch paths recorded above. Claude retains review, acceptance, commits, and task-record ownership.

## 1. Assignment and outcome

B21 is implemented in `/data/ycfeng/Frontier/.worktrees/b21-v4-census`. The observer, comparison, reproducibility scripts, and Acceptance 1–6 checks are complete. Final review and acceptance remain with Claude. Full P2-32/P2-64 A10 replays were explicitly reserved for Claude and were not run here.

Allowed repository writes were limited to `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/`, the same prefix's `b21/`, `tests/analysis/v4_replay_census.py`, `tests/unit/test_v4_replay_census.py`, and `tests/unit/test_dsv4_a10_tools.py`. All run roots, caches, bytecode/temp storage, logs, native reductions, and comparison outputs stayed in private scratch:

`/data/ycfeng/tmp/codex-implement/fcf9eca4-0142-4d72-a3ea-443401fae6fe`.

The supplied assignment, applicable AGENTS.md instructions, H3o design §1 (A11/A13), §2, §3.7 and entry 18, R1 reducer, routing/runtime evidence, native counter inputs, A10 tools, and named event/batch/DP synchronization sources informed the implementation. A11 remains diagnostic: differences never change a predictor cost or gate a replay.

No simulator source, profiling table, scheduler, simulator configuration field, dependency, native input, GPU operation, commit, deployment, or lead task record was changed. No delegation was used. Existing A10/common/cache-comparison and R1 implementations were reused; existing base work was preserved. Both completed runtime receipts identify source HEAD `a27d3adeaf5ee929be48b645c10b99f29c054ac7` and empty dirty-path lists under `frontier/` and `data/`. B20's concurrent changes were not present in these runs.

A material evidence limitation was found: native dummy counters have no phase, some first-tail counters lack the context/generation split, and R1 has no warmup/formal attribution. The implementation preserves explicit unknowns. This is reported below for Claude's decision.

## 2. Changes

Task prefix for paths beginning `a10/` or `b21/` below:

`task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/`.

| Changed file | Status | Behavior and relevant symbols |
| --- | --- | --- |
| [a10/a10_driver.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py:115) | Modified | `main` attaches the observer by default and writes trace/A11 receipts. `build_config` accepts existing-field overrides; CLI adds `--without-observer` and `--trace`. |
| [a10/a10_observer.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_observer.py:98) | New | `ForwardObserver`, `batch_shape`, `frontier_step_class`, and `collective_request` record lane forwards, exposed costs, and successful measured collective requests. |
| [tests/analysis/v4_replay_census.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/tests/analysis/v4_replay_census.py:226) | New | `read_counters`, `runtime_records`, `compare_run`, and `summary` produce weighted all-node census, R1 time comparison, collective shares, and JSON/text outputs. |
| [tests/unit/test_v4_replay_census.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/tests/unit/test_v4_replay_census.py:30) | New | Step shapes, counter weighting/phases, arithmetic, interpolation/unequal coordinates, observer restoration/failure, final snapshots, scope composition, and actual MoE dispatch coverage. |
| [tests/unit/test_dsv4_a10_tools.py](/data/ycfeng/Frontier/.worktrees/b21-v4-census/tests/unit/test_dsv4_a10_tools.py:176) | Modified | Receipt checks cover observer on/off and successful/failed runs. |
| `b21/generate_trace.py:8` | New | Generates the specified eight-request CSV and SHA-256 in each run root. |
| `b21/run_small.py:15` | New | Runs cold-cache P2-32 with independent successful collective and stage-end counters. |
| `b21/check_identity.py:14` | New | Exact parsed-artifact/cache-name comparison, row/content/hash checks, independent counts, and timing report. |
| `b21/run_identity.sh:1` | New | Serial cold-cache identity reproducer with memory limit and timeouts. |
| `b21/run_comparison.sh:1` | New | Produces one comparison JSON/text pair for a completed observed replay. |
| `b21/check_comparison.py:8` | New | Independently sums native source rows and validates comparison counts/output presence. |
| `b21/identity.json` | New | Final corrected identity result, 17 PASS checks, timings, cache counts and census. |
| `b21/comparison_validation.json` | New | Final comparison result, six PASS checks and independent native totals. |
| `b21/report.md` | New | Detailed commands, environment, results, scope contract, limitations and review handoff. |

Detailed durable report: [b21/report.md](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/report.md).

### Observation points

All wrappers exist only inside the A10 process and are restored when the observer exits:

- `BatchStageEndEvent.__init__` snapshots the final descriptor, shape, stage time and timestamps before request mutation. `handle_event` publishes after the original handler completes and counts stale events separately.
- `dp_dummy_forward._synchronize_dp_descriptors` captures local classifier/desire before synchronization and final width vector, peer modes and peer classes afterwards.
- `SklearnMoEExecutionTimePredictor.predict_stage_execution_time` reads existing single-layer source-attention results with `include_ffn=False`, once per physical layer.
- `BaseClusterScheduler._prepare_moe_ep_wave_plan` provides source-batch ownership; static `_get_shared_ep_phase_times_ms` exposes existing TP-rank-zero EP costs for each DP source lane.
- `MeasuredCCBackend.predict_operator_collective` records only successful calls after the original returns. A failing original call produces no record and propagates its exception unchanged.

No hook is installed inside B20's edited modules. The observer changes only its own dictionaries and output files; it requests no additional predictions and modifies no simulator entity, descriptor, input, or cache.

Each forward row includes role, replica, DP lane, batch/stage/epoch identity, start/end simulated time, real/dummy, final mode, local tokens/requests, synchronized width/vector, classifier inputs, desired mode, peer modes/classes, native step class, composed milliseconds, scope costs and segment costs. Step classification reuses R1's `step_class`; Frontier shape uses `dsv4_request_is_decoding` so the first DECODE tail follows the V2 prefilling rule.

Collectives aggregate by role, curve key, requested bytes and unequal ordered rows or null. Records include call count, predicted milliseconds/total, exact/interpolated status, bracketing measured bytes/times and interpolation log-distance ratio. Exact coordinates have identical brackets and a null ratio. Unequal combine admission remains exact only. These counts are prediction API calls, including repeated estimator/metrics calls, rather than executed-collective counts.

### Scope-to-segment mapping

| Segment/column | Mapping |
| --- | --- |
| `prepare` | `dsv4_input_preparation` |
| `post` | `dsv4_lm_head`, `dsv4_logits_gather`, `dsv4_sampling` |
| `dispatch` | Registered V4 attention/mHC, MoE/shared-expert and V4 communication scopes, excluding post scopes; includes embedding, mHC head, final norm and final mHC gather. Exact names are stored in receipts/comparison outputs. |
| Own columns | `kv_cache_block_zero`, `dsv4_block_table_write`, and each unknown exposed scope. |
| `coordination_and_overlap` | Signed composed duration minus mapped segments and own columns; preserves coordination/overlap without introducing a cost. |

Scope values retain their exposed gaps. Attention dispatch attribution uses the existing `_dsv4_attention_schedule.elapsed_ms` critical path while retaining individual scope costs. EP attribution is TP rank zero per DP source lane; it is not an exact reconstruction of EP barriers/rank maxima.

Native R1 assigns the next forward's pre-DP work to the preceding post segment. No cross-forward reassignment is inferred. Whole native time uses device span; native segment time uses device union, converted to milliseconds. Each statistic includes its contributing forward count. Relative error is null at native zero or unavailable inputs.

## 3. Verification

Date: October 29, 2026. Interpreter: `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python`, Python 3.10.6, GCC 11.3.0. NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3, pytest 9.1.1. Linux x86_64; 64 logical CPUs. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` were unset.

Commands below use these aliases from the working root:

```bash
PYTHON=/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python
TASK=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs
SCRATCH=/data/ycfeng/tmp/codex-implement/fcf9eca4-0142-4d72-a3ea-443401fae6fe
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export TMPDIR="$SCRATCH/tmp"
ulimit -v 62914560
```

| Acceptance | Expected | Actual | Result |
| --- | --- | --- | --- |
| 1 | Requested Frontier shapes match R1; weighted all-node phase reading; arithmetic including native zero | All covered in final regression suite | PASS |
| 2 | Two cold caches; exact parsed-artifact/cache-name identity; observer time within 2% or both times reported | Zero artifact/name differences; 1,758 cache files each; observed time −0.285581% | PASS, timing caveat below |
| 3 | Both roles/lanes, required modes/classes, one row per stage end, independent collective counts | 320 unique rows = 320 stage ends; all required examples; 267,170 collective calls agree | PASS |
| 4 | Small comparison JSON/text and exact native source count sums | Four nodes; 67,600 weighted observations; six independent checks pass | PASS, unknown native attributes retained |
| 5 | Four requested unit files; dummy matrix 12/12 | 74 passed in 4.20 s; 12/12 smokes | PASS |
| 6 | B21 scripts/records/report; modules ≤2,000 lines | All present; nine touched Python files parse; maximum 302 lines; shell syntax passes | PASS |

### Executed regression and dummy matrix

```bash
timeout 600 "$PYTHON" -m pytest \
 tests/unit/test_dsv4_a10_tools.py tests/unit/test_dsv4_graph_dispatch.py \
 tests/unit/test_dsv4_dp_sync.py tests/unit/test_v4_replay_census.py \
 -q -p no:cacheprovider --basetemp="$SCRATCH/pytest_source_scope_fixed"

PYTHON_BIN="$PYTHON" METRICS_OUTPUT_DIR="$SCRATCH/dummy/metrics" timeout 600 \
 bash examples/architecture/run_dummy_smoke_matrix.sh -- \
 --metrics_config_cache_dir "$SCRATCH/dummy/cache"
```

Observed logs: `$SCRATCH/unit_source_scope_fixed.log` reports **74 passed in 4.20 s**. `$SCRATCH/dummy.log` reports **Completed 12 profiling-independent dummy smoke case(s)**, covering dense/MoE, offline/online and all three architectures.

### Executed cold-cache runs and identity

```bash
timeout 7200 "$PYTHON" "$TASK/b21/run_small.py" "$SCRATCH/small_plain" --without-observer
timeout 7200 "$PYTHON" "$TASK/b21/run_small.py" "$SCRATCH/small_observed_complete"
timeout 600 "$PYTHON" "$TASK/b21/check_identity.py" \
 "$SCRATCH/small_plain" "$SCRATCH/small_observed_complete" "$TASK/b21/identity.json"
```

Both receipts are FINISHED. Logs are `$SCRATCH/plain.log`, `$SCRATCH/observed_complete.log`, and `$SCRATCH/identity_complete.log`.

Each run generated arrivals at 0.00, 0.05, …, 0.35 s, with 6,144 prompt and 48 output tokens per request. Both trace hashes, independently checked against the files, are:

`5418058ebec3922a08eed2e083ff863329effc86d7432b8e658cf0477f8dfcbc`.

| Run | Initialization seconds | Simulator.run seconds |
| --- | ---: | ---: |
| Plain | 495.152905 | 1,140.308637 |
| Final corrected observer | 492.573880 | 1,137.052128 |

Observed change: **−0.285581%**, numerically within 2%. The plain run overlapped the preliminary observer run; the corrected observer ran later. The ratio is descriptive, not an isolated overhead estimate. `run_identity.sh` supplies a serial reproducer; it was syntax-checked but not rerun end-to-end after the completed individual runs.

[identity.json](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/identity.json) records **17/17 PASS**. Artifact comparison uses existing `compare_artifact_directories` and `substitutions_for`; cache-name comparison uses `classify_cache_differences`. No comparison tolerance was introduced.

| Role/lane | Rows | Dummies | PIECEWISE tails | NONE real prefills |
| --- | ---: | ---: | ---: | ---: |
| PREFILL dp0 | 32 | 28 | 0 | 4 |
| PREFILL dp1 | 32 | 28 | 0 | 4 |
| DECODE dp0 | 128 | 40 | 4 | 0 |
| DECODE dp1 | 128 | 42 | 4 | 0 |

There are **320 unique forwards matching 320 independently counted stage ends**, with zero stale ends; **138 dummies**, **8 PIECEWISE tails** (2 alone, 6 with decodes), **114 FULL multi-request decodes**, and **8 NONE prefills**. Attention/preparation scopes occur on all 320 rows. Block zeroing is retained on 81 rows and block writes on 87 rows.

Final observer SHA-256 values:

- `forwards.jsonl.gz`: `844be98462a6b25cad4ebc68020c8f0227869a1b3cdc65838314d1750657f7e3`
- `collectives.json`: `686bace6c0d195e03456ea4eaacea03434d9ecfade936d252cf33c22c81440d2`

Independent successful collective calls equal observer counts and plain-run counts: PREFILL **50,192**, DECODE **216,978**, total **267,170**.

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

All completed successfully. R1 found **408 profiler-window forwards**: DECODE dp0 120/dp1 64 and PREFILL dp0 96/dp1 128. The comparison imports and invokes R1 rather than copying its functions.

Final outputs are `$SCRATCH/comparison_complete/comparison.json` and `comparison.txt`. [comparison_validation.json](/data/ycfeng/Frontier/.worktrees/b21-v4-census/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/comparison_validation.json) records **6/6 PASS**: every node, phase/kind sum, formal lane/real/mode sum, unassigned lane/real/mode sum, native window counts and output presence.

These small-run times compare the whole Frontier serving replay with a native profiler window containing different request/context distributions. Values are diagnostic milliseconds; all segment values and absolute/relative errors are in JSON/text.

| Class | Frontier n | Native window n | Frontier median / p90 | Native median / p90 |
| --- | ---: | ---: | ---: | ---: |
| DECODE batch 1 | 51 | 69 | 19.429212 / 19.707921 | 18.327267 / 18.572705 |
| DECODE batch 2 | 48 | 48 | 19.603487 / 19.691029 | 18.381844 / 18.650757 |
| DECODE batch 3 | 51 | 0 | 19.603487 / 19.627395 | undefined |
| DECODE batch 4 | 16 | 0 | 19.604319 / 19.618418 | undefined |
| DECODE dummy | 82 | 64 | 18.724965 / 19.426461 | 18.322917 / 18.540742 |
| First tail alone | 2 | 2 | 20.389033 / 20.389033 | 25.222273 / 29.582595 |
| First tail with decodes | 6 | 1 | 20.272969 / 20.349388 | 34.185530 / 34.185530 |
| PREFILL dummy | 56 | 198 | 18.724965 / 18.724965 | 19.338276 / 19.903310 |
| PREFILL real | 8 | 26 | 245.233954 / 245.233954 | 307.438003 / 393.199666 |

Role-wide serving/profiler-window distributions include real and dummy rows:

| Role | Frontier n | Native n | Frontier median / p90 (ms) | Native median / p90 (ms) |
| --- | ---: | ---: | ---: | ---: |
| DECODE | 256 | 184 | 19.426596 / 19.691576 | 18.350388 / 18.637093 |
| PREFILL | 64 | 224 | 18.724965 / 223.064147 | 19.422502 / 304.408176 |

### Collective shares for the small run

Every group below has **100% exact calls and predicted time, 0% interpolated calls and time**. Coordinate/bracket details remain in `collectives.json`. This result does not establish the payloads requested by the full replays.

| Curve key | DECODE calls | PREFILL calls | Exact call/time share | Interpolated call/time share |
| --- | ---: | ---: | ---: | ---: |
| all_gatherv / ep / 16 | 88,064 | 22,016 | 100% / 100% | 0% / 0% |
| all_reduce / tp / 8 | 430 | 64 | 100% / 100% | 0% / 0% |
| logits_all_gather / tp / 8 | 1,392 | 64 | 100% / 100% | 0% / 0% |
| reduce_scatterv / ep / 16 | 88,064 | 22,016 | 100% / 100% | 0% / 0% |
| sp_all_gather / tp / 8 | 20,538 | 3,272 | 100% / 100% | 0% / 0% |
| sp_reduce_scatter / tp / 8 | 18,490 | 2,760 | 100% / 100% | 0% / 0% |

### Static checks and intermediate corrections

AST parsing and the ≤2,000-line assertion passed for all nine touched Python files. Line counts are 196, 287, 302, 248, 239, 58, 84, 21 and 49 respectively; evidence is `$SCRATCH/module_validation.json`. Both shell scripts passed:

```bash
bash -n "$TASK/b21/run_identity.sh" "$TASK/b21/run_comparison.sh"
```

The preliminary observer at `$SCRATCH/small_observed` was superseded for forward-scope evidence. Inspection found that patching the dense predictor base missed A10's MoE override, leaving source attention/preparation absent. Only the task-local hook was corrected, a real disaggregation-to-MoE dispatch test was added, and a fresh cold-cache observer replay completed.

The first new test fixture failed with `TypeError` because the predictor class was abstract (`1 failed, 73 passed`, `$SCRATCH/unit_source_scope.log`). A concrete fixture subclass fixed it; final regression passed 74 tests. A source-scope probe also overrequired block zeroing on every real row. Actual first tails can have no new blocks, so the final check requires exposed pre-DP scopes to be retained exactly and both names to occur in the run. No simulator difference was fixed. Final identity and comparison use only `small_observed_complete`.

## 4. Remaining work and decisions for Claude

No assigned implementation/check failure remains. Full A10 replays, final acceptance and B20 integration are **NOT_RUN** here.

Observed native evidence:

- Real serving rows use `phase: warmup` or `formal`; every dummy counter row in both cases has `phase: null`. Capture/profile kinds are excluded regardless of phase. Formal dummy counts/differences are therefore null, with unassigned rows retained separately.
- Multi-request first-tail counters lack the context/generation request split. They use `first_tail_unresolved`; R1 supplies the exact split only in its profiler window.
- R1 has no warmup/formal tag. Time distributions are explicitly labeled `serving` versus `profiler_window`, not phase-matched.

| Case | Nodes | Weighted total | Formal real | Unassigned dummies | Excluded | Ambiguous formal tails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2-32 | 4 | 67,600 | 36,108 | 20,750 | 10,742 | 328 |
| P2-64 | 8 | 111,904 | 52,494 | 41,449 | 17,961 | 267 |

P2-32 formal counts are DECODE 35,724/PREFILL 384; unassigned dummies are DECODE 2,290/PREFILL 18,460. P2-64 formal counts are DECODE 52,110/PREFILL 384; unassigned dummies are DECODE 6,913/PREFILL 34,536. Both inventories were independently reconciled in `$SCRATCH/native_counter_validation.json`; the final P2-32 comparison has additional durable validation.

Inference: the supplied schemas cannot support exact phase-matched dummy/class/time comparisons. Recommended next action is to retain these explicit diagnostic limitations, or obtain native exports containing dummy phase, context/generation counts and profiler phase attribution before requiring the stronger comparison. Observed differences authorize no predictor or scheduler correction in B21.

## 5. Review handoff

Review hook restoration, actual MoE dispatch coverage, layer deduplication/EP source ownership, snapshot timing before request mutation, attention critical-path attribution, signed coordination remainder, receipt hash/completion checks and unknown native attributes.

Rerun the four-file regression suite after merging B20. Use `run_identity.sh` with a fresh private `B21_RUN_ROOT` if an isolated serial overhead measurement is needed. Run the full A10 cases with the observer enabled, then invoke `run_comparison.sh` against each matching native artifacts directory and separate output roots. Keep full-run collective shares separate from this small scenario's all-exact result.

Durable records are `b21/report.md`, `b21/identity.json`, and `b21/comparison_validation.json`. Large run/observer/comparison artifacts remain at the private scratch paths above. Claude retains review, acceptance, commits and task-record ownership.

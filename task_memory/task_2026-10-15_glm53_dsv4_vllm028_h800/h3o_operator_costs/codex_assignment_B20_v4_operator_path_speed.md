# Task: make the V4 operator path fast enough for the A10 replays, with identical results (stage 4 assignment B20, CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Set the working tree and the base worktree: both start from the B18 commit `b6c28858`; the lead creates the base worktree. |
| 2026-10-29 | Created after the lead's cProfile probes of P2-32 on the operator path. Covers caching of four repeated pure computations, the identity check against the base commit, and the speed measurement. |

## Goal

The A10 replays of P2-32 and P2-64 run on the operator path. At the current speed they cannot finish in practical time:

| Probe | Code | Wall window | Simulated time reached | Layer waves |
| --- | --- | --- | --- | --- |
| `probe/p2_32_prof` | `e52ecec4` plus the B18 candidate classifier edit | 300 s of `Simulator.run` under cProfile | 0.4406 s | 1,031 |
| `probe/p2_32_perf` | the same plus caches 1 and 2 below | 300 s | 0.8640 s | 2,795 |
| `probe/p2_32_perf2` | the same plus caches 1-4 below | 300 s | 1.7114 s | 5,827 |

Logs and `profile.pstats` files are under `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/probe/` (read-only for you; the drivers are `probe_profile.py`, `probe_profile2.py` and `run_perf*.sh` in the parent directory). The P2-32 run simulates 146 s (128 requests of 6,144 prompt and 1,024 output tokens, diagnostic run `h200_mainline/diagnostic/results/p2_32e2`). At 0.44 s per 300 s it needs about a day; with the four caches about 7 hours. P2-64 has twice the EP ranks per layer wave.

The lead's prototype lives in a scratch worktree (`/data/ycfeng/Frontier/.worktrees/probe-perf`, detached, never merged). It used module-global dictionaries keyed by `id()`. That is not acceptable code; it only measured the gain. The deliverable is the clean version below.

Measured hot spots at the base (first probe, 295 s profiled):

| Function | Cumulative | Cause |
| --- | --- | --- |
| `DeepSeekV4OperatorPredictor.grouped_gemm` (`dsv4_operator_predictor.py:359`) | 202 s | Each call builds the balanced workload of the whole layer twice (`materialize_balanced_rank_classes` and `materialize_balanced_ep_workload`), with full validation of 256 experts and 16 lane descriptors, about 5.8 ms per build, for every EP rank of every layer |
| `operator_tensor_parallel_size` (`dsv4_operator_predictor.py:377`) and the shared-expert contract in `predict_source_execution` (`deepseek_v4.py`) | 22 s and 20 s | `resolve_layer_contract` for the same operator and layer on every call |
| `build_operator_manifest(model, global_layer_id=...)` in `predict_source_execution` | 18 s (second probe) | The same per-layer manifest rebuilt on every call, only to read its operator names |
| `DeepSeekV4AttentionPredictor.scope_times` (`dsv4_predictor.py:310`) | 120 s of 300 s once grouped GEMM is cached | The fitted random forest (about 250 trees, through joblib) is called for features that are identical across the layers of one variant in one forward. The prototype's count: 13,440 repeated feature tuples, 72 distinct, in 300 s |

## Approach and decisions

### Read first

- `frontier/moe_ep_workload.py` (`LayerEPWorkload`, frozen, with `MappingProxyType` fields; `materialize_layer_ep_workload`, `materialize_balanced_ep_workload`, `materialize_balanced_rank_classes`).
- `frontier/execution_time_predictor/dsv4_operator_predictor.py` (`DeepSeekV4OperatorPredictor`, `_layer_contract`, `operator_tensor_parallel_size`).
- `frontier/execution_time_predictor/deepseek_v4.py` (`predict_attention_operator_times`, `predict_source_execution`, `predict_structural_execution`).
- `frontier/execution_time_predictor/dsv4_predictor.py` (`DeepSeekV4AttentionPredictor.scope_times`).
- Where the predictor that owns `_model_config`, `_replica_config` and `_predictions` is built for a V4 cluster (`sklearn_execution_time_predictor.py` and the V4 training registration).
- The A10 driver from B18: `h3o_operator_costs/a10/a10_driver.py` (committed before this assignment; read its usage line).
- `tests/e2e/refactor_fidelity/compare.py` (`compare_artifact_directories`, `classify_cache_differences`, `substitutions_for`).

### Decisions

1. **Balanced EP workload.** Decorate `materialize_balanced_ep_workload` and `materialize_balanced_rank_classes` with `functools.lru_cache(maxsize=None, typed=True)`.
   - Both are pure functions of four integers; their results are immutable (the frozen dataclass with read-only mappings, and a tuple of tuples).
   - `typed=True` keeps a bool or NumPy integer argument out of an entry made by an exact `int`, so such a call still reaches `_require_int` and raises as today.
   - No other change in this module. The scheduler-side `materialize_layer_ep_workload` call per wave stays uncached.
2. **Layer facts.** The per-operator TP from `operator_tensor_parallel_size`, the shared-expert layer contract and the set of operator names of the per-layer manifest are computed once for the predictor that owns the fixed model and replica configuration, eagerly at V4 predictor construction or lazily on first use.
   - Do not key a module-global dictionary by object `id()`.
   - The non-V4 predictors must not compute or hold these facts.
   - The validation that `predict_source_execution` performs on each batch stays per call. Only the facts that depend on the configuration and the layer id are cached.
3. **Attention interpolation.** In `DeepSeekV4AttentionPredictor.scope_times`, memoize the fitted model's value per instance, keyed by the model name and the feature tuple.
   - The coverage check, the exact lookup and the finite and sign checks keep running on every call, in the current order. Only `model.predict` on a tuple already seen is skipped.
   - Another assignment (B19, in a separate worktree) changes the coverage check in the same function. Keep your edit to the prediction line and the memo, so the two merge with a small, local conflict at most.
4. **Nothing else changes.** `ExecutionTime`, `Request`, the EP lane batch builder, the communication payload validation and the scheduler wave code are shared with other models. List what they cost after your change (Acceptance 4) and do not change them here.

## Acceptance

1. **Unit tests** (new `tests/unit/test_v4_prediction_memo.py`):
   - The two balanced materializers return results equal to their uncached `__wrapped__` versions over a grid of populations, top-k and EP sizes. After a valid call has filled an entry, `True` and `numpy.int64` arguments for the same value still raise.
   - The attention memo: a second call with the same features does not call `model.predict` (counted by a wrapper on the model) and returns the same float. A call with different features does call it. Out-of-coverage features still raise on the first and on every later call.
   - The layer facts: two predictors with different replica configurations each get their own values, and the values equal the uncached computation for every layer of the V4 model config.
2. **Identity against the base.**
   - Base: the commit you start from, `b6c28858`. The lead has created the detached worktree `/data/ycfeng/Frontier/.worktrees/b20-base` at that commit for this check. Run the base from it and do not edit it.
   - Scenario: the P2-32 config of `a10_driver.py` with one change, a trace file of 8 requests arriving at 0.00, 0.05, ..., 0.35 s, each with 6,144 prompt tokens and 48 output tokens. Write the trace into the output root and record its SHA-256. If `a10_driver.py` has no way to set one more field, use `h4a4_driver.build_config` with the same fields and say so.
   - Run it cold on the base and on your change.
   - Pass: every artifact identical after parsing (`compare_artifact_directories` with `substitutions_for` for paths), and `classify_cache_differences` reports no predictor cache name difference.
   - Also show from the run artifacts that the scenario exercised: prefill on both PREFILL lanes, NIXL first tails on DECODE dispatched PIECEWISE, decode forwards with more than one request dispatched FULL, and at least one DP dummy forward.
3. **Speed.**
   - Wall time of `Simulator.run` and the peak RSS for both identity runs.
   - The simulated time reached by P2-32 in a 300 s window of `Simulator.run`, with the warm cache copied from one of your identity runs. Use your own copy of the lead's probe driver.
   - Expected: the 300 s window reaches at least 1.5 s of simulated time (the lead's prototype reached 1.71 s); the peak RSS rises by less than 1 GB.
   - Below that, report the profile; do not widen the scope.
4. **Remaining cost.** The top 15 functions by cumulative time in the 300 s window after your change, with a one-line cause each, and an estimate of the full P2-32 and P2-64 wall times from the measured simulated-time rate.
5. **Regression.**
   - `tests/unit/test_dsv4_operator_consumer.py`, `test_dsv4_runner_consumer.py`, `test_dsv4_runner_ledger.py`, `test_dsv4_kv_block_table_writes.py`, `test_dsv4_predictor.py`, `test_v4_attention_coverage.py`, `test_balanced_routing.py`, `test_moe_rank_plan.py`, `test_moe_rank_routing.py`, `test_dsv4_dp_sync.py`, `test_dsv4_structural_pd.py`, `test_dsv4_structural_execution.py`, `test_dsv4_source_execution.py`, `test_dsv4_attention_runtime.py`, `test_dsv4_attention_workload.py`, `test_dsv4_features.py`, and the B18 tests.
   - The dummy smoke matrix (`PYTHON_BIN` set) 12 of 12.
   - The known failure `test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` predates this work and may remain.
6. **Records.** `h3o_operator_costs/b20/` holds the trace generator, the run and comparison scripts, the comparison JSON and a `report.md` with the exact commands, Python version, environment, expected and actual values and PASS/FAIL per item. Touched modules stay at or below 2,000 lines.

## Writable paths

Your working tree is `/data/ycfeng/Frontier/.worktrees/b20-v4-speed` (branch `b20-v4-speed`, started at `b6c28858`, which includes B18). All paths below are relative to it. B19 runs at the same time in `.worktrees/b19-v4-heldout`; its changes are not part of your base.


- `frontier/moe_ep_workload.py`, `frontier/execution_time_predictor/dsv4_operator_predictor.py`, `frontier/execution_time_predictor/deepseek_v4.py`, `frontier/execution_time_predictor/dsv4_predictor.py`.
- The V4 predictor construction site, only if Decision 2 places the facts there.
- `tests/unit/test_v4_prediction_memo.py` (new).
- `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b20/` (new).

Do not edit B18's or B19's files, the profiling tables, or the lead's scratch directories. Temporary outputs go under `/data/ycfeng/tmp/b20/`. Simulations run with `ulimit -v 62914560` and a timeout.

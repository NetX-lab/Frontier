**1. Assignment and outcome.** B20 caches four repeated pure computations on the DeepSeek-V4 operator path used by A10 P2-32/P2-64, preserving results against `b6c288588ae2a19c0b86620d782c8a3d838cae9d`. Outcome: **partial**. Implementation, memo tests, cold identity comparison, final DP dispatch proof, speed measurement, remaining-cost profile, smoke matrix and records are complete. One required regression test cannot read a retained fixture absent in both worktrees and outside writable scope. Final acceptance remains with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/b20-v4-speed`. Untouched base: `/data/ycfeng/Frontier/.worktrees/b20-base`. Writable source paths were `frontier/moe_ep_workload.py`, `frontier/execution_time_predictor/deepseek_v4.py`, `frontier/execution_time_predictor/dsv4_operator_predictor.py`, `frontier/execution_time_predictor/dsv4_predictor.py`, and `frontier/execution_time_predictor/dsv4_training.py`; also new `tests/unit/test_v4_prediction_memo.py` and the B20 record directory. All outputs, caches and temporary files reside under `/data/ycfeng/tmp/codex-implement/4bcc70f2-4440-43f7-9489-12ff832c399a/b20`.

The applicable `AGENTS.md`, named workload/predictor modules, V4 construction/training paths, B18 assignment/report, A10 driver, fidelity comparator and lead probes were inspected. Requirements preserved: typed unlimited balanced caches; configuration facts owned by one fixed V4 predictor; attention predictions memoized per instance/model/features; validation remains per call; touched Python modules remain below 2,000 lines. Fixed configuration and fitted models during predictor lifetime are the assignment's settled assumptions. B18 remains the base and B19 is excluded. No unrelated pre-existing edits were encountered in the changed paths. No Git operation, commit, dependency installation/upgrade, GPU operation, delegation, profiling-table edit, scheduler/entity edit or deployment was performed.

The complete durable record is [report.md](/data/ycfeng/Frontier/.worktrees/b20-v4-speed/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b20/report.md).

**2. Changes.** Three production files changed. The other allowed production files remain byte-identical to the base.

| Changed file | Relevant symbols/lines | Behavior and reason |
| --- | --- | --- |
| `frontier/moe_ep_workload.py` | Import at 13; balanced materializers at 888 and 909 | Add only `lru_cache(maxsize=None, typed=True)` to `materialize_balanced_ep_workload` and `materialize_balanced_rank_classes`. Their immutable results are reused across EP ranks and layers. Bool and NumPy integer arguments cannot reuse entries populated by exact integers and still reach validation. Scheduler `materialize_layer_ep_workload` stays uncached. |
| `frontier/execution_time_predictor/deepseek_v4.py` | `DeepSeekV4LayerFacts` at 122; `_layer_facts` at 160; source prediction at 168 | Lazily attach configuration facts to the owning V4 predictor: immutable operator-name sets per layer, shared-expert contracts per layer, and TP per operator. Reuse these at existing prediction sites. Existing layer-ID validation runs on hits; batch validation remains per call. Non-V4 execution never creates these facts. |
| `frontier/execution_time_predictor/dsv4_predictor.py` | Memo at 309; `scope_times` at 311; prediction branch at 335 | Skip fitted `model.predict` for a seen `(model_name, feature_tuple)`. Coverage/schema/exact lookup/finite/sign checks retain their order and run on every call. The coverage block is unchanged for B19 integration. |
| `tests/unit/test_v4_prediction_memo.py` — new | Materializer tests at 34; attention tests at 75; layer facts at 123 | 350 cases cover cached/uncached equality, numeric-alias rejection after hits, counted fitted-model predictions, different features/models/instances, repeated coverage and finite/sign checks, two replica configurations and every V4 layer, and cache-hit layer-ID validation. |

No construction-site edit was needed because lazy ownership fits in `deepseek_v4.py`. `dsv4_operator_predictor.py`, `dsv4_training.py` and the inspected `sklearn_execution_time_predictor.py` remain unchanged. There is no module-global dictionary keyed by object identity. `ExecutionTime`, `Request`, lane batching, communication payload validation and scheduler wave logic remain unchanged. No behavior/interface/scope deviation was introduced.

Every new file in `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b20/` is listed below:

| Files | Purpose |
| --- | --- |
| `generate_trace.py` | Generate/hash the eight-request identity trace. |
| `run_replay.py`, `run.sh` | Run A10 configurations with scratch outputs, limits/timeouts, timing/RSS receipts and read-only dispatch observers. |
| `run_checks.sh`, `run_smoke.sh` | Required pytest selection and 12-case dummy matrix with CPU interpreter and scratch storage. |
| `run_window.sh` | Copy a completed identity cache, retain copy hashes and run the 300-second P2-32 profile. |
| `compare_runs.py`, `comparison.json` | Parsed artifact comparison with path substitutions and cache-name classification. |
| `verify_final_dispatch.py`, `final-dp-dispatch.json` | Verify post-synchronization DP metadata and observer-run identity. |
| `profile_report.py`, `profile.json` | Top 15 functions, remaining costs, RSS deltas, speedup and replay estimates. |
| `verify_retained_plan.py`, `retained-plan-check.json` | Check the missing fixture's retained source without writing outside scope or relabeling the required test. |
| `verify_static.py`, `static-checks.json` | Syntax, sizes, whitespace, JSON parsing and measured source hashes. |
| `environment.json`, `checks.json`, `report.md` | Environment, retained commands/results/log hashes and complete handoff. |

**3. Verification.** Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6, GCC 11.3.0. Packages: NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, joblib 1.6.0, pytest 9.1.1. CPU: Intel Xeon Gold 6330 at 2.00 GHz, effective CPUs 0–63. Bytecode and W&B disabled; BLAS/OMP/MKL threads each 1. `TMPDIR` and `FRONTIER_TMP_ROOT` point to private scratch. Simulations use `ulimit -v 62914560` and timeouts; pytest disables cacheprovider and uses scratch basetemp/output cwd with read-only repository data access.

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| New memo tests | Equality, isolation and repeated validation | 350 passed in 2.17 s | PASS |
| Cold identity | Every parsed artifact equal; cache names unchanged | Six artifacts equal; 1,758 cache entries each; no differences | PASS |
| Final DP coverage | Both PREFILL lanes; PIECEWISE first tails; FULL multi-request decode; dummy forward | Lanes 0/1; 8 first tails, all PIECEWISE; 114 multi-request FULL forwards; 138 dummy forwards | PASS |
| 300-second speed window | At least 1.5 simulated seconds; RSS growth below 1 GB | 2.0053377476662515 simulated seconds; probe RSS growth 209,604,608 bytes over cold base | PASS |
| Required regression plus B18 tests | Required suite complete | Initial 851 passed, 2 skipped, 3 failed in 471.85 s; two output failures cleared on reruns, missing fixture remains | FAIL: unavailable prerequisite |
| Final memo/source/structural rerun | Final changes and scratch-cwd fix pass | 460 passed in 18.15 s | PASS |
| Operator training/cache parity rerun | Both training paths and warm cache preserve values | 1 passed in 191.00 s, three pandas mixed-type warnings | PASS |
| Dummy smoke matrix | 12/12 with explicit interpreter | 12/12, exit 0 | PASS |
| Static/records | Syntax, sizes, whitespace and measured hashes valid | AST, shell syntax, JSON and whitespace checks pass; source hashes match measured candidate | PASS |

The default `run_checks.sh` includes all 16 named regression files from the assignment, the new memo test, and B18's graph-dispatch and A10-tool tests. B18's modified DP-sync tests are already in the required selection. Counts from overlapping executions must not be added. The two skips concern the optional pinned balanced-routing registry (`VLLM_BS_SOURCE_ROOT`) and unavailable retained P1h GPU evidence.

Two initial failures came from existing fixtures constructing default `MetricsConfig()` and attempting to create `outputs/` in the read-only repository cwd. The final runner executes from scratch with a read-only link to `data`. Both affected tests passed without production or fixture edits: operator training/cache parity, and EP-expansion source-ID quota. An earlier targeted run had 459 passes plus the same output error; the corrected rerun had 460 passes. The missing frozen-plan fixture is a separate failure.

The missing fixture and three supplemental README assertions reproduce on the untouched base: 4 failed and 2 passed in 2.11 s. A supplementary exploratory release/graph/dummy/DP run had 99 passed and 4 failed in 80.59 s: three README assertions and the assignment-known DP late-join failure, lane 2 `(0, False)` instead of `(0, True)`. Its historical suite-selection command was not retained; it is diagnostic evidence and is not counted as a required green acceptance check. No README or DP fix was made.

Identity uses A10's P2-32 fields. `a10_driver.py` accepts only `<case> <output root>` and has no extra-field argument, so the permitted `h4a4.build_config` fallback receives the same A10 case and `a10.operator_fields`, dropping the same diagnostic override/table entries. Identity adds only the trace-file field. Both roles use fitted operator costs; dummy mode is false and no step-cost table is bound.

Each identity root contains eight requests arriving at 0.00–0.35 s in 0.05 s steps, prompt 6,144 and output 48. Trace SHA-256: `cec2c473c51d43518ac4bb0781ae3d020e63f7a841297b4ff268d2a97d962186`.

`compare_artifact_directories` with `substitutions_for` reports no differences in `config.json`, `frontier_transfer_ledger.jsonl`, `metrics_ground_truth.jsonl`, `op_precision_metadata.csv`, `request_metrics.csv` and `system_metrics.json`. `classify_cache_differences` reports none: each cache has 879 pickle files and 879 lock files. Neither run produces `cc_cache`; that absence is recorded on both sides. Local dispatch observer records are byte-identical.

Because DP synchronization can revise descriptors, a separate warm replay records final metadata after native `_synchronize_dp_descriptors` returns. Its six simulator artifacts, cache names and local observer also match the cold candidate. The final coverage counts above come from those post-synchronization snapshots.

| Scratch run root | Initialization s | Simulator.run s | Simulated s | Peak RSS KiB |
| --- | ---: | ---: | ---: | ---: |
| `base-cold` | 510.09502797201276 | 1139.313491472043 | 2.702534436729753 | 997996 |
| `candidate-final-cold` | 520.145190033596 | 170.79701200406998 | 2.702534436729753 | 998380 |
| `candidate-dp-proof` — supplementary warm observer | 54.55038211029023 | 171.63454188778996 | 2.702534436729753 | 1202668 |
| `window` — warm, profiled P2-32 | 55.903996157925576 | 300.0001518148929 | 2.0053377476662515 | 1202688 |

Cold `Simulator.run` speedup: **6.670570392911171×**. Candidate cold RSS growth: 393,216 bytes, or 384 KiB. Probe RSS growth over cold base: 209,604,608 bytes. RSS is the whole-process peak from `resource.getrusage(RUSAGE_SELF).ru_maxrss`, including initialization. The correct probe peak is **1,202,688 KiB**. Cache-copy provenance and hashes remain in `window/warm_cache_copy.json`. Two preliminary aborted initialization attempts never reached `Simulator.run` and are excluded. The probe and base run overlapped on the shared CPU host; these are single measurements, not repeated paired benchmarks.

Top 15 functions by cumulative time in the 300-second window:

| # | Function | Cumulative s | Cause |
| --- | --- | ---: | --- |
| 1 | `Simulator.run` | 300.000 | Top-level DES run and profiled event loop. |
| 2 | `_run_sequential` | 300.000 | Processes ordered events and invokes handlers. |
| 3 | `enter_layer_sync` | 251.600 | Collects DP participants and starts ready EP layer waves. |
| 4 | `schedule_layer_wave` | 249.722 | Schedules EP plans, phases and completion events; 6,689 waves. |
| 5 | `_prepare_moe_ep_wave_plan` | 247.714 | Delegates workload and rank-phase planning. |
| 6 | `prepare_moe_wave_from_inputs` | 247.651 | Materializes scheduler workload and all lane timings. |
| 7 | `predict_ep_wave_phase_times` | 212.337 | Builds and prices every EP rank, including wrapper and communication work. |
| 8 | `_predict_layer_with_kernel_gap` | 151.065 | Prices layers and validates/applies operator kernel gaps. |
| 9 | `decode_sync_event.handle_event` | 149.567 | Runs decode synchronization handler. |
| 10 | `on_decode_sync` | 149.288 | Passes lane completion to shared synchronization. |
| 11 | `enter_decode_sync` | 149.256 | Registers completion and starts the cohort's next wave. |
| 12 | `_on_decode_ep_wave_ready` | 148.237 | Schedules ready decode waves and continuations. |
| 13 | `_predict_dsv4_structural_execution` | 144.818 | V4 prediction entry for attention lanes and routed ranks. |
| 14 | `predict_structural_execution` | 143.798 | Builds timing, workloads, runner scopes and communication payloads. |
| 15 | `prefill_sync_event.handle_event` | 102.543 | Runs prefill synchronization handler. |

Nested cumulative times overlap and must not be summed. `profile.json` retains source paths/lines, calls, own time and causes. The requested unchanged costs are:

| Function | Calls | Cumulative s | Cause |
| --- | ---: | ---: | --- |
| `ExecutionTime.__init__` | 125,807 | 48.511 | Constructs and validates layer fields/operator contracts. |
| `_build_prefill_ep_lane_batch` | 107,009 | 40.572 | EP participant lane-batch construction. |
| `build_ep_lane_batch` | 107,009 | 40.309 | Copies requests and binds immutable workload. |
| `materialize_layer_ep_workload` | 6,689 | 32.669 | Per-wave source/routing/descriptor/expert validation. |
| `build_payload_bytes` | 511,116 | 30.297 | Builds and validates communication payloads. |
| `_dsv4_ep_payload` | 428,036 | 23.813 | Validates EP routing/transfer tensor contracts. |
| `_predict_comm_operator_with_context` | 255,558 | 23.774 | Validates metadata/payload and looks up measured latency. |
| `Request.__init__` | 1,712,144 | 22.911 | Initializes lifecycle state for copied requests. |
| `scope_times` | 18,798 | 10.377 | Repeated selection/coverage/schema/exact/finite/sign checks; interpolation hits use the memo. |
| `_predict_comm_operator` | 40,076 | 4.589 | Resolves communication ownership and validates runtime payload. |

Progress rate: 0.006684455776221046 simulated seconds per wall second. Extrapolation to retained diagnostic durations gives P2-32 **21,869.96 s / 6.075 h**, and P2-64 **45,775.77 s / 12.715 h**. P2-64 assumes twice the cost per simulated second because EP ranks double; it was not measured. Estimates use the profiled initial P2-32 rate and exclude cold initialization. Later phases, cProfile overhead and shared-host variation can change the rate. Full A10 replays and client-gate readings were not run in B20.

Exact retained commands follow. Completed run roots must be preserved; use fresh roots for reruns. Scripts contain the expanded pytest selection, environment and resource limits.

```bash
W=/data/ycfeng/Frontier/.worktrees/b20-v4-speed
BASE=/data/ycfeng/Frontier/.worktrees/b20-base
S=/data/ycfeng/tmp/codex-implement/4bcc70f2-4440-43f7-9489-12ff832c399a/b20
R=$W/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b20
PY=/data/ycfeng/envs/frontier-py310/bin/python
export PYTHON_BIN=$PY PYTHONDONTWRITEBYTECODE=1

bash "$R/run.sh" "$BASE" "$S/base-cold" > "$S/base-cold.log" 2>&1
bash "$R/run.sh" "$W" "$S/candidate-final-cold" > "$S/candidate-final-cold.log" 2>&1
bash "$R/run_window.sh" "$S/candidate-final-cold" "$S/window" > "$S/window.log" 2>&1
# The proof root's cache was copied from the completed cold candidate in scratch.
B20_TIMEOUT=3600 bash "$R/run.sh" "$W" "$S/candidate-dp-proof" --observe-dp > "$S/candidate-dp-proof.log" 2>&1

"$PY" "$R/compare_runs.py" "$S/base-cold" "$S/candidate-final-cold" "$R/comparison.json"
"$PY" "$R/verify_final_dispatch.py" "$S/candidate-final-cold" "$S/candidate-dp-proof" "$R/final-dp-dispatch.json"
"$PY" "$R/profile_report.py" "$S/window" "$S/base-cold" "$S/candidate-final-cold" "$R/profile.json"

bash "$R/run_checks.sh" > "$S/regression.log" 2>&1
bash "$R/run_checks.sh" "$W/tests/unit/test_v4_prediction_memo.py" > "$S/fitted-memo.log" 2>&1
bash "$R/run_checks.sh" "$W/tests/unit/test_v4_prediction_memo.py" "$W/tests/unit/test_dsv4_source_execution.py" "$W/tests/unit/test_dsv4_structural_execution.py" "$W/tests/unit/test_dsv4_structural_pd.py" > "$S/final-check.log" 2>&1
bash "$R/run_checks.sh" "$W/tests/unit/test_dsv4_operator_consumer.py::test_both_training_paths_and_warm_cache_exact_identity" > "$S/operator-rerun.log" 2>&1
bash "$R/run_smoke.sh" > "$S/smoke.log" 2>&1
B20_REPO="$BASE" B20_TEST_TMP="$S/base-audit-tmp" bash "$R/run_checks.sh" "$BASE/tests/unit/test_v4_attention_coverage.py::test_package03_plan_json_frozen" "$BASE/tests/unit/test_pdd_public_surface_docs.py" > "$S/base-audit.log" 2>&1

RETAINED=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json
PYTHONPATH="$W:$W/tests/unit" "$PY" "$R/verify_retained_plan.py" "$RETAINED" "$R/retained-plan-check.json" > "$S/retained-plan-audit.log" 2>&1
"$PY" "$R/verify_static.py" "$W" "$BASE" "$S/candidate-final-cold" "$R/static-checks.json"
```

Final module sizes: workload 948 lines, `deepseek_v4.py` 706, `dsv4_predictor.py` 345, new memo tests 174; unchanged allowed operator predictor 386 and training module 58. Every new record Python script is below 200 lines. Static verification uses filesystem comparisons rather than Git. Logs and raw `profile.pstats` remain in scratch; durable JSON files retain results and relevant hashes.

**4. Remaining work and decisions for Claude.** Required failure: `tests/unit/test_v4_attention_coverage.py::test_package03_plan_json_frozen`. Its fixture is `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json`. Observed: absent in both candidate/base; both raise `FileNotFoundError`. The retained source at the path above contains 11,881 bytes, SHA-256 `8bb82d07eedaea7382bb39d25e3356417fee30ce7121bbd480beff711f406011`. The same JSON-equality and exact-serialization assertions pass against that read-only copy. This supplementary result does not change the required pytest verdict. Recommended action: Claude restore the retained fixture in the acceptance worktree and rerun that test. The evidence indicates an unavailable fixture, not a production regression.

The supplemental README failures reproduce at the base and are outside B20 scope. The known DP late-join failure may remain under the assignment. No materially different behavior/interface choice was introduced. Full P2-32/P2-64 replays and gate readings remain lead work. Caches are intentionally unbounded; the RSS limit is demonstrated for this scenario/window, not arbitrary long traces. Future configuration/model mutation or concurrent access would require separate cache-lifetime/concurrency decisions.

**5. Review handoff.** Inspect the narrow balanced decorators, lazy predictor-owned facts and preserved per-call validation, and prediction-only attention memo. Review two-replica/all-layer tests and counted fitted-model calls. Integrate B19 around the untouched coverage block. Review `comparison.json`, `final-dp-dispatch.json`, `profile.json` and `checks.json` for evidence and limitations. After restoring the fixture, rerun the frozen-plan test and default required `run_checks.sh`; after combining B19, rerun memo/coverage/source/structural checks and the dummy smoke matrix. Use fresh scratch roots for identity/window reruns. No acceptance, commit or task-record update by Claude is claimed.

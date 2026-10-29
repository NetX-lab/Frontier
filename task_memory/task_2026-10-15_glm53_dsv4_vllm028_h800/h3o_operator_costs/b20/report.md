B20 implementation and verification report — October 29, 2026

**1. Assignment and outcome.** Cache four repeated pure computations on the DeepSeek-V4 operator path used by the A10 P2-32 and P2-64 replays, preserving results against base commit `b6c288588ae2a19c0b86620d782c8a3d838cae9d`. Outcome: **partial**. The implementation, unit coverage, cold identity comparison, final DP dispatch proof, speed threshold, profile and dummy smoke matrix are complete. One required regression test cannot read a retained fixture absent in both worktrees and outside writable scope. Final review and acceptance remain with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/b20-v4-speed`. Untouched base: `/data/ycfeng/Frontier/.worktrees/b20-base`. Allowed source paths are `frontier/moe_ep_workload.py`, `frontier/execution_time_predictor/deepseek_v4.py`, `frontier/execution_time_predictor/dsv4_operator_predictor.py`, `frontier/execution_time_predictor/dsv4_predictor.py`, and `frontier/execution_time_predictor/dsv4_training.py`; also new `tests/unit/test_v4_prediction_memo.py` and `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b20/`. Outputs and caches reside only in private scratch `/data/ycfeng/tmp/codex-implement/4bcc70f2-4440-43f7-9489-12ff832c399a/b20`.

The working root's `AGENTS.md` was applied: preserve numeric and intermediate results, use existing component boundaries, and keep touched Python modules at or below 2,000 lines. No applicable nested `AGENTS.md` or `CLAUDE.md` was found. The B18 assignment/report, A10 driver, fidelity comparator, named workload/predictor modules, V4 construction/training paths and lead probe drivers were inspected. B18 remains the base; B19 is excluded. No unrelated pre-existing edits were encountered in the changed source paths. No Git operation, commit, dependency installation or upgrade, GPU operation, deployment, profiling-table edit, scheduler/entity edit or delegation was performed.

Settled decisions preserved: both balanced functions use unlimited typed LRU caches; layer facts belong to the fixed owning V4 predictor; the attention memo belongs to each attention predictor and keys by model name and feature tuple; every batch validation and the attention coverage/schema/exact/finite/sign checks remain per call. Model and replica configuration, and fitted models, are assumed fixed for the lifetime of their owning predictor, as specified by the assignment. There is no module-global dictionary keyed by object identity.

**2. Changes.** Three production files changed; the other allowed production files remain byte-identical to the base.

| Changed file | Symbols / lines | Behavior and reason |
| --- | --- | --- |
| `frontier/moe_ep_workload.py` | import at 13; `materialize_balanced_ep_workload` at 888 and `materialize_balanced_rank_classes` at 909 | Add only `lru_cache(maxsize=None, typed=True)` to the two pure balanced materializers. Immutable workloads/rank tuples are reused across EP ranks and layers. Exact integers cannot share a hit with bool or NumPy integer arguments, so invalid types still reach validation. `materialize_layer_ep_workload` stays uncached. |
| `frontier/execution_time_predictor/deepseek_v4.py` | `DeepSeekV4LayerFacts` at 122; `_layer_facts` at 160; `predict_source_execution` at 168 | Lazily attach configuration facts to one V4 serving predictor: immutable operator-name sets per layer, shared-expert contracts per layer, and TP per operator. Reuse these at the existing source prediction sites. Existing layer-ID validation also runs on hits to reject numeric aliases. Other per-batch checks remain in their existing order. Non-V4 execution never creates these facts. |
| `frontier/execution_time_predictor/dsv4_predictor.py` | memo at 309; `scope_times` at 311, interpolation branch at 335 | Skip fitted `model.predict` for a seen `(model_name, feature_tuple)`. Exact lookup and validation still execute each call, including memo hits. The coverage block is untouched to keep the B19 integration conflict local. |
| `tests/unit/test_v4_prediction_memo.py` (new) | materializer tests at 34; attention tests at 75; layer facts at 123 | 350 cases cover cached/uncached equality across populations 0–6,144, top-k 1/2/4, experts 4/16/256 and EP 1/2/4; invalid bool/NumPy integer arguments after hits; counted calls to fitted two-tree forests; feature/model/instance isolation; repeated coverage and finite/sign checks; per-replica facts and every model layer; cache-hit layer-ID validation. |

No predictor construction-site edit was necessary: lazy ownership is within `deepseek_v4.py`. `dsv4_operator_predictor.py`, `dsv4_training.py` and `sklearn_execution_time_predictor.py` remain unchanged. `ExecutionTime`, `Request`, EP lane batching, communication payload validation and scheduler wave logic remain unchanged. There was no behavior or scope deviation. Test harness adjustments to place default `outputs/` in scratch are recorded below.

Every new file under the record directory is listed here:

| New record file(s) | Purpose |
| --- | --- |
| `generate_trace.py` | Generate and hash the eight-request identity trace. |
| `run_replay.py`, `run.sh` | Run the base/candidate A10 configuration with scratch output, limit/timeout, timing/RSS receipts, and read-only dispatch observers. |
| `run_checks.sh`, `run_smoke.sh` | Reproduce the required pytest selection and 12-case dummy smoke matrix with the CPU interpreter and scratch storage. |
| `run_window.sh` | Copy a completed identity run's predictor cache, retain copy hashes, and run the 300-second P2-32 window. |
| `compare_runs.py`, `comparison.json` | Strict parsed artifact comparison with repository/output path substitutions and predictor-cache name classification. |
| `verify_final_dispatch.py`, `final-dp-dispatch.json` | Verify metadata after native DP synchronization and compare the supplementary warm observer run with the cold candidate artifacts. |
| `profile_report.py`, `profile.json` | Retain top 15 functions, remaining specified costs, RSS deltas, identity speedup and replay estimates. |
| `verify_retained_plan.py`, `retained-plan-check.json` | Independently check the missing fixture's retained source without writing outside scope or relabeling the failed required test. |
| `verify_static.py`, `static-checks.json` | Check syntax, sizes, changed-source whitespace, JSON parsing and source hashes against the measured candidate. |
| `environment.json`, `checks.json`, `report.md` | Environment, exact retained commands/results/log hashes, limitations and this report. |

**3. Verification.** Existing CPU interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6, GCC 11.3.0. Packages: NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, joblib 1.6.0, pytest 9.1.1. Host: Intel Xeon Gold 6330 at 2.00 GHz, effective CPUs 0–63. `PYTHONDONTWRITEBYTECODE=1`, W&B disabled, BLAS/OMP/MKL threads each 1; `TMPDIR` and `FRONTIER_TMP_ROOT` point to scratch. Simulations use `ulimit -v 62914560` and `timeout 14400s`; the supplementary warm observer uses 3,600 s. Pytest disables cacheprovider and uses scratch basetemp, output cwd and a read-only link to the repository data. The profile and base run overlapped on the shared CPU host; these are single measurements, not paired repeated benchmarks.

| Acceptance item | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| 1: new memo tests | Equality, isolation and repeated validation | 350 passed in 2.17 s | PASS |
| 2: cold base/candidate identity | Every parsed artifact equal, no cache name difference | Six artifacts equal; 1,758 entries per cache (879 pickle files and 879 locks); no differences | PASS |
| 2: scenario coverage | Both PREFILL lanes, PIECEWISE NIXL tails, FULL multi-request decode, DP dummy | Final DP snapshots: lanes 0 and 1; 8 first tails, all PIECEWISE; 114 multi-request FULL forwards; 138 dummy forwards | PASS |
| 3: 300-second P2-32 window | At least 1.5 s simulated; peak RSS growth below 1 GB | 2.0053377476662515 s in 300.0001518148929 s; probe growth 209,604,608 bytes over cold base | PASS |
| 4: profile and estimates | Top 15 with causes and specified unchanged costs | Tables below and `profile.json` | PASS |
| 5: required regressions plus new B18 tests | Required suite complete | Initial run 851 passed, 2 skipped, 3 failed in 471.85 s. Both default-output failures passed on scratch-cwd reruns; missing plan fixture remains | FAIL: unavailable prerequisite |
| 5: targeted final source/structural/memo checks | Passing final changes and output fix | 460 passed in 18.15 s; operator training/cache parity separately 1 passed in 191.00 s | PASS |
| 5: dummy smoke matrix | 12/12 with explicit interpreter | 12/12, exit 0 | PASS |
| 6: records and size | Durable scripts/results/report; touched modules <=2,000 lines | All listed records present; AST, shell syntax, JSON and changed-source whitespace checks pass; measured source hashes match | PASS |

The default required `run_checks.sh` selection contains every 16 named regression file from the assignment, the new memo test, and B18's `test_dsv4_graph_dispatch.py` and `test_dsv4_a10_tools.py`; B18's changed `test_dsv4_dp_sync.py` is already in the required selection. Counts from different executions overlap and must not be added. Two skips concern the optional pinned native balanced-routing registry (`VLLM_BS_SOURCE_ROOT`) and unavailable retained P1h GPU evidence. Three pandas mixed-type warnings remain in the successful operator cache-parity rerun.

The initial operator-parity and EP-expansion tests failed because existing fixtures instantiate default `MetricsConfig()` and attempted to create `outputs/` in the read-only repository cwd. The final runner instead executes in scratch with read-only access to `data`; the affected tests passed without production or test-fixture changes. An earlier targeted run had 459 passes and the same output error before the cwd correction. The missing `test_package03_plan_json_frozen` fixture failure remains distinct.

The missing fixture and three supplemental README assertions reproduce on the untouched base: 4 failed and 2 passed in 2.11 s (`base-audit.log`). A supplementary exploratory release/graph/dummy/DP run had 99 passed and 4 failed in 80.59 s: the three README failures plus the assignment-known DP late-join failure, lane 2 `(0, False)` instead of `(0, True)`. Its historical suite-selection command was not retained; that run is diagnostic evidence only and is not counted as a required green check. The exact repeatable base audit is retained below. No README or DP fix was made.

Identity configuration and evidence: A10's public usage is `a10_driver.py <case> <output root>` and `build_config` has no extra-field parameter. The permitted fallback `h4a4.build_config` is called with the same A10 case and `a10.operator_fields`, dropping the same diagnostic table/override entries as A10. Identity adds only the trace-file field. Both roles use fitted operator costs, dummy mode is false, and no step-cost table is bound. The trace in each run root has arrivals `0.00, 0.05, ..., 0.35`, prompt 6,144, output 48, SHA-256 `cec2c473c51d43518ac4bb0781ae3d020e63f7a841297b4ff268d2a97d962186`.

`compare_artifact_directories` with `substitutions_for` reports no difference in `config.json`, `frontier_transfer_ledger.jsonl`, `metrics_ground_truth.jsonl`, `op_precision_metadata.csv`, `request_metrics.csv` and `system_metrics.json`. `classify_cache_differences` reports none. Neither run creates `cc_cache`; this absence is recorded on both sides rather than counted as a produced artifact. Local scheduler dispatch observer records are also byte-identical. Because DP synchronization can revise descriptors, a separate warm replay recorded final metadata **after** `_synchronize_dp_descriptors` returned; its six artifacts, cache names and local observer also exactly match the cold candidate. Final coverage counts above are from those final snapshots, not merely local pre-synchronization dispatch.

| Run root under scratch | Initialization s | `Simulator.run` s | Simulated s | Peak RSS KiB |
| --- | ---: | ---: | ---: | ---: |
| `base-cold` | 510.09502797201276 | 1139.313491472043 | 2.702534436729753 | 997996 |
| `candidate-final-cold` | 520.145190033596 | 170.79701200406998 | 2.702534436729753 | 998380 |
| `candidate-dp-proof` (supplementary warm observer) | 54.55038211029023 | 171.63454188778996 | 2.702534436729753 | 1202668 |
| `window` (warm, profiled P2-32) | 55.903996157925576 | 300.0001518148929 | 2.0053377476662515 | 1202688 |

Cold identity `Simulator.run` speedup is 6.670570392911171x. Candidate cold peak RSS increases 393,216 bytes (384 KiB); window peak increases 209,604,608 bytes relative to the cold base. RSS is `resource.getrusage(RUSAGE_SELF).ru_maxrss`, the peak of the whole process including initialization. The correct probe peak is **1,202,688 KiB**. Warm cache copy provenance and hashes are in `window/warm_cache_copy.json`. Two preliminary aborted initialization attempts (`base/`, `candidate-cold/`) never reached `Simulator.run` and are excluded from timing and identity results.

Top 15 functions by cumulative time in the 300-second window follow. Nested cumulative times overlap and must not be summed; full paths, lines, calls and own time are in `profile.json`.

| # | Function (source) | Cumulative s | Cause |
| --- | --- | ---: | --- |
| 1 | `run` (`frontier/simulator.py:423`) | 300.000 | Top-level DES run, including the profiled sequential event loop. |
| 2 | `_run_sequential` (`frontier/simulator.py:595`) | 300.000 | Processes the ordered event queue and invokes handlers. |
| 3 | `enter_layer_sync` (`frontier/scheduler/utils/sync_entry.py:34`) | 251.600 | Collects attention-DP participants and starts each ready EP layer wave. |
| 4 | `schedule_layer_wave` (`frontier/scheduler/utils/ep_wave_schedule.py:52`) | 249.722 | Schedules each layer's EP plan, phases and completion events. |
| 5 | `_prepare_moe_ep_wave_plan` (`frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py:1035`) | 247.714 | Delegates immutable workload and per-rank phase planning for each layer wave. |
| 6 | `prepare_moe_wave_from_inputs` (`frontier/scheduler/utils/ep_wave.py:80`) | 247.651 | Materializes the scheduler workload and all lane timings for each wave. |
| 7 | `predict_ep_wave_phase_times` (`frontier/scheduler/utils/expert_parallel.py:91`) | 212.337 | Builds and prices every EP rank, including shared wrapper and communication work. |
| 8 | `_predict_layer_with_kernel_gap` (`frontier/execution_time_predictor/kernel_gap.py:209`) | 151.065 | Prices each layer then validates/applies per-operator kernel gaps. |
| 9 | `handle_event` (`frontier/events/decode_sync_event.py:36`) | 149.567 | Runs the event-specific synchronization, admission or completion handler. |
| 10 | `on_decode_sync` (`frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py:1282`) | 149.288 | Passes a DECODE lane's layer completion to the shared synchronization boundary. |
| 11 | `enter_decode_sync` (`frontier/scheduler/utils/sync_entry.py:197`) | 149.256 | Registers DECODE lane completion and starts a ready cohort's next layer wave. |
| 12 | `_on_decode_ep_wave_ready` (`frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py:1109`) | 148.237 | Schedules ready DECODE EP waves and their continuation events. |
| 13 | `_predict_dsv4_structural_execution` (`frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:95`) | 144.818 | V4 prediction entry for each attention lane and every routed EP rank. |
| 14 | `predict_structural_execution` (`frontier/execution_time_predictor/deepseek_v4.py:342`) | 143.798 | Constructs complete layer timing, operator workloads, runner scopes and communication payloads. |
| 15 | `handle_event` (`frontier/events/prefill_sync_event.py:36`) | 102.543 | Runs the event-specific synchronization, admission or completion handler. |

The specified shared costs remain visible and unchanged:

| Function | Calls | Cumulative s | Cause |
| --- | ---: | ---: | --- |
| `execution_time.__init__` | 125,807 | 48.511 | ExecutionTime constructs and validates a finalized layer's many fields and operator contracts. |
| `_build_prefill_ep_lane_batch` | 107,009 | 40.572 | Copies source requests and constructs a lane batch for each EP participant. |
| `build_ep_lane_batch` | 107,009 | 40.309 | Copies source requests into one EP lane batch and binds its immutable workload. |
| `materialize_layer_ep_workload` | 6,689 | 32.669 | Validates source/routing rows, lane descriptors and all expert ownership for each scheduler wave. |
| `build_payload_bytes` | 511,116 | 30.297 | Builds and validates each communication operator payload using runtime source/routing rows. |
| `_dsv4_ep_payload` | 428,036 | 23.813 | Validates the V4 EP routing and transfer tensor contract per payload request. |
| `_predict_comm_operator_with_context` | 255,558 | 23.774 | Validates communication metadata/payload then looks up the measured backend latency. |
| `request.__init__` | 1,712,144 | 22.911 | Request copies made by EP lane batching initialize full request lifecycle state. |
| `scope_times` | 18,798 | 10.377 | Repeats attention model selection, coverage/schema/exact/finite/sign checks; interpolation hits use the instance memo. |
| `_predict_comm_operator` | 40,076 | 4.589 | Resolves communication ownership and validates each runtime payload before backend lookup. |

Measured progress is 0.006684455776221046 simulated seconds per wall second. Extrapolating to retained diagnostic durations (P2-32 146.1887597443391 s, P2-64 152.99305567235012 s) gives P2-32 **21,869.96 s / 6.075 h** and P2-64 **45,775.77 s / 12.715 h**. P2-64 assumes twice the cost per simulated second because EP ranks double; it was not measured. These estimates use the profiled initial P2-32 rate and exclude cold initialization. Later workload phases, cProfile overhead and shared-host variation can change the rate. Neither full A10 replay nor a client-gate reading was run in B20.

Commands used for the retained required checks and final comparisons are below. The scripts retain the full expanded pytest selection, environment, limit and timeout. Existing completed run roots must be preserved; use fresh roots when rerunning cold simulations or cache-copy windows.

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
# candidate-dp-proof/cache was a scratch copy of the completed cold candidate cache.
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

Final module sizes: `moe_ep_workload.py` 948, `deepseek_v4.py` 706, `dsv4_predictor.py` 345, new memo tests 174; unchanged allowed `dsv4_operator_predictor.py` 386 and `dsv4_training.py` 58. Every new record Python script is below 200 lines. Static checks use filesystem comparisons instead of Git. Test/run logs and raw `profile.pstats` remain in scratch; durable JSON records retain results, relevant values and log hashes.

**4. Remaining work and decisions for Claude.** The required fixture path is `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json`. Observed: it is absent in both candidate and base, and both required tests raise `FileNotFoundError`. A read-only retained copy exists at the `RETAINED` path above: 11,881 bytes, SHA-256 `8bb82d07eedaea7382bb39d25e3356417fee30ce7121bbd480beff711f406011`. The same JSON-equality and exact-serialization assertions pass against that copy. This is supplementary evidence and does not change the required pytest verdict. Recommended action: Claude restore the retained fixture in the acceptance worktree, then rerun that test; no production fix is indicated by this failure.

The three README assertions are outside the required B20 selection and reproduce at the base; leave their disposition to Claude. The known DP late-join failure may remain under the assignment. No design or interface decision was broadened. Full P2-32/P2-64 A10 replays and gate readings remain lead work. Caches are intentionally unbounded; the memory threshold is demonstrated for this scenario/window, not for arbitrary long traces. Configuration/model mutation or future concurrent access would need separate cache-lifetime/concurrency decisions.

**5. Review handoff.** Inspect the narrow balanced-materializer decorators, the lazy owning-predictor facts and preserved per-call validations, and the prediction-only attention memo branch. Check the two-replica/all-layer tests and counted fitted-model calls. Integrate B19 around the untouched coverage block. Review `comparison.json`, `final-dp-dispatch.json`, `profile.json` and `checks.json` for actual results and limitations. After restoring the fixture, rerun the frozen-plan test and the default required `run_checks.sh`; after combining B19, rerun the memo/coverage/source/structural tests and the dummy smoke matrix. Identity/window reruns should use fresh scratch run roots. No acceptance, commit or task-record update by Claude is claimed.

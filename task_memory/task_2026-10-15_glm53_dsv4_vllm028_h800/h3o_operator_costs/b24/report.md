# B24: distinct V4 EP predictions per wave

## 1. Assignment and outcome

**Outcome: partial.** The implementation, audit, new tests, three exact identity comparisons, smoke matrix, speed checks and 300-second profile are complete. The supplied interpreter lacks `torch`, so the required `test_moe_rank_routing.py` module and one GLM expert-parallel layout test were skipped. Those checks remain unverified; installing dependencies is outside the assignment's authorization. Final review and acceptance remain with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/b24-v4-ep-lane-reuse`. Authorized edits: `frontier/scheduler/utils/expert_parallel.py`, the new `tests/unit/test_v4_ep_lane_reuse.py`, only `test_ep_wave_uses_the_segment_gap_wrapper_for_every_lane` in `tests/unit/test_dsv4_operator_consumer.py`, and this `b24/` records directory. Caches, bytecode alternatives, logs, JUnit files and run outputs belong to `/data/ycfeng/tmp/codex-implement/9de7af1c-59f2-4ff4-a3ca-bfa1bf52e247`. No Git operations, commits, task records outside B24, dependency installations, GPU operations, external messages or deployment were performed.

The repository's root AGENTS.md was inspected, including its vLLM TP/DP/EP ownership contract and 2,000-line gate. No nested AGENTS.md or CLAUDE.md was found on the edited paths. The assignment, B20 generators/replay/comparison tools, A10/H4a-4 config owners, the lead launcher/driver/profile and the required prediction sources/tests were inspected. Source text did not broaden the assignment.

Settled behavior: optimize only `source_batches is not None` with `capture_lane_timings=False`; use the DP source index plus every named EPLaneWorkload field except `ep_id` and canonical `owned_expert_ids`; fail when the dataclass schema gains an uncovered field; retain first-participant prediction order; share only the five validated phases within this call. Lane construction, SP assignment, workload logging and six output tuples stay per participant. Lane-record and non-V4 paths keep separate predictions.

The supplied base is `/data/ycfeng/Frontier/.worktrees/b26-base` at the user-declared `627253d5`; the candidate started at user-declared `a420db84`. A byte comparison found 509 candidate `frontier/` files with only the intended `expert_parallel.py` difference and 220 `data/` files with no differences. Git commit identities were not independently queried because Git operations belong to Claude. No pre-existing production changes were encountered in that comparison; B24 was empty when inspected. B26's concurrent working tree was not used as the base.

## 2. Changes and audit

- `frontier/scheduler/utils/expert_parallel.py:91`: added `_V4_EP_LANE_KEY_FIELDS` and `_v4_ep_lane_prediction_key`. The key explicitly covers `moe_expert_parallel_size`, `total_expert_num`, `local_token_counts`, `global_token_counts`, `routed_token_count`, and `router_topk`, and compares this coverage with `dataclasses.fields(lane_workload)`. It uses values, with no object identity key.
- `frontier/scheduler/utils/expert_parallel.py:114`: `predict_ep_wave_phase_times` creates a local phase memo only for the selected branch. A miss runs the existing kernel-gap wrapper, stage assembly and phase getter; a hit reuses the five phases. The first matching participant remains the predictor caller. Final module length: **867 lines**.
- `tests/unit/test_v4_ep_lane_reuse.py` is new. It reuses the trained H200 V4 fixture, exercises TP8/DP2/EP16 and TP8/DP4/EP32 with one 6,144-token prefill source plus dummy sources and decode sources with different request counts. Layers 0, 3 and 42 cover ordinary and endpoint behavior. It checks exact `==` of all six numeric tuples, prediction/assembly/phase-getter counts and first-rank order, per-participant logging and lane construction, SP assignments, source-rank metadata with records on, memo lifetime across calls, and an uncovered added dataclass field.
- `tests/unit/test_dsv4_operator_consumer.py:287`: only the assigned test changed. It checks every distinct prediction passes the segment-gap wrapper and every participant receives its group's routed time. An AST comparison confirms no other top-level node in that file changed.

The audit found no extra rank-dependent phase input. The initial new decode fixture failed six cases because it copied `original_total_tokens=16` onto batches with fewer scheduled rows. The validator failed before prediction. `dataclasses.replace` now sets the true original token/request count while retaining the FULL descriptor's padded width; the final 13 new tests pass. Production design and interfaces did not change beyond the agreed reuse.

The following audit is also retained as `audit.md`:

Audit scope: the `source_batches is not None` operator branch, `include_attention=False`, `include_ffn=True`, `skip_padding=False`, with one fixed predictor and layer per wave. References are to the starting source except `expert_parallel.py`, whose helper moves the original function down. No unaccounted source-rank or destination-rank input to the five phase times was found.

| Site | Reads of rank, workload, or batch | Effect on phase time versus metadata | State writes / reuse implication |
| --- | --- | --- | --- |
| `expert_parallel.py:128`, `predict_ep_wave_phase_times` | Participant order; `ep_id`; `source_batches[ep_id // attn_tp]`; `ep_id % attn_tp`; lane batch workload and SP widths | Source DP selects all source costs; source TP rank only reaches the source-row metadata below. Destination workload selects grouped FFN. | A call-local dictionary retains only the five validated floats. First encounter preserves prediction order. |
| `batch_builders.py:37`, `build_ep_lane_batch` | `layer_workload.lane(ep_id)`, local expert counts, source replica/time/IDs/global ID/effective tokens/AFD stage | Lane-batch geometry and identifiers are needed for logging/reporting. V4 prediction receives the source batch rather than this lane batch. | Creates `Request` and `EPBatchGroup` objects and their IDs for every participant, copies source metadata, assigns global ID. These calls remain per participant. |
| SP assignment in `predict_ep_wave_phase_times` | `sp_token_widths.local_tokens[ep_id // replica.attn_tensor_parallel_size]` | Source DP width is a potential prediction input; the explicit wave widths, not source TP valid widths, price communication. | Assigns lane batch `moe_sequence_parallel_token_widths` and `moe_pre_routing_effective_total_tokens` for every participant. |
| `kernel_gap.py:209`, `_predict_layer_with_kernel_gap` | Source batch at `args[0]`; layer family; configured gap table | Source graph mode sets segment gaps. No lane-batch or rank read. | Pushes fresh `LayerKernelCounts` on `_kernel_count_scopes`, restores it in `finally`. V4 uses explicit operator kernel counts instead of generic accumulated counts. |
| `sklearn_moe_execution_time_predictor.py:95`, `_predict_dsv4_structural_execution` | Batch/layer/rank/context forwarded; `_dsv4_step_costs` | Dispatch only. A10 has no step-cost binding; the separate step-cost EP shortcut already omits this work with records off. | None. |
| `deepseek_v4.py:342`, `predict_structural_execution` | Workload topology (`moe_expert_parallel_size`, `total_expert_num`, `router_topk`) | Validates topology against replica; all covered by key. | None on predictor; builds local time/workload/count dictionaries. |
| `deepseek_v4.py:168`, `predict_source_execution` source validation | Source batch type, request presence, idle state, spec/AFD metadata; query/context/decode vectors and aggregate counts | Validation only on this FFN-only branch. Attention geometry is not priced. Same source object within each DP group. | None; `_validate_source_batch` and `compute_rows` are read-only. |
| `predict_source_execution` TP rank | Validates `tp_rank`; `sequence_parallel_moe_token_widths([tokens], tp, rank)` | `local_tokens[0] = ceil(tokens / tp)` does not depend on rank. `valid`, `padding_mask_rows` depend on rank but are metadata; structural `skip_padding=False` rejects masked routing. | No persistent writes. |
| `predict_source_execution` source physical/actual rows | `sum(batch.num_tokens)`, `compute_rows(batch)`, source graph original/padded rows, last-layer `selected_logits_rows` | Physical local and descriptor rows price mHC/shared operators and final endpoints; actual rows validate logits bounds and compute metadata. Same DP source preserves them. | None. |
| `deepseek_v4.py:622`, `serving_context` | Source graph runtime mode and `len(batch.requests)` at last layer | Execution mode and selected logits rows affect source scopes; included by retaining source DP index. No rank reads. | None. |
| `predict_source_execution.local_scope` / `DeepSeekV4LayerFacts` | Source local/descriptor/logits rows; model manifest/shared TP/operator TP | These price selected physical source operators; no source TP rank or expert identity in any coordinate. | Lazily creates `_dsv4_layer_facts` and fills `_operator_names`, `_shared_contracts`, `_operator_tp_sizes` with configuration facts. First group trains/initializes in the original order. |
| `dsv4_operator_predictor.py:353`, `operator` / `_exact` | Operator name, selected TP, source rows; exact time/kernel-count model dictionaries | Exact lookup returns time and kernel count. | Read-only model lookup; no prediction counters, runtime-cache updates, training, or sklearn `.predict` on this path. |
| `dsv4_runtime.py:61`, `add_runner_scopes` | Source dummy type and last layer; source role and request count through `sampling_coordinate(batch)` | With attention excluded, only final-layer non-dummy sampling can run. Source role/request count determine price. Zeroing/block-table/input-preparation branches are disabled. | Writes local time/count/workload dictionaries, not predictor or batch state. |
| `dsv4_operator_predictor.py:356`, `gate` | Source `physical_local_rows` only | Exact gate time and count; independent of rank and destination workload. | Read-only exact lookup. |
| `dsv4_operator_predictor.py:359`, `grouped_gemm` | `global_token_counts`, `router_topk`, topology, `ep_id` selecting balanced rank class/measured lane | Coordinate is `(EP, population, local_routed_rows, experts_with_extra_row, base_rows)`. For validated balanced lanes the last three are determined by local/global count tuples. `ep_id` additionally selects the lane used for global-vector validation; matching vectors and topology give the same validation result. `owned_expert_ids` is canonical and does not enter pricing. | Global `materialize_balanced_ep_workload` and `materialize_balanced_rank_classes` LRU caches gain the same population/topology entries; hit counters decrease with reuse, with no numeric effect. `LayerEPWorkload.lane` reads its already materialized descriptor tuple. |
| `predict_structural_execution` routed row workload | `lane_workload.routed_token_count` | Grouped-GEMM input/output shapes and `synthetic_destination_assignments` metadata. Key includes the count even though the measured grouped price is selected through rank class. | Local metadata only. |
| Source communication (`predict_source_execution.communication`) | Source batch, model/replica/role through `CommPayloadContext`; local source rows at final mHC gather | FFN-only branch has final mHC TP gather at the final layer. Rank-free source geometry determines payload and cost. | Communication predictions described below. |
| Structural EP communication | Wave `sp_token_widths` (ordered `local_tokens`, `gathered_tokens`), model hidden width, top-k, dtype and topology | Dispatch uses largest SP local input; combine uses gathered rows and exact ordered SP row tuple for unequal measured combine. No source TP rank, destination EP ID, or owned expert IDs. | No payload-validation changes; first prediction of every group still validates its complete communication payload. |
| Structural final logits communication | Source `len(batch.requests)` / selected logits rows, vocabulary, attention TP | Source-DP-specific endpoint payload and time. | Read-only. |
| `sklearn_execution_time_predictor.py:6279/6298`, communication wrappers | Payload context source batch, role, model/config, wave widths | Size/domain/device count/implementation are rank-free. Zero payload policy and measured range validation unchanged. | A10 measured route reads `_cc_backend` only. Legacy fallback allgather/reduce-scatter wrappers delegate to the configured backend or configured dummy constant. |
| `measured_cc_backend.py:250`, `predict_operator_collective` | Alias, implementation, size, devices/domain, optional ordered SP rows | Read-only curve interpolation or exact unequal-combine lookup. No rank input. | No prediction caches/counters or training writes. |
| `predict_structural_execution` source metadata | `tp_rank`, source valid rows/dummy kind/graph mode/ID/request IDs; routed count | `_dsv4_source_rows`: rank-dependent `valid_local_rows`, `request_valid_local_rows`, `source_tp_rank`; all other source metadata fixed per DP group. Metrics require individual objects only when records are captured. | Writes mapping proxies to the new timing object. With records on, every participant still predicts. |
| `dsv4_runtime.py:54`, `add_dsv4_operator_gaps` | Source batch graph mode, layer op times/counts/scope gaps, configured gap table | Phase costs include count × segment gap. All inputs are fixed per key/source DP. No rank read. | Writes `_dsv4_operator_gap_ms` on the new timing object only. |
| `base_execution_time_predictor.py:81`, `_assemble_stage` | Layer result, model-owned layer identities | One-layer wrapping/finalization preserves numbers and metadata. No batch/workload/rank reads. | No predictor writes. Timing construction advances the `ExecutionTime` class ID counter once per prediction; internal stage copies do not allocate IDs. Skipping discarded timings changes that internal counter, not Request/Batch counters or phase values; artifact identity checks verify observable output. |
| `get_ep_phase_times_ms` and output/logging | Five stage accessors, total post-attention time; `ep_id` only in error context; participant's own lane workload | Validates finite/nonnegative phases and conservation once per prediction group. Six returned tuples retain every participant in order; logger receives each participant's own workload. | No predictor writes. `StageExecutionTime` may memoize its own aggregate model time on access; no stage object is retained for a reused lane. |

Excluded calls: attention feature/prediction and `compose_trained_attention` do not run with `include_attention=False`; zeroing, KV-table writes and preparation scopes are gated by `include_attention=True`. The generic `_record_kernel_count` runtime-cache insertion is not reached by V4 exact operator or measured communication lookups. Trained models are initialized before `Simulator.run`; any configuration-fact/cache miss still occurs at the first original participant for that group.

## 3. Verification and exact commands

Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, **Python 3.10.6**; Linux 5.10.25, 64 reported CPUs; local date October 30, 2026, timezone Asia/Hong_Kong. Simulation/test launches use `ulimit -v 62914560` and timeouts; analysis uses `ulimit -v 31457280`. Environment:

```bash
REPO=/data/ycfeng/Frontier/.worktrees/b24-v4-ep-lane-reuse
B24=/data/ycfeng/Frontier/.worktrees/b24-v4-ep-lane-reuse/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b24
SCRATCH=/data/ycfeng/tmp/codex-implement/9de7af1c-59f2-4ff4-a3ca-bfa1bf52e247
PY=/data/ycfeng/envs/frontier-py310/bin/python
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$SCRATCH/tmp" FRONTIER_TMP_ROOT="$SCRATCH/tmp"
export PYTHONPATH="$REPO"
```

Scripts `run_one.sh`, `run_pair.py`, `run_regression.py`, `run_additional.py` and `run_smoke.sh` retain the expanded environment, limits, timeouts and output ownership. JSON command records retain exact expanded subprocess arguments. Base runs use their own repository cwd and PYTHONPATH.

| Acceptance item | Expected | Observed | Status |
| --- | --- | --- | --- |
| 1: new tests and changed consumer test | Exact timings, correct counts/logging, schema guard; changed test passes | 13 new tests passed; changed consumer test passed in the initial targeted run and the regression run | PASS |
| 2 S1: P2-32, eight requests, cold | No parsed artifact or cache-name differences | No metrics/CC/cache-name differences; dispatch observer equal; both end at 2.702498831804029 simulated s | PASS |
| 2 S2: P2-64, eight requests, independent warm copies | Exact artifacts/cache names; demonstrate sharing | No differences; 22,102 predictions for 176,128 participants over 5,504 waves; records off | PASS |
| 2 S3: S1 with operation metrics and warm S1 cache copies | Exact artifacts/cache names with records on | No differences; source config has store_operation_metrics=True and write_metrics=True, which enables lane reporting | PASS |
| 3: S1/S2 run wall time and peak RSS | Measured base/candidate | Table below; S1 1.725x and S2 2.116x speedup | PASS |
| 3: full P2-64, simultaneous 900-second unprofiled windows from time 0 | Candidate simulated progress >=1.5x base | 17.403888793087 / 8.635213637374 = 2.015455x | PASS |
| 4: candidate 300-second profile and replay estimates | Top 15 cumulative functions, causes, estimates | Profile table and estimate table below; measured 300.000140 s window | PASS |
| 5: regression selection | Named suites and all matching test names run | 882 passed, 2 skip entries; 14 additional parameter-name matches passed. Required torch-dependent module/function unverified | PARTIAL / NOT_RUN for skips |
| 5: dummy smoke | All 12 architecture/mode/density cases | 12 of 12 completed successfully | PASS |
| 6: records and module gate | B24 scripts/comparisons/audit/report; <=2,000 lines | Records retained here; source 867 lines | PASS |

Executed test and smoke commands:

```bash
cd "$REPO"
ulimit -v 62914560
timeout 600s "$PY" -m pytest tests/unit/test_v4_ep_lane_reuse.py -q -p no:cacheprovider \
  --basetemp="$SCRATCH/tmp/lane-tests-final" --junitxml="$SCRATCH/lane-tests.xml"
timeout 1900s "$PY" "$B24/run_regression.py" --scratch-cwd
timeout 700s "$PY" "$B24/run_additional.py"
bash "$B24/run_smoke.sh"
```

The first targeted command also included `tests/unit/test_dsv4_operator_consumer.py::test_ep_wave_uses_the_segment_gap_wrapper_for_every_lane`, using `--basetemp="$SCRATCH/tmp/lane-tests"`; it produced 6 fixture failures and 8 passes. After correcting the fixture, the final new-test command produced 13 passes. Regression includes the changed consumer test and all ten named suites. The selection is recorded in `regression_targets.json` and expanded in `regression_candidate_scratch_command.json`.

The initial repository-cwd regression command (`timeout 1900s "$PY" "$B24/run_regression.py"`) produced 839 passes, 43 failures and 2 skips. Every failure was `OSError: [Errno 30] Read-only file system: 'outputs'`. The base command with `--base` reproduced the **same 43 node IDs and messages**, recorded in `verification_results.json`; no failure was caused by reuse. The final run uses a scratch cwd, absolute test paths, repository PYTHONPATH, and a scratch `data` symlink pointing to read-only repository data. This keeps default outputs/caches in authorized scratch without editing the tests or repository output paths. It yielded **882 passed, 2 skipped**.

For parameterized name coverage, these additional executed discovery commands were used:

```bash
timeout 180s "$PY" "$B24/collect_regression.py"
timeout 180s "$PY" "$B24/collect_regression.py" --base
```

Both broad collections reported the same 22 unrelated module import errors (torch, aiohttp, matplotlib or absent native analysis tools). Their test functions and decorators contain no requested matching names; `unavailable_collection_audit.json` records that audit. The successfully collected matching nodes exposed 14 extra parameter-name cases, all executed and passed by `run_additional.py`. No dependencies were installed. The known balanced-routing I-002 and DP dummy-forward failures were not part of the selected cases and were not claimed to pass.

Executed cache preparation and replay commands:

```bash
WARM=/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_g1hs_r2/cache
ulimit -v 31457280
"$PY" "$B24/prepare_cache.py" "$WARM" "$SCRATCH/s2_base"
"$PY" "$B24/prepare_cache.py" "$WARM" "$SCRATCH/s2_candidate"
"$PY" "$B24/prepare_cache.py" "$WARM" "$SCRATCH/window_base"
"$PY" "$B24/prepare_cache.py" "$WARM" "$SCRATCH/window_candidate"
"$PY" "$B24/prepare_cache.py" "$WARM" "$SCRATCH/profile_candidate"
```

These independent copies were made concurrently; each contained 1,758 files and 324,536,393 bytes. Each pre-run file inventory includes SHA-256, including lock files, and all normalized inventories have digest `6ca1d0629afb11f23414d08fc4c8ed345f7a90f84b1dcf6a30ca6c2abfaff556`. Each source remained read-only. S1 used fresh cold roots. After S1 finished, `after_s1.py` byte-copied each side's own S1 cache into S3 and the candidate S1 cache into the extra P2-32 rate window; their inventories also match.

```bash
ulimit -v 62914560
"$PY" "$B24/run_pair.py" s1
"$PY" "$B24/run_pair.py" s2
"$PY" "$B24/run_pair.py" window
"$PY" "$B24/after_s1.py"
bash "$B24/run_one.sh" "$REPO" "$SCRATCH/profile_candidate" --case p2_64 --window 300 --profile
```

`run_pair.py` launches base and candidate together. `after_s1.py` launches `run_pair.py s3` after finished S1 caches are copied and concurrently runs the candidate's original full P2-32 trace for an unprofiled 300-second window. Each simulator subprocess has the 60-GiB virtual-memory limit and 3,600-second timeout. The profile uses the original full P2-64 trace and excludes initialization. The full trace has 128 requests, one arrival per second from 0 to 127, 6,144 prompt tokens and 1,024 output tokens. Identity runs use B20's eight arrivals at 0.00, 0.05, ..., 0.35 s and 48 output tokens; trace SHA-256 is `cec2c473c51d43518ac4bb0781ae3d020e63f7a841297b4ff268d2a97d962186`.

`--window` now controls only wall-time stopping; profiling requires explicit `--profile`, so both 900-second speed windows had no profiler. Candidate S2 alone used `--count-ep`; its lightweight observer wraps `_predict_dsv4_structural_execution` within EP-wave prediction and is absent from the base and both full speed windows. It reported 5,418 waves with four predictions and 86 waves with five predictions. All 5,504 waves had records off.

Executed comparisons and analysis:

```bash
ulimit -v 31457280
"$PY" "$B24/compare_runs.py" "$SCRATCH/s1_base" "$SCRATCH/s1_candidate" "$B24/s1_comparison.json"
"$PY" "$B24/compare_runs.py" "$SCRATCH/s2_base" "$SCRATCH/s2_candidate" "$B24/s2_comparison.json"
"$PY" "$B24/compare_runs.py" "$SCRATCH/s3_base" "$SCRATCH/s3_candidate" "$B24/s3_comparison.json"
"$PY" "$B24/profile_report.py"
"$PY" "$B24/summarize_runs.py"
```

All three comparisons use the repository's `compare_artifact_directories` with `substitutions_for` and `classify_cache_differences`. They compare config.json, frontier_transfer_ledger.jsonl, metrics_ground_truth.jsonl, op_precision_metadata.csv, request_metrics.csv and system_metrics.json; all parsed differences and cache-name findings are empty. Each side has 1,758 cache files. The forward-dispatch observers are byte-identical and show required prefill DP lanes, first-tail PIECEWISE forwards, multirequest FULL forwards and dummy forwards. No CC cache directory was produced.

Comparison JSON absolute paths:

- `/data/ycfeng/Frontier/.worktrees/b24-v4-ep-lane-reuse/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b24/s1_comparison.json`
- `/data/ycfeng/Frontier/.worktrees/b24-v4-ep-lane-reuse/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b24/s2_comparison.json`
- `/data/ycfeng/Frontier/.worktrees/b24-v4-ep-lane-reuse/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b24/s3_comparison.json`

Wall time is `Simulator.run`, including its normal final metric work, and excludes initialization. Peak RSS is process `ru_maxrss`, in KiB, including initialization.

| Scenario | Base init s | Candidate init s | Base run s | Candidate run s | Speedup | Base peak RSS KiB | Candidate peak RSS KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 550.821 | 559.387 | 171.649 | 99.503 | 1.725x | 1044868 | 1046328 |
| S2 | 74.131 | 74.072 | 232.548 | 109.900 | 2.116x | 1348524 | 1348812 |
| S3 | 76.044 | 72.511 | 233.734 | 230.507 | 1.014x | 1368524 | 1369096 |

| Full P2-64 window | Wall s | Simulated start s | Simulated end s | Peak RSS KiB |
| --- | ---: | ---: | ---: | ---: |
| Base | 900.000186 | 0 | 8.635213637374 | 1365352 |
| Candidate | 900.000180 | 0 | 17.403888793087 | 1389316 |

The candidate/base progress ratio is **2.015455x**, exceeding 1.5x. Both were launched together on the shared host; other CPU work, including concurrent B26 work, short replays and tests, was present. These are shared-host measurements, not isolated hardware benchmarks.

# Candidate 300-second cProfile window

| Function | Calls | Self s | Cumulative s | Cause |
| --- | ---: | ---: | ---: | --- |
| `simulator.py:423:run` | 1 | 0.000 | 300.000 | Total sequential event-loop execution in the measured window. |
| `simulator.py:595:_run_sequential` | 1 | 0.495 | 300.000 | Sequential event queue processing. |
| `sync_entry.py:34:enter_layer_sync` | 33788 | 0.413 | 191.694 | Collects DP lanes in a layer waiting room and launches a complete EP wave. |
| `ep_wave_schedule.py:52:schedule_layer_wave` | 8447 | 0.373 | 188.031 | Builds source-cohort inputs, predicts the wave, and schedules its lane/collective events. |
| `base_cluster_scheduler.py:1035:_prepare_moe_ep_wave_plan` | 8447 | 0.077 | 185.163 | Delegates workload, phase prediction, logging and barrier preparation. |
| `ep_wave.py:80:prepare_moe_wave_from_inputs` | 8447 | 0.414 | 185.086 | Materializes wave workload, predictions and collective barriers. |
| `kernel_gap.py:209:_predict_layer_with_kernel_gap` | 79753 | 0.749 | 131.579 | Calls selected layer predictor and adds segment-specific gaps. |
| `expert_parallel.py:114:predict_ep_wave_phase_times` | 8447 | 5.404 | 130.058 | Remaining distinct V4 predictions plus required per-participant batch construction and logging. |
| `decode_sync_event.py:36:handle_event` | 22780 | 0.388 | 130.023 | Event-handler work, including scheduling and phase prediction. |
| `base_cluster_scheduler.py:1282:on_decode_sync` | 22780 | 0.061 | 129.435 | Delegates each decode-lane arrival to the layer synchronization path. |
| `sync_entry.py:197:enter_decode_sync` | 22780 | 0.033 | 129.374 | Runs the shared layer rendezvous for a decode source lane. |
| `base_cluster_scheduler.py:1109:_on_decode_ep_wave_ready` | 5695 | 0.138 | 126.981 | Schedules the completed decode cohort through the shared wave helper. |
| `sklearn_moe_execution_time_predictor.py:95:_predict_dsv4_structural_execution` | 79753 | 0.680 | 126.823 | Dispatches V4 operator prediction for attention sources or distinct FFN groups. |
| `deepseek_v4.py:342:predict_structural_execution` | 79753 | 4.560 | 126.142 | Builds operators, collective contexts, workload metadata and timing objects. |
| `sklearn_disaggregation_execution_time_predictor.py:1391:predict_stage_execution_time` | 34075 | 0.271 | 102.155 | Source attention stage prediction and reporting assembly. |

Wall time: 300.000140 s; simulated time reached: 3.612744670137 s; peak RSS: 1352464 KiB.
Cumulative values overlap because parent functions include their callees.

The lead's 1,200-second mid-run profile was read separately; `lead_profile.json` confirms 12,214 EP waves, 838.877738 s cumulative wave prediction, and 390,848 lane builders. Its cumulative counts include additional non-EP calls, whereas the assignment's nested counts are EP-specific. The candidate profile above starts at simulated 0 and is not a direct same-phase comparison to that mid-run profile.

| Full candidate replay estimate | Unprofiled sample | Measured sim s / wall s | Historical full simulated duration s | Estimated Simulator.run h | Estimated warm total h |
| --- | --- | ---: | ---: | ---: | ---: |
| p2_32 | p2_32_rate_candidate | 0.019560847 | 147.428532227 | 2.094 | 2.114 |
| p2_64 | window_candidate | 0.019337650 | 153.266943310 | 2.202 | 2.222 |

The P2-32 sample reached 5.868256507505 simulated seconds in 300.000123 wall seconds. Estimates divide previously completed A10 g1hs_r2 simulated durations by these early unprofiled rates; warm totals add measured initialization. `speed_results.json` records the historical metric paths and SHA-256. **These are extrapolations, not completed candidate full replays**; occupancy, long-run memory and final metric-output cost can change the rate. Historical full Simulator.run times were 16,584.308 s for P2-32 and 20,343.941 s for P2-64.

Final static verification was executed with:

```bash
ulimit -v 31457280
timeout 60s "$PY" "$B24/check_static.py"
timeout 60s "$PY" "$B24/build_report.py"
```

`check_static.py` parsed the three changed source/test files and every B24 Python script, ran `bash -n` on both shell launchers, confirmed the one-test-only consumer diff, and measured 867 source lines: **PASS**. It also confirmed all ten receipts have FINISHED or WINDOW_COMPLETE status and no owned simulator or replay-launcher process remains running. Results and expanded check arguments are in `static_checks.json`; final source/test diffs are retained as evidence without Git operations.

## 4. Remaining work and decisions for Claude

The only assigned verification prerequisite still open is CPU `torch` in the prescribed interpreter, or a Claude-approved equivalent CPU test environment. Evidence: `torch_prerequisite.json` records exit 1 and `ModuleNotFoundError: No module named 'torch'`. No compatible torch package was found in the inspected developer Python-3.10 environment directories. `test_moe_rank_routing.py:9` skips its entire module via `pytest.importorskip('torch')`; `test_glm_moe_profiling.py:88` skips the one matching layout function. The base also produced these skip entries.

After Claude provides that dependency/environment, rerun, from the scratch cwd with the read-only data link and repository PYTHONPATH:

```bash
cd "$SCRATCH/regression_cwd_candidate"
ulimit -v 62914560
timeout 600s "$PY" -m pytest "$REPO/tests/unit/test_moe_rank_routing.py" \
  "$REPO/tests/unit/test_glm_moe_profiling.py::test_expert_parallel_layout_accepts_only_contiguous_aligned_maps" \
  -q -p no:cacheprovider --basetemp="$SCRATCH/tmp/torch-rerun"
```

That command is proposed, **NOT_RUN**; installing dependencies is left to Claude. No source change is recommended for the missing package. No observed numerical or speed blocker remains. No material alternative behavior or interface design was discovered. Full replay estimates need completed full candidate runs for confirmation; such runs are not part of this assigned window acceptance.

## 5. Review handoff and changed record files

Review the six-field key and schema failure, the memo's local lifetime and guard, prediction order, and the placement of lane construction/SP assignment/logger outside reuse. Check the single changed consumer test and the endpoint/decode/dummy cases. Rerun the torch-dependent checks first; after review, the new tests, regression runner with `--scratch-cwd`, three comparison commands and smoke wrapper are the acceptance commands. Commits, acceptance and subsequent B26/B24 orchestration belong to Claude.

All B24 files below are new in the initially empty records directory. Raw run outputs and caches remain in private scratch; receipts and pre-run SHA-256 inventories are copied here for durable evidence. The report and audit explain state effects, and command JSON records expand wrapper invocations. Every added record file is listed:

- `b24/after_s1.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/after_s1_commands.json` — Expanded executed subprocess commands and exit codes.
- `b24/audit.md` — Human-readable audit, remaining-cost analysis, or self-contained handoff.
- `b24/build_report.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/cache_inventories/p2_32_rate_candidate.json` — Pre-run byte count and SHA-256 for every copied cache file.
- `b24/cache_inventories/profile_candidate.json` — Pre-run byte count and SHA-256 for every copied cache file.
- `b24/cache_inventories/s2_base.json` — Pre-run byte count and SHA-256 for every copied cache file.
- `b24/cache_inventories/s2_candidate.json` — Pre-run byte count and SHA-256 for every copied cache file.
- `b24/cache_inventories/s3_base.json` — Pre-run byte count and SHA-256 for every copied cache file.
- `b24/cache_inventories/s3_candidate.json` — Pre-run byte count and SHA-256 for every copied cache file.
- `b24/cache_inventories/window_base.json` — Pre-run byte count and SHA-256 for every copied cache file.
- `b24/cache_inventories/window_candidate.json` — Pre-run byte count and SHA-256 for every copied cache file.
- `b24/check_static.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/collect_regression.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/compare_runs.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/environment.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/expert_parallel.diff` — Read-only base/candidate source comparison for review.
- `b24/generate_trace.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/lead_profile.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/prepare_cache.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/profile.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/profile.md` — Human-readable audit, remaining-cost analysis, or self-contained handoff.
- `b24/profile_report.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/receipts/p2_32_rate_candidate.json` — Exact run configuration provenance, cache inventory, timing, RSS and completion status.
- `b24/receipts/profile_candidate.json` — Exact run configuration provenance, cache inventory, timing, RSS and completion status.
- `b24/receipts/s1_base.json` — Exact run configuration provenance, cache inventory, timing, RSS and completion status.
- `b24/receipts/s1_candidate.json` — Exact run configuration provenance, cache inventory, timing, RSS and completion status.
- `b24/receipts/s2_base.json` — Exact run configuration provenance, cache inventory, timing, RSS and completion status.
- `b24/receipts/s2_candidate.json` — Exact run configuration provenance, cache inventory, timing, RSS and completion status.
- `b24/receipts/s3_base.json` — Exact run configuration provenance, cache inventory, timing, RSS and completion status.
- `b24/receipts/s3_candidate.json` — Exact run configuration provenance, cache inventory, timing, RSS and completion status.
- `b24/receipts/window_base.json` — Exact run configuration provenance, cache inventory, timing, RSS and completion status.
- `b24/receipts/window_candidate.json` — Exact run configuration provenance, cache inventory, timing, RSS and completion status.
- `b24/regression_additional_command.json` — Expanded executed subprocess commands and exit codes.
- `b24/regression_base_command.json` — Expanded executed subprocess commands and exit codes.
- `b24/regression_candidate_scratch_command.json` — Expanded executed subprocess commands and exit codes.
- `b24/regression_collection.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/regression_collection_base.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/regression_command.json` — Expanded executed subprocess commands and exit codes.
- `b24/regression_targets.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/report.md` — Human-readable audit, remaining-cost analysis, or self-contained handoff.
- `b24/run_additional.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/run_one.sh` — Reproducer for assigned execution, analysis, or report generation.
- `b24/run_pair.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/run_regression.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/run_replay.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/run_smoke.sh` — Reproducer for assigned execution, analysis, or report generation.
- `b24/s1_commands.json` — Expanded executed subprocess commands and exit codes.
- `b24/s1_comparison.json` — Exact parsed-artifact/cache-name comparison and exercised-path evidence.
- `b24/s2_commands.json` — Expanded executed subprocess commands and exit codes.
- `b24/s2_comparison.json` — Exact parsed-artifact/cache-name comparison and exercised-path evidence.
- `b24/s3_commands.json` — Expanded executed subprocess commands and exit codes.
- `b24/s3_comparison.json` — Exact parsed-artifact/cache-name comparison and exercised-path evidence.
- `b24/source_inventory.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/speed_results.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/static_checks.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/summarize_runs.py` — Reproducer for assigned execution, analysis, or report generation.
- `b24/test_dsv4_operator_consumer.diff` — Read-only base/candidate source comparison for review.
- `b24/torch_prerequisite.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/unavailable_collection_audit.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/verification_results.json` — Structured environment, selection, prerequisite, source, static, or verification evidence.
- `b24/window_commands.json` — Expanded executed subprocess commands and exit codes.

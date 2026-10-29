# Task: price the V4 model-runner operators and admit the PD roles (stage 3 assignment B17, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created after B16b (`61c2f9d3`). Covers the four runner operators, their tables, the block-table write coordinate, the PREFILL and unified DECODE admission (design §6 entry 4), `runtime_limitations` (entry 11), the module-size record (entry 12) and the A3 ledger sentinel. |

## Goal

After B16b a non-dummy V4 forward prices every model operator from the trained H200 tables, except the four operators that run on the V2 model runner's objects:

| Operator | Native work | Phase (design §4) | Measured by |
| --- | --- | --- | --- |
| `kv_cache_block_zero` | async H2D copy of the block ids and `_zero_kv_blocks_kernel` | before the DP rendezvous, real lanes only | G2h-03 path B (direct calls on a native-layout engine) |
| `dsv4_block_table_write` | `_apply_write_kernel` and its H2D copies | before the DP rendezvous, real lanes only | G2h-03 path B |
| `dsv4_input_preparation` | input and attention-metadata preparation kernels | after the DP rendezvous, real and dummy forwards | G2h replay (G2h-04, still queued) |
| `dsv4_sampling` | the sampler kernels and their copies | after the model, real forwards only | G2h replay (G2h-04) |

Today `kv_cache_block_zero` raises `DeepSeekV4CoverageError` in non-dummy mode, and the other three are not priced. The V4 predictor also admits only MONOLITHIC, but the H200 A10 cases are PD: P2-32 is P16+D16 with TP8/DP2/EP16 in each role ([../requirements.md](../requirements.md) D-H200-H4B-LIFECYCLE). Its decode role receives each request through NIXL and starts it with a first step of query 1 and 6,143 cached tokens.

B17 delivers:

1. The runner tables: the block table from G2h-03 now, committed; the replay table through the same promotion script, which the lead runs on G2h-04 when it is downloaded.
2. The consumer: exact lookup, cost and gaps, phase placement, and the real and dummy ledger.
3. The block-table write coordinate of each real forward, computed by the KV allocation code.
4. The PREFILL and unified DECODE admission of the V4 predictor (PD-AF stays rejected).
5. The `runtime_limitations` text, the module-size record and the A3 ledger sentinel.

## Approach and decisions

### Read first

- [design.md](design.md): §1 (gates A1, A3, A5, A9, A12, A13), §3.3 (gaps; the runner work pays the `eager_preparation` gap per activity), §3.4 (host time), §3.7 (the DP-synchronized descriptor, the pre-DP and post-DP timeline, the V2 dummy workload), §4 (the ledger), §6 entries 3, 4, 8, 11, 12 and 19, and the implementation notes (the B13r split, the B15 notes on the PD admission).
- [c1_coverage.md](c1_coverage.md) §3.2-3.3 and §4.2 (the zeroing count rule, the write pairs of the native cases, the coordinate margins), and `evidence/coverage/c1_workload.py:300-326` (the native-derived count rules for zeroing and writes per forward).
- The B16b assignment and report: [codex_assignment_B16b_v4_operator_consumer_runtime.md](codex_assignment_B16b_v4_operator_consumer_runtime.md), [codex-implement-86155181-0e36-4ecf-9c67-6e5ed04b9aa2/implementation.md](codex-implement-86155181-0e36-4ecf-9c67-6e5ed04b9aa2/implementation.md), and part 2 of [../test_report_2026-10-29_b16_v4_operator_consumer.md](../test_report_2026-10-29_b16_v4_operator_consumer.md).
- The producer and its rows: [codex_assignment_B13r_v4_runner_operators.md](codex_assignment_B13r_v4_runner_operators.md), [codex_assignment_B13s_g2h_sampling_scope_and_split_jobs.md](codex_assignment_B13s_g2h_sampling_scope_and_split_jobs.md), [../test_report_2026-10-29_g2h_runner_02.md](../test_report_2026-10-29_g2h_runner_02.md), `frontier/profiling/model_runner/dsv4_runner_rows.py` and `dsv4_block_operators.py` (`write_staging_plan`, line 98, defines the write coordinate).
- The B16b runtime: `frontier/execution_time_predictor/deepseek_v4.py` (`timing_operator_times`, `predict_structural_execution`, `serving_context`), `dsv4_operator_predictor.py` (`DeepSeekV4OperatorPredictor`, `_exact`), `dsv4_training.py` (`train_dsv4_models`), `dsv4_runtime.py` (`V4_SEGMENT_OWNERS`, "B17 adds the runner scopes here"), `dsv4_composition.py` (`_PREFIX`), `frontier/scheduler/utils/dp_dummy_forward.py:225-250` (the rendezvous pricing reads the `pre_dp` phase).
- The KV state: `frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py` (`_allocate_request` and the zero queue at line 675, `reserve_kv_transfer_target` at line 185, `_free_request_resources`, `_preempt_request`), the group tables (`_allocate_group_slots`, `get_group_block_ids`, `reclaim_skipped_blocks`), `vllm_v1_engine_replica_scheduler.py:447-467` (`_create_batch` builds `KVCacheBlockZeroing`), `frontier/entities/batch.py:14` (`KVCacheBlockZeroing`).
- The pinned V2 runner for the native semantics: `vllm/v1/worker/gpu/model_runner.py` (request add, update and removal; the zeroing call near line 1038; the order of zeroing, staged writes, DP synchronization and preparation, lines 1417-1546), `vllm/v1/worker/gpu/block_table.py` or the module that owns `BlockTables.append_block_ids`, `vllm/v1/worker/gpu/buffer_utils.py:211-271` (staged writes), `vllm/v1/worker/utils.py:211-232` (the zeroer), and the sampler path.

### Inputs (read-only)

- **Block rows (G2h-03, accepted):** `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_03/download/block/G2H_BLOCK_OPERATORS/` (`block_operators.csv`, `layout.json`, the per-row `*.trace.intervals.json`; do not search it recursively, list it). 394 rows: zeroing at n = 1-32 and 2,520; writes at w_c 0-32 × r 0-4 without (0, 0); both methods. Each repeat is 1 kernel and 1 memcpy. The KERNEL_ONLY rows are the `record_function` rows.
- **Replay rows, structure fixture only:** G2h-02, `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_02/download/replay/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/` (`dsv4_raw_eN.runner.csv`, `dsv4_raw_eN.jsonl` for N = 0, 1, 2; the e0 JSONL is 344 MB). Its sampling scope ran without the native sampling contract (no logit-bias path; fixed in `994d293c`). Use these rows only in scratch to build and test the replay table and the consumer. Do not commit a replay table built from G2h-02. The lead runs the promotion on G2h-04 (same layout under `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_04/download/`; the script takes the raw directory as an argument) after accepting it.
- **Descriptor join:** the committed `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv`, one descriptor per `case_id`. The lead's scratch check `/data/ycfeng/tmp/b17_prep/prep_act.py` shows the join: 138 preparation keys over the fields (`phase`, `graph_runtime_mode`, `padded_rows`, `dispatch_num_requests`, `batch_size`, `num_prefill_seqs`, `num_decode_seqs`, `total_tokens`), with constant activity counts per key (54-70 kernels, 7-11 copies). Dummy keys: FULL and PIECEWISE at W 1, 2, 4, 8, 16, 24, NONE at W1. Prefill key: NONE, 6,144 rows, 1 request.
- **Gaps:** the committed `kernel_gap.csv` (`eager_preparation` 1.2208671118771937 µs on H200).

### Decisions

1. **Two runner tables, one promotion script.**
   - A torch-free reducer module in `frontier/profiling/model_runner/` turns the producer rows into consumer tables; a new script `h3o_operator_costs/h200_tables/promote_runner_tables.py` calls it and writes `h200_tables/runner_manifest.json` (inputs and outputs with SHA-256, row counts, the source run tags). The existing `promote_tables.py` and its four tables stay unchanged.
   - Outputs in `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/`: `model_runner_block_kernel_only.csv` (zeroing and writes, from the `record_function` path-B rows) and `model_runner_replay_kernel_only.csv` (preparation and sampling, from the replay `cost` rows). Commit-ready now: the block table only.
   - Each output row carries: the operator; its coordinate columns (zeroing `n`; writes `w_c`, `r` and the per-group entries of one new row; preparation the eight descriptor fields; sampling `b`); the producer's `kernel_stats.<op>.*` and `memcpy_stats.<op>.*` columns unchanged; the activity count per invocation (kernels, copies and memsets of the scope, from the trace intervals); and the provenance (source case or row id, engine index, run tag).
   - The new-row entries come from the producer grid (`BlockGrid.entries`, `first_chunk_tokens` and the block sizes). They are not literals in the reducer or the consumer.
   - The activity count of a coordinate must be the same in every source row of that coordinate. If not, the reducer raises and names the coordinate.
   - Preparation rows take their coordinate from the attention table by `case_id`. A `case_id` with no descriptor or with more than one raises. Sampling `b` is the number of real requests of the step; take it from the descriptor and check it against the native sampler's input size in the pinned source.
   - The `validation` and `diagnostic` replay rows do not enter the tables. Report the replay write rows against path B: for the replay steps with write activity, their cost medians beside the path-B range; for the steps without write activity, confirm from the step records that nothing was staged, if the records say so. This is a reported comparison, not a gate.
2. **Exact lookup only** (as B16 for operators). The runner operators train through `train_dsv4_models` with the existing exact-lookup models, in both training paths; duplicate coordinates use the trainer's mean.
   - A coordinate without a row raises `DeepSeekV4CoverageError` naming the operator and the coordinate. A missing runner table raises the same error naming the file.
   - No row and no charge at zeroing n = 0 or writes (w_c, r) = (0, 0): the native runner launches nothing then.
3. **Cost and gaps.** Cost per invocation = the kernel median + the memcpy median of the row. Gap = activity count × the `eager_preparation` value, through `V4_SEGMENT_OWNERS` and `_operator_gaps` (B16b). No other gap class and no host time.
4. **Phases and the ledger (design §4).**
   - `pre_dp`: `kv_cache_block_zero` and `dsv4_block_table_write`, in the native order (check it in the pinned source), at layer 0 of real forwards only. The rendezvous pricing (`dp_dummy_forward.py`) already starts the pre-DP device work at the lane's entry and the first layer at the later of the rendezvous and the pre-DP device end; it needs no change unless the pinned order says otherwise.
   - `dsv4_input_preparation`: at the head of the layer-0 attention prefix (`timing_operator_times` and the composition `_PREFIX`), on real and dummy forwards, at the synchronized descriptor. Its gaps are its own (`eager_preparation`), not the attention segment class.
   - `dsv4_sampling`: in `post_combine` after `dsv4_logits_gather`, at the last layer of real forwards only, keyed by the number of real requests.
   - A dummy forward charges preparation only, and no pre-DP work, LM head, logits gather or sampling.
   - In dummy predictor mode the new scopes cost `_dummy_execution_time`, as the other V4 scopes; report the change of V4 dummy-mode outputs.
5. **The write coordinate**, computed where the KV state lives.
   - A new frozen record beside `KVCacheBlockZeroing` in `frontier/entities/batch.py`, attached by `_create_batch` like the zeroing record. It holds, for one real forward of one lane: the new-block count of each continuing (group, row) pair, and the per-group entries of each new row. Name it after what it is (for example `KVBlockTableWrites`).
   - Native semantics to reproduce (check each against the pinned source and cite the lines): a request that the runner adds this step (first schedule; the decode role's first step after the NIXL load; resume after preemption) writes its full row, one entry per block in each group, null entries included; a running request appends its new block ids per group; a removed request (finished or preempted) gives up its row.
   - Frontier side: keep, per lane, each request's per-group table length at its last forward (from `get_group_block_ids`, which includes the `None` entries). A request without a recorded length is a new row; otherwise its new entries per group are the length difference. `reclaim_skipped_blocks` does not change lengths and writes nothing. Remove the record on free and on preemption.
   - Coordinate: w_c = the number of continuing (group, row) pairs with a new entry; r = the number of new rows. A continuing pair with more than 1 new entry, or a new row whose per-group entries differ from the table's, raises `DeepSeekV4CoverageError`; the measured shape is 1 entry per continuing pair and the 6,144-token first-chunk row.
   - The decode role's NIXL request: its prompt blocks come from `reserve_kv_transfer_target`, which bypasses the zero queue. Its first step is therefore a new row with the prompt's entries and no zeroing of the prompt blocks, which matches C1's count rule.
   - Keep the logic in `vllm_v1_kv_allocation.py` (1,073 lines). `vllm_v1_engine_replica_scheduler.py` (1,914 lines) gets only the lines that attach the record and clear the state, and stays at or below 2,000 lines. Non-grouped models get no record and no new state.
6. **PREFILL and unified DECODE admission (entry 4).**
   - `_validate_attention_context` admits `PREFILL` and `DECODE` beside `MONOLITHIC`; `DECODE_ATTN` and `DECODE_FFN` stay rejected with the existing message style.
   - `compute_rows` takes the cluster type of the batch's role instead of the hard-coded `ClusterType.MONOLITHIC`.
   - Remove the `serving_context` gate that rejects `KV_CONNECTOR != "none"` without a step-cost table. The diagnostic step-cost path stays as it is.
   - The PREFILL role samples its one request (native `max_tokens=1` for the P request). Check in the native P2 package's proxy what sampling parameters the P request carries. If they differ from the measured contract (min_tokens equal to max_tokens, the generation-config EOS as a stop token, so the logit-bias path is active), raise `DeepSeekV4CoverageError` for PREFILL sampling and report the evidence; do not price it from the decode-contract rows.
   - The first decode-role step of a NIXL request is first-tail shaped (query 1, 6,143 cached tokens) and uses the `first_tail` keys of the attention and preparation tables.
7. **`runtime_limitations` (entry 11).** Replace the stale V4 text in `frontier/model_architectures.py:704-709` ("measured execution is limited to a diagnostic whole-step table ...") with the current scope: the trained operator path for MONOLITHIC, PREFILL and unified DECODE on the measured H200 coordinates; the step-cost table as a diagnostic only; PD-AF, prefix caching, speculative decoding / MTP and CPU-overhead modeling unsupported; sampling measured under the native min_tokens contract only. Update the tests that pin the old text.
8. **Module-size record (entry 12).** Add a dated section to `docs/general/module-size-exceptions.md` in the existing format (see `V4 diagnostic step-cost dispatch, 2026-10-25`) for the stage-3 changes to `sklearn_execution_time_predictor.py` and `metrics_store.py` (B15, B16b and B17): line counts before stage 3 (the design's 8,666 and 5,744; confirm against Git) and after B17, the scoped cleanup inspection, the boundary reason, and where the change sits in the planned split. Neither file gets more than registration and routing lines in B17.
9. **Use B16b's names and paths:** `DeepSeekV4OperatorPredictor`, `OPERATOR_MODEL_PREFIX`, `operator_model_name`, `V4_SEGMENT_OWNERS`, `train_dsv4_models`, the `dsv4_predictions` entries. No second loader, trainer or gap path.

## Out of scope

- Promoting a replay table from G2h-04 (the lead does it after acceptance) or from G2h-02 (never committed).
- The `add_requests` staged copies that the replay leaves unowned (about 15-25 µs on steps with new requests): a known limit, listed in the report, not priced.
- Pricing the replay zeroing rows: they stay diagnostic.
- Changes to the four committed tables, `promote_tables.py`, the GPU producers' measured behavior, `balanced_routing.py`, the collective consumer and the DP descriptor and rendezvous functions (entries 20-22), unless decision 4's order check requires a rendezvous change; then stop that part and report.
- End-to-end replays (A10), the diagnostic cross-check (A11) and the 77-case fidelity matrix (the lead runs them).
- Any GPU, platform or network action.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `61c2f9d3` or later). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md), [c1_coverage.md](c1_coverage.md).
- **Writable paths:**
  - `frontier/execution_time_predictor/` (the V4 modules, `kernel_gap.py`, `shared_prediction_model_manager.py`, new modules beside them; `sklearn_execution_time_predictor.py` registration and routing lines only);
  - `frontier/attention/deepseek_v4.py`, `frontier/attention/dsv4_features.py`;
  - `frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py`, `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` (attachment lines only), `frontier/scheduler/utils/dp_dummy_forward.py` (only per decision 4);
  - `frontier/entities/batch.py` (the record and its field);
  - `frontier/operators/families.py`, `frontier/metrics/constants.py` (runner operator metadata and catalog names, if needed);
  - `frontier/model_architectures.py` (the V4 `runtime_limitations` text only);
  - `frontier/profiling/model_runner/` (the new reducer module only; the producer modules stay unchanged);
  - `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/model_runner_block_kernel_only.csv` (new);
  - `docs/general/module-size-exceptions.md`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/` (the new script and `runner_manifest.json` only);
  - `tests/unit/`.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b17/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No moves. Delete only code that this change supersedes, and list each deletion.
  - Python 3.10. Keep the touched modules at or below 2,000 lines, apart from `sklearn_execution_time_predictor.py` and `metrics_store.py` (registration lines only). Report the line counts before and after for every touched module above 1,000 lines.
  - No magic literals in code paths: block sizes, entries, gap values, operator names and coordinates come from the model config, the registry and the tables. No guards or defaults for states the code cannot reach, and no fallback that hides a missing row. No second implementation of the zeroing count rule, the descriptor synchronization or the gap selection.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command. The raw replay JSONL files are large: stream them.
  - Search with `/usr/bin/grep` and quote `--include='*.py'`. Do not search recursively under `/data/ycfeng/tmp/glm53_dsv4/`; list the exact directories above.
- **Environment:**
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. Run pytest from the repository root.
  - The pinned vLLM source is `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`), read-only.

## Acceptance

1. **Tables.**
   - `promote_runner_tables.py` on G2h-03 writes the block table: 33 zeroing rows (n = 1-32 and 2,520) and 164 write rows, each with activity count 2 (1 kernel, 1 memcpy), the new-row entries (24, 96, 96, 1,536, 768) from the grid, and the manifest. A second run gives the same SHA-256.
   - The same script on the G2h-02 replay in scratch writes a replay table with 138 preparation keys and the sampling keys by b, with constant activity counts per key; report the counts and the per-key cost ranges.
   - The replay write comparison of decision 1, with the numbers.
2. **Consumer.**
   - Exact lookup: one worked example per operator (coordinate, row medians, cost, activity count, gap), matching the table by hand.
   - `DeepSeekV4CoverageError` for: an unmeasured n, (w_c, r) or preparation key or b; a continuing pair with 2 entries; a new row with other entries; a missing runner table. No charge at n = 0 and (w_c, r) = (0, 0).
   - Both training paths give the same predictions, and the A9 cache identity covers the runner tables (a changed table changes the cache key).
3. **Write coordinate.** Unit tests on the KV allocation: first schedule (r = 1 with the full first-chunk entries); running decodes crossing block boundaries (w_c); the decode role's NIXL first step (r = 1, no zeroing of the prompt blocks); preemption and resume (a new row); free (record removed); a non-grouped model (no record). One consistency test: across a short V4 PDD run in dummy predictor mode, every batch's (w_c, r) and zeroing n equal the C1 rules of `c1_workload.py:316-326` evaluated on the same batch's (old, new, is_new) tuples.
4. **Ledger (A3 sentinel).** One test with the §4 expected table as its only copy: counts per operator over all 43 layers for a real decode forward, a decode-role first-tail forward, a prefill forward, and FULL, PIECEWISE and NONE dummy forwards. Pre-DP work, LM head, logits gather and sampling are 0 on dummies; preparation is 1 on every forward.
5. **PD admission.** Non-dummy V4 predictors for PREFILL and unified DECODE (TP8/DP2/EP16, `skip_cpu_overhead_modeling=True`) construct on the committed tables plus the scratch replay table and price a prefill forward, a first-tail decode forward and a dummy forward; `DECODE_ATTN` and `DECODE_FFN` still raise. The P-request sampling contract evidence of decision 6.
6. **A1.** With the committed tables plus the scratch replay table, MONOLITHIC and PDD non-dummy forwards price every operator from tables and read no `_dummy_execution_time` (the stage components of the decode FULL, first-tail and prefill NONE forwards). With the committed tables alone, construction or the first forward raises the typed error naming `model_runner_replay_kernel_only.csv`.
7. **A12 and regression.** The affected unit suites, the dummy smoke matrix (`bash examples/architecture/run_dummy_smoke_matrix.sh`, 12/12), and the evidence that non-V4 predictions and metrics are unchanged. The V4 dummy-mode changes of decision 4, listed.
8. **Records.** The `runtime_limitations` text and the tests that pin it; the module-size section; the line counts.

## Expected return

- Report language: English.
- Changed, new and deleted paths, each with a summary.
- The results of Acceptance 1-8, with the commands, the expected and actual values, and the counts.
- The native source lines that settle decisions 4, 5 and 6.
- The expected effect on a decode step on H200: the added pre-DP, preparation and sampling time and gaps for a typical b = 8 FULL step, from the tables.
- Scratch paths, deviations and their reasons, and limits.
- If the code, the native source or the tables contradict a decision above, stop that part. Report the evidence and the options rather than choosing a different design. Continue with the parts that do not depend on it.

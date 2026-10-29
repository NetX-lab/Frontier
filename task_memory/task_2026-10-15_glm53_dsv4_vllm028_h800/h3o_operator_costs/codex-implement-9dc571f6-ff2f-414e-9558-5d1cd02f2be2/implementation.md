**1. Assignment and outcome**

B17 adds pricing for the four V4 model-runner operators, records each lane’s block-table write coordinate, admits PREFILL and unified DECODE, and updates the runtime limitations, module-size record and A3 sentinel.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (HEAD `1fcc6880`; B16b runtime baseline `61c2f9d3`). Scratch root: `/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17`.

The writable scope was the assignment’s V4 predictor and attention modules, the two named scheduler modules, Batch, operator metadata, V4 runtime-limitations text, the new runner reducer, the new block table, the new promotion script/manifest, unit tests and module-size document. `frontier/metrics/metrics_store.py` is outside that scope.

The implementation preserves the agreed exact lookup, existing training/cache and gap paths, zero-coordinate no-ops, native phase ordering, no host-time charge, dummy-forward exclusions, and PD-AF rejection. Runner tables are resolved beside the attention input table. G2h-02 replay data was used only in scratch; no replay table was added to the committed data directory.

Outcome is **partial** for two concrete reasons:

- Native PREFILL sampling contradicts the measured decode sampling contract. The implementation raises the typed error required by decision 6. PREFILL attention/preparation are admitted, but a complete non-dummy PREFILL forward cannot finish sampling.
- Two actual MetricsStore trace tests fail because the emitted runner names lack OperationMetrics entries. Adding entries also requires V4-only metric admission in `metrics_store.py` to preserve non-V4 output columns. That file is not writable here.

Applicable AGENTS instructions, design/C1 records, requirements/plan, B13r/B13s/B16b artifacts and pinned native source were read. The pre-existing untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` directories were left intact. No Git state changes, dependency installation, GPU/network actions, commits or task-record updates were performed. Final acceptance remains with Claude.

**2. Changes**

New files:

| File | Behavior and purpose |
| --- | --- |
| [dsv4_runner.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/dsv4_runner.py:10) | Registry-owned runner names, table names, descriptor/activity columns and synchronized preparation coordinates. |
| [dsv4_runner_tables.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/model_runner/dsv4_runner_tables.py:54) | Torch-free block/replay reducer; streams JSONL, joins descriptors exactly, enforces constant per-coordinate activity counts and reports replay-write comparisons. |
| [promote_runner_tables.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/promote_runner_tables.py:1) | One CLI for block and replay promotion, with input/output hashes, source record counts and run tags. Refuses G2h-02 replay promotion into the committed table directory. |
| [runner_manifest.json](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/runner_manifest.json:1) | Provenance for the accepted G2h-03 block promotion. |
| [model_runner_block_kernel_only.csv](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/model_runner_block_kernel_only.csv:1) | 197 accepted path-B rows: 33 zeroing coordinates and 164 write coordinates. |

Changed production/documentation files:

| File | Affected behavior |
| --- | --- |
| [dsv4_operator_predictor.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/dsv4_operator_predictor.py:209) | Existing operator loader/trainer extended for runner rows. Kernel plus memcpy medians become exact costs; all device activities feed the existing paired count model. Exact misses and unsupported write shapes raise DeepSeekV4CoverageError. |
| [dsv4_training.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/dsv4_training.py:19) | Both existing training paths admit both runner tables and retain the existing dsv4_predictions entries. Missing replay data raises the typed file error. |
| [dsv4_runtime.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/dsv4_runtime.py:15) | Registers eager_preparation ownership and prices zero/write before DP, preparation once per forward, and real sampling at the final layer. Adds the explicit PREFILL sampling coverage guard. |
| [predictor deepseek_v4.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/deepseek_v4.py:19) | Runner phase ordering and trace metadata; replaces the synthetic-only zeroing block with runner pricing and removes the non-diagnostic KV_CONNECTOR rejection. |
| [dsv4_composition.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/dsv4_composition.py:27) | Preparation heads the layer-0 attention prefix and retains its own activity gaps. |
| [attention deepseek_v4.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/attention/deepseek_v4.py:179) | Admits MONOLITHIC/PREFILL/DECODE, rejects PD-AF roles, and uses the batch role for compute rows. PREFILL consumes the synchronized graph metadata through its existing width selector. |
| [dsv4_features.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/attention/dsv4_features.py:132) | Recognizes a newly added NIXL decode row with query 1 and prompt-minus-one context as first_tail, without changing request KV/accounting snapshots. |
| [batch.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/entities/batch.py:38) | Frozen KVBlockTableWrites record, batch field and batch role. Records continuing group/row append counts and full per-group new-row lengths. |
| [vllm_v1_kv_allocation.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py:28) | Tracks each grouped request’s logical table lengths, including null entries. New requests write full rows; continuing requests append length differences; free/preemption removes saved lengths. |
| [vllm_v1_engine_replica_scheduler.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:336) | Four attachment/initialization lines: grouped row state, real batch role/write record and dummy batch role. |
| [model_architectures.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/model_architectures.py:705) | Replaces stale diagnostic-only limitations with the trained H200 operator scope and current unsupported features/sampling contract. |
| [families.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/operators/families.py:1) | Updates stale runner-family descriptive text. |
| [module-size-exceptions.md](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/docs/general/module-size-exceptions.md:199) | Dated stage-3 cleanup/boundary/split record, confirmed counts, and the remaining metrics writable prerequisite. |

New unit files:

- `tests/unit/test_dsv4_runner_tables.py`: descriptor join, activity consistency, sampling-size checks and exclusion of validation rows.
- `tests/unit/test_dsv4_runner_consumer.py`: all exact runner coordinates, duplicate means, gaps, typed misses, cache identity, role admission, missing replay construction and all-layer non-dummy PD/dummy coverage.
- `tests/unit/test_dsv4_kv_block_table_writes.py`: first schedule, null-preserving reclamation, C1 consistency across the NIXL lifecycle, preemption/resume/free and a real non-grouped forward with no new state or record.
- `tests/unit/test_dsv4_runner_ledger.py`: the sole §4 expected ledger across 43 layers for six real/dummy forward geometries.

Changed unit files:

| File under tests/unit/ | Reason |
| --- | --- |
| test_dsv4_attention_runtime.py | Explicit batch role for role-aware compute rows. |
| test_dsv4_attention_workload.py | PREFILL admission and PD-AF rejection expectations. |
| test_dsv4_cache_zero_execution.py | Runner ordering/metadata expectations; retains the two actual metrics failures. |
| test_dsv4_dp_sync.py | Explicit role for synchronized descriptor checks. |
| test_dsv4_graph_measured.py | Preparation scope and role-aware expectations. |
| test_dsv4_operator_consumer.py | Scratch replay fixture selection, cache reuse, explicit role and an unmeasured zero coordinate. |
| test_dsv4_source_execution.py | New runner scopes and current PD-AF error wording. |
| test_dsv4_structural_execution.py | New preparation/sampling costs and explicit roles. |
| test_dsv4_structural_pd.py | Runner scopes in PD lifecycle expectations. |
| test_grouped_kv_cache_scheduler.py | Grouped table-length state in the lightweight scheduler fixture. |

No files were deleted. Superseded code removed from predictor deepseek_v4.py comprises the old synthetic-only cache-zero block and the serving_context KV_CONNECTOR rejection. The original four tables, existing promote_tables.py, GPU producers, collective consumer, DP descriptor and rendezvous functions are unchanged. Neither sklearn_execution_time_predictor.py nor metrics_store.py was edited in B17.

The new script and manifest are ignored by the repository’s current ignore rules; Claude must include them explicitly when staging an accepted change.

**3. Verification**

All checks ran from the repository root with Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), no torch/vLLM, the prescribed PYTHONPATH, disabled bytecode/W&B, scratch TMPDIR/caches/outputs, `ulimit -v 31457280`, and a timeout on every shell command.

For the commands below, `P` denotes that interpreter and `S` denotes the scratch root above. The environment is recorded in [env.sh](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/env.sh:1). Consumer checks set `FRONTIER_DSV4_TEST_MODEL_ROOT=$S/tmp/runner_tmp/v4_models0` to reuse the completed training cache.

**Acceptance 1 — tables: PASS for accepted blocks and scratch replay structure.**

Executed block promotion, then repeated it with scratch output:

```bash
timeout 90s $P task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/promote_runner_tables.py --block-dir /data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_03/download/block/G2H_BLOCK_OPERATORS --block-run-tag v4-h200-g2h-runner-20261029-03-block

timeout 90s $P task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/promote_runner_tables.py --block-dir /data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_03/download/block/G2H_BLOCK_OPERATORS --block-run-tag v4-h200-g2h-runner-20261029-03-block --output-dir $S/block_repeat --manifest-path $S/block_repeat_manifest.json
```

Observed 197 rows: zero n=1–32 and 2,520 (33 rows); writes w_c=0–32 × r=0–4 excluding (0,0) (164 rows). Every row has kernel=1, memcpy=1, memset=0, activity=2. New-row entries are grid-derived `(24,96,96,1536,768)`. Both outputs have SHA-256 `e0a6dadfa844883603e22be2906e34d987060bc487f3290b821c9a0a6c3139cd`. The manifest includes counts for all inputs and outputs; the input block CSV has 394 rows.

The earlier completed scratch replay promotion used the same script with:

```bash
--replay-raw-dir /data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_02/download/replay/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw
--replay-run-tag v4-h200-g2h-runner-20261029-02-replay-STRUCTURE_ONLY
--output-dir $S/tables
--manifest-path $S/replay_manifest.json
```

Observed 1,965 retained rows: 989 preparation and 976 sampling; 138 preparation keys and 24 sampling keys b=1–24. SHA-256 is `b6d5349e75864d6659392bdda4f8b62a3ee9f0b41407a14313fd8729a2d12a5b`. Activity counts are constant within every coordinate. Validation/diagnostic rows are excluded. The raw JSONL was streamed; source record counts are 514,190 / 221,714 / 5,191 for e0/e1/e2. The corresponding runner CSV counts are 3,044 / 1,340 / 32.

Ranges below are ranges of the trained per-key mean costs, in milliseconds:

| Preparation phase/mode | Keys | Activities per invocation | Cost range |
| --- | ---: | --- | --- |
| decode/FULL | 39 | 75 or 81 | 0.08440999349–0.10137717014 |
| decode/PIECEWISE | 39 | 75 or 76 | 0.08508740234–0.09953754340 |
| first_tail/PIECEWISE | 46 | 76 or 77 | 0.08831542969–0.10344514974 |
| prefill/NONE | 1 | 76 | 0.14371044922 |
| dummy/FULL | 6 | 66 | 0.06588671875–0.07763134766 |
| dummy/PIECEWISE | 6 | 61 or 66 | 0.06178759766–0.07746875000 |
| dummy/NONE | 1 | 66 | 0.06617138672 |

Sampling has 24 keys, activity 9, and per-key mean cost range 0.02515501739–0.04725339988 ms. These are **G2h-02 structure-fixture figures under the superseded sampler contract**, not final accepted sampling measurements.

Replay write comparison: 976 validation rows, 499 with nonzero cost; replay median costs range 0–0.0096640625 ms, beside path-B’s 0.0043837890625–0.00896044921875 ms. There are 490 steps without write intervals. None of their retained step records contains staging fields, so those records cannot independently prove that nothing was staged. The comparison remains reported evidence, not a gate.

All per-key costs, source-median min/max ranges and aggregate counts are in [table_evidence.json](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/table_evidence.json:1); source hashes and write comparisons are in [replay_manifest.json](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/replay_manifest.json:1).

**Acceptance 2 — consumer: PASS.**

Every retained coordinate was checked against the table’s kernel-plus-memcpy median, including the trainer’s duplicate-coordinate mean. The eager_preparation gap is 1.2208671118771937 µs per activity.

Worked H200 example, in milliseconds:

| Operator/coordinate | Kernel median mean | Memcpy median mean | Cost | Kernel/copy/memset | Gap | Cost + gap |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| zero n=8 | 0.008063964844 | 0.001023925781 | 0.009087890625 | 1/1/0 | 0.002441734224 | 0.011529624849 |
| write (8,0) | 0.003808105469 | 0.001023925781 | 0.004832031250 | 1/1/0 | 0.002441734224 | 0.007273765474 |
| preparation (decode,FULL,8,8,8,0,8,8) | 0.080760416667 | 0.009610351563 | 0.090370768229 | 64/11/0 | 0.091565033391 | 0.181935801620 |
| sampling b=8 | 0.027516876776 | 0 | 0.027516876776 | 9/0/0 | 0.010987804007 | 0.038504680782 |

Preparation averages nine source rows; sampling averages 55. The latter two figures are provisional G2h-02 fixture values. Preparation plus sampling adds 0.220440482402 ms to a b=8 FULL step. If that step also has n=8 and (w_c,r)=(8,0), pre-DP work adds 0.018803390323 ms, giving 0.239243872725 ms total. Pre-DP work is zero when no blocks or writes are staged. Full arithmetic is in [worked_examples.json](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/worked_examples.json:1).

Typed errors passed for n=33, writes (33,0), an unmeasured preparation key, b=25, a continuing pair appending two entries, a differently sized new row and missing runner files. Zero n and writes (0,0) return zero cost/activity and emit no charge.

Both existing training paths and warm cache matched model keys, every exact cost/count map and FULL/NONE attention stage ledgers. Changing either runner table changes the cache hash. The completed combined command was:

```bash
timeout 900s $P -m pytest tests/unit/test_dsv4_runner_consumer.py tests/unit/test_dsv4_operator_consumer.py tests/unit/test_dsv4_runner_ledger.py tests/unit/test_dsv4_kv_block_table_writes.py -q -p no:cacheprovider --basetemp $TMPDIR/runner_final_tmp
```

Its log reports 33 passed and six ledger assertion failures. Those six failures were a test-only dispatch-duration multiplier; the existing dummy dispatch endpoint costs one configured scope duration. After correcting that assertion, all six ledger cases passed in the structural run below. The training parity test passed in the completed combined run and was not needlessly cold-trained again. [runner_final.log](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/runner_final.log:1) preserves the actual result.

**Acceptance 3 — write coordinate: PASS.**

The KV tests cover full first-chunk entries, null-preserving reclamation, running decode boundary appends, NIXL first-tail r=1 with no prompt zeroing, preemption/resume as a new row, free cleanup and a real non-grouped schedule/free with no record or state. A 17-step decode lifecycle compares every zero/write coordinate against C1’s old/new/is_new formulas on the same forward geometry.

Final executed command:

```bash
timeout 90s $P -m pytest tests/unit/test_dsv4_kv_block_table_writes.py tests/unit/test_dsv4_runner_tables.py -q -p no:cacheprovider --basetemp $TMPDIR/kv_final_tmp
```

Observed **9 passed**; [kv_final.log](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/kv_final.log:1).

**Acceptance 4 — A3 ledger: PASS.**

All 43 layers match the sole EXPECTED table for real decode, decode-role first_tail, prefill and FULL/PIECEWISE/NONE dummy forwards. Preparation occurs once on every forward. Dummies have no pre-DP work, LM head, logits gather or sampling. Native phase ownership is checked, including preparation at the attention prefix and sampling after logits gather.

**Acceptance 5 — PD admission: PARTIAL, with the required native sampling rejection.**

Non-dummy TP8/DP2/EP16 PREFILL and DECODE predictors construct with the scratch replay table and skip_cpu_overhead_modeling=True, including KV_CONNECTOR=nixl. PREFILL preparation/attention and decode first_tail price from exact tables. DECODE_ATTN and DECODE_FFN still raise. All-layer decode/dummy tests pass; PREFILL explicitly raises at final-layer sampling.

Native evidence:

- P2 proxy `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/p2_32_reference/package_01/tests/comparison/calibration/pd_proxy.py:103–108` sets max_tokens=1 and removes both min_tokens and min_completion_tokens.
- Pinned `vllm/v1/worker/gpu/model_runner.py:1122–1130` sets total_num_logits=num_reqs without draft tokens; `1339–1353` selects those hidden rows, computes logits and invokes the sampler. Sampling b therefore counts real requests, not graph padding.

The P request cannot be priced from decode-contract sampler rows. The guard preserves the assignment’s explicit decision instead of changing the proxy or claiming measurement coverage.

**Acceptance 6 — A1: PASS for supported measured phases; full PREFILL sampling remains blocked.**

The non-dummy fixture forbids reading _dummy_execution_time. MONOLITHIC FULL/NONE stage components, all-layer DECODE FULL/first_tail, and FULL/PIECEWISE/NONE dummy forwards price from tables. PREFILL prices through the supported phases and raises the deliberate sampling coverage error. With committed tables alone, construction raises DeepSeekV4CoverageError naming model_runner_replay_kernel_only.csv.

Executed:

```bash
timeout 240s $P -m pytest tests/unit/test_dsv4_runner_consumer.py tests/unit/test_dsv4_kv_block_table_writes.py -q -p no:cacheprovider --basetemp $TMPDIR/consumer_verified_tmp
```

Observed **21 passed**, with four existing pandas mixed-type warnings; [consumer_verified.log](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/consumer_verified.log:1).

**Acceptance 7 — A12/regression: PARTIAL because two metrics trace checks fail.**

Executed affected structural suites:

```bash
timeout 180s $P -m pytest tests/unit/test_dsv4_runner_ledger.py tests/unit/test_dsv4_cache_zero_execution.py tests/unit/test_grouped_kv_cache_scheduler.py tests/unit/test_dsv4_structural_pd.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_graph_measured.py tests/unit/test_dsv4_source_execution.py tests/unit/test_dsv4_attention_runtime.py tests/unit/test_dsv4_dp_sync.py tests/unit/test_dsv4_attention_workload.py -q -p no:cacheprovider --basetemp $TMPDIR/structural_verified_tmp
```

Observed **251 passed, 2 failed**. Both failures are test_real_stage_trace_places_cache_zero_before_embedding, with chunking False/True:

`ValueError: 'dsv4_block_table_write' is not a valid OperationMetrics`

These failures were retained, not xfailed or hidden. [structural_verified.log](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/structural_verified.log:1).

Executed related suites:

```bash
timeout 240s $P -m pytest tests/unit/test_dsv4_features.py tests/unit/test_dsv4_predictor.py tests/unit/test_dsv4_model_config.py tests/unit/test_dsv4_operator_manifest.py tests/unit/test_dsv4_operator_admission.py tests/unit/test_dsv4_composition.py tests/unit/test_dsv4_composition_check.py tests/unit/test_dsv4_runner_operators.py tests/unit/test_dsv4_runner_tables.py tests/unit/test_dsv4_replay_rows.py tests/unit/test_dsv4_replay_gaps.py tests/unit/test_dsv4_h200_tables.py tests/unit/test_dsv4_step_cost_input.py tests/unit/test_kernel_gap_pricing.py tests/unit/test_operator_registry.py tests/unit/test_shared_prediction_model_manager_eager_attention_decode.py tests/unit/test_shared_prediction_model_manager_eager_attention_mla.py tests/unit/test_shared_prediction_model_manager_mixed_layer_moe.py -q -p no:cacheprovider --basetemp $TMPDIR/related_verified_tmp
```

Observed **451 passed**; [related_verified.log](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/related_verified.log:1).

The B16 consumer rerun, excluding only the already completed cold training parity test:

```bash
timeout 180s $P -m pytest tests/unit/test_dsv4_operator_consumer.py -k 'not both_training_paths_and_warm_cache_exact_identity' -q -p no:cacheprovider --basetemp $TMPDIR/b16_verified_tmp
```

Observed **18 passed, 1 deselected**, one existing pandas warning; [b16_verified.log](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/b16_verified.log:1).

Executed dummy smoke matrix:

```bash
timeout 600s bash -c 'source /data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/env.sh; PYTHON_BIN=/data/ycfeng/envs/frontier-py310/bin/python METRICS_OUTPUT_DIR=$FRONTIER_TMP_ROOT/smoke bash examples/architecture/run_dummy_smoke_matrix.sh -- --metrics_config_cache_dir $FRONTIER_TMP_ROOT/smoke_cache'
```

Observed **12/12 passed**. A read-only HEAD source overlay in scratch also passed 12/12. `timeout 60s $P $S/compare_non_v4.py` compared all 36 CSV/system-metrics files: **byte and parsed equality, identical schemas, no omitted fields**. [non_v4_comparison.json](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/non_v4_comparison.json:1) records the paths and hashes.

V4 dummy-mode output change was checked against the same baseline overlay across 43 layers. With a 2 ms configured scope duration and a staged write, real decode/first_tail/prefill geometry changes from 2,320 to 2,326 ms of summed device scope costs: preparation +2, write +2, sampling +2. FULL/PIECEWISE dummies change 2,314→2,316 ms; NONE dummies 2,188→2,190 ms, preparation only. Existing operator costs, including zeroing, are unchanged. This is a MONOLITHIC comparison of identical geometries because the baseline rejects PD roles; these sums are not request latency or critical-path measurements. [v4_dummy_comparison.json](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/v4_dummy_comparison.json:1).

Cold reducer/promotion imports confirmed neither torch nor vLLM was imported. `git diff --check` passed. The original attention, linear, MoE and gap tables are byte-identical to HEAD; [unchanged_tables.json](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/unchanged_tables.json:1).

An initial 180-second consumer invocation did not finish; a later complete 900-second run superseded it. Earlier fixture/state-order failures and the ledger assertion were diagnosed and corrected. No outstanding acceptance failure within the writable implementation scope remains beyond the two reported prerequisites.

**Acceptance 8 — records and size: PASS.**

Current limitations and their tests describe the trained operator path, diagnostic-only step table, unsupported PD-AF/prefix caching/MTP/CPU overhead, and native sampling restriction.

Touched Python modules above 1,000 lines:

| Module | Before B17 | After B17 |
| --- | ---: | ---: |
| frontier/entities/batch.py | 1,612 | 1,634 |
| frontier/model_architectures.py | 1,342 | 1,342 |
| frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py | 1,914 | 1,918 |
| frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py | 1,073 | 1,109 |

No touched Python module exceeds 2,000 lines. Git-confirmed pre-stage-3 counts at d6ea9661 are 8,666 for sklearn_execution_time_predictor.py and 5,744 for metrics_store.py; current counts are 8,640 and 5,768. B17 changes neither file. The dated exception section records cleanup inspection, boundary reasons and the existing split sequence. [line_counts.json](/data/ycfeng/tmp/codex-implement/9dc571f6-ff2f-414e-9558-5d1cd02f2be2/b17/line_counts.json:1).

**4. Remaining work and decisions for Claude**

1. **Metrics writable prerequisite:** authorize/edit `frontier/metrics/metrics_store.py`, particularly its operation admission at lines 1645–1673, and add the three new runner OperationMetrics entries in constants.py. Exclude those entries for non-V4 clusters using the existing structural metric admission mechanism. The present failure is observed; the need for that filter follows from the existing initialization of every admitted enum as a metric series. Rerun both failed trace cases and the 36-file non-V4 comparison. Neither enums nor an out-of-scope metrics workaround were added here.
2. **PREFILL sampling coverage:** measure the proxy’s actual P contract separately, or make an explicit high-level decision to change the native P sampling contract and measurement scope. Current native max_tokens=1/min_tokens-unset requests must not reuse the decode-contract sampler rows. The implementation intentionally stops at this boundary.
3. **Accepted replay promotion:** accept G2h-04 and run the promotion script on that raw directory. To write a manifest covering both tables, pass both the accepted G2h-03 block input and G2h-04 replay input in the invocation. No G2h-04 promotion was attempted here.
4. **Known unpriced work:** add_requests staged copies, approximately 15–25 µs on steps with new requests, remain unowned as assigned. Replay zeroing remains diagnostic. No-write step records lack independent staging evidence.
5. **NOT_RUN, assigned to the lead:** A10 end-to-end replays, A11 diagnostic cross-check, the 77-case fidelity matrix and all GPU/platform/network checks. Scratch replay sampling values must not be reported as accepted fidelity measurements.

Native source settles the ordering and KV decisions at pinned revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`, under `/data/ycfeng/tmp/opencode/vllm-v0.28.0`:

- gpu/model_runner.py:1034–1038 zeroes newly allocated IDs; 1420–1424 performs finish/free/add/update and then applies block writes; DP dispatch/synchronization follows at 1432–1465; preparation follows at 1498–1546. No rendezvous change is needed.
- gpu/model_runner.py:992–994 adds full rows with overwrite=True; 1022–1025 appends cached-request blocks with overwrite=False; 913–940 removes finished/preempted requests; 964 removes old state before re-add.
- gpu/block_table.py:107–120 stages group entries and updates logical lengths.
- gpu/buffer_utils.py:239–271 skips absent staged writes, performs H2D, launches the write kernel and clears staging.
- worker/utils.py:211–232 skips empty zero IDs and otherwise performs H2D followed by the zero kernel.

**5. Review handoff**

Inspect the existing operator/training integration, runner activity ownership, pre-DP/attention/post-combine ordering, logical table-length lifecycle, NIXL first_tail classification, and explicit PREFILL coverage guard. Review the new block CSV and hashed manifest together; grid-derived new-row sizes must remain metadata, not consumer literals.

Rerun the consumer, ledger/KV/reducer suites and the two currently failing trace cases after metrics integration. Repeat the non-V4 baseline comparison after adding metric enums/filtering. After G2h-04 acceptance, rerun all exact runner-coordinate and cache-identity checks against the accepted table, then perform the lead-owned fidelity checks.

The implementation is reviewable within scope, but B17 is not fully accepted or complete until the metrics prerequisite and native PREFILL sampling coverage are resolved.

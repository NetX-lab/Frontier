# Test report: B17, the V4 model-runner operators and the PD roles

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: B17 (Codex `9dc571f6`, partial) review, the lead metrics correction, the lead checks, and the PREFILL sampling finding that opens B17b. |

## Scope

Assignment: [h3o_operator_costs/codex_assignment_B17_v4_runner_operators_and_pd.md](h3o_operator_costs/codex_assignment_B17_v4_runner_operators_and_pd.md). Codex report: `h3o_operator_costs/codex-implement-9dc571f6-ff2f-414e-9558-5d1cd02f2be2/implementation.md`.

Codex outcome: partial, for two reasons.

1. Two trace tests failed: `ValueError: 'dsv4_block_table_write' is not a valid OperationMetrics`. The runner names had no metric entries, and `metrics_store.py` was outside the writable paths.
2. PREFILL sampling raises `DeepSeekV4CoverageError` on purpose. The sampling rows measure the native decode request contract, and the P request uses a different one (see the finding below).

## Delivered (Codex)

| Area | Files | Behavior |
| --- | --- | --- |
| Runner tables | `frontier/profiling/model_runner/dsv4_runner_tables.py` (new, torch-free), `h3o_operator_costs/h200_tables/promote_runner_tables.py` (new), `runner_manifest.json` (new) | Streams the raw JSONL, joins the preparation descriptor by `case_id`, requires a constant activity count per coordinate, writes hashed provenance |
| Block table | `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/model_runner_block_kernel_only.csv` (new) | 197 rows from the G2h-03 path-B `record_function` rows: zeroing n = 1-32 and 2,520 (33), writes w_c = 0-32 x r = 0-4 without (0, 0) (164); every row 1 kernel + 1 memcpy |
| Consumer | `dsv4_runner.py` (new registry), `dsv4_operator_predictor.py`, `dsv4_training.py`, `dsv4_runtime.py`, `deepseek_v4.py`, `dsv4_composition.py` | Exact lookup; cost = kernel median + memcpy median; gap = activity count x `eager_preparation`; zeroing and writes before the DP rendezvous; preparation at the head of the layer-0 prefix on real and dummy forwards; sampling after the logits gather on real forwards; zero coordinates charge nothing; typed errors for every unmeasured coordinate |
| Write coordinate | `entities/batch.py` (`KVBlockTableWrites`), `vllm_v1_kv_allocation.py`, `vllm_v1_engine_replica_scheduler.py` (4 lines) | Per-request group table lengths: a new row writes its full lengths, a continuing row appends the length differences, free and preemption drop the saved lengths |
| PD roles | `attention/deepseek_v4.py`, `attention/dsv4_features.py` | MONOLITHIC, PREFILL and unified DECODE admitted; DECODE_ATTN and DECODE_FFN still rejected; the NIXL D-side new row with query 1 and context prompt-1 is the `first_tail` table phase |
| Records | `model_architectures.py` (`runtime_limitations`), `families.py` (text), `docs/general/module-size-exceptions.md` | Current scope text; dated module-size section |

## Lead review

- The `KV_CONNECTOR` rejection and the synthetic-only zeroing block in `deepseek_v4.py` are removed; the zeroing now goes through the same runner path as the other three scopes.
- `batch._cluster_type` is read outside `Batch`. This follows the existing scheduler idiom (`m2n_arrival.py:113`, `replica_state.py:29`), so it stays.
- `tests/unit/test_dsv4_operator_consumer.py` reads the scratch replay table through `FRONTIER_DSV4_REPLAY_TABLE` while no replay table is committed. B17b removes that override once the G2h-04 table is committed.
- The reducer's per-row scans over all steps are quadratic. The promotion is a one-off CPU tool and finished within its timeout, so this is not changed.

## Lead correction: runner metric registration

- `frontier/metrics/constants.py`: three entries, `DSV4_BLOCK_TABLE_WRITE`, `DSV4_INPUT_PREPARATION`, `DSV4_SAMPLING`.
- `frontier/metrics/metrics_store.py` (+5 lines, registration only): the existing non-V4 line that excluded `kv_cache_block_zero` now excludes every operator of `KV_CACHE_FAMILY` and `DEEPSEEK_V4_RUNNER_FAMILY`.
- `tests/unit/test_glm53_structural_execution.py`: the expected non-V4 catalog subtracts the same two families instead of the single zeroing entry.

## Finding: the P request has no logits processing (opens B17b)

- My earlier note (progress row of 2026-10-29, commit `994d293c` message) said the PD proxy sends the P request `min_tokens = max_tokens = 1`. That holds only for the pre-0.28 branch. On the vLLM 0.28.0 branch, `pd_proxy.py:103-108` sets `max_tokens = 1` and removes `min_tokens` and `min_completion_tokens`. The D request keeps the original body (`pd_proxy.py:141`), including `min_tokens`.
- Native rule (pinned `2cf0a691`): `sampler.py:74-87` sets `needs_logits_processing` per request (logit bias, which `min_tokens > 0` with stop tokens turns on in `logit_bias.py:89-107`, penalties, bad words, non-greedy settings). `apply_sampling_params` (`sampler.py:192-199`) skips the FP32 copy and every processing kernel when no request in the batch needs it.
- Native evidence: `evidence/v4_operators/p2_32_prefill_dp0.txt` has no `_bias_kernel`; its post section has `_gumbel_sample_kernel` 22.70 µs and the argmax reduce. The decode traces have `_bias_kernel`.
- The sampling-state staged writes run in `add_requests` (`model_runner.py:1008-1011`), outside the preparation scope, so preparation does not depend on the contract.
- G2h-02 sent no `min_tokens` (the request contract before `994d293c`), so its 976 sampling rows measure the batch-level `needs_logits_processing = False` branch at b = 1-24. G2h-04 measures the `True` branch. Both are H200 measurements of one native branch each; no step table and no scaling are involved.
- B17b: the sampling coordinate becomes (`needs_logits_processing`, b); the role decides the value (PREFILL False because of the proxy; MONOLITHIC and DECODE True because the replay sends `min_tokens = max_tokens`); the PREFILL guard is removed.

## Lead checks

Environment: Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6 (no torch, no vLLM); `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`; `TMPDIR=/data/ycfeng/tmp/lead_b17/tmp`; `ulimit -v 31457280` (smoke 62914560). Consumer suites also set `FRONTIER_DSV4_REPLAY_TABLE` to the Codex scratch G2h-02 structure table and `FRONTIER_DSV4_TEST_MODEL_ROOT` to its trained cache.

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Structural suites (10 files: runner ledger, cache zero, grouped KV scheduler, structural PD and execution, graph measured, source execution, attention runtime, DP sync, attention workload) | All pass, including the 2 Codex failures | 253 passed | PASS |
| 2 | Consumer suites (`test_dsv4_runner_consumer.py`, `test_dsv4_kv_block_table_writes.py`, `test_dsv4_runner_tables.py`, `test_dsv4_operator_consumer.py`) | All pass | 45 passed (230 s), pandas mixed-type warnings only | PASS |
| 3 | Block table reproduction (`promote_runner_tables.py` on G2h-03 into scratch) | Same SHA-256 as the committed CSV | Both `e0a6dadf...c3139cd`, 197 rows | PASS |
| 4 | Dummy smoke matrix (`examples/architecture/run_dummy_smoke_matrix.sh`, output `/data/ycfeng/tmp/lead_b17/smoke`) | 12 of 12 | 12 of 12, exit 0 | PASS |
| 5 | Non-V4 outputs against the HEAD baseline (`/data/ycfeng/tmp/lead_b17/compare_non_v4.py` against Codex's HEAD-overlay `baseline_smoke`) | Every CSV and `system_metrics.json` equal after parsing | 36 of 36 equal | PASS |
| 6 | Metrics and V4 related regression (33 files: metrics stage and scope, transfer metrics, GLM structural, DP placement, preemption, PP admission and overlap, spec decode, V4 features, predictor, composition, replay rows and gaps, H200 tables, gap pricing, operator registry, and others) | Only failures that also fail on HEAD | 810 passed, 5 failed. 4 GLM catalog failures were the old expected set (fixed above; rerun 31 passed). 1 failure, `test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` (`{2: (0, False)} != {2: (0, True)}`), fails identically at HEAD `1fcc6880`, at `c428295c` and at pre-stage-3 `d6ea9661` (HEAD sources from `git archive`) | PASS (1 failure outside B17) |
| 7 | `git diff --check` | Clean | Clean | PASS |
| 8 | Module sizes | Touched modules at or below 2,000 lines; the two large modules take registration lines only | `batch.py` 1,634, `vllm_v1_engine_replica_scheduler.py` 1,918, `vllm_v1_kv_allocation.py` 1,109; `sklearn_execution_time_predictor.py` 8,640 (unchanged), `metrics_store.py` 5,773 (+5) | PASS |

`tests/unit/test_vllm_rocm_attention_wrapper_increment9.py` needs torch and was left out of check 6.

## Limits

- The replay table (preparation and sampling) is not committed yet. Every consumer check above used the G2h-02 structure table from scratch; the committed table follows G2h-04 acceptance and B17b.
- PREFILL forwards stop at sampling until B17b.
- The `add_requests` staged copies stay unowned and unpriced (about 15-25 µs on steps with new requests).
- No E2E replay, no fidelity matrix and no A11 cross-check were run here.

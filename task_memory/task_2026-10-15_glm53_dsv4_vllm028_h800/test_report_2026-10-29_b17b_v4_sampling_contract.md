# Test report: B17b, the V4 sampling branch and the committed replay table

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: B17b (Codex `66e252e1`) review and lead checks. |

## Scope

Assignment: [h3o_operator_costs/codex_assignment_B17b_v4_sampling_contract_and_replay_table.md](h3o_operator_costs/codex_assignment_B17b_v4_sampling_contract_and_replay_table.md). Codex report: `h3o_operator_costs/codex-implement-66e252e1-2862-4858-abb7-ed7bf9edf9ee/implementation.md`. Codex outcome: complete, no open blocker.

## Delivered (Codex)

| Area | Files | Behavior |
| --- | --- | --- |
| Registry | `frontier/execution_time_predictor/dsv4_runner.py` | `SAMPLING_COLUMNS = ("needs_logits_processing", "b")`; the role mapping MONOLITHIC 1, unified DECODE 1, PREFILL 0 with its source comment; the identifying kernel `_bias_kernel`; `sampling_coordinate(batch)` with b the number of real requests |
| Consumer | `dsv4_runtime.py`, `dsv4_operator_predictor.py` | `add_runner_scopes` prices sampling at the role coordinate; the PREFILL `DeepSeekV4CoverageError` is removed; the loader validates both columns (branch 0 or 1, b positive) and the trainer uses both as features; exact lookup unchanged |
| Reducer and promotion | `frontier/profiling/model_runner/dsv4_runner_tables.py`, `h3o_operator_costs/h200_tables/promote_runner_tables.py` | Each replay run declares its branch; every real step must have sampling activity and must show `_bias_kernel` exactly when the declared branch is 1, otherwise the promotion stops and names the run, case and step. The second run contributes sampling rows only. The two runs must declare different branches. The G2h-02 structure-fixture guard is removed |
| Tables | `data/.../DeepSeek-V4-Flash/model_runner_replay_kernel_only.csv` (new), `runner_manifest.json` | 2,941 rows: preparation 989 (138 keys) from G2h-04; sampling 976 per branch (24 keys each, b = 1-24) from G2h-04 (branch 1) and G2h-02 (branch 0). The block table is rewritten byte-identical. The manifest records 213 hashed inputs, both replay runs with their row counts, and three source tags |
| Text | `frontier/model_architectures.py` | V4 `runtime_limitations`: both measured request contracts; other sampling parameters unmeasured |
| Tests | `test_dsv4_runner_tables.py`, `test_dsv4_runner_consumer.py`, `test_dsv4_operator_consumer.py` | Branch mismatch in both directions, partial branch and missing sampling, two-source promotion and manifest; role-specific sampling for the three roles; A1 with PREFILL sampling; worked b = 8 examples; the `FRONTIER_DSV4_REPLAY_TABLE` and `FRONTIER_DSV4_TEST_MODEL_ROOT` overrides and the pending-table skip are removed |

## Lead review

- The role mapping reads `batch._cluster_type`, which the vLLM V1 replica scheduler sets on real and dummy forwards (`vllm_v1_engine_replica_scheduler.py:448,729`), as B17 already does in `attention/deepseek_v4.py` and `dsv4_features.py`. PD-AF roles have no entry; they are rejected before this point.
- The loader converts the two sampling columns to numbers before validation, because preparation rows leave them blank and pandas then reads the column as strings. A malformed value raises the typed coverage error. This is input validation of a table, so it stays.
- `test_dsv4_runner_tables.py` loads the promotion script from the task directory, as nine other unit tests already do with task-directory scripts.
- No correction was needed.

## Lead checks

Environment: Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6 (no torch, no vLLM); `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`; `TMPDIR=FRONTIER_TMP_ROOT=/data/ycfeng/tmp/lead_b17b/tmp`; `ulimit -v 31457280` (smoke 62914560). Scripts and logs: `/data/ycfeng/tmp/lead_b17b/` (`env.sh`, `run_consumers.sh`, `run_smoke.sh`, `wrong_branch.py`, `logs/`). No environment override of any table path.

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Promotion into scratch (`promote_runner_tables.py` with G2h-03 block, G2h-04 replay `--replay-logits-processing 1`, G2h-02 `--sampling-logits-processing 0`, `--output-dir /data/ycfeng/tmp/lead_b17b/promote`) | Both CSVs equal to the working-tree files; block table unchanged from B17 | Block `e0a6dadf...c3139cd` (197 rows), replay `de7607f1...e1da197211b9ec26d9981e10c0cba15da63f81f58f` (2,941 rows) in both places. The manifests differ only in the `outputs` paths (scratch against the data directory) | PASS |
| 2 | Independent table summary (standard-library CSV) | Preparation from G2h-04 only; sampling medians equal to the G2h-04 acceptance (check 9 of [test_report_2026-10-29_g2h_runner_04.md](test_report_2026-10-29_g2h_runner_04.md)) | Preparation 989 rows, 138 keys, all G2h-04, median 0.09367 ms. Sampling branch 1: 976 rows, all G2h-04, 11 kernels, median 0.05057 ms. Branch 0: 976 rows, all G2h-02, 9 kernels, median 0.03024 ms. b = 1-24 in both | PASS |
| 3 | Wrong declaration on real inputs (`wrong_branch.py`) | Typed stop at the first real step | G2h-02 declared 1: `Run g2h-02, case g1-e0-0000-full, step 0: declared needs_logits_processing=1, observed 0 from _bias_kernel`; G2h-04 declared 0: the same message with 0 / 1 | PASS |
| 4 | Consumer, reducer and ledger suites (`test_dsv4_runner_tables.py`, `test_dsv4_runner_consumer.py`, `test_dsv4_operator_consumer.py`, `test_dsv4_runner_ledger.py`, `test_dsv4_kv_block_table_writes.py`) on the committed tables | All pass | 62 passed in 560 s, 7 pandas mixed-type warnings from the attention reader | PASS |
| 5 | Structural suites (11 files: cache zero, grouped KV scheduler, V4 structural PD and execution, graph measured, source execution, attention runtime, DP sync, attention workload, GLM structural execution and PD) | All pass | 291 passed | PASS |
| 6 | Related regression (23 files: V4 features, predictor, model config, operator manifest and admission, composition, runner operators, replay rows and gaps, H200 tables, step-cost input, gap pricing, operator registry, three shared-manager files, metrics stage and scope, transfer metrics, DP placement, pilot cases, attention coverage) | Only the failure known from HEAD | 730 passed, 1 failed: `test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`, the failure that also fails at `1fcc6880`, `c428295c` and `d6ea9661` (B17 report, check 6) | PASS (1 failure outside B17b) |
| 7 | Dummy smoke matrix (`run_dummy_smoke_matrix.sh`, output `/data/ycfeng/tmp/lead_b17b/smoke`) | 12 of 12 | 12 of 12, rc 0. A first attempt used the host `python3` because my script did not set `PYTHON_BIN`, and stopped at `No module named 'plotly'` (kept as `logs/smoke_wrong_python.log`); the rerun set `PYTHON_BIN` | PASS |
| 8 | Non-V4 outputs against the B17 lead smoke (`/data/ycfeng/tmp/lead_b17/compare_non_v4.py /data/ycfeng/tmp/lead_b17/smoke /data/ycfeng/tmp/lead_b17b/smoke`) | Every CSV and `system_metrics.json` equal after parsing | 36 of 36 equal | PASS |
| 9 | `git diff --check` and module sizes | Clean; touched modules at or below 2,000 lines | Clean; `dsv4_runner.py` 66, `dsv4_operator_predictor.py` 386, `dsv4_runtime.py` 101, `dsv4_runner_tables.py` 169, `model_architectures.py` 1,343 | PASS |

Codex also ran a HEAD-overlay smoke and the same 36-file comparison (36 equal) and a standard-library hand check of all 2,941 rows against the trained exact lookups; I did not repeat those two.

## Effect on pricing (from the committed tables)

- A real PREFILL forward (b = 1) now prices sampling at 0.035243 ms plus 9 × 1.2208671 µs = 0.010988 ms of gap, where B17 raised a coverage error.
- A b = 8 DECODE or MONOLITHIC forward prices 0.043093 ms plus 11 × 1.2208671 µs = 0.013430 ms, which is 0.018017 ms more than B17's branch-0 value at the same b.
- Preparation now comes from G2h-04; its per-case values are within 1.7 % (median ratio) of the G2h-02 values B17's tests used (G2h-04 report, check 9).

## Limits

- Sampling covers the two measured request contracts only; other sampling parameters (penalties, bad words, non-greedy settings) are unmeasured.
- Sampling rows come from one GPU with TP1; the TP8 path samples the same gathered full-vocabulary logits (inference from the vLLM V1 code path), with no TP8 measurement.
- The `add_requests` staged copies stay unowned and unpriced (about 15-25 µs on steps with new requests).
- No E2E replay, no 77-case fidelity matrix and no A11 cross-check are part of this report.

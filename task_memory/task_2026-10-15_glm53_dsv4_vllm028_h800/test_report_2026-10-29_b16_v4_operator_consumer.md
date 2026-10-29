# Test report: B16, the V4 operator consumer on the H200 tables

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Part 2: B16b (Codex `86155181`) accepted: the trained V4 forward, the lead test correction and the lead checks. The B16 limits section is updated. |
| 2026-10-29 | Created: B16 part 1 (Codex `54e2e32d`, partial): table promotion, gap precision fix, the three admission prerequisites that stopped the runtime work, and the lead checks. |

## Part 1: Codex `54e2e32d` (outcome partial)

Assignment: [h3o_operator_costs/codex_assignment_B16_v4_operator_consumer.md](h3o_operator_costs/codex_assignment_B16_v4_operator_consumer.md). Codex report: `h3o_operator_costs/codex-implement-54e2e32d-92b5-49b8-b578-4119b950ab6b/implementation.md`.

### Delivered

| Path | Change |
| --- | --- |
| `h3o_operator_costs/h200_tables/promote_tables.py` | New. Builds the four H200 V4 tables with the existing reducers (`dsv4_attention_main aggregate`, `dsv4_replay_gaps`) on scratch copies of the G1h-02 raw records, and the row unions of the G2h-01 operator + EP16 shared-expert rows and the G3h EP16 + EP32 MoE rows. Writes `manifest.json` (12 inputs with SHA-256, 4 outputs with SHA-256 and row counts, revisions). |
| `h3o_operator_costs/h200_tables/manifest.json` | New provenance manifest. |
| `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/` | New: `attention_graph_replay_kernel_only.csv` (4,945 rows), `kernel_gap.csv` (4), `linear_op_kernel_only.csv` (41), `moe_kernel_only.csv` (33). |
| `frontier/execution_time_predictor/kernel_gap.py` | `read_dsv4_kernel_gap_ms` parses with `float_precision="round_trip"`, so every recorded digit survives; the generic reader is unchanged. |
| `tests/unit/test_dsv4_h200_tables.py` | New, 13 cases: union, revision reader, hashes and counts against the manifest, KERNEL_ONLY only, exact gaps, invalid gap rows. |

Lead correction: the promotion script's scratch directory was the Codex request scratch by default. `--scratch-dir` is now required, because the scratch location belongs to the executor.

### Not delivered, and why

Codex stopped the runtime part (training function, trained pricing, composition, segment gaps, host time, runner coverage error). Three admission guards block a non-dummy V4 predictor, and their files were outside the assignment's writable paths. The lead missed them when writing B16. Observed by Codex and re-checked by the lead:

1. `frontier/operators/families.py:314-318`: `DEEPSEEK_V4_MHC_FAMILY` has `execution_enabled=False` ("source inventory only"). `bind_operator_query("dsv4_mhc_first_pre")` and `validate_typed_operator_contracts(...)` raise `NotImplementedError` for every row of the promoted linear-op table (lead probe: 0 of 41 rows pass).
2. The assignment told the loader to validate every operator contract with `profile.resolve_layer_contract`. That is wrong for the mHC and endpoint operators: the producer writes them with the non-layer contract of `frontier/profiling/linear_op/profiling_plan.py:153` (`layer_kind: null`), and the `deepseek_v4` profile has no layer contract for that family (`get_layer_contract_for_family` finds 0). The shared expert and the MoE rows do resolve through layer contracts (15 and 132 contracts passed).
3. `frontier/attention/families.py:652`: `DEEPSEEK_V4_ATTENTION_FAMILY` has `execution_enabled=False`. `base_execution_time_predictor.py:44-47` admits it only in dummy mode or with a bound step-cost table, so a non-dummy TP8/DP2/EP16 predictor fails at construction.

These go to the follow-up assignment B16b, with the corrected loader rule for non-layer contracts.

### Lead checks (HEAD `1fe9603d` plus the B16 changes)

Environment: Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6); `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `TMPDIR=/data/ycfeng/tmp/b16_accept/tmp`, `ulimit -v 31457280`.

| # | Check | Command | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | Promotion reproduces after the lead correction | `promote_tables.py --scratch-dir /data/ycfeng/tmp/b16_accept/promotion` | Exit 0; the four tables and `manifest.json` byte-identical to the Codex outputs | Exit 0; `sha256sum` of the four tables equal; `cmp` of the manifest equal | PASS |
| 2 | Focused and import regression | `pytest tests/unit/test_dsv4_h200_tables.py test_dsv4_runner_operators.py test_kernel_gap_pricing.py test_dsv4_replay_rows.py test_dsv4_replay_gaps.py test_dsa_predictor.py -k 'not test_every_collective_launches_one_kernel_by_default'` | All pass | 259 passed, 1 deselected (22.27 s) | PASS |
| 3 | Whitespace | `git diff --check` | Clean | Clean | PASS |
| 4 | mHC registry validation (prerequisite 1) | Probe: `validate_typed_operator_contracts` on each row of the promoted `linear_op_kernel_only.csv` | Documents the blocker | `NotImplementedError: Operator family deepseek_v4_mhc is not enabled for execution` on the first row; 0 of 41 pass | Blocker confirmed |

Codex's own checks (not rerun by the lead): two promotions byte-identical; 4,945 attention identities equal to `/data/ycfeng/tmp/b13r_accept/after/rows.json` over 95 fields; the four gap values equal to G1h-02 to every digit; placement suite 18 passed / 1 failed before and after (the known late-join failure); dummy smoke matrix 12 of 12; cold imports load no torch, vLLM or CUDA.

### Observations reported by Codex (not gated)

- EP16 and EP32 shared-expert medians agree within 1 % at every L (largest gap: `share_expert_up_proj` L = 1, 8.8485 against 8.7685 µs).
- The existing trainer reduces duplicate feature rows by the arithmetic mean (`sklearn_execution_time_predictor.py:191`). The G3h gate rows have 12, 4, 4, 3 and 10 duplicates at L = 1, 2, 3, 4 and 768.
- All seven V4 collective specs report `kernel_count=1`.

## Part 2: B16b, Codex `86155181` (outcome complete)

Assignment: [h3o_operator_costs/codex_assignment_B16b_v4_operator_consumer_runtime.md](h3o_operator_costs/codex_assignment_B16b_v4_operator_consumer_runtime.md). Codex report: `h3o_operator_costs/codex-implement-86155181-0e36-4ecf-9c67-6e5ed04b9aa2/implementation.md`. Codex scratch: `/data/ycfeng/tmp/codex-implement/86155181-0e36-4ecf-9c67-6e5ed04b9aa2/b16b`.

### Delivered

| Path | Change |
| --- | --- |
| `frontier/execution_time_predictor/dsv4_operator_predictor.py` | New (257 lines). Loads the KERNEL_ONLY linear-op and MoE tables. mHC and endpoint rows are checked against their non-layer contract; shared-expert and MoE rows against the resolved layer contract. Checks the FP8 DeepGEMM backend of the shared-expert projections, the MXFP4 MARLIN expert identity, balanced routing and the rank-class coordinates. `DeepSeekV4OperatorPredictor` answers only measured coordinates (time and kernel count) and raises `DeepSeekV4CoverageError` otherwise. |
| `dsv4_training.py`, `dsv4_runtime.py`, `dsv4_measurement.py` | New. One V4 training entry for the predictor and the shared manager; the segment-owner table, the trained attention composition and the operator-gap publication; KERNEL_ONLY selection for every V4 role and the typed error when host time is not the explicit zero. |
| `deepseek_v4.py` | Standalone, shared-expert, gate and grouped-GEMM scopes use trained predictions and counts outside dummy mode; composed attention; host time zero; `kv_cache_block_zero` raises a coverage error until B17 prices it. `read_isolated_scope` and `moe_shuffling` (V4 does its permutation inside the MARLIN grouped GEMM) removed. 807 to 676 lines. |
| `kernel_gap.py` | V4 reads all four segment gaps at attention TP and fails when `kernel_gap.csv` is missing; the V4 layer wrapper publishes per-operator gaps once. |
| `families.py` (attention and operators) | `DEEPSEEK_V4_ATTENTION_FAMILY` and `DEEPSEEK_V4_MHC_FAMILY` enabled. |
| Routing-only edits | `sklearn_execution_time_predictor.py` (8,626 to 8,640, imports and early returns), `shared_prediction_model_manager.py`, `sklearn_moe_execution_time_predictor.py` (blanket rejection removed), `moe_dataset_training.py`, `base_execution_time_predictor.py`, `expert_parallel.py` (EP lanes priced through `_predict_layer_with_kernel_gap`), `profiling_metadata.py` (`CHECKPOINT_MXFP4_BACKEND_IDENTITIES`), `deepseek_v4_axes.py` (`select_deepseek_v4_operator_tp`, shared with the producer plan). |
| Tests | New `test_dsv4_operator_consumer.py` (19 cases, including the A1 sentinel for a FULL decode and a NONE prefill forward) and `test_dsv4_operator_admission.py` (14 cases). 19 existing test files flip the disabled-family assertions to enabled, drop the removed reader's tests, or complete model stubs. |

### Lead review

- Gap accounting checked by reading the code: attention elapsed time comes from the composed schedule (`execution_time.py:1138`), which already holds the attention gaps; the other phases add the published per-operator gap once through `timing_operator_times` (`deepseek_v4.py:70`). The composition reads the operator costs before the gaps are published (`kernel_gap.py:212-218`), so no gap is counted twice.
- The test changes only flip the disabled-family assertions, remove the tests of the deleted reader, and complete model stubs. No assertion on numeric results was relaxed.
- Lead correction: `tests/unit/test_dsv4_runner_operators.py::test_package_check_writes_no_destination` still pinned `package_03` and its job tags; my G2h-04 plan change (`18161ea2`) moved them to `package_04`. The test now pins `package_04` and the `-04-` tags.
- The test `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default`, deselected in earlier reports as a known failure, passes since B15 (`5896339e` added `kernel_count`). The lead run includes it.

### Lead checks (HEAD `c428295c` plus the B16b changes)

Environment as in part 1, with `TMPDIR=/data/ycfeng/tmp/lead_b16b/tmp`, `FRONTIER_TMP_ROOT=/data/ycfeng/tmp/lead_b16b`.

| # | Check | Command | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | Affected-module regression, both new files, no deselection | Codex's argv (`S/affected_rerun_command.json`, 107 test files) with `--basetemp /data/ycfeng/tmp/lead_b16b/pytest_tmp` and the `--deselect` removed; argv in `/data/ycfeng/tmp/lead_b16b/argv.json` | Only the existing failures | 2,538 passed, 17 failed, 3 skipped (395 s). The 17 failures equal Codex's list minus the corrected runner test: 10 in `test_colocation_release_review_contracts.py` (sources absent at HEAD: `frontier/config_optimizer/config_explorer/main.py` and three `tests/debug/e2e-level/...` scripts), 2 Stage-2 CLI tests (flashinfer absent), 5 in `test_profiling_governance_minimal_red.py` (torch absent) | PASS (existing failures only) |
| 2 | Dummy smoke matrix | `timeout 900s env METRICS_OUTPUT_DIR=/data/ycfeng/tmp/lead_b16b/dummy_outputs bash examples/architecture/run_dummy_smoke_matrix.sh -- --metrics_config_cache_dir /data/ycfeng/tmp/lead_b16b/dummy_cache` | 12 of 12 | 12 of 12 completed | PASS |

Codex's checks (not rerun by the lead, because the lead changed no runtime code): consumer 19 passed (217 s), admission 14 passed; placement suite 18 passed / 1 failed before and after (the known `test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`); 568 trained models; cold imports load no torch, vLLM or CUDA; flag-consumer probe before and after with identical operation sets.

## Limits

- The runner scopes are not priced yet: `kv_cache_block_zero`, the block-table writes, input preparation and sampling (the LM head and the logits gather are priced, from the linear-op table and the collective consumer). A non-dummy forward with a zeroing request raises `DeepSeekV4CoverageError`. B17 adds them (gates A3 and A4 complete there).
- PREFILL and unified DECODE roles are not admitted yet (B17 entry 4).
- Gates A6 (accuracy), A10 (E2E) and A12's 77-case fidelity matrix are not run; they follow B17.

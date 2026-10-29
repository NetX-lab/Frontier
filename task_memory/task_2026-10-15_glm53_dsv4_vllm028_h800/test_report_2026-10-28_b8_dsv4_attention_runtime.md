# Test report: stage 1a assignment B8, DeepSeek-V4 attention at runtime (2026-10-28)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: lead review and verification of Codex assignment B8 (`bb6c54ec`), lead corrections R1, R3 and R4, and the Git state incident caused by the lead's whole-suite comparison run. |
| 2026-10-28 | Git repair applied after YC's approval and verified; whole-suite after half rerun without `GIT_DIR` (only R4 differs); whole-suite row PASS. |

## Scope

- Assignment: [h3o_operator_costs/codex_assignment_B8_dsv4_attention_runtime.md](h3o_operator_costs/codex_assignment_B8_dsv4_attention_runtime.md) (entries 4, 5 and the attention part of entry 3; entry 10's wiring is deferred).
- Codex report: [h3o_operator_costs/codex-implement-bb6c54ec-e30e-4efd-928b-36dce74bd0b0/implementation.md](h3o_operator_costs/codex-implement-bb6c54ec-e30e-4efd-928b-36dce74bd0b0/implementation.md) (outcome `partial`; the only open item was literal A9 hygiene, resolved below).
- Base: `dd4c4317`.
- Changed files (all inside the assigned scope):
  - `frontier/attention/deepseek_v4.py` (419 lines): one V2 builder, `build_deepseek_v4_attention_workload`, for real and dummy batches in NONE, PIECEWISE and FULL.
    - The compressor store runs on every call with compression ratio > 1.
    - The CSA indexer uses the short-context fill only under NONE when max(s)//r ≤ top-k.
    - The `dsv4_index_scores` scope is present when some request has q ≤ 1, or when the q > 1 requests have Σ s//r > 0.
    - A dummy uses vLLM's geometry: FULL has R requests of one row; PIECEWISE and NONE have one request of R rows. Valid-row fields are 0 and serving fields are empty.
    - The result records `indexer_short_context` and `plans_tile_schedule`.
  - `frontier/attention/families.py` (710): `deepseek_v4_main_kv_cache_dtype`, the shared alias fp8 → fp8_ds_mla.
  - `frontier/entities/batch.py` (1612): the `DummyForwardBatch` docstring only.
  - `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` (8626, registration lines only): V4 replay predictor registration, a single-row `_train_model` branch, and the threshold attached to the models.
  - `frontier/execution_time_predictor/deepseek_v4.py` (804): entry 3, `predict_attention_operator_times`, which returns the scope costs, `_dsv4_scope_gap_ms` and `_dsv4_composition_inputs`.
  - `frontier/execution_time_predictor/dsv4_predictor.py` (333): `load_attention_dataframe` and `validate_balanced_routing` moved here from `deepseek_v4.py`.
  - `frontier/execution_time_predictor/profiling_dataframe_loaders.py` (975): the V4 branch is removed. The shared manager path now raises `NotImplementedError` for V4 until final admission.
  - Deleted (the four authorized files): `frontier/execution_time_predictor/dsv4_attention_input.py`, `tests/comparison/calibration/prepare_dsv4_attention_input.py`, `tests/integration/check_dsv4_measured_scopes.py`, `tests/fixtures/deepseek_v4_attention_layouts.json`.
  - `tests/integration/run_dsv4_structural.py`: the boundary store rule becomes `store = ratio > 1`.
  - Tests:
    - `tests/unit/test_dsv4_attention_runtime.py` (new, 235 lines): the owner pattern of Acceptance 2;
    - `test_dsv4_attention_workload.py`, `test_dsv4_structural_execution.py` (455), `test_dsv4_source_execution.py`, `test_dsv4_predictor.py`, `test_profiling_runtime_boundary.py`;
    - `test_dsv4_graph_measured.py` (215): the isolated-attention tests were removed with the isolated path;
    - `test_glm53_attention_cost_contract.py`: lead correction R4.

## Lead review

- **Builder rule.** The scope rule and the dummy geometry match design §3.4 and the lead's rule script in `b8_design/`. Acceptance 1 checks them against every row of the real table.
- **Registration.** It follows the DSA owner pattern (`tests/unit/test_dsa_predictor.py:404-505`). A key with one row trains through the single-row branch rather than through cross-validation. The `multi_stream_gemm_token_threshold` of the table travels with the models. A second training from the cache leaves the files and the model hashes unchanged (Acceptance 2).
- **Lead corrections** (small, made directly):
  - **R1** `frontier/attention/deepseek_v4.py`. The `dsv4_sparse_attention` call record repeated four request-order fields of the top-level result: `semantic_is_decoding`, `source_request_order`, `num_decode_query_tokens` and `num_prefill_query_tokens`. They are removed from that call; the top-level result keeps them. The call keeps `decode_request_indices` and `prefill_request_indices`, which the sparse kernel consumes. `test_dummy_attention_uses_v2_descriptor_geometry` now checks the top-level fields and the two call fields separately.
  - **R3** `frontier/execution_time_predictor/dsv4_predictor.py:78-89`. Three `validate_balanced_routing` messages still said "balanced isolated attention" after the isolated path was deleted; they now say "balanced replay attention".
  - **R4** `tests/unit/test_glm53_attention_cost_contract.py`. The whole-suite comparison found one regression caused by B8: `test_unmeasured_family_rejects_csv_loading_and_standalone_training` expected the deleted V4 loader branch to raise `DeepSeekV4CoverageError`. The intended B8 behavior is that the shared manager path raises `NotImplementedError("deepseek_v4_attention is not enabled for execution")` until final admission. The expectation was updated and the unused import removed. 41 passed together with `test_dsv4_attention_runtime.py`.
- **Callers.** `serving_context` and `predict_source_execution` stay in `execution_time_predictor/deepseek_v4.py`; `sklearn_moe_execution_time_predictor.py` and `scheduler/utils/expert_parallel.py` still import them from there. `read_isolated_scope` remains only for the non-attention scopes in `test_dsv4_graph_measured.py`. No caller of the moved `load_attention_dataframe` or `validate_balanced_routing` uses the old module.
- **A9 hygiene exception (accepted).**
  - `sklearn_execution_time_predictor.py` has 5 unused imports. They are identical at `dd4c4317` and are not touched by B8's registration-only scope.
  - Stale `.pyc` files of the deleted modules exist in untracked `__pycache__` directories. The removed-name grep, restricted to `*.py` source files, finds nothing. The `.pyc` files were left in place (no deletion was authorized).
- **Module sizes.** All at or below 2,000 lines except `sklearn_execution_time_predictor.py` (registration lines only, as authorized).

## Checks

Environment:
- Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vllm).
- Variables: `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`.
- Memory limits: `ulimit -v 31457280` for tests and analysis; `ulimit -v 62914560` for structural runs.
- Lead scratch: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b8_review/`; scripts and outputs under `checks/`.
- Real table: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_review/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv`, sha256 `277626cd766865bd...`.

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| A1 scope sets | `python checks/a1_scope_sets.py` (after R1) | all 505 rows: scope set, `indexer_short_context`, `plans_tile_schedule` and R equal the row's | 505 rows, 0 mismatches. By (phase, mode): decode FULL 320, decode PIECEWISE 80, first_tail PIECEWISE 50, prefill NONE 5, dummy FULL 25, dummy PIECEWISE 20, dummy NONE 5 | PASS |
| A2 owner pattern | `pytest tests/unit/test_dsv4_attention_runtime.py -q` (inside the A8 run) | passes | passed | PASS |
| A3 trained runtime path | `python checks/a3_runtime_table.py checks/a3_runtime_table.json`: lead's own script with an independent exact lookup (`DecisionTreeRegressor`), the h200 TP8 / attn_dp 8 / EP64 replica, `kv_cache_dtype="fp8"`, batches built from the row columns | B7 per-row values; median ≤ 5%, p90 ≤ 10% | 554 models; median 0.4909%, p90 1.3427%, max 6.6727%; rows 13 and 204 above 5% with B7's predicted values; `b7_equal` true | PASS |
| A4 unchanged | `python .../b6_baseline/a4_baseline.py checks/a4_report.json`; `cmp` with `a4_report_d2a226fe.json` | byte-identical | `PASS 125`; `cmp` exit 0 | PASS |
| A5 import boundary | `pytest tests/unit/test_profiling_runtime_boundary.py -q` | passes (it failed at B7 on the `dsa_rows` import) | passed | PASS |
| A6 step-cost path | `pytest tests/unit/test_dsv4_step_cost_input.py -q`, file unchanged | passes | passed; no diff in the file | PASS |
| A7 dummy-mode differences | `run_dsv4_structural.py --dp 4 --case both` on a `git archive dd4c4317` snapshot (`run_before2.sh`) and on the working tree after R1 and R3 (`run_after.sh`); `checks/structural_summary.py` on each run, then a field-by-field diff | both exit 0; idle HCA 9 → 11, CSA 13 → 16, uncompressed 8 → 8 calls; real HCA calls gain the store | both exit 0. Every difference is listed below; offline and online are identical | PASS |
| A8 regression suites | `checks/run_regression.sh` (every `test_dsv4_*.py` plus the named suites) | all pass except the known failure | 1255 passed, 8 skipped (torch unavailable), 1 failed: `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default` (`OperatorSpec` has no `kernel_count`, unchanged since before B7) | PASS |
| A9 hygiene | line counts; `/data/ycfeng/tmp/unused_imports.py` on the changed files; cold imports of the three modules; removed-name grep over `frontier tests docs examples` restricted to `*.py`; `git diff --check` | no unused imports, imports exit 0 without torch/vllm/cuda, grep finds nothing | line counts as above; no unused imports except sklearn's 5 baseline names; cold imports exit 0 with no native modules; grep exit 1; `git diff --check` clean | PASS (baseline exception above) |
| Whole unit suite, before vs after | `checks/run_unit_pair.sh`, compared test by test with `checks/unit_compare.py`; the after half was rerun after the Git repair with no `GIT_DIR` (`unit_after_rerun.xml`) | only intended changes | First after-run: 134 failed, 6649 passed. One B8-caused regression found and fixed (R4). Rerun: 133 failed, 6650 passed, 133 skipped, 109 errors. Rerun vs first after-run: only the R4 test changed (failure → passed). Rerun vs before: every transition is explained below | PASS |

A7 differences (`evidence/p1h_bridge/package_03/b8_structural_{before,after}_{offline,online}.json`):
- **Real HCA layers:** 13 → 14 attention calls per layer for both q=1 and q=16; `dsv4_compressor_fused_compress_store` 0 → 240 scheduler-rank calls. CSA and uncompressed real layers are unchanged.
- **Idle EP waves:** in the dummy timing of 0.01 ms per call, the all-idle wave gap rises from 0.103529173 to 0.123529173 ms (HCA, 580 waves) and from 0.143529173 to 0.173529173 ms (CSA, 609 waves). The uncompressed gap is unchanged at 0.093529173 ms. These are 2 more HCA calls and 3 more CSA calls, i.e. 9 → 11 and 13 → 16.
- **Stage totals:** +0.2 ms per forward (6.171754453 → 6.371754453 ms and the other stage groups by the same amount).
- **Request metrics:** TTFT 19.76059264 → 20.16059264 ms, TPOT 9.69986421 → 9.89986421 ms, E2E 29.46045685 → 30.06045685 ms, and the throughput values that follow from the longer duration. No other field differs.

Whole-suite transitions, before (`git archive dd4c4317`) vs the rerun (`checks/unit_compare_rerun_vs_before.json`):
- **Test set changes from B8** (expected): new tests pass (`test_dsv4_structural_execution` 16, `test_dsv4_attention_runtime` 13, `test_dsv4_attention_workload` 8, `test_dsv4_graph_measured` 4, `test_dsv4_predictor` 3, `test_dsv4_source_execution` 3); removed tests are absent (`test_dsv4_graph_measured` 103 passed and 75 skipped, `test_dsv4_attention_workload` 10, `test_dsv4_structural_execution` 5, `test_dsv4_source_execution` 4).
- **Fixed by B8:** `test_profiling_runtime_boundary`, failure → passed.
- **Harness artifacts, not B8 behavior:**
  - one `test_dsv4_attention_runtime` setup error: the runner's `-p no:logging` removes the `caplog` fixture; the test passes in A8;
  - `test_native_groundtruth_check`: 19 skipped → error (`No module named 'torch'`) and 2 skipped → passed. The archive lacks an untracked fixture, so the module skips before; on the working tree it runs and needs torch;
  - failures and errors in the before run only, all caused by the exported `GIT_DIR`:
    - failures: `test_native_reader_binding` 29, `test_native_pd_manifest` 1, `test_refactor_fidelity_gate` 1, `test_moe_ep_non_dummy_matrix` 1;
    - setup errors: `test_native_serving_inventory` 5 and `test_refactor_fidelity_gate` 3.

    These tests read Git state. Git therefore returned the polluted worktree directory's state: for example, the fixture HEAD `85ccd7db` instead of the pinned native commit `2cf0a691`. The fixtures' own `git` calls also failed when Git sent them to that directory;
  - `test_dsv4_pilot_cases` failed in the before run because the archive has no ignored `task_memory/` files.

Codex's own diff (`/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/structural_diff.json`) agrees. It also shows two further changes:
- idle lanes no longer price the warmup-only scopes `dsv4_query_padding` and `dsv4_attention_output_zero`;
- `trace_rows` goes from 481144 to 481232.

Lead evidence in `h3o_operator_costs/evidence/p1h_bridge/package_03/`:
- `b8_runtime_table_composition.py`, `.json`, `.log`: A3;
- `b8_structural_summary.py` and `b8_structural_{before,after}_{offline,online}.json`: A7.

## Git state incident (lead error)

- **Cause.** For the "before" half of the whole-suite comparison, `checks/run_unit_pair.sh` exported `GIT_DIR=/data/ycfeng/Frontier/.git/worktrees/glm53-dsv4-vllm028`. The aim was to give the `git archive` snapshot a Git identity. Fixture tests that build temporary repositories then ran `git init`, `git config` and `git commit` against this worktree's Git directory instead of their temporary directories. The tests involved: `test_moe_ep_baseline_replay.py:167`, `test_moe_ep_non_dummy_matrix.py:5615`, `test_native_pd_manifest.py:35` and `test_refactor_fidelity_gate.py:464`.
- **Effects observed** (from 10:43:04 to 10:45:31 UTC):
  - the branch `feat/glm53-dsv4-vllm028` moved from `dd4c4317` to the fixture commit `80aacd8e` ("baseline", parent `dd4c4317`, deletes 4994 files);
  - this worktree's HEAD is detached at the fixture commit `4a107a47` ("initial"), after 29 fixture commits;
  - this worktree's index holds the 9 fixture files;
  - the shared `/data/ycfeng/Frontier/.git/config` now has `core.bare = true` (written by `git init`) and a new `[user]` section `name = t`, `email = t@example.com`.
  - The working-tree files were not affected: the fixtures' work tree was their temporary directory, and the B8 edits are still on disk. The main checkout's own `HEAD` and `index` are unchanged. No other branch, and no remote-tracking ref, was moved.
- **Effect on this review.**
  - A1, A3, A4 and A8 finished before 10:43 and are unaffected.
  - The structural after-run (A7) read source files from disk and matches the expected values exactly.
  - The whole-suite after-run overlapped the window for 91 s. Its rerun after the repair differs only by the R4 test, so the overlap changed no result.
- **Repair.** The permission check first blocked it as a destructive Git action. YC approved it ("批准，继续") and the lead applied it on 2026-10-28; exit 0. The polluted files are backed up in `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b8_review/git_repair/`. The repair restores the earlier state exactly; the reflog keeps the old values:
  1. `git config --file /data/ycfeng/Frontier/.git/config core.bare false`
  2. `git config --file /data/ycfeng/Frontier/.git/config --remove-section user`
  3. `git update-ref refs/heads/feat/glm53-dsv4-vllm028 dd4c4317606789e5cafcba148f6c1955f3b96f7f 80aacd8e282bd1ae2d877caeb794e5718dbf7686`
  4. in this worktree: `git symbolic-ref HEAD refs/heads/feat/glm53-dsv4-vllm028`, then `git read-tree dd4c4317606789e5cafcba148f6c1955f3b96f7f` (index only; working tree untouched).
- **Checks after the repair** (all as expected):
  - `git -C /data/ycfeng/Frontier rev-parse --is-bare-repository` prints `false`, and `git status` works in the main checkout (branch `main`);
  - `git config user.name` / `user.email` resolve to the global `fwyc0573` identity;
  - this worktree is on `feat/glm53-dsv4-vllm028` at `dd4c4317`;
  - `git status --short` lists exactly the B8 changes, the task records and the untracked `.cortexkit/`, `.pi/`, `remaining_measurements/` and `tests/unit/test_dsv4_attention_runtime.py`.

  The whole-suite after-run was then repeated with no `GIT_DIR` (table above).
- **Prevention.** Never export `GIT_DIR` or `GIT_WORK_TREE` around a test suite. Give a before-snapshot its own Git identity with a separate worktree, or accept that its Git-reading tests fail.

## Limits

- A3 is in-sample: an exact lookup trained on the same 505 rows. It shows that the runtime path reproduces B7's composition. Group-held-out accuracy is A6 of the route (stage 1a), still open.
- The structural harness runs in dummy mode with synthetic timing; A7 shows call-count and timing structure, not latency.
- CPU only; no GPU run.
- The simulator does not yet use this path for V4:
  - the shared manager path raises `NotImplementedError` until final admission;
  - the structural and step-cost paths still use `serial_attention_phase` (entries 6-9);
  - entry 10 (V4 gap selection in `kernel_gap.py`) and the kernel-count pairing are deferred;
  - mixed decode/prefill batches raise `DeepSeekV4CoverageError` in trained mode.
- The static warmup-only operators remain in the V4 operator list.
- The whole-suite before half ran on a snapshot without its own Git identity. Its Git-reading tests are therefore not a valid baseline; the rerun-vs-first-after comparison covers them instead.

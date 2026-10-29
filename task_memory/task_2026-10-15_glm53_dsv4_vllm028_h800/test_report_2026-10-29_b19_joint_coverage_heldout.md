# Test report: B19, per-row-group attention admission (A5) and the A6 held-out audit

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: B19 (Codex `70321f85`, outcome partial) review, the lead correction of the collective part, lead checks and the diagnosis of the failing scopes. |

## Scope

Assignment: B19 as submitted with B18 in `3831c459` (A5 admission per measured row group, replacing B7's per-column bounds; the A6 held-out tool and result). Codex record: [h3o_operator_costs/a6/verification.md](h3o_operator_costs/a6/verification.md) (handoff text plus the lead addendum). Working tree `.worktrees/b19-v4-heldout`, branch `b19-v4-heldout`, commit `0dc74fd2`, merged into `feat/glm53-dsv4-vllm028` as `84b26b6d`.

Codex returned "partial": the uniform collective part stopped because the attention unit of the assignment does not exist in the collective table, and the frozen package 03 plan file was absent from its worktree.

## Delivered

| Area | Files | Behavior |
| --- | --- | --- |
| A5 admission | `frontier/execution_time_predictor/dsv4_predictor.py` (357 lines) | Training stores, per fitted V4 attention model, the measured range of every feature and geometry column for each measured `padded_rows` group (`_frontier_dsv4_workload_group_ranges`). `scope_times` first requires a measured row count, then checks every other value against that group's range. Either failure raises `DeepSeekV4CoverageError` naming the operator, model key, group, column, value and range. Dummy groups admit measured row counts only. The global per-column bounds of B7, which let a batch borrow geometry measured at another row count, are removed |
| Predictor tests | `tests/unit/test_dsv4_predictor.py` (327 lines) | Unmeasured row counts between measured sizes are rejected; geometry from another row group is rejected while the own group keeps its value; interpolation stays inside its row group; dummy admission needs a measured row count and no geometry |
| A6 tool | `tests/analysis/v4_attention_heldout.py` (532 lines after the lead edit) | Complete geometry units `(graph_runtime_mode, phase, padded_rows, dispatch_num_requests, prefill_seq_lens, prefill_kv_cache_sizes, decode_kv_cache_sizes)`, sklearn `KFold` (K=5, seed 1234), each fold trained through `ExecutionTimePredictionModelManager._train_single_model` with the P2 config, held-out rows priced through the runtime scope selection, joint against old admission, per-scope and per-key statistics, and the collective curves (lead correction) |
| Tool tests | `tests/unit/test_v4_attention_heldout.py` (163 lines) | Six Codex tests (splitting, equal geometries, rejection accounting, legally zero selection, admitted-row statistics) and one lead test for the collective leave-one-out |
| Results | `h3o_operator_costs/a6/result.json`, `summary.md`, `families.md`, `verification.md` | The audit output, the family classification with the collective table, and the verification record |

## Lead review

- **A5 change.** Checked the training and lookup sides against the assignment: the range metadata is reattached after a cache load and after a fit, so cache names do not change (the consumer suite's warm-cache identity test passes). The exact lookup and the `model.predict` line follow the new check unchanged; B20 edits the same function at that line (the memo), so the B20 merge resolves one local conflict there.
- **Lead correction (collective part).** Codex stopped correctly under the contradiction rule. The lead set the unit (requirements item 36, D-A6-COLLECTIVE-UNIT): one measured payload point of a uniform curve, evaluated with the existing `_Curve.describe` leave-one-out, thresholds per curve. Edit: `collective_heldout` and the collective entry of `family_classification`; the backend is built as `Cluster._initialize_cc_backend` builds it for the P2 DECODE cluster; the `BLOCKED_UNIT_CONTRACT` fields are removed. The backend's `describe_holdouts` compares single-active and unequal-token rows with the uniform curves, a different workload, so it is not used.
- **Frozen plan.** The package 03 plan was ignored by `.gitignore` and existed only in the main worktree. Committed on main (`c4df84e3`).
- **Failing attention scopes.** Diagnosed below; the verdicts stay FAIL and no threshold, feature or table was changed.

## Checks

Environment: Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6 (sklearn 1.7.2, pandas 2.3.3, numpy 2.2.6; no torch, no vLLM); `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`; `TMPDIR=FRONTIER_TMP_ROOT=/data/ycfeng/tmp/lead_b19/tmp`; `ulimit -v 31457280`. Lead scripts and logs: `/data/ycfeng/tmp/lead_b19/` (`env.sh`, `run_tests.sh`, `order_check.sh`, `run_a6.sh`, `run_a6_warm.sh`, `main_checks.sh`, `logs/`). Input table `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv`, SHA-256 `34c4f3a86822994b126d195a54fce934e208fd3ba9dff2cc4fddb560c129321f`, 4,945 rows, 989 units of 5 rows.

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Required suite in the B19 worktree (`run_tests.sh`: predictor, held-out tool, attention coverage, attention runtime, operator consumer, structural PD and execution, source execution) | All pass | 321 passed, 1 failed: `test_package03_plan_json_frozen`, `FileNotFoundError` for the plan file absent from that worktree (`logs/required.log`) | PASS except the missing file (check 7) |
| 2 | Order check in the B19 worktree (`order_check.sh`: held-out tool, predictor, DP dummy pricing, DP sync, features; the B18 files do not exist at its base `3831c459`) | All pass | 117 passed (`logs/order.log`) | PASS |
| 3 | Tool and predictor tests after the lead edit | All pass | 54 passed (7 tool tests including the lead test, 47 predictor tests) | PASS |
| 4 | Cold reproduction of the Codex A6 run (`run_a6.sh`, Codex tool unchanged, own scratch `heldout_cold`) | `result.json` and `summary.md` identical to Codex's | `EXIT 0`, 3,031 s; `RESULT_IDENTICAL`, `SUMMARY_IDENTICAL` (`logs/a6_rerun.log`) | PASS |
| 5 | Regeneration with the corrected tool (`run_a6_warm.sh`, fold caches of check 4) | Only the collective part changes | `EXIT 0`, 529 s; against check 4 only the top-level key `other_families` differs; `attention` identical (`compare_attention.py`). New `result.json` 2,578,014 bytes, SHA-256 `02ec668557f5db166629dbbc04320917e3223f6ec66712a64be63e2d0dedff9d` | PASS |
| 6 | A5 admission on the held-out rows (from `result.json`) | Joint admission rejects at least what the old admission rejects; shared predictions unchanged | 67,244 held-out scope predictions; joint rejects 1,806, old rejects 478; the 65,438 predictions both admit are equal (0 mismatches); 11,876 legally zero checks with 0 mismatches | PASS |
| 7 | Frozen plan test on main after committing the plan (`c4df84e3`) | Passes | 1 passed | PASS |
| 8 | Post-merge required suite on main (`main_checks.sh`, the list of check 1) | All pass | 323 passed in 286 s, rc 0 (`logs/main_checks.log`; 322 of check 1 plus the lead test) | PASS |
| 9 | Post-merge order check on main with the B18 files (`main_checks.sh`: held-out tool, predictor, A10 tools, DP dummy pricing, DP sync, graph dispatch, features) | All pass | 146 passed, rc 0 (`logs/main_checks.log`) | PASS |
| 10 | `git diff --check` and module sizes | Clean; at or below 2,000 lines | Clean; 357, 327, 532, 163 | PASS |

## A6 results

Kernel error is `(predicted - measured) / measured` on the held-out rows that joint admission accepts; the gate is median absolute error at or below 5% and p90 at or below 10%.

Attention: 16 scopes PASS, 3 FAIL, 2 LEGALLY_ZERO, 1 NOT_EVALUABLE (`index_short_context_fill`: its 2 held-out rows are rejected). Per model key: 103 PASS, 17 FAIL, 157 NOT_EVALUABLE (dummy keys are not evaluable by construction because a held-out dummy unit removes its row group; prefill and NONE keys have one measured workload).

| Failing scope | Evaluable | Exact lookups | Median | p90 | Signed median | Share of FULL decode attention kernel time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `index_fused_query_prepare` | 1,930 | 1,930 | 4.593% | 12.207% | 0.40% | 0.95% |
| `compressor_fused_compress_store` | 3,846 | 3,696 | 2.968% | 11.518% | 0.00% | 5.73% |
| `index_compressor_fused_compress_store` | 1,926 | 1,838 | 3.159% | 17.476% | 0.44% | 2.28% |

Diagnosis (lead, `/data/ycfeng/tmp/lead_b19/variance.py`, `boundary2.py`, `repeat_noise.py`, `repeat_noise2.py`):

- **Observation.** 95-100% of the held-out predictions of these scopes are exact lookups: the same feature tuple is already in the training fold, so the prediction is the training measurement and the error is the spread between measurements with equal features.
- **Observation.** One forward measures each variant in two layer groups (`plans_tile_schedule` true and false) with equal geometry. For these kernels (2-20 µs) the two groups deviate from their median by a p90 of 7.5% (`index_fused_query_prepare`), 7.6% (`compressor_fused_compress_store`) and 17.9% (`index_compressor_fused_compress_store`).
- **Observation.** The compressor store time has two levels: about 2-3 µs when no row completes a compression group, about 19-20 µs when one or more do, nearly independent of how many. The existing feature `num_compressed_writes` (`(kv + 1) // ratio - kv // ratio` summed over rows) separates the two levels.
- **Inference.** The failures sit at the measurement spread of these small kernels, not at a missing feature or a fit error. The signed medians (0.0-0.4%) show no shift of the typical row, so the errors largely cancel when summed over layers and steps; their effect on E2E error is not measured here.

Collectives (D-A6-COLLECTIVE-UNIT, P2 DECODE backend, `data/profiling/network/h200_dgx/summary.csv`, SHA-256 `f4d52911770cc39939dddc0fa98ce30b530b1cd60ea2bd94070d293e200ca559`):

| Curve | Points | Held out | Median | p90 | Max | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| all_gatherv, ep, 16 | 5 | 3 | 11.404% | 12.091% | 12.263% | FAIL |
| all_gatherv, ep, 32 | 5 | 3 | 9.012% | 11.218% | 11.769% | FAIL |
| all_reduce, tp, 8 | 8 | 6 | 8.828% | 18.793% | 20.233% | FAIL |
| logits_all_gather, tp, 8 | 24 | 22 | 5.396% | 9.022% | 12.869% | FAIL |
| reduce_scatterv, ep, 16 | 5 | 3 | 0.640% | 6.920% | 8.490% | PASS |
| reduce_scatterv, ep, 32 | 5 | 3 | 30.644% | 39.754% | 42.032% | FAIL |
| sp_all_gather, tp, 8 | 9 | 7 | 3.527% | 22.978% | 26.420% | FAIL |
| sp_reduce_scatter, tp, 8 | 8 | 6 | 5.251% | 8.929% | 9.644% | FAIL |

A left-out point is interpolated across two measured intervals (about 28 times in bytes for the five-point curves), where the runtime interpolates across one, so these errors overstate the runtime error. Which payloads the A10 runs interpolate is recorded by B21 (decision 1b, resubmitted as Codex `fcf9eca4`).

## Limits

- A6 measures the attention table and the uniform collective curves only. The standalone operator, MoE, runner, gap and unequal-combine families are exact lookups covered by A5.
- The held-out errors are per scope and per row; their effect on step and E2E time is read in A10 and A11, not here.
- The stricter A5 can reject A10 replay coordinates that the per-column bounds accepted. The runtime then raises `DeepSeekV4CoverageError`; that would be a measurement gap to report, not a reason to widen admission.
- The dummy smoke matrix and the 77-case fidelity matrix were not rerun: B19 changes only `dsv4_predictor.py`, which only V4 models reach, and neither matrix contains a V4 case (`tests/e2e/refactor_fidelity/cases.py` names no DeepSeek model).
- The known DP placement failure (`test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`) and the KV admission difference (2,940 against 2,944 blocks) are outside B19 and unchanged.

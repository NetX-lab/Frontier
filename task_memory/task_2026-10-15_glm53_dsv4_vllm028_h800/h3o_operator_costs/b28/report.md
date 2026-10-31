B28 / I-008 implements isolated H200 eager EP timings and exact measured pricing for uneven V4/GLM dispatch. The implementation and targeted checks are complete. The assignment outcome is **partial**: the required regression cannot finish collection because `aiohttp` is unavailable, and the supplemental pair retains existing failures and nonidentical diagnostic messages. Final acceptance and commits remain with Claude.

Assignment and scope: working root `ROOT=/data/ycfeng/Frontier/.worktrees/b28-eager-ep-exact-rows`; read-only base `BASE=/data/ycfeng/Frontier/.worktrees/b28-base`. Both started at `833c7e3145b1bafd33928d850266ebeb8fa160b2`. No pre-existing tracked changes were encountered. Records are under `B=ROOT/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b28`; scratch is `S=/data/ycfeng/tmp/codex-implement/d684ea95-b7ff-4a6d-8fec-63b038641a41/b28`. Below, these aliases expand to those absolute paths.

D1–D5 are preserved: replace only eager EP rows containing a 6144-token lane; select exact dispatch/combine evidence by collective alias/domain/implementation; reject missing uneven vectors or bytes without fallback; retain equal-SP nonuniform measurements as holdouts; and share one routing rule with A10. Geometry uses the existing SP padding and `ep_workload` producer. Applicable AGENTS instructions and the named issue, revised G5h decision, stage metadata and expected-price artifacts were read. W7b was absent in this tree and was read from the named canonical main task tree. No behavior outside the agreed change, Git mutation, install, network/GPU operation or A10 replay operation was performed.

Every changed tracked file:

| Path relative to ROOT | Change and reason |
| --- | --- |
| `frontier/cc_backend/backends/measured_cc_backend.py` | `_ExactSPCall`/`_EXACT_SP_CALLS` (54–71) hold one evidence entry per call. `_load_curves` (150), `_retained_sp_rows` (214), `_exact_sp_times` and `_predict_exact_sp` generalize combine-only loading/lookup. `uses_exact_sp_rows` (270) owns routing and is called by `predict_operator_collective` (275). Errors/docstring describe both calls; combine validation remains intact. |
| `frontier/operators/sequence_parallel_moe.py` | `ep_sp_rows` (151) replaces `combine_sp_rows`, supplying ordered widths to both SP EP calls and retaining `None` for ordinary all-to-all. |
| `frontier/operators/families.py` | `_ep_sp_rows` (526) replaces `_combine_sp_rows`; GLM dispatch/combine (715–743) and V4 EP specs (792–820) share the builder. Alias/payload semantics are unchanged. |
| `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_observer.py` | `collective_request` (73–82) uses the backend predicate and `_exact_sp_times`. Fields/bracket meaning are unchanged; uneven dispatch now reports exact evidence. |
| `data/profiling/network/h200_dgx/summary.csv` | Replaced 108 isolated lines: nine EP16 dispatch, nine EP16 combine, 45 EP32 dispatch and 45 EP32 combine. Header, order, CRLF, 384 rows and the other 276 lines are preserved byte-for-byte. |
| `data/profiling/network/h200_dgx/summary_provenance.json` | Updated table hash/rule and added the specified isolated sources/stage metadata with `replaced_rows`. Both original source objects and `output_rows` remain identical. |
| `docs/general/measured-communication.md` | Only the unequal-SP bullet (78–85) changes to the exact dispatch/combine rule, typed missing coverage and equal-SP holdouts. |
| `tests/unit/test_glm_sp_moe_comm.py` | Replaces the uniform dispatch test with distinguishable H800 exact evidence; extends equal-SP and scalar-backend checks. |
| `tests/unit/test_measured_cc_backend.py` | Generalizes row helper/names/census, preserves combine assertions and adds dispatch kernel/input/output rejection. |
| `tests/unit/test_v4_measured_collective_consumer.py` | Extends grids/renamed specs to dispatch; adds missing shape/size, decision-price, producer geometry and coverage checks; updates H200 census. |
| `tests/unit/test_v4_replay_census.py` | Observer fixture shares the backend predicate and verifies exact brackets for both EP calls. |

New B records: `build_table.py`, `table_check.json`, `build_table.txt`, `price_check.py`, `price_check.txt`, `curve_diff.py`, `curve_diff.txt`, `run_checks.py`, `compare_regression.py`, `regression_targets.json`, `cache_audit.md`, `changed_test_targets.json`, `changed_test_results.json`, `changed_test_lines.md`, `scope_check.json`, `diff_stat.txt`, the named command receipts below, `focused_comparison.json`, `regression_comparison.json` and `report.md`. Existing task-memory Git rules ignore B records, so they do not appear in tracked diff stat; staging is Claude's responsibility.

Table check: **PASS**. `build_table.py` verifies the three prescribed SHA-256 inputs before writing, 108 selected rows, unique `(group_size, pass, scenario, call)` matches and every prescribed geometry/kernel column. `table_check.json` records all 108 0-based data-row indices, keys and old/new medians. Builder output:

```json
{
  "input_sha256": {
    "base": "f4d52911770cc39939dddc0fa98ce30b530b1cd60ea2bd94070d293e200ca559",
    "16": "e26e697a948c8f1dcb9a5190cd828450219697b254c4d1205d163e47a5719886",
    "32": "8ed9d86330cf7c462867efc23e83a7e2d23bc7d8eb7a1a8e50b9150c1bc907a0"
  },
  "output_sha256": "1a7c701f2701aa94440f091fb6c5d03477b61f81ad7d43924f0a51008f3d4c65",
  "rows": 384,
  "replaced_rows": 108,
  "counts": [
    {
      "group_size": 16,
      "call": "moe_combine_reduce_scatterv",
      "rows": 9
    },
    {
      "group_size": 16,
      "call": "moe_dispatch_all_gatherv",
      "rows": 9
    },
    {
      "group_size": 32,
      "call": "moe_combine_reduce_scatterv",
      "rows": 45
    },
    {
      "group_size": 32,
      "call": "moe_dispatch_all_gatherv",
      "rows": 45
    }
  ]
}
```

Price check: **PASS**. Separate base/candidate interpreters use real V4 specs, contexts and producer payloads. Candidate prices equal unrounded medians of three isolated passes and the decision values rounded to 0.1 µs. Output of `price_check.py`, saved in `price_check.txt`:

```text
base: /data/ycfeng/Frontier/.worktrees/b28-base
P2-32 EP16 dispatch: 801.7 us, input 6346752 bytes, SP rows (768, 1); isolated median 560.2 us
P2-32 EP16 combine: 763.9 us, input 50397184 bytes, SP rows (768, 1); isolated median 734.6 us
P2-32 43 layers, planned change from base: dispatch -10.38 ms, combine -1.26 ms, total -11.64 ms
P2-64 EP32 dispatch: 1076.0 us, input 6346752 bytes, SP rows (768, 1, 1, 1); isolated median 1043.0 us
P2-64 EP32 combine: 2011.1 us, input 50528256 bytes, SP rows (768, 1, 1, 1); isolated median 2172.0 us
P2-64 43 layers, planned change from base: dispatch -1.42 ms, combine +6.92 ms, total +5.50 ms
PASS: operator prices and forward changes match B28 decisions

candidate: /data/ycfeng/Frontier/.worktrees/b28-eager-ep-exact-rows
P2-32 EP16 dispatch: 560.2 us, input 6346752 bytes, SP rows (768, 1); isolated median 560.2 us
P2-32 EP16 combine: 734.6 us, input 50397184 bytes, SP rows (768, 1); isolated median 734.6 us
P2-32 43 layers, actual change from base: dispatch -10.38 ms, combine -1.26 ms, total -11.64 ms
P2-64 EP32 dispatch: 1043.0 us, input 6346752 bytes, SP rows (768, 1, 1, 1); isolated median 1043.0 us
P2-64 EP32 combine: 2172.0 us, input 50528256 bytes, SP rows (768, 1, 1, 1); isolated median 2172.0 us
P2-64 43 layers, actual change from base: dispatch -1.42 ms, combine +6.92 ms, total +5.50 ms
PASS: operator prices and forward changes match B28 decisions
```

Combine retains its BF16 Reduce kernel proof, gathered input/largest-local output and BF16-multiple row width. Dispatch requires AllGatherV, largest-local input/gathered output and integer row width. H200 `(6144,1,1,1)` dispatch input is 768 × 8264 = 6,346,752 bytes and output is 6168 × 8264 = 50,972,352. Tests check producer geometry for all 96 H200 and 60 H800 retained raw rows. Bad dispatch kernel/input/output raises `ValueError` at load.

Curve/coverage check: **PASS**. H200's only uniform changes are four 6144-token EP points; its 16 existing combine entries change only from replaced scenarios. Forty-eight dispatch holdouts (six EP16, 42 EP32) become 16 pooled exact entries: total exact 16→32, holdouts 48→0. H800 curves/combine stay identical; 30 dispatch rows become nine exact entries: total exact 9→18, holdouts 282→252. Nine equal-SP dispatch holdouts remain. No combine-only or dispatch-only vector exists on either table; therefore no measured H800 combine vector loses dispatch coverage. Other missing uneven vectors/sizes intentionally raise without fallback. Every difference and ordered vector follows, from `curve_diff.txt`:

```text
h200_dgx
  curve ('all_gatherv', 'ep', 16) bytes=6346752: 801.660776138 -> 567.487984896 us
  curve ('all_gatherv', 'ep', 32) bytes=6346752: 1076.022386551 -> 1138.767957687 us
  curve ('reduce_scatterv', 'ep', 16) bytes=100663296: 768.513607979 -> 454.272001982 us
  curve ('reduce_scatterv', 'ep', 32) bytes=201326592: 865.223979950 -> 895.792007446 us
  exact (('reduce_scatterv', 'ep', 16), (1, 768), 50397184): 798.739218712 -> 728.287994862 us
  exact (('reduce_scatterv', 'ep', 16), (768, 1), 50397184): 763.857579231 -> 734.575986862 us
  exact (('reduce_scatterv', 'ep', 32), (1, 1, 1, 768), 50528256): 2008.993625641 -> 2145.391941071 us
  exact (('reduce_scatterv', 'ep', 32), (1, 1, 768, 1), 50528256): 2066.531181335 -> 2196.176052094 us
  exact (('reduce_scatterv', 'ep', 32), (1, 1, 768, 768), 100794368): 1789.123153687 -> 1857.855975628 us
  exact (('reduce_scatterv', 'ep', 32), (1, 768, 1, 1), 50528256): 2077.182388306 -> 2176.752090454 us
  exact (('reduce_scatterv', 'ep', 32), (1, 768, 1, 768), 100794368): 1852.000045776 -> 1916.992008686 us
  exact (('reduce_scatterv', 'ep', 32), (1, 768, 768, 1), 100794368): 1849.689579010 -> 1898.447990417 us
  exact (('reduce_scatterv', 'ep', 32), (1, 768, 768, 768), 151060480): 2061.390399933 -> 2106.559991837 us
  exact (('reduce_scatterv', 'ep', 32), (768, 1, 1, 1), 50528256): 2011.121559143 -> 2171.999931335 us
  exact (('reduce_scatterv', 'ep', 32), (768, 1, 1, 768), 100794368): 1780.782318115 -> 1922.335982323 us
  exact (('reduce_scatterv', 'ep', 32), (768, 1, 768, 1), 100794368): 1807.839965820 -> 1918.832004070 us
  exact (('reduce_scatterv', 'ep', 32), (768, 1, 768, 768), 151060480): 2067.867183685 -> 2292.079925537 us
  exact (('reduce_scatterv', 'ep', 32), (768, 768, 1, 1), 100794368): 1787.827205658 -> 1871.951997280 us
  exact (('reduce_scatterv', 'ep', 32), (768, 768, 1, 768), 151060480): 2091.500854492 -> 2298.143982887 us
  exact (('reduce_scatterv', 'ep', 32), (768, 768, 768, 1), 151060480): 2052.043247223 -> 2145.328044891 us
  new exact (('all_gatherv', 'ep', 16), (1, 768), 6346752): 568.015992641 us
  new exact (('all_gatherv', 'ep', 16), (768, 1), 6346752): 560.192018747 us
  new exact (('all_gatherv', 'ep', 32), (1, 1, 1, 768), 6346752): 1060.063958168 us
  new exact (('all_gatherv', 'ep', 32), (1, 1, 768, 1), 6346752): 1048.352003098 us
  new exact (('all_gatherv', 'ep', 32), (1, 1, 768, 768), 6346752): 1096.656024456 us
  new exact (('all_gatherv', 'ep', 32), (1, 768, 1, 1), 6346752): 1058.127999306 us
  new exact (('all_gatherv', 'ep', 32), (1, 768, 1, 768), 6346752): 1085.168004036 us
  new exact (('all_gatherv', 'ep', 32), (1, 768, 768, 1), 6346752): 1099.840044975 us
  new exact (('all_gatherv', 'ep', 32), (1, 768, 768, 768), 6346752): 1126.464009285 us
  new exact (('all_gatherv', 'ep', 32), (768, 1, 1, 1), 6346752): 1043.007969856 us
  new exact (('all_gatherv', 'ep', 32), (768, 1, 1, 768), 6346752): 1091.152012348 us
  new exact (('all_gatherv', 'ep', 32), (768, 1, 768, 1), 6346752): 1083.680033684 us
  new exact (('all_gatherv', 'ep', 32), (768, 1, 768, 768), 6346752): 1134.383976460 us
  new exact (('all_gatherv', 'ep', 32), (768, 768, 1, 1), 6346752): 1087.199985981 us
  new exact (('all_gatherv', 'ep', 32), (768, 768, 1, 768), 6346752): 1133.791983128 us
  new exact (('all_gatherv', 'ep', 32), (768, 768, 768, 1), 6346752): 1137.392044067 us
  holdouts ('all_gatherv', 'ep', 16): 6 -> 0
  holdouts ('all_gatherv', 'ep', 32): 42 -> 0
  totals: curve changes=4, combine changes=16, new dispatch entries=16, exact entries=16 -> 32, holdouts=48 -> 0
  SP-vector coverage: dispatch=16, combine=16, combine-only=[], dispatch-only=[]
  ordered raw-lane coverage: dispatch=16, combine=16, combine-only=[], dispatch-only=[]
  covered ordered lane vectors: [(16, (1, 6144)), (16, (6144, 1)), (32, (1, 1, 1, 6144)), (32, (1, 1, 6144, 1)), (32, (1, 1, 6144, 6144)), (32, (1, 6144, 1, 1)), (32, (1, 6144, 1, 6144)), (32, (1, 6144, 6144, 1)), (32, (1, 6144, 6144, 6144)), (32, (6144, 1, 1, 1)), (32, (6144, 1, 1, 6144)), (32, (6144, 1, 6144, 1)), (32, (6144, 1, 6144, 6144)), (32, (6144, 6144, 1, 1)), (32, (6144, 6144, 1, 6144)), (32, (6144, 6144, 6144, 1))]
h800_dgx
  new exact (('all_gatherv', 'ep', 32), (4, 1, 1, 1), 49408): 334.529602528 us
  new exact (('all_gatherv', 'ep', 32), (8, 1, 1, 1), 98816): 336.737596989 us
  new exact (('all_gatherv', 'ep', 32), (16, 1, 1, 1), 197632): 336.091196537 us
  new exact (('all_gatherv', 'ep', 32), (32, 1, 1, 1), 395264): 343.172794580 us
  new exact (('all_gatherv', 'ep', 32), (64, 1, 1, 1), 790528): 364.342403412 us
  new exact (('all_gatherv', 'ep', 32), (128, 1, 1, 1), 1581056): 402.873611450 us
  new exact (('all_gatherv', 'ep', 32), (256, 1, 1, 1), 3162112): 496.985602379 us
  new exact (('all_gatherv', 'ep', 32), (512, 1, 1, 1), 6324224): 1242.256021500 us
  new exact (('all_gatherv', 'ep', 32), (1024, 1, 1, 1), 12648448): 2188.248062134 us
  holdouts ('all_gatherv', 'ep', 32): 39 -> 9
  totals: curve changes=0, combine changes=0, new dispatch entries=9, exact entries=9 -> 18, holdouts=282 -> 252
  SP-vector coverage: dispatch=9, combine=9, combine-only=[], dispatch-only=[]
  ordered raw-lane coverage: dispatch=9, combine=9, combine-only=[], dispatch-only=[]
  covered ordered lane vectors: [(32, (32, 1, 1, 1)), (32, (64, 1, 1, 1)), (32, (128, 1, 1, 1)), (32, (256, 1, 1, 1)), (32, (512, 1, 1, 1)), (32, (1024, 1, 1, 1)), (32, (2048, 1, 1, 1)), (32, (4096, 1, 1, 1)), (32, (8192, 1, 1, 1))]
PASS: only the planned curve, exact-entry and holdout differences; coverage is equal
```

Cache audit: no out-of-scope fix is needed. The evidence is source inspection, not an A10/GPU replay; `cache_audit.md` provides the complete references.

| Cache/memo | Evidence and conclusion |
| --- | --- |
| V4 `_prediction_memo`, `dsv4_predictor.py:324,357–366` | Stores attention scope cost/gap by model/features, never EP or composed forward time. No dispatch invalidation needed. |
| B24 `phase_memo`, `scheduler/utils/expert_parallel.py:138,164–183` | Holds dispatch-containing phase tuples, but is created per wave. SP widths, predictor/config, source batches and layer stay fixed throughout its lifetime. Missing SP rows in its key cannot reuse a price across vectors. |
| A10 `_collectives`, `a10_observer.py:272–283` | Aggregation key already includes role, curve, bytes and unequal rows. The shared predicate now includes dispatch rows. Other maps use batch/epoch/event identity and do not cache across forwards. |
| Fitted/runtime compute caches, `dsv4_training.py`, `prediction_model_registry.py`, `sklearn_execution_time_predictor.py:6312–6320` | Measured communication returns directly before compute-model lookup; loading the backend reloads the CSV. No compute cache invalidation required. |
| Routing caches, `moe_routing_workload.py:400–445`; stage-local attention query cache | Store descriptors or attention predictions, not dispatch/composed forward time. SP widths are attached independently per wave. |
| `StageExecutionTime._model_time_ms_cache`, `stage_execution_time.py:135,144–174,340–352` | Belongs to an already finalized stage instance, not a global shape cache. |
| Diagnostic step-cost table, `dsv4_step_cost_input.py:31–58` | Separate opt-in native-step input bypasses measured CC; A10 rejects this path. It does not cache a forward assembled from these calls. |
| Backend/non-measured caches | Exact keys contain curve, ordered SP rows and bytes; state is instance-local. Non-measured backends retain scalar alias/bytes/domain/implementation results and ignore SP rows. |

Observed cache ownership/lifetimes support the inference that no inspected serving path can return a stale cross-vector dispatch price. No persistent caches or running A10 processes were modified.

Verification used `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `ulimit -v 31457280`, bounded commands, disabled bytecode and scratch outputs. Executed build/price/curve program arguments (with the interpreter above):

```text
B/build_table.py
B/price_check.py BASE base --json-output S/base_prices.json
B/price_check.py ROOT candidate --json-output S/candidate_prices.json --baseline S/base_prices.json
B/curve_diff.py
```

Price invocations ran from `S/base` and `S/candidate`; the successful curve check used scratch cwd. An initial price invocation without a `data` symlink failed with unknown-model/no-JSON; the established scratch directories/data symlinks resolved that harness error. No pricing fallback was used.

Regression selection is 165 matching unit modules plus 27 additional B24 selectors, totaling 192 in `regression_targets.json`. The specified strings are `measured|MeasuredCC|families|sequence_parallel_moe|sp_rows|get_comm_operator|dsv4|v4_|glm|a10|h200_dgx|h800_dgx`. Six redundant B24 node selectors are already covered by their selected whole modules; an independent audit verified complete coverage. No selected module was omitted.

These commands expand to the executed runner argv from ROOT; all aliases have their absolute definitions above:

```bash
ulimit -v 31457280
timeout 3700 /data/ycfeng/envs/frontier-py310/bin/python "$B/run_checks.py" "$BASE" base
timeout 3700 /data/ycfeng/envs/frontier-py310/bin/python "$B/run_checks.py" "$ROOT" candidate
timeout 3700 /data/ycfeng/envs/frontier-py310/bin/python "$B/run_checks.py" "$BASE" base_continued --continue-collection
timeout 3700 /data/ycfeng/envs/frontier-py310/bin/python "$B/run_checks.py" "$ROOT" candidate_continued --continue-collection
timeout 60 /data/ycfeng/envs/frontier-py310/bin/python "$B/compare_regression.py" base_continued candidate_continued
```

The runner uses `S/<label>` cwd, a data symlink to the tested root, absolute test paths, `timeout 3600`, `-q -p no:cacheprovider`, `--basetemp=S/tmp/<label>` and `--junitxml=S/<label>.xml`. Environment: `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, `VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0`, `TMPDIR=FRONTIER_TMP_ROOT=S/tmp`, and `PYTHONPATH=<tested root>:<tested root>/tests/unit:<tested root>/tests/comparison/calibration:<tested root>/tests/analysis:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages`. The same command/environment is used for each root except root/label paths. Supplemental runs add only `--continue-on-collection-errors`.

Fully expanded commands, environment, cwd, limits and exit status are in `base_command.json`, `candidate_command.json`, `base_continued_command.json`, `candidate_continued_command.json`, `focused_base_command.json`, `focused_final_command.json` and `changed_final_command.json`. Logs/XML are `S/<label>.log` and `S/<label>.xml`. The final 20-function rerun uses the selectors in `changed_test_targets.json` and label `changed_final`.

| Executed check | Observed result | Status |
| --- | --- | --- |
| Table/byte/hash/provenance | 384 rows, 108 replacements, 276 unchanged; all assertions pass | PASS |
| Base/candidate operator prices | All eight values and both 43-layer changes match | PASS |
| Curve/exact/holdout/coverage | Only planned differences; equal coverage | PASS |
| Final changed/new selection | 20 functions, 66 cases pass in 3.33 s, exit 0 | PASS |
| Focused base/candidate | 305/320 passed; same two failures | FAIL, baseline; no new failure |
| Required full base/candidate | Same one collection error, exit 2, 24.02/21.07 s | FAIL, unavailable prerequisite |
| Supplemental base | 4980 passed, 75 failed, 55 skipped, 87 errors, five subtests passed, 16 warnings; exit 1, 1028.83 s | FAIL, baseline |
| Supplemental candidate | 4995 passed; same 75 failed, 55 skipped, 87 errors, five subtests passed, 16 warnings; exit 1, 1037.98 s | FAIL, matches baseline outcomes |
| Node comparison | Zero common outcome changes or candidate-only failures; added/removed nodes pass | PASS outcome parity; strict messages unresolved |
| Scope/syntax/size/CRLF diff | Eleven approved tracked files; all changed/new Python parses and is ≤2000 lines | PASS |

Required collection fails at `test_native_cold_start.py:14` → `test_native_pd_canonical_http.py:14` → unavailable `aiohttp`. Supplemental runs expose collectable outcomes and do not replace the required command. Existing skips remain 55 per side; no skip or xfail was introduced.

All changed/new functions passed on final source:

| Test under tests/unit | Cases |
| --- | ---: |
| `test_glm_sp_moe_comm.py::test_equal_sp_rows_with_unequal_raw_tokens_are_covered` | 4 |
| `test_glm_sp_moe_comm.py::test_unequal_dispatch_uses_its_exact_measured_row` | 1 |
| `test_glm_sp_moe_comm.py::test_unequal_ep_other_backends_keep_scalar_prediction` | 2 |
| `test_measured_cc_backend.py::test_checked_in_unequal_ep_samples_match_producer_medians` | 1 |
| `test_measured_cc_backend.py::test_equal_sp_holdout_stays_out_of_exact_and_uniform_samples` | 2 |
| `test_measured_cc_backend.py::test_invalid_retained_unequal_combine_kernel_or_shape_is_rejected` | 11 |
| `test_measured_cc_backend.py::test_invalid_retained_unequal_dispatch_kernel_or_shape_is_rejected` | 3 |
| `test_measured_cc_backend.py::test_unequal_combine_does_not_interpolate_between_nonmonotone_points` | 1 |
| `test_measured_cc_backend.py::test_unequal_combine_exact_match_pools_samples_without_changing_uniform_curve` | 2 |
| `test_measured_cc_backend.py::test_unequal_combine_requires_exact_ordered_rows_bytes_and_group` | 5 |
| `test_measured_cc_backend.py::test_unequal_rows_outside_the_ep_combine_curve_use_no_exact_entry` | 1 |
| `test_measured_cc_backend.py::test_uneven_dispatch_uses_exact_ordered_rows_and_caller_supplied_bytes` | 1 |
| `test_v4_measured_collective_consumer.py::test_exact_dispatch_and_combine_vectors_have_identical_coverage` | 2 |
| `test_v4_measured_collective_consumer.py::test_exact_ep_collective_is_spec_driven_with_renamed_operator` | 4 |
| `test_v4_measured_collective_consumer.py::test_grouped_dispatch_and_combine_match_ordered_eager_grid` | 18 |
| `test_v4_measured_collective_consumer.py::test_h200_eager_ep_prices_equal_isolated_decision` | 2 |
| `test_v4_measured_collective_consumer.py::test_h200_row_classes_are_separate` | 1 |
| `test_v4_measured_collective_consumer.py::test_unmeasured_combine_size_or_ordered_vector_is_typed` | 2 |
| `test_v4_measured_collective_consumer.py::test_unmeasured_dispatch_size_or_ordered_vector_is_typed` | 2 |
| `test_v4_replay_census.py::test_collective_brackets_exact_unequal_and_shares` | 1 |

Two final strengthening edits (preserved combine kernel proof and expanded producer geometry) followed broad candidate collection; all affected tests passed in the 66-case final rerun. Production code stayed unchanged. An initial new scalar-fixture assertion incorrectly used combine's divisor 5000 for dispatch; it was corrected to the fixture's existing dispatch divisor 3000, preserving combine's assertion. The two focused baseline failures were left intact: `test_v4_wave_passes_widths_without_mutating_source_batches` asserts six observations equal 32; `test_observer_snapshots_final_descriptor_and_records_end_once` lacks `_formed` state. No assertion was relaxed to hide them.

Every changed test line and reason follows; ranges are inclusive and include blanks. `changed_test_lines.md` also preserves each literal added/removed line.

Each line number or inclusive range below identifies an added or removed test line. The reason applies to every listed line, including blanks. The full literal line audit is in `changed_test_lines.md`.

| File | Base removed lines | Candidate added lines | Reason |
| --- | --- | --- | --- |
| `tests/unit/test_glm_sp_moe_comm.py` | 100, 104–105 | 100–101, 105–106 | Check uniform routing for dispatch and combine; rename the exact-path fail hook. |
| `tests/unit/test_glm_sp_moe_comm.py` | 134, 138, 140–142 | 135–138, 143–151 | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| `tests/unit/test_glm_sp_moe_comm.py` | 145, 148–150 | — | Extend the unchanged scalar-backend contract to dispatch; preserve the combine assertion. |
| `tests/unit/test_glm_sp_moe_comm.py` | — | 154–156, 159–162 | Extend the unchanged scalar-backend contract to dispatch using the fixture actual alias-specific divisors. |
| `tests/unit/test_measured_cc_backend.py` | 61 | 61, 69–72 | Rename the synthetic row helper and add dispatch evidence with its reversed input/output geometry. |
| `tests/unit/test_measured_cc_backend.py` | 156–168 | 160–171 | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| `tests/unit/test_measured_cc_backend.py` | 175, 191, 199, 235, 238 | 178, 194, 202, 261, 264 | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| `tests/unit/test_measured_cc_backend.py` | 216, 218 | 219, 221 | Use the shared row helper and call-neutral validation error; preserve the full combine rejection matrix. |
| `tests/unit/test_measured_cc_backend.py` | 222–223, 229, 231 | 237–239, 243–245, 248–249, 251–257 | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| `tests/unit/test_measured_cc_backend.py` | — | 225–234 | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| `tests/unit/test_measured_cc_backend.py` | — | 235–236 | Spacing supporting the adjacent test change. |
| `tests/unit/test_measured_cc_backend.py` | 426, 436, 439–440, 442, 444–452 | — | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| `tests/unit/test_measured_cc_backend.py` | — | 452, 455–458, 463–465, 469, 472–473, 475–478, 480–497 | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| `tests/unit/test_v4_measured_collective_consumer.py` | 130–132 | 130–139 | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| `tests/unit/test_v4_measured_collective_consumer.py` | 145, 147, 149, 154–155 | 150, 152, 154, 156, 158, 163–165 | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| `tests/unit/test_v4_measured_collective_consumer.py` | 164 | 174 | Use the call-neutral exact-measurement error text; keep the combine typed-key assertion. |
| `tests/unit/test_v4_measured_collective_consumer.py` | — | 175–176, 185–186, 211–212 | Spacing supporting the adjacent test change. |
| `tests/unit/test_v4_measured_collective_consumer.py` | — | 177–184 | Exercise V4 operator/context rejection for both an absent byte size and an absent SP vector. |
| `tests/unit/test_v4_measured_collective_consumer.py` | — | 187–210 | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| `tests/unit/test_v4_measured_collective_consumer.py` | — | 213–248 | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| `tests/unit/test_v4_measured_collective_consumer.py` | 296–299 | 380–383 | Require 32 entries (16 per call) and zero H200 holdouts after dispatch retention. |
| `tests/unit/test_v4_replay_census.py` | 90, 104–107 | 90, 93, 105–112 | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |


Node differences are exactly 24 added and nine removed passing cases, representing the intended extensions/replacements above. No common ID changes outcome. Exact added IDs:

```text
tests/unit/test_glm_sp_moe_comm.py::test_equal_sp_rows_with_unequal_raw_tokens_are_covered[expert_parallel_alltoall_combine-decode]
tests/unit/test_glm_sp_moe_comm.py::test_equal_sp_rows_with_unequal_raw_tokens_are_covered[expert_parallel_alltoall_combine-prefill]
tests/unit/test_glm_sp_moe_comm.py::test_equal_sp_rows_with_unequal_raw_tokens_are_covered[expert_parallel_alltoall_dispatch-decode]
tests/unit/test_glm_sp_moe_comm.py::test_equal_sp_rows_with_unequal_raw_tokens_are_covered[expert_parallel_alltoall_dispatch-prefill]
tests/unit/test_glm_sp_moe_comm.py::test_unequal_dispatch_uses_its_exact_measured_row
tests/unit/test_glm_sp_moe_comm.py::test_unequal_ep_other_backends_keep_scalar_prediction[expert_parallel_alltoall_combine-reduce_scatter-3440640]
tests/unit/test_glm_sp_moe_comm.py::test_unequal_ep_other_backends_keep_scalar_prediction[expert_parallel_alltoall_dispatch-allgather-395264]
tests/unit/test_measured_cc_backend.py::test_checked_in_unequal_ep_samples_match_producer_medians
tests/unit/test_measured_cc_backend.py::test_equal_sp_holdout_stays_out_of_exact_and_uniform_samples[False]
tests/unit/test_measured_cc_backend.py::test_equal_sp_holdout_stays_out_of_exact_and_uniform_samples[True]
tests/unit/test_measured_cc_backend.py::test_invalid_retained_unequal_dispatch_kernel_or_shape_is_rejected[change0]
tests/unit/test_measured_cc_backend.py::test_invalid_retained_unequal_dispatch_kernel_or_shape_is_rejected[change1]
tests/unit/test_measured_cc_backend.py::test_invalid_retained_unequal_dispatch_kernel_or_shape_is_rejected[change2]
tests/unit/test_measured_cc_backend.py::test_uneven_dispatch_uses_exact_ordered_rows_and_caller_supplied_bytes
tests/unit/test_v4_measured_collective_consumer.py::test_exact_dispatch_and_combine_vectors_have_identical_coverage[h200_dgx]
tests/unit/test_v4_measured_collective_consumer.py::test_exact_dispatch_and_combine_vectors_have_identical_coverage[h800_dgx]
tests/unit/test_v4_measured_collective_consumer.py::test_exact_ep_collective_is_spec_driven_with_renamed_operator[deepseek-ai/DeepSeek-V4-Flash-lanes0-dsv4_ep_output_reduce_scatter]
tests/unit/test_v4_measured_collective_consumer.py::test_exact_ep_collective_is_spec_driven_with_renamed_operator[deepseek-ai/DeepSeek-V4-Flash-lanes1-dsv4_ep_dispatch_allgather]
tests/unit/test_v4_measured_collective_consumer.py::test_exact_ep_collective_is_spec_driven_with_renamed_operator[zai-org/GLM-5.3-lanes2-expert_parallel_alltoall_combine]
tests/unit/test_v4_measured_collective_consumer.py::test_exact_ep_collective_is_spec_driven_with_renamed_operator[zai-org/GLM-5.3-lanes3-expert_parallel_alltoall_dispatch]
tests/unit/test_v4_measured_collective_consumer.py::test_h200_eager_ep_prices_equal_isolated_decision[lanes0-560.2-734.6]
tests/unit/test_v4_measured_collective_consumer.py::test_h200_eager_ep_prices_equal_isolated_decision[lanes1-1043.0-2172.0]
tests/unit/test_v4_measured_collective_consumer.py::test_unmeasured_dispatch_size_or_ordered_vector_is_typed[0-lanes1]
tests/unit/test_v4_measured_collective_consumer.py::test_unmeasured_dispatch_size_or_ordered_vector_is_typed[1-lanes0]
```

Exact removed IDs:

```text
tests/unit/test_glm_sp_moe_comm.py::test_equal_sp_rows_with_unequal_raw_tokens_are_covered[decode]
tests/unit/test_glm_sp_moe_comm.py::test_equal_sp_rows_with_unequal_raw_tokens_are_covered[prefill]
tests/unit/test_glm_sp_moe_comm.py::test_unequal_combine_other_backends_keep_scalar_prediction
tests/unit/test_glm_sp_moe_comm.py::test_unequal_dispatch_keeps_largest_rank_uniform_curve
tests/unit/test_measured_cc_backend.py::test_checked_in_unequal_combine_samples_match_producer_medians
tests/unit/test_measured_cc_backend.py::test_equal_sp_holdout_stays_out_of_exact_and_uniform_samples
tests/unit/test_measured_cc_backend.py::test_uneven_dispatch_uses_caller_supplied_largest_rank_bytes
tests/unit/test_v4_measured_collective_consumer.py::test_exact_combine_is_spec_driven_with_renamed_operator[deepseek-ai/DeepSeek-V4-Flash-lanes0-dsv4_ep_output_reduce_scatter]
tests/unit/test_v4_measured_collective_consumer.py::test_exact_combine_is_spec_driven_with_renamed_operator[zai-org/GLM-5.3-lanes1-expert_parallel_alltoall_combine]
```

All 162 failing/error nodes (75 failures, 87 errors including collection) match by status. `regression_comparison.json` retains full raw messages/traces. Of these messages, 147 are byte-identical; the 15 differences are:

| Node | Raw difference |
| --- | --- |
| `tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_base_does_not_default_to_private_conda_path` | Worktree path; normalized messages match. |
| `tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_base_resolves_latest_canonical_metrics_run_dir` | Same missing `test_base.sh`, missing helper and exit 127; path-length-dependent CompletedProcess truncation remains after narrow normalization. |
| `tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_base_uses_safe_pythonpath_expansion` | Worktree path; normalized messages match. |
| `tests/unit/test_colocation_release_review_contracts.py::test_debug_e2e_conda_activation_temporarily_disables_nounset` | Worktree path; normalized messages match. |
| `tests/unit/test_colocation_release_review_contracts.py::test_readme_debug_scripts_do_not_use_post_increment_under_set_e` | Worktree path; normalized messages match. |
| `tests/unit/test_colocation_release_review_contracts.py::test_readme_moe_debug_script_header_uses_current_tp_terms` | Worktree path; normalized messages match. |
| `tests/unit/test_colocation_release_review_contracts.py::test_readme_moe_debug_script_satisfies_shared_parallel_domain` | Worktree path; normalized messages match. |
| `tests/unit/test_colocation_release_review_contracts.py::test_readme_moe_debug_script_uses_valid_shared_parallel_domain` | Worktree path; normalized messages match. |
| `tests/unit/test_colocation_release_review_contracts.py::test_release_debug_scripts_use_canonical_metrics_resolver` | Worktree path; normalized messages match. |
| `tests/unit/test_dsv4_attention_runtime.py::test_sklearn_owner_trains_single_rows_reuses_cache_and_prices_replay` | Same log assertion; fixture memory address differs and normalizes. |
| `tests/unit/test_dsv4_pilot_cases.py::test_package_plan_cli_rebuilds_pilot_and_summary` | Worktree path; normalized messages match. |
| `tests/unit/test_mha_stage2_profile_modeling.py::test_stage2_cli_writes_reproducible_artifacts` | Same subprocess failure; root/basetemp paths normalize. |
| `tests/unit/test_mqa_stage2_profile_modeling.py::test_mqa_stage2_cli_writes_reproducible_artifacts` | Same subprocess failure; root/basetemp paths normalize. |
| `tests/unit/test_v4_replay_census.py::test_observer_snapshots_final_descriptor_and_records_end_once` | Same missing `_formed` entry; `KeyError: ('decode', 8378, 0)` versus `('decode', 8389, 0)`. Batch-ID normalization matches. |
| `tests/unit/test_v4_trace_audit_interval_metrics.py::IntervalMetricsTests::test_pins_and_caps` | Same original-environment assertion against populated prescribed PYTHONPATH; path length changes `[490 chars]`/656-character diff to `[550 chars]`/716-character diff. Narrow normalization retains it. |

Observed: 13 differences disappear using only root/scratch-path, address and generated-batch-ID normalization. Two truncated path/environment diagnostics remain unequal. Inference: these are presentation differences with the same baseline causes, not new product failures. Strict byte-identical message parity is not claimed; no broader masking was added.

Other baseline blockers include 42 `TarFile.extractall(filter=...)` fixture errors on Python 3.10, 25 CUPTI setup assertion errors, 19 structural setup errors attempting read-only `outputs`, 15 unavailable `torch.float8_e8m0fnu` failures, missing legacy scripts/package fixtures, native bootstrap/binding failures and unchanged assertions. Candidate nodes/statuses are identical; full evidence is in paired logs/XML and comparison JSON. Repairs, installs/upgrades or fixture relaxations were outside scope and were not attempted.

Scope evidence: `git diff --stat` shows only the eleven approved tracked paths, 526 insertions/219 deletions. `scope_check.json` independently verifies byte preservation, original provenance objects, Python parsing and all changed/new module sizes. Largest edited module: `families.py`, 969 lines; backend 325; largest edited test 522. Plain `git diff --check` flags required CRLF as trailing whitespace; `git -c core.whitespace=cr-at-eol diff --check` passes. B records remain ignored and unstaged.

Remaining work/decisions for Claude: provide approved `aiohttp` availability and resolve or classify existing fixture/environment failures before claiming complete required regression. Decide whether the 15 volatile raw differences, particularly the two truncated messages, satisfy the intended acceptance wording. No production work or out-of-scope cache repair remains. Table, prices, curve diff, changed tests and writable scope passed; full required collection and strict message equality remain unresolved.

Review handoff: inspect per-call kernel/geometry evidence, the one backend routing predicate, both spec builders and unchanged A10 attribution fields. Check all 108 indexed replacements against sources and all 276 preserved lines. Review line-level reasons for preserved combine assertions and replacing uniform dispatch behavior. Rerun the four affected modules/66 changed selectors, price/curve scripts and the recorded 192-selector base/candidate pair in the approved environment. CPU pricing/coverage is demonstrated; no hardware run, deployment, acceptance or commit is claimed.

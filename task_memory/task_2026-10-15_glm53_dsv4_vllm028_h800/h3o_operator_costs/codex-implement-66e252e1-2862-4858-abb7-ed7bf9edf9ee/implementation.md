**1. Assignment and outcome**

Implemented B17b in `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, starting from HEAD `77c5c1f61beb3f466feab930783620c87bf08ef5` (later than B17). The assigned implementation and CPU checks are finished. Final review, acceptance, commits, and task-record updates remain with Claude.

The writable scope was the named V4 predictor modules, `frontier/profiling/model_runner/dsv4_runner_tables.py`, the V4 `runtime_limitations` text, the named promotion script and manifest, the two runner CSV targets, and `tests/unit/`. All other generated files were written under scratch:

`/data/ycfeng/tmp/codex-implement/66e252e1-2862-4858-abb7-ed7bf9edf9ee/b17b/`

Read the applicable root AGENTS.md files, the B17 assignment/report/test report, both G2h reports, the identified requirements/plan/design/issue material, and the affected implementation and tests. No applicable CLAUDE.md or more-specific AGENTS.md was found along the edited paths. The pinned vLLM checkout reports revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`.

Preserved decisions:

- Sampling uses `(needs_logits_processing, b)`, with b equal to the number of real requests.
- MONOLITHIC and unified DECODE select processing on; PREFILL selects processing off.
- Preparation comes only from G2h-04. Sampling comes from G2h-04/on and G2h-02/off.
- Exact lookup, duplicate-coordinate means, and the existing activity-count gap path remain in use.
- Unmeasured coordinates raise `DeepSeekV4CoverageError`; PD-AF roles remain rejected.
- Preparation coordinates, attention rows, block costs, and gap data are unchanged.

The pinned sampler and proxy agree with these decisions: `sampler.py:75–87,192–199` selects/skips logits processing; `logit_bias.py:89–107` activates the bias for positive min_tokens with stop tokens; `model_runner.py:1122–1130,1339–1353` samples real-request rows. The proxy removes min_tokens on P at `pd_proxy.py:103–108` and forwards the original request to D at line 141. The inspected native P evidence contains no `_bias_kernel`; D evidence contains it. No contradictory evidence required a design change.

**2. Changes**

| Path | Change and reason |
| --- | --- |
| [dsv4_runner.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/dsv4_runner.py:23) | Added `SAMPLING_COLUMNS`, branch constants, `SAMPLING_BRANCH_KERNEL`, the immutable role mapping, and `sampling_coordinate` at line 38. The comment records the native request/proxy contract. |
| [dsv4_operator_predictor.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/dsv4_operator_predictor.py:210) | Loader admits and normalizes the two sampling columns, validates branch/b, and checks activity constancy per full coordinate. `train_dsv4_runner_models` at line 267 trains with both features. Existing `runner` exact lookup consumes the trained feature names without another lookup path. |
| [dsv4_runtime.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/dsv4_runtime.py:61) | `add_runner_scopes` prices sampling through the batch-role coordinate. Removed the PREFILL sampling rejection. Dummy forwards still do not sample. |
| [model_architectures.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/model_architectures.py:704) | Changed only V4 `runtime_limitations`: identifies the measured on/off request contracts and says other sampling parameters are unmeasured. |
| [dsv4_runner_tables.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/model_runner/dsv4_runner_tables.py:94) | Streams the identifying-kernel evidence alongside activity counters. `reduce_replay_rows` at line 114 requires a declared branch and checks every real step before promotion. Errors name run, case, and step. `include_preparation=False` admits only sampling from the second source. |
| [promote_runner_tables.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/promote_runner_tables.py:40) | Added primary/secondary branch arguments, different-branch validation, sampling-only secondary reduction, and per-run hashed provenance/row counts. Removed the obsolete G2h-02 structure-fixture guard. |
| [runner_manifest.json](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/runner_manifest.json:1) | Regenerated with all three source tags, both replay sources, both outputs, and the primary replay write comparisons. |
| [model_runner_replay_kernel_only.csv](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/model_runner_replay_kernel_only.csv:1) | New, commit-ready table: 989 preparation rows plus 976 sampling rows per branch. It is currently untracked and awaits Claude’s commit. |
| [test_dsv4_runner_tables.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_runner_tables.py:14) | Added branch-aware fixtures, mismatch checks in both directions, later-step/missing-sampling rejection, and deterministic two-source promotion/manifest checks. |
| [test_dsv4_runner_consumer.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_runner_consumer.py:66) | Checks both sampling features, typed missing coordinates, role-specific sampling, FULL/first-tail/PREFILL pricing, all-layer A1 coverage, b=8 worked examples, activity constancy, and invalid coordinates. Pins the new limitations text. |
| [test_dsv4_operator_consumer.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_operator_consumer.py:56) | Removed both test-only environment overrides and the pending-table skip. The trained fixture remains module-scoped and uses a scratch cache. |

The promotion also rewrote `model_runner_block_kernel_only.csv`; it is byte-identical to HEAD and therefore has no Git diff. `dsv4_training.py` needed no edit: both entry paths already call the updated common loader/trainer. `test_dsv4_runner_ledger.py` needed no edit: B17’s sentinel EXPECTED table already charges sampling once on real PREFILL forwards and zero times on dummies. The non-dummy A1 expectation changed from zero to one for PREFILL sampling.

No files were deleted or moved. Superseded code deletions were the PREFILL error block and its unused imports, the G2h-02 structure-fixture guard, the two fixture overrides and pending-table skip, the obsolete missing-replay constructor test, and unused `os`/`csv` imports in the affected tests.

Pre-existing work: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/progress.md` was modified before execution and remains intact. `.cortexkit/`, `.pi/`, and `remaining_measurements/` were left alone.

**3. Verification**

All checks ran from the repository root using Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6). Scratch `env.sh` sets the required PYTHONPATH, `PYTHONDONTWRITEBYTECODE=1`, W&B-disable variables, scratch TMPDIR/FRONTIER_TMP_ROOT, and `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Every command used a timeout and `ulimit -v 31457280`. No torch/vLLM installation, GPU, network, or package action was performed.

For the commands below:

```bash
S=/data/ycfeng/tmp/codex-implement/66e252e1-2862-4858-abb7-ed7bf9edf9ee/b17b
source "$S/env.sh"
```

Acceptance results:

| Acceptance | Observed result | Verdict |
| --- | --- | --- |
| 1 — tables | Block 197 rows with B17 SHA; replay 2,941 rows = preparation 989/138 keys + sampling 976/24 keys per branch. Counts constant per coordinate. Second promotion reproduced both CSVs and the manifest. Fixture mismatches in both directions and one real-input mismatch check passed. | PASS |
| 2 — consumer | All exact runner coordinates match duplicate means. b=8 examples pass for both branches. Missing b=25 raises with the branch/b coordinate. Both training paths and warm cache agree; table mutation changes cache identity. Independent source-statistics/cache checks passed. | PASS |
| 3 — roles | Non-dummy TP8/DP2/EP16 PREFILL, DECODE, and MONOLITHIC construct from target tables; PREFILL, first-tail, and FULL forwards price sampling with the expected branch. Dummies do not sample. DECODE_ATTN/DECODE_FFN still raise. | PASS |
| 4 — A1 | The fixture makes any `_dummy_execution_time` read raise. MONOLITHIC and PDD forwards, including PREFILL final-layer sampling, pass on target tables alone. | PASS |
| 5 — A3 | Six 43-layer real/dummy ledger scenarios pass with the existing EXPECTED table and phase ownership. | PASS |
| 6 — regression | 886 selected tests passed; current and HEAD-overlay smoke matrices each passed 12/12; all 36 parsed non-V4 CSV/JSON output files match. | PASS |
| 7 — text/size | New limitations text is pinned by tests. All nine touched Python modules compile and are below 2,000 lines. `git diff --check` passes. | PASS |

Promotion command, executed twice with the same arguments:

```bash
timeout 240s "$PYTHON_BIN" task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/promote_runner_tables.py \
  --block-dir /data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_03/download/block/G2H_BLOCK_OPERATORS \
  --block-run-tag v4-h200-g2h-runner-20261029-03-block \
  --replay-raw-dir /data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_04/download/replay/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw \
  --replay-run-tag v4-h200-g2h-runner-20261029-04-replay \
  --replay-logits-processing 1 \
  --sampling-raw-dir /data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_02/download/replay/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw \
  --sampling-run-tag v4-h200-g2h-runner-20261029-02-replay \
  --sampling-logits-processing 0
```

| Artifact | Data rows | SHA-256 |
| --- | ---: | --- |
| Block CSV | 197 | `e0a6dadfa844883603e22be2906e34d987060bc487f3290b821c9a0a6c3139cd` |
| Replay CSV | 2,941 | `de7607f1142f95a6730b57e1da197211b9ec26d9981e10c0cba15da63f81f58f` |
| Manifest | JSON object | `4ca324a62a1c20a24d20cd5a82271e4d467a34220f89f8c436050f93edc50d93` |

Manifest summary: 213 distinct input entries with hashes and row counts, two output entries, both replay-run records, all three source tags, and 1,466 primary replay write-comparison entries. G2h-04 contributes preparation 989 and sampling 976; G2h-02 contributes sampling 976 only. The manifest is 408,780 bytes. Its second-run bytes equal the first-run copy at `$S/first_manifest.json`. Block, attention, and `kernel_gap.csv` bytes were also compared directly with `git show HEAD:<path>` and match.

Preparation key inventory uses the unchanged eight-column coordinate. Define `(W,B)` pairs as W=1/B=1; W=2/B=1–2; W=4/B=1–4; W=8/B=1–8; W=16/B=1–16; W=24/B=17–24 (39 pairs).

- Decode FULL: those 39 pairs, dispatch=W, prefill=0, decode=B, tokens=B.
- Decode PIECEWISE: the same 39 pairs, dispatch absent, prefill=0, decode=B, tokens=B.
- First-tail PIECEWISE: those 39 pairs with prefill=1/decode=B−1, plus seven `(W,B,prefill,decode)` coordinates: `(2,2,2,0)`, `(4,3,2,1)`, `(4,3,3,0)`, `(4,4,4,0)`, `(8,5,2,3)`, `(8,7,2,5)`, `(8,8,3,5)`. Dispatch is absent and tokens=B.
- Dummy FULL/PIECEWISE: six keys each at W=1,2,4,8,16,24; real geometry absent. FULL dispatch=W; PIECEWISE dispatch absent.
- Dummy NONE: W=1, dispatch=1, real geometry absent.
- PREFILL NONE: W=6144, dispatch=1, B=1, prefill=1, decode=0, tokens=6144.

These total 138 keys. Their mean-cost ranges, in ms:

| Preparation class | Keys | Minimum–maximum per-key mean cost |
| --- | ---: | --- |
| Decode FULL | 39 | 0.086507324–0.103006999 |
| Decode PIECEWISE | 39 | 0.086557210–0.100579210 |
| First-tail PIECEWISE | 46 | 0.089600586–0.105775391 |
| Dummy FULL | 6 | 0.067075684–0.078684082 |
| Dummy PIECEWISE | 6 | 0.063040527–0.077379395 |
| Dummy NONE | 1 | 0.067235352 |
| PREFILL NONE | 1 | 0.143780273 |

Every preparation key’s complete coordinate, source-row count, minimum/mean/maximum cost, and activity count is recorded in [preparation_keys.csv](/data/ycfeng/tmp/codex-implement/66e252e1-2862-4858-abb7-ed7bf9edf9ee/b17b/preparation_keys.csv:1). Full-precision results for both operators are in [table_stats.json](/data/ycfeng/tmp/codex-implement/66e252e1-2862-4858-abb7-ed7bf9edf9ee/b17b/table_stats.json:1).

Sampling has all 48 keys: branches 0 and 1, each at b=1–24. Branch 0 has 9 kernels; branch 1 has 11. Both have zero memcpy and memset activities at every key. The following lists mean cost and the minimum–maximum source-row cost in ms; row counts are equal between branches:

| b | Rows per branch | Branch 0 mean [min, max] | Branch 1 mean [min, max] |
| ---: | ---: | --- | --- |
| 1 | 66 | 0.035242861 [0.024384766, 0.042080566] | 0.045851015 [0.042304688, 0.052672363] |
| 2 | 97 | 0.030118678 [0.022176270, 0.040381836] | 0.044186755 [0.032800781, 0.057277832] |
| 3 | 83 | 0.028226898 [0.021600098, 0.042500000] | 0.043902591 [0.037089355, 0.050621094] |
| 4 | 82 | 0.025155017 [0.021407227, 0.038722168] | 0.039264339 [0.031009766, 0.046238770] |
| 5 | 55 | 0.031821342 [0.023007812, 0.048191895] | 0.052254847 [0.043294434, 0.062015625] |
| 6 | 54 | 0.027639838 [0.022881348, 0.035903809] | 0.046249195 [0.039073242, 0.058786621] |
| 7 | 55 | 0.033277255 [0.024927734, 0.051391602] | 0.055326864 [0.044544434, 0.068930176] |
| 8 | 55 | 0.027516877 [0.023167480, 0.041088867] | 0.043092525 [0.038461914, 0.051967773] |
| 9 | 27 | 0.031779478 [0.026177734, 0.051137207] | 0.057634729 [0.049761719, 0.070849121] |
| 10 | 27 | 0.029276259 [0.025278809, 0.042751953] | 0.053661947 [0.044735840, 0.060575684] |
| 11 | 27 | 0.029806424 [0.024989258, 0.043071777] | 0.049960865 [0.044322266, 0.056737305] |
| 12 | 27 | 0.031065972 [0.024512207, 0.043007812] | 0.049885290 [0.044895508, 0.057631836] |
| 13 | 24 | 0.032814290 [0.026875977, 0.048480469] | 0.063482544 [0.055808594, 0.072927734] |
| 14 | 27 | 0.034110749 [0.026240234, 0.049088379] | 0.061086968 [0.054369141, 0.070655762] |
| 15 | 27 | 0.035596770 [0.026976563, 0.054718750] | 0.064637207 [0.047711426, 0.080353027] |
| 16 | 27 | 0.034271900 [0.026463867, 0.052479004] | 0.058272461 [0.050463867, 0.068834961] |
| 17 | 27 | 0.037678277 [0.027263672, 0.055775879] | 0.064416486 [0.058367188, 0.068929688] |
| 18 | 27 | 0.045394893 [0.028545898, 0.061344727] | 0.068616717 [0.062304688, 0.075809570] |
| 19 | 27 | 0.047253400 [0.029248047, 0.058496094] | 0.069787959 [0.063585449, 0.076319824] |
| 20 | 27 | 0.042601834 [0.027583984, 0.055167480] | 0.064381583 [0.043199707, 0.071105957] |
| 21 | 27 | 0.044218985 [0.029375488, 0.063293457] | 0.070266204 [0.062882812, 0.075966309] |
| 22 | 27 | 0.042057635 [0.028640137, 0.057889160] | 0.065292390 [0.044640137, 0.073536621] |
| 23 | 27 | 0.043908077 [0.029504883, 0.058112793] | 0.074221625 [0.063715332, 0.080068848] |
| 24 | 27 | 0.041532480 [0.029151855, 0.057216797] | 0.072384096 [0.065824219, 0.078048340] |

The full-precision sampling inventory is [sampling_keys.csv](/data/ycfeng/tmp/codex-implement/66e252e1-2862-4858-abb7-ed7bf9edf9ee/b17b/sampling_keys.csv:1).

Worked consumer examples use the mean of each coordinate’s source-row kernel medians plus the mean memcpy medians. The gap is activity count × unchanged `eager_preparation` value, 0.0012208671118771937 ms:

| Coordinate | Rows | Mean kernel median, ms | Mean memcpy median, ms | Activities | Gap, ms | Sampling plus gap, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (0, 8) | 55 | 0.027516876775568185 | 0 | 9 | 0.010987804006894744 | 0.038504680782462927 |
| (1, 8) | 55 | 0.04309252485795454 | 0 | 11 | 0.013429538230649131 | 0.05652206308860367 |
| (0, 1), PREFILL | 66 | 0.03524286073626894 | 0 | 9 | 0.010987804006894744 | 0.04623066474316369 |

The independent hand check uses standard-library CSV reading and `math.fsum`, verifies that all 2,941 promoted rows retain the producer kernel/memcpy statistics, and compares saved model exact lookups. Five sampling cost artifacts and five count artifacts from predictor/role/manager paths agree on all 48 coordinates. The warm/shared training test also compares all 578 model keys and exact lookup/count dictionaries.

Expected H200 effect: b=1 PREFILL now prices sampling at 0.035242861 ms plus 0.010987804 ms gap instead of raising. A b=8 DECODE FULL forward prices branch 1 at 0.043092525 ms plus 0.013429538 ms gap. Relative to B17’s tested G2h-02 b-only key at b=8, sampling increases by 0.015575648 ms and the gap by 0.002441734 ms, totaling 0.018017382 ms. B17 had no successful PREFILL sampling result to subtract. Preparation now uses G2h-04 measurements as assigned; no scaling or gap-data change was introduced.

Exact executed test commands:

```bash
timeout 600s "$PYTHON_BIN" -m pytest tests/unit/test_dsv4_runner_tables.py tests/unit/test_dsv4_runner_consumer.py tests/unit/test_dsv4_operator_consumer.py tests/unit/test_dsv4_runner_ledger.py tests/unit/test_dsv4_kv_block_table_writes.py -q -p no:cacheprovider --basetemp "$B17B_SCRATCH/pytest/consumers_corrected" > "$B17B_SCRATCH/logs/consumers_corrected.log" 2>&1
```

PASS: 62 passed, 7 existing attention-reader mixed-type warnings, 494.34 seconds. The initial execution with the same file list used basetemp `consumers` and log `consumers.log`; it produced 31 passed, 2 failed, and 29 setup errors because mixed CSV union columns arrived as strings. Normalizing the sampling columns before integer validation fixed that acceptance blocker. The corrected full run above passed; no failures were skipped or xfailed.

```bash
timeout 240s "$PYTHON_BIN" -m pytest tests/unit/test_dsv4_cache_zero_execution.py tests/unit/test_grouped_kv_cache_scheduler.py tests/unit/test_dsv4_structural_pd.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_graph_measured.py tests/unit/test_dsv4_source_execution.py tests/unit/test_dsv4_attention_runtime.py tests/unit/test_dsv4_dp_sync.py tests/unit/test_dsv4_attention_workload.py tests/unit/test_glm53_structural_execution.py tests/unit/test_glm53_structural_pd.py -q -p no:cacheprovider --basetemp "$B17B_SCRATCH/pytest/structural" > "$B17B_SCRATCH/logs/structural.log" 2>&1
```

PASS: 291 passed, 30.47 seconds.

```bash
timeout 240s "$PYTHON_BIN" -m pytest tests/unit/test_dsv4_features.py tests/unit/test_dsv4_predictor.py tests/unit/test_dsv4_model_config.py tests/unit/test_dsv4_operator_manifest.py tests/unit/test_dsv4_operator_admission.py tests/unit/test_dsv4_composition.py tests/unit/test_dsv4_composition_check.py tests/unit/test_dsv4_runner_operators.py tests/unit/test_dsv4_replay_rows.py tests/unit/test_dsv4_replay_gaps.py tests/unit/test_dsv4_h200_tables.py tests/unit/test_dsv4_step_cost_input.py tests/unit/test_kernel_gap_pricing.py tests/unit/test_operator_registry.py tests/unit/test_shared_prediction_model_manager_eager_attention_decode.py tests/unit/test_shared_prediction_model_manager_eager_attention_mla.py tests/unit/test_shared_prediction_model_manager_mixed_layer_moe.py tests/unit/test_metrics_stage_execution_time.py tests/unit/test_metrics_full_stage_scope.py tests/unit/test_transfer_metrics_contract.py -q -p no:cacheprovider --basetemp "$B17B_SCRATCH/pytest/related" > "$B17B_SCRATCH/logs/related.log" 2>&1
```

PASS: 533 passed, 12.82 seconds.

```bash
METRICS_OUTPUT_DIR="$B17B_SCRATCH/smoke" RUN_ID_PREFIX=b17b_smoke timeout 600s bash examples/architecture/run_dummy_smoke_matrix.sh -- --metrics_config_cache_dir "$B17B_SCRATCH/smoke_cache" > "$B17B_SCRATCH/logs/smoke.log" 2>&1

export PYTHONPATH="$B17B_SCRATCH/baseline:$PYTHONPATH"
"$PYTHON_BIN" -c 'import frontier.model_architectures as m; assert "b17b/baseline" in m.__file__; print(m.__file__)'
METRICS_OUTPUT_DIR="$B17B_SCRATCH/baseline_smoke" RUN_ID_PREFIX=b17b_smoke timeout 600s bash examples/architecture/run_dummy_smoke_matrix.sh -- --metrics_config_cache_dir "$B17B_SCRATCH/baseline_smoke_cache" > "$B17B_SCRATCH/logs/baseline_smoke.log" 2>&1
```

PASS: both matrices completed 12/12 with exit 0. The baseline overlay contains HEAD bytes obtained with read-only `git show` for the five affected production modules; its `sitecustomize.py` redirects only those module imports. Current and baseline share the same smoke run IDs. The baseline import assertion passed.

Additional executed commands:

```bash
timeout 120s "$PYTHON_BIN" "$B17B_SCRATCH/wrong_branch_check.py" > "$B17B_SCRATCH/logs/wrong_branch.log" 2>&1
timeout 60s "$PYTHON_BIN" "$B17B_SCRATCH/table_stats.py" > "$B17B_SCRATCH/logs/table_stats.log" 2>&1
timeout 60s "$PYTHON_BIN" "$B17B_SCRATCH/hand_check.py" > "$B17B_SCRATCH/logs/hand_check.log" 2>&1
timeout 60s "$PYTHON_BIN" "$B17B_SCRATCH/compare_non_v4.py" > "$B17B_SCRATCH/logs/non_v4_comparison.log" 2>&1
timeout 30s git diff --check
```

All PASS. The real-input wrong declaration produced:

`Run v4-h200-g2h-runner-20261029-02-replay, case g1-e0-0000-full, step 0: declared needs_logits_processing=1, observed 0 from _bias_kernel`

Non-V4 comparison: 36/36 parsed files equal across 12 cases, covering request metrics, system metrics, and operator precision metadata. Details are in `$S/non_v4_comparison.json`. Nine touched Python files were compiled with `compile(source, path, 'exec')` in a timed, bytecode-free Python command.

Final Python line counts:

| Module | Lines |
| --- | ---: |
| dsv4_runner.py | 66 |
| dsv4_operator_predictor.py | 386 |
| dsv4_runtime.py | 101 |
| dsv4_runner_tables.py | 169 |
| model_architectures.py | 1,343 (HEAD: 1,342; V4 text only) |
| promote_runner_tables.py | 136 |
| test_dsv4_runner_tables.py | 120 |
| test_dsv4_runner_consumer.py | 269 |
| test_dsv4_operator_consumer.py | 329 |

**4. Remaining work and decisions for Claude**

No implementation blocker or material design deviation remains. Review and commit the new replay CSV together with the code and regenerated manifest. No Git state-changing command or task-record update was performed.

Observations: all assigned selected CPU checks passed; the block, attention, and gap files are byte-identical to HEAD; producer kernel/memcpy statistics are preserved in every promoted row. The b=8 sampling delta above follows directly from the measured coordinate means and unchanged gap value.

Limits: no E2E replay, 77-case fidelity matrix, or A11 diagnostic cross-check was run; these remain explicitly assigned to the lead. The known `add_requests` staged-copy cost remains unpriced. Sampling coverage is limited to the two measured request contracts; other sampling parameters remain unmeasured. The shared native sampling input argument for TP8 follows the inspected gathered-logits path; this implementation does not claim a new TP8 GPU measurement.

**5. Review handoff**

Inspect the registry role mapping and evidence comment, loader normalization and two-feature training, every-real-step branch verification, sampling-only secondary promotion, source provenance, and the removed PREFILL/fixture gates. Check the replay CSV and manifest hashes against the values above. The A3 expected ledger is intentionally unchanged because it already described one real PREFILL sampling invocation.

Rerun the 62-test consumer/reducer/ledger command and the real-input wrong-branch check first. The promotion command can reproduce the targets and manifest when the read-only measured inputs remain available. The structural/related suites and dummy smoke commands above provide the regression rerun set. Complete the lead-owned E2E/fidelity/A11 checks before final release acceptance.

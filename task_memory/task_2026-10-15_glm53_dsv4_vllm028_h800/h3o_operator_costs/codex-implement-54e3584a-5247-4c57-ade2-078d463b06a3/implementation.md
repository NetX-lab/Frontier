**1. Assignment and outcome**

Implemented B7 in `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`: shared batch features, a replay loader/trainer/predictor, shared measured admission helpers, and attention composition using one signed gap per scope. The assigned CPU checks are complete. Final review and acceptance remain with Claude.

Edits are confined to the explicitly listed files within `frontier/attention/`, `frontier/execution_time_predictor/`, `frontier/profiling/attention/`, `tests/unit/`, and `tests/comparison/calibration/`. Validation scripts, test directories, logs and reports are under `/data/ycfeng/tmp/codex-implement/54e3584a-5247-4c57-ade2-078d463b06a3/b7/`.

Preserved decisions: phase labels match the profiler; request features use closed forms; only `dsv4_sparse_attention` separates tile-planner roles; other scopes pool those roles; costs and gaps have separate models; gaps may be negative; dummy features contain only dispatch rows. No runtime registration, serving admission, real sklearn fitting, kernel-gap wiring, cache writes, GPU work, network work, dependency changes or Git state changes were performed.

Read the applicable repository AGENTS.md guidance, the B7 assignment, route/design material, planner-role evidence script, existing DSA and V4 implementations, the replay fixtures and the A4 baseline script. No nested AGENTS.md or CLAUDE.md files were found under the affected source/test directories.

The initial tracked tree was clean. The pre-existing untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` directories were preserved. During execution, concurrent changes appeared in `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/stage1a_replay_route.md` and `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/progress.md`; I did not edit those files.

**2. Changes**

Paths below are relative to the working root. Line counts cover every executor-changed Python file.

| File | Lines | Change and relevant reference |
| --- | ---: | --- |
| `frontier/attention/dsv4_features.py` — new | 134 | Frozen `DeepSeekV4BatchFeatures`, named phases, explicit 22-scope feature table, phase-qualified column lookup and closed-form causal sums. `build_dsv4_features`:78; dummy builder:132. |
| `frontier/attention/deepseek_v4.py` | 471 | Named FlashMLA padded widths and `deepseek_v4_padded_q_heads`:138, used by both existing workload builders. |
| `frontier/execution_time_predictor/dsv4_predictor.py` — new | 280 | Replay filename resolution:48; strict table loader:121; deterministic keys/names:198/204; cost/gap trainer:209; frozen result:238; covered exact/interpolated prediction:243. |
| `frontier/execution_time_predictor/deepseek_v4.py` | 975 | Optional schema keyword:560; public `validate_measured_identity`:930; shared balanced-routing validation:961. Isolated reads retain explicit eager selection after shared identity validation. Error text identifies device, fingerprint, parallel-group and routing columns. |
| `frontier/execution_time_predictor/dsv4_composition.py` | 194 | `compose_attention_phase`:165 now accepts costs and one signed gap per scope. Requires equal key sets. Stream program, predecessor calculation, serial composition and `segment_class` remain unchanged. |
| `frontier/profiling/attention/dsv4_replay_rows.py` | 213 | Public `DSV4_REPLAY_OP_NAME`:42 replaces the main module’s private constant. No row-format change. |
| `frontier/profiling/attention/dsv4_attention_main.py` | 272 | Aggregate output naming uses the shared replay op constant. |
| `tests/comparison/calibration/dsv4_composition_check.py` | 200 | Own-gap composition supplies measured scope gaps directly; class-gap diagnostics supply count × class value per scope at lines 142/151. |
| `tests/unit/test_dsv4_features.py` — new | 95 | Phase/mix/decode contracts; random and boundary closed-form checks across all variants; ratio restrictions; dummy/table coverage; padded-head helper. |
| `tests/unit/test_dsv4_predictor.py` — new | 268 | Aggregate round trip over 60 rows spanning all phases/modes; key pooling; signed gaps; absent scopes; bounds/schema/missing keys; interpolation; result validity; loader rejection and filename contracts. |
| `tests/unit/test_dsv4_composition.py` | 261 | Per-scope signed-gap test:113, segment-class mapping:138, unknown mode:142, missing/extra gap-key rejection:147. |
| `tests/unit/test_dsv4_replay_rows.py` | 270 | Row composition test uses separate own-cost and own-gap maps at line 137. |
| `tests/unit/test_profiling_runtime_boundary.py` | 91 | Commented allowlist entries for composition’s segment schema and predictor’s CPU replay schema module at line 13. |

No changes were needed in `profiling_dataframe_loaders.py`, `sklearn_execution_time_predictor.py`, `test_dsv4_graph_measured.py` or `test_dsv4_attention_workload.py`.

The new predictor test adapts the reused synthetic records to H200 and `fp8_ds_mla` before aggregation. The original fixture records describe H800 and distinguish requested `fp8` from realized `fp8_ds_mla`; those facts otherwise trigger the intended new identity/layout checks. This adaptation changes only the new test input, preserving existing fixtures and production admission.

**3. Verification**

All CPU checks used `/data/ycfeng/envs/frontier-py310/bin/python`, verified as Python 3.10.6, with this setup from the working root:

```sh
ulimit -v 31457280
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
```

Every command had a timeout. Pytest cache creation was disabled, and test temporary directories were placed in scratch.

The main regression command was:

```sh
timeout 240s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_features.py tests/unit/test_dsv4_predictor.py tests/unit/test_dsv4_composition.py tests/unit/test_dsv4_composition_check.py tests/unit/test_dsv4_replay_rows.py tests/unit/test_dsv4_replay_gaps.py tests/unit/test_dsv4_graph_bridge_check.py tests/unit/test_dsv4_kernel_identity_check.py tests/unit/test_dsv4_graph_replay_profiling.py tests/unit/test_dsv4_attention_profiling.py tests/unit/test_dsv4_kv_cache_groups.py tests/unit/test_dsv4_graph_measured.py tests/unit/test_dsv4_attention_workload.py tests/unit/test_dsv4_step_cost_input.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_source_execution.py tests/unit/test_dsv4_structural_pd.py tests/unit/test_dsa_predictor.py tests/unit/test_dsa_attention_profiling.py -q -r fEs -p no:cacheprovider --basetemp=/data/ycfeng/tmp/codex-implement/54e3584a-5247-4c57-ade2-078d463b06a3/b7/pytest_regression
```

PASS. Output tail in `b7/regression.log`:

```text
883 passed, 75 skipped in 126.43s (0:02:06)
```

All 75 skips are unchanged accepted-prefill tests in `test_dsv4_graph_measured.py`, reporting `Accepted prefill receipts/package are unavailable`. No failures occurred.

The new predictor suite also passed separately: `41 passed in 6.90s`, recorded in `b7/predictor_second.log`. Its first run failed because the reused fixture’s H800/requested-fp8 identity did not satisfy the intended H200/realized-cache contract; the test input adaptation above resolved that cause.

After the final diagnostic-text change identifying `moe_routing`, ran:

```sh
timeout 120s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_predictor.py tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/codex-implement/54e3584a-5247-4c57-ade2-078d463b06a3/b7/pytest_identity_final
```

PASS. `b7/identity_final.log` ends with `162 passed, 75 skipped in 87.65s`. A direct CPU check also verified that malformed balanced-routing metadata raises `DeepSeekV4CoverageError` naming `moe_routing`.

Executed the documented failure checks:

```sh
timeout 90s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_profiling_runtime_boundary.py tests/unit/test_kernel_gap_pricing.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/codex-implement/54e3584a-5247-4c57-ade2-078d463b06a3/b7/pytest_known_failures
```

FAIL, exactly as expected: `2 failed, 22 passed in 1.83s`. `b7/known_failures.log` records:

```text
AssertionError: deepseek_v4.py
Extra items in the left set:
'frontier.profiling.attention.dsa_rows'

FAILED test_every_collective_launches_one_kernel_by_default
AttributeError: 'OperatorSpec' object has no attribute 'kernel_count'
```

The independent AST check examined every predictor module rather than stopping at the first mismatch:

```sh
timeout 60s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/codex-implement/54e3584a-5247-4c57-ade2-078d463b06a3/b7/boundary_check.py
```

PASS. `b7/boundary_ast.log` contains exactly:

```json
[
  {
    "module": "deepseek_v4.py",
    "extra": ["frontier.profiling.attention.dsa_rows"],
    "missing": []
  }
]
```

A4 baseline verification:

```sh
timeout 180s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_baseline.py /data/ycfeng/tmp/codex-implement/54e3584a-5247-4c57-ade2-078d463b06a3/b7/a4_report.json
timeout 10s cmp /data/ycfeng/tmp/codex-implement/54e3584a-5247-4c57-ade2-078d463b06a3/b7/a4_report.json /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_report_d2a226fe.json
```

PASS. `b7/a4.log` reports `PASS 125`; `cmp` produced no differences. Both reports have SHA256 `0505669f349d2a9ae4b4b0bbc164bb29d4d84c6291c2d62e96ead0cc1ebb313b`.

Real-table validation:

```sh
timeout 120s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/codex-implement/54e3584a-5247-4c57-ade2-078d463b06a3/b7/real_table_check.py
```

PASS. The scratch script loads the specified read-only table with the full DeepSeek-V4-Flash config, TP8 replica namespace, H200, `fp8_ds_mla`, balanced routing, block size 256 and KERNEL_ONLY. It fits through the exact-lookup fake and predicts/composes every row from its own reconstructed batch.

Input SHA256 is `277626cd766865bd0d006d73bf01b31292d7e3f536acce919b1c7ceca7f30ff6`, matching the supplied B6 identity. Loaded all 505 rows and trained 554 models:

| Phase | Cost models | Gap models | Total |
| --- | ---: | ---: | ---: |
| decode | 80 | 80 | 160 |
| dummy | 117 | 117 | 234 |
| first_tail | 40 | 40 | 80 |
| prefill | 40 | 40 | 80 |
| Total | 277 | 277 | 554 |

Absolute relative composition error against `layer_window_stats.median`:

| Statistic | Observed | Assigned limit |
| --- | ---: | ---: |
| Median | 0.490904% | ≤5% |
| p90 | 1.342730% | ≤10% |
| Maximum | 6.672723% | No maximum limit specified |

Two rows exceed 5%:

| Row | Case / model key | Cached decode tokens | Error |
| --- | --- | --- | ---: |
| 13 | `e0-decode-b1-p6144`; csa / FULL / decode / planner=False / padded_rows=1 | `[6151]` | 5.751993% |
| 204 | `e0-decode-b4-p6144`; csa / FULL / decode / planner=True / padded_rows=4 | `[6150, 6151, 6152, 6153]` | 6.672723% |

`b7/real_table_report.json` records their complete keys, predicted/target windows and every pooled-scope cost/gap difference. The largest absolute pooled scope-weight difference in each is `dsv4_compressor_fused_compress_store`: −5.419922 µs for row 13 and +8.420944 µs for row 204. This identifies a leading contributor; it does not imply that scope deltas simply add through the fork/join schedule. No key or feature tuning was performed.

Cold imports:

```sh
timeout 90s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/codex-implement/54e3584a-5247-4c57-ade2-078d463b06a3/b7/cold_import_check.py
```

PASS. The script uses a fresh subprocess for each module and checks loaded modules. `b7/cold_import.log` reports:

```text
PASS frontier.execution_time_predictor.dsv4_predictor no torch/vllm/cuda modules loaded
PASS frontier.attention.dsv4_features no torch/vllm/cuda modules loaded
PASS frontier.execution_time_predictor.dsv4_composition no torch/vllm/cuda modules loaded
```

Size and whitespace checks:

```sh
timeout 10s git diff --check -- frontier/attention frontier/execution_time_predictor frontier/profiling/attention tests/unit tests/comparison/calibration/dsv4_composition_check.py
```

PASS with no output. `wc -l` produced the counts in the change table; every changed module is below 2,000 lines, including the shared predictor module at 975.

**4. Remaining work and decisions for Claude**

There is no unresolved B7 implementation blocker or material design deviation. The real-table median and p90 limits pass using the agreed keys/features.

The 75 skipped accepted-prefill regressions require the unavailable receipts/package to execute. They were not fabricated or replaced. Restoring that prerequisite would allow Claude to rerun those cases.

The two documented failures remain: the isolated path’s extra `dsa_rows` import, scheduled for B8 removal, and the collective-default kernel-count assertion. No unrelated fix was attempted.

B8 remains responsible for predictor registration and real sklearn training, PREFILL/DECODE admission, trained runtime composition, isolated-path removal and kernel-gap reader wiring. Single-row sklearn training behavior and serving-capacity comparisons remain untested and out of this assignment.

**5. Review handoff**

Inspect the new feature table and causal formulas; loader identity/layout/ownership checks; planner-only key split; target naming and bounds; exact-lookup behavior; and conversion of all assigned composition callers to per-scope gap maps. The existing graph-measured tests were preserved and passed wherever their data prerequisites were available.

For acceptance, rerun the main regression command, A4 baseline plus `cmp`, and `b7/real_table_check.py`. Review `b7/real_table_report.json` for the two >5% rows. The boundary AST listing should continue to contain only the expected `deepseek_v4.py` extra import until B8 removes the isolated path.

All scratch scripts and artifacts are available under `/data/ycfeng/tmp/codex-implement/54e3584a-5247-4c57-ade2-078d463b06a3/b7/`. No commit or acceptance action was taken.

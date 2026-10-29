**1. Assignment and outcome**

B8 runtime implementation and all substantive CPU acceptance checks are complete. The outcome is **partial solely for two literal hygiene conditions in Acceptance 9**: the supplied import scanner reports pre-existing sklearn names, and the required unfiltered grep finds removed symbols in pre-existing bytecode. Resolving those findings requires cleanup beyond the assignment’s “registration lines only” and “only four deletions” restrictions. Final review and acceptance remain with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Baseline: `dd4c4317606789e5cafcba148f6c1955f3b96f7f`. Scratch root: `/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8`.

Writes stayed within the specifically assigned files under `frontier/attention`, `frontier/entities`, `frontier/execution_time_predictor`, `tests/unit`, and `tests/integration`, the four authorized deletions under those directories plus `tests/comparison/calibration` and `tests/fixtures`, and private scratch. No commits, Git state changes, network access, dependency changes, GPU operations, delegation, or task-record writes were performed.

Applicable guidance included the root `AGENTS.md`, `docs/general/module-size-exceptions.md`, the B8 assignment, the B7 assignment and evidence, and design §3.7. The B8 brief’s narrower scope governed the implementation; broader design text was treated as evidence for later assignments.

The read-only H200 replay table contains 505 rows and retains SHA256 `277626cd766865bd0d006d73bf01b31292d7e3f536acce919b1c7ceca7f30ff6`. Its measured GEMM threshold is 1024. Tests use the genuine H200 V4 replica configuration with TP8, attention DP8, EP64, balanced routing, `FLASHMLA_SPARSE_DSV4`, and `kv_cache_dtype="fp8"`.

Settled behavior is preserved: trained attention returns scope costs, signed gaps, workloads, and composition inputs; structural and step-cost execution retain serial composition. Full measured serving admission, PD admission, manager registration, kernel-gap wiring, and removal of static warmup operators remain deferred as assigned.

**2. Changes**

Fourteen existing files were modified and one test module was added. Final line counts are included below; sklearn decreased from 8,666 to 8,626 lines and remains covered by the documented size exception.

| File | Final lines | Change and reason |
| --- | ---: | --- |
| [frontier/attention/deepseek_v4.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/attention/deepseek_v4.py:257) | 421 | Unified real and dummy workload construction. Uses V2 NONE/FULL/PIECEWISE geometry; compressed stores always execute; CSA short fill occurs only under NONE; scores follow the request-geometry rule. Publishes planner and short-context flags. Dummy valid-row fields are zero and serving-request fields are empty. Removed the separate dummy builder and metadata flag. |
| [frontier/attention/families.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/attention/families.py:130) | 710 | Added the named alias table and public `deepseek_v4_main_kv_cache_dtype()` helper. Cache resolution retains its previous behavior, and replay admission now accepts the replica’s `fp8` alias for measured `fp8_ds_mla`. |
| [frontier/entities/batch.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/entities/batch.py:1353) | 1,612 | Docstring-only change distinguishing V1 dummy metadata facts from V2 descriptor geometry, padding, dummy metadata, and FULL restaging. |
| [frontier/execution_time_predictor/deepseek_v4.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/deepseek_v4.py:75) | 804 | Added `predict_attention_operator_times()` for both synthetic and trained pricing. Rewired source execution, attached per-scope gaps and composition inputs, and removed the isolated attention branch, layout helper, and metadata flag. Retained non-attention isolated readers and identity selection. No attention kernel counts are recorded. |
| [frontier/execution_time_predictor/dsv4_predictor.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/dsv4_predictor.py:51) | 333 | Moved the replay-only dataframe loader and balanced-routing validator here. Applied the shared KV alias. Training attaches the measured threshold to every returned model, including cache hits; the predictor constructor derives and validates one common threshold from its own models. |
| [frontier/execution_time_predictor/sklearn_execution_time_predictor.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/sklearn_execution_time_predictor.py:1436) | 8,626 | Added the assigned V4 loader, trainer, predictor-registration, and runtime branches. One-row keys fit directly and log “trained on its single row”; later identity, exact-lookup, cache, and kernel-count pairing steps remain shared. Removed the old dummy attention method. Changes are confined to the assigned locations. |
| [frontier/execution_time_predictor/profiling_dataframe_loaders.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/execution_time_predictor/profiling_dataframe_loaders.py:315) | 975 | Deleted only the V4 special branch. The manager path again reaches the family’s execution-admission guard, as required. |
| [tests/integration/run_dsv4_structural.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/integration/run_dsv4_structural.py:98) | 483 | Updated the boundary validator to require a store whenever compression ratio is greater than one. |
| [tests/unit/test_dsv4_attention_workload.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_attention_workload.py:108) | 365 | Updated unconditional HCA-store expectations, removed metadata-flag arguments, and replaced old dummy-method tests with tests of the shared synthetic pricing entry point. |
| [tests/unit/test_dsv4_attention_runtime.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_attention_runtime.py:62) — new | 235 | Added synthetic scope-rule branches and a genuine sklearn owner/cache/runtime test. Includes one-row training, unchanged cache mtimes/hashes/estimator count on retraining, threshold recovery, FULL decode and FULL/NONE dummy costs, signed gaps, composition, and absence of attention kernel-count recording. |
| [tests/unit/test_dsv4_graph_measured.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_graph_measured.py:25) | 215 | Removed obsolete isolated-attention tests. Retained graph/source geometry and non-attention coverage/error tests; adapted PIECEWISE expectations to the V2 scope rule. |
| [tests/unit/test_dsv4_predictor.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_predictor.py:269) | 282 | Updated constructor usage and added missing/conflicting threshold and KV-alias tests. |
| [tests/unit/test_dsv4_source_execution.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_source_execution.py:250) | 331 | Removed metadata-flag arguments/case and updated the synthetic HCA total for its newly required store. |
| [tests/unit/test_dsv4_structural_execution.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_structural_execution.py:287) | 453 | Updated boundary-store and context expectations; replaced old dummy-builder assumptions with V2 scope, descriptor, valid-row, serving-field, and graph-pricing checks. |
| [tests/unit/test_profiling_runtime_boundary.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_profiling_runtime_boundary.py:13) | 89 | Set the allowlist to the final modules’ exact AST profiling imports, with the required explanatory comments. |

Deleted exactly these four authorized files because they served the superseded isolated-eager attention path:

- `frontier/execution_time_predictor/dsv4_attention_input.py`
- `tests/comparison/calibration/prepare_dsv4_attention_input.py`
- `tests/integration/check_dsv4_measured_scopes.py`
- `tests/fixtures/deepseek_v4_attention_layouts.json`

There were no tracked changes at the initial inspection. A concurrent external change later appeared in `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/stage1a_replay_route.md`; it was preserved and is not part of this executor’s changes. The existing untracked `.cortexkit/`, `.pi/`, and `remaining_measurements/` were left alone.

Two verification-harness adaptations were needed, without changing runtime interfaces or coverage keys. The synthetic owner fixture now writes complete replay runtime identity and normalizes layer timestamps to sequential execution; the original fixture’s fixed layer offsets produced unequal eager entry gaps for pooled model roles. The archived baseline runner uses a scratch wrapper that supplies only the archival `git rev-parse HEAD` result, because a Git archive has no `.git` directory. Archived source files were not edited.

**3. Verification**

Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6, without torch or vLLM. All checks used timeouts. Tests and analysis used virtual-memory limit 31,457,280 KiB; structural simulations used 62,914,560 KiB. Bytecode, test caches, predictor caches, and generated outputs were directed to scratch or suppressed.

The command payloads below use these shared path substitutions; logging redirections and outer timeout wrappers are omitted from the saved-run payload listings. The final hygiene recheck additionally records its exact subprocess argv and return codes in [final_scope_hygiene.json](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/final_scope_hygiene.json).

```bash
R=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
S=/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8
P=/data/ycfeng/envs/frontier-py310/bin/python
export PYTHONPATH="$R:$R/tests/unit:$R/tests/comparison/calibration:$R/tests/analysis"
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
```

Executed analysis payloads:

```bash
$P "$S/real_runtime.py"
$P "$S/b7_runtime_parity.py"
$P /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_baseline.py "$S/a4_report.json"
cmp "$S/a4_report.json" /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_report_d2a226fe.json
$P -B "$S/hygiene.py"
timeout 30s $P -B "$S/final_scope_hygiene.py"
```

Executed regression selection:

```bash
$P -m pytest -q -ra -p no:cacheprovider --basetemp "$S/pytest_regression" \
  tests/unit/test_dsv4_*.py \
  tests/unit/test_dp_dummy_forward_pricing.py \
  tests/unit/test_forward_sync_state.py \
  tests/unit/test_monolithic_mixed_forward_sync.py \
  tests/unit/test_dsa_predictor.py \
  tests/unit/test_dsa_attention_profiling.py \
  tests/unit/test_profiling_runtime_boundary.py \
  tests/unit/test_kernel_gap_pricing.py
```

The final regression used the same selection with scratch basetemp `"$S/pytest_final"` and only this deselection:

```bash
--deselect tests/unit/test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default
```

Executed structural payloads:

```bash
git archive dd4c4317 --output="$S/before.tar"
tar -xf "$S/before.tar" -C "$S/before"
$P "$S/run_before.py" --dp 4 --case both --output-root "$S/structural_before"
$P tests/integration/run_dsv4_structural.py --dp 4 --case both --output-root "$S/structural_after"
```

The baseline census imported Frontier from the archive using `PYTHONPATH` rooted at `"$S/before"`; the after census used the working-tree environment:

```bash
$P "$S/structural_census.py" "$S/structural_before" "$S/counts_before.json"
$P "$S/structural_census.py" "$S/structural_after" "$S/counts_after.json"
$P "$S/structural_diff.py"
```

**Acceptance 1 — PASS.** All 505 rows match scopes, `indexer_short_context`, `plans_tile_schedule`, and padded compute width. The committed synthetic tests cover eager short/long CSA, both score branches, HCA without a crossing, FULL, PIECEWISE widths 2 and 4, and NONE dummies.

| Phase / mode | Matched rows |
| --- | ---: |
| decode / FULL | 320 |
| decode / PIECEWISE | 80 |
| dummy / FULL | 25 |
| dummy / NONE | 5 |
| dummy / PIECEWISE | 20 |
| first_tail / PIECEWISE | 50 |
| prefill / NONE | 5 |
| Total | 505 |

**Acceptance 2 — PASS.** The sklearn owner test passes in the final regression. It demonstrates direct one-row fitting, unchanged cache-file mtimes, model hashes, and estimator count on the second training, threshold attachment/recovery, FULL decode and FULL/NONE dummy predictions, exact scope costs and signed gaps, composition equivalence to synthetic row medians, and no attention kernel counts.

**Acceptance 3 — PASS.** [real_runtime.py](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/real_runtime.py) exercises the actual sklearn-owned runtime entry with exact lookup and the H200 replica. It trains 554 models using 554 estimator creations.

| Absolute relative error | Observed |
| --- | ---: |
| Median | 0.4909042604% |
| p90 | 1.3427296634% |
| Maximum | 6.6727232002% |
| Rows above 5% | 2 |

Median ≤ 5% and p90 ≤ 10% both pass. The summary has zero delta from B7’s reference. The independent [B7 parity check](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/b7_runtime_parity.json) confirms **all 505 composed runtime values equal the B7 lookup/composition values exactly**.

| Zero-based row | Coordinate | Predicted ms | Measured ms | Error |
| --- | --- | ---: | ---: | ---: |
| 13 | `e0-decode-b1-p6144`, CSA/FULL, follower, R=1, decode context `[6151]` | 0.09087017313639323 | 0.096416015625 | 5.7519930197% |
| 204 | `e0-decode-b4-p6144`, CSA/FULL, planner, R=4, decode contexts `[6150,6151,6152,6153]` | 0.12353565979003907 | 0.11580810546875 | 6.6727232002% |

These are the same above-5% rows as B7. Saved per-scope deltas show the largest residual contributions in the pooled compressor-store costs/gaps; no keys, features, or model thresholds were tuned. Full evidence is in [real_runtime.json](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/real_runtime.json) and [real_runtime.log](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/real_runtime.log).

Relevant output tails:

```text
"scope_mismatches": 0
"models": 554
"estimators": 554
"limits_passed": true
"reference_above_rows_equal": true

B7 parity:
"rows": 505
"differing_rows": []
"all_rows_equal": true
```

**Acceptance 4 — PASS.** A4 returned `PASS 125`; `cmp` returned 0 with no output. The generated [a4_report.json](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/a4_report.json) is byte-identical to `a4_report_d2a226fe.json`. The comparison was also rechecked after the final source changes.

**Acceptance 5 — PASS.** The runtime import-boundary tests pass. [profiling_imports.json](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/profiling_imports.json) lists every one of the 32 predictor modules. Eight have profiling imports; the other 24 have none:

| Predictor module | Profiling imports |
| --- | --- |
| `deepseek_v4.py` | `frontier.profiling.common.runtime_identity` |
| `dsv4_predictor.py` | `frontier.profiling.attention.dsv4_attention_rows`, `frontier.profiling.attention.dsv4_replay_rows` |
| `dsv4_composition.py` | `frontier.profiling.kernel_gap.schema` |
| `kernel_gap.py` | `frontier.profiling.kernel_gap.schema` |
| `measurement_input_paths.py` | `frontier.profiling.cpu_overhead.schema` |
| `prediction_family_trainers.py` | `frontier.profiling.cpu_overhead.validation` |
| `profiling_dataframe_loaders.py` | `frontier.profiling.cpu_overhead.validation` |
| `sklearn_execution_time_predictor.py` | `frontier.profiling.cpu_overhead.schema`, `frontier.profiling.cpu_overhead.validation`, `frontier.profiling.other_overhead.validation` |

**Acceptance 6 — PASS.** `tests/unit/test_dsv4_step_cost_input.py` passes in the final regression and was not edited.

**Acceptance 7 — PASS.** Both archived-before and working-tree-after structural runs exit 0 for offline and online. Their scope and system-metric difference lists are identical. The idle per-layer totals are HCA 9→11, CSA 13→16, and uncompressed 8→8, as expected.

Every changed scope-count entry is listed below. Counts apply separately to each offline/online run. “Scheduler” counts deduplicate EP destinations for each source batch/layer; “physical” counts include the eight physical destination ranks. Real counts are independently corroborated by attention traces. Idle counts are reconstructed from the recorded dummy ledger geometry because idle attention emits no serving trace.

| Lane / variant | Scope | Scheduler before→after | Physical before→after |
| --- | --- | ---: | ---: |
| idle / CSA | `dsv4_attention_output_zero` | 2436→0 | 19488→0 |
| idle / CSA | `dsv4_compressor_fused_compress_store` | 0→2436 | 0→19488 |
| idle / CSA | `dsv4_compressor_save_partial_states` | 0→2436 | 0→19488 |
| idle / CSA | `dsv4_fused_q_kv_prepare` | 0→2436 | 0→19488 |
| idle / CSA | `dsv4_index_compressor_fused_compress_store` | 0→2436 | 0→19488 |
| idle / CSA | `dsv4_index_compressor_save_partial_states` | 0→2436 | 0→19488 |
| idle / CSA | `dsv4_index_fused_query_prepare` | 2436→0 | 19488→0 |
| idle / CSA | `dsv4_index_query_projection` | 2436→0 | 19488→0 |
| idle / CSA | `dsv4_index_short_context_fill` | 0→2436 | 0→19488 |
| idle / CSA | `dsv4_query_padding` | 2436→0 | 19488→0 |
| idle / CSA | `dsv4_sparse_attention` | 0→2436 | 0→19488 |
| idle / HCA | `dsv4_attention_output_zero` | 2320→0 | 18560→0 |
| idle / HCA | `dsv4_compressor_fused_compress_store` | 0→2320 | 0→18560 |
| idle / HCA | `dsv4_compressor_save_partial_states` | 0→2320 | 0→18560 |
| idle / HCA | `dsv4_fused_q_kv_prepare` | 0→2320 | 0→18560 |
| idle / HCA | `dsv4_query_padding` | 2320→0 | 18560→0 |
| idle / HCA | `dsv4_sparse_attention` | 0→2320 | 0→18560 |
| idle / uncompressed | `dsv4_attention_output_zero` | 232→0 | 1856→0 |
| idle / uncompressed | `dsv4_fused_q_kv_prepare` | 0→232 | 0→1856 |
| idle / uncompressed | `dsv4_query_padding` | 232→0 | 1856→0 |
| idle / uncompressed | `dsv4_sparse_attention` | 0→232 | 0→1856 |
| real / HCA | `dsv4_compressor_fused_compress_store` | 0→240 | 0→1920 |

All other scope counts are unchanged, including real CSA and real uncompressed scopes. The eager idle lanes now execute short-context CSA scopes; the warmup-only padding/zero scopes disappear from all dummy variants. Real HCA gains its unconditional store even without a boundary crossing.

There are 29 changed system-metric fields per case. The latency rows below each apply to all eight statistics: `max`, `mean`, `median`, `min`, `p50`, `p90`, `p95`, and `p99`.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| `ttft_statistics.*`, ms | 19.76059264000007 | 20.160592640000058 | +0.4 |
| `tpot_statistics.*`, ms | 9.699864213333399 | 9.899864213333391 | +0.2 |
| `request_e2e_time_statistics.*`, ms | 29.46045685333347 | 30.06045685333345 | +0.6 |
| `throughput_metrics.decode_tokens_per_second` | 271.5504392829806 | 266.1303532089489 | −5.420086074031701 |
| `throughput_metrics.requests_per_second` | 135.7752196414903 | 133.06517660447446 | −2.7100430370158506 |
| `throughput_metrics.tokens_per_second` | 2579.7291731883156 | 2528.238355485015 | −51.49081770330076 |
| `throughput_metrics.total_duration_ms` | 29.46045685333347 | 30.06045685333345 | +0.6 |
| `throughput_metrics.total_duration_seconds` | 0.02946045685333347 | 0.03006045685333345 | +0.0006 |

Each real forward gains 20 HCA store calls at 0.01 ms per call. The two prefill forwards therefore add 0.4 ms to TTFT, and the decode forward adds 0.2 ms. Throughput changes follow the longer simulated duration.

Three driver-summary fields differ per case: `run_dir` points to the respective before/after output root; `trace_rows` increases 481144→481232 (+88); and `max_stage_conservation_error_ms` changes from `1.0000018590972104e-09` to `1.0000009709187907e-09` (−8.881784197001252e-16, consistent with floating arithmetic). The trace increase consists of 80 expanded first-forward HCA store events and eight later aggregate store records.

All remaining driver-summary fields are unchanged: four completed requests, 12 stage rows, 1,376 waves, 4,128 real EP lanes, 39,904 idle EP lanes, cache-zero counts/reservation extent, and boundary geometry. Phase conservation remains `6.938893903907228e-18` ms. Native GPU assignments, physical cache retirement, and GPU latency remain unverified by this CPU structural harness.

Complete evidence: [structural_diff.json](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/structural_diff.json), [before report](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/structural_before/report.json), [after report](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/structural_after/report.json), `counts_before.json`, `counts_after.json`, and their census/diff scripts under the scratch root.

**Acceptance 8 — PASS with the explicitly allowed failure.** The unfiltered regression returned:

```text
AttributeError: 'OperatorSpec' object has no attribute 'kernel_count'
FAILED tests/unit/test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default
1 failed, 1255 passed, 8 skipped in 54.62s
```

The final run, deselecting only that known failure, exited 0:

```text
1255 passed, 8 skipped, 1 deselected in 55.88s
```

The eight skips are one V4 linear-profiling test and seven V4 MoE-profiling tests requiring unavailable torch. No new failures remain. Logs: [regression.log](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/regression.log) and [final_regression.log](/data/ycfeng/tmp/codex-implement/bb6c54ec-e30e-4efd-928b-36dce74bd0b0/b8/final_regression.log).

Earlier diagnostic runs exposed and corrected three test-expectation/fixture issues: the newly required HCA store changed an expected synthetic total from 20 to 22 ms; the first owner fixture lacked the replay `schema_version` identity column; and its fixed layer offsets produced unequal pooled eager gaps. Their logs remain in `focused.log`, `owner.log`, and `owner2.log`. The final regression covers the corrected files. Runtime keys/features were not changed to accommodate these fixture failures.

**Acceptance 9 — PARTIAL.** All changed modules except the explicitly exempt sklearn module are below 2,000 lines. `git diff --check` exits 0. Cold imports of all three requested modules exit 0, with output:

```text
{"module": "frontier.execution_time_predictor.deepseek_v4", "native_modules": []}
{"module": "frontier.execution_time_predictor.dsv4_predictor", "native_modules": []}
{"module": "frontier.attention.deepseek_v4", "native_modules": []}
```

The supplied `/data/ycfeng/tmp/unused_imports.py` returns `[]` for every changed file except sklearn. For sklearn it reports the same five names in both the archive and final source:

```text
['BaseCCBackend', 'EPLaneWorkload', 'ExecutionTimePredictionModelManager', 'MEMORY_FAMILY', 'product']
```

The first three are used in quoted type annotations, which this scanner does not recognize. `MEMORY_FAMILY` and `product` are genuinely unused baseline imports. The scanner exits 0 regardless of findings; the literal “no unused imports” condition therefore remains unmet. Removing unrelated sklearn imports is outside the allowed registration edits.

The required command `/usr/bin/grep -rn 'has_attention_metadata\|build_deepseek_v4_dummy_attention_workload\|_predict_dsv4_dummy_attention_time\|build_isolated_prefill_layouts\|dsv4_attention_input' frontier tests` returns binary matches only in these retained bytecode files:

```text
frontier/attention/__pycache__/deepseek_v4.cpython-310.pyc
frontier/execution_time_predictor/__pycache__/sklearn_execution_time_predictor.cpython-310.pyc
frontier/execution_time_predictor/__pycache__/deepseek_v4.cpython-310.pyc
frontier/execution_time_predictor/__pycache__/dsv4_attention_input.cpython-310.pyc
tests/integration/__pycache__/check_dsv4_measured_scopes.cpython-310.pyc
tests/unit/__pycache__/test_dsv4_structural_execution.cpython-310-pytest-9.1.1.pyc
tests/unit/__pycache__/test_dsv4_attention_workload.cpython-310-pytest-9.1.1.pyc
tests/unit/__pycache__/test_dsv4_graph_measured.cpython-310-pytest-9.1.1.pyc
tests/unit/__pycache__/test_dsv4_source_execution.cpython-310-pytest-9.1.1.pyc
```

Both `rg -n` and `/usr/bin/grep -Irn` with the same symbols return exit 1 and no matches: the obsolete source symbols are gone. The stale caches were not deleted because only the four named source/fixture deletions were authorized. Exact argv, outputs, baseline comparison, and line counts are captured in `final_scope_hygiene.json`; cold-import output is in `hygiene.log`.

**4. Remaining work and decisions for Claude**

No functional B8 implementation blocker remains within the assigned scope. To satisfy literal Acceptance 9, Claude should either authorize/perform targeted cleanup of the two genuine unused baseline sklearn imports and the nine stale bytecode files, while accounting for quoted-annotation scanner false positives, or explicitly accept a baseline/source-only hygiene exception. The evidence distinguishes those baseline findings from new imports or surviving source references.

There was no material runtime design deviation. The synthetic timestamp normalization and archival commit-identity wrapper are bounded verification adaptations described above. Full serving/PD/manager admission, native structural composition, kernel-gap integration, mixed trained decode/prefill support, and removal of registered warmup operators remain later work, not unfinished B8 edits. Task records, commits, and acceptance remain Claude’s responsibility.

**5. Review handoff**

Review the unified workload builder’s dummy geometry, unconditional stores, CSA score condition, and planner flag; the shared KV alias and model threshold attributes; the new runtime pricing function’s costs/gaps/composition inputs; and the narrowly scoped sklearn registration/one-row branch. Confirm the four authorized deletions and that the external task-record diff is excluded from the B8 commit.

For acceptance reruns, use the regression selection above with only the known kernel-gap failure deselected, then `real_runtime.py`, `b7_runtime_parity.py`, A4 plus `cmp`, and both structural cases if additional implementation changes are made. Rerun the literal hygiene commands after Claude resolves their scope decision. The scratch scripts and logs are retained for reproducibility; no GPU/native latency acceptance is claimed.

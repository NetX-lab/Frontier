# Affected CORE fixture validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive five-test RED, complete corrected fixtures, full coherent union, exact commands and ownership accounting. |
| 2026-10-23 | Seal scoped verified test commit f03a1157 without additional checks or code changes. |

TaskID: **D29-CORE-AFFECTED-FIXTURES**. Worktree `/data/ycfeng/Frontier/.worktrees/step5-core-review-fixes`; branch feat/step5-core-review-fixes; basee49b26f8. Environment `/data/ycfeng/envs/frontier-py310/bin/python`,Python3.10.6. Every test command uses bytecode disabled,OMP/OPENBLAS threads1,TMPDIR/data/ycfeng/tmp,PYTHONPATH repo and existing tests/unit. No package/dependency installations or GPU/platform/network/CLI operations.

## Expected behavior and observed results

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Current five-test RED, after authorized registry merge before fixture edit | Same five failure modes as root logs | 5FAIL,0PASS,2.50s | Expected RED |
| Initial fixture correction | Source consumers repaired; runtime helper wiring verified | 3PASS/2FAIL,24.11s; nonexistent test helper name | Intermediate failure retained |
| Initial full affected files | All assertions pass | 28PASS,32.20s | PASS |
| Initial coherent full union | No new failures or exclusion | 221PASS/7SKIP,196.13s | PASS, intermediate version |
| Source Batch/forward/materializer completion | Full padded file passes with real derived descriptor/census | 20PASS,7.39s | PASS |
| Final two full affected files | All tests, including original five failures, pass | 28PASS,32.13s;0FAIL/ERROR/SKIP | PASS |
| Final entire coherent seven-file union | Existing producer/registry/trainer/consumer selection has no failure | 221PASS/7SKIP,186.10s;0FAIL/ERROR | PASS |
| Effective grouped caps/AST/whitespace | <=80repair;runtime<=110,padded<=210,CORE<=4404 | 72/80;53/110,197/210;3833/4404 | PASS |

Overlapping suites are not summed. Seven existing union skips remain; no skip/exclusion was added by this repair. Complete affected files have zero skips. The union uses no `-k` exclusion or explicit-test deselection. Existing BF16/deferred prepare applicability and torch-limited CPU cases do not establish native execution; independent root full-unit/strict77/source/native/E2E gates remain separate.

## Exact RED reproduction

Authorized immutable registry source was merged without peer edits:

```bash
git merge --no-ff --no-edit a5189e199208022a341b377b0fb8c542030907f3
```

Result: mergeacea99e4, no conflicts; original registry history retained. Independent registry acceptance remains root-owned.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_routing_runtime.py::test_position_runtime_binds_retained_source_and_full_plan_policy tests/unit/test_moe_grouped_gemm_padded_features.py::test_selected_sources_reach_real_preflights_loader_and_both_trainers_before_old_helpers tests/unit/test_moe_grouped_gemm_padded_features.py::test_independent_selected_actual_fit_cache_binds_native_context tests/unit/test_moe_grouped_gemm_padded_features.py::test_selected_output_inventory_uses_declared_cells_without_structural_zero_fill > /data/ycfeng/tmp/d29-core-fixtures-red01.log 2>&1
```

Actual failures:
- `test_position_runtime_binds_retained_source_and_full_plan_policy[None]`
- `test_position_runtime_binds_retained_source_and_full_plan_policy[foreign1]`

Both: `AttributeError: 'types.SimpleNamespace' object has no attribute 'model_type'`, at real routed-group resolution, before point recovery.

- `test_selected_sources_reach_real_preflights_loader_and_both_trainers_before_old_helpers`
- `test_independent_selected_actual_fit_cache_binds_native_context`
- `test_selected_output_inventory_uses_declared_cells_without_structural_zero_fill`

All three: `ValueError: MoE physical scope metadata missing ['routing_input']` at existing admission. No assertion or source validator was weakened.

## Corrected fixture mechanisms

Source rows now create actual Batch or DummyForwardBatch, bind actual existing forward agreement with explicit graph policy/capture size8, freeze through build_wave_routing_source and inspect existing materialize output. Canonical MoESourceFrame queries/context and physical width derive from immutable source; profiling runtime context stays an explicit fixture input. MoERoutingProfileInput/Cell serialize the existing protocol; local physical/live/masked counts and IDs derive from those objects. Full zero-live width8 and live4+masked4 cases remain. Actual source/materializer live census is asserted. All shared/independent preflight, fit/count pair, cold/warm cache, context/hash, precision and cell-inventory assertions remain unchanged.

Runtime test imports the existing real `bound` fixture. It fits both actual routed groups with formal ReplicaConfig/model and existing shared trainer into the real registry, then uses the existing shared predictor. Step layer metadata88 and its resolved group select actual BF16 retained physical rows. Source materializer yields actual local16/dummy16 gathered32, matching the producer's forward-bound layout. Existing position_collective_costs writer creates explicit CPU artifact keys with full context and buffer/count receipts. Real grouped-feature/model selection is no longer replaced by a lambda. Tests assert actual native context, source-local features, every measured plan context, complete buffer compute_context, existing model marker/physical rows. Original foreign-context and absent-point rejection assertions remain.

Numerical and collective values remain explicitly synthetic CPU fixture inputs, not native observations or admission.

## Iterations and exact final commands

Initial focused correction ran the exact RED test list into `d29-core-fixtures-green01.log`:3PASS/2FAIL24.11s. The only remaining error was nonexistent `_describe_step5_layer`; actual helper is step5_layer_metadata. Subsequent change reads this real helper, resolves the model's group and retains assertions; no default/fallback was introduced.

Initial full files/union commands below first ran to green01/union01 logs with480s tool limits. Completing the requested source producer/materializer path then ran the full padded file:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_grouped_gemm_padded_features.py > /data/ycfeng/tmp/d29-core-fixtures-bound-source01.log 2>&1
```

Final version, exact commands (tool limits480s, no CLI limit changes):

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_grouped_gemm_padded_features.py > /data/ycfeng/tmp/d29-core-fixtures-full-green02.log 2>&1
env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_prediction_measurement_bindings.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_moe_pricing.py tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/integration/test_step5_measured_main.py > /data/ycfeng/tmp/d29-core-fixtures-coherent-union02.log 2>&1
git diff --check
git diff --numstat e49b26f8 -- tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_grouped_gemm_padded_features.py
```

Final outputs: `28 passed in 32.13s`; `221 passed, 7 skipped in 186.10s`. No unresolved failures.

## Accounting and limits

Final exact diff:affected_fixtures.diff. Budget:budget.json. Apply root's live registry_scope_amendment then updated core_review_scope_amendment; no root file copied or rewritten. Current two-test patch+45/-27=72/80,8remaining. Original seven-path review228/230 unchanged. Runtime53/110 andpadded197/210 relative1aee0af;allothergroupedcaps andASTPASS. Combined effectiveCORE3833/4404,571remaining;46allocatedCOREpaths within109path union and10000globalceiling. Root-owned collective-sim checkout3dac remains modified/unstaged and excluded from owned commit.

Current source/test gates pass for this scoped fixture repair. Root independently reviews/integrates immutable result, manages gitlink and executes full-unit/strict77/freshactual-main/native/GPU/global acceptance. No new independent public mode, registry API or production behavior is supported by this repair.

Immutable source/tests/records commit:f03a1157. Code/tests exactly match the final full-file and full-union commands above. Post-commit status preserves only root-owned collective-sim gitlink; source path check and whitespace remain PASS.

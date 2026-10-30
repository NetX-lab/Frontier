**1. Assignment and outcome**

Implemented B27 / I-002 on October 30, 2026: update `test_selected_producer_rows_train_and_reuse_distinct_cache` to the existing DeepSeek-V4 kernel-only measurement contract. The required CPU pytest run completed with **128 passed, 0 failed, 0 skipped**. Final review, acceptance, task-record updates, and commits remain with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/b27-balanced-routing-v4-test`. Initial HEAD: `b1cf3447d523465337aa9c4fae96256fe20dbcdd`. Initial `git status --short` was empty; the test had no pre-existing edits. The named `.cortexkit/`, `.pi/`, and `remaining_measurements/` directories were left untouched.

Allowed repository writes were limited to:

- `tests/unit/test_balanced_routing.py`;
- `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b27/` for this new `report.md`.

Scratch outputs are under `/data/ycfeng/tmp/codex-implement/7653c18e-0ac9-4ec6-bd74-0044d43e038d/b27`. No file under `frontier/` was modified. No Git state mutation, task-record update, dependency installation, network access, GPU operation, job submission, or interaction with the running A10 simulations occurred.

Read the identical parent and worktree `AGENTS.md` files, the named B27 assignment artifact, the read-only main-tree `issues.md` I-002 entry, the complete test module, and the relevant predictor measurement/view code. No applicable `CLAUDE.md` or nested instruction file was found under the test/report paths. The assignment's focused CPU/test-only acceptance governs this change; the broader simulator scenario matrix was not run.

Settled behavior: retain both routing-order parametrizations, separate cache identities per routing path, exact predictions 2.0 and 7.0, the warm lookup's no-fit guard, and both wrong-row `ValueError` checks matching `routing.*path`. Use explicit `KERNEL_ONLY` fixture values and the `kernel_only` cluster view, and add an empty-`eager` assertion for each V4 role exercised by this test (`PREFILL`, `DECODE`). No new interface or scope decision was needed.

Observed contract evidence: `frontier/execution_time_predictor/dsv4_measurement.py:9` returns `MeasurementType.KERNEL_ONLY` for V4; `shared_prediction_model_manager.py:248` limits its cluster measurement types accordingly; `shared_prediction_model_manager.py:598` returns an empty event-family view when that measurement type is excluded. `prediction_model_registry.py:194` rejects fixture/active-measurement mismatches, so both profiling fixtures and the manager must change together.

**2. Changes and line-by-line reasons**

Only the named test function changed. The test diff contains five replaced lines and one added line (6 insertions, 5 deletions). All 11 previous Python `assert` statements remain, with the two cluster-view assertions redirected to the required family; there are now 12. No skip, `xfail`, test removal, assertion weakening, or unrelated formatting change was introduced.

| Original line | Current line | Change | Reason |
| --- | --- | --- | --- |
| 285 | 285 | MoE row `measurement_type`: `CUDA_EVENT` → `KERNEL_ONLY` | Train both routing producers from the V4 measurement family. Typed contracts, routing metadata, and values 2.0/7.0 are unchanged. |
| 296 | 296 | Linear/shared-expert row `measurement_type`: `CUDA_EVENT` → `KERNEL_ONLY` | The same FFN training call also loads these operators; its fixture must match the active V4 measurement type. |
| 301 | 301 | `_active_measurement_type`: `MeasurementType.CUDA_EVENT` → `MeasurementType.KERNEL_ONLY` | Register and retrieve the trained models within the V4 family and satisfy active-measurement validation. |
| 316 | 316 | Cold cluster-view identity assertion: `eager` → `kernel_only` | Assert that the cluster selects the exact cold model in the supported V4 family, preserving the `is cold[role]` identity check. |
| — | 317 | Add `assert manager.get_models_for_cluster(role)["eager"] == {}` | Explicitly assert the V4 contract for each configured role, in both routing orders. |
| 324 | 325 | Warm cluster-view identity assertion: `eager` → `kernel_only` | Assert that cache reuse exposes the exact warm model through the V4 family, preserving the `is warm[role]` identity check. |

New file: this `report.md`, which records scope, evidence, every test line change, the diff, commands/results, and review handoff. No pre-existing report was overwritten.

Exact test diff:

```diff
diff --git a/tests/unit/test_balanced_routing.py b/tests/unit/test_balanced_routing.py
index 1f4e2aed..1c2fe427 100644
--- a/tests/unit/test_balanced_routing.py
+++ b/tests/unit/test_balanced_routing.py
@@ -282,7 +282,7 @@ def test_selected_producer_rows_train_and_reuse_distinct_cache(tmp_path, order):
                       num_tensor_parallel_workers=1, expert_parallel_size=1)
     features = MoELoadImbalanceInput(8, model.num_experts, model.embedding_dim,
                                     model.routed_mlp_hidden_dim, model.num_experts_per_tok).to_features_dict()
-    rows = pd.DataFrame([dict(dimensions, num_tokens=8, profiling_precision="FP16", measurement_type="CUDA_EVENT",
+    rows = pd.DataFrame([dict(dimensions, num_tokens=8, profiling_precision="FP16", measurement_type="KERNEL_ONLY",
                              gating_runtime_context="standalone_legacy", gating_runtime_context_impl="none",
                              model_architecture_profile=profile.profile_id,
                              typed_operator_contracts=serialize_typed_operator_contracts(metadata),
@@ -293,12 +293,12 @@ def test_selected_producer_rows_train_and_reuse_distinct_cache(tmp_path, order):
     rows.to_csv(source, index=False)
     shared_width = profile.resolve_layer_contract(model, operator_name="share_expert_up_proj", ffn_tp_size=1).width
     pd.DataFrame([dict(num_tokens=8, num_tensor_parallel_workers=1, n_expanded_embd=shared_width,
-        profiling_precision="FP16", measurement_type="CUDA_EVENT",
+        profiling_precision="FP16", measurement_type="KERNEL_ONLY",
         **{f"time_stats.{name}.median": 1.0 for name in ["post_attention_layernorm",
             *get_family_profiling_names(SHARE_EXPERT_FAMILY)]})]).to_csv(linear, index=False)
     manager = ExecutionTimePredictionModelManager({}, SimpleNamespace(cache_dir=str(tmp_path)))
     manager._cluster_configs, manager._all_dummy_mode = configs, False
-    manager._active_measurement_type = MeasurementType.CUDA_EVENT
+    manager._active_measurement_type = MeasurementType.KERNEL_ONLY
     config, signatures = RandomForrestExecutionTimePredictorConfig(num_estimators=[1], max_depth=[1], min_samples_split=[2]), set()
     def train(role):
         return manager._train_ffn_models_for_cluster(role, configs[role].replica_config, config,
@@ -313,7 +313,8 @@ def test_selected_producer_rows_train_and_reuse_distinct_cache(tmp_path, order):
         selected = manager.get_model(op, layer_contract=contracts[role])
         assert selected.predict(pd.DataFrame({"num_tokens": [8]}))[0] == {"standard_fused_topk": 2.0, "balanced_topk": 7.0}[path]
         assert manager.get_model(op, layer_contract=contracts[role]) is cold[role]
-        assert manager.get_models_for_cluster(role)["eager"][op] is cold[role]
+        assert manager.get_models_for_cluster(role)["kernel_only"][op] is cold[role]
+        assert manager.get_models_for_cluster(role)["eager"] == {}
         assert cold[role].predict(pd.DataFrame({"num_tokens": [8]}))[0] == {"standard_fused_topk": 2.0, "balanced_topk": 7.0}[path]
     manager._create_estimator_and_params = lambda *_: pytest.fail("Warm lookup must not fit")
     signatures.clear()
@@ -321,7 +322,7 @@ def test_selected_producer_rows_train_and_reuse_distinct_cache(tmp_path, order):
     for role in configs:
         assert warm[role]._frontier_model_hash == cold[role]._frontier_model_hash
         assert manager.get_model(op, layer_contract=contracts[role]) is warm[role]
-        assert manager.get_models_for_cluster(role)["eager"][op] is warm[role]
+        assert manager.get_models_for_cluster(role)["kernel_only"][op] is warm[role]
         assert warm[role].predict(pd.DataFrame({"num_tokens": [8]}))[0] == cold[role].predict(pd.DataFrame({"num_tokens": [8]}))[0]
     for wrong in [rows.iloc[:1], rows.drop(columns="routing_runtime_path")]:
         wrong.to_csv(source, index=False)
```

**3. Verification**

Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python **3.10.6**. Pytest **9.1.1**. Torch imports from `/usr/local/lib/python3.10/dist-packages/torch/__init__.py`, version **2.1.0+cu121**, compiled for CUDA **12.1**. This is a CUDA-capable wheel exercised with CPU tensors; no GPU operation was invoked. The required source root was `/data/ycfeng/tmp/opencode/vllm-v0.28.0`. Each shell command used a timeout and `ulimit -v 31457280`; pytest had its required 900-second timeout, within a 930-second outer shell timeout. Bytecode and pytest cache-provider writes were disabled.

Before: **FAIL, as expected**. Executed in the working root with this inner-shell command (outer wrapper: `timeout 930 /bin/zsh -f -c`):

```bash
ulimit -v 31457280
env PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0 \
  PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages \
  timeout 900 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_balanced_routing.py -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/codex-implement/7653c18e-0ac9-4ec6-bd74-0044d43e038d/b27/pytest \
  > /data/ycfeng/tmp/codex-implement/7653c18e-0ac9-4ec6-bd74-0044d43e038d/b27/before.txt 2>&1
result=$?
tail -8 /data/ycfeng/tmp/codex-implement/7653c18e-0ac9-4ec6-bd74-0044d43e038d/b27/before.txt
exit $result
```

Observed exit code **1** and summary `2 failed, 126 passed in 16.03s`. Both `[order0]` and `[order1]` failed with `KeyError: 'moe_gating_routing_topk'` at the old line 316 (`eager` cold-model identity assertion). This reproduces the lead's inspected `/data/ycfeng/tmp/i002_before.txt` summary (`2 failed, 126 passed in 17.45s`) on this assignment's actual worktree.

After: **PASS**. Executed in the same working root/environment with this inner-shell command (same outer wrapper):

```bash
ulimit -v 31457280
env PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0 \
  PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages \
  timeout 900 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_balanced_routing.py -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/codex-implement/7653c18e-0ac9-4ec6-bd74-0044d43e038d/b27/pytest \
  > /data/ycfeng/tmp/codex-implement/7653c18e-0ac9-4ec6-bd74-0044d43e038d/b27/after.txt 2>&1
result=$?
cat /data/ycfeng/tmp/codex-implement/7653c18e-0ac9-4ec6-bd74-0044d43e038d/b27/after.txt
exit $result
```

Observed exit code **0**, summary `128 passed in 18.13s`, and no failures, skips, or warnings reported. Pytest rebuilt its `--basetemp` in authorized scratch; the baseline log was kept separately. Both routing orders completed the cold and warm checks and both wrong-row cases.

Assertion/change audit: **PASS**, executed from the working root under a 30-second shell timeout and the same memory limit:

```bash
env PYTHONDONTWRITEBYTECODE=1 timeout 20 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/codex-implement/7653c18e-0ac9-4ec6-bd74-0044d43e038d/b27/audit.py
```

The scratch audit reads `git show HEAD:tests/unit/test_balanced_routing.py`, compares the final file to exactly the five prescribed replacements plus the one assertion, and checks prior assertions through Python ASTs. Exit code **0**; output preserved in `/data/ycfeng/tmp/codex-implement/7653c18e-0ac9-4ec6-bd74-0044d43e038d/b27/audit.txt`:

```text
PASS: only the five required line replacements and one added assertion changed.
PASS: all 11 prior assert statements preserved; 12 after.
PASS: routing-order parameters, warm-fit guard, wrong-row checks, and existing skips unchanged.
PASS: the rest of the module is byte-for-byte unchanged.
```

Diff and scope checks: **PASS**, executed under a 30-second shell timeout and the same memory limit:

```bash
git diff --check
git diff --stat
git status --short --untracked-files=all
git status --short --untracked-files=all --ignored -- tests/unit/test_balanced_routing.py task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b27/report.md
git diff --no-index --stat -- /dev/null task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b27/report.md
```

`git diff --check` returned **0** with no output. Default `git diff --stat` lists only `tests/unit/test_balanced_routing.py`, with **6 insertions and 5 deletions**. The new report is ignored by the pre-existing `.gitignore:171` rule `task_memory/*`; normal status/stat therefore omit it. The targeted status command with `--ignored` displays `M` for the test and `!!` for this report. The explicit no-index report stat displays only `report.md` (exit **1** is expected for a new-file diff). Only these two allowed repository files were written. The requested combined two-file default stat is unavailable without an index change, which this assignment reserves to Claude. Claude may force-include the report in its own staging workflow and inspect a combined staged stat; no ignore rule or index was changed here.

One initial edit-command wrapper failed with `zsh:1: = not found` before modifying the file. A corrected heredoc wrapper applied the intended patch; no implementation iteration or production-code workaround was required.

Broader simulator tests and GPU checks: **NOT_RUN**, outside this CPU/test-only assignment. No unresolved required-check failure remains.

**4. Remaining work and decisions for Claude**

No implementation blocker or missing prerequisite remains. No change under `frontier/` was needed. The observed before/after results support the issue's diagnosis that the fixture/view contract was stale; they do not claim new simulator performance or numeric-fidelity evidence.

Claude retains review, acceptance, staging/commit, and I-002/task-record updates. The report is currently ignored by the existing `task_memory/*` rule. If Claude wants a combined tracked two-file stat or to commit the report, it must force-include the new report in its own staging workflow; this executor did not mutate the index. This is a reporting/staging limitation, not a test or implementation blocker. No further writable path is requested.

**5. Review handoff**

Inspect the six test line changes above: both profiling fixtures, the manager's active measurement type, cold/warm `kernel_only` identity assertions, and the new empty-`eager` assertion. Verify that the routing-order parameters, typed cache-identity checks, two distinct model hashes, exact 2.0/7.0 predictions, warm estimator failure guard, and wrong-row rejection loop are intact.

Rerun the after pytest command with the specified torch/source/PYTHONPATH environment and scratch `--basetemp`; expected result is **128 passed**. Rerun `git diff --check` and inspect the test/report scope. The existing module-level torch import skip and pinned-source requirements remain unchanged, so a run without the specified environment does not demonstrate this acceptance condition. Final acceptance remains with Claude.

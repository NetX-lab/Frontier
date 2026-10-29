**1. Assignment and outcome**

B23 implements D-ROUTING-ORIGIN-PROVENANCE: accept balanced-routing descriptors from different packages when every descriptor passes `routing_columns` and their identities are equal after removing only the top-level `origin`.

Working root: `/data/ycfeng/Frontier/.worktrees/b23-routing-origin`.

Allowed source paths, and the only source files changed:

- `frontier/execution_time_predictor/dsv4_predictor.py`, restricted to `validate_balanced_routing`.
- `tests/unit/test_dsv4_predictor.py`.
- `tests/unit/test_v4_attention_promotion.py`.

Scratch root: `/data/ycfeng/tmp/codex-implement/ce6d9226-b4d3-4e5b-84de-01a40f3c9e20/b23/`.

Read the applicable repository AGENTS.md instructions, the B23 assignment, requirements items 37, 39 and 40 in the main task tree, the B22 assignment and report, and the post-job script and failure log. Preserved the settled decisions: runtime path and strategy digest remain single-valued; every descriptor still requires a valid package origin; nested native-source origins remain identity; package reducers remain strict; measured CSV descriptors are retained unchanged.

Outcome is **partial** because acceptance 4 cannot be completed within the supplied environment and write scope. The implementation and acceptance checks 1, 2, 3 and 5 are complete and pass. The required test run has one filesystem prerequisite failure and one missing-Torch module skip. Final review, acceptance, commits and task records remain with Claude.

Initial `git status --short` was clean. No pre-existing tracked edits were encountered. No source files were added. No commits, state-changing Git commands, dependency installations, GPU operations, network operations or writes under `data/profiling/` were performed.

**2. Changes**

| Changed file | Relevant symbols/lines | Change and reason |
| --- | --- | --- |
| [dsv4_predictor.py](/data/ycfeng/Frontier/.worktrees/b23-routing-origin/frontier/execution_time_predictor/dsv4_predictor.py:78) | `validate_balanced_routing` | Separates descriptor presence from the existing runtime-path/digest uniqueness checks. Parses and validates each distinct descriptor, removes its top-level origin from the temporary parsed object, and compares identities using the existing `canonical_json`. Preserves the dataframe and existing error strings. |
| [test_dsv4_predictor.py](/data/ycfeng/Frontier/.worktrees/b23-routing-origin/tests/unit/test_dsv4_predictor.py:336) | `balanced_routing`, `write_routing`, new balanced-loader tests; `aggregate` at line 67 | Adds a validated synthetic routing fixture and 11 parametrized regression cases: one/two package origins accepted with provenance and existing columns preserved; valid descriptors differing in native digest, native origin or router inventory rejected; missing column/value rejected; malformed JSON, missing origin, wrong origin suffix and wrong strategy hash rejected. Extends the existing aggregation helper to carry routing in the raw engine record. |
| [test_v4_attention_promotion.py](/data/ycfeng/Frontier/.worktrees/b23-routing-origin/tests/unit/test_v4_attention_promotion.py:55) | `test_synthetic_supplement_merge_loads_with_different_package_origin` | Adds a synthetic two-package merge regression using the real reducers, supplement raw-copy filter and CSV union helper. The supplement engine records a different package origin. The merged loader result has one fingerprint, retains two descriptors and equals the concatenated independently loaded package tables, including costs and gaps. |

The checked-in B22 promotion test contained repeat filtering only; its synthetic full merge was recorded as a lead check in the B22 report. Added the merge regression beside the existing repeat-filter test using existing production helpers and synthetic records. This is a test-structure choice, with no production interface or scope change.

The production diff is confined to the assigned function:

```diff
@@ -78,19 +78,27 @@
 def validate_balanced_routing(rows, *, require_descriptor=True):
-    """Select the measured balanced routing identity when the replica requests it."""
-    columns = ("routing_runtime_path", "routing_strategy_sha256")
-    if require_descriptor:
-        columns += ("moe_routing",)
-    for name in columns:
+    """Validate routing identity while retaining each package's origin provenance."""
+    from frontier.profiling.common.runtime_identity import canonical_json
+
+    for name in ("routing_runtime_path", "routing_strategy_sha256"):
         if name not in rows or rows[name].isna().any() or len(rows[name].unique()) != 1:
             raise DeepSeekV4CoverageError(f"V4 balanced replay attention missing/ambiguous {name}")
+    if require_descriptor and ("moe_routing" not in rows or rows.moe_routing.isna().any()):
+        raise DeepSeekV4CoverageError("V4 balanced replay attention missing/ambiguous moe_routing")
     rows = _select_rows(rows, {"routing_runtime_path": "balanced_topk",
         "routing_strategy_sha256": BALANCED_ROUTING_SHA256}, "balanced replay attention")
     if not require_descriptor:
         return rows
+    identities = set()
     try:
-        routing_columns(json.loads(rows.moe_routing.iloc[0]))
+        for value in rows.moe_routing.unique():
+            descriptor = json.loads(value)
+            routing_columns(descriptor)
+            descriptor.pop("origin")
+            identities.add(canonical_json(descriptor))
     except (ValueError, TypeError, AttributeError, KeyError) as exc:
         raise DeepSeekV4CoverageError("V4 balanced replay attention routing identity mismatch (moe_routing)") from exc
+    if len(identities) != 1:
+        raise DeepSeekV4CoverageError("V4 balanced replay attention missing/ambiguous moe_routing")
     return rows
```

An executed AST comparison against `git show HEAD:frontier/execution_time_predictor/dsv4_predictor.py` confirmed that the predictor module outside `validate_balanced_routing` is unchanged. Full diff: `b23/changes.diff`.

**3. Verification**

Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6. Installed versions observed: NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2. Torch and vLLM are absent.

All execution used `ulimit -v 31457280` and command timeouts. The scratch `run_cpu.sh` invokes that interpreter and sets:

```bash
PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
PYTHONDONTWRITEBYTECODE=1
WANDB_DISABLED=true
VIDUR_DISABLE_WANDB=1
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TMPDIR=/data/ycfeng/tmp/codex-implement/ce6d9226-b4d3-4e5b-84de-01a40f3c9e20/b23/tmp
FRONTIER_TMP_ROOT=/data/ycfeng/tmp/codex-implement/ce6d9226-b4d3-4e5b-84de-01a40f3c9e20/b23/tmp
```

For the commands below, the variables identify the exact paths used:

```bash
W=/data/ycfeng/Frontier/.worktrees/b23-routing-origin
S=/data/ycfeng/tmp/codex-implement/ce6d9226-b4d3-4e5b-84de-01a40f3c9e20/b23
MERGED=/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/g1h_s01_post/promote/output/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv
COMMITTED=data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv
PLAN=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g1h_supplement/package_plan.json
cd "$W"
ulimit -v 31457280
```

| Acceptance | Expected | Observed | Result |
| --- | --- | --- | --- |
| 1: merged loader | 7,105 rows; one fingerprint; revision `a689ed05f462eb269b9dffe640da151e6614a5ef` | Exact expected values; fingerprint `5237f2bd076fe132`; 554 models; two preserved descriptors belonging to 4,945 and 2,160 rows | PASS |
| 2: committed-table regression | Same loaded rows and all model workload-group ranges equal before/after | 4,945 rows; all 554 model keys/ranges equal; dataframe hash, columns, dtypes and table hash equal | PASS |
| 3: measured replay closure | Zero `recomputed_current` misses for 17,000 P2-32 and 16,000 P2-64 requests | Exact request counts; zero missed requests, calls, scope requests and scope calls for both | PASS |
| 4: nine required test files | All execute and pass | 492 passed, 1 failed, 1 module skipped | FAIL / incomplete prerequisites |
| 5: hygiene | Each changed file ≤2,000 lines; no unused imports; clean diff | 372, 423 and 80 lines; all unused-import lists empty; `git diff --check` exit 0 | PASS |

Loader and baseline commands executed:

```bash
# Executed before the production change.
timeout 600s bash "$S/run_cpu.sh" "$S/check_table.py" \
  --table "$COMMITTED" --expected-rows 4945 \
  --output "$S/committed_before.json"

# Executed after the production change.
timeout 600s bash "$S/run_cpu.sh" "$S/check_table.py" \
  --table "$COMMITTED" --expected-rows 4945 \
  --baseline "$S/committed_before.json" \
  --output "$S/committed_after.json" > "$S/committed_after.log" 2>&1

timeout 600s bash "$S/run_cpu.sh" "$S/check_table.py" \
  --table "$MERGED" --expected-rows 7105 \
  --output "$S/merged_after.json" > "$S/merged_after.log" 2>&1
```

All three commands exited 0. `check_table.py` calls `load_dsv4_attention_dataframe` with TP8, H200, `fp8_ds_mla`, balanced routing, block size 256 and `MeasurementType.KERNEL_ONLY`. It invokes the unchanged trainer with the existing `_fit` test callback, which builds exact lookups, and records every returned model's workload-group ranges. Thus the range comparison covers all 554 trainer outputs; it does not claim before/after equality of serialized Random Forest estimators.

Committed dataframe hash before and after: `49bb65fc0351f7ceb75e35bccd07810e1d0b9d59425d5fddc2b93986eb03f1dc`. Committed CSV SHA-256 before and after: `34c4f3a86822994b126d195a54fce934e208fd3ba9dff2cc4fddb560c129321f`. Merged CSV SHA-256: `01a9d51b4389db15c0d61eaa7347482e6a99df421540929abc7f0c55052365eb`, also checked against the existing promotion manifest. A separate executed provenance check validated both real descriptors and confirmed one identity after removing only their top-level origins.

Measured replay command executed:

```bash
timeout 1800s bash "$S/run_cpu.sh" tests/analysis/v4_attention_replay_coverage.py \
  --table "$MERGED" --supplement-plan "$PLAN" \
  --census /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/census_snapshots/1930/p2_32 \
           /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/census_snapshots/1930/p2_64 \
  --output "$S/replay_measured.json" > "$S/replay_measured.log" 2>&1

timeout 30s bash "$S/run_cpu.sh" "$S/verify_results.py"
```

Both exited 0. The verifier explicitly asserts the required `recomputed_current` results, rather than relying on the analysis command's union-only exit condition.

| Census | Requests | Calls | Scope calls | Recomputed measured misses |
| --- | ---: | ---: | ---: | ---: |
| P2-32 | 17,000 | 580,307 | 8,566,545 | 0 |
| P2-64 | 16,000 | 642,725 | 9,483,839 | 0 |

The analysis also reports 554 current models, 554 union models and 2,160 projected rows. Its console `current` field refers to the old recorded census ranges: P2-32 still shows the historical 480 misses there. The requested `recomputed_current` field uses the merged measured table and has zero misses. `current_ranges_equal_recorded=false` is expected because the recorded ranges predate the supplement.

Required suite command executed:

```bash
timeout 1200s bash "$S/run_cpu.sh" -m pytest \
  tests/unit/test_dsv4_predictor.py \
  tests/unit/test_v4_attention_promotion.py \
  tests/unit/test_v4_attention_supplement.py \
  tests/unit/test_v4_attention_heldout.py \
  tests/unit/test_dsv4_attention_runtime.py \
  tests/unit/test_dsv4_operator_consumer.py \
  tests/unit/test_v4_prediction_memo.py \
  tests/unit/test_balanced_routing.py \
  tests/unit/test_dsv4_replay_rows.py \
  -q -ra -p no:cacheprovider \
  --basetemp "$S/pytest_acceptance" --junitxml "$S/acceptance.xml" \
  > "$S/acceptance.log" 2>&1
```

Exit 1: **492 passed, 1 failed, 1 skipped, 3 warnings in 146.76 seconds**. Counts inspected from the JUnit artifact:

| Required file | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| `test_dsv4_predictor.py` | 58 | 0 | 0 |
| `test_v4_attention_promotion.py` | 2 | 0 | 0 |
| `test_v4_attention_supplement.py` | 9 | 0 | 0 |
| `test_v4_attention_heldout.py` | 7 | 0 | 0 |
| `test_dsv4_attention_runtime.py` | 14 | 0 | 0 |
| `test_dsv4_operator_consumer.py` | 18 | 1 | 0 |
| `test_v4_prediction_memo.py` | 350 | 0 | 0 |
| `test_balanced_routing.py` | 0 | 0 | 1 module |
| `test_dsv4_replay_rows.py` | 34 | 0 | 0 |

The three warnings are existing pandas mixed-type warnings for columns 49 and 50.

Regression sensitivity and focused checks executed:

```bash
# Before the production change.
timeout 120s bash "$S/run_cpu.sh" -m pytest \
  tests/unit/test_dsv4_predictor.py tests/unit/test_v4_attention_promotion.py \
  -q -p no:cacheprovider --basetemp "$S/pytest_before" \
  -k 'accepts_package_origins or synthetic_supplement_merge' \
  > "$S/regression_before.log" 2>&1

# After the production change.
timeout 120s bash "$S/run_cpu.sh" -m pytest \
  tests/unit/test_dsv4_predictor.py tests/unit/test_v4_attention_promotion.py \
  -q -p no:cacheprovider --basetemp "$S/pytest_focused" \
  > "$S/focused.log" 2>&1
```

Before: expected regression failures, 2 failed, 1 passed and 57 deselected. Both origin-only acceptance failures were `missing/ambiguous moe_routing`. After: **60 passed in 6.82 seconds**, exit 0. The real merged loader was also run before the change and reproduced the assigned error; evidence is in `merged_before.log`.

Hygiene commands executed:

```bash
timeout 30s wc -l frontier/execution_time_predictor/dsv4_predictor.py \
  tests/unit/test_dsv4_predictor.py tests/unit/test_v4_attention_promotion.py

timeout 30s bash "$S/run_cpu.sh" /data/ycfeng/tmp/unused_imports.py \
  frontier/execution_time_predictor/dsv4_predictor.py \
  tests/unit/test_dsv4_predictor.py tests/unit/test_v4_attention_promotion.py

timeout 30s git diff --check
```

PASS: line counts 372/423/80; all three unused-import lists `[]`; diff check exit 0. Final diff totals: three files, 143 insertions and seven deletions.

**4. Remaining work and decisions for Claude**

Two observed prerequisites prevent acceptance 4:

- **Read-only output directory.** `test_dsv4_operator_consumer.py::test_both_training_paths_and_warm_cache_exact_identity` fails at line 266 while constructing `MetricsConfig(cache_dir=...)` without an output override. `MetricsConfig.__post_init__`, line 173, attempts to create the default `outputs/metrics` and raises `OSError: [Errno 30] Read-only file system: 'outputs'`. The additional directory it requires is `/data/ycfeng/Frontier/.worktrees/b23-routing-origin/outputs/metrics`. That path is outside the write scope, and the test file is also outside the writable set. Recommended action: Claude amend that test to pass `output_dir=str(tmp_path / 'metrics')`, then rerun the operator-consumer file. This keeps output in scratch and avoids broadening repository write access.
- **Missing Torch.** `test_balanced_routing.py:17` uses `pytest.importorskip('torch')`; the entire module was skipped with `No module named 'torch'`. Recommended action: rerun this file in an authorized Python 3.10 environment with CPU Torch, setting `VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0` for the native-source checks. The pinned routing source exists. No Torch dependency was installed and no GPU was used here.

The filesystem error is observed in unchanged test/config code during metrics construction. Its independence from B23 is an inference supported by the traceback and the verified restriction of the production diff to routing validation. No repeated retries were made because neither prerequisite can be fixed in the assigned source scope. No routing implementation failures remain after the focused check.

No material design decision is outstanding for the loader. Promotion of the real table and final census/full simulation acceptance remain lead-owned work and were not performed.

**5. Review handoff**

Inspect the function diff for validation-before-origin-removal, temporary-object mutation only, retained nested identity fields and the existing ambiguity error. Inspect the rejection tests to confirm the native digest/origin and router-inventory variants independently pass `routing_columns` before the loader rejects their disagreement. Inspect the promotion regression for raw-engine provenance, one merged fingerprint and unchanged concatenated measurements.

Rerun the two changed test files, the committed/merged loader checks and the measured closure command after integration. Resolve the two prerequisites above and rerun acceptance 4 before marking the assignment fully accepted.

Scratch artifacts include `check_table.py`, `run_cpu.sh`, `verify_results.py`, `committed_before.json`, `committed_after.json`, `merged_after.json`, `replay_measured.json`, `provenance_environment.json`, `verification_summary.json`, `acceptance.xml`, the associated logs, `scope_check.txt` and `changes.diff`. Scratch usage observed at final verification was approximately 23.8 MB; no real table copies were made.

Coverage evidence applies to the two specified 19:30 census snapshots. It does not establish closure for later or finished live censuses. Claude acceptance is not claimed.

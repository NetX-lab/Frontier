**1. Assignment and outcome**

Implemented the permitted parts of Claude’s stage 1a B3 assignment in `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, against HEAD `e59ac014`. The objective was to add native-coordinate pilot cases, dummy and first-tail execution, DP-peer padding emulation, calibrated device holds, sampling-inclusive step ranges, launch-shape identity checking, the own-gap A4 gate, and package 03 planning support.

The explicit writable file list governed edits within `frontier/profiling/attention/`, `frontier/profiling/kernel_gap/`, `tests/comparison/calibration/`, `tests/analysis/`, `tests/unit/`, and the task’s `p1h_bridge/` directory. All executor changes are within that list. Private scratch is `/data/ycfeng/tmp/codex-implement/1042c9ef-35cf-4c99-a6a7-52795291a84a/b3`.

Outcome is **partial** because section 1.8 requires a guard in `build_dsv4_rows`, which resides in `frontier/profiling/attention/dsv4_attention_rows.py`; that file is absent from the explicit edit list. It was left untouched. The remaining permitted implementation and checks are finished. Final review, acceptance, commits, task records, revision updates, package generation, and GPU execution remain with Claude.

Applied the settled own-gap remedy and retained class-value composition as a diagnostic. Kept native dispatch logic responsible for padding and graph-mode synchronization. Existing eager-chunked and native-default plan serialization and summaries retain byte identity. GPU imports remain lazy. No GPU, network, dependency installation, Git mutation, package generation, or delegation was performed. `package_01/`, `package_02/`, `.cortexkit/`, `.pi/`, and `remaining_measurements/` were left intact.

Applicable AGENTS instructions, the replay route and design, existing implementations and tests, B2 rebuild inputs and evidence, package 02 configuration, and the pinned native DP utilities informed the work.

**2. Changes**

Paths below are relative to the working root. The `p1h_bridge/` abbreviation means `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge/`.

| File | Change and purpose | Lines |
|---|---|---:|
| `frontier/profiling/attention/dsa_plan.py` | Added frozen `DsaDpPeer`, optional case peers, first-tail/dummy phases, hold validation and serialization, strict `build_pilot_cases`, and a shared decode allowance formula. Graph summaries include hold and phase counts. Legacy optional fields remain omitted. | 1058 |
| `frontier/profiling/attention/dsv4_attention_plan.py` | Accepts pilot payloads, enforces serving-option exclusivity, and applies the shared V4 startup query-length check to pilot engines while preserving existing messages. | 517 |
| `frontier/profiling/attention/dsv4_attention_main.py` | Adds V4-only `--pilot_cases` and `--step_hold_us` and passes them into planning. | 259 |
| `frontier/profiling/attention/dsa_vllm_runner.py` | Binds native execute signatures through `wraps`, classifies dummy and first-tail steps, applies peers during matched warmups, clears peer state in `finally`, maintains persistent step ranges, implements native dummy and first-tail drivers, and writes interval grid/block fields. Ordinary case RPC argument lengths remain compatible. | 930 |
| `frontier/profiling/attention/dsv4_graph_replay_runner.py` | Emulates the peer column using native DP synchronization with temporary module attributes restored in `finally`. Records synchronized widths while returning `None` for the local forward’s DP tensor. Adds hold calibration/inventory and sampling-inclusive ranges. | 561 |
| `frontier/profiling/attention/graph_replay_trace.py` | Adds hold annotations/activity, containment and exactly-one-kernel validation, required kernel launch dimensions, and hold exclusion from costs, windows, outside counts, and A2. | 419 |
| `frontier/profiling/kernel_gap/dsv4_replay_gaps.py` | Validates hold fields/counts, treats holds as dependency endpoints only, excludes overlapping activities, and reports duration/slack/late-launch diagnostics. Overlap exclusion precedes eager-position rejection. | 517 |
| `tests/comparison/calibration/dsv4_graph_bridge_check.py` | Replaces old configuration keys with frozen run records, validates unique mode/engine pairs, uses per-engine artifacts/logs/bootstrap files, reads listed traces, and reports hold counts. Old keys fail explicitly. | 227 |
| `tests/comparison/calibration/dsv4_composition_check.py` | Uses each call’s own signed scope gaps for A4 acceptance; retains class-value errors and missing values in `diagnostic`. Adds host-ahead fractions and scope gap/ratio percentiles. Planner lookup includes engine index. | 216 |
| `tests/comparison/calibration/dsv4_kernel_identity_check.py` — **new** | Expands planner/reuse attention kernel multisets to the full model, deduplicates native candidates, checks containment by class/mode, reports missing kernels and grid differences, and supports margin cases and CLI exit gates. | 246 |
| `tests/analysis/v4_runtime_census.py` | Shares trace parsing/forward segmentation between the existing census and new frozen `ForwardKernels`/`dispatch_kernel_multisets` API. Census outputs retain byte identity. | 455 |
| `p1h_bridge/prepare_package.py` | Copies/hashes the pilot file, builds plans per distinct mode, validates cases per run/engine, and writes run-based configuration and estimated-step manifest keys. Existing pins and fresh-destination checks remain. Not executed. | 175 |
| `p1h_bridge/package_plan.json` | Selects package 03, native limits, 200000 µs hold, 35 native capture sizes, two engine runs, and the 3600-second timeout. Source revision and planning GPU cap are unchanged. | 133 |
| `p1h_bridge/pilot_cases.json` — **new** | Defines the exact 25 assigned cases across engines 0 and 1, including peer padding and dummy steps. | 279 |
| `tests/unit/test_dsv4_pilot_cases.py` — **new** | Covers strict loader admission/rejection, phase bounds, startup checking, allowances, windows, hold validation, serialization, and package coordinates. | 201 |
| `tests/unit/test_dsv4_kernel_identity_check.py` — **new** | Uses synthetic gzip native traces and raw pilot records to cover containment, missing/grid mismatches, class/mode matching, expansion, deduplication, margins, and CLI exits. | 129 |
| `tests/unit/test_dsv4_graph_replay_profiling.py` | Adapts hold inventory, dispatch metadata, dimensions, native signature, and range fixtures; adds fake DP/hold/calibration/matching/driver coverage. Includes first-tail-alone execution and single-install nonzero-hold inventory coverage. | 976 |
| `tests/unit/test_dsv4_replay_gaps.py` | Adds mandatory hold fields to fixtures and tests hold dependencies, overlap priority, count invariants, missing fields, slack, and late launches. | 369 |
| `tests/unit/test_graph_replay_trace.py` | Adds hold attribution/validation and launch-dimension checks. | 432 |
| `tests/unit/test_dsv4_composition_check.py` | Adapts run keys and the new diagnostic contract; tests own-gap pass/fail/admission and scope distributions. | 179 |
| `tests/unit/test_dsv4_graph_bridge_check.py` | Adapts configuration/layout fixtures to runs and covers old-key rejection, duplicate runs, per-engine invocations, failure reporting, two-engine reports, and holds. | 165 |

Relevant entry points: [pilot loader](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/attention/dsa_plan.py:788), [case drivers](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/attention/dsa_vllm_runner.py:673), [DP dispatch wrapper](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/attention/dsv4_graph_replay_runner.py:328), [trace attribution](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/attention/graph_replay_trace.py:210), [own-gap gate](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/comparison/calibration/dsv4_composition_check.py:112), and [identity gate](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/comparison/calibration/dsv4_kernel_identity_check.py:181).

Public additions include `DsaDpPeer`, the two phases and DP mode constants, `build_pilot_cases`, optional pilot/hold arguments, `run_dummy_step`, hold annotation APIs and interval dimensions, `BridgeRun`, `ForwardKernels`, `dispatch_kernel_multisets`, and the identity checker APIs. `profile_args` now accepts a run; `graph_mode_flags` accepts mode and capture sizes.

Two other actors’ task-record edits were visible on resumption: `h3o_operator_costs/stage1a_replay_route.md` and the task’s `progress.md`. They were not edited by this executor and remain intact. The new pilot JSON is Git-ignored; Claude must explicitly include it when staging.

**3. Verification**

All CPU execution used `/data/ycfeng/envs/frontier-py310/bin/python`, Python **3.10.6**, from the working root in Bash, with:

```bash
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/data/ycfeng/tmp/codex-implement/1042c9ef-35cf-4c99-a6a7-52795291a84a/b3
ulimit -v 31457280
```

Tests used `-q -p no:cacheprovider` and explicit scratch `--basetemp` directories. Commands had timeout wrappers. Exact individual command arrays, working directories, exit codes, and result tails are retained in `required_test_commands.json`, `final_exact_test_commands.json`, `plan_check_commands.json`, `native_check_commands.json`, and `final_trace_check_commands.json` under scratch.

| Executed pytest group | Observed result | Status |
|---|---|---|
| `test_dsv4_replay_gaps.py test_dsv4_composition_check.py test_dsv4_graph_replay_profiling.py test_dsv4_graph_bridge_check.py test_graph_capture.py test_graph_replay_trace.py` | Final exact run: **262 passed**, exit 0 | PASS |
| `test_dsa_attention_profiling.py test_dsv4_composition.py` | **72 passed**, exit 0 | PASS |
| `test_dsa_query_one_profiling.py test_dsv4_attention_profiling.py` | **184 passed**, exit 0 | PASS |
| `test_dsa_extend_profiling.py` | **6 passed**, exit 0 | PASS |
| `test_kernel_gap_pricing.py test_kernel_gap_profiler.py` | **29 passed, 1 failed**, exit 1; `test_every_collective_launches_one_kernel_by_default`: `AttributeError: 'OperatorSpec' object has no attribute 'kernel_count'` | Known baseline FAIL, unchanged |
| `test_dsa_full_dummy_profiling.py test_dsv4_graph_measured.py test_profiling_runtime_boundary.py` | **133 passed, 76 skipped, 1 failed**, exit 1; `test_runtime_predictors_use_only_allowlisted_profiling_helpers` sees the pre-existing extra `frontier.profiling.attention.dsa_rows` import | Known baseline FAIL, unchanged |
| `test_dsv4_pilot_cases.py test_dsv4_kernel_identity_check.py` | Final exact run: **65 passed**, exit 0 | PASS |

After the final field-constant changes, the core/new files also passed together (**327 passed**), and the three legacy groups passed together (**262 passed**). Logs are `final_exact_core.log`, `final_exact_new.log`, `final_core_and_new.log`, and `final_legacy.log`. Intermediate schema/signature and new fixture failures were diagnosed and corrected; their logs remain in scratch. No new failure remains in these suites.

The principal executed orchestration commands were:

```bash
timeout 1800 /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/required_tests.py"
timeout 600 /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/final_test_subset.py"
timeout 600 /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/plan_checks.py"
timeout 120 /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/final_import_checks.py"
timeout 1800 /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/final_trace_checks.py"
timeout 60 /data/ycfeng/envs/frontier-py310/bin/python "$TMPDIR/evidence_checks.py"
```

Plan compatibility: extracted HEAD’s `frontier` and `data/config` trees into scratch `head/`. Compared HEAD and working outputs using identical output destinations. V4 grid, query-one, exact-prefill, and native-default plans, plus DSA eager and native-default plans, passed JSON and stdout byte comparisons. The three package 02 graph plans differ only by `limits.step_hold_us: 0.0`; summaries differ only by the permitted hold and phase-count fields. They also match the frozen package 02 plans after removing that addition. Results are in `plan_checks.log` and the paired plan files.

Verification adaptation: the existing CLI does not expose the native-default policy for these entry points. The scratch entry parses a normal Namespace, sets `engine_policy='native_defaults'`, and invokes the existing plan command. Existing timestamped warning logging was disabled identically for HEAD/work comparisons; product logging was not changed.

The standalone package 03 CLI plan build used `package_plan.json` plan arguments, `package_02/model/config.json`, the new pilot JSON, and FULL_AND_PIECEWISE flags with all 35 capture sizes. It passed, exit 0:

```text
num_cases=25
cases_by_phase={prefill: 1, decode: 10, first_tail: 4, dummy: 10}
estimated_steps=335
engine 0: 280 estimated steps
engine 1: 55 estimated steps
step_hold_us=200000.0
```

Verified `max_tokens`: b24/p6144 = **31**, b16/p6144 = **25**, b4/p7150 = **16**, with the latter reaching 7166 tokens, within 7168. Evidence: `package03_plan.json`, `package03_plan.stdout`, and the loader tests. The package builder itself was **NOT_RUN**, as instructed; its import and the standalone planning path were checked.

B2 re-reduction: adapted the supplied script into scratch `rebuild_bridge_raw.py`. Executed it with the source bridge tree at `/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_02/download/P1H_BRIDGE/bridge/`, package 02’s bridge configuration, and output `scratch/bridge_reduced/`, under the required memory limit and a timeout. All **12 mode/case traces** were re-attributed; every rebuilt call row equals B2 exactly. It added hold fields, DP-width nulls, grid/block intervals, and the run-based layout. Traces are symlinks to the read-only source. Evidence: `rebuild.log` and `bridge_reduced/rebuild_summary.json`.

Final saved-artifact CLI checks used:

- `dsv4_graph_bridge_check.py --config <scratch>/bridge_reduced/bridge_config.json --report`
- `python -m frontier.profiling.kernel_gap.dsv4_replay_gaps --raw_files <three mode raw files> --device h200 --model DeepSeek-V4-Flash --output_dir <scratch>/gap_csv --diagnostics_path <scratch>/gap_diagnostics.json`
- `dsv4_composition_check.py --config <scratch>/bridge_reduced/bridge_config.json`

Actual tails:

```text
[dsv4_bridge] PASS: 3 runs, FULL=24, PIECEWISE=72
[dsv4_gaps] eager_preparation: no host-ahead sample
[dsv4_composition] FAIL: 74 groups, {'no_host_ahead_call': 38, 'pass': 36}
```

Bridge exited **0**. The reducer and A4 exited **1**, exactly as required for this host-bound B2 input. The class diagnostic statuses are `{no_host_ahead_call: 38, fail: 28, pass: 8}`. Maximum own-gap passing-group p90 error is **1.8497845729582155e-16**.

| Gap class | Value, µs | Samples | Host-ahead | Host-bound excluded |
|---|---:|---:|---:|---:|
| cuda_graph | 0.4029155405916909 | 6216 | 6192 | 24 |
| cuda_graph_piecewise | 0.9618163320170886 | 17304 | 17102 | 202 |
| eager | 1.934530550687183 | 7996 | 3156 | 4840 |
| eager_preparation | unavailable | 8698 | 0 | 8602 |

Recursive comparison of diagnostics against the B2 evidence copy passed exactly after removing only the new root `hold_by_step_mode_and_case` block. No CSV was written. Evidence: `evidence_checks.log`, `gap_diagnostics.json`, and `final_trace_checks.log`.

Census: saved the HEAD census script into scratch and ran HEAD and current scripts on P2-32 with `--trace-dir`, `--output`, and `--summary`. Both exited 0; the final current script was rerun after refactoring. JSON and summary are byte-identical. SHA256 values are `02d5dc92067b93ae84b689a3c7f1369bcfc8ad9fff3987460e035ee6382f1382` for JSON and `ba6befdea0c9db7bff9c5517cd0a8aadf54d1dab43bd169c486b1730936d556e` for the summary.

Identity: executed `dsv4_kernel_identity_check.py --config <scratch>/bridge_reduced/bridge_config.json --native_trace_dirs <native directory> --output <scratch>/identity_p32.json` and again for P2-64, each with a 1200-second timeout. Native directories were the assigned `dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces` and `dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces` trees. Each run read TP0 lanes, completed without a crash, and exited 1:

```text
P2-32: FAIL, 144 FAIL steps, 12 FAIL cases; 10 deduplicated candidates, 408 native forwards
P2-64: FAIL, 144 FAIL steps, 12 FAIL cases; 17 deduplicated candidates, 956 native forwards
```

These are B2 shapes, not the new native-coordinate pilots; the assignment permits either identity verdict for this check. No margin cases were supplied. Full reports and logs are `identity_p32.json`, `identity_p64.json`, and their corresponding logs.

Import isolation passed for all **19 changed/new Python modules**, including tests and an import-only package builder check: no `torch`, `vllm`, or `cuda` module loaded. `git diff --check` passed with empty output. Executed `wc -l` for all 21 changed/new files; the largest is 1058 lines. Full output is `final_line_counts.log`.

**4. Remaining work and decisions for Claude**

- **Writable prerequisite:** authorize or implement the explicit first-tail/dummy rejection in `frontier/profiling/attention/dsv4_attention_rows.py`, then add its focused unit cases. Observed evidence: `build_dsv4_rows(pilot_plan, ())` currently returns `{'CUDA_EVENT': [], 'KERNEL_ONLY': []}` instead of raising. This was executed and recorded in `final_import_checks.log`. No indirect workaround was added.
- **Layer-count discrepancy:** the assignment prose says decoder layer 42 is uncompressed. Both canonical and package 02 configuration have `num_hidden_layers=43`, 44 ratio entries, and ending entries `(40,4), (41,128), (42,4), (43,0)`. The mandated `resolve_dsv4_layers` API therefore yields **2 uncompressed, 21 CSA, 20 HCA** decoder layers. The checker follows that API/configuration, rather than changing existing model semantics. The extra zero is outside the 43 decoder layers, consistent with the MTP entry. Claude should reconcile the prose with the pinned native/model configuration before interpreting pilot identity results.
- Include the ignored `pilot_cases.json` when staging. Preserve the two separately visible task-record edits and the pre-existing untracked directories.
- After review and commit, set `source_revision` as planned and generate package 03. The revision remains the old value intentionally; the builder’s snapshot check will require the lead’s revision update.
- GPU behavior is **NOT_RUN**: native runner execution, hold calibration and slack, DP emulation, graph recapture, first-tail/dummy drivers, and sampling-inclusive ranges have CPU fake coverage only. The next GPU run must establish actual host-ahead coverage and qualify the new pilot kernels. Apply the assigned margin IDs `e0-decode-b24-p6144` and `e0-dummy-full16` when gating that pilot identity run.

**5. Review handoff**

Inspect the strict pilot formulas and bounds, native signature binding, warmup peer lifetime, temporary DP attribute restoration, persistent range closure at sampling/dummy completion, hold dependency and overlap rules, own-gap gate versus class diagnostic, per-engine artifact keys, and full-model multiset expansion. Review the package builder diff without generating package 03 until the lead’s commit/revision step.

Rerun the core and new CPU suites after the row guard is added, and repeat the plan/census compatibility and B2 artifact checks if review changes their paths. The exact commands and all check outputs are available under the reported scratch directory. The two documented baseline failures are unrelated to this assignment and were left unchanged. Acceptance has not been claimed.

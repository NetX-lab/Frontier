**1. Assignment and outcome**

B11 implements the CPU preparation for G1h attention measurements: pilot cases, per-request context and multi-tail support, per-step state prediction, a streaming C1 coverage checker, package planning inputs, and estimated GPU time. The outcome is **partial** because the complete requested grid cannot be realized under its stated first-tail compression mixes. Final review, acceptance, commits, package building, and submission remain with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

Allowed paths were `frontier/profiling/attention/`, new files under `tests/analysis/`, `tests/unit/`, and `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g1h/`. All assignment changes stay within those paths. Scratch is `/data/ycfeng/tmp/codex-implement/9abfeda9-9f73-4dec-9cb7-ddb151fbbdac/b11`.

The implementation follows the applicable AGENTS.md instructions, B11, requirements and plan, design §6 entry 14, C1 §4.1, the graph-replay route, and the S1 stream evidence. No applicable CLAUDE.md was found. Package 03 and GLM plans preserve their serialized outputs. The V4 predictor, loader, trainer, census, collective benchmark, and other producers were not changed. No GPU, network, installation, package build, submission, Git mutation, deletion, move, or delegation was performed.

The full manifest retains **1,074 requested windows**. The generated executable subset has **494 cases and measured steps**, predicting **524 windows**, with **550 missing** and no predicted step outside the grid.

Two findings prevent complete coverage:

- **Unconditional first-tail contradiction: 234 windows.** With cached tokens `c = 6143` and query tokens `q = 1`, both `floor((c + q) / 4) > floor(c / 4)` and `floor((c + q) / 128) > floor(c / 128)` hold. Every first tail writes at both ratios. Therefore, the one-tail targets `(0, 0)` and `(k4, 0)` cannot occur. There are 39 classes × 3 profiles × 2 impossible mixes = 234 windows. Warm-up changes, more cases, or longer measurements cannot resolve this contradiction.
- **Profile allowance needs a decision: another 316 windows at the provisional allowance.** B11 specifies “near” without a numeric bound. The planning example uses an explicitly provisional ±7-token allowance, restricted to continuing cached contexts 6144–7166. A continuing ratio-128 writer in that interval needs cached 6271, 6399, …, 7039; low and high profiles cannot reach those boundaries at ±7. These 316 failures are conditional on that planning assumption, not a proof that every broader interpretation is impossible.

The tolerance and executable subset are reviewable planning inputs, not an accepted amendment to the grid. Missing targets have individual reasons in `coverage_grid.json`; no target was silently removed or widened. The package wrapper refuses incomplete planning.

**2. Changes**

Paths below are relative to the working root. `G1H` denotes `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g1h/`.

| File | Status | Behavior and reason |
| --- | --- | --- |
| `frontier/profiling/attention/dsa_plan.py` | Modified | Extends `DsaCase` with optional per-request `prompt_lens` and `tail_count`; admits optional per-case `warmup`/`measured` and engine `chunk_budget`; permits a first-tail DP peer. Strict validation covers new keys, integer ranges, context feasibility, token lifetime, KV capacity, budget, and sequence limits. Default new fields are omitted from legacy JSON. Relevant symbols: `DsaCase:151`, `build_pilot_cases:809`, `_extended_pilot_case:921`. |
| `frontier/profiling/attention/dsa_vllm_runner.py` | Modified | Submits heterogeneous prompts and simultaneous tail requests. Recorder matching uses the submitted case definition for extended cases and first tails. Prefill repeat windows use the actual submitted length. No raw-record field was added. Relevant symbols: `_WorkerState:156`, `begin_case:514`, `_case_window:555`, `run_case:675`. |
| `frontier/profiling/attention/dsa_case_prediction.py` | New | CPU synchronous FCFS scheduling prediction, shared request matching, output lifetimes, local graph dispatch, and DP redispatch. Relevant symbols: `requests_match_case:20`, `predict_case_requests:101`, `predict_dispatch:145`, `predict_case:178`. |
| `frontier/profiling/attention/v4_attention_windows.py` | New | Single C1 class/profile/compression table used by builder and checker. Describes W, R, own requests, tails, cached/query tokens, ratio writers, and virtual dummy/padding states. Assigns steps to targets and checks native class membership. Relevant symbols: `required_windows:66`, `describe_step:104`, `assign_windows:133`, `native_class_in_grid:157`. |
| `frontier/profiling/attention/v4_attention_cases.py` | New | Finds compression-phase placements, inverts FCFS construction through prediction, admits cases through the existing pilot loader, deduplicates identical submissions, and preserves infeasible targets. Relevant symbols: `_align_context_sum:18`, `_contexts:51`, `_place:78`, `build_attention_cases:105`. |
| `tests/analysis/v4_attention_coverage.py` | New | `cases` emits predicted step JSONL. `check` streams raw records, compares complete ordered request states and dispatch descriptors, lists predicted/covered/missing windows and class coverage, and reports outside, extra, missing, or mismatched steps. Supports the three native C1 JSON inputs and a complete-grid exit check. Relevant symbols: `read_steps:40`, `native_coverage:54`, `check_steps:80`, `main:146`. |
| `tests/unit/test_v4_attention_coverage.py` | New | 70 parametrized test cases cover format bounds, heterogeneous contexts, simultaneous-tail scheduling and matching, runner submissions, infeasible prediction, frozen package03 plans, all 25 native pilot cases, streaming checks, native classes, manifest blockers, and package-wrapper reuse. |
| `G1H/build_cases.py` | New | Generates pilot cases, the complete target manifest, run lists, and GPU-time estimates. Accepts source revision and context tolerance as planning inputs. |
| `G1H/prepare_package.py` | New | Imports P1h’s unchanged package builder, redirects its planning directory to G1h, and refuses incomplete coverage before invoking snapshot/package creation. |
| `G1H/pilot_cases.json` | New, generated | Contains 494 executable cases across three engines, with explicit request contexts and one measured step per case. |
| `G1H/package_plan.json` | New, generated | Destination `package_01`, name `G1H_ATTENTION`, output `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/g1h_01`, and explicit execution blockers. Pins pre-B11 HEAD `5083d0369a5f9b86c333b30fce78154a3364225f`; Claude replaces it after committing. |
| `G1H/coverage_grid.json` | New, generated | Retains all 1,074 targets, individual prediction/infeasibility status, shared-case aliases, provisional tolerance, limits, counts, and timing inputs. |
| `G1H/gpu_time_estimate.json` | New, generated | Per-engine subset estimate, conditional full-grid envelope, P1h timing basis, and limitations. |
| `G1H/README.md` | New | Documents blockers, provisional decisions, counts, construction evidence, reproduction commands, timing estimates, and lead handoff. |

Ordinary implementation choices were zero per-case warm-up and one measured step per executable case, using native engine warm-up/capture unchanged. This avoids intermediate mixes outside the requested targets and reduces time, but gives no repeated-window variance. Physically identical submissions share a case: a one-request spread degenerates to low, and profiles with no continuing request share the same coordinate. FULL W1 dummy is measured once for both planner roles, turning 14 requested dummy uses into 13 physical windows.

A construction defect found during verification was corrected within scope. W24/own24/low/ratio-4 initially left 21 tokens in the final FCFS construction step, which needs at least 24. Placement now moves two non-writers from 6144 to 6145 and 6146, preserving the profile and compression mix while making the final step feasible. No FCFS construction failures remain in the executable subset.

Pre-existing/concurrent work was left intact: the modified B9 report and `tests/analysis/v4_runtime_census.py`, untracked B10 collective checker/test, `.cortexkit/`, `.pi/`, and `remaining_measurements/`. `progress.md` also became modified concurrently and was not edited by this executor. G1h files are ignored by `.gitignore:171` (`task_memory/*`); Claude must explicitly include them when staging.

The multi-tail construction uses engine 2 with budget **24576 = 4 × 6144** and `prefill_query_len = 6143`. Engines 0 and 1 retain budget 8192. Continuing requests first reach decode; all tails are then submitted together. Every tail’s 6143-token chunk fits in the same earlier step, alongside continuing decodes. The next measured step has cached 6143/query 1 for each tail.

Native-source evidence comes from `/data/ycfeng/tmp/opencode/vllm-v0.28.0`, verified at revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`:

- `vllm/v1/core/sched/scheduler.py:525,563,751,952–966,1389–1397`: running requests precede waiting requests, chunk threshold and budget cap prefill work, and scheduled tokens advance computed state.
- `vllm/v1/worker/gpu/cudagraph_utils.py:170–282` and `dp_utils.py:16–95`: local dispatch and DP redispatch.
- `vllm/v1/attention/backends/mla/sparse_swa.py:531,712–745,765–788` and `vllm/models/deepseek_v4/nvidia/flashmla.py:148–169`: measured query-one rows take the decode attention branch; decode planning depends on actual decode tokens and layer type; prefill budget metadata belongs to attention-prefill work.
- Compressor actual-row slicing at `compressor.py:348` onward and scratch slicing at `eager_scratch.py:98–109,130–136` support unchanged measured attention shapes and branches.

This source evidence does **not** establish GPU kernel identity for the larger-budget engine. The fill/zero kernel generated by `decode_swa_lens[num_decode_tokens:] = 0` can process a larger allocation (`sparse_swa.py:551–552`); its exact CUDA symbol was not qualified on CPU. Construction prefill workspace/chunking can also change (`flashmla.py:317–323`), as can allocation/autotuning. These limits are reported rather than claiming identity. Only the measured 6144-query prefill crosses the 1024-row threshold; construction chunks are unmeasured.

**3. Verification**

All executed checks used Python **3.10.6** at `/data/ycfeng/envs/frontier-py310/bin/python`. Commands had `ulimit -v 31457280` and a timeout. The common environment and aliases below expand the exact paths used:

```bash
ulimit -v 31457280
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
B11_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python
B11_SCRATCH=/data/ycfeng/tmp/codex-implement/9abfeda9-9f73-4dec-9cb7-ddb151fbbdac/b11
export TMPDIR="$B11_SCRATCH"
B11_H3O=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs
B11_G1H="$B11_H3O/g1h"
B11_RAW=/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_03/download/P1H_BRIDGE/bridge/FULL_AND_PIECEWISE/raw
```

| Acceptance | Result | Observed evidence |
| --- | --- | --- |
| 1. Package03 unchanged | **PASS** | Rebuilt plan is JSON-equal and byte-identical to `p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json`: 11,881 bytes, SHA256 `8bb82d07eedaea7382bb39d25e3356417fee30ce7121bbd480beff711f406011`. |
| 2. Existing tests / GLM preservation | **PASS, one optional check skipped** | Final group: 561 passed, 1 skipped, exit 0. This comprises 491 existing and 70 new passing cases. Also compared 64 GLM DSA plans against HEAD’s original planner; every serialization was byte-identical. |
| 3. Package03 prediction | **PASS** | All 175 measured steps of all 25 cases match complete ordered request dictionaries, including every `num_computed_tokens`, and dispatch dictionaries. Zero mismatches or missing steps. |
| 4. Complete G1h grid | **BLOCKED** | Subset check succeeds with 494 steps, 524 predicted windows, 550 missing, zero outside steps and zero prediction mismatches. Complete-grid command exits 2, correctly exposing incomplete coverage. All 145 native serving classes and seven multi-tail coordinates lie inside the class grid. |
| 5. Streaming coverage on P1h raw | **PASS** | Exit 0; 20 covered and 1,054 missing windows; 79 outside steps, including records from `e0-decode-b4-p7150`. Outside coverage is distinct from prediction mismatch: all 175 predictions match. |
| 6. New tests | **PASS** | 70 new parametrized test cases passed in the final group. |
| 7. GPU-time estimate | **PASS as a CPU estimate** | Subset 0.678018 GPU-hours; conditional full-grid envelope 1.086529. Both below C1’s 1.5-hour G1h share and the 7-hour G1h–G4h cap. No GPU timing was run. |
| 8. Hygiene | **PASS** | Nine Python files below 2,000 lines; unused-import checker reports `[]` for each; cold imports exit 0 with no torch/vllm/cuda modules; `git diff --check` exits 0. Final file hashes still match the saved hygiene evidence. |

Executed compatibility command:

```bash
timeout 60s "$B11_PYTHON" "$B11_SCRATCH/check_compatibility.py"
```

Expected and observed: exit 0, frozen package03 bytes equal, and 64 unchanged GLM plans. The GLM matrix covers sequence limits `{1,2,4,8}`, contexts `{4160,8192}`, budgets `{(256,1024),(2048,)}`, KV capacities `{20000,200000}`, and measured decode counts `{3,8}`, with block size 64, decode warm-up 2, prefill warm-up/measured 1/2, and target TP8. Evidence: `plan_compatibility.json` in scratch.

Executed final pytest command:

```bash
timeout 120s "$B11_PYTHON" -m pytest \
  tests/unit/test_dsv4_pilot_cases.py \
  tests/unit/test_dsv4_graph_replay_profiling.py \
  tests/unit/test_dsv4_attention_profiling.py \
  tests/unit/test_dsa_attention_profiling.py \
  tests/unit/test_dsa_extend_profiling.py \
  tests/unit/test_dsa_query_one_profiling.py \
  tests/unit/test_dsv4_features.py \
  tests/unit/test_dsv4_replay_rows.py \
  tests/unit/test_dsv4_kernel_identity_check.py \
  tests/unit/test_dsa_full_dummy_profiling.py \
  tests/unit/test_v4_attention_coverage.py \
  -q -rs -p no:cacheprovider --basetemp="$B11_SCRATCH/pytest-final-5"
```

Expected: no failures. Observed: **561 passed, 1 skipped in 10.48 s**, exit 0. The optional pinned-native Git-object check at `test_dsa_full_dummy_profiling.py:166` was **NOT_RUN** because `FRONTIER_VLLM_BS_ROOT` was unset. Exact argv/environment and output are retained in `pytest-final-5.command.json` and `pytest-final-5.log`. For reruns, use a fresh basetemp; pytest removes an existing basetemp.

Import consumers were identified with:

```bash
timeout 30s rg -l 'dsa_plan|dsa_vllm_runner|dsa_case_prediction|v4_attention_cases|v4_attention_windows|v4_attention_coverage' tests --glob '*.py'
```

The resulting unit-test consumers are included above; calibration drivers are included in cold-import verification.

Executed case generation and prediction:

```bash
timeout 120s "$B11_PYTHON" "$B11_G1H/build_cases.py"
timeout 120s "$B11_PYTHON" tests/analysis/v4_attention_coverage.py cases \
  --package-plan "$B11_G1H/package_plan.json" \
  --output "$B11_SCRATCH/g1h_predictions.jsonl"
```

Both exited 0. Generation retained all targets and emitted 494 executable cases; prediction emitted 494 step records.

Executed P1h streaming check and native-class check:

```bash
timeout 120s "$B11_PYTHON" tests/analysis/v4_attention_coverage.py check \
  --package-plan "$B11_H3O/p1h_bridge/package_plan.json" \
  --context-tolerance 7 --raw "$B11_RAW" \
  --c1-json "$B11_H3O/evidence/coverage/p2_32.json" \
            "$B11_H3O/evidence/coverage/p2_64.json" \
            "$B11_H3O/evidence/coverage/p0.json" \
  --output "$B11_SCRATCH/p1h_coverage.json"
```

Expected and observed: exit 0, all predictions match, missing and outside windows are listed. Native class counts were P2-32 **43/43 inside**, P2-64 **54/54**, and P0 **48/48**, totaling 145 case-specific classes with zero outside. P1h step categories: FULL 64, PIECEWISE 16, first-tail 12, dummy 80, prefill 3.

Executed G1h subset check:

```bash
timeout 120s "$B11_PYTHON" tests/analysis/v4_attention_coverage.py check \
  --package-plan "$B11_G1H/package_plan.json" \
  --raw "$B11_SCRATCH/g1h_predictions.jsonl" --predicted \
  --c1-json "$B11_H3O/evidence/coverage/p2_32.json" \
            "$B11_H3O/evidence/coverage/p2_64.json" \
            "$B11_H3O/evidence/coverage/p0.json" \
  --output "$B11_SCRATCH/g1h_coverage.json"
```

Expected and observed: exit 0; zero mismatches, missing steps, or outside steps. The same command was executed with `--require-complete-grid` and output `g1h_complete_grid.json`; it exited **2**, as expected for the blocker, rather than passing complete-grid acceptance.

| Category | Required windows | Predicted windows | Missing windows | Cases / measured steps |
| --- | ---: | ---: | ---: | ---: |
| FULL | 351 | 239 | 112 | 229 |
| PIECEWISE without tails | 351 | 239 | 112 | 229 |
| PIECEWISE with one tail | 351 | 25 | 326 | 15 |
| Multi-tail | 7 | 7 | 0 | 7 |
| Dummies | 13 | 13 | 0 | 13 |
| Prefill | 1 | 1 | 0 | 1 |
| Total | 1,074 | 524 | 550 | 494 |

The seven multi-tail entries are singleton observed windows with low continuing contexts and mix `(tails, tails)`, rather than a further Cartesian expansion.

Executed hygiene commands include:

```bash
timeout 60s "$B11_PYTHON" "$B11_SCRATCH/check_hygiene.py"
timeout 30s "$B11_PYTHON" /data/ycfeng/tmp/unused_imports.py \
  frontier/profiling/attention/dsa_plan.py \
  frontier/profiling/attention/dsa_vllm_runner.py \
  frontier/profiling/attention/dsa_case_prediction.py \
  frontier/profiling/attention/v4_attention_cases.py \
  frontier/profiling/attention/v4_attention_windows.py \
  tests/analysis/v4_attention_coverage.py \
  tests/unit/test_v4_attention_coverage.py \
  "$B11_G1H/build_cases.py" "$B11_G1H/prepare_package.py"
timeout 30s git diff --check
```

All exited 0. The hygiene helper executes a fresh interpreter import of the changed/new modules, new test module, both builders, and existing bridge/composition/identity drivers. No `torch`, `vllm`, `cuda`, `pycuda`, or `cupy` module loaded. Evidence: `hygiene.json`, `cold_import.json`, and `unused_imports.command.json`/`.log` in scratch.

| Python module | Lines |
| --- | ---: |
| `dsa_plan.py` | 1,151 |
| `dsa_vllm_runner.py` | 937 |
| `dsa_case_prediction.py` | 184 |
| `v4_attention_cases.py` | 173 |
| `v4_attention_windows.py` | 163 |
| `v4_attention_coverage.py` | 189 |
| `test_v4_attention_coverage.py` | 365 |
| `g1h/build_cases.py` | 135 |
| `g1h/prepare_package.py` | 31 |

Earlier verification exposed a legacy validation/error regression, which was corrected. The final FCFS placement correction also exposed two new test expectations that assumed three +1 shifts; deterministic placement instead selected +1 and +2. Those expectations were corrected, and the final full group passed. Earlier logs remain in scratch; the final results supersede them. `handoff_details.json` is stale from before that final construction correction; use the final coverage reports and `hygiene.json`.

GPU-time estimate:

| Engine | Budget | Cases | Estimated total steps | GPU seconds | GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 8,192 | 472 | 4,160 | 1,433.304 | 0.398140 |
| 1 | 8,192 | 15 | 131 | 515.959 | 0.143322 |
| 2 | 24,576 | 7 | 20 | 491.604 | 0.136557 |
| Total | | 494 | 4,311 | 2,440.966 | **0.678018** |

The estimate charges **486.104 s** cold process/start/recapture time to each engine, conservatively derived from P1h engine0’s 576.104 s total minus about 90 s of cases. Observed compile/autotuning was about 431.78 s for engine0; engine1 took 90.291 s total with about 25.23 s compile/autotuning. Removing 163 × 0.2 s holds from the 90 s case interval and dividing by 280 estimated steps gives **0.205 s per unheld step**, or **0.405 s per measured step including hold**.

The conditional full-grid envelope is **1.086529 GPU-hours**: engine0 0.589704, engine1 0.360041, engine2 0.136784. It is not an executable estimate while targets remain impossible. Both estimates are below 1.5 and 7 GPU-hours, so the requested reduction proposal for estimates above three hours is unnecessary. Per-case profiler overhead, abort/drain overhead, and larger-budget autotuning remain unqualified.

**4. Remaining work and decisions for Claude**

The complete-grid blocker needs a scope decision, not another placement attempt. Resolve whether first-tail compression targets should count continuing requests separately from mandatory tail writes, or whether the requested tail mixes should be amended explicitly. Either changes the agreed coverage definition; neither was selected here.

Set the numeric profile phase allowance and settle whether continuing cached 6143 is permitted in the low profile. This can resolve some phase-placement failures but cannot remove mandatory tail writes. The generated ±7 files should be reviewed as a provisional example. Regenerate the cases and rerun complete-grid coverage after the decision.

After reviewing and committing B11, replace `source_revision` with the resulting committed revision, regenerate planning inputs, and resolve the explicit `execution_blockers`. The current revision is `5083d0369a5f9b86c333b30fce78154a3364225f`. Package snapshot validation remains unchanged. Building and submitting the package were **NOT_RUN**, as assigned; there is no `package_01` build from this work.

No additional writable prerequisite was needed. The larger-budget multi-tail engine requires later GPU kernel-identity and timing evidence, within Claude’s orchestration. It should not be presented as already GPU-qualified.

**5. Review handoff**

Review the legacy serialization omissions and extended admission in `dsa_plan.py`, submission/matching changes in `dsa_vllm_runner.py`, FCFS state advancement and DP dispatch prediction, and the single shared C1 table. Inspect the complete manifest’s 550 missing targets, provisional tolerance, deduplication aliases, and package refusal before considering GPU execution.

Rerun package03 compatibility, the listed pytest group with a fresh scratch basetemp, P1h raw comparison, G1h prediction and coverage, complete-grid enforcement, and hygiene after any substantive edits. The optional native Git-object test can be rerun with the appropriate `FRONTIER_VLLM_BS_ROOT` supplied.

The durable planning README and manifest contain the same final counts as this report. Acceptance 4 remains incomplete; the CPU prediction, compatibility, checker, test, and hygiene evidence is ready for Claude’s review. No acceptance or commit by Claude is claimed.

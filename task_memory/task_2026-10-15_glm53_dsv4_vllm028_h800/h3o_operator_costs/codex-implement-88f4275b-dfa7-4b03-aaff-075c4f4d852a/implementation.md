# B13r implementation handoff

## 1. Assignment and outcome

Implemented the CPU portion of B13r: owned runner scopes in the graph replay, direct calls to native zeroing/block-table objects, profiling registry targets, and a G2h runner package plan/builder. Final acceptance remains with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

Allowed write scope was respected:

- `frontier/profiling/attention/`
- `frontier/profiling/model_runner/`
- `frontier/profiling/kernel_gap/dsv4_replay_gaps.py`
- `frontier/operators/families.py`
- `tests/unit/`
- `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/`

Scratch: `/data/ycfeng/tmp/codex-implement/88f4275b-dfa7-4b03-aaff-075c4f4d852a/b13r`.

The root AGENTS.md, assignment constraints, named task artifacts, existing producers/package builders, native source and recorded evidence informed the implementation. No consumers, loaders, trainers, execution flags, B13 producer files, G1h grids/packages or H800 files were modified. No GPU, network, installation, submission, package generation, commit or state-changing Git command was executed.

Initial HEAD was `57d456e4f49e6ac0ed99c4639dbd10dccce2ba1f`. During execution, HEAD advanced externally to `4ed287fc03a24b8b5d014508917828911c8df492` (the B13 GPU result record). The intervening commit changed none of the assigned source paths. The package plan now pins that current HEAD; Claude must replace it after committing B13r. The initially untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` were preserved.

Settled implementation choices:

- Step owners are `(-1, scope)`, preserving the existing integer-layer annotation/record format. Physical attention layers remain nonnegative. Both owner fields are validated together.
- Sampling is an unannotated ownership container around adjacent sampling/logits/sampling segments. The original bound `sample()` implementation is called unchanged. CUDA events and kernel-only annotations both separate compute_logits from sampling.
- Eager model-forward membership is determined by CPU launch correlations within a dedicated forward range. Graph activities outside attention owners are model-forward work. Ownership is never inferred from kernel names.
- Direct planning reuses the existing attention engine builder and G1h arguments, with 43 layers and grid-specified 8 routed experts. Direct startup is eager; its unused replay hold/capture settings are cleared. Measured holds come from the direct grid.
- Exact zeroer segment construction requires allocated native tensor views. CPU planning checks B4 group membership and expected values; the worker must verify 139 segments and 10 chunks before timing.

## 2. Changes

Twenty files were changed or added. Paths below are relative to the working root.

| File | Change and purpose |
| --- | --- |
| `frontier/operators/families.py` | At 835, enables profiling for existing `kv_cache_block_zero` and adds `dsv4_block_table_write` to disabled KV_CACHE_FAMILY. At 864, adds disabled DEEPSEEK_V4_RUNNER_FAMILY for preparation/sampling. Predictor/e2e flags remain false; registry implementation is unchanged. |
| `frontier/profiling/attention/dsv4_graph_replay_runner.py` | Adds five top-level scopes, bound runner wrappers, split sampling/logits timing, a model-forward marker, runtime layout inventory and ownership/runner sidecars. Principal wrappers are at 337 and 356. Step opening/closing remains execute_model through sample_tokens. |
| `frontier/profiling/attention/dsv4_step_scopes.py` — new | Defines the reserved owner and validates step versus attention ownership. |
| `frontier/profiling/attention/dsa_vllm_runner.py` | `graph_replay_calls` at 617 records forward membership when the new marker exists. Old traces omit this extra field. |
| `frontier/profiling/attention/dsv4_attention_rows.py` | `build_dsv4_rows` at 214 filters step call rows before indexing attention calls. |
| `frontier/profiling/attention/dsv4_attention_runner.py` | `build_dsv4_engine` at 459 accepts an optional routed-expert override and forwards it through the existing builder/finalizers. |
| `frontier/profiling/kernel_gap/dsv4_replay_gaps.py` | Excludes step scopes from attention chains and preserves their former outside-work/graph classes and hold diagnostics. Relevant functions are build_layer_calls at 249 and activity_gaps at 309. |
| `frontier/profiling/model_runner/__init__.py` — new | Establishes the CPU-importable package. |
| `frontier/profiling/model_runner/dsv4_block_operators.py` — new | Grid loading/capacity checks, native layout checks, native staging, direct timing/traces and CLI. Main symbols: load_grid 70, check_layout 160, measure_worker 175, direct_engine_plan 264. GPU imports occur only in worker/runtime functions. |
| `frontier/profiling/model_runner/dsv4_runner_rows.py` — new | Correlation-based classification, per-step/per-kind ownership census, replay/direct CSV rows. forward_correlations is at 22; ownership_report at 59. |
| `tests/unit/test_dsv4_graph_replay_profiling.py` | Extends stand-ins with the pinned runner surface, updates container annotation expectations, supports the additional layout RPC and isolates sidecar writing. Removes an unused import. |
| `tests/unit/test_dsv4_runner_operators.py` — new | 44 collected CPU cases covering scopes, events, exception restoration, actual signed-owner trace parsing, classification, gap invariance, grids, staging, layout rejection, override forwarding, registry and package checks. |
| `tests/unit/test_mla_native_profiling_wrapper.py` | Adds a test-only scratch cwd fixture. The existing helper otherwise attempted to create relative `unused/` output outside the writable scope. Product behavior is unchanged. |
| `.../g2h_runner/package_plan.json` — new | Job settings, exact G1h replay lists/arguments, two phases, current HEAD pin and GPU estimate. |
| `.../g2h_runner/block_operator_grid.json` — new | All direct coordinates, layer/expert counts, timing settings and expected layout. |
| `.../g2h_runner/expected_rows.json` — new | 989 path A case IDs and 394 path B coordinates. |
| `.../g2h_runner/prepare_package.py` — new | Reuses P1h snapshot construction, validates unchanged G1h inputs and exact expected rows/cases, probes worker imports, and defines the direct phase. Only its plan-check path was executed. |
| `.../g2h_runner/profile_dsv4_block_operators.py` — new | Reuses the accepted bootstrap, substituting the direct profiler CLI. |
| `.../g2h_runner/README.md` — new | Commands, schema, ownership interpretation, layout checks, registry effects, GPU allowance and the lead’s four-gap comparison command. |
| `.../g2h_runner/native_kernel_notes.md` — new | Pinned call/launch locations, expert/layout/finalizer evidence, matching native dump lines and both permitted G1h trace summaries. |

The package prefix abbreviated above is `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs`.

### Native bound call sites

All runner call-site lines refer to pinned `vllm/v1/worker/gpu/model_runner.py`:

| Callable | Call site |
| --- | --- |
| kv_block_zeroer.zero_block_ids | 1038 |
| block_tables.apply_staged_writes | 1424 |
| prepare_inputs | 1477 |
| prepare_attn | 1480 |
| model_state.preprocess_state | 1485 |
| model_state.prepare_attn | 1535 |
| model_state.prepare_inputs | 1600 |
| sample | 1749; its row gather, logits and sampler are at 1339, 1340 and 1353 |
| postprocess_sampled | 1809; post_update/state hook at 1381/1394 |
| model.compute_logits | 1340, also passed as callback at 1764 |

AsyncOutput’s output copy remains unowned and unpriced.

### Layout, rows and source findings

The grid has zero counts 1–32 plus 2520, and writes w_c=0–32/r=0–4 excluding (0,0). This gives 33 zeroing and 164 write coordinates per method, 394 total. Warm-up is 3, repeats 20, and measured hold 20000 µs.

Expected group member counts are `[62,22,21,42,20]`, derived with the existing B4 helper. Expected table block sizes `[256,64,64,4,8]` give first-chunk entries `[24,96,96,1536,768]` for 6144 tokens, total 2520. The write grid needs at most 11 request rows; 128 are planned. IDs are consecutive free IDs starting at 1, excluding null block 0.

Native zeroing selects addresses without ID-dependent branching; shape/count determine launched work. Arbitrary nonconsecutive address patterns are not claimed timing-equivalent. Native append_block_ids stages distinct continuing pairs with overwrite false and new rows with overwrite true. Initial row restoration is outside timing.

The expert/layout source chain is documented in native_kernel_notes.md: expert count is used by FFN/gate/hash construction in nvidia/model.py:554-593/694; cache specs derive from attention/cache/compression fields in attention.py:716-779 and compressor.py:182-205; grouping operates on those specs in kv_cache_utils.py:1545/1741; zeroer metadata depends on actual tensor pointers/strides/backend dimensions in worker/utils.py:139-209. Allocation capacity can change with expert memory use and is checked separately. Exact segment count is not inferred from 167 logical cache members.

Eight experts satisfy native top-k six and physical-TP1 divisibility. Native num_hash_layers=3 remains intact; dummy hash IDs use the reduced valid range. Existing finalizers and balanced routing accept the reduced count by source inspection. Actual startup/layout remains a worker validation item.

The native notes correct two name-based assumptions: global-top-k mapping also executes inside model forward, and the large prefill vectorized gather belongs to dispatch. The optional bias kernel is absent from both selected G1h steps.

Producer schema is `dsv4_model_runner.v1`, using DSA-style time_stats columns in ms, with separate kernel_stats/memcpy_stats, coordinates, repeats and layout JSON. Replay writes adjacent ownership JSON and runner CSV; direct calls write layout JSON, CSV, Chrome traces and separate interval JSON. The future loader consumes these artifacts; no consumer admission was added.

## 3. Verification

Commands ran under `ulimit -v 31457280`, with individual timeouts. Common environment:

```bash
ROOT=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
S=/data/ycfeng/tmp/codex-implement/88f4275b-dfa7-4b03-aaff-075c4f4d852a/b13r
PY=/data/ycfeng/envs/frontier-py310/bin/python
export PYTHONPATH="$ROOT:$ROOT/tests/unit:$ROOT/tests/comparison/calibration:$ROOT/tests/analysis"
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
```

Interpreter was the supplied Python 3.10.6. CPU torch checks additionally used `/usr/local/lib/python3.10/dist-packages`; native-source tests used the supplied VLLM_BS_SOURCE_ROOT. Existing system setuptools was made available by appending `/usr/lib/python3/dist-packages` to the broad test process. Nothing was installed.

### Acceptance 1: existing records — PASS

Executed before and after:

```bash
timeout 600s "$PY" "$S/baseline.py" "$S/before"
timeout 600s "$PY" "$S/baseline.py" "$S/after"
timeout 60s cmp "$S/before/rows.json" "$S/after/rows.json"
timeout 60s cmp "$S/before/gaps.json" "$S/after/gaps.json"
```

The script invokes build_dsv4_replay_rows and reduce_dsv4_gaps on all three G1h-02 raw JSONL files. Both comparisons exit 0. All 4945 attention rows and complete gap diagnostics are byte-identical.

| Class | Gap µs | Sample count |
| --- | ---: | ---: |
| cuda_graph | 0.49773414267747085 | 169232 |
| cuda_graph_piecewise | 0.6249537992538311 | 289730 |
| eager | 0.5446049595929223 | 30243 |
| eager_preparation | 1.2208671118771937 | 91170 |

Executed `timeout 300s "$PY" "$S/trace_evidence.py"`. It runs the bridge report with old/new container definitions on the two permitted traces and compares all report fields after normalizing output paths. Both reports PASS and are equal. FULL has 488 graph/76 eager kernel intervals; multi-tail has 442/121. Existing outside counts 323/322 and unattributed-launch counts 624/688 are retained. This is a two-trace bridge regression, not a reread of all 989 trace files. The FULL/PIECEWISE/NONE fixtures also preserve gap classes and diagnostics when formerly unowned intervals receive step owners.

### Acceptance 2–3: scopes and ownership census — PASS

Stand-ins verify every selected callable has a top-level owner, nested step calls fail, sampling excludes logits, event segments/exception restoration work, and execute/sample_tokens boundaries persist. A final test passes a signed step owner through the actual trace reader with both kernel and memcpy activities. Classification fixtures cover attention, step, graph/eager model-forward and unowned activities, plus all requested case kinds and unowned grid summaries.

### Acceptance 4–5: direct grid and layout evidence — PASS on CPU

Executed:

```bash
timeout 120s "$PY" -m frontier.profiling.model_runner.dsv4_block_operators --grid "$ROOT/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/block_operator_grid.json" --model_config_path "$ROOT/data/config/models/deepseek-ai__DeepSeek-V4-Flash.json" --dry_run
```

Observed 394 rows, equal to expected_rows.json after coordinate-order normalization, and gpu_operations=0. Tests reject n=2521 against planned free capacity, r=129 against 128 request rows, and five experts against top-k six. Staging counts and runtime layout rejection are tested. Native file:line evidence is supplied; allocated 139-segment validation is deliberately deferred to the worker.

### Acceptance 6: registry preservation — PASS

Executed `timeout 120s "$PY" "$S/registry_regression.py"`. It loads HEAD’s prior family definitions, compares serialized family fields and rebuilds plans against both registries. All 11 unaffected families are equal. Fifty plans are identical: five models (Llama 3.1 8B, Llama 3.3 70B, DeepSeek V3, step-moe-noquant and GLM-5.3), TP 1/2/4/8/16, replicated enabled/disabled. New target/admission assertions also pass.

The README records every production lookup/filter category. Explicit profiling queries expose the four targets; canonical operator vocabulary gains the three new keys. Existing model contracts do not select the new family, and fixed-family/attention-only callers remain unchanged.

### Acceptance 7–8: tests — PASS for available checks

Final focused command:

```bash
timeout 120s "$PY" -m pytest -q -ra -p no:cacheprovider --basetemp "$S/focused_trace_final_temp" tests/unit/test_dsv4_runner_operators.py tests/unit/test_dsv4_graph_replay_profiling.py tests/unit/test_dsv4_replay_gaps.py tests/unit/test_profiling_runtime_boundary.py
```

Observed **214 passed**, including **44 new cases**.

The broader selection contains 93 files; its exact list is `$S/selected_tests.json`. It includes the requested attention/replay/bridge/coverage, operator/family/kernel-gap/runtime-boundary tests and changed-module importers. Executed with the CPU torch path and native-source override:

```bash
timeout 600s "$PY" - <<'PYTEST'
import json, sys, pytest
from pathlib import Path
s = Path('/data/ycfeng/tmp/codex-implement/88f4275b-dfa7-4b03-aaff-075c4f4d852a/b13r')
sys.path.append('/usr/lib/python3/dist-packages')
raise SystemExit(pytest.main(['-q','-ra','-p','no:cacheprovider','--basetemp',str(s/'broad_final_temp'),'-k','not test_every_collective_launches_one_kernel_by_default and not test_stage2_cli_writes_reproducible_artifacts and not test_mqa_stage2_cli_writes_reproducible_artifacts',*json.loads((s/'selected_tests.json').read_text())]))
PYTEST
```

Observed **2551 passed, 5 skipped, 3 deselected**. The final signed-owner case was added afterward and passed in the focused suite.

Skipped: one V4 linear and two V4 MoE cases requiring unavailable vLLM; MHA/MQA live-artifact checks without their `/tmp` artifacts. Deselected: the known collective test and two unrelated MHA/MQA CLI checks requiring missing FlashInfer. The latter two were executed earlier and failed with `flashinfer-python is not installed in the active Python env`; they remain NOT_RUN successfully.

Earlier environment diagnostics are retained: minimal collection failed for missing torch; an initial CPU run had 19 dependency/output-path failures; a whole-suite scratch-cwd experiment broke relative model-config lookup (426 failures/467 errors) and was abandoned. The final run restored repository cwd, used existing setuptools and isolated the wrapper’s relative output in a test fixture. These were harness/environment failures, not unresolved producer failures. Logs are broad_minimal.log, broad_cpu.log, broad_scratch.log and broad_final.log.

### Acceptance 9: package plan — PASS

Executed the builder’s `--check-plan` with timeout 120s, saved as `$S/package_check_head.json`. Observed current pin `4ed287fc03a24b8b5d014508917828911c8df492`, phases G2H_RUNNER_REPLAY/G2H_BLOCK_OPERATORS, cases 686/296/7, 394 direct rows, worker import probe PASS, package_built=false and zero GPU/platform operations. Explicitly checked that package_01 does not exist.

Estimate is 1.00–1.19 GPU hours: G1h reference job 3055s, scope allowance 120–360s, direct startup 180–480s, conservative hold allowance 181.24s and other direct work 60–180s. Original cap comparison is 2.15–2.71 of seven hours. The newly available lead B13 report records 407s/0.113 GPU hours; using that observation gives approximately 2.03–2.22 of seven hours after this plan. B13r GPU timing remains unmeasured.

### Acceptance 10: hygiene — PASS

Executed `timeout 180s "$PY" "$S/hygiene.py"` and `timeout 60s git diff --check`. All unused-import lists from `/data/ycfeng/tmp/unused_imports.py` are empty. Twelve separate cold imports exit 0 with torch/vLLM/cuda/fasteners/sklearn blocked and absent from loaded modules.

Python line counts: families 964; dsa runner 942; attention rows 313; attention runner 534; replay runner 673; step helper 27; gap tool 566; model-runner init 1/direct 309/rows 138; package builder 145/adapter 26; replay tests 987/new tests 333/wrapper tests 575. Every Python module is below 2000 lines. Full evidence is `$S/hygiene.json`.

## 4. Remaining work and decisions for Claude

No unresolved CPU implementation blocker remains. Required external validation is still pending:

- Review/commit B13r and replace source_revision with that commit. The new g2h_runner directory is Git-ignored; include it explicitly under Claude’s task-record policy. No Git staging was performed here.
- Build and submit the package, then confirm native startup and runtime layout before accepting direct costs. The exact segment count is an observation requirement, not a fabricated CPU result.
- Inspect one-kernel direct calls, separate HtoD intervals, hold slack, actual table groups/widths and all unowned summaries.
- Run the README’s post-run reducer/comparison command and report all four G2h gaps, counts and deltas against G1h-02. Compare attention rows for the same 989 IDs and match owned kernels with native evidence.
- Treat path A zeroing/full-TP1 logits as diagnostic. Native pricing sources are path B zero/write and path A preparation/sampling. Output copies remain unpriced.
- Rerun the unavailable optional checks in an environment with vLLM/FlashInfer and the required live artifacts if they are required for final acceptance. Dependency changes remain Claude’s responsibility.

GPU startup, actual timing, full trace comparison and consumer admission were not executed or accepted in this session.

## 5. Review handoff

Focus the diff review on sampling’s adjacent scope/event segments and exception restoration; signed owner handling in all readers; correlation-based forward classification; native staging/restoration and capacity/layout checks; and the package phase/hash/bootstrap wiring. The native notes preserve source distinctions where identical kernel names occur in different phases.

Rerun the focused command and builder plan check after integration. The broader exact test list and logs are retained in scratch for reproducibility. Final limitations are the two-trace CPU bridge sample, unavailable optional dependencies/artifacts, and the intentionally pending native H200 runtime/layout/timing validation. No acceptance, commit, package build or submission by Claude is claimed.

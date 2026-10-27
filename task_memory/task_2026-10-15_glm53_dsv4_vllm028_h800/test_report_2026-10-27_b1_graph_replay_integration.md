# Test report: stage 1a assignment B1 (graph-replay integration) and the P1h package

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Created: B1 review and CPU checks (`03d246f9`), P1h package build and checks (`b6550140`), P1h submission. |

## Scope

- Commit `03d246f9`: DeepSeek-V4 graph-replay profiling runner and the P1h bridge driver (Codex assignment B1, request `00f96d2b-b7a0-4e7a-bb9b-3915a678a8c8`, assignment [h3o_operator_costs/codex_assignment_B1_graph_replay_integration.md](h3o_operator_costs/codex_assignment_B1_graph_replay_integration.md), report `h3o_operator_costs/codex-implement-00f96d2b-b7a0-4e7a-bb9b-3915a678a8c8/implementation.md`, outcome `implemented`). 9 files, +1,791 / −70.
- Commit `b6550140`: the P1h package builder `h3o_operator_costs/p1h_bridge/prepare_package.py` and its plan `package_plan.json`. The built package `h3o_operator_costs/p1h_bridge/package_01` is not tracked (22 MB; model directory and wheelhouse copied from the accepted base package).
- Design: [h3o_operator_costs/stage1a_replay_route.md](h3o_operator_costs/stage1a_replay_route.md) §3.4-§3.5 and design §5.

## Environment

- Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch, no vLLM).
- `export PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1; ulimit -v 31457280`, from the worktree root `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
- Native vLLM source for the pin checks: `/data/ycfeng/tmp/opencode/vllm-v0.28.0` at `2cf0a6915ce544dc493a0990f2ea38d81601128a`.

## Lead review findings and corrections (B1)

| Item | Finding | Action |
| --- | --- | --- |
| Driver invocation | The accepted bootstrap `native_dummy_bootstrap.py` runs its `script` with exactly `--config PATH`. The delivered driver required a subcommand and a configured interpreter path. | Lead correction: `main` takes `--config` and an optional `--report`; each mode runs with `sys.executable`, the interpreter that the bootstrap started. `BridgeConfig.python` removed. |
| Job environment | The first builder version put `env VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1` before `{python}`; the worker replaces only an exact `{python}` element, and the driver already sets both variables for each mode. | Removed the `env` prefix from the job argv. |
| Scope tree | `SCOPE_TREE` (26 scopes) and `CONTAINER_SCOPES` (6) checked against the pinned native call sites of attention.py, sparse_attn_indexer.py, compressor.py, flashmla.py and o_proj.py (stage1a §3.5). | Accepted. |
| Re-capture | `recapture_graphs` clears the compilation graphs and the Breakable runner graphs, captures inside the CUPTI clone recorder, and fails fast when vLLM resolved a different mode or size list. | Accepted. |

## CPU checks

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| New and stage 1a suites | `timeout 600 $PY -m pytest tests/unit/test_dsv4_graph_replay_profiling.py tests/unit/test_dsv4_graph_bridge_check.py tests/unit/test_graph_capture.py tests/unit/test_graph_replay_trace.py -q -p no:cacheprovider` | all pass | 146 passed in 1.32 s | PASS |
| Existing DSA and composition | `timeout 600 $PY -m pytest tests/unit/test_dsa_attention_profiling.py tests/unit/test_dsv4_composition.py -q -p no:cacheprovider` | 34 passed (baseline) | 34 passed in 1.93 s | PASS |
| Existing V4 and query-one profiling | `timeout 900 $PY -m pytest tests/unit/test_dsa_query_one_profiling.py tests/unit/test_dsv4_attention_profiling.py -q -p no:cacheprovider` | 184 passed (baseline before B1: 184) | 184 passed in 1.85 s | PASS |
| Import isolation | `runpy.run_path("tests/comparison/calibration/dsv4_graph_bridge_check.py", run_name="__main__")` with `--help`, then import `dsv4_graph_replay_runner` and `dsv4_attention_main`; assert no `torch`, `vllm`, `cuda` or `pandas` module loaded; every scope parent is a scope; containers are scopes | PASS | `PASS: no torch/vllm/cuda/pandas imported; 26 scopes, 6 containers` | PASS |
| Module size | `wc -l` of the 9 changed files | each at most 2,000 lines | largest `dsa_plan.py` 868 | PASS |
| Dry-run plans | `dsv4_attention_main plan` with the P1h plan arguments, `--engine_policy graph_replay`, each of NONE, PIECEWISE, FULL_AND_PIECEWISE (sizes `1 2 4 8 16 32 64 128 256 512` except NONE), `--output /data/ycfeng/tmp/b1_plans/<MODE>.json` | exit 0; NONE sizes `[]`; the 4 P1h cases present | exit 0 for all three; 30 cases each; sizes `[]` for NONE and the 10 sizes otherwise; 0 missing cases | PASS |
| Eager and native serialization (Codex check, report §4; not rerun by the lead) | The pre-B1 `dsa_plan.py` loaded from scratch; `json.dumps(..., indent=2)` of DSA and V4 plans and summaries for the `eager_chunked` and `native_defaults` policies compared | byte-identical | all four byte-identical; `require_scope_timing` bytecode unchanged | PASS (reported) |

Observation: the first run of the new suite in the Codex session reported 3 failed, 140 passed (a fake gave a string to the real scheduler classifier; a test assumed an old layout); Codex fixed the tests before delivery. The lead's runs above are on the committed code.

## P1h package checks (`package_01`)

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Snapshot paths equal the pinned revision `03d246f9` (`git diff --quiet`) | no difference | no difference (builder ran) | PASS |
| Base package manifest `9714cdb7…09cf` and all its files | equal before and after the build | equal | PASS |
| Manifest files | every SHA-256 equal | 538 files, 0 mismatches (rechecked before submission by the lead and by Pi) | PASS |
| `manifest.json` SHA-256 | recorded | `953f99f14916782c45a5aa8d75fb070a369a488da5866192049b11bfbb58c9c7` | — |
| Tool hashes in `bridge_config.json` | every file equal | 529 entries, 0 mismatches; no bytecode file listed | PASS |
| Native pins | 63 non-overlay pins equal git at `2cf0a69`; the `model.py` pin equals the overlay SHA-256 | 64 pins, 0 git mismatches; overlay pin equal | PASS |
| Job file | 1 H200, `step_main`, not preemptible, timeout 5400 s, `bridge_run` then `bridge_report` | as expected | PASS |
| Driver CLI inside the package (`-I`, package root on `sys.path`) | `--help` exits 0 | exit 0 | PASS |
| Estimated steps per mode | prefill p4095 32, decode b1 12, b4 28, b8 44 | equal for the three modes | — |

Observation: after the manifest was written, the lead's import check created `cpython-310` bytecode files under `package_01/frontier/`. They are not in the manifest or the tool hashes, and their sources are unchanged, so they do not affect the hash checks. Left in place.

## P1h submission

- Pi run `p1h-graph-bridge-1027a` (prompt `/data/ycfeng/tmp/ds4_runs/prompts/p1h-graph-bridge-1027a.md`). Quota at 18:42:58Z (`/data/ycfeng/tmp/glm53_dsv4/h200/monitor/quota_20261027T184258Z.json`): `step_main` H200 total 512, non-preemptible used 440, free 72, `gpu_usable` 72.
- Pre-submission: manifest SHA-256 equal, 538/538 files, cloud output directory absent, no existing P1h launcher.
- Job identity, platform fields and results: see the next section (pending).

## Limits

- All checks above are CPU checks with fakes for vLLM, CUDA and CUPTI. The accepted risks for P1h stay open until the GPU run: cuda-bindings 13.3.1 return shapes, extra kernels inside containers (A2 errors), a lazy Breakable capture at run time ("not a clone id"), and a `timeout` SIGTERM that skips the worker's result copy.

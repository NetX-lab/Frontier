# Test report: stage 1a assignment B1 (graph-replay integration) and the P1h package

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Created: B1 review and CPU checks (`03d246f9`), P1h package build and checks (`b6550140`), P1h submission. |
| 2026-10-27 | P1h attempt 1 (failed at the first re-capture), its cause and the fix `548ec6c0`, `package_02` (`2bdf5506`), attempt 2 (bridge PASS), the lead's CPU rerun of the report and the diagnostic per-scope readings. |

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

- Pi run `p1h-graph-bridge-1027a`, turns 1-3 (prompts `/data/ycfeng/tmp/ds4_runs/prompts/p1h-graph-bridge-1027a.md`, `-turn2.md`, `-turn3.md`). Authorization: YC's P1h approval (1 H200 on `step_main`, not preemptible, cap 2 GPU-hours). Quota at 18:42:58Z (`/data/ycfeng/tmp/glm53_dsv4/h200/monitor/quota_20261027T184258Z.json`): `step_main` H200 total 512, non-preemptible used 440, free 72, `gpu_usable` 72.
- Both attempts: identity fields read with `status.sh` while Running, all equal to the expected values: creator `i-fengyicheng`, `preemptible` `no`, queue `step-main-default`, 1 replica, mount `100.96.128.195:/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` at the same path, image `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e853…b06635`. Worker runtime: NVIDIA H200 (143,771 MiB, driver 570.124.06), Python 3.12.3, torch 2.13.0+cu130, vLLM 0.28.0, FlashInfer 0.6.16.post3.

### Attempt 1: `exp-1028-024316-296035` (failed)

- Rjob `by-stepmind/b0182`, created 18:43:16Z, Running 18:46:39Z, Failed 18:53:13Z ("Some critical tasks failed"). Package `package_01`, source revision `03d246f9`. Local evidence: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge/` (`status_1-2.json`, `cloud_status_1-2.json`, `launch.log`, `replica_logs/`, `download/`).
- `bridge_run.json`: NONE exit 0 (299.8 s); PIECEWISE exit 1 (45.4 s); FULL_AND_PIECEWISE exit 1 (42.5 s). The worker stops at the first failed step, so `bridge_report` did not run; cloud `status.json` error `RuntimeError: bridge_run exited 1`.
- First error, both graph modes, at the first `capture_begin` of the hooked re-capture (`PIECEWISE/run.log:126-127`): `RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED at ".../aten/src/ATen/core/CachingHostAllocator.h":815`.
- Cause: `recapture_graphs` released every graph of the startup capture and then captured again on the same private pool id. When all graphs of a private pool are released, its `use_count` drops to 0. PyTorch then refuses to share that pool for a new capture; both the device caching allocator and the host caching allocator check `use_count > 0`. vLLM itself takes a fresh pool from `current_platform.graph_pool_handle()` when it discards captures (V1 profiling path), and the Breakable runner keeps the pool it was built with.
- Fix (lead, `548ec6c0`, 2 files, +15 / −2): before the re-capture, `recapture_graphs` sets `cm.pool = current_platform.graph_pool_handle()` and gives the same pool to `breakable_cg_runner.graph_pool`. The unit fixture now starts from a released pool (`"released_pool"`), asserts that both the manager and the Breakable runner see the fresh pool during `capture()`, and asserts that NONE (no re-capture) keeps the old pool.
- Local CPU report on the attempt 1 NONE output (`local_report_01/`): verdict FAIL as expected (the graph modes are missing); NONE `cg_mode_counts` `{"NONE": 48}`, no trace error.

### Fix checks and `package_02` (`2bdf5506`)

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Stage 1a suites after the fix (same command as above) | all pass | 146 passed in 1.30 s | PASS |
| Existing DSA, composition, V4 and query-one suites | 218 passed (34 + 184) | 218 passed in 3.42 s | PASS |
| `package_plan.json` | destination `package_02`, job tag `…-20261027-02`, output `…/p1h_bridge_02`, source revision `548ec6c0` | as expected | PASS |
| Manifest | every SHA-256 equal | 538 files, 0 mismatches; `manifest.json` SHA-256 `acc8f6b699e80e45275a75f61fe7cf69a18dbf204b5d16ef2215b5f2b044079f` | PASS |
| Tool hashes | every file equal, no bytecode listed | 529 entries, 0 mismatches | PASS |
| Difference from `package_01` | only the fix and the identity fields | `bridge_config.json`, `frontier/profiling/attention/dsv4_graph_replay_runner.py`, `jobs/P1H_BRIDGE.json`, `overlay.json` | PASS |
| Driver CLI in the package | `--help` exits 0 | exit 0 when run as a script (without `-I`; `-I` drops the script directory from `sys.path`, which the bootstrap does not use) | PASS |

### Attempt 2: `exp-1028-031147-398843` (bridge PASS)

- Rjob `by-stepmind/af3b5`, created 19:11:47Z, Running 19:16:06Z, Succeeded 19:21:29Z (`task_status` `{"succeeded": 1}`); launcher `FINAL_STATUS succeeded`. Worker source revision `548ec6c0`, phase seconds 401.6. Local evidence: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_02/` (`status_1-4.json`, `cloud_status_1-2.json`, `launch.log`, `download/`; copy check: 29 files, 405,893,411 bytes on both sides, nothing skipped). Cloud: `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_graph_bridge/p1h_bridge_02`.
- `bridge_run.json`: NONE exit 0 (293.2 s), PIECEWISE exit 0 (48.0 s), FULL_AND_PIECEWISE exit 0 (46.7 s). Both graph modes complete the re-capture (PIECEWISE 10/10; FULL_AND_PIECEWISE PIECEWISE 10/10 and FULL 4/4). The only traceback in each `run.log` is vLLM's non-fatal chat-template warmup warning (`Tokenizer not available when skip_tokenizer_init=True`), also present in attempt 1's NONE run.
- Worker `bridge_report.json`: verdict `PASS`, `reasons` `[]`, `[dsv4_bridge] PASS: 3 modes, FULL=24, PIECEWISE=72`.

| Mode | Captures (graph / segments) | Nodes (kernel) | Eager breaks | Step modes | Traces (graph, eager, outside, unattributed launches) |
| --- | --- | --- | --- | --- | --- |
| NONE | 0 / 0 | 0 | 0 | NONE 48 | b1 (0, 2588, 1692, 1113); b4 (0, 2672, 1696, 7363); b8 (0, 2668, 1692, 13788); prefill (0, 8529, 5637, 3168) |
| PIECEWISE | 0 / 70 | 2403 (2403) | 60, owner `dsv4_attention_layer` | PIECEWISE 48 | b1 (1864, 804, 1692, 454); b4 (1864, 808, 1696, 2973); b8 (1864, 804, 1692, 5560); prefill (5856, 2724, 5604, 1289) |
| FULL_AND_PIECEWISE | 4 / 70 | 3443 (3439) | 60, owner `dsv4_attention_layer` | FULL 24, PIECEWISE 24 | b1 (2072, 596, 1692, 390); b4 (2072, 600, 1696, 2909); b8 (2072, 596, 1692, 5496); prefill (5856, 2724, 5604, 1289) |

- Every trace has `error` `null`. Under the attribution of `graph_replay_trace.py`, this means: every graph activity of a measured step maps through the clone map to one captured node of one capture, with equal kernel name, grid and block and equal activity type; each launch matches the full set of activity nodes of its capture; and no kernel is owned directly by a container scope (A2). The §5 stop rule does not apply. "Unattributed launches" are launches outside every measured step range (warmup and context-prefill steps).
- The NONE numbers equal attempt 1's NONE numbers exactly (all 16 counts).
- Lead rerun on CPU: `local_report_02/` holds a copy of the downloaded `bridge/` tree and `bridge_config.json` with `output_root` pointed at it; `dsv4_graph_bridge_check.py --config <copy> --report` (frontier-py310, 8 s) printed `PASS: 3 modes, FULL=24, PIECEWISE=72`, and its `bridge_report.json` equals the worker's report except for the trace path prefix. PASS.
- GPU use, submission to finish: attempt 1 0.166 and attempt 2 0.162 GPU-hours, 0.33 in total of the 2 GPU-hour cap.

### Diagnostic readings from the P1h traces (not costs, not a prediction check)

Script `h3o_operator_costs/evidence/p1h_bridge/scope_summary.py`, output `scope_summary_p1h_02.json` (per mode, case, layer and scope: kernels and kernel-only time per measured step, graph share). Reduced engine: 6 layers, one engine per mode, chunk budget 512.

| Case | Attention kernels per step (NONE / PIECEWISE / FULL_AND_PIECEWISE) | Attention kernel-only time per step, µs | Kernel sum − union (overlap), µs | Layer windows − union (idle inside the layer windows), µs |
| --- | --- | --- | --- | --- |
| decode b1 p256 | 112 / 122 / 122 | 558.7 / 634.8 / 648.7 | — | — |
| decode b4 p2048 | 122 / 122 / 122 | 616.1 / 665.4 / 678.3 | 22.0 / 125.4 / 145.6 | 9205.0 / 667.6 / 23.3 |
| prefill p4095 | 120.5 / 124 / 124 | 1185.5 / 1266.6 / 1264.5 | 100.4 / 193.1 / 193.3 | 7072.5 / 468.2 / 463.8 |

- Observation: at decode b1 p256, NONE runs the indexer's short-context branch (`dsv4_index_short_context_fill`, 2 kernels per step) instead of the query projection, query prepare, scores and top-k kernels (12 per step): 10 fewer kernels per step. vLLM skips that branch while capturing, so graph steps always run the long path. The prefill case shows the same in its early chunks (NONE 120.5 kernels per step on average).
- Observation: in decode, the graph-replayed projections take longer per kernel than in eager (b4: fused QKV projection 52.0 → 68.3 µs, query projection 33.4 → 43.3 µs, index weights projection 11.4 → 18.7 µs, compressor compress-store 15.3 → 27.6 µs per step), while `dsv4_sparse_attention` and the output projection stay within 2%. At the same time the kernels overlap far more across streams in graph replay (overlap 22.0 µs eager, 145.6 µs FULL), and the union of kernel time is lower (594.1 eager, 532.7 FULL).
- Inference (not verified by a controlled test): in eager mode the host's launch spacing keeps the aux-stream projections of `execute_in_parallel` mostly serial, so each kernel runs with the GPU to itself; graph replay launches them together, and the concurrent kernels share the SMs. Eager per-kernel costs therefore underprice graph-replay kernels on the aux streams, and kernel-only sums overstate graph-replay layer time. This supports taking V4 costs from graph replay (design §5, F016) and composing them with the overlap, not by summation.

## Limits

- The B1 checks are CPU checks with fakes for vLLM, CUDA and CUPTI. P1h closed the GPU risks listed for it: the cuda-bindings 13.3.1 return shapes, extra kernels inside containers (no A2 error), and a lazy Breakable capture at run time (no "not a clone id" error). The `timeout` SIGTERM risk did not arise (both attempts ended well under 5400 s).
- P1h ran the reduced engine (6 layers, TP1, one H200). It shows that the node bridge holds and that the attribution reads every graph kernel; it is not a cost table and does not show that Frontier predicts anything. The per-scope readings above come from 8 or 24 measured steps per case, one run each.

# D21-NATIVE-REPLAY verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded actual-source red/green caller regression and125 CPU checks. |

Environment: local Linux, Python3.12.13 at `/data/ycfeng/envs/asyncopd-py312/bin/python`; existing simulator-export fixture uses `/data/ycfeng/envs/frontier-py310/bin/python` through its documented override. No conda activation, GPU, auth, network, SSH or further delegation. CPU tests are not native replay evidence.

Working directory for commands: `/data/ycfeng/Frontier/.worktrees/step5-priority-runs`.

## Red signal and correction

Read-only inline `python3 -B` over P0 `stage/evidence/nodes/*/server.log` asserted at least one `Engine\\s+\\d{3}` row. Actual: four complete files, zero rows, assertion failure/exit1. This retained-run symptom intentionally remains unchanged after the fix: historical logs are immutable.

Deterministic CPU loop, before the caller correction:

```bash
env PYTHONDONTWRITEBYTECODE=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_native_source_contract.py -k canonical_caller -q --tb=short -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d21_native_replay_red02
```

Expected/observed: **2 failed in0.33s**, DP2/DP4 each `Multiple API clients disable native EngineNNN/FULL graph logging`, zero LoggingStatLogger adapters while Prometheus survives. It exercises the real builder/node_plan/server_command, actual locked ServeSubcommand.cmd, and actual locked StatLoggerManager constructor with CPU-only process/logger factories. It checks the suppression condition at the real caller seam, not just existence of source strings. Existing actual-source tests separately exercise MRv2 replay dispatch and graph table emission. First test-fixture attempt failed before that seam with `KeyError: skip_tokenizer_init`; using the existing builder's normalized engine fixed the fixture, not production behavior.

Minimal correction: canonical config freezes `api_server_count=1`; native command emits it only for API heads. Headless nodes omit it because native rejects positive counts with headless. All TP/DP/EP values, request populations, NIXL behavior, original logging level, auxiliary error isolation and metric gates remain unchanged. Frozen runtime bindings now include frontend CLI, API entry and periodic logger caller.

## Green and affected regressions

```bash
env PYTHONDONTWRITEBYTECODE=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_native_source_contract.py tests/unit/test_step5_priority_runs.py tests/unit/test_step5_canonical_native.py tests/unit/test_step5_native_pd.py tests/unit/test_native_pd_worker.py -q --tb=short -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d21_native_replay_green02
git diff --check
```

Expected: single frontend with per-engine logger covering everyDP lane, no API flag on headless peers, all prior exact-request/idle/worker/soft-observation checks unchanged. Actual: **125 passed in86.13s — PASS**; diff whitespace checkPASS. No tests skipped. This exceeds50 concrete affected scenarios without touching simulator files.

Preserved environment error: first broad invocation used the same asyncopd interpreter without `FRONTIER_SIMULATOR_PYTHON`; actual17 passed/32 setup errors (`ModuleNotFoundError: plotly`) in3.15s. The unchanged fixture's documented existing interpreter override resolved it; no install or production fix. No GPU replay is claimed from the green CPU result.

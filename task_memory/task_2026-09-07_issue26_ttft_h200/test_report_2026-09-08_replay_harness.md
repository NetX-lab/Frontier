## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded replay preparation and directly reproduced incremental DP assignment failure. |

# Replay harness preparation verification

Working directory: current issue26 worktree. Code commit: 897d2482. Modified files: tests/e2e/issue26_token_id_client.py, issue26_h200_diagnostics_worker.sh, issue26_h200_replay_worker.sh.

## Checks and evidence

- `bash -n tests/e2e/issue26_h200_replay_worker.sh tests/e2e/issue26_h200_diagnostics_worker.sh`: PASS, exit0. Detects shell parse errors before GPU launch.
- `git diff --check`: PASS before commit.
- Client localhost SSE check: PASS. A temporary aiohttp server verified request IDs, exact four prompt tokens, ignore_eos and return_token_ids; emitted two token events. Actual client observed2tokens, first-token latency2.115614ms, E2E2.163259ms; exact dispatch-lag formula passed. Raw record: /data/ycfeng/tmp/issue26-dispatch-check-1pippwg6/client.jsonl. This checks transport/recording only, not LLM numerical performance.
- Initial attempt in conda dev-vidur-v03-hopper-e2e (Python3.13.13) failed `ModuleNotFoundError: No module named 'aiohttp'`. Existing master environments also lacked the dependency. Verified recovery: create isolated venv with that Python, install aiohttp through company proxy, execute the same localhost check using /data/ycfeng/tmp/issue26-client-validation-20260908/bin/python. Installed aiohttp3.14.3. GPU image dependencies remain unchanged.
- vLLM source check: clean source46f7b179 and diagnostic source361d941c, both branch feature/frontier-comparison-instrumentation, clean trees, expected origin. Fresh remote tip ea95f571 was observed with company proxy.
- H200 predict-only: three eligible eight-GPU nodes. Actual job scheduled gpu-h200-0844. Runtime/GPU topology checks are in the worker and remain pending at this preparation checkpoint.

## Reproducible preparation and launch

```bash
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1" no_proxy="$no_proxy,127.0.0.1,localhost,::1"
export TMPDIR=/data/ycfeng/tmp PIP_CACHE_DIR=/data/ycfeng/tmp/issue26-client-pip-cache
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m venv /data/ycfeng/tmp/issue26-client-validation-20260908
/data/ycfeng/tmp/issue26-client-validation-20260908/bin/python -m pip install aiohttp==3.14.3
bash /data/ycfeng/tmp/issue26-h200-network/launch-historical-replay-01.sh
```

The exact expanded launch is persisted at runs/h200-historical-replay-01/launch.sh. It selects step_main, eight H200, pinned image digest, NFS /data mount and issue26_h200_replay_worker.sh. Each mode has an isolated run_manifest.json and run_check_status artifact. Do not rerun against existing artifact paths.

Numerical accuracy is NOT EVALUATED for this new generation until clean and Frontier runs complete. The independent round-robin failure reproduction is recorded with its exact command and raw output in analysis/round_robin_incremental_rca.md / .json.

## Batch analyzer extension

Commit91f7147a adds a batch-only branch to the existing identity analyzer. Direct synthetic fixture with400client identities and8worker files passed all formal prefill coverage checks and retained the first request ID correctly. A mutated empty token vector raised AssertionError as expected. Python3.12.3 system runtime, no GPU. Fixture/output /data/ycfeng/tmp/issue26-batch-identity-check-fck1by8w/good.json. This tests record validation, not a vLLM execution. Exact real diagnostic command, once worker completion is observed:

```bash
python tests/e2e/issue26_diagnostic_identity_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-historical-replay-01/batch/runtime/batch --mode batch --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/historical-replay-01-batch-validation.json
```

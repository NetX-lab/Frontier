## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded shared-forward regression and trained reproduction validation. |

# Shared-forward validation

Environment: CPU master, conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13. Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.

## Focused checks

Detect split phase rendezvous, wrong source attention shapes/timing, duplicate completion or request progression, and sequential PDD regressions. PASS requires one shared wave for all nine phase combinations in both arrival orders; original batch objects/token/KV state reach prediction; completion preserves old TTFT and advances only the relevant requests; existing targeted suites pass.

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_LOG_LEVEL=ERROR /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/unit/test_monolithic_mixed_forward_sync.py tests/unit/test_forward_sync_state.py tests/unit/test_shared_forward_group_admission.py tests/unit/test_prefill_ep_wave_materialization.py tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_collective_timing.py tests/unit/test_dense_layer_complete_event.py tests/integration/test_online_pdd_forward_groups.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/issue26-shared-forward-pytest-02 --tb=short
```

Observed: **73 passed in 10.45s**. Raw log: `/data/ycfeng/tmp/issue26-h200-network/mixed-forward-focused-02.log`. `git diff --check` PASS. Initial red log `mixed-forward-red.log` contains eight production cross-phase zero-wave failures and ten fixture-only AttributeErrors from direct `_forward_cohort_id` access; the fixture now calls the existing step getter. No production fallback was introduced for that fixture failure.

## Trained original three-request reproduction

Detect the actual predictor-timed stall previously observed at simulated 0.29563780122719624 seconds, and ensure no token loss or double credit. Same first three arrivals and 4096/1024 lengths, trained fresh-task H200 profiles, collective_sim/nvlink_analytic. Exact expanded simulator command: `analysis/shared-forward-three-request/command.json`; raw run: `/data/ycfeng/tmp/issue26-shared-forward-03req-01`.

Reproduce with the recorded argv in a new output directory (the recorded command uses the completed run's output paths):

```python
import json, os, subprocess
from pathlib import Path
command = json.loads(Path("analysis/shared-forward-three-request/command.json").read_text())
old = "/data/ycfeng/tmp/issue26-shared-forward-03req-01"
new = "/data/ycfeng/tmp/issue26-shared-forward-03req-review"
command = [value.replace(old, new) for value in command]
subprocess.run(command, check=True, env=dict(os.environ, PYTHONPATH="/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907", PYTHONDONTWRITEBYTECODE="1", TMPDIR="/data/ycfeng/tmp", FRONTIER_LOG_LEVEL="ERROR"))
```

Run the snippet with the conda Python from the task directory. Criteria and observations: exit0; system completed_requests=3; exactly IDs0/1/2 complete; each request has one4096-token prefill and1023single-token decode ledger entries, totaling1024generated tokens with prefill's existing first-token credit;2050stage batches; final completion56.488028970837675s. **PASS**. Selected JSON/CSV and assertions are in `analysis/shared-forward-three-request/validation.json`. Request TTFTs65.790076904/103.767020949/119.313703029ms are diagnostic only, not comparable to the full100request gate. This control reuses only newly trained current-task caches. The final full run will use fresh caches.

The three-request process loaded core source before the final cross-lane duplicate check and zero-valued speculative-tail accounting addition; those are covered by the final focused suite and the forthcoming final-source full run. No timing correction factor or CPU residual adjustment is applied.

## Batch membership analyzer check

Used the original three-request Frontier output against the first three formal request identities in the fresh vLLM diagnostic. This verifies helper identity joins and local mixed token vectors; it is not whole-case batch parity. Formal IDs/count come from the mapping rather than a hard-coded case name or count. **PASS**:3/3sameDP,3/3sameMembers, firstrequestDP0contains onlyrequest0/4096.

```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_prefill_batch_comparison.py --ledger /data/ycfeng/tmp/issue26-shared-forward-03req-01/metrics/qwen3_a3b_30b_moe/online_serving/runtime/frontier_stage_batch_ledger.jsonl --mapping /data/ycfeng/tmp/issue26-shared-forward-03req-01/request_id_map.csv --diagnostic task_memory/task_2026-09-07_issue26_ttft_h200/analysis/historical-replay-02-batch-validation.json --clients task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-historical-replay-02/batch/runtime/batch/client.jsonl --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/shared-forward-three-request/batch_comparison.json
```

The three-row mapping is the header and first three data rows of analysis/historical-replay-02-clean/request_id_map.csv. Output creation is exclusive; select a new output path for repeats.

## Independent runtime metric helper validation

Extended the existing request analyzer with optional `--groundtruth` input for six directly joined metrics. The original official-server TTFT comparator remains unchanged. Direct verification used the completed three-request Frontier artifacts and their mapped reference rows. This is a helper check, not a full-case numerical gate. **PASS**: six rows, each sample_count=eligible_count=3/excluded_count=0; Frontier TTFT96.29026696068973ms,TPOT54.88189071772979ms,E2E56240.46447119826ms match the observed three-request output; throughput numerators3requests/15360tokens/3072decode tokens and duration56.488028970837675s match endpoints and token counts. `git diff --check` PASS.

Exact invocation below reproduces the calculation from the active repository root; Python and conda are the environment stated above. Choose a fresh output directory for a repeat.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PYCODE'
import csv, importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location("analysis", "tests/e2e/issue26_frontier_baseline_analysis.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
task = Path("task_memory/task_2026-09-07_issue26_ttft_h200")
source = Path("/data/ycfeng/tmp/issue26-shared-forward-03req-01")
metrics = source / "metrics/qwen3_a3b_30b_moe/online_serving/runtime"
completions = {r["request_id"]: r for r in map(json.loads, (metrics / "metrics_ground_truth.jsonl").open()) if r["event_type"] == "request_completion"}
mapped = {int(r["frontier_request_id"]): r for r in csv.DictReader((source / "request_id_map.csv").open())}
output = task / "analysis/shared-forward-three-request/metric-helper-check-repeat"
output.mkdir()
module.compare_runtime_metrics(completions, mapped, task / "runs/h200-historical-replay-02/clean/runtime/clean/server.request_metrics.jsonl", output)
print((output / "runtime_comparison.json").read_text())
PYCODE
```

Original outputs: analysis/shared-forward-three-request/metric-helper-check. A separate exact-reconstruction assertion for vLLM TPOT failed: native TPOT uses first/last token timestamps while E2E ends at output-record construction (pinned vLLM output_processor.py:268–280). Maximum absolute reconstruction difference0.002321042939812ms/token, mean0.000073436742334ms/token. The helper correctly retains native TPOT. No tolerance relaxation or timing correction was applied.

## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded historical harness review and current replay preparation. |

# Historical harness component validation

## Execution

Reference checkout: /data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908, branch v0.3-hopper-testbed at 9fd7fea584dba998d3b8c12af7233c6f0c7deeed. CPU conda environment: `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e`; Python 3.13.13. No GPU or old Frontier numerical simulation was executed.

Exact commands:

```bash
cd /data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908
export TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python --version
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/comparison/chunked_prefill_online/test_extract_vllm_online_request_metrics.py tests/comparison/chunked_prefill_online/test_replay_openai_workload.py tests/comparison/chunked_prefill_online/test_run_vllm_colo_clean_trace_selection.py -q -p no:cacheprovider
```

Clone command (company proxy sourced before network access):

```bash
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY" TMPDIR=/data/ycfeng/tmp
git clone --single-branch --branch v0.3-hopper-testbed --depth 1 https://github.com/fwyc0573/dev-vidur.git /data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908
```

## Criteria and observed evidence

- Detect incompatibility in the reusable extractor, request replay and trace-selection behavior before porting their controls. PASS: `22 passed in 0.96s`, exit 0. These are existing regression tests, not newly generated mocks of the desired E2E result.
- Verify clone provenance and source cleanliness. PASS: `git rev-parse HEAD` returned 9fd7fea584dba998d3b8c12af7233c6f0c7deeed and `git status --short` was empty after validation.
- Preserve the current Frontier branch. PASS: branch remains task/issue26-ttft-h200-20260907; the two pre-existing untracked D005 files remain untouched.
- Initial document-name assumption failed: historical requirements.md and summary.md are absent. Resolved by reading the actual README/task_plan/notes/progress/reports layout; no runtime failure resulted.
- Numerical replay criterion: NOT EVALUATED. No fresh historical-harness GPU run has occurred. Existing current mean TTFT is Frontier 103.51040122462338 ms versus vLLM 127.38051176071167 ms, absolute error 23.87011053608829 ms, relative error 18.739217016908402%; this is a baseline only.

## Reproduce the added baseline percentiles

Run from the active Frontier worktree using the CPU Python above:

```python
import csv, importlib.util, json
from pathlib import Path
r = Path("task_memory/task_2026-09-07_issue26_ttft_h200")
p = Path("/data/ycfeng/stepfun-performance-optimization/dev-vidur-v03-hopper-reference-20260908/task_memory/task_2026-04-19_frontier_vllm_v1_accuracy_matrix/scripts/compare_case_metrics.py")
spec = importlib.util.spec_from_file_location("historical_compare", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
f = list(csv.DictReader((r / "runs/cpu-uniform-frontier-01/runtime/metrics/qwen3_a3b_30b_moe/online_serving/runtime/request_metrics.csv").open()))
v = [json.loads(line) for line in (r / "runs/h200-uniform-clean-01/runtime/clean/server.request_metrics.jsonl").open() if "warmup:" not in line]
assert len(f) == len(v) == 100
for key in ("ttft", "tpot", "request_e2e_time"):
    for percentile in (90, 95):
        a = m._percentile([float(row[key]) for row in f], percentile)
        b = m._percentile([float(row[key]) for row in v], percentile)
        print(key, percentile, a, b, abs(a-b), abs(a-b)/abs(b)*100)
```

Limits: component tests establish executable helper behavior. They do not validate H200 runtime semantics, historical measurement boundaries, first-batch equality or E2E accuracy. See `analysis/historical_accuracy_matrix_review.md` for the source-backed differences and pending replay scope.

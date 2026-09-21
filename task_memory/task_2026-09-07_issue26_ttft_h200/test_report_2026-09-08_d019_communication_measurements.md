## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Validated fresh eight-H200 communication primitives and committed their verified benchmark; physical parameter calibration remains open. |

# D019 communication direct verification

Result: **PASS for execution identity and numerical collective correctness** on eight ranks; **FAIL for current numerical attention-AR calibration**, with Frontier 0.372905067 ms versus vLLM 0.098904997 ms per call, absolute error 0.274000070 ms and signed relative error +277.033595%. No production backend parameter was changed. YC's current ideal EP abstraction is retained; naive protocol modeling is deferred.

GPU execution: root-managed job `yc26-h200-d019-profiles-20260908-02`, dedicated eight H200 / `step_main`, approved image `sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`; conda `vllm-bs-0.10.2`, Python 3.10.16, Torch 2.8.0+cu128. Exact launcher and commands/environment are recorded in `analysis/h200-d019-profiles-02/launch.sh`, its frozen `issue26_h200_d019_profiles_worker.sh`, and `analysis/d019-communication-measurements.md`. Every rank JSON records actual runtime versions, NCCL settings and the vLLM commit.

Verification purpose: detect a wrong group/message shape, incorrect runtime primitive selection, arithmetic failure, invalid samples, misreported summary statistics, or mismatch between committed code and the code actually run. Success establishes an auditable measured primitive, not calibrated E2E simulation.

CPU validation ran from the active worktree with conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13. Reproducible command:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import json, math, statistics, subprocess
from pathlib import Path
task=Path('task_memory/task_2026-09-07_issue26_ttft_h200')
run=task/'analysis/h200-d019-profiles-02'
committed=subprocess.check_output(['git','show','08f16e58:tests/performance/issue26_h200_collective_microbenchmark.py'])
assert committed==(run/'issue26_h200_collective_microbenchmark.py').read_bytes()
for rank in range(8):
 r=json.loads((run/f'runtime/communication/rank_{rank}.json').read_text())
 assert r['rank']==rank and r['correctness']=='PASS'
 assert r['tp_ranks']==list(range(rank//4*4,(rank//4+1)*4))
 assert r['dp_ranks']==[rank%4,rank%4+4] and r['ep_ranks']==list(range(8))
 assert r['contract']['local_hidden_bytes']==[16777216,4096]
 assert r['contract']['global_hidden_bytes']==16781312
 assert r['implementation']['tp_runtime_selected']==('pynccl_all_reduce' if rank<4 else 'custom_all_reduce')
 assert len(r['measurements'])==8
 for m in r['measurements']:
  assert len(m['samples'])==7
  values=[s['per_call_ms'] for s in m['samples']]
  assert all(math.isfinite(v) and v>0 for v in values)
  assert statistics.median(values)==m['median_per_call_ms']
print('PASS 8 ranks, 64 operation rows, 448 samples; committed source equals GPU snapshot')
PY
```

Observed output:

```text
PASS 8 ranks, 64 operation rows, 448 samples; committed source equals GPU snapshot
```

The full artifact audit also checks all local/router/global byte fields, uniform runtime versions and source commit, custom maximum/disabled fields, PyNCCL and symmetric-memory status, exact operation order, per-block versus per-call ratios and all min/max summaries. Machine-readable evidence and all per-rank timings are in `analysis/d019-communication-measurements.json`; the comparison table and limits are in its Markdown sibling.

Baseline benchmark commit: `08f16e58` (`test: measure reached vLLM TP and DP collective primitives on H200`). Only the verified benchmark was staged and committed; unrelated work was preserved. The subsequent approved multi-size extension is not covered by this GPU PASS and awaits separate execution.

The current 50-us per-step setting contributes 300 us/call and is inconsistent with the complete measured approximately 99-us call. Removing it alone predicts 72.905067 us, still -26.287782% relative to the actual primitive. A five-size PyNCCL sweep with 16 MiB held out and kernel-family capture is the approved next step; no residual-fitting constant or global factor has been applied.

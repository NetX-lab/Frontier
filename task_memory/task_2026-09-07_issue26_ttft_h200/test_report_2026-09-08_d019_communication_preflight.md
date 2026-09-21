## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded CPU-only exact group/message checks and current backend decomposition; GPU validation remains pending. |

# D019 communication preflight

Execution directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
Environment: conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13; executable `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`; `PYTHONDONTWRITEBYTECODE=1`. No GPU or Docker command was executed by this lane.

Purpose: detect syntax/import problems in the CPU-safe contract path, wrong TP/DP rank layouts, and accidental mixing of local tokens, global tokens or routed assignments in message sizes. A failure would prevent freezing the microbenchmark for root-managed GPU execution.

Exact command:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
p=Path('tests/performance/issue26_h200_collective_microbenchmark.py')
ast.parse(p.read_text())
spec=importlib.util.spec_from_file_location('benchmark',p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
a=SimpleNamespace(tp_size=4,dp_tokens=[4096,1],hidden_size=2048,experts=128,warmup=20,calls=48,repeats=7)
c=m.describe(a)
assert c['tp_groups']==[[0,1,2,3],[4,5,6,7]]
assert c['dp_groups']==[[0,4],[1,5],[2,6],[3,7]]
assert c['local_hidden_bytes']==[16777216,4096]
assert c['local_router_bytes']==[1048576,256]
assert c['global_hidden_bytes']==16781312
print('PASS AST and exact group/message contract; GPU execution pending')
PY
```

Observed output:

```text
PASS AST and exact group/message contract; GPU execution pending
```

The CLI was also executed directly:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/performance/issue26_h200_collective_microbenchmark.py --describe
```

PASS: JSON reported world size 8, the expected TP/DP groups, BF16, `[4096,1]` source populations, local hidden bytes `[16777216,4096]`, local router bytes `[1048576,256]`, global hidden bytes `16781312`, global router bytes `1048832`, 20 warmups, 48 calls/sample and 7 samples.

Current-backend direct calculation is preserved in `analysis/d019-communication-contract.json`: TP4, 16-MiB input, 25,165,824 modeled bytes/rank, 6 steps, 300-us configured launch term/layer, 0.3729050667 ms/layer and 17.8994432 ms/48 layers. The source-proven zero-launch counterfactual is 3.4994432 ms/48 layers; removing the term changes prediction by 14.4 ms. The current trace's 17.899440-ms rounded value matches this backend calculation. This is formula validation, not agreement with ground truth. Existing lower-density actual attention AR is 5.537248–5.708768 ms across real-lane TP ranks; it remains diagnostic and is not a fitted calibration target.

Practical limits: the CPU check does not import the GPU-only vLLM path, validate distributed initialization, measure any collective, or establish numerical calibration. The root-managed H200 worker must produce eight valid rank records and demonstrate actual dispatch/combine correctness and positive event spans. Root owns the final pre-start addition of `torch.inference_mode()` around the measured operations to match the decorated `GPUModelRunner.execute_model`; group initialization remains outside that context. No production backend change is delivered in this preflight.

Inspection failures: several initial searches used nonexistent guessed module/glob names (for example `frontier/cc_backend/base.py`); `rg --files` and actual symbol search located `base_cc_backend.py`, `operators/families.py` and `operators/spec.py`. These were read-only path-selection errors, not runtime failures or suppressed calibration errors.

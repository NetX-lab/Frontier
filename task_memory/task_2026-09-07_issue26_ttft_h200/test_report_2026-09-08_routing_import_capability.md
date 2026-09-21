## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded routing input capability checks. |

# Routing input capability check

## Execution

Working directory: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907.
Conda environment: dev-vidur-v03-hopper-e2e; Python3.13.13. No GPU, network, training or numeric calibration inputs were used. The following reproduces the exact checked source boundaries and fixtures:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PYCODE'
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
source = Path('frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py')
cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'SklearnMoEExecutionTimePredictor')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_init_global_routing_allocations')
namespace = {'Dict': dict, 'np': np, 'ClusterType': SimpleNamespace(DECODE_ATTN='decode_attn')}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
probe = SimpleNamespace(_replica_config=SimpleNamespace(total_expert_num=128, moe_routing_trace_path='/data/ycfeng/tmp/nonexistent-routing-capability-probe.jsonl'), _cluster_type='monolithic', _model_config=SimpleNamespace(is_moe=True, num_layers=48), _moe_ep_size=8, _moe_routing_distribution_type='balanced', _moe_routing_seed=42)
ratios = namespace['_init_global_routing_allocations'](probe)
assert len(ratios) == 48 and set(ratios[0].values()) == {1 / 128}
spec = importlib.util.spec_from_file_location('routing_capability_workload', 'frontier/moe_ep_workload.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
counts = {i: 256 for i in range(128)}
counts[0] += 91
counts[1] -= 91
kwargs = dict(target_replica_id=0, global_layer_id=0, router_topk=8, total_expert_num=128, moe_expert_parallel_size=8, expert_to_ep=module.build_contiguous_expert_ownership(128, 8))
same = module.materialize_layer_ep_workload(routing_ratios={i: n / 32768 for i, n in counts.items()}, routing_token_count=4096, **kwargs)
assert dict(same.global_per_expert_tokens) == counts
other = module.materialize_layer_ep_workload(routing_ratios={i: n / 32768 for i, n in counts.items()}, routing_token_count=4097, **kwargs)
assert sum(other.global_per_expert_tokens.values()) == 32776
assert dict(other.global_per_expert_tokens) != counts
print('PASS: ignored trace path; exact same-population counts; different population changes counts')
PYCODE
```

## Criteria and observed evidence

1. Detect whether the selected routing generator consults an external path. Observed48 synthetic balanced layers despite a nonexistent path; source reference audit shows no file consumer.
2. Detect whether the existing allocator can preserve a known full count vector at the same token population. PASS, all128 expert counts identical; total32768.
3. Detect whether reusing ratios with a different token population preserves those counts. Counts differ as expected; total32776 at4097tokens.

The original invocation additionally printed Python version and a JSON receipt; its assertions and source methods are reproduced above. Process success is supported by all explicit assertions. No predicted/actual TTFT was measured in this check; the existing baseline remains98.783926557ms versus115.982880592ms, absolute17.198954035ms andrelative14.828872974%, with routing still unaligned.

## Limits

Synthetic fixtures establish a reusable internal allocator only, not a working import interface, arbitrary-ratio numerical proof, full simulator execution or observed groundtruth equality. Findings and the scoped next proposal are maintained in analysis/routing_import_capability.md.

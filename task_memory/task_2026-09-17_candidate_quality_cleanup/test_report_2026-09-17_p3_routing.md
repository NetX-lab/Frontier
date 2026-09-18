## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded exact routing characterization and shared integerization verification. |

# P3 S7 — One Hamilton integerization owner

## Preserved invariant and change

`moe_ep_workload` owns normalization, floor allocation, largest-remainder ranking, expert-ID tie order and conservation. Extracted those existing operations as `materialize_expert_token_counts` in the same module. The runtime materializer and experimental replay both call it. This avoids a second implementation and avoids manufacturing replica/EP identities to obtain only a histogram. The existing ratio-generator export remains an alias; explicit-histogram validation and deterministic assignment reconstruction are unchanged.

Initial characterization compared 3,407,872 generated histograms (experts 4/8/16/32/64/128/256, four distributions, seeds 0/1/42/999, sizes 1–4096, top-k 1–min(experts,8)). No feasible old histogram differed. This vectorized diagnostic checks rounding sensitivity, not all possible floating values or native GPU execution.

## Focused tests

Python `/data/ycfeng/tmp/quality-review-env/bin/python` 3.12.3, uv venv, no conda. Working directory is the active worktree for after; `../quality-baseline-c288a19f` for before.

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_sglang_experimental_increment13.py tests/unit/test_sglang_routed_sorting.py tests/unit/test_moe_ep_workload_materializer.py tests/unit/test_typed_ep_lane_contract.py
```

- Frozen candidate: **52 PASS**, 5.01 s; `/data/ycfeng/tmp/quality-p3-routing-frozen.log`.
- After: **60 PASS**, 5.07 s; `/data/ycfeng/tmp/quality-p3-routing-after.log` (eight new registry-consumer checks).
- An initial asynchronous local pre-check overlapped the patch window, so it is not used as baseline evidence. The replacement baseline above executed the immutable frozen checkout.

## Exact before/after comparison

Run from the active worktree:

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python - <<'PY'
import importlib.util, itertools, subprocess, sys
from frontier.profiling.experimental.sglang import routed_moe_replay as candidate
from frontier import moe_ep_workload as runtime
reference = 'c288a19f59bec09529ee18d782fa57218da2c781'
def frozen_module(name, path):
    source = subprocess.check_output(['git', 'show', f'{reference}:{path}'], text=True)
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    sys.modules[name] = module
    exec(compile(source, f'{reference}:{path}', 'exec'), module.__dict__)
    return module
baseline = frozen_module('frontier.profiling.experimental.sglang._quality_frozen_routes', 'frontier/profiling/experimental/sglang/routed_moe_replay.py')
old_runtime = frozen_module('_quality_frozen_ep_workload', 'frontier/moe_ep_workload.py')
passed = rejected = workload_count = 0
for n, size, k, kind, seed, layer in itertools.product((8,32,64,512), (1,7,64,255), (1,2,4,10), ('balanced','random','skewed','zipf'), (0,17), (0,3)):
    if k > n:
        continue
    kwargs = dict(physical_size=size,total_expert_num=n,router_topk=k,distribution_type=kind,seed=seed,layer_id=layer)
    try:
        old = baseline.input_from_frontier_config(**kwargs)
    except ValueError:
        try:
            candidate.input_from_frontier_config(**kwargs)
        except ValueError:
            rejected += 1
            continue
        raise AssertionError(('candidate unexpectedly accepts',kwargs))
    new = candidate.input_from_frontier_config(**kwargs)
    assert old.as_dict() == new.as_dict(), kwargs
    passed += 1
    ratios = runtime.generate_moe_routing_ratios(total_expert_num=n,distribution_type=kind,seed=seed,layer_id=layer)
    for ep in (1,2,4):
        args = dict(routing_ratios=ratios,target_replica_id=2,global_layer_id=layer,routing_token_count=size,router_topk=k,total_expert_num=n,moe_expert_parallel_size=ep,expert_to_ep=runtime.build_contiguous_expert_ownership(n,ep))
        a,b = old_runtime.materialize_layer_ep_workload(**args),runtime.materialize_layer_ep_workload(**args)
        for field in ('global_per_expert_tokens','per_ep_per_expert_tokens','per_ep_routed_tokens','participant_ep_ids','expert_to_ep','total_routed_assignments'):
            assert getattr(a,field) == getattr(b,field),(args,field)
        workload_count += 1
print('PASS exact replay dictionaries',passed,'unchanged infeasible rejection',rejected,'exact EP workloads',workload_count)
PY
```

Observed **PASS: 893 exact replay dictionaries, 67 unchanged infeasible rejections, 2,679 exact EP workloads**. Full replay dictionaries include every per-token assignment, source and histogram. All assertions are exact, not tolerance-based. Log: `/data/ycfeng/tmp/quality-p3-routing-exact.log`.

This is generated CPU evidence, not universal floating-point equivalence. Final simulator fidelity and timing include the extracted runtime call. No numerical discrepancy was observed or expected value changed.

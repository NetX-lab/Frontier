## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified GDN variant source and collective setup reuse. |

# P3 — Small contract reuse

## Execution and criteria

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`. Python: `/data/ycfeng/tmp/quality-review-env/bin/python`, 3.12.3, uv venv (no conda).
The tests detect changes to hybrid topology, GDN identities, collective precision/metadata and the CPU import boundary.

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_gdn_semantic_core.py tests/unit/test_hybrid_runtime_family.py tests/unit/test_collectives_rocm_runner.py tests/unit/test_collectives_increment11.py tests/unit/test_profiling_runtime_boundary.py
```

PASS before: 39 tests, 7.94 s (`/data/ycfeng/tmp/quality-p3-small-before.log`). PASS after: 39 tests, 7.98 s (`/data/ycfeng/tmp/quality-p3-small-after.log`).

The GDN topology variant is now unpacked from the family's singleton supported-variant declaration; no arbitrary first-variant selection or new fallback. Collective BenchmarkRunner invokes the existing `_configure_collective_environment` owner after visibility selection and before distributed initialization; local-worker ordering and CPU-only imports remain unchanged.

## Direct setup equivalence

The following command was run once with `frontier_setup_root=../quality-baseline-c288a19f` and once with `frontier_setup_root=.`:

```bash
env PYTHONPATH="$frontier_setup_root" TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python - "$frontier_setup_root" <<'PY'
import json, os, sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
sys.path[0] = sys.argv[1]
sys.modules['ray'] = SimpleNamespace(remote=lambda **kwargs: lambda cls: cls)
with redirect_stdout(sys.stderr):
    from frontier.profiling.collectives import benchmark_runner as runner
keys = ('NCCL_ASYNC_ERROR_HANDLING', 'NCCL_GRAPH_MIXING_SUPPORT', 'KINETO_LOG_LEVEL', 'NCCL_IGNORE_DISABLED_P2P')
with patch.dict(os.environ, {key:'original' for key in keys}):
    with patch.object(runner, 'set_process_visible_device', return_value='cpu-contract') as visible:
        obj = runner.BenchmarkRunner(3, 8, '127.0.0.1')
    visible.assert_called_once_with(3, torch_module=runner.torch)
    assert obj._accelerator_visibility == 'cpu-contract'
    actual = {key:os.environ.get(key) for key in keys}
    assert actual == dict(zip(keys, (None, '0', '5', '1')))
    print(json.dumps(actual, sort_keys=True))
PY
```

PASS: both outputs are identical and all assertions passed. Artifact: `/data/ycfeng/tmp/quality-p3-collective-env-comparison.log`. This is CPU setup equivalence, not NCCL/RCCL execution or device timing evidence.

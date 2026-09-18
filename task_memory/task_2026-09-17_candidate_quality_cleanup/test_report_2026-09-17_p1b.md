## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded memory ownership preservation and environment repair. |

# P1b: memory model invariant

Environment: `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, uv venv with same-interpreter system-site-packages inheritance; no conda. System Torch 2.12.0+cu132 imports successfully; CPU-only tests do not establish GPU support. Matplotlib installed using the internal mirror. No shared environment modified.

Run in both `../quality-baseline-c288a19f` and active worktree, replacing `CHECK` with `quality-p1b-before` / `quality-p1b`:

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_increment6_memory.py tests/unit/test_gdn_scheduler_slots.py tests/unit/test_ffn_memory_operator_families.py tests/unit/test_mla_model_config_contracts.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/CHECK
```

Criteria: identical resident-stage selection, parameter/KV/GDN bytes and capacity, valid state slot lifecycle, dense/MLA behavior and unsupported layout rejection. Observed **47 PASS before (8.60s), 47 PASS after (8.24s)**. No changed expectations.

Direct numerical comparison (execute in both roots):

```python
from tests.unit.test_gdn_increment6_memory import _replica_config, _replica
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType
for tp in (1, 2):
    for pp in (1, 2):
        p = MemoryPlanner(_replica_config(tp_size=tp, pp_size=pp), _replica(pp_size=pp), ClusterType.MONOLITHIC, max_num_seqs=4)
        print(tp, pp, p.get_parameter_memory_per_device_bytes(), p.get_gdn_state_memory_per_device_per_request_bytes(), p._get_kv_cache_memory_per_layer_per_block(16), p.get_num_blocks(block_size=16))
```

Output logs `/data/ycfeng/tmp/quality-memory-quality-baseline-c288a19f.log` and `/data/ycfeng/tmp/quality-memory-feature-amd-sglang-gdn.log` compare exactly with `diff -u` (exit 0). Four configurations, all integer outputs identical. Suite logs: `/data/ycfeng/tmp/quality-p1b-before.log`, `/data/ycfeng/tmp/quality-p1b.log`.

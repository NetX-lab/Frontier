## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded full-runtime regression RCA and valid fixture verification. |

# Runtime fixture contract verification

Environment: `/data/ycfeng/tmp/quality-review-env`, Python 3.12.3, uv venv (no conda), same-interpreter system-site-packages enabled.

```bash
PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_moe_shared_routing_helper.py tests/unit/test_pd_transfer_predictors.py tests/unit/test_gdn_runtime_guards.py -q -p no:cacheprovider
```

Criteria: preserve exact ratio expectations, analytical KV byte/time calculations (including MLA), and GDN unsupported-runtime errors. Resolve the four newly exposed missing-member errors without restoring production reflection or changing expected values.

Observed: **PASS**, 25 tests in 5.62 s. Log: `/data/ycfeng/tmp/quality-p1-fixtures.log`. The preceding full run had 22 failures, of which 18 match frozen candidate and main. Four fixture failures were `AttributeError` for `is_moe` (one node) and `get_num_gdn_layers` (three nodes). Only fixtures were changed; full regression remains required after subsequent phases.

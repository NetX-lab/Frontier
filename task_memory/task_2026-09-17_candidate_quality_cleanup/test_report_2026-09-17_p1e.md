## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded deterministic GDN slot allocation preservation. |

# P1e validation

CPU, Python 3.12.3, dedicated uv venv (no conda).

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_state_lifecycle.py tests/unit/test_gdn_scheduler_slots.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p1e
```

Criteria: preserve lowest free slot, stable retain/resume identity, idempotent release, exhaustion and scheduler rollback. **16 passed in 5.62s** (`/data/ycfeng/tmp/quality-p1e.log`).

Direct before/after check loaded each revision's `frontier/attention/gdn/state.py` with `runpy.run_path`. With capacity 32 and `random.Random(42)`, process request IDs 0..999: release a randomly chosen active request when full or when `rng.random() < 0.45`, then allocate when capacity permits. Record operation/request/slot triples and assert exact transcript equality. **PASS**. This tests deterministic behavior, not a claimed Simulator wall-clock improvement.

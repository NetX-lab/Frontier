## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded GDN runtime invariant verification. |

# P1c: GDN runtime contracts

Environment: dedicated `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, no conda; CPU.

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_runtime_guards.py tests/unit/test_gdn_scheduler_slots.py tests/unit/test_pdaf_decode_attn_preemption.py tests/unit/test_gdn_training_predictor_increment8.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p1c
```

Observed **46 passed in 16.44s**, log `/data/ycfeng/tmp/quality-p1c.log`. All these cases also passed in the fresh pre-refactor full suite (`quality-pre-unit-complete.xml`); its unrelated environment/document failures are not suppressed. Preserved checks include exact decode feature vector/state counts, saved/reloaded GDN predictions, mixed-phase rejection, slot release/rollback and unchanged preemption request counters. Fixture repair changes only construction validity, not expected outputs. Broader artifact fidelity remains a phase-level gate.

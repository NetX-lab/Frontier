## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded registry-owned GDN schema/task cleanup. |

# P3 GDN schema and task ownership

Environment: dedicated `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, no conda, CPU. Before cleanup the P3 suite passed 253 tests; this focused after command passed **184 in 16.34 s**:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_profiler_cpu_increment7.py tests/unit/test_gdn_profile_samples.py tests/unit/test_gdn_training_predictor_increment8.py tests/unit/test_gdn_artifact_boundary.py tests/unit/test_gdn_semantic_core.py tests/unit/test_model_architecture_registry.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p3-schema
```

Log `/data/ycfeng/tmp/quality-p3-schema.log`. Criteria: identical profiling field order/requiredness, model metadata, phase prediction outputs, fitted artifact identity, exact-profile and estimator predictions. A direct AST comparison of the `_GDN_FEATURE_COLUMNS` literal from `git show c288a19f:frontier/profiling/gdn/inputs.py` against `get_required_gdn_profiling_columns()` passes for all 27 ordered fields. Both phase-filtered GDN_TASKS lists equal their previous input/core/output order; the quantization name set remains the same four operators.

Removed zero-mean fallback is unreachable after constructor validation rejects empty/nonpositive query lengths. Kept optional masks and physical batch size, null versus empty quantization selectors, and explicit inactive-phase zero entries. CPU checks make no native GPU performance or numerical claim.

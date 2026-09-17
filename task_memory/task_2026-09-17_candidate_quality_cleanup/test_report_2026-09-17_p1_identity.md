## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded identity precedence characterization and import-cycle correction. |

# Architecture identity consolidation

Environment: Python 3.12.3, `/data/ycfeng/tmp/quality-review-env`, uv venv, no conda.

```bash
PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_semantic_core.py tests/unit/test_attention_family_binding.py tests/unit/test_model_architecture_registry.py -q -p no:cacheprovider
```

Criteria: identical registry matching, structural identity, topology admission, error precedence, explicit-profile behavior, whitespace normalization, and real model layer counts. Nine matrix cases characterize distinct entrance contracts.

- Before: **100 PASS**, 9.88 s; `/data/ycfeng/tmp/quality-p1-identity-before.log`.
- Initial after: **collection FAIL**, two errors; direct module import caused `attention.model_binding -> gdn.config -> model_architectures -> operators.families -> config.model_config -> attention.model_binding`. Registry validation reported `ValueError: model architecture profile 'step3_text' references unknown operator family`, with the actual circular `ImportError` as its cause. Log: `/data/ycfeng/tmp/quality-p1-identity-after.log`.
- Corrected after: **100 PASS**, 10.78 s; `/data/ycfeng/tmp/quality-p1-identity-after-fixed.log`. The existing exported GDN function defers import and delegates to the architecture-owned policy. No exception suppression, fallback, cache, or duplicated classification was introduced.

This validates preservation and CPU import order; it does not claim a measurable runtime speedup.

## Binding follow-up

The next bounded change uses the registered GDN family ID, directly calls the existing homogeneous binder after proving the schedule homogeneous, and patches the actual topology lookup in the cache regression. Same command plus `tests/unit/test_mla_model_config_contracts.py`: **110 PASS**. Log: `/data/ycfeng/tmp/quality-p1-binding.log`. Family IDs, layer identities and cache object identity remain unchanged; the earlier 100-test run is the pre-change control. The cache test now rejects re-entry at the actual imported resolver, rather than patching an unused function.

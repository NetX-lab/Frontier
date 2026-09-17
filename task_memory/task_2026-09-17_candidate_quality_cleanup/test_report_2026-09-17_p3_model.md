## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified canonical profiling family resolution and removal of redundant parsed overlays. |

# P3 S5 — Profiling model configuration

## Execution

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.
Environment: `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, isolated uv venv with same-interpreter system packages; no conda environment.

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true MPLCONFIGDIR=/data/ycfeng/tmp/quality-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_mla_model_config_contracts.py tests/unit/test_model_architecture_registry.py tests/unit/test_gdn_semantic_core.py tests/unit/test_gdn_campaign_preflight.py tests/unit/test_sglang_experimental_increment13.py
cmp <(sed -n '/^{/,$p' /data/ycfeng/tmp/quality-p3-model-before.json) <(sed -n '/^{/,$p' /data/ycfeng/tmp/quality-p3-model-after.json)
```

Snapshot generation used `ModelConfig.from_model_name(path.stem.replace('__', '/')).to_dict()` for every sorted `data/config/models/*.json` path, serialized with sorted keys and indentation. Each failure was recorded as its exception class and message instead of being suppressed. The before snapshot predates S5; after snapshot uses the two-file S5 patch on `3d29f6e4`. Logger warning timestamps precede the JSON; the comparison selects the JSON starting at its standalone opening brace, without ignoring any configuration field.

## Criteria and evidence

- Family selection must preserve homogeneous/hybrid head and cache semantics while resolving through one authoritative owner.
- BaseModelConfig-owned profile, architecture, topology, interval and gate fields must serialize identically after removing their second JSON overlay.
- PASS: 129 focused tests, 10.37 s; `/data/ycfeng/tmp/quality-p3-model.log`.
- PASS: all 22 snapshot entries exactly equal: 21 configurations and one existing `deepseek-v3` ValueError requiring `is_checkpoint_fp8_serialized=True` for block-wise FP8. No new rejection and no expected-value update.
- Retained raw `model_type` and five linear-shape overlays: explicit null and string-number serialization are observable and not proven redundant.
- Limits: CPU configuration/adapter tests, not native GPU execution or device timing parity.

## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | W01 topology normalization and focused evidence. |

# W01 topology sub-step

R01/N01: general topology resolution now lives in attention/model_binding.py and reuses its homogeneous classifier without invoking the layer getter recursively. GDN owns only schedule/shape parsing. Runtime and profiling configs cache the ordered specs and GDN shape together. bind_layer_attention delegates to the model-owned validated getter, removing swallowed exceptions and repeated schedule parsing. Full-attention capacity classification uses existing enabled-family memory-layout metadata, covering MLA without activating frozen DSA.

Execution: `/usr/bin/python` 3.12.3 (no conda), `env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /usr/bin/python -m pytest tests/unit/test_attention_family_binding.py tests/unit/test_gdn_semantic_core.py tests/unit/test_hybrid_runtime_family.py tests/unit/test_mla_predictor_runtime_operator_times.py tests/unit/test_attention_query_cache.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w00-20260916/w01-green3`.

Observed: 51 passed. Before repair, new tests reproduced homogeneous out-of-range binding and repeated hybrid resolution. Existing MLA real BaseModelConfig constructor test now also asserts model-owned specs/counts. Checked-in dense/Qwen3-Next/Qwen3.5 models exercised. Deepseek-v3 JSON could not construct because its existing FP8 quantization metadata violates is_checkpoint_fp8_serialized; that attempted case is recorded as unavailable, not MLA E2E PASS. MLA uses real BaseModelConfig with explicit supported metadata; public non-dummy/stage/cache MLA acceptance remains pending W03/W10.

Initial implementation check found profiling asdict forwarding the new private cache; fixed by excluding it alongside existing private caches. The malformed-topology test now constructs a new config, because topology is a construction-time snapshot; the existing architecture validator reports its established structural error before binding. No silent homogeneous fallback.

W01 implementation sub-step passes; full W01 acceptance remains PARTIAL until producer stage identities and integrated MLA evidence in W03/W10. No timing semantics changed in this commit. Runtime model_config.py is 1223 lines and profiling model_config.py is 681 lines after duplicate shape construction removal; neither exceeds the 2000-line threshold.

## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified requested-model dtype compatibility at GDN load. |

# R05 — PASS

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`; Python `/data/ycfeng/tmp/quality-review-env/bin/python` 3.12.3, uv environment, no conda activation. Prefix: `PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

```bash
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_training_predictor_increment8.py -k receiving_model_dtype -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r05-red
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_training_predictor_increment8.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py tests/unit/test_gdn_artifact_boundary.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r05-green
```

Criteria: otherwise identical receiving model and BF16 artifact accept bfloat16/torch.bfloat16/BF16 aliases and reject float16/torch.float16. Use existing PrecisionType normalization, once at loading, with no per-operator check or new alias table.

Evidence: before **2 FAIL / 3 PASS**, 8.58 s, FP16 loads did not raise. After **60 PASS**, 25.49 s, including real hybrid model and manager loading. Updated one incomplete test model to supply constructor-owned torch_dtype; no production getattr/default fallback.

Related gate inspection: GatedDeltaNetConfig accepts normalized silu/swish and sigmoid; artifact identity does not contain output_gate_type. The confirmed dtype defect is repaired. The experimental SGLang dense primitive explicitly passes gdn.output_gate_type to its native builder and uses different silu/sigmoid reference functions (frontier/profiling/experimental/sglang/dense.py). The standard vLLM producer instead constructs QwenGatedDeltaNetAttention from the checkpoint hf_config (frontier/profiling/gdn/vllm_wrapper.py), without mapping the Frontier output_gate_type into that constructor. These source paths do not establish interchangeable native gate contracts. No claim that all variants share a verified backend contract is made, and no speculative gate/backend schema expansion is implemented here.

Logs: `/data/ycfeng/tmp/pr33-r05-red.log` and `/data/ycfeng/tmp/pr33-r05-green.log`. These are CPU identity/contract checks, not native precision benchmarks.

## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified hybrid ROCm full-attention family binding. |

# R02 — PASS for CPU contracts; native execution pending hardware

Execution directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`. Interpreter `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, uv environment, no conda activation. Prefix commands with `PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

```bash
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_vllm_rocm_attention_wrapper_increment9.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r02-red
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_vllm_rocm_attention_wrapper_increment9.py tests/unit/test_attention_family_binding.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r02-green
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/integration/test_pr33_native_profiling_acceptance.py --collect-only -q -k rocm_attention -p no:cacheprovider
```

Criteria: the actual Qwen3.8 hybrid ModelConfig reaches ROCm implementation setup; homogeneous MHA/GQA/MQA work; latent MLA is rejected; the whole-model homogeneous binder still rejects hybrid configs. The wrapper uses the existing runtime family resolver and accepts only DENSE_KV.

Evidence: pre-fix **1 FAIL / 9 PASS**, 28.17 s; exact hybrid binder ValueError. Post-fix **25 PASS**, 3.13 s. Native regression now parametrizes Llama and actual Qwen3.8 configs across prefill/decode and retains its causal numerical-reference and operation-sample checks. Collection verifies availability of those tests, not native correctness. No ROCm hardware execution occurred.

Logs: `/data/ycfeng/tmp/pr33-r02-red.log`, `pr33-r02-green.log`, `pr33-r02-native-collection.log`. Test script absolute paths are the execution directory joined with the paths above.

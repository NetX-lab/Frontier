## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded Increment 4 hybrid GDN semantic, profiling schema, and quantization checks. |

# Increment 4 Test Report — Hybrid GDN Semantic Core

## Execution

- Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Python: 3.12.3
- Environment: CPU master; PyTorch 2.5.1+cu124 is installed, while `vllm`, `sglang`, and `aiter` are unavailable.
- Compile check:

  ```text
  python -m compileall -q frontier tests
  ```

- Focused regression:

  ```text
  python -m pytest \
    tests/unit/test_gdn_semantic_core.py \
    tests/unit/test_attention_family_binding.py \
    tests/unit/test_attention_family_specs.py \
    tests/unit/test_attention_memory_layout.py \
    tests/unit/test_model_architecture_registry.py \
    -q -p no:cacheprovider
  ```

- Diff whitespace check:

  ```text
  git diff --check
  ```

- Qwen3.8 selective quantization smoke:

  ```python
  from frontier.config.model_config import BaseModelConfig
  from frontier.config.quantization_manager import QuantizationManager

  config = BaseModelConfig.create_from_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")
  manager = QuantizationManager()
  manager.configure_from_model_config(config)
  assert manager.get_precision("moe_grouped_gemm").name == "FP4"
  assert manager.get_precision("attn_pre_proj").name == "BF16"
  assert manager.get_precision("share_expert_up_proj").name == "BF16"
  manager.load_config()
  ```

## Criteria

| Criterion | Expected result |
| --------- | --------------- |
| Qwen3.8 structural schedule | 92 layers, full-attention IDs `3, 7, 11, ..., 91`, 69 GDN and 23 full-attention layers |
| Eight-layer synthetic schedule | Full-attention IDs `[3, 7]`, six GDN and two full-attention layers |
| Binding safety | Hybrid whole-model binder fails; layer binding requires an explicit valid global layer ID |
| Legacy behavior | Dense and MLA binders remain homogeneous; Qwen3-Next reduced configs remain dense despite `linear_*` fields |
| GDN runtime semantics | `FIXED_STATE`, no runtime KV helpers or KV factor, and deterministic fail-fast behavior |
| Profiling contract | Synthetic fixture validates against all GDN `DEVICE_EVENT` feature and timing-stat columns |
| Quantization | FP4 applies to `moe_grouped_gemm` only; attention/shared expert remain BF16 |
| Optional stack imports | GDN semantic and input modules import without vLLM/SGLang/AITER/ROCm |

## Evidence

- **PASS** — `python -m compileall -q frontier tests` exited 0.
- **PASS** — Focused combined suite: **134 passed** in 6.63s.
- **PASS** — GDN semantic suite alone: **12 passed** in 2.81s after correcting the expected tensor-parallel conv-state dimension to the contract-derived value `(3, 128)`.
- **PASS** — Qwen3.8 load observed `num_layers=92`, `num_gdn_layers=69`, `num_full_attention_layers=23`, and full-attention IDs beginning `[3, 7, 11, 15, 19]`.
- **PASS** — Qwen3.8 quantization manager observed `moe_grouped_gemm=FP4`, `attn_pre_proj=BF16`, and `share_expert_up_proj=BF16`; `load_config()` completed.
- **PASS** — `git diff --check` exited 0.
- **PASS** — No optional GPU framework was imported by the CPU-safe GDN modules.
- **SKIP: AMD/MI355X hardware unavailable** — real ROCm GDN profiling, `DEVICE_EVENT` writer execution, and hardware timing fidelity were not attempted.

The synthetic fixture validates schema and control-flow contracts only. Its timing values are test data and do not establish benchmark or ground-truth parity.

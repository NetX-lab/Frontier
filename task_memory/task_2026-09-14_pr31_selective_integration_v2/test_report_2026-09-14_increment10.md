## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded standard vLLM functional MoE and MXFP4 CPU contract verification. |

# Increment 10 Verification Report — Standard vLLM/AITER MoE + MXFP4

## Execution

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Python: `3.12.3`; CPU environment; `torch 2.5.1+cu124` and `triton 3.1.0` are importable. vLLM, ROCm, AITER, and SGLang are unavailable.
- Commands:

  ```bash
  python -m pytest \
    tests/unit/test_moe_mxfp4_increment10.py \
    tests/unit/test_moe_profiling_output_metadata.py \
    tests/unit/test_profiling_governance_minimal_red.py \
    tests/unit/test_profiling_accelerator.py -q -p no:cacheprovider
  python -m compileall -q frontier/profiling/moe tests/unit/test_moe_mxfp4_increment10.py
  git diff --check
  ```

## Criteria and evidence

| Criterion | Evidence | Result |
| --- | --- | --- |
| Invalid quantization combinations fail before GPU allocation | `validate_moe_quantization_mode(use_fp8=True, use_mxfp4=True)` raises a mutual-exclusion `ValueError`; BF16/FP8/MXFP4 selections resolve deterministically. | PASS |
| Physical MXFP4 planning | `plan_mxfp4_weight_layout()` validates positive dimensions, group size 32, even packed dimensions, two FP4 values per byte, and E8M0 byte scales; tiny BF16-independent shapes are asserted. | PASS |
| Current vLLM API selection | `profile_fused_moe_kernel()` exposes `use_mxfp4`; current vLLM import fallback selects package-level `fused_experts` when the legacy low-level API is absent. | PASS (CPU import/source contract) |
| Explicit implementation metadata | `resolve_grouped_gemm_backend()` returns `frontier_loop`, `vllm_fused`, or `vllm_aiter_mxfp4`; profile rows include `moe_grouped_gemm_backend` and `moe_quantization_mode`. | PASS |
| Optional dependency isolation | Importing the MoE kernel module succeeds without vLLM; GPU-only calls retain an explicit missing-vLLM error. | PASS |
| Existing MoE metadata/governance behavior | Existing metadata and governance tests remain green with the new mode handling. | PASS; combined suite `63 passed` |
| Compilation and whitespace | Targeted compileall and `git diff --check` exited successfully. | PASS |

## Hardware boundary

`SKIP: AMD/MI355X hardware unavailable` — BF16 fused-experts execution, physical MXFP4 packing through current vLLM `FusedMoEFactory`/`OnlineQuantizationConfig`, AITER backend selection, ROCm `DEVICE_EVENT` rows, and GPU timing were not executed. The CPU layout contract and mocked/import checks do not establish AMD runtime correctness, benchmark parity, or groundtruth parity. No SGLang data enters the standard MoE path.

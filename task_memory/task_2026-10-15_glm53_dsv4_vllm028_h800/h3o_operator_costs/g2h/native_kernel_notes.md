# G2h native kernel evidence

Pinned vLLM revision: `2cf0a6915ce544dc493a0990f2ea38d81601128a`. These are native H200 P2 reference measurements, not results of G2h. Durations are microseconds. The dump generator (`evidence/v4_operators/native_all_kernels.py:35-37`) reports mean launches per selected forward and median duration per launch, grouped by kernel name and grid; it truncates kernel names at 150 characters. Shortened names below are prefixes, not complete template identities.

| Key | Dump | Selected population |
| --- | --- | --- |
| 32D | `../evidence/v4_operators/p2_32_decode_dp0.txt:1` | FULL decode, 58 forwards; actual token counts 1 and 2 |
| 32P | `../evidence/v4_operators/p2_32_prefill_dp0.txt:192` | NONE prefill, 2 forwards of 6144 tokens |
| 64D | `../evidence/v4_operators/p2_64_decode_dp0.txt:1` | FULL decode, 59 forwards; actual token counts 1 and 9 |
| 64P | `../evidence/v4_operators/p2_64_prefill_dp0.txt:192` | NONE prefill, 1 forward of 6144 tokens |

For the 64D nine-token forwards, the graph descriptor is padded: head and final norm have W=16, and the layer scopes have L=2. The dumps therefore must not be read as grid=actual token count. For prefill, L=768 and W=6144. Only the selected real populations are tabulated; FULL dummy rows at the top of the prefill dumps are excluded.

Aggregate attribution limits: decode scaling has 236 launches per forward across shared and attention linears, so the `scale_1x128_kernel` entries repeated under up/down are evidence of the backend, not 236 launches owned by either shared scope. Both fused mHC names share the same aggregate kernel entries: 43 FFN-side calls (`dsv4_mhc_attn_post_ffn_pre`) and 42 later-attention-side calls (`dsv4_mhc_ffn_post_attn_pre`), 85 in total, from native `model.py:898-990`. Each fused call owns the corresponding kernels once; each standalone measurement times one call. Prefill `mhc_post` has 86 launches per forward: 85 fused calls plus one final post, whose duration is not separately identifiable from this grouping. Fractional counts across different grids sum to the per-forward count (rounding can leave 0.01 difference). Shared operators have 43 calls per forward, one per decoder layer; endpoints and first/final mHC each have one.

## `share_expert_up_proj`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:2 | `void tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float>(__nv_…` | `[1056, 1, 1]` | 236.00 | 1.60 |
| 32D:22 | `void deep_gemm::fp8_gemm_kernel_swapAB<4096u, 4096u, 128u, 16u, 128u, 1u, 8u, 128u, 128u, 1u, deep_gemm::Norma…` | `[132, 1, 1]` | 43.00 | 8.90 |
| 32P:212 | `void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10::BFloat16 con…` | `[1536, 1, 1]` | 43.00 | 9.06 |
| 32P:213 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 4096u, 1u, 128u, 192u, 128u, 128u, 12…` | `[132, 1, 1]` | 43.00 | 19.90 |
| 64D:2 | `void tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float>(__nv_…` | `[1056, 1, 1]` | 236.00 | 1.60 |
| 64D:17 | `void deep_gemm::fp8_gemm_kernel_swapAB<4096u, 4096u, 128u, 16u, 128u, 1u, 8u, 128u, 128u, 1u, deep_gemm::Norma…` | `[132, 1, 1]` | 43.00 | 9.06 |
| 64P:212 | `void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10::BFloat16 con…` | `[1536, 1, 1]` | 43.00 | 9.06 |
| 64P:213 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 4096u, 1u, 128u, 192u, 128u, 128u, 12…` | `[132, 1, 1]` | 43.00 | 19.90 |

## `share_expert_act`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:23 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(…` | `[1, 1, 1]` | 43.00 | 2.34 |
| 32P:214 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(…` | `[768, 1, 1]` | 43.00 | 4.32 |
| 64D:28 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(…` | `[1, 1, 1]` | 32.07 | 2.34 |
| 64D:62 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(…` | `[2, 1, 1]` | 10.93 | 2.37 |
| 64P:214 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(…` | `[768, 1, 1]` | 43.00 | 4.26 |

## `share_expert_down_proj`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:2 | `void tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float>(__nv_…` | `[1056, 1, 1]` | 236.00 | 1.60 |
| 32D:24 | `void deep_gemm::fp8_gemm_kernel_swapAB<4096u, 2048u, 128u, 16u, 128u, 1u, 8u, 128u, 128u, 1u, deep_gemm::Norma…` | `[132, 1, 1]` | 43.00 | 5.95 |
| 32P:215 | `void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10::BFloat16 con…` | `[768, 1, 1]` | 43.00 | 5.41 |
| 32P:216 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 2048u, 1u, 128u, 192u, 128u, 128u, 12…` | `[132, 1, 1]` | 43.00 | 12.51 |
| 64D:2 | `void tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float>(__nv_…` | `[1056, 1, 1]` | 236.00 | 1.60 |
| 64D:18 | `void deep_gemm::fp8_gemm_kernel_swapAB<4096u, 2048u, 128u, 16u, 128u, 1u, 8u, 128u, 128u, 1u, deep_gemm::Norma…` | `[132, 1, 1]` | 43.00 | 6.02 |
| 64P:215 | `void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10::BFloat16 con…` | `[768, 1, 1]` | 43.00 | 5.50 |
| 64P:216 | `void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 2048u, 1u, 128u, 192u, 128u, 128u, 12…` | `[132, 1, 1]` | 43.00 | 12.54 |

## `dsv4_mhc_first_pre`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:71 | `void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 4096u, 64u, 32u, 64u, 16u, 128u, 12u, 128u, 128u>(unsigned…` | `[16, 1, 1]` | 1.00 | 3.28 |
| 32D:72 | `mhc_pre_big_fuse_broadcast_with_norm_tilelang_kernel` | `[1, 1, 1]` | 1.00 | 4.91 |
| 32P:252 | `void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 4096u, 64u, 32u, 64u, 11u, 128u, 12u, 128u, 128u>(unsigned…` | `[132, 1, 1]` | 1.00 | 6.83 |
| 32P:253 | `mhc_pre_big_fuse_broadcast_with_norm_tilelang_kernel` | `[768, 1, 1]` | 1.00 | 12.03 |
| 64D:84 | `void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 4096u, 64u, 32u, 64u, 16u, 128u, 12u, 128u, 128u>(unsigned…` | `[16, 1, 1]` | 1.00 | 3.39 |
| 64D:92 | `mhc_pre_big_fuse_broadcast_with_norm_tilelang_kernel` | `[1, 1, 1]` | 0.75 | 5.12 |
| 64D:101 | `mhc_pre_big_fuse_broadcast_with_norm_tilelang_kernel` | `[2, 1, 1]` | 0.25 | 5.09 |
| 64P:252 | `void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 4096u, 64u, 32u, 64u, 11u, 128u, 12u, 128u, 128u>(unsigned…` | `[132, 1, 1]` | 1.00 | 6.66 |
| 64P:253 | `mhc_pre_big_fuse_broadcast_with_norm_tilelang_kernel` | `[768, 1, 1]` | 1.00 | 12.29 |

## `dsv4_mhc_ffn_post_attn_pre`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:5 | `mhc_fused_tilelang_kernel` | `[1, 12, 8]` | 85.00 | 5.57 |
| 32D:6 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `[1, 1, 1]` | 85.00 | 5.15 |
| 32P:195 | `mhc_post_tilelang_kernel` | `[768, 1, 1]` | 86.00 | 14.43 |
| 32P:197 | `void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 11u, 128u, 12u, 128u, 128u>(unsigne…` | `[132, 1, 1]` | 85.00 | 13.82 |
| 32P:198 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `[768, 1, 1]` | 85.00 | 12.51 |
| 64D:6 | `mhc_fused_tilelang_kernel` | `[1, 12, 8]` | 63.39 | 5.57 |
| 64D:7 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `[1, 1, 1]` | 63.39 | 5.12 |
| 64D:35 | `mhc_fused_tilelang_kernel` | `[2, 12, 8]` | 21.61 | 5.70 |
| 64D:36 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `[2, 1, 1]` | 21.61 | 5.15 |
| 64P:195 | `mhc_post_tilelang_kernel` | `[768, 1, 1]` | 86.00 | 14.37 |
| 64P:197 | `void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 11u, 128u, 12u, 128u, 128u>(unsigne…` | `[132, 1, 1]` | 85.00 | 13.79 |
| 64P:198 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `[768, 1, 1]` | 85.00 | 12.64 |

## `dsv4_mhc_attn_post_ffn_pre`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:5 | `mhc_fused_tilelang_kernel` | `[1, 12, 8]` | 85.00 | 5.57 |
| 32D:6 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `[1, 1, 1]` | 85.00 | 5.15 |
| 32P:195 | `mhc_post_tilelang_kernel` | `[768, 1, 1]` | 86.00 | 14.43 |
| 32P:197 | `void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 11u, 128u, 12u, 128u, 128u>(unsigne…` | `[132, 1, 1]` | 85.00 | 13.82 |
| 32P:198 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `[768, 1, 1]` | 85.00 | 12.51 |
| 64D:6 | `mhc_fused_tilelang_kernel` | `[1, 12, 8]` | 63.39 | 5.57 |
| 64D:7 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `[1, 1, 1]` | 63.39 | 5.12 |
| 64D:35 | `mhc_fused_tilelang_kernel` | `[2, 12, 8]` | 21.61 | 5.70 |
| 64D:36 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `[2, 1, 1]` | 21.61 | 5.15 |
| 64P:195 | `mhc_post_tilelang_kernel` | `[768, 1, 1]` | 86.00 | 14.37 |
| 64P:197 | `void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 11u, 128u, 12u, 128u, 128u>(unsigne…` | `[132, 1, 1]` | 85.00 | 13.79 |
| 64P:198 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `[768, 1, 1]` | 85.00 | 12.64 |

## `dsv4_mhc_final_post`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:73 | `mhc_post_tilelang_kernel` | `[1, 1, 1]` | 1.00 | 3.55 |
| 32P:195 | `mhc_post_tilelang_kernel` | `[768, 1, 1]` | 86.00 | 14.43 |
| 64D:93 | `mhc_post_tilelang_kernel` | `[1, 1, 1]` | 0.75 | 3.82 |
| 64D:102 | `mhc_post_tilelang_kernel` | `[2, 1, 1]` | 0.25 | 3.94 |
| 64P:195 | `mhc_post_tilelang_kernel` | `[768, 1, 1]` | 86.00 | 14.37 |

## `dsv4_mhc_head`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:77 | `hc_head_fuse_tilelang_kernel` | `[2, 1, 1]` | 0.83 | 12.53 |
| 32D:81 | `hc_head_fuse_tilelang_kernel` | `[1, 1, 1]` | 0.17 | 10.83 |
| 32P:254 | `hc_head_fuse_tilelang_kernel` | `[6144, 1, 1]` | 1.00 | 174.75 |
| 64D:95 | `hc_head_fuse_tilelang_kernel` | `[1, 1, 1]` | 0.75 | 9.92 |
| 64D:104 | `hc_head_fuse_tilelang_kernel` | `[16, 1, 1]` | 0.25 | 14.37 |
| 64P:254 | `hc_head_fuse_tilelang_kernel` | `[6144, 1, 1]` | 1.00 | 175.42 |

## `dsv4_final_norm`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:78 | `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, …` | `[2, 1, 1]` | 0.83 | 2.05 |
| 32D:82 | `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, …` | `[1, 1, 1]` | 0.17 | 2.10 |
| 32P:255 | `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, …` | `[6144, 1, 1]` | 1.00 | 27.94 |
| 64D:96 | `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, …` | `[1, 1, 1]` | 0.75 | 2.06 |
| 64D:105 | `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, …` | `[16, 1, 1]` | 0.25 | 2.27 |
| 64P:255 | `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, …` | `[6144, 1, 1]` | 1.00 | 27.81 |

## `dsv4_embedding`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:65 | `triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0` | `[1, 1, 1]` | 1.00 | 0.77 |
| 32D:66 | `triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1` | `[1, 1, 1]` | 1.00 | 2.24 |
| 32D:67 | `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, 2, 2, -2>(at:…` | `[32, 1, 1]` | 1.00 | 2.82 |
| 32D:75 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespac…` | `[16, 1, 1]` | 0.83 | 2.43 |
| 32D:79 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespac…` | `[8, 1, 1]` | 0.17 | 2.58 |
| 32P:247 | `triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0` | `[3072, 1, 1]` | 1.00 | 2.90 |
| 32P:248 | `triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1` | `[3072, 1, 1]` | 1.00 | 2.88 |
| 32P:249 | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` | `[6144, 2, 1]` | 1.00 | 18.75 |
| 32P:250 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespac…` | `[49152, 1, 1]` | 1.00 | 59.50 |
| 64D:83 | `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, 2, 2, -2>(at:…` | `[32, 1, 1]` | 1.00 | 2.21 |
| 64D:85 | `triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0` | `[1, 1, 1]` | 0.75 | 0.77 |
| 64D:86 | `triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1` | `[1, 1, 1]` | 0.75 | 2.42 |
| 64D:87 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespac…` | `[8, 1, 1]` | 0.75 | 2.59 |
| 64D:97 | `triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0` | `[8, 1, 1]` | 0.25 | 0.80 |
| 64D:98 | `triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1` | `[8, 1, 1]` | 0.25 | 2.43 |
| 64D:99 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespac…` | `[128, 1, 1]` | 0.25 | 2.69 |
| 64P:247 | `triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0` | `[3072, 1, 1]` | 1.00 | 2.88 |
| 64P:248 | `triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1` | `[3072, 1, 1]` | 1.00 | 2.91 |
| 64P:249 | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` | `[6144, 2, 1]` | 1.00 | 18.62 |
| 64P:250 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespac…` | `[49152, 1, 1]` | 1.00 | 59.33 |

## `dsv4_lm_head`

| Population / dump line | Kernel prefix | Grid | Mean launches / forward | Median µs / launch |
| --- | --- | --- | --- | --- |
| 32D:83 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | `[2, 64, 1]` | 1.00 | 34.30 |
| 32P:257 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | `[2, 64, 1]` | 1.00 | 39.94 |
| 64D:112 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | `[2, 64, 1]` | 0.75 | 34.37 |
| 64D:121 | `nvjet_sm90_tst_128x16_64x11_2x1_v_bz_TNT` | `[2, 64, 1]` | 0.25 | 34.88 |
| 64P:257 | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` | `[2, 64, 1]` | 1.00 | 39.94 |

## Corrections and call binding

- Shared decode uses scale + FlashInfer swapAB FP8 GEMM for each projection; L=768 uses native per-token group quantization plus DeepGEMM. The up/down templates have output/input widths 4096/4096 and 4096/2048 respectively. Activation has grid L. The unscoped scale aggregate also includes attention.
- First-pre has the TF32 prenorm GEMM even in decode (`32D:71`, `64D:84`), then the broadcast/norm tilelang kernel. This is missing from the initial reading.
- Fused decode uses the `(L,12,8)` mHC fused kernel and `(L,1,1)` pre/norm kernel; prefill replaces the fused kernel with post + TF32 GEMM + pre/norm. The two names are timed separately with distinct attention/FFN parameter sets.
- Head and final norm run after SP all-gather on W, rather than L (`vllm/models/deepseek_v4/nvidia/model.py:1197-1211`). The W=6144 grids in 32P/64P agree.
- Embedding decode uses `indexSelectSmallIndex`, while prefill uses `vectorized_gather_kernel`, plus the two compiled mask kernels and `masked_fill_kernel`. The masked-fill grid follows the native W descriptor (8W in these populations). The producer calls the native index function, native compiled masking, embedding method, and masked fill; the constructor and forward otherwise require the live TP group and would reduce. The local TP0 shard excludes that collective (`vocab_parallel_embedding.py:279-282,361-402,487-506`).
- LM head is the unquantized local vocabulary GEMM. The b=2 dump uses `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT`; 64D additionally shows the b=9 `128x16` variant (`64D:121`). No logits all-gather is included.
- Final norm selects native `RMSNorm.forward_cuda` with the vLLM IR `vllm_c` provider; that calls `_C.rms_norm` (`layernorm.py:74-114`; `vllm/kernels/vllm_c.py:23-43`). Provider selection, tilelang/Inductor warmup, and actual names/grids remain GPU checks.

## Full trace names and clamp activation

Exactly one decode and one prefill TP0/DP0 rank file from each P2 run were read under a timeout and the 30 GiB virtual-memory limit. No further rank files were read. The retained extraction is `b13/native_full_names.json` in the execution scratch. Its line references are decompressed JSON lines. All four full activation names have the template tail `true, true, true, false`. Native `csrc/libtorch_stable/activation_kernels.cu:98-102` names those booleans `act_first`, `use_vec`, `HAS_CLAMP`, `use_256b`: this is the clamp variant, in agreement with `SiluAndMulWithClamp(swiglu_limit)` (`model.py:140-141`).

| Full reference trace | Activation JSON line | Template tail |
| --- | --- | --- |
| `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792942701901162313.pt.trace.json.gz` | 987283 | `true, true, true, false` |
| `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces/prefill/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792942682770529608.pt.trace.json.gz` | 1678504 | `true, true, true, false` |
| `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792946239761662380.pt.trace.json.gz` | 1083386 | `true, true, true, false` |
| `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces/prefill/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792946186223519831.pt.trace.json.gz` | 1807301 | `true, true, true, false` |

G2h has not been built or run. After the lead commits B13, replace `source_revision`, build and submit the package, then compare each saved `profiler_traces/<operator>_<rows>/` scope against these references. Kernel-only costs remain sums of owned kernels, with CUDA-event rows for validation.

# Qwen3.5 GDN profiler

This profiler measures the gated-delta-network layer used by the Qwen3.5
family, including the hybrid Qwen3.8 checkpoint. It constructs vLLM's real
`QwenGatedDeltaNetAttention` module, disables the checkpoint's unrelated Quark
quantizer, and initializes synthetic BF16 projection weights. `A_log` and the
recurrent state remain FP32, matching the runtime contract. Launch TP profiles
with `torchrun --nproc-per-node=<TP>` and the matching
`--tensor-parallel-size`; every timing sample is reduced to the slowest rank.

Run it inside the pinned ROCm container described in
`docs/profiling/ROCM_MI355X.md`. For the AITER dispatch, set
`VLLM_ROCM_USE_AITER=1` before Python imports vLLM. The output defaults to:

```text
data/profiling/compute/{DEVICE}/{MODEL}/gdn.csv
```

Configure Frontier's eager predictor with `gdn_input_file`; use
`gdn_kernel_only_input_file` for a separately collected kernel-only dataset.
The predictor rejects rows whose architecture, quantization, dimensions,
state layout, or complete runtime stack do not agree.

Before emitting a row, the profiler compares the complete vLLM layer with its
input-projection, GDN-core, and output-projection decomposition, including both
BF16 convolution state and FP32 recurrent state. A failed equivalence or
non-finite output aborts profiling. Cache block zero is reserved as vLLM's null
block; real request state begins at block one. Do not remove this offset—the
causal-convolution kernel intentionally skips requests assigned to block zero.

For serving-faithful SGLang measurements with checkpoint weights and graph
decode, import `sglang.benchmark.one_batch` traces with
`python -m frontier.profiling.gdn.sglang_trace`. This writes
`gdn_kernel_only.csv` and deliberately excludes the decoder input norm and
post-GDN TP all-reduce from the GDN operator samples.

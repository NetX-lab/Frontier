# ROCm MI355X Profiling

This workflow profiles Frontier operators on an eight-GPU AMD Instinct MI355X
UBB node (`gfx950`). It uses vLLM's ROCm attention implementation, AITER for
MXFP4 MoE, and PyTorch distributed collectives backed by RCCL.

## Validated stack

The initial port was validated with this immutable image:

```text
vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032
```

The image contains PyTorch `2.12.0+git6bbd260`, HIP `7.2.53211`, vLLM
`0.28.0`, Triton `3.7.1`, and AITER. Pin the digest when comparing results;
changing the image also changes the kernel stack under test.

## Start the profiling container

Run this from a Frontier checkout. Replace `/path/to/models` only if a later
workflow needs model weights; the Frontier operator profilers synthesize their
inputs and weights from the checked-in model configuration.

```bash
IMAGE=vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032

docker run --rm -it \
  --name frontier-mi355x-dev \
  --network host \
  --ipc host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e PYTHONPATH=/workspace/Frontier \
  -v "$PWD:/workspace/Frontier" \
  -v /path/to/models:/models:ro \
  -w /workspace/Frontier \
  --entrypoint bash \
  "$IMAGE"
```

Verify that PyTorch sees the expected platform and device count:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("devices:", torch.cuda.device_count())
print("device 0:", torch.cuda.get_device_name(0))
PY
```

PyTorch intentionally retains its `torch.cuda` API and the distributed backend
name `nccl` on ROCm. The runtime routes those calls to HIP and RCCL.

## Device visibility

Use the native ROCm variable for subprocess profiling:

```bash
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
```

Do not set `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` to the same physical
nonzero ID. Some runtime versions apply the second variable after the first and
hide the selected device through a second index remapping.

## Qwen3.8 operator smokes

The checked-in `Qwen3.8-2.4T-A95B-Quark-MXFP4` configuration describes the
92-layer, 512-expert checkpoint. Its base activations are BF16; only
`moe_grouped_gemm` is marked MXFP4.

Profile common and shared-expert linear operations:

```bash
python -m frontier.profiling.linear_op.main \
  --disable_ray \
  --num_gpus 1 \
  --device mi355x \
  --models Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --num_tensor_parallel_workers 1 \
  --num_tokens_list 1 16 \
  --is_moe \
  --profile_method cuda_event \
  --output_dir data/profiling \
  --yes
```

Profile the model's full-attention operator:

```bash
python -m frontier.profiling.attention.main \
  --disable_ray \
  --num_gpus 1 \
  --device mi355x \
  --models Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --num_tensor_parallel_workers 1 \
  --attention_backend VLLM_ROCM \
  --max_model_len 16 \
  --max_seq_len 16 \
  --min_batch_size 1 \
  --max_batch_size 1 \
  --profile_only_prefill \
  --profile_method cuda_event \
  --output_dir data/profiling \
  --yes
```

Profile an isolated Qwen3.8 gated-delta-network layer with the real vLLM
Qwen3.5 module, synthetic BF16 weights, BF16 convolution state, and FP32
recurrent state:

```bash
export VLLM_ROCM_USE_AITER=1

python -m frontier.profiling.gdn.main \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --model-path /models/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --device mi355x \
  --prefill-seq-lens 16 128 512 2048 \
  --decode-batch-sizes 1 8 32 128 \
  --decode-context-len 512 \
  --continuation-context-len 512 \
  --include-continuation-prefill \
  --include-mixed \
  --max-model-len 4096 \
  --max-batch-size 129 \
  --output-dir data/profiling
```

The profiler runs vLLM in eager mode, records the input projections, phase-
specific GDN core, output projection, and complete layer separately, and
checks numerical equivalence before writing a CSV. The runtime stack and
standard/AITER dispatch choice are part of every row's compatibility
contract. Do not merge measurements produced by different runtime stack
signatures.

For a serving-faithful TP=8 profile, SGLang's static-batch runner can capture
the real Quark-weighted Qwen3.8 kernels. The importer recognizes the filenames
written by `sglang.benchmark.one_batch`, excludes the decoder input norm and
the post-GDN TP all-reduce, and writes the kernel-only predictor dataset:

```bash
python -m frontier.profiling.gdn.sglang_trace \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --device mi355x \
  --tensor-parallel-size 8 \
  --runtime-stack-signature 'sglang=<commit>;torch=<version>;rocm=<version>;aiter=<version>' \
  --trace /path/outside/repo/prefill.trace.json.gz \
  --trace /path/outside/repo/decode.trace.json.gz \
  --output-dir data/profiling
```

The output is
`data/profiling/compute/mi355x/Qwen3.8-2.4T-A95B-Quark-MXFP4/gdn_kernel_only.csv`.
The trace must contain a whole number of 69-layer GDN passes. Decode graph
profiling may omit a boundary replay, so the importer validates complete model
passes rather than requiring the requested profiler-step count exactly.

Enable AITER before importing vLLM, then profile the routed experts with the
node's natural EP=8 distribution (64 local experts per device):

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1

python -m frontier.profiling.moe.main \
  --disable_ray \
  --num_gpus 1 \
  --device mi355x \
  --models Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --num_tensor_parallel_workers 1 \
  --expert_parallel_sizes 8 \
  --num_tokens_list 1 \
  --load_distributions uniform \
  --num_samples_per_distribution 1 \
  --profile_method cuda_event \
  --output_dir data/profiling \
  --yes
```

The resulting MoE row must report
`moe_grouped_gemm_backend=vllm_aiter_mxfp4` and a quantization signature ending
in `operations=moe_grouped_gemm`. A BF16 `profiling_precision` value describes
the base activation dtype; the quantization signature and backend identify the
selective MXFP4 kernel.

## RCCL collective smoke

The local runner does not require Ray. To profile a two-GPU BF16 all-reduce:

```bash
export ROCR_VISIBLE_DEVICES=0,1
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

python -m frontier.profiling.collectives.main \
  --disable_ray \
  --num_gpus 2 \
  --num_workers_per_node_combinations 2 \
  --max_collective_size 1024 \
  --collective all_reduce \
  --precision BF16 \
  --output_dir data/profiling
```

The CSV `size` column is bytes, so 1,024 BF16 elements are recorded as 2,048
bytes. Frontier uses Kineto to capture the RCCL kernel time.

## Current Qwen3.8 fidelity boundary

For reproducible TP=8 batch capture, held-out GDN validation and native
predictor replay, see [MI355X correlation](MI355X_CORRELATION.md).

Frontier models Qwen3.8 as the exact 92-layer hybrid schedule: layers 3, 7,
..., 91 use full attention and the other 69 layers use gated delta networks.
Prediction aggregates each layer family independently. Memory planning counts
the dense-attention KV cache only for the 23 full-attention layers and reserves
the fixed convolution/recurrent GDN state for every active request.

The standalone wrapper reserves cache block zero because vLLM defines it as
`NULL_BLOCK_ID`; real request state uses blocks `1..N`. Assigning a request to
zero makes the causal-convolution kernel skip that request and invalidates both
correctness and timing results.

Record numerical-equivalence checks, selected kernel backends, fallback paths,
runtime commit identities, tuning availability and all measured timings in the
external experiment directory. Do not commit generated traces, profiles,
latency ratios or backend performance conclusions to the source repository.

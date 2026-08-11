# MI355X (ROCm) Profiling Cookbook

Everything discovered getting Frontier's profiling pipeline working end-to-end
on a shared AMD MI355X (CDNA4, gfx950) server. Companion to
[HARDWARE_COOKBOOK.md](HARDWARE_COOKBOOK.md) — that doc is hardware-agnostic;
this one is the concrete, hard-won MI355X/ROCm path, written front-loaded with
the fixes already known rather than as a chronological debugging log.

## The short version

```bash
# 1. Confirm hardware + driver
rocm-smi
rocminfo | grep -i gfx        # expect: gfx950

# 2. Use AMD's combined ROCm+PyTorch+vLLM image — don't hand-build this stack
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 8G -v ~/frontier-work:/workspace -w /workspace \
  --name mi355x-frontier-vllm \
  rocm/vllm:rocm7.12.0_gfx950-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0 bash

# 3. Inside the container: verify, then set up Frontier
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python3 -c "import vllm; print(vllm.__version__)"
cd /workspace/Frontier
export PYTHONPATH=$PWD
pip install -e ".[test]"

# 4. The two env vars that make Frontier's profiling scripts work on ROCm
export CUDA_VISIBLE_DEVICES=0                        # yes, this exact name — see gotcha #2
export FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1 # needed for linear_op only — see gotcha #4

# 5. Profile
DEVICE=mi355x MODEL=meta-llama/Llama-2-7b-hf TP_SIZES="1 2 4 8" PROFILE_METHOD=cuda_event \
  bash examples/profiling/profile_linear_op.sh

DEVICE=mi355x MODEL=meta-llama/Llama-2-7b-hf TP_SIZES="1 2 4 8" \
  ATTENTION_BACKEND=TORCH_SDPA PROFILE_METHOD=cuda_event \
  bash examples/profiling/profile_attention_chunked_prefill.sh

# MoE profiling additionally needs a patched frontier/profiling/moe/moe_impl.py
# (vLLM import-path fix, see gotcha #6) and a routing-path switch:
DEVICE=mi355x MODEL=Qwen3-30B-A3B-tiny TP_SIZES="1 2" EP_SIZES="1 2" \
  ROUTING_RUNTIME_PATH=standard_fused_topk PROFILE_METHOD=cuda_event \
  bash examples/profiling/profile_moe.sh
```

Resuming later — same box, same install already done:
```bash
sudo docker start -ai mi355x-frontier-vllm
```

## The gotchas, and why each fix is what it is

### 1. `gfx950` needs ROCm 7.0+ — a stock `pip install torch` (or an older ROCm-tagged wheel) silently fails
Symptom: `rocm-smi` shows the GPUs fine, permissions are correct, but
`torch.cuda.is_available()` is `False` and `torch.cuda.get_device_name(0)`
raises `RuntimeError: No HIP GPUs are available`. This combination —
`rocm-smi` working, HIP runtime seeing zero devices — means the installed
PyTorch build has no compiled kernels for this architecture, not a permissions
or driver problem. MI350/MI355X (gfx950) support landed in ROCm 7.0; anything
older (we hit this with `torch 2.5.1+rocm6.2`) will never see the GPU no
matter how correct the rest of your setup is.

Don't try to fix this with a plain `pip install torch --index-url .../rocm6.x`
— gfx950 wheels are handled specially. Either use AMD's own wheel index with
the `device-gfx950` extra (`--index-url https://repo.amd.com/rocm/whl-multi-arch/
"torch[device-gfx950]==..."`), or — what we actually used, and recommend —
AMD's prebuilt Docker image, which bundles a version combination AMD has
already validated together rather than one you're assembling yourself for
launch-day hardware.

### 2. Frontier's profiling scripts check `CUDA_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES` — even on AMD
`frontier/profiling/{linear_op,attention,moe}/main.py` each have their own
`_get_available_gpus()` helper (three separate copies of the same code) that:
1. Checks `os.environ.get("CUDA_VISIBLE_DEVICES", "")` first — if set, uses it directly.
2. Otherwise, shells out to `nvidia-smi --query-gpu=index ...` to discover GPUs.

Step 2 is a hard NVIDIA-only assumption with no ROCm equivalent wired in —
worth an upstream fix at some point (falling back to `rocm-smi`/`amd-smi`),
but for now, step 1's escape hatch means the fix is a one-line env var:
```bash
export CUDA_VISIBLE_DEVICES=0
```
Yes, that literal NVIDIA-named variable, on AMD hardware — this code path
just reads it as a plain string, it never calls anything CUDA-specific with it.

`scripts/profile_true_mixed_batch.sh` and `scripts/profile_deepseek_aiter_mla.sh`
already export both `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` (derived
from their `NUM_GPUS`/`GPU_IDS` config) before invoking `attention.main` — this
gotcha is baked in, not something you need to remember when using them.

### 3. FlashInfer (attention profiling's real kernel path) doesn't exist on ROCm at all
There are only two attention backends in `frontier/profiling/attention/backends/`:
`FLASHINFER` (real kernel, but NVIDIA/CUDA-only — hard `ImportError` on ROCm)
and `NO_OP` (runs everywhere, but returns an empty tensor — collects zero real
attention cost by design, not a real substitute).

We wrote a third backend, `TORCH_SDPA`
(`frontier/profiling/attention/backends/torch_sdpa_attention_wrapper.py`),
using only `torch.nn.functional.scaled_dot_product_attention` — portable to
ROCm since SDPA is core PyTorch, no vendor-specific kernel. Use it explicitly:
```bash
ATTENTION_BACKEND=TORCH_SDPA bash examples/profiling/profile_attention_chunked_prefill.sh
```
Real caveats (documented in the module docstring too): it loops per-request
rather than running one fused ragged-batch kernel, so it's slower than a real
paged-attention kernel would be — treat its numbers as a correctness-first
reference baseline, not peak achievable MI355X attention performance. Verify
a non-chunked (single-prefill-chunk) case looks sane before trusting chunked
rows, since the causal-masking logic for chunked prefill relies on PyTorch
SDPA's bottom-right-aligned causal semantics.

### 4. vLLM API drift: `get_rope()` signature mismatch (linear_op profiling only)
Symptom: `TypeError: get_rope() got an unexpected keyword argument 'rotary_dim'`,
raised from `frontier/profiling/common/layers/rotary_embedding.py` calling
`vllm.model_executor.layers.rotary_embedding.get_rope(...)`. The vLLM dev
build bundled in the ROCm image (`0.16.1.dev10+...`) has evidently changed
this function's signature since Frontier's wrapper code was written against it.

Rather than patching around a moving-target upstream API, use the fallback
that's already built into that same file:
```bash
export FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1
```
This skips the vLLM call entirely and uses a local pure-PyTorch
`RotaryEmbedding` implementation instead — portable, and avoids depending on
vLLM's RoPE API matching whatever Frontier expected. Only needed for
**linear_op** profiling — attention profiling doesn't import
`rotary_embedding.py` at all, so this one doesn't apply there. There's also a
broader `FRONTIER_PROFILING_FORCE_TORCH_FALLBACK` (forces more than just RoPE)
— we didn't need it since FP16 profiling doesn't exercise the FP8-quantization
vLLM paths elsewhere in the codebase; reach for it only if you hit a *different*
vLLM-API-mismatch error elsewhere and want the broadest possible workaround.

### 5. Container GPU passthrough flags matter, and get forgotten on restart
The `--device=/dev/kfd --device=/dev/dri --group-add=video` flags are what
make the container see the GPU at all — easy to leave out if you `docker run`
from memory later. Avoid re-typing them by naming the container once and
reusing it:
```bash
sudo docker start -ai mi355x-frontier-vllm    # not `docker run` again
```

### 6. MoE profiling: more vLLM API drift, in `moe_impl.py` this time — needs both an import patch and a routing-path switch
Two separate issues, both from the same underlying cause (this vLLM version
being much newer than whatever Frontier's MoE profiling code was written
against):

**a) `routing_runtime_path=uniform_topk` requires vLLM's fused routing kernel,
`standard_fused_topk` doesn't.** The release wrapper defaults to
`uniform_topk`; switch it explicitly:
```bash
ROUTING_RUNTIME_PATH=standard_fused_topk bash examples/profiling/profile_moe.sh
```
This isn't just a workaround — `standard_fused_topk` pairs with Frontier's
default `--replica_config_moe_routing_mode simulation` (realistic routing),
while `uniform_topk` only matters if you specifically plan to simulate with
`uniform_legacy`/`uniform_random` routing modes later. Pick based on what
you'll actually simulate, not just what currently runs.

**b) `moe_impl.py`'s vLLM imports point at the old locations.** Two of the
four imported names moved: `fused_topk` (moved into a new `router/`
submodule) and `get_config_dtype_str` (renamed `_get_config_dtype_str`,
moved to `fused_moe/config.py`). The other two (`try_get_optimal_moe_config`,
`moe_align_block_size`) are unaffected — they only *looked* broken because a
single missing name fails Python's whole `from x import (a, b, c)` statement.
Fixed with a nested try/except in `moe_impl.py` that tries the old locations
first, falls back to the new ones — see the diff already applied to this repo
(`frontier/profiling/moe/moe_impl.py`, the `HAS_VLLM` import block).

Separately, `moe_vllm_kernel.py`'s `invoke_fused_moe_kernel` import is
**still broken** and genuinely gone from this vLLM version (not moved,
verified across the whole package) — but that one only gates
`--enable_load_imbalance` (the wrapper script passes `--disable_load_imbalance`
by default), so it doesn't block a normal profiling run. Leave it broken
until you specifically need load-imbalance profiling.

### 7. Attention training needs decode data — the release wrapper only profiles prefill
`profile_attention_chunked_prefill.sh` hardcodes `--profile_only_prefill`
unconditionally (not exposed as a toggle), so `attention.csv` collected via
that script has **zero decode rows**. Training fails with `Training data for
model attn_decode is empty` the moment it gets to that model. Since
`attention/main.py` writes its CSV with plain `df.to_csv(output_file,
index=False)` — no append mode, silently overwrites — don't just re-run
profiling with `--profile_only_decode` afterward, or you'll lose the prefill
data collecting the decode data. Instead, invoke `attention.main` directly
(bypassing the wrapper) with **neither** `--profile_only_prefill` nor
`--profile_only_decode` set — that profiles both phases in one pass, one
clean CSV, no merge required:
```bash
python3 -m frontier.profiling.attention.main --disable_ray \
  --models meta-llama/Llama-2-7b-hf --num_gpus 1 --max_seq_len 128 \
  --num_tensor_parallel_workers 1 2 4 8 --max_pipeline_parallel_size 1 \
  --attention_backend TORCH_SDPA --block_size 16 --min_batch_size 1 --max_batch_size 2 \
  --fixed_chunked_prefill_size 64 --enable_chunked_prefill_grid_search \
  --device mi355x --profile_method cuda_event \
  --output_dir /workspace/Frontier/data/profiling
```

Both `scripts/profile_true_mixed_batch.sh` and `scripts/profile_deepseek_aiter_mla.sh`
already invoke `attention.main` directly for exactly this reason (neither sets
`--profile_only_prefill`/`--profile_only_decode`) — this gotcha doesn't apply
when using them.

### 8. Training CLI defaults don't match what the release wrapper scripts profiled with
Both the `routing_runtime_path` and `gating_runtime_context` training flags
default to values (`uniform_topk` in one case; `standalone_legacy` for
gating) that don't match what `profile_moe.sh` actually profiled with
(`standard_fused_topk` / `prefill_hot`, its own defaults). Symptom:
`ValueError: No MoE gating profiling rows match the requested
gating_runtime_context='standalone_legacy'. Available contexts:
['prefill_hot']`. Fix is just passing both explicitly to match whatever you
profiled with:
```bash
--routing_runtime_path standard_fused_topk --gating_runtime_context prefill_hot
```
General lesson for this whole pipeline: profiling-time and training-time
flags need to agree on routing/gating context, same as the
`num_tensor_parallel_workers` used in profiling has to match
`--tensor_parallel_size` used in training. Mismatches fail loudly (empty
dataset after filtering) rather than silently training on the wrong slice.

**c) Fidelity caveat, not a bug**: expect a console warning like
`Using default MoE config. Performance might be sub-optimal! Config file not
found at .../configs/E=16,N=768,device_name=AMD_Radeon_Graphics.json`. vLLM
ships pre-tuned grouped-GEMM kernel configs per device name, and doesn't have
one for this device string yet — it falls back to an untuned default
config rather than failing. The profiling run still completes and produces
real measured numbers, but `moe_grouped_gemm` timings collected this way are
likely somewhat slower than what a properly MI355X-tuned kernel config would
achieve. Worth knowing if grouped-GEMM latency specifically matters for
whatever you're comparing.

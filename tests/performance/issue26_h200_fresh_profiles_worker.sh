#!/usr/bin/env bash
# Collect new operator profiles for the single 4096/1024 H200 case.
set -euo pipefail
source "$(dirname "$0")/../e2e/issue26_h200_environment_probe.sh" "${1:?Provide a fresh output directory.}"
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1" no_proxy="$no_proxy,127.0.0.1,localhost,::1"
export PATH="/usr/local/nvidia/bin:$PATH"
export PYTHONPATH="$REPO_ROOT:/data/ycfeng/tmp/vLLM-BS"
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_FRONTIER_INSTRUMENTATION=0
export TRITON_CACHE_DIR="$TMPDIR/$(basename "$PROBE_ROOT")/triton-cache"
export CUDA_CACHE_PATH="$TMPDIR/$(basename "$PROBE_ROOT")/cuda-cache"
COMMON=(--disable_ray --models qwen3-a3b-30b-moe --device h200 --num_gpus 8
        --precision BF16 --profile_method cuda_event --output_dir "$PROBE_ROOT/profiles" --yes)
TOKENS=(1 2 4 8 16 24 32 48 64 80 96 100 128)
for BASE in 4096 8192 12288; do
  for OFFSET in 0 16 32 48 64 80 96 112 128; do TOKENS+=("$((BASE + OFFSET))"); done
done
TOKENS+=(16384)
set -x
"$PY" -m frontier.profiling.linear_op.main "${COMMON[@]}" \
  --num_tensor_parallel_workers 4 --attn_tp 4 --ffn_tp 4 --moe_tp 1 --is_moe \
  --max_tokens 16384 --num_tokens_list "${TOKENS[@]}" > "$PROBE_ROOT/linear.log" 2>&1
"$PY" -m frontier.profiling.attention.main "${COMMON[@]}" \
  --num_tensor_parallel_workers 4 --max_pipeline_parallel_size 1 \
  --max_model_len 16384 --max_seq_len 4096 --block_size 16 \
  --attention_backend FLASHINFER --fixed_chunked_prefill_size -1 \
  --batch_size_list 1 2 3 4 8 16 32 48 64 96 100 \
  --decode_kv_cache_size_list 4096 4352 4608 4864 5119 \
  --enable_true_mixed --true_mixed_prefill_batch_sizes 1 2 3 \
  --true_mixed_prefill_chunk_sizes 4096 \
  --true_mixed_decode_batch_sizes 1 2 4 8 16 32 48 64 96 \
  --true_mixed_decode_kv_cache_sizes 4096 4608 5119 \
  --true_mixed_prefill_kv_cache_size 0 > "$PROBE_ROOT/attention.log" 2>&1
"$PY" -m frontier.profiling.moe.main "${COMMON[@]}" \
  --num_tensor_parallel_workers 1 --expert_parallel_sizes 8 \
  --routing_runtime_path standard_fused_topk --gating_runtime_context prefill_hot \
  --enable_load_imbalance --load_distributions uniform skewed extremely_skewed \
  --num_samples_per_distribution 3 --max_tokens 32768 --num_tokens_list "${TOKENS[@]}" 24576 32768 \
  > "$PROBE_ROOT/moe.log" 2>&1
echo FRESH_PROFILING_EXECUTION_COMPLETE

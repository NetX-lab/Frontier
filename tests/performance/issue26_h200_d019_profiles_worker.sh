#!/usr/bin/env bash
# Validate the corrected MoE path, then measure exact inputs and collectives serially.
set -euo pipefail
PHASE="${2:-all}"
case "$PHASE" in all|attention_communication|timing_context|linear_context) ;; *) exit 1 ;; esac
source "$(dirname "$0")/../e2e/issue26_h200_environment_probe.sh" "${1:?Provide a fresh output directory.}"
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1" no_proxy="$no_proxy,127.0.0.1,localhost,::1"
export PATH="/usr/local/nvidia/bin:$PATH"
export PYTHONPATH="$REPO_ROOT:/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908"
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_FRONTIER_INSTRUMENTATION=0 VLLM_ALL2ALL_BACKEND=naive
export TRITON_CACHE_DIR="$TMPDIR/d019-profiles-$(basename "$(dirname "$PROBE_ROOT")")/triton-cache"
export CUDA_CACHE_PATH="$TMPDIR/d019-profiles-$(basename "$(dirname "$PROBE_ROOT")")/cuda-cache"
git config --global --add safe.directory /data/ycfeng/tmp/issue26-vllm-diagnostics-20260908
test "$(git -C /data/ycfeng/tmp/issue26-vllm-diagnostics-20260908 rev-parse HEAD)" = "${ISSUE26_DIAGNOSTIC_VLLM_COMMIT:?Pin the reviewed diagnostic checkout.}"
test -z "$(git -C /data/ycfeng/tmp/issue26-vllm-diagnostics-20260908 status --porcelain)"
git -C "$REPO_ROOT" diff -- frontier/profiling/moe/moe_vllm_kernel.py > "$PROBE_ROOT/frontier_moe_patch.diff"
set -x
if [[ "$PHASE" == all ]]; then
"$PY" -m pytest tests/unit/test_moe_fused_expert_numerical_parity.py -q -p no:cacheprovider > "$PROBE_ROOT/numerical.log" 2>&1
"$PY" tests/performance/issue26_moe_exact_profile.py --output-dir "$PROBE_ROOT/moe" \
  --model qwen3-a3b-30b-moe --ep-size 8 --tokens 4095 4096 4097 \
  --ep-ranks 0 1 7 --samples 20 --seeds 0 1 2 > "$PROBE_ROOT/moe.log" 2>&1
for REPEAT in 1 2 3; do
  "$PY" -m frontier.profiling.linear_op.main --disable_ray --models qwen3-a3b-30b-moe \
    --device h200 --num_gpus 1 --precision BF16 --profile_method cuda_event \
    --output_dir "$PROBE_ROOT/linear-$REPEAT" --yes \
    --num_tensor_parallel_workers 4 --attn_tp 4 --ffn_tp 4 --moe_tp 1 --is_moe \
    --max_tokens 4096 --num_tokens_list 4096 > "$PROBE_ROOT/linear-$REPEAT.log" 2>&1
done
fi
if [[ "$PHASE" == timing_context || "$PHASE" == linear_context ]]; then
if [[ "$PHASE" == timing_context ]]; then
"$PY" -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/performance/issue26_h200_collective_microbenchmark.py \
  --output "$PROBE_ROOT/communication" \
  --tp-message-bytes 8388608 12582912 16777216 25165824 33554432 \
  --capture-tp-kernels > "$PROBE_ROOT/communication.log" 2>&1
fi
"$PY" tests/performance/issue26_linear_timing_context.py \
  --output "$PROBE_ROOT/timing-context" > "$PROBE_ROOT/timing-context.log" 2>&1
else
"$PY" tests/performance/issue26_attention_exact_profile.py --output "$PROBE_ROOT/attention" \
  --model qwen3-a3b-30b-moe --tokens 4096 --tp 4 --blocks 310809 \
  --max-model-len 16384 --repeats 3 > "$PROBE_ROOT/attention.log" 2>&1
"$PY" -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/performance/issue26_h200_collective_microbenchmark.py \
  --output "$PROBE_ROOT/communication" > "$PROBE_ROOT/communication.log" 2>&1
fi
set +x
echo D019_PROFILES_EXECUTION_COMPLETE

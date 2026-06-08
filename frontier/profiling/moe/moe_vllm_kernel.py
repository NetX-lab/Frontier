"""
vLLM fused MoE kernel wrapper for accurate load imbalance profiling.

This module provides a wrapper around vLLM's optimized fused_moe_kernel,
which is necessary for accurately capturing the performance impact of
expert load imbalance in grouped GEMM operations.

Design rationale:
- Per-expert loop implementation (moe_impl.py) cannot capture true load imbalance effects
- vLLM's fused_moe_kernel uses optimized grouped GEMM with real parallelism
- Only used when --enable_load_imbalance is specified for accurate profiling

Supported vLLM versions:
- vLLM 0.10.x: Current supported version with extended invoke_fused_moe_kernel API

Note: vLLM 0.3.x support has been removed. Please use vLLM >= 0.10.0.
"""

import torch
import triton
import triton.language as tl
from typing import Dict, List, Optional, Tuple

VLLM_AVAILABLE = False
VLLM_VERSION = None
VLLM_API_VERSION = None
FP8_QUANT_AVAILABLE = False

try:
    import vllm
    VLLM_VERSION = vllm.__version__

    # Import vLLM 0.10.x functions
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_moe_kernel,
        invoke_fused_moe_kernel,
        moe_align_block_size,
        try_get_optimal_moe_config,
        get_config_dtype_str,
    )

    VLLM_API_VERSION = "0.10.x"
    VLLM_AVAILABLE = True
    print(f"vLLM {VLLM_VERSION} loaded successfully (API: {VLLM_API_VERSION})")

    # Try to import FP8 quantization utilities
    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
        from vllm.utils.deep_gemm import (
            per_block_cast_to_fp8 as _per_block_cast_to_fp8,
        )
        FP8_QUANT_AVAILABLE = True
        print("FP8 quantization utilities loaded successfully")
    except ImportError:
        FP8_QUANT_AVAILABLE = False
        _per_block_cast_to_fp8 = None
        print("Warning: FP8 quantization utilities not available")

except ImportError as e:
    print(f"vLLM import error: {e}")
    VLLM_AVAILABLE = False
    VLLM_API_VERSION = None
    _per_block_cast_to_fp8 = None
    print("Warning: vLLM >= 0.10.0 required. Load imbalance profiling will not work.")


def check_vllm_available():
    """Check if vLLM is available for fused MoE kernel."""
    return VLLM_AVAILABLE


def check_fp8_available():
    """Check if FP8 quantization is available."""
    return FP8_QUANT_AVAILABLE


# FP8 E4M3 max value
FP8_E4M3_MAX = 448.0


def _validate_block_shape(
    block_shape: Optional[List[int]],
) -> Optional[Tuple[int, int]]:
    if block_shape is None:
        return None
    if len(block_shape) != 2:
        raise ValueError(
            f"block_shape must have 2 elements, got {len(block_shape)}."
        )
    block_n, block_k = block_shape
    if block_n <= 0 or block_k <= 0:
        raise ValueError(
            f"block_shape must be positive, got {block_shape}."
        )
    return block_n, block_k


def quantize_weights_to_fp8(
    weights: torch.Tensor,
    per_channel: bool = False,
    block_shape: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert weights to FP8 and compute scales.

    Args:
        weights: Weight tensor to quantize
        per_channel: Whether to use per-channel quantization
        block_shape: Block dimensions for block-wise quantization (e.g., [128, 128])

    Returns:
        Tuple of (quantized_weights, scales)
    """
    block_dims = _validate_block_shape(block_shape)
    if block_dims is not None:
        if per_channel:
            raise ValueError(
                "Block-wise FP8 quantization does not support per-channel quantization."
            )
        if _per_block_cast_to_fp8 is None:
            raise ImportError(
                "Block-wise FP8 quantization requires vLLM deep_gemm utilities."
            )
        block_n, block_k = block_dims
        if weights.ndim == 2:
            w_fp8, w_scale = _per_block_cast_to_fp8(
                weights, block_size=[block_n, block_k]
            )
            return w_fp8, w_scale
        if weights.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D weight tensor, got shape {tuple(weights.shape)}."
            )
        w_list = []
        s_list = []
        for expert_idx in range(weights.size(0)):
            w_fp8, w_scale = _per_block_cast_to_fp8(
                weights[expert_idx].contiguous(),
                block_size=[block_n, block_k],
            )
            w_list.append(w_fp8)
            s_list.append(w_scale)
        return torch.stack(w_list), torch.stack(s_list)
    if per_channel:
        # Per-channel: scale per output channel (dim 1 for MoE weights)
        # weights shape: [num_experts, out_dim, in_dim]
        absmax = weights.abs().amax(dim=-1, keepdim=True)
        scale = absmax / FP8_E4M3_MAX
        # Avoid division by zero
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        w_fp8 = (weights / scale).clamp(
            -FP8_E4M3_MAX, FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn)
        return w_fp8, scale.squeeze(-1)
    else:
        # Per-tensor: single scale per expert
        # weights shape: [num_experts, out_dim, in_dim]
        absmax = weights.abs().amax(dim=(-2, -1), keepdim=True)
        scale = absmax / FP8_E4M3_MAX
        # Avoid division by zero
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        w_fp8 = (weights / scale).clamp(
            -FP8_E4M3_MAX, FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn)
        # Return scale as [num_experts] for per-tensor
        return w_fp8, scale.squeeze(-1).squeeze(-1)


def quantize_activations_to_fp8(
    activations: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to FP8 using per-token-group quantization.

    Args:
        activations: Activation tensor [num_tokens, hidden_dim]
        group_size: Group size for quantization (default 128)

    Returns:
        Tuple of (quantized_activations, scales)
    """
    if not FP8_QUANT_AVAILABLE:
        raise ImportError(
            "vLLM FP8 activation quantization utilities are required for FP8 "
            "MoE activation profiling. Install the profiling environment from "
            "environment_profiling.yml or use an existing vLLM environment with "
            "per_token_group_quant_fp8 available."
        )
    return per_token_group_quant_fp8(activations, group_size=group_size)


def _invoke_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict,
    A_scale: Optional[torch.Tensor] = None,
    B_scale: Optional[torch.Tensor] = None,
    use_fp8: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[List[int]] = None,
) -> None:
    """
    Wrapper for vLLM 0.10.x invoke_fused_moe_kernel.

    Args:
        A: Input activations tensor
        B: Expert weights tensor
        C: Output tensor
        topk_weights: Routing weights for top-k experts
        sorted_token_ids: Sorted token indices for kernel
        expert_ids: Expert indices for each token block
        num_tokens_post_padded: Number of tokens after padding
        mul_routed_weight: Whether to multiply by routing weights
        top_k: Number of experts per token
        config: Kernel configuration (BLOCK_SIZE_M, BLOCK_SIZE_N, etc.)
        A_scale: Activation scale for FP8 quantization
        B_scale: Weight scale for FP8 quantization
        use_fp8: Whether to use FP8 W8A8 quantization
        per_channel_quant: Whether to use per-channel quantization
        block_shape: Block dimensions for block-wise quantization
    """
    # Determine compute_type - for FP8, we accumulate in FP16/BF16
    if use_fp8:
        compute_type = tl.float16  # FP8 accumulates in FP16
    else:
        dtype = A.dtype
        if dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif dtype == torch.float16:
            compute_type = tl.float16
        elif dtype == torch.float32:
            compute_type = tl.float32
        else:
            raise ValueError(f"Unsupported dtype for fused MoE compute_type: {dtype}")

    invoke_fused_moe_kernel(
        A=A,
        B=B,
        C=C,
        A_scale=A_scale,
        B_scale=B_scale,
        B_zp=None,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=mul_routed_weight,
        top_k=top_k,
        config=config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        B_bias=None,
    )


def _run_fused_moe_iteration(
    A: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    intermediate_cache1: torch.Tensor,
    intermediate_cache2: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    config: Dict,
    expert_hidden_dim_per_partition: int,
    block_dims: Optional[Tuple[int, int]],
    A_scale: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    use_fp8: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[List[int]] = None,
) -> None:
    _invoke_kernel(
        A=A.contiguous(),
        B=w1.contiguous(),
        C=intermediate_cache1.contiguous(),
        topk_weights=topk_weights.contiguous(),
        sorted_token_ids=sorted_token_ids.contiguous(),
        expert_ids=expert_ids.contiguous(),
        num_tokens_post_padded=num_tokens_post_padded.contiguous(),
        mul_routed_weight=False,
        top_k=top_k,
        config=config,
        A_scale=A_scale,
        B_scale=w1_scale,
        use_fp8=use_fp8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
    )

    intermediate_cache1_flat = intermediate_cache1.view(-1, intermediate_cache1.shape[-1])
    intermediate_cache2_input = intermediate_cache1_flat[:, :expert_hidden_dim_per_partition].contiguous()

    intermediate_A_scale = None
    if use_fp8:
        group_size = block_dims[1] if block_dims else 128
        intermediate_cache2_input, intermediate_A_scale = quantize_activations_to_fp8(
            intermediate_cache2_input,
            group_size=group_size,
        )

    _invoke_kernel(
        A=intermediate_cache2_input,
        B=w2.contiguous(),
        C=intermediate_cache2.contiguous(),
        topk_weights=topk_weights.contiguous(),
        sorted_token_ids=sorted_token_ids.contiguous(),
        expert_ids=expert_ids.contiguous(),
        num_tokens_post_padded=num_tokens_post_padded.contiguous(),
        mul_routed_weight=True,
        top_k=1,
        config=config,
        A_scale=intermediate_A_scale,
        B_scale=w2_scale,
        use_fp8=use_fp8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
    )


def _collect_cuda_event_stats(step_fn, active_steps: int) -> Dict:
    times = []
    for _ in range(active_steps):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        step_fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times_tensor = torch.tensor(times)
    return {
        "min": float(times_tensor.min()),
        "max": float(times_tensor.max()),
        "mean": float(times_tensor.mean()),
        "median": float(times_tensor.median()),
        "std": float(times_tensor.std()),
    }


def _collect_record_function_stats(
    step_fn,
    active_steps: int,
    output_dir: Optional[str],
    operation_name: str,
) -> Dict:
    if not output_dir:
        raise ValueError(
            "output_dir is required for record_function profiling of fused MoE grouped GEMM."
        )

    from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer

    tracer = RecordFunctionTracer(output_dir)
    with tracer:
        for _ in range(active_steps):
            with torch.profiler.record_function(f"vidur_{operation_name}"):
                step_fn()

    time_stats = tracer.get_operation_time_stats()
    if operation_name not in time_stats:
        raise ValueError(
            f"RecordFunctionTracer did not capture '{operation_name}' in fused MoE profiling."
        )
    return time_stats[operation_name]


def profile_fused_moe_kernel(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    expert_hidden_dim: int,
    top_k: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    tensor_parallel_size: int = 1,
    dtype: torch.dtype = torch.float16,
    warmup_steps: int = 2,
    active_steps: int = 20,
    use_fp8: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[List[int]] = None,
    profile_method: str = "cuda_event",
    output_dir: Optional[str] = None,
    global_num_experts: Optional[int] = None,
    expert_map: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Profile vLLM's fused MoE kernel with given routing decisions.

    This function directly profiles the optimized grouped GEMM kernel,
    capturing the true performance impact of load imbalance.

    Args:
        num_tokens: Number of input tokens.
        num_experts: Number of local experts materialized in ``w1``/``w2``.
        hidden_dim: Model hidden dimension (FULL dimension, will be partitioned by TP).
        expert_hidden_dim: Expert FFN hidden dimension (FULL dimension, will be partitioned by TP).
        top_k: Number of experts selected per token.
        topk_weights: Routing weights with shape ``[num_tokens, top_k]``.
        topk_ids: Expert indices with shape ``[num_tokens, top_k]``. When ``expert_map`` is
            provided these are interpreted as global expert ids; otherwise they are local ids.
        tensor_parallel_size: Tensor parallel size (for dimension partitioning).
        dtype: Data type for computation (ignored if ``use_fp8=True``).
        warmup_steps: Number of warmup iterations.
        active_steps: Number of active profiling iterations.
        use_fp8: Whether to use FP8 W8A8 quantization.
        per_channel_quant: Whether to use per-channel quantization (only for FP8).
        block_shape: Block dimensions for block-wise quantization.
        profile_method: Profiling method (``cuda_event`` or ``record_function``).
        output_dir: Trace output directory for ``record_function`` profiling.
        global_num_experts: Global expert count for alignment when profiling EP-local workloads.
        expert_map: Optional mapping from global expert ids to local expert ids.

    Returns:
        Dictionary containing timing statistics.

    Raises:
        RuntimeError: If vLLM is not available.
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError(
            "vLLM is not available. Cannot use fused_moe_kernel for load imbalance profiling. "
            "Please install vLLM or disable --enable_load_imbalance."
        )

    if profile_method not in {"cuda_event", "record_function"}:
        raise ValueError(
            "profile_fused_moe_kernel only supports 'cuda_event' and 'record_function'. "
            f"Got profile_method={profile_method!r}."
        )

    align_num_experts = int(global_num_experts) if global_num_experts is not None else int(num_experts)
    if align_num_experts <= 0:
        raise ValueError(f"global_num_experts must be positive, got {align_num_experts}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if expert_map is not None:
        if expert_map.numel() != align_num_experts:
            raise ValueError(
                "expert_map size must match global_num_experts. "
                f"Got expert_map.numel()={expert_map.numel()}, global_num_experts={align_num_experts}"
            )
        expert_map = expert_map.to(device=topk_ids.device, dtype=torch.int32)

    device = "cuda"

    expert_hidden_dim_per_partition = expert_hidden_dim // tensor_parallel_size
    if expert_hidden_dim % tensor_parallel_size != 0:
        raise ValueError(
            f"expert_hidden_dim ({expert_hidden_dim}) must be divisible by "
            f"tensor_parallel_size ({tensor_parallel_size})"
        )

    base_dtype = torch.bfloat16 if use_fp8 else dtype
    A = torch.randn(num_tokens, hidden_dim, dtype=base_dtype, device=device)

    w1 = torch.randn(
        num_experts,
        2 * expert_hidden_dim_per_partition,
        hidden_dim,
        dtype=base_dtype,
        device=device,
    )
    w2 = torch.randn(
        num_experts,
        hidden_dim,
        expert_hidden_dim_per_partition,
        dtype=base_dtype,
        device=device,
    )

    w1_scale = None
    w2_scale = None
    A_scale = None

    block_dims = _validate_block_shape(block_shape)
    if use_fp8:
        w1, w1_scale = quantize_weights_to_fp8(
            w1,
            per_channel=per_channel_quant,
            block_shape=block_shape,
        )
        w2, w2_scale = quantize_weights_to_fp8(
            w2,
            per_channel=per_channel_quant,
            block_shape=block_shape,
        )
        group_size = block_dims[1] if block_dims else 128
        A, A_scale = quantize_activations_to_fp8(A, group_size=group_size)

    config_dtype = get_config_dtype_str(base_dtype)
    config = try_get_optimal_moe_config(
        w1_shape=w1.shape,
        w2_shape=w2.shape,
        top_k=top_k,
        dtype=config_dtype,
        M=num_tokens,
        block_shape=block_shape,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        align_num_experts,
        expert_map=expert_map,
    )

    output_dtype = base_dtype
    intermediate_cache1 = torch.empty(
        num_tokens,
        top_k,
        w1.shape[1],
        device=device,
        dtype=output_dtype,
    )
    intermediate_cache2 = torch.empty(
        num_tokens,
        top_k,
        hidden_dim,
        device=device,
        dtype=output_dtype,
    )

    def _step() -> None:
        _run_fused_moe_iteration(
            A=A,
            w1=w1,
            w2=w2,
            intermediate_cache1=intermediate_cache1,
            intermediate_cache2=intermediate_cache2,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            top_k=top_k,
            config=config,
            expert_hidden_dim_per_partition=expert_hidden_dim_per_partition,
            block_dims=block_dims,
            A_scale=A_scale,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            use_fp8=use_fp8,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        )

    for _ in range(warmup_steps):
        _step()
    torch.cuda.synchronize()

    if profile_method == "record_function":
        return _collect_record_function_stats(
            step_fn=_step,
            active_steps=active_steps,
            output_dir=output_dir,
            operation_name="moe_grouped_gemm",
        )

    return _collect_cuda_event_stats(
        step_fn=_step,
        active_steps=active_steps,
    )


def generate_expert_weights(
    num_experts: int,
    hidden_dim: int,
    expert_hidden_dim: int,
    dtype: torch.dtype = torch.float16,
    use_gated: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random expert weights for profiling.
    
    Args:
        num_experts: Number of experts
        hidden_dim: Model hidden dimension
        expert_hidden_dim: Expert FFN hidden dimension
        dtype: Data type
        use_gated: Whether to use gated FFN (SwiGLU)
    
    Returns:
        Tuple of (w1, w2) weight tensors
    """
    device = "cuda"
    
    if use_gated:
        # Gated FFN: w1 produces 2x intermediate_size
        w1 = torch.randn(
            num_experts, 2 * expert_hidden_dim, hidden_dim,
            dtype=dtype, device=device
        )
    else:
        w1 = torch.randn(
            num_experts, expert_hidden_dim, hidden_dim,
            dtype=dtype, device=device
        )
    
    w2 = torch.randn(
        num_experts, hidden_dim, expert_hidden_dim,
        dtype=dtype, device=device
    )
    
    return w1, w2


if __name__ == "__main__":
    # Simple test
    print("Testing vLLM fused MoE kernel wrapper...")
    
    if not check_vllm_available():
        print("❌ vLLM not available, skipping test")
        exit(1)
    
    # Test configuration
    num_tokens = 1024
    num_experts = 8
    hidden_dim = 4096
    expert_hidden_dim = 11008
    top_k = 2
    
    # Generate routing data (uniform distribution)
    from frontier.profiling.moe.load_distribution import generate_expert_routing
    
    topk_weights, topk_ids = generate_expert_routing(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        load_distribution="uniform",
        seed=42
    )

    print(topk_ids)

    # Count the token number of each expert
    from frontier.profiling.moe.load_distribution import compute_expert_token_counts
    expert_token_counts = compute_expert_token_counts(topk_ids, num_experts)
    print(f"Expert token counts: {expert_token_counts}")
    
    # Profile
    print(f"Profiling with {num_tokens} tokens, {num_experts} experts, top_k={top_k}")
    stats = profile_fused_moe_kernel(
        num_tokens=num_tokens,
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        expert_hidden_dim=expert_hidden_dim,
        top_k=top_k,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    
    print(f"✅ Profiling completed:")
    print(f"  - Mean: {stats['mean']:.3f} ms")
    print(f"  - Median: {stats['median']:.3f} ms")
    print(f"  - Std: {stats['std']:.3f} ms")

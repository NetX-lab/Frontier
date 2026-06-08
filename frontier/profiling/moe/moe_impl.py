"""
MoE operation implementations for profiling.

This module implements MoE-specific operations:
1. moe_gating: Router/gating network
2. moe_shuffling: Local token shuffling (GPU memory operations)
3. moe_grouped_gemm: Expert computation using grouped GEMM

Design principle: EP (expert_parallel_size) is a distribution parameter,
not a compute parameter. We profile operations per-device, using num_experts_per_device.

Implementation: Uses Frontier profiling operators for consistency with MLP profiling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.profiling.common.layers.activation import SiluAndMul
from frontier.profiling.common.parallel_utils.tensor_parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from frontier.profiling.common.utils import raise_if_fp8_requested
try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_topk,
        get_config_dtype_str,
        try_get_optimal_moe_config,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    HAS_VLLM = True
    _VLLM_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - fail fast when vLLM missing
    HAS_VLLM = False
    _VLLM_IMPORT_ERROR = exc

try:
    from vllm.model_executor.layers.linear import ReplicatedLinear

    HAS_VLLM_REPLICATED_LINEAR = True
except ImportError:
    ReplicatedLinear = None  # type: ignore[assignment]
    HAS_VLLM_REPLICATED_LINEAR = False


def uniform_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    indices_type: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """vLLM-equivalent uniform routing fallback for profiling."""
    num_tokens = hidden_states.size(0)
    num_experts = gating_output.size(1)

    topk_weights = torch.full(
        (num_tokens, topk),
        1.0 / topk,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    topk_ids = torch.empty(
        num_tokens,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_indices = torch.arange(num_tokens, device=hidden_states.device)
    for k in range(topk):
        topk_ids[:, k] = (token_indices * topk + k) % num_experts

    token_expert_indices = topk_ids.clone().to(torch.int32)
    return topk_weights, topk_ids, token_expert_indices


class MoEGatingNetwork(nn.Module):
    """
    MoE Gating/Router Network.

    Computes routing scores for all experts and selects top-K experts per token.
    This operation is independent of EP (expert parallelism) because:
    - Gating happens before token dispatch
    - Each token needs routing scores for ALL experts
    - EP only affects subsequent expert execution, not routing decision

    Implementation: Uses ColumnParallelLinear from Frontier profiling for consistency.

    Timing is split into two scopes to match vLLM's implementation:
    - moe_gating_linear: Gate linear layer (hidden_dim -> num_experts)
    - moe_gating_routing_topk: TopK selection + Softmax normalization
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        router_topk: int,
        tensor_parallel_size: int = 1,
        use_vllm_fused_topk: bool = True,
        renormalize: bool = True,
        routing_runtime_path: str = "standard_fused_topk",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.router_topk = router_topk
        self.tensor_parallel_size = tensor_parallel_size
        self.use_vllm_fused_topk = use_vllm_fused_topk
        self.renormalize = renormalize
        self.routing_runtime_path = routing_runtime_path
        get_routing_runtime_metadata(self.routing_runtime_path)
        if self.use_vllm_fused_topk and not HAS_VLLM:
            raise ImportError(
                "vLLM fused_topk is required for MoE gating profiling but is not available."
            ) from _VLLM_IMPORT_ERROR
        if (
            self.routing_runtime_path == "uniform_topk"
            and not self.use_vllm_fused_topk
        ):
            raise ValueError(
                "uniform_topk routing profiling requires the vLLM routing path."
            )

        if self.use_vllm_fused_topk and HAS_VLLM_REPLICATED_LINEAR:
            # Align gating linear kernel family with vLLM runtime contract.
            # disable_tp=True avoids requiring TP group initialization in profiling jobs.
            self.gate = ReplicatedLinear(
                hidden_dim,
                num_experts,
                bias=False,
                disable_tp=True,
            )
        else:
            # Fall back to native torch linear only when vLLM kernel alignment is disabled.
            self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Split gating into two separate timers to match vLLM's scope separation
        self.gating_linear_timer = CudaTimer("moe_gating_linear")
        self.gating_routing_topk_timer = CudaTimer("moe_gating_routing_topk")
        raise_if_fp8_requested(
            "moe_gating",
            "FP8 gating kernel is unavailable for moe_gating profiling.",
        )
        raise_if_fp8_requested(
            "moe_gating_linear",
            "FP8 gating kernel is unavailable for moe_gating_linear profiling.",
        )
        raise_if_fp8_requested(
            "moe_gating_routing_topk",
            "FP8 routing kernel is unavailable for moe_gating_routing_topk profiling.",
        )

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [num_tokens, hidden_dim]

        Returns:
            routing_weights: [num_tokens, router_topk]
            selected_experts: [num_tokens, router_topk]
        """
        # Gate linear layer (hidden_dim -> num_experts)
        with self.gating_linear_timer:
            logits = self.gate(hidden_states)  # [num_tokens, num_experts]
            if isinstance(logits, tuple):
                logits = logits[0]

        # TopK selection + Softmax normalization
        with self.gating_routing_topk_timer:
            routing_runtime_path = getattr(
                self,
                "routing_runtime_path",
                "standard_fused_topk",
            )
            if getattr(self, "use_vllm_fused_topk", False):
                if routing_runtime_path == "uniform_topk":
                    routing_weights, selected_experts, _ = uniform_topk(
                        hidden_states=hidden_states,
                        gating_output=logits,
                        topk=self.router_topk,
                        indices_type=None,
                    )
                else:
                    routing_weights, selected_experts, _ = fused_topk(
                        hidden_states=hidden_states,
                        gating_output=logits,
                        topk=self.router_topk,
                        renormalize=getattr(self, "renormalize", True),
                        indices_type=None,
                    )
            else:
                if routing_runtime_path != "standard_fused_topk":
                    raise ValueError(
                        "PyTorch fallback routing only supports standard_fused_topk."
                    )
                routing_weights, selected_experts = torch.topk(
                    logits, self.router_topk, dim=-1
                )
                routing_weights = F.softmax(
                    routing_weights,
                    dim=-1,
                    dtype=torch.float32,
                ).to(hidden_states.dtype)

        return routing_weights, selected_experts


def get_routing_runtime_metadata(
    routing_runtime_path: str,
) -> dict[str, object]:
    if routing_runtime_path == "standard_fused_topk":
        return {
            "routing_runtime_path": "standard_fused_topk",
            "routing_assignment_policy": "logit_topk",
            "routing_weight_policy": "softmax_renorm",
            "routing_uses_router_logits": True,
        }
    if routing_runtime_path == "uniform_topk":
        return {
            "routing_runtime_path": "uniform_topk",
            "routing_assignment_policy": "round_robin_uniform",
            "routing_weight_policy": "uniform_1_over_topk",
            "routing_uses_router_logits": False,
        }
    raise ValueError(
        f"Unsupported routing_runtime_path for MoE profiling: {routing_runtime_path}"
    )


class MoETokenShuffler(nn.Module):
    """
    Local token shuffling for MoE.
    
    This profiles the GPU memory operations for reordering tokens based on
    routing decisions. Only profiles LOCAL shuffling (within a single GPU).
    Cross-device shuffling (all-to-all) is handled separately in communication profiling.
    """
    
    def __init__(
        self,
        num_experts: int,
        router_topk: int,
        hidden_dim: int,
        expert_hidden_dim: int,
        dtype: torch.dtype,
        use_gated: bool,
        num_local_experts: Optional[int] = None,
    ):
        super().__init__()
        if not HAS_VLLM:
            raise ImportError(
                "vLLM is required for MoE shuffling alignment but is not available."
            ) from _VLLM_IMPORT_ERROR
        self.num_experts = num_experts
        self.num_local_experts = (
            num_experts if num_local_experts is None else num_local_experts
        )
        self.router_topk = router_topk
        self.hidden_dim = hidden_dim
        self.expert_hidden_dim = expert_hidden_dim
        self.dtype = dtype
        self.use_gated = use_gated
        self._block_size_cache = {}
        
        self.shuffling_timer = CudaTimer("moe_shuffling")
        raise_if_fp8_requested(
            "moe_shuffling",
            "FP8 shuffling kernel is unavailable for moe_shuffling profiling.",
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        *,
        global_num_experts: Optional[int] = None,
        expert_map: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states: [num_tokens, hidden_dim]
            selected_experts: [num_tokens, router_topk]
        
        Returns:
            shuffled_states: Reordered tokens grouped by expert
        
        Note:
            The shuffling timer wraps vLLM's moe_align_block_size scope, matching
            the runtime scope semantics captured by CUDA event instrumentation.
        """
        num_tokens = hidden_states.shape[0]
        block_size = self._get_block_size_m(num_tokens)
        topk_ids = selected_experts.to(dtype=torch.int32)

        aligned_num_experts = (
            self.num_experts if global_num_experts is None else global_num_experts
        )

        with self.shuffling_timer:
            sorted_token_ids, _, _ = moe_align_block_size(
                topk_ids=topk_ids,
                block_size=block_size,
                num_experts=aligned_num_experts,
                expert_map=expert_map,
            )

        valid_token_ids = sorted_token_ids[sorted_token_ids < num_tokens]
        shuffled_states = hidden_states[valid_token_ids]
        return shuffled_states

    def _get_block_size_m(self, num_tokens: int) -> int:
        cached = self._block_size_cache.get(num_tokens)
        if cached is not None:
            return cached
        num_local_experts = getattr(self, "num_local_experts", self.num_experts)
        w1_dim = self.expert_hidden_dim * (2 if self.use_gated else 1)
        w1_shape = (num_local_experts, w1_dim, self.hidden_dim)
        w2_shape = (num_local_experts, self.hidden_dim, self.expert_hidden_dim)
        config_dtype = get_config_dtype_str(dtype=self.dtype)
        config = try_get_optimal_moe_config(
            w1_shape=w1_shape,
            w2_shape=w2_shape,
            top_k=self.router_topk,
            dtype=config_dtype,
            M=num_tokens,
            block_shape=None,
        )
        block_size = config["BLOCK_SIZE_M"]
        self._block_size_cache[num_tokens] = block_size
        return block_size


class MoEGroupedGEMM(nn.Module):
    """
    Grouped GEMM for MoE expert computation.
    
    This profiles the core expert computation. Key design principle:
    - Use num_experts_per_device (not expert_parallel_size)
    - EP only determines workload distribution, not computation pattern
    - Profiling on single GPU with N experts = profiling on EP=K with N/K experts per device
    
    Implementation note:
    We use a simple implementation with per-expert linear layers for profiling.
    In production, this would use optimized grouped GEMM kernels (e.g., FlashInfer, vLLM).
    For profiling purposes, the timing characteristics should be similar.
    """
    
    def __init__(
        self,
        num_experts_per_device: int,
        hidden_dim: int,
        expert_hidden_dim: int,
        tensor_parallel_size: int = 1,
        use_gated: bool = True,
    ):
        super().__init__()
        self.num_experts_per_device = num_experts_per_device
        self.hidden_dim = hidden_dim
        self.expert_hidden_dim = expert_hidden_dim
        self.tensor_parallel_size = tensor_parallel_size
        self.use_gated = use_gated
        
        # Create expert layers
        # In practice, these would be implemented as grouped GEMM
        # For profiling, we use ModuleList to simulate the computation
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(num_experts_per_device)
        ])
        
        self.grouped_gemm_timer = CudaTimer("moe_grouped_gemm")
    
    def _create_expert(self):
        """
        Create a single expert (FFN) using Frontier profiling operators.

        This ensures consistency with MLP profiling and uses optimized operators.
        """
        if self.use_gated:
            # Gated FFN: up_proj produces 2x intermediate_size for SwiGLU
            # Use Frontier profiling parallel linear layers
            return nn.ModuleDict({
                "up_proj": ColumnParallelLinear(
                    self.hidden_dim,
                    2 * self.expert_hidden_dim,
                    bias=False,
                    gather_output=False,  # Keep output partitioned for activation
                    world_size=self.tensor_parallel_size,
                    linear_metric_name="moe_expert_up_proj",
                ),
                "down_proj": RowParallelLinear(
                    self.expert_hidden_dim,
                    self.hidden_dim,
                    bias=False,
                    input_is_parallel=True,  # Input is partitioned from activation
                    reduce_results=False,  # Avoid needing parallel state for profiling
                    world_size=self.tensor_parallel_size,
                    linear_metric_name="moe_expert_down_proj",
                ),
                "act_fn": SiluAndMul(),  # Custom CUDA activation
            })
        else:
            # Standard FFN with GELU
            return nn.ModuleDict({
                "up_proj": ColumnParallelLinear(
                    self.hidden_dim,
                    self.expert_hidden_dim,
                    bias=False,
                    gather_output=False,
                    world_size=self.tensor_parallel_size,
                    linear_metric_name="moe_expert_up_proj",
                ),
                "down_proj": RowParallelLinear(
                    self.expert_hidden_dim,
                    self.hidden_dim,
                    bias=False,
                    input_is_parallel=True,
                    reduce_results=False,  # Avoid needing parallel state for profiling
                    world_size=self.tensor_parallel_size,
                    linear_metric_name="moe_expert_down_proj",
                ),
                "act_fn": nn.GELU(),  # Standard PyTorch GELU
            })
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_allocation: torch.Tensor,
    ):
        """
        Args:
            hidden_states: [total_tokens, hidden_dim]
                          where total_tokens = num_tokens * router_topk (after token replication)
            expert_allocation: [num_experts_per_device] - number of tokens assigned to each expert

        Returns:
            expert_outputs: [total_tokens, hidden_dim]
        """
        with self.grouped_gemm_timer:
            outputs = []
            start_idx = 0

            # Process each expert's tokens
            for expert_id, num_tokens_for_expert in enumerate(expert_allocation):
                if num_tokens_for_expert > 0:
                    end_idx = start_idx + num_tokens_for_expert
                    expert_input = hidden_states[start_idx:end_idx]

                    # Apply expert FFN using Frontier profiling operators
                    if self.use_gated:
                        # Gated FFN: up_proj -> SiluAndMul -> down_proj
                        up_output, _ = self.experts[expert_id]["up_proj"](expert_input)
                        intermediate = self.experts[expert_id]["act_fn"](up_output)  # SiluAndMul
                        expert_output, _ = self.experts[expert_id]["down_proj"](intermediate)
                    else:
                        # Standard FFN: up_proj -> GELU -> down_proj
                        up_output, _ = self.experts[expert_id]["up_proj"](expert_input)
                        intermediate = self.experts[expert_id]["act_fn"](up_output)  # GELU
                        expert_output, _ = self.experts[expert_id]["down_proj"](intermediate)

                    outputs.append(expert_output)
                    start_idx = end_idx

            # Concatenate all expert outputs
            if outputs:
                result = torch.cat(outputs, dim=0)
            else:
                result = torch.empty(0, self.hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

        return result


class MoELayer(nn.Module):
    """
    Complete MoE layer combining gating, shuffling, and grouped GEMM.
    
    This is used for integrated profiling to ensure realistic timing.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        num_experts_per_device: int,
        router_topk: int,
        tensor_parallel_size: int = 1,
        use_gated: bool = True,
    ):
        super().__init__()
        
        self.gating = MoEGatingNetwork(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            router_topk=router_topk,
            tensor_parallel_size=tensor_parallel_size,
        )
        
        self.shuffler = MoETokenShuffler(
            num_experts=num_experts,
            router_topk=router_topk,
        )
        
        self.grouped_gemm = MoEGroupedGEMM(
            num_experts_per_device=num_experts_per_device,
            hidden_dim=hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            tensor_parallel_size=tensor_parallel_size,
            use_gated=use_gated,
        )
        
        self.num_experts = num_experts
        self.num_experts_per_device = num_experts_per_device
        self.router_topk = router_topk
    
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [num_tokens, hidden_dim]
        
        Returns:
            output: [num_tokens, hidden_dim]
        """
        # 1. Gating
        routing_weights, selected_experts = self.gating(hidden_states)
        
        # 2. Token shuffling
        shuffled_states = self.shuffler(hidden_states, selected_experts)
        
        # 3. Simulate expert allocation (uniform distribution for profiling)
        total_tokens = hidden_states.shape[0] * self.router_topk
        tokens_per_expert = total_tokens // self.num_experts_per_device
        expert_allocation = torch.full(
            (self.num_experts_per_device,),
            tokens_per_expert,
            dtype=torch.long,
            device=hidden_states.device,
        )
        # Handle remainder
        remainder = total_tokens % self.num_experts_per_device
        if remainder > 0:
            expert_allocation[:remainder] += 1
        
        # 4. Grouped GEMM
        expert_outputs = self.grouped_gemm(shuffled_states, expert_allocation)
        
        # 5. Combine outputs (weighted by routing weights)
        # For profiling, we skip the actual weighted combination
        # and just return the expert outputs
        return expert_outputs

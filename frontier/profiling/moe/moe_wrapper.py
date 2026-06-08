"""
MoE profiling wrapper.

This module wraps MoE operations for profiling, following the same pattern as the linear-op wrapper.
"""

import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.config import PrecisionType
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.common.utils import (
    configure_quantization_manager_for_model_name,
    get_operation_precision,
)
from frontier.profiling.moe.moe_impl import (
    MoEGatingNetwork,
    MoETokenShuffler,
    MoEGroupedGEMM,
    get_routing_runtime_metadata,
)
from frontier.profiling.moe.moe_vllm_kernel import (
    check_fp8_available,
    check_vllm_available,
)
from frontier.profiling.utils import ProfileMethod, normalize_profile_method
from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_HOT_MOE_GATING_PREFIX_REPEATS,
    get_moe_gating_runtime_context_metadata,
)

WARMUP_STEPS = 2
ACTIVE_STEPS = 20


class MoEWrapper:
    """
    Wrapper for profiling MoE operations.

    Profiles three core MoE compute operations:
    1. moe_gating: Router/gating network
    2. moe_shuffling: Local token shuffling
    3. moe_grouped_gemm: Expert computation
    """

    def __init__(
        self,
        model_config: ModelConfig,
        num_tensor_parallel_workers: int,
        expert_parallel_size: int,
        profile_method: str,
        rank: int,
        output_dir: str,
        use_vllm_kernel: bool = True,  # New parameter for load imbalance profiling
        use_fp8: bool = False,  # Enable FP8 W8A8 quantization profiling
        per_channel_quant: bool = False,  # Per-channel vs per-tensor quantization
        block_shape: Optional[List[int]] = None,  # Block-wise quantization shape (e.g., [128, 128])
        routing_runtime_path: str = "standard_fused_topk",
        gating_runtime_context: str = DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    ):
        super().__init__()

        # Validate that this is a MoE model
        if not model_config.is_moe:
            raise ValueError(
                f"MoEWrapper requires a MoE model (is_moe=True), "
                f"but got model {model_config.name} with is_moe=False"
            )

        self.profile_method = normalize_profile_method(profile_method)
        self.timer_stats_store = TimerStatsStore(profile_method=self.profile_method)

        # Store model config and extract MoE-specific parameters
        self.model_config = model_config
        configure_quantization_manager_for_model_name(self.model_config.name)
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.expert_parallel_size = expert_parallel_size
        self.rank = rank
        self.output_dir = output_dir
        self.use_vllm_kernel = use_vllm_kernel  # Flag for using vLLM kernel
        self.use_fp8 = use_fp8  # Flag for FP8 W8A8 quantization
        self.per_channel_quant = per_channel_quant  # Per-channel quantization flag
        self.block_shape = block_shape  # Block-wise quantization shape
        self.routing_runtime_path = routing_runtime_path
        self.routing_runtime_metadata = get_routing_runtime_metadata(
            routing_runtime_path
        )
        self.gating_runtime_context_metadata = get_moe_gating_runtime_context_metadata(
            gating_runtime_context
        )
        self.gating_runtime_context = self.gating_runtime_context_metadata[
            "gating_runtime_context"
        ]

        # Extract MoE parameters from ModelConfig
        self.hidden_dim = model_config.embedding_dim
        self.expert_hidden_dim = model_config.mlp_hidden_dim  # moe_intermediate_size
        self.num_experts = model_config.num_experts
        self.router_topk = model_config.num_experts_per_tok
        self.use_gated = model_config.use_gated_mlp
        self._dtype = model_config.dtype
        precision = get_operation_precision("moe_grouped_gemm")
        use_fp8_from_manager = precision == PrecisionType.FP8
        if use_fp8_from_manager and not self.use_vllm_kernel:
            raise ValueError(
                "FP8 moe_grouped_gemm requires vLLM fused kernel profiling (use_vllm_kernel=True)."
            )
        if use_fp8_from_manager:
            if not check_vllm_available():
                raise ImportError(
                    "vLLM fused MoE kernel is unavailable for FP8 moe_grouped_gemm profiling."
                )
            if not check_fp8_available():
                raise ImportError(
                    "vLLM FP8 quantization utilities are unavailable for moe_grouped_gemm profiling."
                )
        self.use_fp8 = use_fp8_from_manager

        # Calculate num_experts_per_device based on EP
        # EP is a distribution parameter: it determines how experts are distributed across devices
        # For profiling, we calculate the number of experts each device handles
        if self.num_experts % expert_parallel_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"expert_parallel_size ({expert_parallel_size})"
            )
        self.num_experts_per_device = self.num_experts // expert_parallel_size

        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        # Initialize MoE components (convert to float16 for profiling)
        self.gating = MoEGatingNetwork(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            router_topk=self.router_topk,
            tensor_parallel_size=num_tensor_parallel_workers,
            use_vllm_fused_topk=self.use_vllm_kernel,
            routing_runtime_path=self.routing_runtime_path,
        ).to(dtype=self._dtype).cuda().eval()

        self.shuffler = MoETokenShuffler(
            num_experts=self.num_experts,
            num_local_experts=self.num_experts_per_device,
            router_topk=self.router_topk,
            hidden_dim=self.hidden_dim,
            expert_hidden_dim=self.expert_hidden_dim,
            dtype=self._dtype,
            use_gated=self.use_gated,
        ).to(dtype=self._dtype).cuda().eval()

        self.grouped_gemm = MoEGroupedGEMM(
            num_experts_per_device=self.num_experts_per_device,
            hidden_dim=self.hidden_dim,
            expert_hidden_dim=self.expert_hidden_dim,
            tensor_parallel_size=num_tensor_parallel_workers,
            use_gated=self.use_gated,
        ).to(dtype=self._dtype).cuda().eval()

        # Initialize dummy weights
        self._initialize_weights()
        self._init_gating_runtime_context_state()

    def _initialize_weights(self):
        """Initialize dummy weights for profiling."""
        for module in [self.gating, self.shuffler, self.grouped_gemm]:
            for param in module.parameters():
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)

    def _init_gating_runtime_context_state(self) -> None:
        prefix_up_dim = self.expert_hidden_dim * (2 if self.use_gated else 1)
        self._gating_prefix_up_weight = torch.empty(
            prefix_up_dim,
            self.hidden_dim,
            device="cuda",
            dtype=self._dtype,
        )
        self._gating_prefix_down_weight = torch.empty(
            self.hidden_dim,
            self.expert_hidden_dim,
            device="cuda",
            dtype=self._dtype,
        )
        torch.nn.init.xavier_uniform_(self._gating_prefix_up_weight)
        torch.nn.init.xavier_uniform_(self._gating_prefix_down_weight)

    def _materialize_gating_hidden_states(self, num_tokens: int) -> torch.Tensor:
        return torch.randn(
            num_tokens,
            self.hidden_dim,
            device="cuda",
            dtype=self._dtype,
        )

    def _run_gating_with_runtime_context(
        self,
        hidden_states: torch.Tensor,
    ):
        if self.gating_runtime_context == PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT:
            self._run_prefill_hot_gating_prefix(hidden_states)
        return self.gating(hidden_states)

    def _run_prefill_hot_gating_prefix(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        prefix_hidden_states = hidden_states
        for _ in range(PREFILL_HOT_MOE_GATING_PREFIX_REPEATS):
            up_proj = torch.matmul(
                prefix_hidden_states, self._gating_prefix_up_weight.transpose(0, 1)
            )
            if self.use_gated:
                gate_proj, value_proj = up_proj.chunk(2, dim=-1)
                activated = F.silu(gate_proj) * value_proj
            else:
                activated = F.silu(up_proj)
            prefix_hidden_states = torch.matmul(
                activated, self._gating_prefix_down_weight.transpose(0, 1)
            )
        return prefix_hidden_states

    def _get_profiling_ep_rank(self) -> int:
        """Use a fixed representative EP rank for deterministic profiling rows."""
        return 0

    def _build_local_expert_map(self, device: torch.device) -> torch.Tensor:
        if self.expert_parallel_size <= 1:
            raise ValueError("Local expert map is only valid when expert_parallel_size > 1")
        if self.num_experts % self.expert_parallel_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by expert_parallel_size ({self.expert_parallel_size})"
            )

        local_num_experts = self.num_experts_per_device
        ep_rank = self._get_profiling_ep_rank()
        start_idx = ep_rank * local_num_experts
        expert_map = torch.full(
            (self.num_experts,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        expert_map[start_idx:start_idx + local_num_experts] = torch.arange(
            local_num_experts,
            dtype=torch.int32,
            device=device,
        )
        return expert_map

    def _compute_local_expert_token_counts(self, global_topk_ids: torch.Tensor) -> List[int]:
        if self.expert_parallel_size <= 1:
            raise ValueError("Local expert token counting is only valid when expert_parallel_size > 1")

        expert_map = self._build_local_expert_map(global_topk_ids.device)
        local_ids = expert_map[global_topk_ids.long()]
        valid_local_ids = local_ids[local_ids >= 0]
        if valid_local_ids.numel() == 0:
            return [0] * self.num_experts_per_device
        counts = torch.bincount(
            valid_local_ids.detach().cpu(),
            minlength=self.num_experts_per_device,
        )
        return counts.numpy().astype(int).tolist()

    def _prepare_routing_inputs(
        self,
        num_tokens: int,
        load_distribution: str = "uniform",
        seed: Optional[int] = None,
        expert_token_counts: Optional[List[int]] = None,
    ) -> dict:
        from frontier.profiling.moe.load_distribution import (
            compute_expert_token_counts,
            generate_expert_routing,
        )

        profiling_global_num_experts: Optional[int] = None
        profiling_expert_map: Optional[torch.Tensor] = None

        if self.expert_parallel_size > 1:
            topk_weights, topk_ids = generate_expert_routing(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                top_k=self.router_topk,
                load_distribution=load_distribution,
                seed=seed,
            )
            if expert_token_counts is None:
                expert_token_counts = self._compute_local_expert_token_counts(topk_ids)
            profiling_global_num_experts = self.num_experts
            profiling_expert_map = self._build_local_expert_map(topk_ids.device)
        else:
            topk_weights, topk_ids = generate_expert_routing(
                num_tokens=num_tokens,
                num_experts=self.num_experts_per_device,
                top_k=self.router_topk,
                load_distribution=load_distribution,
                seed=seed,
            )
            if expert_token_counts is None:
                expert_token_counts = compute_expert_token_counts(
                    topk_ids,
                    self.num_experts_per_device,
                )

        return {
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "expert_token_counts": expert_token_counts,
            "global_num_experts": profiling_global_num_experts,
            "expert_map": profiling_expert_map,
        }
    
    @torch.inference_mode()
    def profile_gating(self, num_tokens: int):
        """
        Profile MoE gating network.

        Args:
            num_tokens: Number of input tokens

        Returns:
            dict with timing statistics and parameters
        """
        hidden_states = self._materialize_gating_hidden_states(num_tokens)

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # Warmup
            self._run_gating_with_runtime_context(hidden_states)
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(self.output_dir)
            with record_function_tracer:
                self._run_gating_with_runtime_context(hidden_states)

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            # Warmup
            for _ in range(WARMUP_STEPS):
                self._run_gating_with_runtime_context(hidden_states)
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            # Active profiling
            for _ in range(ACTIVE_STEPS):
                self._run_gating_with_runtime_context(hidden_states)
            torch.cuda.synchronize()

            time_stats = self.timer_stats_store.get_stats()

        stats = {
            "time_stats": time_stats,
            "num_tokens": num_tokens,
            "num_experts": self.num_experts,
            **self.routing_runtime_metadata,
            **self.gating_runtime_context_metadata,
            "router_topk": self.router_topk,
            "hidden_dim": self.hidden_dim,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }
        self.timer_stats_store.clear_stats()

        return stats
    
    @torch.inference_mode()
    def profile_shuffling(
        self,
        num_tokens: int,
        load_distribution: str = "uniform",
        expert_token_counts: Optional[List[int]] = None,
        seed: Optional[int] = None,
        routing_inputs: Optional[dict] = None,
    ):
        """
        Profile MoE token shuffling using the same routing contract as grouped GEMM.

        Args:
            num_tokens: Number of input tokens
            load_distribution: Load distribution type for routing generation
            expert_token_counts: Optional precomputed local expert token counts
            seed: Random seed for reproducibility
            routing_inputs: Optional precomputed routing sample shared with grouped GEMM

        Returns:
            dict with timing statistics and parameters
        """
        from frontier.profiling.moe.moe_input import MoELoadImbalanceInput

        if routing_inputs is None:
            prepare_kwargs = {
                "num_tokens": num_tokens,
                "load_distribution": load_distribution,
                "seed": seed,
            }
            if expert_token_counts is not None:
                prepare_kwargs["expert_token_counts"] = expert_token_counts
            routing_inputs = self._prepare_routing_inputs(**prepare_kwargs)

        hidden_states = torch.randn(
            num_tokens,
            self.hidden_dim,
            device="cuda",
            dtype=self._dtype,
        )
        selected_experts = routing_inputs["topk_ids"]
        expert_token_counts = routing_inputs["expert_token_counts"]
        global_num_experts = routing_inputs["global_num_experts"]
        expert_map = routing_inputs["expert_map"]

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # Warmup
            self.shuffler(
                hidden_states,
                selected_experts,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(self.output_dir)
            with record_function_tracer:
                self.shuffler(
                    hidden_states,
                    selected_experts,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                )

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            # Warmup
            for _ in range(WARMUP_STEPS):
                self.shuffler(
                    hidden_states,
                    selected_experts,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                )
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            # Active profiling
            for _ in range(ACTIVE_STEPS):
                self.shuffler(
                    hidden_states,
                    selected_experts,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                )
            torch.cuda.synchronize()

            time_stats = self.timer_stats_store.get_stats()

        expert_hidden_dim = getattr(self, "expert_hidden_dim", self.hidden_dim)
        num_tensor_parallel_workers = getattr(self, "num_tensor_parallel_workers", 1)
        load_input = MoELoadImbalanceInput(
            num_tokens=num_tokens,
            num_experts_per_device=self.num_experts_per_device,
            hidden_dim=self.hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            router_topk=self.router_topk,
            load_distribution=load_distribution,
            expert_token_counts=expert_token_counts,
            seed=seed,
            tensor_parallel_size=num_tensor_parallel_workers,
        )

        stats = {
            "time_stats": time_stats,
            "num_tokens": num_tokens,
            "num_experts": self.num_experts,
            **load_input.to_features_dict(),
            "num_tensor_parallel_workers": num_tensor_parallel_workers,
        }
        self.timer_stats_store.clear_stats()

        return stats
    
    @torch.inference_mode()
    def profile_grouped_gemm(
        self,
        num_tokens: int,
        load_distribution: str = "uniform",
        expert_token_counts: Optional[List[int]] = None,
        seed: Optional[int] = None,
        routing_inputs: Optional[dict] = None,
    ):
        """
        Profile MoE grouped GEMM with load imbalance support.

        Two profiling modes:
        1. ``use_vllm_kernel=False``: Use per-expert loop implementation.
        2. ``use_vllm_kernel=True``: Use vLLM fused MoE kernel.

        For EP>1, grouped GEMM profiling must preserve vLLM's runtime contract:
        routing is selected over the global expert space first, then each local EP
        rank only executes the subset of routed experts that map to its local shard.
        """
        from frontier.profiling.moe.moe_input import MoELoadImbalanceInput

        if routing_inputs is None:
            prepare_kwargs = {
                "num_tokens": num_tokens,
                "load_distribution": load_distribution,
                "seed": seed,
            }
            if expert_token_counts is not None:
                prepare_kwargs["expert_token_counts"] = expert_token_counts
            routing_inputs = self._prepare_routing_inputs(**prepare_kwargs)

        topk_weights = routing_inputs["topk_weights"]
        topk_ids = routing_inputs["topk_ids"]
        expert_token_counts = routing_inputs["expert_token_counts"]
        profiling_global_num_experts = routing_inputs["global_num_experts"]
        profiling_expert_map = routing_inputs["expert_map"]

        if self.use_vllm_kernel:
            time_stats = self._profile_with_vllm_kernel(
                num_tokens=num_tokens,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=profiling_global_num_experts,
                expert_map=profiling_expert_map,
            )
            grouped_gemm_backend = "vllm_fused"
        else:
            time_stats = self._profile_with_loop(
                expert_token_counts=expert_token_counts,
            )
            grouped_gemm_backend = "frontier_loop"

        load_input = MoELoadImbalanceInput(
            num_tokens=num_tokens,
            num_experts_per_device=self.num_experts_per_device,
            hidden_dim=self.hidden_dim,
            expert_hidden_dim=self.expert_hidden_dim,
            router_topk=self.router_topk,
            load_distribution=load_distribution,
            expert_token_counts=expert_token_counts,
            seed=seed,
            tensor_parallel_size=self.num_tensor_parallel_workers,
        )

        stats = {
            "time_stats": time_stats,
            "num_tokens": num_tokens,
            **load_input.to_features_dict(),
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
            "moe_grouped_gemm_backend": grouped_gemm_backend,
        }

        return stats
    
    def _profile_with_vllm_kernel(
        self,
        num_tokens: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: Optional[int] = None,
        expert_map: Optional[torch.Tensor] = None,
    ) -> dict:
        """Profile using vLLM fused MoE kernel with optional EP-local expert mapping."""
        from frontier.profiling.moe.moe_vllm_kernel import profile_fused_moe_kernel

        stats = profile_fused_moe_kernel(
            num_tokens=num_tokens,
            num_experts=self.num_experts_per_device,
            hidden_dim=self.hidden_dim,
            expert_hidden_dim=self.expert_hidden_dim,
            top_k=self.router_topk,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            tensor_parallel_size=self.num_tensor_parallel_workers,
            dtype=self._dtype,
            warmup_steps=WARMUP_STEPS,
            active_steps=ACTIVE_STEPS,
            use_fp8=self.use_fp8,
            per_channel_quant=self.per_channel_quant,
            block_shape=self.block_shape,
            profile_method=self.profile_method,
            output_dir=self.output_dir,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

        return {
            "moe_grouped_gemm": stats,
        }
    
    def _profile_with_loop(
        self,
        expert_token_counts: List[int],
    ) -> dict:
        """Profile using per-expert loop (backward compatible)."""
        total_tokens = sum(expert_token_counts)
        
        # Generate input hidden states
        hidden_states = torch.randn(
            total_tokens,
            self.hidden_dim,
            device="cuda",
            dtype=self._dtype,
        )

        # Convert to tensor
        expert_allocation = torch.tensor(
            expert_token_counts, dtype=torch.long, device="cuda"
        )

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # Warmup
            self.grouped_gemm(hidden_states, expert_allocation)
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(self.output_dir)
            with record_function_tracer:
                self.grouped_gemm(hidden_states, expert_allocation)

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            # Warmup
            for _ in range(WARMUP_STEPS):
                self.grouped_gemm(hidden_states, expert_allocation)
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            # Active profiling
            for _ in range(ACTIVE_STEPS):
                self.grouped_gemm(hidden_states, expert_allocation)
            torch.cuda.synchronize()

            time_stats = self.timer_stats_store.get_stats()

        self.timer_stats_store.clear_stats()

        return time_stats

    @torch.inference_mode()
    def profile(
        self,
        num_tokens: int,
        load_distribution: str = "uniform",
        seed: Optional[int] = None,
    ):
        """
        Main profiling entry point that profiles all MoE operations.

        This method is called by main.py and returns combined statistics
        for all three MoE operations.

        Args:
            num_tokens: Number of input tokens
            load_distribution: Load distribution type for grouped_gemm (uniform/skewed/extremely_skewed)
            seed: Random seed for reproducibility

        Returns:
            dict with combined timing statistics and parameters
        """
        # Profile all three operations
        routing_inputs = self._prepare_routing_inputs(
            num_tokens=num_tokens,
            load_distribution=load_distribution,
            seed=seed,
        )
        gating_stats = self.profile_gating(num_tokens)
        shuffling_stats = self.profile_shuffling(
            num_tokens,
            load_distribution=load_distribution,
            seed=seed,
            routing_inputs=routing_inputs,
        )
        grouped_gemm_stats = self.profile_grouped_gemm(
            num_tokens,
            load_distribution=load_distribution,
            seed=seed,
            routing_inputs=routing_inputs,
        )

        # Combine statistics
        # We merge time_stats from all three operations
        combined_time_stats = {}
        combined_time_stats.update(gating_stats["time_stats"])
        combined_time_stats.update(shuffling_stats["time_stats"])
        combined_time_stats.update(grouped_gemm_stats["time_stats"])

        # Combine all stats (including load imbalance features from grouped_gemm_stats)
        stats = {
            "time_stats": combined_time_stats,
            "num_tokens": num_tokens,
            "num_experts": self.num_experts,
            "num_experts_per_device": self.num_experts_per_device,
            "expert_parallel_size": self.expert_parallel_size,
            **self.routing_runtime_metadata,
            **self.gating_runtime_context_metadata,
            "router_topk": self.router_topk,
            "hidden_dim": self.hidden_dim,
            "expert_hidden_dim": self.expert_hidden_dim,
            "use_gated": self.use_gated,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }
        
        # Add load imbalance features from grouped_gemm_stats (if present)
        # These are added by profile_grouped_gemm when load imbalance is enabled
        for key in grouped_gemm_stats:
            if key not in stats and key != "time_stats":
                stats[key] = grouped_gemm_stats[key]

        return stats

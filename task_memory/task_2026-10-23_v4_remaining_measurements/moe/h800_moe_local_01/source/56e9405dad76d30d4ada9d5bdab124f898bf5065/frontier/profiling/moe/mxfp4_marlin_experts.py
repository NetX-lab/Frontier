"""Checkpoint-MXFP4 routed experts selected by the vLLM 0.28 MXFP4 MoE oracle.

A checkpoint that stores MXFP4 routed experts, such as DeepSeek-V4's, loads
them through vLLM's ``Mxfp4MoEMethod``. That method picks the kernel in
``select_deepseek_v4_mxfp4_moe_backend``; on Hopper the oracle selects
``MARLIN``, a BF16-activation, MXFP4-weight kernel that also applies the
model's SwiGLU clamp. With the all-gather/reduce-scatter dispatch, each EP rank
receives every token of its EP group with global expert ids, and the expert map
restricts ``MarlinExperts`` to the experts that rank owns.

This module builds those experts for one EP rank and exposes their ``apply``,
which the profiler times. Dispatch and combine are communication that Frontier
models separately. Weights are random packed FP4 bytes with unit E8M0 scales;
they set the kernel's shapes and dtypes, not its numerical results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch

from frontier.profiling.common.vllm_compat import vllm_config_context

SUPPORTED_MXFP4_BACKEND = "MARLIN"

_MXFP4_MARLIN_EXPERTS: Dict[Tuple[Any, ...], "Mxfp4MarlinRoutedExperts"] = {}


def _unit_e8m0_scale(shape: Tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    """Scales of 2**0 in E8M0, stored as the uint8 bytes a checkpoint holds."""

    scale = torch.ones(shape, dtype=torch.float32, device=device)
    return scale.to(torch.float8_e8m0fnu).view(torch.uint8)


@dataclass(frozen=True)
class Mxfp4MarlinRoutedExperts:
    """vLLM experts of one EP rank plus the weights their ``apply`` consumes."""

    backend: str
    experts: Any
    w13: torch.Tensor
    w2: torch.Tensor
    activation: Any
    global_num_experts: int
    expert_map: Optional[torch.Tensor]

    def workspaces(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(workspace13, workspace2)`` laid out as vLLM's kernel does.

        ``FusedMoEKernel`` allocates these outside the timed expert computation
        and provisions the output from the same buffer as ``workspace13``, so
        the shared buffer is sized for the larger of the two.
        """

        from vllm.model_executor.layers.fused_moe.utils import _resize_cache

        _, num_tokens, intermediate_dim, hidden_dim, top_k = self.experts.moe_problem_size(
            hidden_states, self.w13, self.w2, topk_ids
        )
        workspace13_shape, workspace2_shape, output_shape = self.experts.workspace_shapes(
            num_tokens,
            intermediate_dim,
            hidden_dim,
            top_k,
            self.global_num_experts,
            self.w13.size(0),
            None,
            self.activation,
        )
        dtype = self.experts.workspace_dtype(hidden_states.dtype)
        shared = torch.empty(
            (max(math.prod(workspace13_shape), math.prod(output_shape)),),
            dtype=dtype,
            device=hidden_states.device,
        )
        workspace2 = torch.empty(workspace2_shape, dtype=dtype, device=hidden_states.device)
        return _resize_cache(shared, workspace13_shape), workspace2

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
    ) -> None:
        self.experts.apply(
            output,
            hidden_states,
            self.w13,
            self.w2,
            topk_weights,
            topk_ids,
            activation=self.activation,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )


def _build_mxfp4_marlin_routed_experts(
    *,
    num_local_experts: int,
    global_num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    swiglu_limit: Optional[float],
    expert_parallel_size: int,
    expert_parallel_rank: int,
    expert_map: Optional[torch.Tensor],
    dtype: torch.dtype,
    model_type: str,
    device: torch.device,
) -> Mxfp4MarlinRoutedExperts:
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        convert_weight_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_quant_config,
        select_deepseek_v4_mxfp4_moe_backend,
    )

    from frontier.profiling.moe.moe_vllm_kernel import plan_mxfp4_weight_layout

    layout = plan_mxfp4_weight_layout(
        num_experts=num_local_experts,
        hidden_dim=hidden_dim,
        expert_hidden_dim=intermediate_dim,
        use_gated=True,
    )
    model_config = SimpleNamespace(
        dtype=dtype,
        hf_config=SimpleNamespace(model_type=model_type),
        hf_text_config=SimpleNamespace(model_type=model_type),
    )
    with vllm_config_context(model_config=model_config):
        parallel_config = FusedMoEParallelConfig(
            tp_size=1,
            pcp_size=1,
            dp_size=1,
            ep_size=expert_parallel_size,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=expert_parallel_rank,
            sp_size=1,
            use_ep=expert_parallel_size > 1,
            all2all_backend=get_current_vllm_config().parallel_config.all2all_backend,
            enable_eplb=False,
        )
        # Routing runs before these experts, and the modular experts accept
        # precomputed top-k ids from any router.
        moe_config = FusedMoEConfig(
            num_experts=global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_dim,
            intermediate_size=intermediate_dim,
            num_local_experts=num_local_experts,
            num_logical_experts=global_num_experts,
            activation=MoEActivation.SILU,
            device=device,
            routing_method=RoutingMethodType.Unspecified,
            moe_parallel_config=parallel_config,
            in_dtype=dtype,
            swiglu_limit=swiglu_limit,
        )
        backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(moe_config)
        if backend.value != SUPPORTED_MXFP4_BACKEND:
            raise RuntimeError(
                f"vLLM selected the {backend.value} MXFP4 MoE backend for EP="
                f"{expert_parallel_size}; this profiler supports only "
                f"{SUPPORTED_MXFP4_BACKEND}, which vLLM selects on Hopper for "
                "checkpoint-MXFP4 experts."
            )

        def packed_fp4(shape: Tuple[int, ...]) -> torch.Tensor:
            return torch.randint(0, 256, shape, dtype=torch.uint8, device=device)

        w13, w2, w13_scale, w2_scale, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=backend,
            layer=SimpleNamespace(params_dtype=dtype),
            w13_weight=packed_fp4(layout["w13_shape"]),
            w2_weight=packed_fp4(layout["w2_shape"]),
            w13_weight_scale=_unit_e8m0_scale(layout["w13_scale_shape"], device=device),
            w2_weight_scale=_unit_e8m0_scale(layout["w2_scale_shape"], device=device),
            activation=MoEActivation.SILU,
        )
        quant_config = make_mxfp4_moe_quant_config(
            mxfp4_backend=backend,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            swiglu_limit=swiglu_limit,
        )
        experts = experts_cls(moe_config=moe_config, quant_config=quant_config)
    return Mxfp4MarlinRoutedExperts(
        backend=backend.value,
        experts=experts,
        w13=w13,
        w2=w2,
        activation=MoEActivation.SILU,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )


def get_mxfp4_marlin_routed_experts(**kwargs: Any) -> Mxfp4MarlinRoutedExperts:
    """Build the experts once per layout; a new layout frees the previous one.

    The profiler calls this once per token count with the same layout, and
    the expert weights of one EP layout can take several GiB of HBM. The EP
    size and rank in the key determine the contiguous expert map.
    """

    device = torch.device("cuda", torch.cuda.current_device())
    key = (device, *sorted((k, v) for k, v in kwargs.items() if k != "expert_map"))
    cached = _MXFP4_MARLIN_EXPERTS.get(key)
    if cached is not None:
        return cached
    if _MXFP4_MARLIN_EXPERTS:
        _MXFP4_MARLIN_EXPERTS.clear()
        torch.cuda.empty_cache()
    experts = _build_mxfp4_marlin_routed_experts(device=device, **kwargs)
    _MXFP4_MARLIN_EXPERTS[key] = experts
    return experts


__all__ = [
    "Mxfp4MarlinRoutedExperts",
    "SUPPORTED_MXFP4_BACKEND",
    "get_mxfp4_marlin_routed_experts",
]

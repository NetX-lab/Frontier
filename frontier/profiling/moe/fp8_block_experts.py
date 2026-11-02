"""Block-FP8 routed experts selected by the vLLM 0.28 FP8 MoE oracle.

vLLM 0.28 picks the FP8 MoE kernel in ``select_fp8_moe_backend`` from the
layer's ``FusedMoEConfig``. On Hopper, block-scaled FP8 weights with expert
parallelism select ``FLASHINFER_CUTLASS``. With the all-gather/reduce-scatter
dispatch, each EP rank receives every token of its EP group with global expert
ids, and ``FlashInferExperts`` computes only the experts that rank owns.

This module builds those experts for one EP rank and exposes their ``apply``,
which the profiler times. Dispatch and combine are communication that Frontier
models separately. Weights are random FP8 values with unit block scales; they
set the kernel's shapes and dtypes, not its numerical results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from frontier.profiling.common.vllm_compat import vllm_config_context

SUPPORTED_FP8_BLOCK_BACKEND = "FLASHINFER_CUTLASS"

_FP8_BLOCK_EXPERTS: Dict[Tuple[Any, ...], "Fp8BlockRoutedExperts"] = {}


def expert_parallel_layout(
    expert_map: Optional[torch.Tensor], *, num_local_experts: int, global_num_experts: int
) -> Tuple[int, int]:
    """Return ``(ep_size, ep_rank)`` for a contiguous global-to-local expert map."""

    if expert_map is None:
        if num_local_experts != global_num_experts:
            raise ValueError(
                "an expert_map is required when local experts "
                f"({num_local_experts}) differ from global experts ({global_num_experts})"
            )
        return 1, 0
    if global_num_experts % num_local_experts != 0:
        raise ValueError(
            f"global experts ({global_num_experts}) must be a multiple of "
            f"local experts ({num_local_experts})"
        )
    owned = torch.nonzero(expert_map.cpu() >= 0).flatten()
    if owned.numel() != num_local_experts:
        raise ValueError(
            f"expert_map owns {owned.numel()} experts, expected {num_local_experts}"
        )
    start = int(owned[0])
    expected = torch.full((global_num_experts,), -1, dtype=torch.int64)
    expected[start : start + num_local_experts] = torch.arange(num_local_experts)
    if start % num_local_experts != 0 or not torch.equal(expert_map.cpu().long(), expected):
        raise ValueError("expert_map must assign one contiguous, aligned block of experts")
    return global_num_experts // num_local_experts, start // num_local_experts


def _random_fp8(shape: Sequence[int], *, device: torch.device) -> torch.Tensor:
    weight = torch.empty(tuple(shape), dtype=torch.bfloat16, device=device)
    return weight.normal_(mean=0.0, std=0.02).to(torch.float8_e4m3fn)


def _block_scale(shape: Sequence[int], block_shape: Sequence[int], *, device) -> torch.Tensor:
    experts, rows, cols = shape
    block_n, block_k = block_shape
    grid = (experts, math.ceil(rows / block_n), math.ceil(cols / block_k))
    return torch.ones(grid, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class Fp8BlockRoutedExperts:
    """vLLM experts of one EP rank plus the weights their ``apply`` consumes."""

    backend: str
    experts: Any
    w13: torch.Tensor
    w2: torch.Tensor
    activation: Any
    global_num_experts: int

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
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
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )


def _build_fp8_block_routed_experts(
    *,
    num_local_experts: int,
    global_num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    block_shape: Sequence[int],
    expert_parallel_size: int,
    expert_parallel_rank: int,
    dtype: torch.dtype,
    model_type: str,
    device: torch.device,
) -> Fp8BlockRoutedExperts:
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        convert_to_fp8_moe_kernel_format,
        make_fp8_moe_quant_config,
        select_fp8_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
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
        )
        backend, experts_cls = select_fp8_moe_backend(
            config=moe_config,
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
            allow_vllm_cutlass=False,
        )
        if backend.value != SUPPORTED_FP8_BLOCK_BACKEND:
            raise RuntimeError(
                f"vLLM selected the {backend.value} FP8 MoE backend for EP="
                f"{expert_parallel_size}; this profiler supports only "
                f"{SUPPORTED_FP8_BLOCK_BACKEND}, which vLLM selects on Hopper "
                "for block-FP8 experts with expert parallelism."
            )

        w13_shape = (num_local_experts, 2 * intermediate_dim, hidden_dim)
        w2_shape = (num_local_experts, hidden_dim, intermediate_dim)
        layer = SimpleNamespace(
            moe_config=moe_config,
            weight_block_size=list(block_shape),
            activation=MoEActivation.SILU,
        )
        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=backend,
            layer=layer,
            w13=_random_fp8(w13_shape, device=device),
            w2=_random_fp8(w2_shape, device=device),
            w13_scale=_block_scale(w13_shape, block_shape, device=device),
            w2_scale=_block_scale(w2_shape, block_shape, device=device),
            w13_input_scale=None,
            w2_input_scale=None,
        )
        quant_config = make_fp8_moe_quant_config(
            fp8_backend=backend,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            a2_scale=None,
            block_shape=list(block_shape),
        )
        experts = experts_cls(moe_config=moe_config, quant_config=quant_config)
    return Fp8BlockRoutedExperts(
        backend=backend.value,
        experts=experts,
        w13=w13,
        w2=w2,
        activation=MoEActivation.SILU,
        global_num_experts=global_num_experts,
    )


def get_fp8_block_routed_experts(**kwargs: Any) -> Fp8BlockRoutedExperts:
    """Build the experts once per layout; a new layout frees the previous one.

    The profiler calls this once per token count with the same layout, and
    the expert weights of one EP layout can take several GiB of HBM.
    """

    kwargs["block_shape"] = tuple(kwargs["block_shape"])
    device = torch.device("cuda", torch.cuda.current_device())
    key = (device, *sorted(kwargs.items()))
    cached = _FP8_BLOCK_EXPERTS.get(key)
    if cached is not None:
        return cached
    if _FP8_BLOCK_EXPERTS:
        _FP8_BLOCK_EXPERTS.clear()
        torch.cuda.empty_cache()
    experts = _build_fp8_block_routed_experts(device=device, **kwargs)
    _FP8_BLOCK_EXPERTS[key] = experts
    return experts


__all__ = [
    "Fp8BlockRoutedExperts",
    "SUPPORTED_FP8_BLOCK_BACKEND",
    "expert_parallel_layout",
    "get_fp8_block_routed_experts",
]

"""Routed experts built by vLLM's native MoE oracles, for one EP rank.

vLLM picks the routed-expert kernel from the layer's ``FusedMoEConfig``:
``select_fp8_moe_backend`` for block-FP8 weights and
``select_unquantized_moe_backend`` for BF16/FP16 weights. The result is a
modular ``FusedMoEKernel`` (prepare, experts, finalize). Native layers time the
same ``FusedMoEKernel.apply`` call this module exposes.

The kernel is built for one EP rank with ``dp_size=1``. vLLM then selects the
no-DP/EP prepare/finalize, which quantizes the activations locally and skips
experts that ``expert_map`` marks as remote. That is the expert compute of one
EP rank without the dispatch and combine communication, which Frontier models
separately. Weights are random values with unit block scales; they set the
kernel's shapes and dtypes, not its numerical results.

Accepted backends are the ones whose weight layout and prepare/finalize this
module was checked against; any other selection fails fast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from frontier.profiling.common.vllm_compat import vllm_config_context

SUPPORTED_EXPERT_BACKENDS = {
    "fp8": frozenset({"FLASHINFER_CUTLASS", "DEEPGEMM", "TRITON"}),
    "bf16": frozenset({"TRITON"}),
}

# Reported when the Triton experts find no tuned config file for the layout.
DEFAULT_TRITON_TUNING_CONFIG = "default"

_ROUTED_EXPERTS: Dict[Tuple[Any, ...], "NativeRoutedExperts"] = {}


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


def _random_weight(shape: Sequence[int], *, dtype: torch.dtype, device) -> torch.Tensor:
    weight = torch.empty(tuple(shape), dtype=torch.bfloat16, device=device)
    return weight.normal_(mean=0.0, std=0.02).to(dtype)


def _block_scale(shape: Sequence[int], block_shape: Sequence[int], *, device) -> torch.Tensor:
    experts, rows, cols = shape
    block_n, block_k = block_shape
    grid = (experts, math.ceil(rows / block_n), math.ceil(cols / block_k))
    return torch.ones(grid, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class NativeRoutedExperts:
    """A native modular MoE kernel of one EP rank plus the weights it consumes."""

    backend: str
    experts_class: str
    triton_tuning_config: Optional[str]
    kernel: Any
    w13: torch.Tensor
    w2: torch.Tensor
    activation: Any
    global_num_experts: int

    def apply(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Keyword arguments: the modular implementation behind this call
        # takes topk_ids before topk_weights.
        output = self.kernel.apply(
            hidden_states=hidden_states,
            w1=self.w13,
            w2=self.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=self.activation,
            global_num_experts=self.global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=False,
        )
        # Only consumers that defer the finalize receive an unfinalized output.
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"FusedMoEKernel.apply returned {type(output).__name__}; the profiler "
                "times the finalized routed-expert output"
            )
        return output


def _triton_tuning_config(experts_cls, *, w2: torch.Tensor, quant_config, in_dtype) -> Optional[str]:
    """Return the tuned Triton config file the experts load, or ``"default"``.

    Returns ``None`` when the selected experts never run the Triton kernel.
    """

    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_config_file_name,
        get_moe_configs,
    )

    # Fallback experts (e.g. TritonOrDeepGemmExperts) name both of their classes.
    get_clses = getattr(experts_cls, "get_clses", None)
    classes = get_clses() if get_clses is not None else (experts_cls,)
    if not any(issubclass(cls, TritonExperts) for cls in classes):
        return None
    # The same lookup arguments as vLLM's try_get_optimal_moe_config.
    num_experts, _, intermediate = w2.shape
    dtype = quant_config.config_name(in_dtype)
    block_shape = quant_config.block_shape
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    if get_moe_configs(num_experts, intermediate, dtype, block_n, block_k) is None:
        return DEFAULT_TRITON_TUNING_CONFIG
    return get_config_file_name(num_experts, intermediate, dtype, block_shape)


def _build_native_routed_experts(
    *,
    quantization_mode: str,
    activation: str,
    num_local_experts: int,
    global_num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    block_shape: Optional[Sequence[int]],
    expert_parallel_size: int,
    expert_parallel_rank: int,
    dtype: torch.dtype,
    model_type: str,
    device: torch.device,
) -> NativeRoutedExperts:
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    if quantization_mode not in SUPPORTED_EXPERT_BACKENDS:
        raise ValueError(
            f"Native routed experts support quantization modes "
            f"{sorted(SUPPORTED_EXPERT_BACKENDS)}, got {quantization_mode!r}"
        )
    if quantization_mode == "fp8" and block_shape is None:
        raise ValueError("FP8 routed experts require a weight block shape")

    moe_activation = MoEActivation(activation)
    model_config = SimpleNamespace(
        dtype=dtype,
        hf_config=SimpleNamespace(model_type=model_type),
        hf_text_config=SimpleNamespace(model_type=model_type),
    )
    with vllm_config_context(model_config=model_config):
        if not is_workspace_manager_initialized():
            init_workspace_manager(device)
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
            activation=moe_activation,
            device=device,
            routing_method=RoutingMethodType.Unspecified,
            moe_parallel_config=parallel_config,
            in_dtype=dtype,
        )
        w13_shape = (num_local_experts, 2 * intermediate_dim, hidden_dim)
        w2_shape = (num_local_experts, hidden_dim, intermediate_dim)

        if quantization_mode == "fp8":
            from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
                convert_to_fp8_moe_kernel_format,
                make_fp8_moe_kernel,
                make_fp8_moe_quant_config,
                select_fp8_moe_backend,
            )
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                GroupShape,
                create_fp8_quant_key,
                kFp8Dynamic128Sym,
            )

            backend, experts_cls = select_fp8_moe_backend(
                config=moe_config,
                weight_key=create_fp8_quant_key(
                    static=True, group_shape=GroupShape(*block_shape)
                ),
                activation_key=kFp8Dynamic128Sym,
                allow_vllm_cutlass=False,
            )
            _require_supported_backend(backend.value, quantization_mode, moe_activation)
            layer = SimpleNamespace(
                moe_config=moe_config,
                weight_block_size=list(block_shape),
                activation=moe_activation,
            )
            w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
                fp8_backend=backend,
                layer=layer,
                w13=_random_weight(w13_shape, dtype=torch.float8_e4m3fn, device=device),
                w2=_random_weight(w2_shape, dtype=torch.float8_e4m3fn, device=device),
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
            kernel = make_fp8_moe_kernel(
                moe_quant_config=quant_config,
                moe_config=moe_config,
                experts_cls=experts_cls,
                fp8_backend=backend,
            )
        else:
            from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
                convert_to_unquantized_kernel_format,
                make_unquantized_moe_kernel,
                select_unquantized_moe_backend,
            )

            backend, experts_cls = select_unquantized_moe_backend(moe_config)
            _require_supported_backend(backend.value, quantization_mode, moe_activation)
            w13, w2 = convert_to_unquantized_kernel_format(
                unquantized_backend=backend,
                moe_config=moe_config,
                w13_weight=_random_weight(w13_shape, dtype=dtype, device=device),
                w2_weight=_random_weight(w2_shape, dtype=dtype, device=device),
            )
            quant_config = FusedMoEQuantConfig.make()
            kernel = make_unquantized_moe_kernel(
                quant_config=quant_config,
                moe_config=moe_config,
                backend=backend,
                experts_cls=experts_cls,
            )
        triton_tuning_config = _triton_tuning_config(
            experts_cls, w2=w2, quant_config=quant_config, in_dtype=dtype
        )
    return NativeRoutedExperts(
        backend=backend.value,
        experts_class=experts_cls.__name__,
        triton_tuning_config=triton_tuning_config,
        kernel=kernel,
        w13=w13,
        w2=w2,
        activation=moe_activation,
        global_num_experts=global_num_experts,
    )


def _require_supported_backend(backend: str, quantization_mode: str, activation) -> None:
    supported = SUPPORTED_EXPERT_BACKENDS[quantization_mode]
    if backend not in supported:
        raise RuntimeError(
            f"vLLM selected the {backend} {quantization_mode} MoE backend for "
            f"activation {activation.value}; this profiler supports "
            f"{sorted(supported)} for {quantization_mode} routed experts."
        )


def get_native_routed_experts(**kwargs: Any) -> NativeRoutedExperts:
    """Build the experts once per layout; a new layout frees the previous one.

    The profiler calls this once per token count with the same layout, and
    the expert weights of one EP layout can take several GiB of HBM.
    """

    if kwargs["block_shape"] is not None:
        kwargs["block_shape"] = tuple(kwargs["block_shape"])
    device = torch.device("cuda", torch.cuda.current_device())
    key = (device, *sorted(kwargs.items()))
    cached = _ROUTED_EXPERTS.get(key)
    if cached is not None:
        return cached
    if _ROUTED_EXPERTS:
        _ROUTED_EXPERTS.clear()
        torch.cuda.empty_cache()
    experts = _build_native_routed_experts(device=device, **kwargs)
    _ROUTED_EXPERTS[key] = experts
    return experts


__all__ = [
    "DEFAULT_TRITON_TUNING_CONFIG",
    "NativeRoutedExperts",
    "SUPPORTED_EXPERT_BACKENDS",
    "expert_parallel_layout",
    "get_native_routed_experts",
]

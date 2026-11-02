"""Deterministic balanced top-k routing, also usable as a flat native extension.

Route every physical row, including graph padding. Consecutive assignment IDs
give the first experts the remainder, matching Frontier's Hamilton tie order.
Importing this module does not import Torch/vLLM or enable simulated routing.
"""

from __future__ import annotations


def balanced_topk(
    hidden_states, router_logits, top_k: int, indices_type=None, *, token_offset: int = 0
):
    """Return float32 weights and unique expert IDs for an EP router chunk."""
    import torch

    num_tokens = hidden_states.shape[0]
    num_experts = router_logits.shape[-1]
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k must be between 1 and the router's expert count")
    if token_offset < 0:
        raise ValueError("token_offset must be non-negative")
    assignments = torch.arange(
        num_tokens * top_k, dtype=torch.int64, device=hidden_states.device
    ).reshape(num_tokens, top_k)
    topk_ids = ((assignments + token_offset * top_k) % num_experts).to(
        torch.long if indices_type is None else indices_type
    )
    topk_weights = torch.full(
        (num_tokens, top_k),
        1.0 / top_k,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return topk_weights, topk_ids


def register_balanced_topk() -> None:
    """Register explicitly before model construction; leave enablement to callers.

    Supported native paths route before modular EP dispatch, with PCP=1 and
    DP-major x SP local sizes matching the EP group. No GPU readback is needed.
    """
    from vllm.distributed import get_ep_group
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
        RoutingSimulator,
        RoutingStrategy,
    )

    class BalancedTopKRouting(RoutingStrategy):
        def route_tokens(self, hidden_states, router_logits, top_k, indices_type=None):
            metadata = get_forward_context().dp_metadata
            group = get_ep_group()
            if metadata is None:
                if group.world_size != 1:
                    raise ValueError("balanced_topk requires DP/SP metadata for EP>1")
                token_offset = 0
            else:
                sizes = metadata.local_sizes
                rank = group.rank_in_group
                if sizes is None or len(sizes) != group.world_size:
                    raise ValueError("balanced_topk requires EP-ordered DP/SP local_sizes")
                if sizes[rank] != hidden_states.shape[0]:
                    raise ValueError("balanced_topk local_sizes does not match router rows")
                token_offset = sum(sizes[:rank])
            return balanced_topk(
                hidden_states, router_logits, top_k, indices_type,
                token_offset=token_offset,
            )

    RoutingSimulator.register_strategy("balanced_topk", BalancedTopKRouting())

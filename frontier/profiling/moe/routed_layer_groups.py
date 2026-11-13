"""Routed MoE layers that share one expert precision, activation and router.

Most MoE models run every routed layer with the model-wide expert precision,
SiLU experts and softmax top-k routing, so the MoE profiler measures one set of
rows per parallel point. Step-5 differs per layer: layers 3-87 keep block-FP8
experts, layers 88-90 keep BF16 experts, every layer uses the clamped
``swiglustep`` activation, and the router is an FP32 gate followed by sigmoid
scores with a selection bias, renormalization and a scaling factor.

A ``RoutedLayerGroup`` names one run of consecutive routed layers whose expert
kernel and router are identical, so the profiler measures each distinct
kernel once and labels the rows with the layers they cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from frontier.config import PrecisionType

# vLLM MoEActivation values for the routed experts.
SILU_ACTIVATION = "silu"
SWIGLUSTEP_ACTIVATION = "swiglustep"


@dataclass(frozen=True)
class RoutedLayerRouter:
    """Router of one routed layer: gate GEMM precision and top-k scoring."""

    scoring_func: str
    uses_selection_bias: bool
    renormalize: bool
    routed_scaling_factor: float
    gate_precision: PrecisionType


@dataclass(frozen=True)
class RoutedLayerGroup:
    """Consecutive routed layers measured by the same expert and router kernels."""

    layer_ids: Tuple[int, ...]
    precision: PrecisionType
    activation: str
    router: RoutedLayerRouter

    @property
    def layer_range(self) -> str:
        return f"{self.layer_ids[0]}-{self.layer_ids[-1]}"

    def row_metadata(self) -> Dict[str, object]:
        return {
            "moe_layer_ids": self.layer_range,
            "moe_routed_precision": self.precision.value,
            "moe_activation": self.activation,
        }


def _step5_routed_layer_groups(model_config) -> Tuple[RoutedLayerGroup, ...]:
    # Resolve this dependency lazily: the attention package imports the model
    # architecture registry, which imports the profiling model config.
    from frontier.attention.step5 import describe_step5_layer

    groups: list[RoutedLayerGroup] = []
    for layer_id in model_config.get_moe_layer_ids():
        # Expert precision, activation and router are per-layer model facts;
        # they do not depend on the parallel sizes, so the neutral size 1 is
        # used for every domain.
        metadata = describe_step5_layer(
            model_config, layer_id,
            attention_tp_size=1, moe_tp_size=1, expert_parallel_size=1,
        )
        expert_precisions = {
            projection.weight_precision
            for projection in metadata.projections
            if projection.operator_name == "moe_grouped_gemm"
        }
        if len(expert_precisions) != 1:
            raise ValueError(
                f"Step-5 layer {layer_id} routed experts must use one weight "
                f"precision for w13 and w2, got {sorted(p.value for p in expert_precisions)}"
            )
        # Native Step5MoE selects swiglustep for any nonzero routed limit.
        activation = (
            SWIGLUSTEP_ACTIVATION if metadata.routed_swiglu_limit else SILU_ACTIVATION
        )
        router = RoutedLayerRouter(
            scoring_func=metadata.router_scoring,
            uses_selection_bias=metadata.router_bias,
            renormalize=metadata.router_renormalize,
            routed_scaling_factor=metadata.router_scaling_factor,
            gate_precision=metadata.router_precision,
        )
        precision = expert_precisions.pop()
        previous = groups[-1] if groups else None
        if (
            previous is not None
            and previous.layer_ids[-1] + 1 == layer_id
            and (previous.precision, previous.activation, previous.router)
            == (precision, activation, router)
        ):
            groups[-1] = RoutedLayerGroup(
                previous.layer_ids + (layer_id,), precision, activation, router
            )
        else:
            groups.append(RoutedLayerGroup((layer_id,), precision, activation, router))
    return tuple(groups)


_ROUTED_LAYER_GROUP_RESOLVERS: Dict[str, Callable[..., Tuple[RoutedLayerGroup, ...]]] = {
    "step5": _step5_routed_layer_groups,
}


def resolve_routed_layer_groups(model_config) -> Tuple[RoutedLayerGroup, ...]:
    """Return the routed layer groups, or ``()`` when one model-wide setting applies."""

    resolver = _ROUTED_LAYER_GROUP_RESOLVERS.get(model_config.model_type)
    if resolver is None:
        return ()
    return resolver(model_config)


__all__ = [
    "RoutedLayerGroup",
    "RoutedLayerRouter",
    "SILU_ACTIVATION",
    "SWIGLUSTEP_ACTIVATION",
    "resolve_routed_layer_groups",
]

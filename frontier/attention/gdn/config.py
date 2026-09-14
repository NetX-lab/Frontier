"""Model-owned GDN shape and hybrid layer schedule contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence


class SequenceMixerType(str, Enum):
    """Sequence mixer selected for one decoder layer."""

    FULL_ATTENTION = "full_attention"
    GATED_DELTA_NET = "gated_delta_net"

    @classmethod
    def from_config_value(cls, value: object) -> "SequenceMixerType":
        if isinstance(value, cls):
            return value
        aliases = {
            "full_attention": cls.FULL_ATTENTION,
            "attention": cls.FULL_ATTENTION,
            "linear_attention": cls.GATED_DELTA_NET,
            "gated_delta_net": cls.GATED_DELTA_NET,
            "gdn": cls.GATED_DELTA_NET,
        }
        normalized = str(value).strip().lower()
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(
                "Unsupported sequence-mixer layer type "
                f"{value!r}; expected one of {sorted(aliases)}"
            ) from exc


@dataclass(frozen=True)
class LayerAttentionSpec:
    """Resolved attention identity for one global decoder layer."""

    global_layer_id: int
    family_id: str
    variant_id: str

    def __post_init__(self) -> None:
        if type(self.global_layer_id) is not int or self.global_layer_id < 0:
            raise ValueError(
                "global_layer_id must be a non-negative int, "
                f"got {self.global_layer_id!r}"
            )
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValueError("family_id must be a non-empty string")
        if not isinstance(self.variant_id, str) or not self.variant_id:
            raise ValueError("variant_id must be a non-empty string")

    @property
    def is_gdn(self) -> bool:
        return self.family_id == "gated_delta_net"

    @property
    def is_full_attention(self) -> bool:
        return self.family_id == "dense_attention"


@dataclass(frozen=True)
class GatedDeltaNetConfig:
    """Architecture dimensions needed to account for one GDN layer."""

    conv_kernel_size: int
    key_head_dim: int
    value_head_dim: int
    num_key_heads: int
    num_value_heads: int
    output_gate_type: str = "silu"

    def __post_init__(self) -> None:
        for field_name in (
            "conv_kernel_size",
            "key_head_dim",
            "value_head_dim",
            "num_key_heads",
            "num_value_heads",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"GDN {field_name} must be positive, got {value!r}")
        normalized_gate = str(self.output_gate_type).strip().lower()
        if normalized_gate == "swish":
            normalized_gate = "silu"
        if normalized_gate not in {"silu", "sigmoid"}:
            raise ValueError(
                "GDN output_gate_type must be silu/swish or sigmoid, got "
                f"{self.output_gate_type!r}"
            )
        object.__setattr__(self, "output_gate_type", normalized_gate)

    @property
    def key_dim(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.num_value_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    def get_state_layout(
        self,
        *,
        tensor_parallel_size: int,
        num_speculative_tokens: int = 0,
        conv_bytes_per_element: int = 2,
        recurrent_bytes_per_element: int = 4,
    ):
        """Return the fixed state layout for one tensor-parallel worker."""

        from frontier.attention.gdn.memory import GatedDeltaNetStateLayout

        tp_size = int(tensor_parallel_size)
        if tp_size <= 0:
            raise ValueError(
                f"tensor_parallel_size must be positive, got {tensor_parallel_size}"
            )
        speculative_tokens = int(num_speculative_tokens)
        if speculative_tokens < 0:
            raise ValueError(
                "num_speculative_tokens must be non-negative, got "
                f"{num_speculative_tokens}"
            )
        for field_name, value in (
            ("num_key_heads", self.num_key_heads),
            ("num_value_heads", self.num_value_heads),
            ("conv_dim", self.conv_dim),
        ):
            if value % tp_size:
                raise ValueError(
                    f"GDN {field_name} must be divisible by tensor_parallel_size: "
                    f"{field_name}={value}, tp={tp_size}"
                )
        return GatedDeltaNetStateLayout(
            conv_state_shape=(self.conv_kernel_size - 1 + speculative_tokens, self.conv_dim // tp_size),
            recurrent_state_shape=(
                self.num_value_heads // tp_size,
                self.value_head_dim,
                self.key_head_dim,
            ),
            conv_bytes_per_element=int(conv_bytes_per_element),
            recurrent_bytes_per_element=int(recurrent_bytes_per_element),
        )


def is_qwen3_5_profile_config(config: Any) -> bool:
    """Return whether *config* explicitly selects the supported Qwen3.5 profile.

    The model type is authoritative when present.  An architecture alias is
    accepted only when it is the exact registered Qwen3.5 class and no
    conflicting model type was supplied.
    """

    model_type = str(getattr(config, "model_type", None) or "").strip().lower()
    profile_id = str(
        getattr(config, "model_architecture_profile", None) or ""
    ).strip().lower()
    if model_type:
        if model_type != "qwen3_5_moe_text":
            if profile_id == "qwen3_5_moe":
                raise ValueError(
                    "qwen3_5_moe profile requires model_type='qwen3_5_moe_text'; "
                    f"got {model_type!r}"
                )
            return False
        return True
    if profile_id == "qwen3_5_moe":
        return True
    architectures = getattr(config, "architectures", None) or ()
    return any(
        str(value).strip().lower() == "qwen3_5moeforcausallm"
        for value in architectures
    )


def _dense_variant(num_q_heads: int, num_kv_heads: int) -> str:
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            "Attention head counts must be positive: "
            f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}"
        )
    if num_q_heads == num_kv_heads:
        return "mha"
    if num_kv_heads == 1:
        return "mqa"
    if num_kv_heads < num_q_heads:
        return "gqa"
    raise ValueError(
        "Unsupported attention head topology: "
        f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}"
    )


def build_sequence_mixer_schedule(
    *,
    num_layers: int,
    layer_types: Sequence[object] | None,
    full_attention_interval: int | None,
    has_gated_delta_net: bool,
) -> tuple[SequenceMixerType, ...]:
    """Normalize explicit Qwen3.5 layer types or interval-derived schedule."""

    normalized_num_layers = int(num_layers)
    if normalized_num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if layer_types is not None:
        schedule = tuple(
            SequenceMixerType.from_config_value(value) for value in layer_types
        )
        if len(schedule) != normalized_num_layers:
            raise ValueError(
                "layer_types length must equal num_layers: "
                f"len(layer_types)={len(schedule)}, num_layers={normalized_num_layers}"
            )
    elif full_attention_interval is not None:
        interval = int(full_attention_interval)
        if interval <= 0:
            raise ValueError(
                "full_attention_interval must be positive, "
                f"got {full_attention_interval}"
            )
        schedule = tuple(
            SequenceMixerType.FULL_ATTENTION
            if (layer_id + 1) % interval == 0
            else SequenceMixerType.GATED_DELTA_NET
            for layer_id in range(normalized_num_layers)
        )
    else:
        schedule = (SequenceMixerType.FULL_ATTENTION,) * normalized_num_layers

    contains_gdn = SequenceMixerType.GATED_DELTA_NET in schedule
    if contains_gdn and not has_gated_delta_net:
        raise ValueError(
            "The decoder layer schedule contains gated-delta-network layers, "
            "but the GDN shape configuration is incomplete"
        )
    if has_gated_delta_net and not contains_gdn:
        raise ValueError(
            "GDN shape fields are configured, but the layer schedule does not "
            "select any gated-delta-network layers"
        )
    return schedule


def _gdn_shape_values(config: Any) -> tuple[Any, ...]:
    return tuple(
        getattr(config, field_name, None)
        for field_name in (
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
        )
    )


def _get_gdn_config(config: Any) -> GatedDeltaNetConfig | None:
    values = _gdn_shape_values(config)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("GDN shape configuration is incomplete")
    return GatedDeltaNetConfig(
        conv_kernel_size=int(values[0]),
        key_head_dim=int(values[1]),
        value_head_dim=int(values[2]),
        num_key_heads=int(values[3]),
        num_value_heads=int(values[4]),
        output_gate_type=str(getattr(config, "gdn_output_gate_type", "silu")),
    )


def resolve_layer_attention_specs(config: Any) -> tuple[LayerAttentionSpec, ...]:
    """Resolve one immutable, ordered attention spec per global layer.

    Only the explicitly registered Qwen3.5 profile activates GDN.  Existing
    Qwen3-Next configs may retain upstream ``linear_*`` fields while remaining
    on Frontier's historical homogeneous dense approximation.
    """

    num_layers = getattr(config, "num_layers", None)
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError("config must declare a positive num_layers")
    if not is_qwen3_5_profile_config(config):
        variant = _dense_variant(
            int(getattr(config, "num_q_heads")),
            int(getattr(config, "num_kv_heads")),
        )
        return tuple(
            LayerAttentionSpec(layer_id, "dense_attention", variant)
            for layer_id in range(num_layers)
        )

    gdn_config = _get_gdn_config(config)
    shape_values = _gdn_shape_values(config)
    has_complete_gdn_shape = all(value is not None for value in shape_values)
    schedule = build_sequence_mixer_schedule(
        num_layers=num_layers,
        layer_types=getattr(config, "layer_types", None),
        full_attention_interval=getattr(config, "full_attention_interval", None),
        has_gated_delta_net=has_complete_gdn_shape,
    )
    if gdn_config is None and SequenceMixerType.GATED_DELTA_NET in schedule:
        raise ValueError("Qwen3.5 GDN layers require complete GDN shape fields")
    dense_variant = _dense_variant(
        int(getattr(config, "num_q_heads")),
        int(getattr(config, "num_kv_heads")),
    )
    return tuple(
        LayerAttentionSpec(
            layer_id,
            "gated_delta_net"
            if mixer is SequenceMixerType.GATED_DELTA_NET
            else "dense_attention",
            "qwen3_5" if mixer is SequenceMixerType.GATED_DELTA_NET else dense_variant,
        )
        for layer_id, mixer in enumerate(schedule)
    )

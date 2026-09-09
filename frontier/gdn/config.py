"""Model-owned gated-delta-network shape and layer-schedule contracts.

The simulator and profiler both consume these helpers.  Keeping the schedule
normalization and state-shape arithmetic here prevents the two configuration
types from developing subtly different Qwen3.5 semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class SequenceMixerType(str, Enum):
    """The sequence-mixing block used by one decoder layer."""

    FULL_ATTENTION = "full_attention"
    GATED_DELTA_NET = "gated_delta_net"

    @classmethod
    def from_config_value(cls, value: object) -> "SequenceMixerType":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {
            "full_attention": cls.FULL_ATTENTION,
            "attention": cls.FULL_ATTENTION,
            "linear_attention": cls.GATED_DELTA_NET,
            "gated_delta_net": cls.GATED_DELTA_NET,
            "gdn": cls.GATED_DELTA_NET,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(
                "Unsupported sequence-mixer layer type "
                f"{value!r}; expected one of {sorted(aliases)}"
            ) from exc


@dataclass(frozen=True)
class GatedDeltaNetStateLayout:
    """Per-worker mutable state shapes for one GDN layer and request slot."""

    conv_state_shape: tuple[int, int]
    recurrent_state_shape: tuple[int, int, int]
    conv_bytes_per_element: int
    recurrent_bytes_per_element: int

    def __post_init__(self) -> None:
        if any(dimension <= 0 for dimension in self.conv_state_shape):
            raise ValueError(
                f"conv_state_shape must be positive, got {self.conv_state_shape}"
            )
        if any(dimension <= 0 for dimension in self.recurrent_state_shape):
            raise ValueError(
                "recurrent_state_shape must be positive, got "
                f"{self.recurrent_state_shape}"
            )
        if self.conv_bytes_per_element <= 0:
            raise ValueError(
                "conv_bytes_per_element must be positive, got "
                f"{self.conv_bytes_per_element}"
            )
        if self.recurrent_bytes_per_element <= 0:
            raise ValueError(
                "recurrent_bytes_per_element must be positive, got "
                f"{self.recurrent_bytes_per_element}"
            )

    @staticmethod
    def _num_elements(shape: Iterable[int]) -> int:
        result = 1
        for dimension in shape:
            result *= int(dimension)
        return result

    @property
    def conv_state_elements(self) -> int:
        return self._num_elements(self.conv_state_shape)

    @property
    def recurrent_state_elements(self) -> int:
        return self._num_elements(self.recurrent_state_shape)

    @property
    def total_elements(self) -> int:
        return self.conv_state_elements + self.recurrent_state_elements

    @property
    def total_bytes(self) -> int:
        return (
            self.conv_state_elements * self.conv_bytes_per_element
            + self.recurrent_state_elements * self.recurrent_bytes_per_element
        )


@dataclass(frozen=True)
class GatedDeltaNetConfig:
    """Architecture parameters required to model a Qwen-style GDN layer."""

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
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"GDN {field_name} must be positive, got {value}")
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
    ) -> GatedDeltaNetStateLayout:
        """Match vLLM's Qwen GDN conv and recurrent cache shapes.

        The physical conv-state orientation is runtime-configurable in vLLM;
        orientation does not affect its element or byte count, so Frontier uses
        the stable ``(width, channels)`` representation.
        """

        tp_size = int(tensor_parallel_size)
        if tp_size <= 0:
            raise ValueError(
                f"tensor_parallel_size must be positive, got {tensor_parallel_size}"
            )
        if int(num_speculative_tokens) < 0:
            raise ValueError(
                "num_speculative_tokens must be non-negative, got "
                f"{num_speculative_tokens}"
            )
        if self.num_key_heads % tp_size != 0:
            raise ValueError(
                "GDN key heads must be divisible by tensor_parallel_size: "
                f"num_key_heads={self.num_key_heads}, tp={tp_size}"
            )
        if self.num_value_heads % tp_size != 0:
            raise ValueError(
                "GDN value heads must be divisible by tensor_parallel_size: "
                f"num_value_heads={self.num_value_heads}, tp={tp_size}"
            )
        if self.conv_dim % tp_size != 0:
            raise ValueError(
                "GDN convolution channels must be divisible by "
                f"tensor_parallel_size: conv_dim={self.conv_dim}, tp={tp_size}"
            )

        conv_width = self.conv_kernel_size - 1 + int(num_speculative_tokens)
        return GatedDeltaNetStateLayout(
            conv_state_shape=(conv_width, self.conv_dim // tp_size),
            recurrent_state_shape=(
                self.num_value_heads // tp_size,
                self.value_head_dim,
                self.key_head_dim,
            ),
            # vLLM 0.28 resolves Qwen3.5's default ``auto`` cache policy to
            # BF16 convolution state and FP32 recurrent state. Callers may
            # override these independently when profiling another policy.
            conv_bytes_per_element=int(conv_bytes_per_element),
            recurrent_bytes_per_element=int(recurrent_bytes_per_element),
        )


def build_sequence_mixer_schedule(
    *,
    num_layers: int,
    layer_types: Sequence[object] | None,
    full_attention_interval: int | None,
    has_gated_delta_net: bool,
) -> tuple[SequenceMixerType, ...]:
    """Normalize explicit or interval-derived decoder layer types."""

    normalized_num_layers = int(num_layers)
    if normalized_num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")

    if layer_types is not None:
        schedule = tuple(
            SequenceMixerType.from_config_value(layer_type)
            for layer_type in layer_types
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
                "full_attention_interval must be positive, got "
                f"{full_attention_interval}"
            )
        schedule = tuple(
            SequenceMixerType.GATED_DELTA_NET
            if (layer_id + 1) % interval
            else SequenceMixerType.FULL_ATTENTION
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
            "GDN shape fields are configured, but layer_types/full_attention_interval "
            "does not select any gated-delta-network layers"
        )
    return schedule

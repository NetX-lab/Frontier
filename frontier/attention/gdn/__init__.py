"""CPU-safe structural contracts for gated-delta-network attention layers.

The package intentionally contains model semantics only.  Runtime kernels and
GPU profiling implementations belong to later, optional integration stages.
"""

from frontier.attention.gdn.config import (
    GatedDeltaNetConfig,
    LayerAttentionSpec,
    SequenceMixerType,
    build_sequence_mixer_schedule,
    is_qwen3_5_profile_config,
    resolve_layer_attention_specs,
)
from frontier.attention.gdn.memory import GatedDeltaNetStateLayout

__all__ = [
    "GatedDeltaNetConfig",
    "GatedDeltaNetStateLayout",
    "LayerAttentionSpec",
    "SequenceMixerType",
    "build_sequence_mixer_schedule",
    "is_qwen3_5_profile_config",
    "resolve_layer_attention_specs",
]

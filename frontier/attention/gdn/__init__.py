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
from frontier.attention.gdn.guards import (
    model_has_gdn,
    validate_gdn_runtime_support,
)
from frontier.attention.gdn.state import (
    GatedDeltaNetStateSlot,
    GatedDeltaNetStateSlotManager,
)

__all__ = [
    "GatedDeltaNetConfig",
    "GatedDeltaNetStateLayout",
    "GatedDeltaNetStateSlot",
    "GatedDeltaNetStateSlotManager",
    "LayerAttentionSpec",
    "SequenceMixerType",
    "build_sequence_mixer_schedule",
    "is_qwen3_5_profile_config",
    "model_has_gdn",
    "resolve_layer_attention_specs",
    "validate_gdn_runtime_support",
]

# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The Sarathi team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Positional Embeddings for profiling."""
import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from frontier.profiling.common.timer_stats_store import TimerStatsStore


_VLLM_GET_ROPE = None
_VLLM_GET_ROPE_IMPORT_ERROR = None
_VLLM_CUSTOM_OPS = None
_VLLM_CUSTOM_OPS_IMPORT_ERROR = None
_LOCAL_ROPE_DICT: Dict[Tuple[Any, ...], Any] = {}


def _freeze_rope_cache_value(value: Any) -> Any:
    """Convert nested rope-cache arguments into hashable values."""
    if isinstance(value, dict):
        return tuple(
            (key, _freeze_rope_cache_value(nested_value))
            for key, nested_value in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_rope_cache_value(nested_value) for nested_value in value)
    if isinstance(value, set):
        return tuple(
            sorted(_freeze_rope_cache_value(nested_value) for nested_value in value)
        )
    return value


def _build_rope_cache_key(
    *,
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: Union[int, float],
    is_neox_style: bool,
    rope_scaling: Optional[Dict[str, Any]],
    dtype: torch.dtype,
) -> Tuple[Any, ...]:
    return (
        int(head_size),
        int(rotary_dim),
        int(max_position),
        base,
        bool(is_neox_style),
        _freeze_rope_cache_value(rope_scaling),
        dtype,
    )


def clear_rope_cache() -> None:
    """Clear local fallback RoPE cache."""
    _LOCAL_ROPE_DICT.clear()


def _load_vllm_get_rope():
    global _VLLM_GET_ROPE
    global _VLLM_GET_ROPE_IMPORT_ERROR

    if _VLLM_GET_ROPE is not None or _VLLM_GET_ROPE_IMPORT_ERROR is not None:
        return _VLLM_GET_ROPE

    try:
        from vllm.model_executor.layers.rotary_embedding import get_rope as vllm_get_rope
    except Exception as exc:
        _VLLM_GET_ROPE_IMPORT_ERROR = exc
        return None

    _VLLM_GET_ROPE = vllm_get_rope
    return _VLLM_GET_ROPE


def _load_vllm_custom_ops():
    global _VLLM_CUSTOM_OPS
    global _VLLM_CUSTOM_OPS_IMPORT_ERROR

    if _VLLM_CUSTOM_OPS is not None or _VLLM_CUSTOM_OPS_IMPORT_ERROR is not None:
        return _VLLM_CUSTOM_OPS

    try:
        from vllm import _custom_ops as vllm_ops
    except Exception as exc:
        _VLLM_CUSTOM_OPS_IMPORT_ERROR = exc
        return None

    _VLLM_CUSTOM_OPS = vllm_ops
    return _VLLM_CUSTOM_OPS


def _should_prefer_torch_rope_fallback() -> bool:
    force_fallback = os.environ.get(
        "FRONTIER_PROFILING_FORCE_TORCH_FALLBACK",
        "",
    ).strip().lower()
    force_rope_fallback = os.environ.get(
        "FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK",
        "",
    ).strip().lower()
    if force_fallback in {"1", "true", "yes", "on"}:
        return True
    if force_rope_fallback in {"1", "true", "yes", "on"}:
        return True
    try:
        return bool(TimerStatsStore().disabled)
    except TypeError:
        return False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    This is a pure PyTorch implementation that replaces the custom CUDA kernel.

    Args:
        q: [num_tokens, num_heads * head_size] - query tensor
        k: [num_tokens, num_heads * head_size] - key tensor
        cos: [num_tokens, rotary_dim] - cosine values
        sin: [num_tokens, rotary_dim] - sine values
        is_neox_style: whether to use GPT-NeoX style rotation

    Returns:
        Tuple of rotated query and key tensors
    """
    # Get rotary dimension from cos/sin
    rotary_dim = cos.shape[-1]

    # Split q and k into rotary and non-rotary parts
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    # Apply rotation to rotary dimensions
    if is_neox_style:
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    else:
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    # Concatenate rotated and non-rotated parts
    q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)

    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style

        cache = self._compute_cos_sin_cache()
        cache = cache.to(torch.get_default_dtype())
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device="cuda")
                / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float, device="cuda")

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding using vLLM or a torch fallback."""
        if _should_prefer_torch_rope_fallback():
            positions = positions.flatten()
            if self.cos_sin_cache.device != query.device or (
                self.cos_sin_cache.dtype != query.dtype
            ):
                self.cos_sin_cache = self.cos_sin_cache.to(
                    query.device, dtype=query.dtype
                )
            cos_sin = self.cos_sin_cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)
            query, key = _apply_rotary_pos_emb(
                query,
                key,
                cos,
                sin,
                self.is_neox_style,
            )
            return query, key

        vllm_ops = _load_vllm_custom_ops()
        if vllm_ops is None:
            raise ImportError(
                "vLLM custom ops are required for RoPE profiling unless torch "
                "fallback is enabled."
            ) from _VLLM_CUSTOM_OPS_IMPORT_ERROR

        # Keep cache dtype/device aligned with the query to match vLLM behavior.
        if self.cos_sin_cache.device != query.device or (
            self.cos_sin_cache.dtype != query.dtype
        ):
            self.cos_sin_cache = self.cos_sin_cache.to(
                query.device, dtype=query.dtype
            )

        positions = positions.flatten()
        vllm_ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        max_len = self.max_position_embeddings * self.scaling_factor
        t = torch.arange(max_len, dtype=torch.float, device="cuda")
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        max_len = self.max_position_embeddings * self.scaling_factor
        base = self.base * (
            (self.scaling_factor * max_len / self.max_position_embeddings)
            - (self.scaling_factor - 1)
        ) ** (self.rotary_dim / (self.rotary_dim - 2))
        inv_freq = self._compute_inv_freq(base)
        t = torch.arange(max_len, dtype=torch.float, device="cuda")

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min: float, max: float, dim: int) -> torch.Tensor:
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor, self.attn_factor)
        )
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        max_len = self.max_position_embeddings * self.scaling_factor
        t = torch.arange(max_len, dtype=torch.float, device="cuda")

        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # Get n-d magnitude scaling corrected for interpolation.
        inv_freq_mask = 1 - _yarn_linear_ramp_mask(
            self.beta_fast, self.beta_slow, self.rotary_dim // 2
        ).to(device=inv_freq.device, dtype=torch.float32)
        inv_freq = inv_freq * (
            (1 - inv_freq_mask) * self.extrapolation_factor
            + inv_freq_mask * self.scaling_factor
        )

        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class Llama3RotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Llama 3.x scaling.
    
    This implements the extended context RoPE scaling used by Meta's Llama 3.x models.
    It uses frequency-based interpolation with high_freq_factor and low_freq_factor
    to smoothly interpolate between different frequency ranges.
    
    Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        original_max_position_embeddings: int,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos/sin cache using Llama 3.x frequency interpolation."""
        # Compute base inverse frequencies
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device="cuda")
                / self.rotary_dim
            )
        )
        
        # Llama 3.x frequency interpolation
        old_context_len = self.original_max_position_embeddings
        low_freq_wavelen = old_context_len / self.low_freq_factor
        high_freq_wavelen = old_context_len / self.high_freq_factor
        
        # Compute wavelengths for each frequency
        wavelens = 2 * math.pi / inv_freq
        
        # Apply smooth interpolation
        new_inv_freq = []
        for i, (wavelen, freq) in enumerate(zip(wavelens, inv_freq)):
            wavelen = wavelen.item()
            freq = freq.item()
            if wavelen < high_freq_wavelen:
                # High frequency: no scaling
                new_inv_freq.append(freq)
            elif wavelen > low_freq_wavelen:
                # Low frequency: full scaling
                new_inv_freq.append(freq / self.scaling_factor)
            else:
                # Smooth interpolation between high and low frequency
                smooth = (old_context_len / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_inv_freq.append((1 - smooth) * freq / self.scaling_factor + smooth * freq)
        
        inv_freq = torch.tensor(new_inv_freq, dtype=torch.float, device="cuda")
        
        # Compute cos/sin cache
        max_len = self.max_position_embeddings
        t = torch.arange(max_len, dtype=torch.float, device="cuda")
        
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


def _normalize_rope_scaling(
    rope_scaling: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if rope_scaling is None:
        return None
    if "rope_type" in rope_scaling:
        if "type" in rope_scaling and rope_scaling["type"] != rope_scaling["rope_type"]:
            raise ValueError(
                "rope_scaling has conflicting 'type' and 'rope_type' values: "
                f"{rope_scaling}"
            )
        return rope_scaling
    if "type" in rope_scaling:
        normalized = dict(rope_scaling)
        normalized["rope_type"] = normalized.pop("type")
        return normalized
    raise ValueError(
        "rope_scaling must contain 'rope_type' (or legacy 'type') key. "
        f"Got: {rope_scaling}"
    )


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool,
    rope_scaling: Optional[Dict[str, Any]],
    dtype: Optional[torch.dtype] = None,
) -> Any:
    """Factory function to create a RoPE embedding with a safe fallback."""
    rope_scaling = _normalize_rope_scaling(rope_scaling)
    rope_dtype = torch.get_default_dtype() if dtype is None else dtype
    cache_key = _build_rope_cache_key(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=base,
        is_neox_style=is_neox_style,
        rope_scaling=rope_scaling,
        dtype=rope_dtype,
    )

    if not _should_prefer_torch_rope_fallback():
        vllm_get_rope = _load_vllm_get_rope()
        if vllm_get_rope is not None:
            return vllm_get_rope(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position=max_position,
                base=base,
                is_neox_style=is_neox_style,
                rope_scaling=rope_scaling,
                dtype=rope_dtype,
            )

    if cache_key in _LOCAL_ROPE_DICT:
        return _LOCAL_ROPE_DICT[cache_key]

    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style
        )
        _LOCAL_ROPE_DICT[cache_key] = rotary_emb
        return rotary_emb

    scaling_type = rope_scaling["rope_type"]
    if scaling_type == "llama3":
        rotary_emb = Llama3RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling["factor"],
            rope_scaling["original_max_position_embeddings"],
            low_freq_factor=rope_scaling["low_freq_factor"],
            high_freq_factor=rope_scaling["high_freq_factor"],
        )
    elif scaling_type == "linear":
        rotary_emb = LinearScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling["factor"],
        )
    elif scaling_type == "dynamic":
        rotary_emb = DynamicNTKScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling["factor"],
        )
    elif scaling_type == "yarn":
        rotary_emb = YaRNScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling["factor"],
            extrapolation_factor=rope_scaling.get("extrapolation_factor", 1),
            attn_factor=rope_scaling.get("attn_factor", 1),
            beta_fast=rope_scaling.get("beta_fast", 32),
            beta_slow=rope_scaling.get("beta_slow", 1),
        )
    elif scaling_type == "default":
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style
        )
    else:
        raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    _LOCAL_ROPE_DICT[cache_key] = rotary_emb
    return rotary_emb

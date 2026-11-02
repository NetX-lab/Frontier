"""Latent-attention projections of vLLM's MLA wrapper for linear-op profiling.

``MlaAttentionLinearOps`` mirrors the projection path of vLLM v0.28
``MultiHeadLatentAttentionWrapper.forward`` (``layers/mla.py``) around the
attention kernel:

- ``attn_pre_proj``: replicated fused q_a/kv_a projection, the q_a and kv_a
  RMSNorms, and the TP-sharded q_b projection.
- ``attn_rope``: rotary embedding on the query and key rope slices.
- ``attn_post_proj``: TP-sharded o_proj.

The latent kv_b up-projection and the DSA indexer run inside the attention
kernel scopes measured by the attention producer, so they are not timed here.
"""

from __future__ import annotations

import torch

from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.profiling.common.layers.layernorm import RMSNorm
from frontier.profiling.common.layers.rotary_embedding import get_rope
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_utils.tensor_parallel_layers import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from frontier.profiling.common.utils import raise_if_fp8_requested

_REQUIRED_MLA_FIELDS = (
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)


def _required_positive_int(config: ModelConfig, name: str) -> int:
    value = getattr(config, name, None)
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"MLA linear-op profiling requires a positive int {name}, got {value!r}"
        )
    return value


class MlaAttentionLinearOps(torch.nn.Module):
    """Projection and RoPE operators of one MLA attention layer on one TP rank."""

    def __init__(self, config: ModelConfig, world_size: int):
        super().__init__()
        dims = {name: _required_positive_int(config, name) for name in _REQUIRED_MLA_FIELDS}
        if config.num_q_heads % world_size != 0:
            raise ValueError(
                f"num_q_heads={config.num_q_heads} must be divisible by TP={world_size}"
            )
        self.q_lora_rank = dims["q_lora_rank"]
        self.kv_lora_rank = dims["kv_lora_rank"]
        self.qk_nope_head_dim = dims["qk_nope_head_dim"]
        self.qk_rope_head_dim = dims["qk_rope_head_dim"]
        self.v_head_dim = dims["v_head_dim"]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.num_heads_per_worker = config.num_q_heads // world_size

        fp8_block_size = None
        if config.quantization_config is not None:
            fp8_block_size = config.quantization_config.weight_block_size
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.fused_qkv_a_proj = ReplicatedLinear(
            config.embedding_dim,
            self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            linear_metric_name=None,
            precision_op_name="attn_pre_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            config.num_q_heads * self.qk_head_dim,
            bias=False,
            gather_output=False,
            linear_metric_name=None,
            precision_op_name="attn_pre_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )
        self.o_proj = RowParallelLinear(
            config.num_q_heads * self.v_head_dim,
            config.embedding_dim,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            linear_metric_name="attn_post_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )
        # vLLM DeepseekV2MLAAttention builds the MLA rope with is_neox_style=False.
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=False,
            rope_scaling=config.rope_scaling,
        )
        raise_if_fp8_requested(
            "attn_rope",
            "FP8 RoPE kernel is unavailable for attn_rope profiling.",
        )
        self._attn_pre_proj_timer = CudaTimer("attn_pre_proj")
        self._attn_rope_timer = CudaTimer("attn_rope")

    def forward(self, hidden_states, positions):
        with self._attn_pre_proj_timer:
            qkv_lora, _ = self.fused_qkv_a_proj(hidden_states)
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            self.kv_a_layernorm(kv_c)
            k_pe = k_pe.unsqueeze(1)
            q, _ = self.q_b_proj(q_c)
            q = q.view(-1, self.num_heads_per_worker, self.qk_head_dim)
        with self._attn_rope_timer:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )
        attn_output = torch.randn(
            q.shape[0],
            self.num_heads_per_worker * self.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        output, _ = self.o_proj(attn_output)
        return output


__all__ = ["MlaAttentionLinearOps"]

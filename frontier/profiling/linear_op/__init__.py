"""
Linear Operations Profiling Module.

This module profiles linear operations (MLP, LayerNorm, projections, etc.)
for LLM inference simulation. These operations have linear complexity
with respect to sequence length.

Profiled operations include:
- MLP layers: mlp_up_proj, mlp_down_proj, mlp_act
- Normalization: input_layernorm, post_attention_layernorm
- Attention projections: attn_pre_proj, attn_post_proj, attn_rope
- Residual: add

Note: When is_moe=True, MLP-specific profiling is skipped because
MoE models use expert layers instead of dense MLP layers.
"""


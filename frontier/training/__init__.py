"""
Training module for Vidur execution time predictors.

This module provides standalone training capabilities for different model structures
(MoE, Attention, Linear Operations, etc.) to pre-train and save model weights for later use in simulations.

Model structure categories:
- attn: Attention operations (prefill, decode, KV cache)
- moe: Mixture of Experts operations (gating, shuffling, grouped GEMM)
- linear_op: Linear operations (MLP, LayerNorm, projections, residual add)
"""

from frontier.training.base_trainer import BaseTrainer
from frontier.training.moe_trainer import MoETrainer
from frontier.training.linear_op_trainer import LinearOpTrainer
from frontier.training.attention_trainer import AttentionTrainer

# Backward compatibility aliases
from frontier.training.linear_op_trainer import MLPTrainer

__all__ = [
    "BaseTrainer",
    "MoETrainer",
    "LinearOpTrainer",
    "AttentionTrainer",
    # Backward compatibility
    "MLPTrainer",
]


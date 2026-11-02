# Copyright 2023 The Sarathi team.
# Adapted from sarathi-serve-vidur/sarathi/model_executor/attention/no_op_attention_wrapper.py

"""No-op attention wrapper for profiling."""

from typing import List, Optional, Tuple

import torch

from frontier.profiling.attention.backends.base_attention_wrapper import (
    BaseAttentionWrapper,
)
from frontier.profiling.attention.sequence_metadata import SequenceMetadata
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig


class NoOpAttentionWrapper(BaseAttentionWrapper):
    """No-op attention wrapper that does not perform actual attention computation.

    This wrapper is useful for profiling other parts of the model without the
    overhead of attention computation. It simply returns an empty tensor with
    the correct shape.
    """

    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        """Initialize the no-op attention wrapper.

        Args:
            model_config: Model configuration.
            parallel_config: Parallel configuration.
            block_size: Size of each KV cache block.
            device: Device to run on.
        """
        self.device = device

    def get_cache_block(
        self, num_blocks: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a cache block (no-op).

        Args:
            num_blocks: Number of blocks.
            **kwargs: Additional arguments.

        Returns:
            Empty tuple (no cache blocks needed for no-op).
        """
        pass

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        """Begin forward pass (no-op).

        Args:
            seq_metadata_list: List of sequence metadata.
        """
        pass

    def end_forward(self):
        """End forward pass (no-op)."""
        pass

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Perform no-op forward pass.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            kv_cache: KV cache tensors.
            softmax_scale: Softmax scale factor.
            layer_id: Layer ID.

        Returns:
            Empty tensor with the same shape as query.
        """
        return torch.empty_like(query, device=self.device)


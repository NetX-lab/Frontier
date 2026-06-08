# Copyright 2023 The Sarathi team.
# Adapted from sarathi-serve-vidur/sarathi/model_executor/attention/base_attention_wrapper.py

"""Base attention wrapper for profiling."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch

from frontier.profiling.attention.sequence_metadata import SequenceMetadata
from frontier.profiling.common.constants import OperationMetrics
from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig


class BaseAttentionWrapper(ABC):
    """Base class for attention wrappers.

    This is an abstract base class that defines the interface for all attention
    backends used in profiling. Each backend (e.g., Flashinfer, NoOp) must
    implement this interface.
    """

    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        """Initialize the attention wrapper.

        Args:
            model_config: Model configuration.
            parallel_config: Parallel configuration.
            block_size: Size of each KV cache block.
            device: Device to run on.
        """
        self.device = device
        self.num_q_heads = model_config.get_num_q_heads(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_dim = model_config.get_head_size()
        self.dtype = model_config.dtype
        self.block_size = block_size
        self._timers = {}

    def get_timer(self, operation: OperationMetrics, layer_id: Optional[int] = None):
        """Get or create a timer for a specific operation and layer.

        For a given model, all layers share the same AttentionWrapper instance.
        However, we cannot have a single timer for all layers because the same
        timer cannot be turned on/off dynamically. So, we have timers for each
        layer separately.

        Args:
            operation: The operation to time.
            layer_id: The layer ID (optional).

        Returns:
            A CudaTimer instance for the operation and layer.
        """
        if self._timers.get((operation, layer_id)) is None:
            self._timers[(operation, layer_id)] = CudaTimer(operation, layer_id)
        return self._timers.get((operation, layer_id))

    @abstractmethod
    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        """Begin a forward pass with the given sequence metadata.

        This method is called before the forward pass to set up any necessary
        state based on the sequences being processed.

        Args:
            seq_metadata_list: List of sequence metadata for the batch.
        """
        pass

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of this attention wrapper.

        Returns:
            The singleton instance.
        """
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    @abstractmethod
    def end_forward(self):
        """End the forward pass and clean up any state."""
        pass

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Perform the attention forward pass.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            kv_cache: KV cache tensor(s).
            softmax_scale: Softmax scale factor.
            layer_id: Layer ID (optional).

        Returns:
            Output tensor.
        """
        pass


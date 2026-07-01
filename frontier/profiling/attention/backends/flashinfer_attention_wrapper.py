# Copyright 2023 The Sarathi team.
# Adapted from sarathi-serve-vidur/sarathi/model_executor/attention/flashinfer_attention_wrapper.py

"""Flashinfer attention wrapper for profiling."""

from typing import List, Optional
import os

import torch

try:
    from flashinfer import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchDecodeWithPagedKVCacheWrapper,
    )

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    BatchPrefillWithPagedKVCacheWrapper = None
    BatchDecodeWithPagedKVCacheWrapper = None

try:
    from vllm.v1.attention.backends.utils import get_kv_cache_layout
    # Import reshape_and_cache_flash from vllm._custom_ops (vLLM 0.10.x API)
    from vllm._custom_ops import reshape_and_cache_flash as _reshape_and_cache_flash

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    get_kv_cache_layout = None
    _reshape_and_cache_flash = None

from frontier.profiling.attention.backends.base_attention_wrapper import (
    BaseAttentionWrapper,
)
from frontier.attention.model_binding import bind_attention_family
from frontier.attention.ops import AttentionMemoryLayout
from frontier.profiling.attention.sequence_metadata import SequenceMetadata
from frontier.profiling.common.constants import OperationMetrics
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig


class FlashinferAttentionWrapper(BaseAttentionWrapper):
    """Flashinfer attention wrapper for profiling.

    This wrapper uses the Flashinfer library for efficient attention computation
    with paged KV cache. It supports both prefill and decode phases.
    """

    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        """Initialize the Flashinfer attention wrapper.

        Args:
            model_config: Model configuration.
            parallel_config: Parallel configuration.
            block_size: Size of each KV cache block.
            device: Device to run on.

        Raises:
            ImportError: If flashinfer is not installed.
        """
        if not HAS_FLASHINFER:
            raise ImportError(
                "flashinfer is not installed. Please install it to use FlashinferAttentionWrapper."
            )
        if not HAS_VLLM:
            raise ImportError(
                "vLLM is required for FlashInfer profiling alignment. "
                "Install vllm or set PYTHONPATH to the vllm source tree."
            )

        self._attention_family = bind_attention_family(model_config).family
        self._uses_latent_mla = (
            self._attention_family.memory_layout is AttentionMemoryLayout.LATENT_MLA
        )
        super().init(model_config, parallel_config, block_size, device)
        if self._uses_latent_mla:
            self._raise_mla_not_implemented()

        self.kv_cache_layout = get_kv_cache_layout()
        if self.kv_cache_layout == "NHD":
            self._kv_cache_stride_order = (0, 1, 2, 3, 4)
        elif self.kv_cache_layout == "HND":
            self._kv_cache_stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown KV cache layout: {self.kv_cache_layout}")

        self.kv_cache_dtype = "auto"
        self.kv_data_type = self.dtype
        self.k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0
        self.softmax_scale = 1.0 / (self.head_dim**0.5)

        # Increase workspace buffer size to handle larger batch sizes and models
        # with many KV heads or large head dimensions (e.g., head_dim=256).
        # NOTE: FlashInfer also maintains a separate internal "int" workspace
        # buffer (default is small, e.g. 8MB in 0.3.0) that can overflow for
        # large mixed prefill/decode planning workloads. We explicitly enlarge it
        # via reset_workspace_buffer below.
        workspace_gb = int(os.environ.get("FRONTIER_FLASHINFER_WORKSPACE_GB", "4"))
        int_workspace_mb = int(
            os.environ.get("FRONTIER_FLASHINFER_INT_WORKSPACE_MB", "512")
        )
        workspace_size = workspace_gb * 1024 * 1024 * 1024
        int_workspace_size = int_workspace_mb * 1024 * 1024
        prefill_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.uint8, device=device
        )
        self._prefill_workspace_buffer = prefill_workspace_buffer
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            prefill_workspace_buffer, self.kv_cache_layout
        )
        prefill_int_workspace_buffer = torch.empty(
            int_workspace_size, dtype=torch.uint8, device=device
        )
        self._prefill_int_workspace_buffer = prefill_int_workspace_buffer
        self.prefill_wrapper.reset_workspace_buffer(
            prefill_workspace_buffer, prefill_int_workspace_buffer
        )

        decode_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.uint8, device=device
        )
        self._decode_workspace_buffer = decode_workspace_buffer
        decode_int_workspace_buffer = torch.empty(
            int_workspace_size, dtype=torch.uint8, device=device
        )
        self._decode_int_workspace_buffer = decode_int_workspace_buffer
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            decode_workspace_buffer,
            self.kv_cache_layout,
            use_tensor_cores=True,
        )
        self.decode_wrapper.reset_workspace_buffer(
            decode_workspace_buffer, decode_int_workspace_buffer
        )

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.contains_prefill = False
        self.contains_decode = False
        self.num_prefill_tokens = 0
        self.num_total_tokens = 0

        self.slot_mapping = None

    def _raise_mla_not_implemented(self) -> None:
        raise NotImplementedError(
            "MLA profiling is not implemented in FlashinferAttentionWrapper. "
            "The dense FlashInfer path cannot be reused for MLA latent KV cache "
            "or vLLM V1 MLA physical scopes; add a dedicated MLA profiling "
            "backend before enabling use_mla here."
        )

    def to_int_tensor(self, data: List[int]) -> torch.Tensor:
        """Convert a list of integers to a CUDA tensor.

        Args:
            data: List of integers.

        Returns:
            CUDA tensor with dtype int32.
        """
        return torch.tensor(data, dtype=torch.int32, device="cuda")

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        """Get a cache block tensor.

        Args:
            num_blocks: Number of blocks.
            **kwargs: Additional arguments for tensor creation.

        Returns:
            Cache block tensor with shape (num_blocks, 2, block_size, num_kv_heads, head_dim).
        """
        if getattr(self, "_uses_latent_mla", False):
            self._raise_mla_not_implemented()
        return torch.randn(
            num_blocks,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        """Begin forward pass and prepare metadata for Flashinfer.

        This method processes the sequence metadata to create the necessary
        index tensors for Flashinfer's batched attention computation.

        Args:
            seq_metadata_list: List of sequence metadata for the batch.
        """
        prefill_qo_indptr: List[int] = [0]
        decode_qo_indptr: List[int] = [0]
        prefill_kv_page_indices: List[int] = []
        decode_kv_page_indices: List[int] = []
        prefill_kv_last_page_len: List[int] = []
        decode_kv_last_page_len: List[int] = []
        prefill_kv_page_indptr: List[int] = [0]
        decode_kv_page_indptr: List[int] = [0]

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        self.contains_prefill = False
        self.contains_decode = False

        slot_mapping: List[int] = []

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            self.contains_prefill = True

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            current_total_len = processed_prompt_len + prompt_chunk_len

            prefill_qo_indptr.append(prefill_qo_indptr[-1] + prompt_chunk_len)
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            prefill_kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            prefill_kv_page_indptr.append(
                prefill_kv_page_indptr[-1] + num_blocks_in_use
            )
            prefill_kv_last_page_len.append(
                current_total_len % self.block_size or self.block_size
            )

            for token_idx in range(processed_prompt_len, current_total_len):
                block_number = seq_metadata.block_table[token_idx // self.block_size]
                block_offset = token_idx % self.block_size
                slot_mapping.append(block_number * self.block_size + block_offset)

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            self.contains_decode = True

            context_len = seq_metadata.seq.get_len()
            decode_qo_indptr.append(decode_qo_indptr[-1] + 1)
            num_blocks_in_use = (context_len + self.block_size - 1) // self.block_size
            decode_kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            decode_kv_page_indptr.append(decode_kv_page_indptr[-1] + num_blocks_in_use)
            decode_kv_last_page_len.append(
                context_len % self.block_size or self.block_size
            )

            token_idx = context_len - 1
            block_number = seq_metadata.block_table[token_idx // self.block_size]
            block_offset = token_idx % self.block_size
            slot_mapping.append(block_number * self.block_size + block_offset)

        if self.contains_prefill:
            # Reset workspace allocator state for each planning iteration.
            # FlashInfer plan() can consume allocator space cumulatively across
            # repeated calls when using a long-lived wrapper instance.
            self.prefill_wrapper.reset_workspace_buffer(
                self._prefill_workspace_buffer,
                self._prefill_int_workspace_buffer,
            )
            prefill_qo_indptr_cpu = torch.tensor(
                prefill_qo_indptr, dtype=torch.int32, device="cpu"
            )
            prefill_kv_page_indptr_cpu = torch.tensor(
                prefill_kv_page_indptr, dtype=torch.int32, device="cpu"
            )
            prefill_kv_last_page_len_cpu = torch.tensor(
                prefill_kv_last_page_len, dtype=torch.int32, device="cpu"
            )
            prefill_kv_page_indices_tensor = self.to_int_tensor(
                prefill_kv_page_indices
            )
            self.prefill_wrapper.plan(
                prefill_qo_indptr_cpu,
                prefill_kv_page_indptr_cpu,
                prefill_kv_page_indices_tensor,
                prefill_kv_last_page_len_cpu,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                causal=True,
                pos_encoding_mode="NONE",
                sm_scale=self.softmax_scale,
                q_data_type=self.dtype,
                kv_data_type=self.kv_data_type,
            )

        if self.contains_decode:
            # Reset workspace allocator state for each planning iteration.
            self.decode_wrapper.reset_workspace_buffer(
                self._decode_workspace_buffer,
                self._decode_int_workspace_buffer,
            )
            decode_kv_page_indptr_cpu = torch.tensor(
                decode_kv_page_indptr, dtype=torch.int32, device="cpu"
            )
            decode_kv_last_page_len_cpu = torch.tensor(
                decode_kv_last_page_len, dtype=torch.int32, device="cpu"
            )
            decode_kv_page_indices_tensor = self.to_int_tensor(decode_kv_page_indices)
            self.decode_wrapper.plan(
                decode_kv_page_indptr_cpu,
                decode_kv_page_indices_tensor,
                decode_kv_last_page_len_cpu,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                pos_encoding_mode="NONE",
                sm_scale=self.softmax_scale,
                q_data_type=self.dtype,
                kv_data_type=self.kv_data_type,
            )

        self.num_prefill_tokens = prefill_qo_indptr[-1]
        self.num_total_tokens = self.num_prefill_tokens + len(decode_qo_indptr) - 1
        self.slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.long, device="cuda"
        )

    def end_forward(self):
        """End forward pass and clean up Flashinfer state."""
        self.is_metadata_initialized = False
        self.slot_mapping = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Perform attention forward pass using Flashinfer.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            kv_cache: KV cache tensor.
            softmax_scale: Softmax scale factor.
            layer_id: Layer ID.

        Returns:
            Output tensor.
        """
        assert self.is_metadata_initialized, "Metadata is not initialized."
        if getattr(self, "_uses_latent_mla", False):
            self._raise_mla_not_implemented()

        if self.is_profiling_iteration:
            # there is no need to call attention in profiling mode
            return torch.zeros_like(query)
        if softmax_scale != self.softmax_scale:
            raise ValueError(
                f"softmax_scale mismatch: expected {self.softmax_scale}, got {softmax_scale}. "
                "Re-plan the wrapper if you need a different scale."
            )

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        output = torch.empty_like(query)

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            if self.slot_mapping is None:
                raise RuntimeError("slot_mapping is not initialized.")
            if _reshape_and_cache_flash is None:
                raise RuntimeError(
                    "reshape_and_cache_flash is not available. "
                    "Please ensure vLLM is properly installed."
                )
            _reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                self.slot_mapping,
                self.kv_cache_dtype,
                self.k_scale,
                self.v_scale,
            )

        kv_cache_permute = kv_cache.permute(*self._kv_cache_stride_order)

        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
            if self.contains_prefill:
                self.prefill_wrapper.run(
                    query[: self.num_prefill_tokens],
                    kv_cache_permute,
                    k_scale=self.k_scale_float,
                    v_scale=self.v_scale_float,
                    out=output[: self.num_prefill_tokens],
                )

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            if self.contains_decode:
                self.decode_wrapper.run(
                    query[self.num_prefill_tokens : self.num_total_tokens],
                    kv_cache_permute,
                    k_scale=self.k_scale_float,
                    v_scale=self.v_scale_float,
                    out=output[self.num_prefill_tokens : self.num_total_tokens],
                )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        return output

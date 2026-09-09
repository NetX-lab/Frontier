"""vLLM ROCm attention backend for Frontier profiling."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import torch

from frontier.attention.model_binding import bind_attention_family
from frontier.attention.ops import AttentionMemoryLayout
from frontier.profiling.attention.backends.base_attention_wrapper import (
    BaseAttentionWrapper,
)
from frontier.profiling.attention.sequence_metadata import SequenceMetadata
from frontier.profiling.common.constants import OperationMetrics
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig


@dataclass(frozen=True)
class RocmSequencePlan:
    """CPU-side metadata needed to materialize one ROCm attention call."""

    query_lengths: List[int]
    sequence_lengths: List[int]
    block_tables: List[List[int]]
    slot_mapping: List[int]


def build_rocm_sequence_plan(
    seq_metadata_list: List[SequenceMetadata],
    block_size: int,
) -> RocmSequencePlan:
    query_lengths: List[int] = []
    sequence_lengths: List[int] = []
    block_tables: List[List[int]] = []
    slot_mapping: List[int] = []

    for seq_metadata in seq_metadata_list:
        if seq_metadata.block_table is None:
            raise ValueError("ROCm attention profiling requires a block table.")

        if seq_metadata.is_prompt:
            query_len = int(seq_metadata.prompt_chunk_len)
            processed_len = int(
                seq_metadata.seq.get_num_prompt_tokens_processed()
            )
            sequence_len = processed_len + query_len
            first_token = processed_len
        else:
            query_len = 1
            sequence_len = int(seq_metadata.seq.get_len())
            first_token = sequence_len - 1

        if query_len <= 0 or sequence_len <= 0:
            raise ValueError(
                "ROCm attention sequence lengths must be positive: "
                f"query_len={query_len}, sequence_len={sequence_len}."
            )

        blocks_in_use = (sequence_len + block_size - 1) // block_size
        block_table = list(seq_metadata.block_table[:blocks_in_use])
        if len(block_table) != blocks_in_use:
            raise ValueError(
                "Block table is too short for ROCm attention sequence: "
                f"required={blocks_in_use}, available={len(block_table)}."
            )

        query_lengths.append(query_len)
        sequence_lengths.append(sequence_len)
        block_tables.append(block_table)
        for token_idx in range(first_token, first_token + query_len):
            block_number = block_table[token_idx // block_size]
            block_offset = token_idx % block_size
            slot_mapping.append(block_number * block_size + block_offset)

    return RocmSequencePlan(
        query_lengths=query_lengths,
        sequence_lengths=sequence_lengths,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
    )


class VllmRocmAttentionWrapper(BaseAttentionWrapper):
    """Profile vLLM's native ROCm paged-attention implementation."""

    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        if not torch.version.hip:
            raise RuntimeError(
                "VLLM_ROCM attention requires a ROCm-enabled PyTorch build."
            )

        attention_family = bind_attention_family(model_config).family
        if attention_family.memory_layout is AttentionMemoryLayout.LATENT_MLA:
            raise NotImplementedError(
                "VLLM_ROCM profiling currently supports dense paged KV cache, "
                "not latent MLA cache layouts."
            )

        try:
            from vllm.v1.attention.backends.rocm_attn import (
                RocmAttentionImpl,
                RocmAttentionMetadata,
            )
        except ImportError as exc:
            raise ImportError(
                "The installed vLLM build does not provide its ROCm attention "
                "backend. Use a current vLLM ROCm image with gfx950 support."
            ) from exc

        super().init(model_config, parallel_config, block_size, device)
        if block_size % 16 != 0:
            raise ValueError(
                "vLLM ROCm paged attention requires block_size to be a "
                f"multiple of 16, got {block_size}."
            )

        self._metadata_cls = RocmAttentionMetadata
        self._scale = 1.0 / (self.head_dim**0.5)
        self._impl = RocmAttentionImpl(
            num_heads=self.num_q_heads,
            head_size=self.head_dim,
            scale=self._scale,
            num_kv_heads=self.num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )
        self._layer = SimpleNamespace(
            _k_scale=torch.tensor(1.0, dtype=torch.float32, device=device),
            _v_scale=torch.tensor(1.0, dtype=torch.float32, device=device),
            _k_scale_float=1.0,
            _v_scale_float=1.0,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
        )
        self.contains_prefill = False
        self.contains_decode = False
        self._prefill_metadata = None
        self._decode_metadata = None
        self._slot_mapping = None
        self._num_prefill_tokens = 0

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        return torch.randn(
            2,
            num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

    def _materialize_metadata(self, plan: RocmSequencePlan):
        query_start = [0]
        for query_len in plan.query_lengths:
            query_start.append(query_start[-1] + query_len)

        max_blocks = max(len(table) for table in plan.block_tables)
        padded_block_tables = [
            table + [0] * (max_blocks - len(table))
            for table in plan.block_tables
        ]
        return self._metadata_cls(
            num_actual_tokens=query_start[-1],
            max_query_len=max(plan.query_lengths),
            query_start_loc=torch.tensor(
                query_start, dtype=torch.int32, device=self.device
            ),
            max_seq_len=max(plan.sequence_lengths),
            seq_lens=torch.tensor(
                plan.sequence_lengths, dtype=torch.int32, device=self.device
            ),
            block_table=torch.tensor(
                padded_block_tables, dtype=torch.int32, device=self.device
            ),
            slot_mapping=torch.tensor(
                plan.slot_mapping, dtype=torch.long, device=self.device
            ),
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            prefix_scheduler_metadata=None,
            causal=True,
        )

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        prefill_sequences = [seq for seq in seq_metadata_list if seq.is_prompt]
        decode_sequences = [seq for seq in seq_metadata_list if not seq.is_prompt]
        ordered_sequences = prefill_sequences + decode_sequences
        plan = build_rocm_sequence_plan(ordered_sequences, self.block_size)

        self.contains_prefill = bool(prefill_sequences)
        self.contains_decode = bool(decode_sequences)
        self._num_prefill_tokens = sum(
            int(seq.prompt_chunk_len) for seq in prefill_sequences
        )
        self._slot_mapping = torch.tensor(
            plan.slot_mapping, dtype=torch.long, device=self.device
        )
        self._prefill_metadata = (
            self._materialize_metadata(
                build_rocm_sequence_plan(prefill_sequences, self.block_size)
            )
            if prefill_sequences
            else None
        )
        self._decode_metadata = (
            self._materialize_metadata(
                build_rocm_sequence_plan(decode_sequences, self.block_size)
            )
            if decode_sequences
            else None
        )

    def end_forward(self):
        self._prefill_metadata = None
        self._decode_metadata = None
        self._slot_mapping = None

    def _run_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata,
        output: torch.Tensor,
    ) -> None:
        self._impl.forward(
            self._layer,
            query,
            key,
            value,
            kv_cache,
            metadata,
            output,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        if self._slot_mapping is None:
            raise RuntimeError("ROCm attention metadata is not initialized.")
        if softmax_scale != self._scale:
            raise ValueError(
                f"softmax_scale mismatch: expected {self._scale}, got "
                f"{softmax_scale}."
            )

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(
                -1, self.num_q_heads, self.head_dim
            )
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(
                -1, self.num_kv_heads, self.head_dim
            )

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            self._impl.do_kv_cache_update(
                self._layer,
                key,
                value,
                kv_cache,
                self._slot_mapping,
            )

        output = torch.empty_like(query)
        prefill_end = self._num_prefill_tokens
        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
            if self._prefill_metadata is not None:
                self._run_attention(
                    query[:prefill_end],
                    key[:prefill_end],
                    value[:prefill_end],
                    kv_cache,
                    self._prefill_metadata,
                    output[:prefill_end],
                )

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            if self._decode_metadata is not None:
                self._run_attention(
                    query[prefill_end:],
                    key[prefill_end:],
                    value[prefill_end:],
                    kv_cache,
                    self._decode_metadata,
                    output[prefill_end:],
                )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)
        return output


__all__ = [
    "RocmSequencePlan",
    "VllmRocmAttentionWrapper",
    "build_rocm_sequence_plan",
]

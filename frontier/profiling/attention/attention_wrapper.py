from math import ceil
import os
from typing import List

import numpy as np
import torch

from frontier.profiling.attention.backends import (
    AttentionBackend,
    get_attention_wrapper,
    set_attention_backend,
)
from frontier.profiling.common.parallel_config import ParallelConfig

from frontier.profiling.attention.attention_input import AttentionInput
from frontier.profiling.attention.sequence_proxy import SequenceMetadataProxy
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.common.utils import (
    configure_quantization_manager_for_model_name,
    raise_if_fp8_requested,
)
from frontier.profiling.utils import ProfileMethod, normalize_profile_method
from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 3
ACTIVE_STEPS = 5
_ALLOW_ZERO_CUDA_OPS = {"attn_input_reshape", "attn_output_reshape"}


class AttentionWrapper:
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        max_num_blocks: int,
        max_model_len: int,
        block_size: int,
        attention_backend: AttentionBackend,
        dtype: torch.dtype,
        profile_method: str = "record_function",
        output_dir: str = "data/profiling",
    ):
        self.profile_method = normalize_profile_method(profile_method)
        self.time_stats_store = TimerStatsStore(profile_method=self.profile_method)
        self.output_dir = output_dir
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        self._model_config = model_config
        configure_quantization_manager_for_model_name(self._model_config.name)
        self._parallel_config = parallel_config
        self._dtype = dtype
        self._device = torch.device("cuda")

        self._max_model_len = max_model_len
        self._n_worker_q_heads = self._model_config.get_num_q_heads(
            self._parallel_config
        )
        self._n_worker_kv_heads = self._model_config.get_num_kv_heads(
            self._parallel_config
        )
        self._head_dim = self._model_config.get_head_size()
        self._softmax_scale = 1.0 / (self._head_dim**0.5)

        self._block_size = block_size

        self._attention_backend = attention_backend
        set_attention_backend(attention_backend)
        get_attention_wrapper().init(
            self._model_config,
            self._parallel_config,
            self._block_size,
            self._device,
        )
        self._max_blocks_per_sequence = ceil(max_model_len / self._block_size)
        # We create (big) KV tensors and reuse them
        self.max_num_blocks = max_num_blocks
        self.kv_cache = get_attention_wrapper().get_cache_block(
            self.max_num_blocks, dtype=self._dtype, device=self._device
        )

    def _validate_precision(self) -> None:
        raise_if_fp8_requested(
            "attn_kv_cache_save",
            "FP8 KV cache save kernel is unavailable for attn_kv_cache_save profiling.",
        )
        raise_if_fp8_requested(
            "attn_prefill",
            "FP8 attention prefill kernel is unavailable for attn_prefill profiling.",
        )
        raise_if_fp8_requested(
            "attn_decode",
            "FP8 attention decode kernel is unavailable for attn_decode profiling.",
        )

    def _get_allow_zero_cuda_ops_for_current_forward(self) -> set[str]:
        allowed_ops = set(_ALLOW_ZERO_CUDA_OPS)
        attention_wrapper = get_attention_wrapper()
        if not getattr(attention_wrapper, "contains_prefill", True):
            allowed_ops.add("attn_prefill")
        if not getattr(attention_wrapper, "contains_decode", True):
            allowed_ops.add("attn_decode")
        return allowed_ops

    def _get_input_tensors(
        self,
        attention_input: AttentionInput,
    ):
        num_tokens_per_seq = (
            attention_input.prefill_chunk_size if attention_input.is_prefill else 1
        )
        batch_size = attention_input.batch_size
        query = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_q_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        key = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        value = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        # Create SequenceMetadataProxy objects corresponding to AttentionInput
        seq_metadata_list: List[SequenceMetadataProxy] = []
        for _ in range(attention_input.batch_size):
            num_blocks = ceil(
                (num_tokens_per_seq + attention_input.kv_cache_size) / self._block_size
            )
            if num_blocks > self.max_num_blocks:
                raise ValueError(
                    "Requested block_table size exceeds max_num_blocks: "
                    f"num_blocks={num_blocks} max_num_blocks={self.max_num_blocks}"
                )
            seq_metadata = SequenceMetadataProxy(
                is_prompt=attention_input.is_prefill,
                total_len=num_tokens_per_seq + attention_input.kv_cache_size,
                processed_len=attention_input.kv_cache_size,
                block_table=list(range(num_blocks)),
            )
            seq_metadata_list.append(seq_metadata)
        return seq_metadata_list, query, key, value, self.kv_cache

    def _get_mixed_input_tensors(
        self,
        mixed_input: "MixedAttentionInput",
    ):
        """
        Generate input tensors for mixed-length batch profiling.
        
        Args:
            mixed_input: MixedAttentionInput specifying the batch configuration.
        
        Returns:
            Tuple of (seq_metadata_list, query, key, value, kv_cache).
        """
        batch_size = mixed_input.batch_size
        seq_lens = mixed_input.seq_lens
        total_tokens = sum(seq_lens)
        
        # Generate query/key/value tensors
        # Shape: (total_tokens, num_heads * head_dim)
        query = torch.randn(
            total_tokens,
            self._n_worker_q_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        key = torch.randn(
            total_tokens,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        value = torch.randn(
            total_tokens,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        
        # Create SequenceMetadataProxy objects for each sequence
        seq_metadata_list: List[SequenceMetadataProxy] = []
        for seq_len in seq_lens:
            # Calculate number of blocks needed for this sequence
            num_blocks = ceil(
                (seq_len + mixed_input.kv_cache_size) / self._block_size
            )
            if num_blocks > self.max_num_blocks:
                raise ValueError(
                    "Requested block_table size exceeds max_num_blocks: "
                    f"num_blocks={num_blocks} max_num_blocks={self.max_num_blocks}"
                )

            # Create metadata for this sequence
            seq_metadata = SequenceMetadataProxy(
                is_prompt=True,  # All sequences are prefill
                total_len=seq_len + mixed_input.kv_cache_size,
                processed_len=mixed_input.kv_cache_size,
                block_table=list(range(num_blocks)),
            )
            seq_metadata_list.append(seq_metadata)
        
        return seq_metadata_list, query, key, value, self.kv_cache

    def _get_true_mixed_input_tensors(
        self,
        true_mixed_input: "TrueMixedBatchInput",
    ):
        """Generate input tensors for true mixed batches.

        A true mixed batch contains both prefill sequences and decode sequences.
        Prefill sequences contribute ``prefill_seq_len`` new tokens each, while
        decode sequences contribute exactly one new token each.
        """
        total_tokens = (
            true_mixed_input.total_prefill_tokens + true_mixed_input.total_decode_tokens
        )
        query = torch.randn(
            total_tokens,
            self._n_worker_q_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        key = torch.randn(
            total_tokens,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        value = torch.randn(
            total_tokens,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )

        seq_metadata_list: List[SequenceMetadataProxy] = []
        next_block_index = 0

        for seq_len, kv_cache_size in zip(
            true_mixed_input.prefill_seq_lens,
            true_mixed_input.prefill_kv_cache_sizes,
        ):
            total_len = seq_len + kv_cache_size
            num_blocks = ceil(total_len / self._block_size)
            if next_block_index + num_blocks > self.max_num_blocks:
                raise ValueError(
                    "Requested block_table size exceeds max_num_blocks: "
                    f"num_blocks={next_block_index + num_blocks} "
                    f"max_num_blocks={self.max_num_blocks}"
                )
            seq_metadata_list.append(
                SequenceMetadataProxy(
                    is_prompt=True,
                    total_len=total_len,
                    processed_len=kv_cache_size,
                    block_table=list(range(next_block_index, next_block_index + num_blocks)),
                )
            )
            next_block_index += num_blocks

        for kv_cache_size in true_mixed_input.decode_kv_cache_sizes:
            total_len = kv_cache_size + 1
            num_blocks = ceil(total_len / self._block_size)
            if next_block_index + num_blocks > self.max_num_blocks:
                raise ValueError(
                    "Requested block_table size exceeds max_num_blocks: "
                    f"num_blocks={next_block_index + num_blocks} "
                    f"max_num_blocks={self.max_num_blocks}"
                )
            seq_metadata_list.append(
                SequenceMetadataProxy(
                    is_prompt=False,
                    total_len=total_len,
                    processed_len=kv_cache_size,
                    block_table=list(range(next_block_index, next_block_index + num_blocks)),
                )
            )
            next_block_index += num_blocks

        return seq_metadata_list, query, key, value, self.kv_cache

    @torch.inference_mode()
    def profile(
        self,
        attention_input: AttentionInput,
    ):
        # batch size is always 1 for prefill and can be different for decode
        assert attention_input.is_valid(self._max_model_len)
        self._validate_precision()

        seq_metadata_list, query, key, value, kv_cache = self._get_input_tensors(
            attention_input,
        )
        get_attention_wrapper().begin_forward(seq_metadata_list)

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # Warmup
            get_attention_wrapper().forward(
                query, key, value, kv_cache, softmax_scale=self._softmax_scale
            )
            torch.cuda.synchronize()

            self.time_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(
                self.output_dir,
                allow_zero_cuda_ops=self._get_allow_zero_cuda_ops_for_current_forward(),
            )
            with record_function_tracer:
                get_attention_wrapper().forward(
                    query, key, value, kv_cache, softmax_scale=self._softmax_scale
                )

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            for _ in range(WARMUP_STEPS):
                get_attention_wrapper().forward(
                    query, key, value, kv_cache, softmax_scale=self._softmax_scale
                )
            torch.cuda.synchronize()

            self.time_stats_store.clear_stats()

            for _ in range(ACTIVE_STEPS):
                get_attention_wrapper().forward(
                    query, key, value, kv_cache, softmax_scale=self._softmax_scale
                )
            torch.cuda.synchronize()

            time_stats = self.time_stats_store.get_stats()

        get_attention_wrapper().end_forward()

        # Derive per-sequence stats so merged CSVs keep mixed-related columns populated
        if attention_input.is_prefill:
            seq_lens = [attention_input.prefill_chunk_size] * attention_input.batch_size
        else:
            seq_lens = [1] * attention_input.batch_size  # decode path processes 1 token/seq

        total_tokens = sum(seq_lens)
        max_seq_len = max(seq_lens)
        min_seq_len = min(seq_lens)
        avg_seq_len = float(total_tokens) / len(seq_lens)
        equal_seq_len = int(np.sqrt(sum(x**2 for x in seq_lens)))
        seq_len_variance = float(np.var(seq_lens)) if len(seq_lens) > 1 else 0.0
        seq_len_std = float(np.sqrt(seq_len_variance))
        seq_len_cv = seq_len_std / avg_seq_len if avg_seq_len != 0 else 0.0

        return {
            "time_stats": time_stats,
            "n_embd": self._model_config.embedding_dim,
            "n_q_head": self._model_config.num_q_heads,
            "n_kv_head": self._model_config.num_kv_heads,
            "block_size": self._block_size,
            "num_tensor_parallel_workers": self._parallel_config.tensor_parallel_size,
            "max_model_len": self._max_model_len,
            "batch_size": attention_input.batch_size,
            "prefill_chunk_size": attention_input.prefill_chunk_size,
            "kv_cache_size": attention_input.kv_cache_size,
            "is_prefill": attention_input.is_prefill,
            "attention_backend": self._attention_backend,
            # Compatibility fields for mixed-batch profiling (even-length baseline)
            "is_mixed_batch": False,
            "mode": "even",
            "seq_lens": seq_lens,
            "total_tokens": total_tokens,
            "max_seq_len": max_seq_len,
            "min_seq_len": min_seq_len,
            "avg_seq_len": avg_seq_len,
            "equal_seq_len": equal_seq_len,
            "seq_len_variance": seq_len_variance,
            "seq_len_std": seq_len_std,
            "seq_len_cv": seq_len_cv,
        }

    @torch.inference_mode()
    def profile_mixed(
        self,
        mixed_input: "MixedAttentionInput",
    ):
        """
        Profile attention performance with mixed-length batch.
        
        This method profiles attention computation when a batch contains
        multiple sequences with potentially different lengths, which is
        common in real serving scenarios.
        
        Args:
            mixed_input: MixedAttentionInput specifying batch configuration.
        
        Returns:
            Dictionary containing profiling results and input metadata.
        """
        from frontier.profiling.attention.mixed_attention_input import MixedAttentionInput
        
        # Validate input
        if not mixed_input.is_valid(self._max_model_len, max_batch_size=128):
            raise ValueError(f"Invalid mixed input: {mixed_input}")
        self._validate_precision()
        
        # Generate input tensors
        seq_metadata_list, query, key, value, kv_cache = self._get_mixed_input_tensors(
            mixed_input,
        )
        
        # Begin forward pass
        get_attention_wrapper().begin_forward(seq_metadata_list)

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # Warmup
            get_attention_wrapper().forward(
                query, key, value, kv_cache, softmax_scale=self._softmax_scale
            )
            torch.cuda.synchronize()

            self.time_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(
                self.output_dir,
                allow_zero_cuda_ops=self._get_allow_zero_cuda_ops_for_current_forward(),
            )
            with record_function_tracer:
                get_attention_wrapper().forward(
                    query, key, value, kv_cache, softmax_scale=self._softmax_scale
                )

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            # Warmup iterations
            for _ in range(WARMUP_STEPS):
                get_attention_wrapper().forward(
                    query, key, value, kv_cache, softmax_scale=self._softmax_scale
                )
            torch.cuda.synchronize()

            # Clear statistics before active profiling
            self.time_stats_store.clear_stats()

            # Active profiling iterations
            for _ in range(ACTIVE_STEPS):
                get_attention_wrapper().forward(
                    query, key, value, kv_cache, softmax_scale=self._softmax_scale
                )
            torch.cuda.synchronize()

            time_stats = self.time_stats_store.get_stats()
        
        # End forward pass
        get_attention_wrapper().end_forward()
        
        # Collect results
        result = {
            "time_stats": time_stats,
            # Model configuration
            "n_embd": self._model_config.embedding_dim,
            "n_q_head": self._model_config.num_q_heads,
            "n_kv_head": self._model_config.num_kv_heads,
            "block_size": self._block_size,
            "num_tensor_parallel_workers": self._parallel_config.tensor_parallel_size,
            "max_model_len": self._max_model_len,
            "attention_backend": self._attention_backend,
            # Standard fields (for compatibility)
            "is_prefill": True,
            "prefill_chunk_size": 0,  # Not applicable for mixed batch
        }
        
        # Add mixed-batch specific fields
        result.update(mixed_input.to_dict())
        
        return result

    @torch.inference_mode()
    def profile_true_mixed(
        self,
        true_mixed_input: "TrueMixedBatchInput",
    ):
        """Profile a batch containing both prefill and decode sequences."""
        if not true_mixed_input.is_valid(self._max_model_len, max_batch_size=128):
            raise ValueError(f"Invalid true mixed input: {true_mixed_input}")
        self._validate_precision()

        seq_metadata_list, query, key, value, kv_cache = (
            self._get_true_mixed_input_tensors(true_mixed_input)
        )
        get_attention_wrapper().begin_forward(seq_metadata_list)

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            get_attention_wrapper().forward(
                query, key, value, kv_cache, softmax_scale=self._softmax_scale
            )
            torch.cuda.synchronize()

            self.time_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(
                self.output_dir,
                allow_zero_cuda_ops=self._get_allow_zero_cuda_ops_for_current_forward(),
            )
            with record_function_tracer:
                get_attention_wrapper().forward(
                    query, key, value, kv_cache, softmax_scale=self._softmax_scale
                )

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            for _ in range(WARMUP_STEPS):
                get_attention_wrapper().forward(
                    query, key, value, kv_cache, softmax_scale=self._softmax_scale
                )
            torch.cuda.synchronize()

            self.time_stats_store.clear_stats()

            for _ in range(ACTIVE_STEPS):
                get_attention_wrapper().forward(
                    query, key, value, kv_cache, softmax_scale=self._softmax_scale
                )
            torch.cuda.synchronize()

            time_stats = self.time_stats_store.get_stats()

        get_attention_wrapper().end_forward()

        result = {
            "time_stats": time_stats,
            "n_embd": self._model_config.embedding_dim,
            "n_q_head": self._model_config.num_q_heads,
            "n_kv_head": self._model_config.num_kv_heads,
            "block_size": self._block_size,
            "num_tensor_parallel_workers": self._parallel_config.tensor_parallel_size,
            "max_model_len": self._max_model_len,
            "attention_backend": self._attention_backend,
        }
        result.update(true_mixed_input.to_dict())
        return result

"""Portable attention wrapper for profiling, built on ``torch.nn.functional.scaled_dot_product_attention``.

Why this backend exists
-----------------------
The only real-kernel attention backend in this package is
:class:`FlashinferAttentionWrapper`, which is CUDA-only — FlashInfer has no ROCm
build, so it raises ``ImportError`` on AMD hardware. The only other option,
:class:`NoOpAttentionWrapper`, returns an empty tensor and therefore collects
zero real attention cost by design. That leaves no way to profile attention on
ROCm (e.g. MI355X / gfx950).

This backend uses only ``torch.nn.functional.scaled_dot_product_attention``,
which is core PyTorch and available on both CUDA and ROCm, so it produces real
measured attention timings on either platform.

Fidelity caveats — read before trusting the numbers
---------------------------------------------------
* **One launch when shapes are uniform, a loop when they are ragged.** A
  production paged attention kernel processes the whole ragged batch in a
  single launch. When every request in the batch shares the same query and
  KV length — which is what most of the profiling grid generates — this
  wrapper stacks them and issues exactly one SDPA call, so the measurement is
  not dominated by per-request launch overhead. Genuinely ragged batches fall
  back to a per-request loop and are pessimistic by roughly one kernel launch
  per extra request. Treat these numbers as a correctness-first reference
  baseline, not peak achievable performance.
* **Gathers the KV cache into contiguous tensors.** Paged KV blocks are gathered
  per request before the SDPA call. That gather is timed inside the prefill /
  decode scopes because it is work a real paged kernel would not do.
* **Bottom-right causal alignment.** With chunked prefill the query is the tail
  of the sequence, so top-left-aligned ``is_causal=True`` would be wrong. This
  wrapper always builds an explicit bottom-right-aligned mask, which is correct
  for both chunked and non-chunked prefill. Verify a non-chunked
  (single-chunk) case looks sane before trusting chunked rows.
* **Dense families only.** MLA's latent KV cache needs a different algorithm and
  a different set of timed scopes; this wrapper refuses those models rather
  than silently mismeasuring them.
"""

from typing import List, Optional, Tuple

import torch

from frontier.attention.model_binding import bind_attention_family
from frontier.attention.ops import AttentionFamilySpec, AttentionMemoryLayout
from frontier.profiling.attention.backends.base_attention_wrapper import (
    BaseAttentionWrapper,
)
from frontier.profiling.attention.sequence_metadata import SequenceMetadata
from frontier.profiling.common.constants import OperationMetrics
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig


def _has_native_gqa() -> bool:
    """Whether this PyTorch build's SDPA accepts ``enable_gqa`` (torch>=2.5)."""
    try:
        query = torch.zeros(1, 2, 1, 1)
        key = torch.zeros(1, 1, 1, 1)
        torch.nn.functional.scaled_dot_product_attention(
            query, key, key, enable_gqa=True
        )
    except TypeError:
        return False
    except Exception:
        # Any other failure means the kwarg was accepted but this CPU-side probe
        # shape was rejected; the kwarg itself exists.
        return True
    return True


class TorchSdpaAttentionWrapper(BaseAttentionWrapper):
    """Dense attention profiling backend using PyTorch SDPA.

    Portable across CUDA and ROCm. See the module docstring for the fidelity
    caveats that apply to every number this backend produces.
    """

    _inst = None

    def supports_attention_family(self, attention_family: AttentionFamilySpec) -> bool:
        """Only dense-compatible families; MLA needs its own algorithm and scopes."""
        return (
            attention_family.dense_compatible
            and attention_family.memory_layout is not AttentionMemoryLayout.LATENT_MLA
        )

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        """Initialize the SDPA attention wrapper.

        Args:
            model_config: Model configuration.
            parallel_config: Parallel configuration.
            block_size: Size of each KV cache block.
            device: Device to run on.

        Raises:
            NotImplementedError: If the model uses an MLA latent KV cache.
        """
        self._attention_family = bind_attention_family(model_config).family
        if self._attention_family.memory_layout is AttentionMemoryLayout.LATENT_MLA:
            raise NotImplementedError(
                "TorchSdpaAttentionWrapper implements the dense attention algorithm "
                "only. MLA uses a compressed latent KV cache and a different set of "
                "timed scopes; add a dedicated MLA backend instead of reusing this one."
            )

        super().init(model_config, parallel_config, block_size, device)

        self.softmax_scale = 1.0 / (self.head_dim**0.5)

        # torch>=2.5 dispatches grouped-query attention natively. Where it is not
        # available we expand KV heads by hand inside the attention scopes.
        self._supports_native_gqa = _has_native_gqa()

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.contains_prefill = False
        self.contains_decode = False
        self.num_prefill_tokens = 0
        self.num_total_tokens = 0

        # Per-request plans built in begin_forward and consumed in forward. Each
        # plan is the (block_index, block_offset) pair covering the whole
        # sequence so far, i.e. every token this request attends over.
        self._prefill_plans: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._prefill_query_lens: List[int] = []
        self._decode_plans: List[Tuple[torch.Tensor, torch.Tensor]] = []
        # Whether every request in the phase shares one (query_len, kv_len)
        # shape, which lets the whole phase run in a single SDPA launch.
        self._prefill_is_uniform = False
        self._decode_is_uniform = False
        # Write targets for this batch's new tokens.
        self._write_block_index: Optional[torch.Tensor] = None
        self._write_block_offset: Optional[torch.Tensor] = None

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        """Allocate a paged KV cache block tensor.

        Layout matches the FlashInfer backend's so that profiling rows collected
        with either backend describe the same memory shape:
        ``(num_blocks, 2, block_size, num_kv_heads, head_dim)``.
        """
        return torch.randn(
            num_blocks,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

    def _sequence_index(
        self, block_table: List[int], num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(block_index, block_offset)`` for every token in ``[0, num_tokens)``.

        Indexing the paged cache by this pair addresses ``kv_cache`` directly.
        A flat "slot" index would require collapsing the block and offset
        dimensions, and because the K/V dimension sits between them, that
        reshape would copy the whole cache on every call.
        """
        num_blocks_in_use = (num_tokens + self.block_size - 1) // self.block_size
        blocks = torch.tensor(
            block_table[:num_blocks_in_use], dtype=torch.long, device=self.device
        )
        offsets = torch.arange(self.block_size, dtype=torch.long, device=self.device)
        block_index = blocks.unsqueeze(1).expand(-1, self.block_size).reshape(-1)
        block_offset = offsets.unsqueeze(0).expand(num_blocks_in_use, -1).reshape(-1)
        return block_index[:num_tokens], block_offset[:num_tokens]

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        """Build per-request slot plans for the batch.

        Prefill sequences are ordered before decode sequences, matching the
        query-tensor layout the profiling harness passes to :meth:`forward`.
        """
        self.is_profiling_iteration = False
        self.is_metadata_initialized = True
        self.contains_prefill = False
        self.contains_decode = False

        self._prefill_plans = []
        self._prefill_query_lens = []
        self._decode_plans = []
        self._prefill_is_uniform = False
        self._decode_is_uniform = False

        write_blocks: List[int] = []
        write_offsets: List[int] = []
        num_prefill_tokens = 0

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

            self._prefill_plans.append(
                self._sequence_index(seq_metadata.block_table, current_total_len)
            )
            self._prefill_query_lens.append(prompt_chunk_len)
            num_prefill_tokens += prompt_chunk_len

            for token_idx in range(processed_prompt_len, current_total_len):
                write_blocks.append(seq_metadata.block_table[token_idx // self.block_size])
                write_offsets.append(token_idx % self.block_size)

        num_decode_tokens = 0
        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            self.contains_decode = True

            context_len = seq_metadata.seq.get_len()
            self._decode_plans.append(
                self._sequence_index(seq_metadata.block_table, context_len)
            )
            num_decode_tokens += 1

            token_idx = context_len - 1
            write_blocks.append(seq_metadata.block_table[token_idx // self.block_size])
            write_offsets.append(token_idx % self.block_size)

        self._prefill_is_uniform = self._plans_are_uniform(
            self._prefill_plans, self._prefill_query_lens
        )
        self._decode_is_uniform = self._plans_are_uniform(
            self._decode_plans, [1] * len(self._decode_plans)
        )

        self.num_prefill_tokens = num_prefill_tokens
        self.num_total_tokens = num_prefill_tokens + num_decode_tokens
        self._write_block_index = torch.tensor(
            write_blocks, dtype=torch.long, device=self.device
        )
        self._write_block_offset = torch.tensor(
            write_offsets, dtype=torch.long, device=self.device
        )

    def end_forward(self):
        """Release per-batch state."""
        self.is_metadata_initialized = False
        self._write_block_index = None
        self._write_block_offset = None
        self._prefill_plans = []
        self._prefill_query_lens = []
        self._decode_plans = []

    def _expand_kv_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads up to the query head count for grouped-query attention.

        Expands the head dimension, which is ``-3`` for both the unbatched
        ``(heads, seq, head_dim)`` and batched ``(batch, heads, seq, head_dim)``
        layouts this wrapper uses.
        """
        if self.num_q_heads == self.num_kv_heads:
            return tensor
        repeats = self.num_q_heads // self.num_kv_heads
        return tensor.repeat_interleave(repeats, dim=-3)

    def _sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run SDPA over ``(..., heads, seq, head_dim)`` inputs."""
        if self._supports_native_gqa and self.num_q_heads != self.num_kv_heads:
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                scale=self.softmax_scale,
                enable_gqa=True,
            )
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            self._expand_kv_heads(key),
            self._expand_kv_heads(value),
            attn_mask=attn_mask,
            scale=self.softmax_scale,
        )

    @staticmethod
    def _plans_are_uniform(
        plans: List[Tuple[torch.Tensor, torch.Tensor]], query_lens: List[int]
    ) -> bool:
        """Whether every request shares one ``(query_len, kv_len)`` shape."""
        if len(plans) < 2:
            return len(plans) == 1
        first_kv_len = plans[0][0].numel()
        first_query_len = query_lens[0]
        return all(
            plan[0].numel() == first_kv_len and query_len == first_query_len
            for plan, query_len in zip(plans, query_lens)
        )

    def _run_batched(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        kv_cache: torch.Tensor,
        plans: List[Tuple[torch.Tensor, torch.Tensor]],
        token_offset: int,
        query_len: int,
        attn_mask: Optional[torch.Tensor],
    ) -> None:
        """Run every request in ``plans`` in a single SDPA launch.

        Only valid when all requests share the same query and KV length, which
        :meth:`begin_forward` checks before setting the uniform flags. Writes
        results back into ``output`` in place.
        """
        batch_size = len(plans)
        block_index = torch.stack([plan[0] for plan in plans])
        block_offset = torch.stack([plan[1] for plan in plans])

        # (batch, kv_len, kv_heads, head_dim) -> (batch, kv_heads, kv_len, head_dim)
        k = kv_cache[block_index, 0, block_offset].permute(0, 2, 1, 3)
        v = kv_cache[block_index, 1, block_offset].permute(0, 2, 1, 3)

        num_tokens = batch_size * query_len
        q = query[token_offset : token_offset + num_tokens].reshape(
            batch_size, query_len, self.num_q_heads, self.head_dim
        ).transpose(1, 2)

        attn_out = self._sdpa(q, k, v, attn_mask)

        output[token_offset : token_offset + num_tokens] = attn_out.transpose(
            1, 2
        ).reshape(num_tokens, self.num_q_heads, self.head_dim)

    def _causal_mask(self, query_len: int, kv_len: int) -> Optional[torch.Tensor]:
        """Build a bottom-right-aligned causal mask.

        The query is the last ``query_len`` tokens of a ``kv_len``-token
        sequence, so query position ``i`` may attend to key positions up to
        ``kv_len - query_len + i``. Returns ``None`` when the mask would admit
        everything (single-query decode), letting SDPA skip masking entirely.
        """
        if query_len == 1:
            return None
        offset = kv_len - query_len
        q_idx = torch.arange(query_len, device=self.device).unsqueeze(1)
        k_idx = torch.arange(kv_len, device=self.device).unsqueeze(0)
        return k_idx <= (q_idx + offset)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Run dense attention for the planned batch and time each scope.

        Emits the same scopes as the FlashInfer backend
        (``ATTN_INPUT_RESHAPE``, ``ATTN_KV_CACHE_SAVE``, ``ATTN_PREFILL``,
        ``ATTN_DECODE``, ``ATTN_OUTPUT_RESHAPE``) so downstream training and
        simulation consume its CSV rows unchanged.
        """
        assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            # Memory-profiling iterations carry no block tables; nothing to compute.
            return torch.zeros_like(query)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        output = torch.empty_like(query)

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            if self._write_block_index is None:
                raise RuntimeError("KV cache write plan is not initialized.")
            # Scatter straight into the paged cache. Indexing by
            # (block, kv_slot, offset) keeps this a real scatter; collapsing
            # block/offset into a flat slot index would copy the whole cache.
            kv_cache[self._write_block_index, 0, self._write_block_offset] = key
            kv_cache[self._write_block_index, 1, self._write_block_offset] = value

        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
            if self.contains_prefill:
                if self._prefill_is_uniform:
                    query_len = self._prefill_query_lens[0]
                    kv_len = self._prefill_plans[0][0].numel()
                    self._run_batched(
                        query,
                        output,
                        kv_cache,
                        self._prefill_plans,
                        token_offset=0,
                        query_len=query_len,
                        attn_mask=self._causal_mask(query_len, kv_len),
                    )
                else:
                    token_offset = 0
                    for (block_index, block_offset), query_len in zip(
                        self._prefill_plans, self._prefill_query_lens
                    ):
                        kv_len = block_index.numel()
                        # (heads, seq, head_dim) for SDPA.
                        q = query[token_offset : token_offset + query_len].transpose(
                            0, 1
                        )
                        k = kv_cache[block_index, 0, block_offset].transpose(0, 1)
                        v = kv_cache[block_index, 1, block_offset].transpose(0, 1)
                        attn_out = self._sdpa(
                            q, k, v, self._causal_mask(query_len, kv_len)
                        )
                        output[token_offset : token_offset + query_len] = (
                            attn_out.transpose(0, 1)
                        )
                        token_offset += query_len

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            if self.contains_decode:
                if self._decode_is_uniform:
                    self._run_batched(
                        query,
                        output,
                        kv_cache,
                        self._decode_plans,
                        token_offset=self.num_prefill_tokens,
                        query_len=1,
                        attn_mask=None,
                    )
                else:
                    token_offset = self.num_prefill_tokens
                    for block_index, block_offset in self._decode_plans:
                        q = query[token_offset : token_offset + 1].transpose(0, 1)
                        k = kv_cache[block_index, 0, block_offset].transpose(0, 1)
                        v = kv_cache[block_index, 1, block_offset].transpose(0, 1)
                        attn_out = self._sdpa(q, k, v, None)
                        output[token_offset : token_offset + 1] = attn_out.transpose(
                            0, 1
                        )
                        token_offset += 1

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        return output

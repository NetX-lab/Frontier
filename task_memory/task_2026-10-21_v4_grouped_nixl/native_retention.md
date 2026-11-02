# Pinned native grouped NIXL decision

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Retained exact pinned method bodies and callsite excerpts for independent review; native rules unchanged. |

Revision: `2cf0a6915ce544dc493a0990f2ea38d81601128a`, read with `git show` from
`/data/ycfeng/Frontier/.real-engine/vLLM-BS`; its working HEAD is not the source pin.

## Callsite evidence

- `vllm/models/deepseek_v4/attention.py:716-738,762-774`: main compressed KV and
  indexer use `MLAAttentionSpec`, with logical block256 and compression metadata.
- `vllm/models/deepseek_v4/compressor.py:155-203`: float32 state, CSA window8 /
  block4, HCA window128 / block8, `SlidingWindowMLASpec`. These are not Mamba
  states. `vllm/v1/kv_cache_interface.py:595` derives that spec from sliding-window
  spec; `single_type_kv_cache_manager.py:1903-1913` selects SlidingWindowManager.
- `vllm/v1/core/kv_cache_manager.py:498-511`: ordinary allocation prunes on the
  processed frontier, excluding in-flight tokens. Thus a single initial6144
  prefill at frontier0 really allocates [24,96,96,1536,768], not the window tail.
- `vllm/v1/core/sched/scheduler.py:2677-2700`: at finish, call remove_skipped_blocks
  with processed computed tokens before computed-token clipping and connector
  `request_finished_all_groups`. `kv_cache_manager.py:714-732` clips to computed
  logical coverage, not a guessed page count.
- `sched/scheduler.py:530-544,1797-1807,1971-1978,2388-2394` and
  `sched/async_scheduler.py:26-41`: the one-output P request gains a placeholder
  after its full prefill, cannot schedule an extra decode step past max_tokens1,
  then drains its6144 in-flight tokens before the finish/free/connector hook.
  Thus the control's handoff in-flight count0 is reachable, not an assumption
  that every async request always has no in-flight work.
- `single_type_kv_cache_manager.py:597-661,1087-1098`: released whole pages become
  nulls; skipped count is floor(max(0,computed-window+1-extra)/group_block_size).
- `nixl/base_scheduler.py:128-150,240-284`: SW groups exchange their last
  ceil(window/block)+1 positions; Mamba speculative/single-state trimming does
  not apply to V4 states. SWA uses the accepted native window128/block64 layout.
- `nixl/pull_scheduler.py:153-167,261-279`: D exchanges unhashed/non-null IDs with
  the same clipper; P advertises clipped group IDs while retaining ownership.
- `nixl/base_worker.py:2412-2446`: without Mamba, take the P suffix matching each
  D group's count after kernel-block expansion. Here both sides share layout,
  so page selection is exactly the corresponding logical group suffix.
- `sched/scheduler.py:918-921,1033-1043`: remote reservation schedules zero local
  tokens and uses external computed tokens. `single_type_kv_cache_manager.py:
  230-279,288-325` creates skipped null prefixes then allocates only required
  external live pages. `sched/scheduler.py:2755-2762` replays prompt-1 after load.

## CPU observations at6144

| Group | Initial P allocation | P held at handoff | P connector list (incl. null) | D reserved | Selected transfer | Component bytes/page | Payload B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compressed | 24 | 24 | 24 | 24 | 24 | 1002240 | 24053760 |
| swa_0 | 96 | 2 | 3 | 2 | 2 | 823680 | 1647360 |
| swa_1 | 96 | 2 | 3 | 2 | 2 | 786240 | 1572480 |
| csa_state_0 | 1536 | 2 | 3 | 2 | 2 | 870912 | 1741824 |
| hca_state_0 | 768 | 16 | 17 | 16 | 16 | 656640 | 10506240 |
| **Total** | **2520** | **46** | **50** | **46** | **46** | n/a | **39521664** |

`native_rules.json` binds executed native bodies, source hashes and results at
prompts191/6144/6145. Fixture objects replace platform/runtime imports; these
are actual extracted methods, not a native GPU execution or wire-byte claim.

The implementation separately snapshots full logical P and D tables (including
nulls) and selected P transfer IDs. Bytes sum existing component-page sizes,
never maximum backing-slab size or ranks. Source and destination block ID spaces
remain separate; numeric ID coincidence across the two pools is harmless.

Structural completion still uses the accepted deterministic D-ready then P-release
approximation at one timestamp. Native readiness/notification ordering and time,
actual wire payloads, measured execution and formal acceptance remain open.

## Retained pinned method bodies

The following source is copied verbatim from the locked git object, not the
working native checkout. Each executed method's complete file hash was checked
against native_rules.json before capture. These excerpts let review inspect the
retention/transfer rules without native checkout access.

### `vllm/v1/core/single_type_kv_cache_manager.py:624-661`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/single_type_kv_cache_manager.py`.

```python
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.
        The removed blocks should be replaced by null_block.

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        differently for each attention type.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for attention types (e.g.
                R-SWA) that evict a middle gap rather than a head prefix. Ignored
                by the default implementation.
        """
        del num_prompt_tokens
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        self._remove_blocks_in_range(request_id, 0, num_skipped_blocks)
```

### `vllm/v1/core/single_type_kv_cache_manager.py:597-622`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/single_type_kv_cache_manager.py`.

```python
    def _remove_blocks_in_range(
        self,
        request_id: str,
        first_block: int,
        last_block: int,
    ) -> None:
        """Free blocks in ``[first_block, last_block)`` and replace with null_block.

        Iterates backward so newly-evictable tail blocks are reached even after
        earlier blocks in the range were nulled in a prior call.
        """
        if request_id not in self.req_to_blocks:
            return
        if first_block >= last_block:
            return
        blocks = self.req_to_blocks[request_id]
        last_block = min(last_block, len(blocks))

        freed: list[KVCacheBlock] = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] == self._null_block:
                break
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)
```

### `vllm/v1/core/single_type_kv_cache_manager.py:229-286`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/single_type_kv_cache_manager.py`.

```python
    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the locally cached (prefix-hit) blocks to the request:
        1. Touch the computed blocks (paired with adding them to `req_blocks`)
           so their ref_cnt exactly tracks the referencing requests.
        1.5. (Optional) For sliding window, skipped blocks are padded with nulls.
        2. Add the remaining computed blocks.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        # The coordinator only calls this for first-time allocations (running
        # requests are short-circuited there), so the request has no blocks yet.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
            # Record the partial tail for the CoW redirect in
            # allocate_new_blocks; cap the cached count at the full blocks so
            # cache_blocks() re-caches the private copy once full.
            block_idx = num_local_computed_tokens // self.block_size
            self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])
            self.num_cached_block[request_id] = block_idx
```

### `vllm/v1/core/single_type_kv_cache_manager.py:288-325`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/single_type_kv_cache_manager.py`.

```python
    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Allocate new blocks for external (KV-connector) computed tokens.

        Must run only after every group's local blocks have been touched via
        `add_local_computed_blocks`, so this group's `get_new_blocks` cannot
        evict another group's cache-hit blocks (issue #33775).

        Args:
            request_id: The request ID.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        if num_skipped_tokens > 0:
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens <= 0:
            return

        req_blocks = self.req_to_blocks[request_id]
        allocated_blocks = self.block_pool.get_new_blocks(
            cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        req_blocks.extend(allocated_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)
```

### `vllm/v1/core/single_type_kv_cache_manager.py:1063-1098`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/single_type_kv_cache_manager.py`.

```python
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.

        The trailing edge of the window is extended by ``extra_retained_tokens``
        so that those extra trailing tokens' blocks are retained (but not
        attended). This is needed for multi-module spec decoding which can
        re-prefill the last num_spec_prefill_tokens - 1 tokens from the end
        of the sequence, and thus needs to delay freeing/caching of blocks.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        return max(
            0,
            num_computed_tokens - self.sliding_window + 1 - self.extra_retained_tokens,
        )
```

### `vllm/v1/core/single_type_kv_cache_manager.py:663-674`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/single_type_kv_cache_manager.py`.

```python
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # The default behavior is to not skip any tokens.
        return 0
```

### `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:240-284`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py`.

```python
    def get_exchange_clipped_blocks(
        self, block_ids: BlockIds, clip_ssm: bool = True
    ) -> BlockIds:
        """Clip a request's block lists down to the transferable blocks.

        Sliding-window groups keep only the in-window tail: the KV cache
        manager allocates blocks for the entire sequence length and cleans up
        out-of-window blocks only prior to the `request_finished_all_groups`
        hook.

        SSM groups keep only their state-bearing slots: the trailing
        speculative scratch slots always go, and in single-state cache modes
        so does everything before the running state (null placeholders and
        the previous step's superseded state). "all" mode keeps its remaining
        slots, which the worker pairs position-wise.

        Use this at every block-id exchange point. Pass ``clip_ssm=False``
        for per-step partial lists (host-buffer save), where the SSM strip
        does not apply.
        """
        if len(block_ids) == 0 or not self._is_hma_required:
            # No blocks to clip eg Full prefix cache hit or not a hybrid model.
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable
        assert len(block_ids) == len(self.blocks_per_sw), (
            "Number of KV cache groups must match"
        )
        clipped = []
        for i, blocks in enumerate(block_ids):
            if n_sw := self.blocks_per_sw[i]:
                blocks = blocks[-n_sw:]
            elif (
                clip_ssm
                and blocks
                and (n_spec_blocks := self._ssm_spec_blocks[i]) is not None
            ):
                if n_spec := min(n_spec_blocks, len(blocks) - 1):
                    blocks = blocks[:-n_spec]
                if not self._ssm_state_slots_are_positional:
                    # Never empty: downstream reads that as a full prefix hit.
                    blocks = blocks[-1:]
            clipped.append(blocks)
        return tuple(clipped)
```

### `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2412-2508`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`.

```python
    def _apply_prefix_caching(
        self,
        decode_block_ids: BlockIds,
        prefill_block_ids: BlockIds,
        decode_physical_per_logical: int,
        prefill_physical_per_logical: int,
    ) -> tuple[BlockIds, BlockIds]:
        """Trim block ID lists so only the uncomputed suffix is transferred.

        Inputs are *kernel* (physical) block IDs, already expanded from logical IDs
        with each side's physical-per-logical ratio. Both pull and push call this after
        that expansion so the trim happens at kernel granularity.

        The prefix-cache hit is always on the decode (D) side, so ``decode``
        holds only its uncomputed blocks while prefill (P) holds the full
        sequence. This is mode-independent: pull passes its own (D) blocks as
        ``decode``, push passes the remote D registration as ``decode``.

        For non-Mamba models: end-trim ``prefill`` to match ``decode`` count, so
        already-cached prefix blocks are skipped in the transfer.

        For Mamba hybrid: SSM groups pair state slots by position and FA
        groups end-trim to the uncomputed suffix when physical-per-logical
        matches.
        """
        # Partial prefix cache hit: just transfer uncomputed blocks.
        # Skip mamba groups — their blocks represent full state (conv+ssm),
        # not per-token data, so trimming would corrupt the transfer.
        prefill_block_ids = list(prefill_block_ids)
        if not self._has_mamba:
            for i, prefill_group in enumerate(prefill_block_ids):
                num_decode_blocks = len(decode_block_ids[i])
                assert num_decode_blocks <= len(prefill_group)
                if num_decode_blocks < len(prefill_group):
                    prefill_block_ids[i] = prefill_group[-num_decode_blocks:]
        else:
            # (NOTE: ZhanqiuHu) HeteroTP can cause different kernel block counts
            # due to logical block rounding.
            # Example: 640 prompt tokens, kernel_block_size=64
            #   prefill physical_per_logical=10, decode physical_per_logical=6
            #   prefill logical ids from kv_transfer_params = [0]
            #   decode logical ids allocated = [0, 1]
            #   prefill kernel blocks: [0..9]  (1*10=10)
            #   decode kernel blocks:  [0..11] (2*6=12)
            #   actual data blocks = ceil(640/64) = 10, trim both to 10
            # Vice versa (prefill physical_per_logical=6, decode=10):
            #   prefill logical ids = [0, 1], decode logical ids = [0]
            #   prefill kernel blocks: [0..11] (2*6=12)
            #   decode kernel blocks:  [0..9]  (1*10=10)
            #   actual data blocks = ceil(640/64) = 10, trim both to 10
            decode_block_ids = list(decode_block_ids)
            for i, prefill_group in enumerate(prefill_block_ids):
                num_decode_blocks = len(decode_block_ids[i])
                num_prefill_blocks = len(prefill_group)
                if _is_ssm_spec(self._group_spec_types[i]):
                    if num_decode_blocks == num_prefill_blocks:
                        continue
                    # Only state-bearing slots reach here, single-state modes
                    # just one (see get_exchange_clipped_blocks), so differing
                    # counts mean position-indexed "all"-mode lists. A longer
                    # prefill list carries earlier positions the decode side
                    # already has (prefix hit) -> take its tail; a longer decode
                    # list holds the position D recomputes itself, which gets
                    # no prefill state.
                    assert num_decode_blocks - num_prefill_blocks <= 1, (
                        f"Group {i}: unpairable SSM state slots, "
                        f"decode={num_decode_blocks} prefill={num_prefill_blocks}"
                    )
                    num_blocks = min(num_decode_blocks, num_prefill_blocks)
                    if num_decode_blocks < num_prefill_blocks:
                        prefill_block_ids[i] = prefill_group[-num_blocks:]
                    else:
                        decode_block_ids[i] = decode_block_ids[i][:num_blocks]
                elif (
                    decode_physical_per_logical == prefill_physical_per_logical
                    and num_decode_blocks < num_prefill_blocks
                ):
                    # Partial prefix cache hit for FA group.
                    prefill_block_ids[i] = prefill_group[-num_decode_blocks:]
                else:
                    # TODO Handle prefix caching with different block_sizes
                    # Allocation rounding legitimately leaves up to
                    # ppl - 1 trailing dead kernel blocks per side (plus one
                    # extra decode block for the recomputed final token), so
                    # the counts may differ by up to the sum of the two
                    # ratios; anything larger indicates mismatched lists.
                    max_padding = (
                        decode_physical_per_logical + prefill_physical_per_logical
                    )
                    assert abs(num_decode_blocks - num_prefill_blocks) <= max_padding, (
                        f"Group {i}: |{num_decode_blocks} - "
                        f"{num_prefill_blocks}| > {max_padding}"
                    )
                    num_blocks = min(num_decode_blocks, num_prefill_blocks)
                    decode_block_ids[i] = decode_block_ids[i][:num_blocks]
                    prefill_block_ids[i] = prefill_group[:num_blocks]
        return decode_block_ids, prefill_block_ids
```

### `vllm/v1/core/sched/scheduler.py:2665-2700`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/sched/scheduler.py`.

```python
    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector, on the processed-token basis (see `allocate_slots`).
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            ),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,
            num_computed_tokens=request.num_computed_tokens,
        )

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)
```

### `vllm/v1/core/kv_cache_manager.py:99-109`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/kv_cache_manager.py`.

```python
    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]
```

### `vllm/v1/core/kv_cache_manager.py:714-732`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/kv_cache_manager.py`.

```python
    def get_block_ids_for_computed_tokens(
        self,
        request_id: str,
        num_computed_tokens: int,
    ) -> tuple[list[int], ...]:
        """Get block ids covering the request's computed tokens."""
        block_ids = self.get_block_ids(request_id)
        clipped_block_ids: list[list[int]] = []
        for group, ids in zip(self.kv_cache_config.kv_cache_groups, block_ids):
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec) or isinstance(
                spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            ):
                clipped_block_ids.append(ids)
                continue

            num_valid_blocks = cdiv(num_computed_tokens, spec.block_size)
            clipped_block_ids.append(ids[:num_valid_blocks])
        return tuple(clipped_block_ids)
```

### `vllm/v1/core/kv_cache_coordinator.py:370-391`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/kv_cache_coordinator.py`.

```python
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length. R-SWA managers use this to
                free gap blocks between the prefill tail and decode window; other
                manager types ignore it.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )
```

### `vllm/models/deepseek_v4/attention.py:700-781`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/models/deepseek_v4/attention.py`.

```python
            self._flashinfer_fp8_q_scale_inv,
            self.eps,
            block_size,
        )
        return q_fp8

    def _global_topk_output_buffers(
        self, topk_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.compress_ratio != 4 or self.eager_scratch_pool is None:
            return None
        return self.eager_scratch_pool.global_topk_outputs(topk_indices)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
        # pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token;
            # head_size stays semantic (512).
            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,
        )


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


```

### `vllm/models/deepseek_v4/compressor.py:140-203`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/models/deepseek_v4/compressor.py`.

```python
            )
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
            num_decode_tokens=num_decode_tokens,
            c128_boundary=(
                _get_c128_boundary(common_attn_metadata)
                if self.block_size == 8
                else None
            ),
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )
```

### `vllm/v1/kv_cache_interface.py:590-619`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/kv_cache_interface.py`.

```python
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None
    # MLA stores a single latent vector per state; there is no separate V.
    head_size_v: int = 0

    def __post_init__(self):
        assert self.model_version in (None, "deepseek_v4"), (
            f"Unsupported model version: {self.model_version}"
        )
        super().__post_init__()
        _apply_alignment_padding(self)

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, SlidingWindowMLASpec) for spec in specs), (
```

### `vllm/v1/core/single_type_kv_cache_manager.py:1885-1918`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/single_type_kv_cache_manager.py`.

```python
        (SlidingWindowSpec, ChunkedLocalAttentionSpec),
    ):
        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
            )
        )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager


def register_all_kvcache_specs(vllm_config):
    """Built-in spec registration"""
    KVCacheSpecRegistry.register(
        FullAttentionSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )

    KVCacheSpecRegistry.register(
        SlidingWindowSpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowSpec,
    )
    KVCacheSpecRegistry.register(
        SlidingWindowMLASpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowMLASpec,
    )

    KVCacheSpecRegistry.register(
        MambaSpec, MambaManager, uniform_type_base_spec=MambaSpec
    )
```

### `vllm/v1/core/sched/scheduler.py:530-544`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/sched/scheduler.py`.

```python
            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue
```

### `vllm/v1/core/sched/scheduler.py:1797-1807`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/sched/scheduler.py`.

```python
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
```

### `vllm/v1/core/sched/scheduler.py:1971-1978`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/sched/scheduler.py`.

```python
            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params, ec_transfer_params = self._free_request(request)
```

### `vllm/v1/core/sched/scheduler.py:2388-2394`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/sched/scheduler.py`.

```python
    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()

        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
```

### `vllm/v1/core/sched/async_scheduler.py:26-41`

Pinned file read: `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/v1/core/sched/async_scheduler.py`.

```python
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
```

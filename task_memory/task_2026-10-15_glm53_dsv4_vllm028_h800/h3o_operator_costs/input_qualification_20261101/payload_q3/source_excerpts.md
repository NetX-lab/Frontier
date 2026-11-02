# Q3 source excerpts

Read-only source; original one-based line numbers. SHA256 is in results.json.

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py
```text
88:         self._is_hma_required = (
89:             not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
90:             # Also handle unlikely SW-only model case instead of checking num_groups>1.
91:             and any(
92:                 not isinstance(g.kv_cache_spec, FullAttentionSpec)
93:                 for g in kv_cache_config.kv_cache_groups
94:             )
95:         )
96:         self._has_mamba = any(
97:             isinstance(g.kv_cache_spec, MambaSpec)
98:             for g in kv_cache_config.kv_cache_groups
99:         )
```
```text
128:         # Gather Sliding Window sizes for each kv cache group (if any) in number of
129:         # blocks per KV cache group. This is used to clip the local attention window.
130:         sw_sizes_tokens: list[tuple[int, int]] = [
131:             (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
132:             if isinstance(g.kv_cache_spec, SlidingWindowSpec)
133:             else (0, self.block_size)
134:             for g in kv_cache_config.kv_cache_groups
135:         ]
136:         # cdiv(n_tokens, block_size) gives blocks/window; add 1 to conservatively
137:         # account for boundary overlap eg window isn't fully aligned with blocks.
138:         self.blocks_per_sw = [
139:             cdiv(n_tokens, block_size) + 1 if n_tokens else 0
140:             for n_tokens, block_size in sw_sizes_tokens
141:         ]
142: 
143:         # Trailing scratch slots that mamba managers co-allocate per request
144:         # for speculative decoding; None for non-SSM groups.
145:         self._ssm_spec_blocks = [
146:             g.kv_cache_spec.num_speculative_blocks
147:             if isinstance(g.kv_cache_spec, MambaSpec)
148:             else None
149:             for g in kv_cache_config.kv_cache_groups
150:         ]
```
```text
240:     def get_exchange_clipped_blocks(
241:         self, block_ids: BlockIds, clip_ssm: bool = True
242:     ) -> BlockIds:
243:         """Clip a request's block lists down to the transferable blocks.
244: 
245:         Sliding-window groups keep only the in-window tail: the KV cache
246:         manager allocates blocks for the entire sequence length and cleans up
247:         out-of-window blocks only prior to the `request_finished_all_groups`
248:         hook.
249: 
250:         SSM groups keep only their state-bearing slots: the trailing
251:         speculative scratch slots always go, and in single-state cache modes
252:         so does everything before the running state (null placeholders and
253:         the previous step's superseded state). "all" mode keeps its remaining
254:         slots, which the worker pairs position-wise.
255: 
256:         Use this at every block-id exchange point. Pass ``clip_ssm=False``
257:         for per-step partial lists (host-buffer save), where the SSM strip
258:         does not apply.
259:         """
260:         if len(block_ids) == 0 or not self._is_hma_required:
261:             # No blocks to clip eg Full prefix cache hit or not a hybrid model.
262:             return block_ids
263:         # NOTE (NickLucche) This logic is currently handled at the connector level
264:         # because offloading connectors might want to receive the whole sequence even
265:         # for SWA groups. We will abstract this logic once the interface is more stable
266:         assert len(block_ids) == len(self.blocks_per_sw), (
267:             "Number of KV cache groups must match"
268:         )
269:         clipped = []
270:         for i, blocks in enumerate(block_ids):
271:             if n_sw := self.blocks_per_sw[i]:
272:                 blocks = blocks[-n_sw:]
273:             elif (
274:                 clip_ssm
275:                 and blocks
276:                 and (n_spec_blocks := self._ssm_spec_blocks[i]) is not None
277:             ):
278:                 if n_spec := min(n_spec_blocks, len(blocks) - 1):
279:                     blocks = blocks[:-n_spec]
280:                 if not self._ssm_state_slots_are_positional:
281:                     # Never empty: downstream reads that as a full prefix hit.
282:                     blocks = blocks[-1:]
283:             clipped.append(blocks)
284:         return tuple(clipped)
```
```text
372:     def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
373:         """D-side only. Returns N-1 for Mamba models since the decoder
374:         always recomputes the last token and must start from h(N-1)."""
375:         if self._has_mamba and num_prompt_tokens > 1:
376:             return num_prompt_tokens - 1
377:         return num_prompt_tokens
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py
```text
61:             # Remote prefill: get all prompt blocks from remote.
62:             token_ids = request.prompt_token_ids or []
63:             actual = self._get_remote_prefill_token_count(len(token_ids))
64:             count = actual - num_computed_tokens
65:             if count > 0:
66:                 return count, True
```
```text
84:         ):
85:             # Decode node has kv blocks for part of prefill request, so, provide them
86:             # as an external token count to scheduler.
87:             # The tokens will be loaded if not already present
88:             # in the prefill node local cache
89:             remote_num_tokens = params.get("remote_num_tokens") or 0
90:             count = (
91:                 min(remote_num_tokens, request.num_prompt_tokens) - num_computed_tokens
92:             )
93:             if count > 0:
94:                 # Check kv_recompute_threshold: skip pull if
95:                 # remote tokens are below the threshold.
96:                 if (
97:                     self.kv_recompute_threshold > 0
98:                     and count < self.kv_recompute_threshold
99:                 ):
100:                     logger.debug(
101:                         "Skipping remote pull for %s: %d remote tokens < threshold %d",
102:                         request.request_id,
103:                         count,
104:                         self.kv_recompute_threshold,
105:                     )
106:                     return 0, False
107:                 return count, True
108: 
```
```text
149:                     # If remote_blocks and num_external_tokens = 0, we have
150:                     # a full prefix cache hit on the local node. We need to call
151:                     # send_notif in _read_blocks to free the memory on the remote node.
152: 
153:                     unhashed_local_block_ids: BlockIds = (
154:                         blocks.get_unhashed_block_ids_all_groups()
155:                         if num_external_tokens > 0
156:                         else ()
157:                     )
158:                     local_block_ids = self.get_exchange_clipped_blocks(
159:                         unhashed_local_block_ids
160:                     )
161: 
162:                     # Get unhashed blocks to pull from remote. Mind that a full prefix
163:                     # cache hit is indicated with an empty list.
164:                     self._reqs_need_recv[request.request_id] = (
165:                         request,
166:                         local_block_ids,
167:                     )
168: 
169:                 else:
170:                     logger.warning(
171:                         "Got invalid KVTransferParams: %s. This "
172:                         "request will not utilize KVTransfer",
173:                         params,
174:                     )
175:             else:
176:                 assert num_external_tokens == 0
177:             # Only trigger 1 KV transfer per request.
178:             params["do_remote_prefill"] = False
179:             params["_remote_blocks_processed"] = True
```
```text
261:             # NOTE HMA will "mark" empty/null blocks in groups with 0s (eg SWA ones),
262:             # trimming down after allocating for the whole sequence length. Empty
263:             # blocks are always at the start of the list.
264:             # Here we "unpad" blocks to send the actual remote blocks to be read.
265:             block_ids = self.get_exchange_clipped_blocks(block_ids)
266: 
267:             remote_num_tokens = request.num_computed_tokens
268: 
269:         return delay_free_blocks, dict(
270:             do_remote_prefill=is_p_node,
271:             do_remote_decode=is_d_node,
272:             remote_block_ids=block_ids,
273:             remote_engine_id=self.engine_id,
274:             remote_request_id=request.request_id,
275:             remote_host=self.side_channel_host,
276:             remote_port=self.side_channel_port,
277:             tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
278:             remote_num_tokens=remote_num_tokens,
279:             remote_blocks_expiry_time=blocks_expiry_time,
280:             transfer_mode=self._TRANSFER_MODE,
281:         )
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py
```text
99:     def _compute_desc_ids(
100:         self,
101:         block_ids: BlockIds,
102:         dst_num_blocks: int,
103:         block_size_ratio: float | None,
104:         physical_blocks_per_logical: int,
105:     ) -> np.ndarray:
106:         """Compute NIXL descriptor IDs for given block IDs."""
107:         num_ssm_regions = 0
108:         if self._has_mamba:
109:             assert self._conv_decomp is not None
110:             # NIXL regions per SSM layer = conv sub-projections + 1 SSM temporal
111:             # (Mamba2/GDN: 3+1=4; Mamba1: 1+1=2).
112:             ssm_regions_per_layer = len(self._conv_decomp.local_conv_offsets) + 1
113:             num_ssm_regions = len(self.block_len_per_layer) * ssm_regions_per_layer
114: 
115:         num_blocks = dst_num_blocks
116:         if block_size_ratio is not None:
117:             num_blocks = int(num_blocks * block_size_ratio)
118:         num_fa_descs = self.num_regions * num_blocks
119: 
120:         # All-attention fast path: single vectorized broadcast.
121:         if num_ssm_regions == 0:
122:             # NOTE (NickLucche) With HMA, every kv group has the same number of layers
123:             # and layers from different groups share the same kv tensor.
124:             # eg block_ids=[[1, 2], [3]]->blocks [1, 2] need to be
125:             # read across all regions, same for [3], but group0-group1 blocks will
126:             # always differ (different areas). Therefore we can just flatten the
127:             # block_ids and compute the descs ids for all groups at once.
128:             block_arr = np.concatenate(block_ids)[None, :]
129:             region_ids = np.arange(self.num_regions)[:, None]
130:             return (region_ids * num_blocks + block_arr).flatten()
```
```text
539:         self._physical_blocks_per_logical_kv_block = 1
540:         self._sync_block_size_with_kernel()
541: 
542:         # Unwrap UniformTypeKVCacheSpecs to get the representative spec type
543:         self._group_spec_types = tuple(
544:             get_representative_spec_type(g.kv_cache_spec)
545:             for g in self.kv_cache_config.kv_cache_groups
```
```text
564:     def _sync_block_size_with_kernel(self) -> None:
565:         backends = get_current_attn_backends(self.vllm_config)
566:         kernel_block_size = select_common_block_size(self.block_size, backends)
567:         # Number of blocks not accounting for kernel block mismatches
568:         self._logical_num_blocks = self.num_blocks
569:         if self.block_size != kernel_block_size:
570:             logger.info_once(
571:                 "User-specified logical block size (%s) does not match"
572:                 " physical kernel block size (%s). Using the latter.",
573:                 self.block_size,
574:                 kernel_block_size,
575:             )
576:             assert self.block_size > kernel_block_size
577:             self._physical_blocks_per_logical_kv_block = (
578:                 self.block_size // kernel_block_size
579:             )
580:             self.block_size = kernel_block_size
581:             self.num_blocks *= self._physical_blocks_per_logical_kv_block
582: 
```
```text
960:     def _register_packed_kv_cache(
961:         self,
962:         storage: torch.UntypedStorage,
963:     ) -> None:
964:         """Register a packed KV cache as a single NIXL region.
965: 
966:         The packed allocation interleaves all layers per block, so each
967:         block_stride-byte chunk is one logical block.  We register 1
968:         NIXL region and create 1 descriptor per block.
969:         """
970:         self.transfer_topo = TransferTopology(
971:             tp_rank=self.tp_rank,
972:             tp_size=self.world_size,
973:             block_size=self.block_size,
974:             engine_id=self.engine_id,
975:             is_mla=self.use_mla,
976:             total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
977:             attn_backends=self.attn_backends,
978:             tensor_shape=None,
979:             is_mamba=self._has_mamba,
980:         )
981:         self.compat_hash = compute_nixl_compatibility_hash(
982:             self.vllm_config,
983:             self.backend_name,
984:             self.transfer_topo.cross_layers_blocks,
985:             transfer_mode=self._TRANSFER_MODE,
986:         )
987: 
988:         total_size = storage.nbytes()
989:         block_stride = total_size // self.num_blocks
990:         base_addr = storage.data_ptr()
991:         device_id = storage.device.index
992:         assert device_id is not None
993: 
994:         logger.info(
995:             "Registering packed KV cache: total_size=%s, block_stride=%s, "
996:             "num_blocks=%s, num_regions=1",
997:             total_size,
998:             block_stride,
999:             self.num_blocks,
1000:         )
1001: 
1002:         self.device_id = device_id
1003:         caches_data = [(base_addr, total_size, self.device_id, "")]
1004: 
1005:         self.block_len_per_layer = [block_stride]
1006:         self.num_regions = 1
1007:         self.num_descs = self.num_blocks
1008:         self.kv_caches_base_addr[self.engine_id][self.tp_rank] = [base_addr]
1009: 
1010:         descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
1011:         self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
1012:         self._registered_descs.append(descs)
1013: 
1014:         self.dst_num_blocks[self.engine_id] = self.num_blocks
1015: 
1016:         self.src_xfer_handles_by_block_size[self.block_size], (self.src_blocks_data) = (
1017:             self.register_local_xfer_handler(self.block_size)
1018:         )
```
```text
1395:         self,
1396:         base_addresses: list[int],
1397:         block_size_ratio: int,
1398:     ) -> np.ndarray:
1399:         """Build local FA descriptors for all layers as an Nx3 uint64 array."""
1400:         assert self.transfer_topo is not None
1401:         assert base_addresses, "Local KV cache base addresses must not be empty."
1402:         num_blocks = self.num_blocks * block_size_ratio
1403:         device_id = self.device_id
1404:         block_arange = np.arange(num_blocks, dtype=np.uint64)
1405:         parts: list[np.ndarray] = []
1406:         for i, base_addr in enumerate(base_addresses):
1407:             # K/V are packed into the content dim, so the whole block transfers
1408:             # as one unit: desc length equals the block stride.
1409:             block_len = self.block_len_per_layer[i] // block_size_ratio
1410:             addrs = base_addr + block_arange * block_len
1411:             parts.append(self._stack_descs(addrs, block_len, device_id))
1412:         return np.concatenate(parts)
1413: 
1414:     def _build_fa_remote(
1415:         self,
1416:         plan: TPMapping,
1417:         nixl_agent_meta: NixlAgentMetadata,
1418:         block_size_ratio: int,
1419:     ) -> np.ndarray:
1420:         """Build remote FA descriptors for all layers as an Nx3 uint64 array."""
1421:         assert self.transfer_topo is not None
1422:         assert nixl_agent_meta.kv_caches_base_addr, (
1423:             "Remote KV cache base addresses must not be empty."
1424:         )
1425:         fa_group_idx = next(
1426:             i for i, t in enumerate(self._group_spec_types) if _is_attention_spec(t)
1427:         )
1428:         # SPLIT regions read their head slice from this many remote ranks at a
1429:         # per-rank offset; REPLICATE regions read the whole block once.
1430:         split_reads = len(plan.source_ranks_per_group[fa_group_idx])
1431:         num_blocks = nixl_agent_meta.num_blocks
1432:         device_id = nixl_agent_meta.device_id
1433:         block_arange = np.arange(num_blocks, dtype=np.uint64)
1434:         parts: list[np.ndarray] = []
1435:         for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
1436:             replicated = self._is_region_replicated(i)
1437:             # Read our whole local region size from remote..
1438:             local_block_len = self.block_len_per_layer[i]
1439:             remote_kv_block_len = local_block_len // block_size_ratio
1440:             if block_size_ratio > 1:
1441:                 # ..using remote kv_block_len as transfer unit
1442:                 local_block_len = remote_kv_block_len
1443: 
1444:             # REPLICATE reads the whole block once at offset 0; SPLIT gathers
1445:             # its head slice from `split_reads` remote ranks at a per-rank offset.
1446:             num_reads = 1 if replicated else split_reads
1447:             rank_offset = (
1448:                 0 if replicated else plan.rank_offset_factor * remote_kv_block_len
1449:             )
1450:             local_block_len = local_block_len // num_reads
1451: 
1452:             page_size = nixl_agent_meta.block_lens[i]
1453:             addrs = base_addr + rank_offset + block_arange * page_size
1454:             parts.append(self._stack_descs(addrs, local_block_len, device_id))
1455:         return np.concatenate(parts)
```
```text
2381:     def _logical_to_kernel_block_ids(self, block_ids: BlockIds, ratio: int) -> BlockIds:
2382:         """
2383:         Convert block ids to kernel physical block ids.
2384:         This is required when the logical block size (the one set by the user)
2385:         does not match the one required by the attn backend.
2386:         `ratio` is the number of physical blocks per logical block.
2387:         We always receive logical blocks from the engine, so we expand them here eg:
2388:         logical block ids: [(SW-clipped) [1], (FA) [2, 3]], ratio=2
2389:         physical block ids: [(SW-clipped) [2, 3], (FA) [4, 5, 6, 7]]
2390:         """
2391:         if ratio == 1:
2392:             # Noop when physical and logical block sizes are the same
2393:             return block_ids
2394:         block_arange = np.arange(0, ratio).reshape(1, -1)
2395:         # Mamba blocks have no logical<>physical discrepancy (block-size=1)
2396:         group_specs = self.kv_cache_config.kv_cache_groups
2397:         physical_block_ids = []
2398:         for i, group in enumerate(block_ids):
2399:             spec = group_specs[i].kv_cache_spec
2400:             if isinstance(spec, MambaSpec):
2401:                 physical_block_ids.append(group)
2402:             else:
2403:                 physical_block_ids.append(
2404:                     BlockTable.map_to_kernel_blocks(
2405:                         np.array(group),
2406:                         ratio,
2407:                         block_arange,
2408:                     ).tolist()
2409:                 )
2410:         return physical_block_ids
```
```text
2437:         # Partial prefix cache hit: just transfer uncomputed blocks.
2438:         # Skip mamba groups — their blocks represent full state (conv+ssm),
2439:         # not per-token data, so trimming would corrupt the transfer.
2440:         prefill_block_ids = list(prefill_block_ids)
2441:         if not self._has_mamba:
2442:             for i, prefill_group in enumerate(prefill_block_ids):
2443:                 num_decode_blocks = len(decode_block_ids[i])
2444:                 assert num_decode_blocks <= len(prefill_group)
2445:                 if num_decode_blocks < len(prefill_group):
2446:                     prefill_block_ids[i] = prefill_group[-num_decode_blocks:]
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py
```text
143:         plan = self.tp_mappings[engine_id]
144:         remote_info = self.transfer_topo.get_engine_info(engine_id)
145:         tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)
146: 
147:         meta.remote.block_ids = self._logical_to_kernel_block_ids(
148:             meta.remote.block_ids,
149:             remote_info.remote_physical_blocks_per_logical,
150:         )
151:         remote_block_ids = meta.remote.block_ids
152:         local_block_ids = meta.local_physical_block_ids
153:         num_groups = len(local_block_ids)
154:         read_specs = [
155:             ReadSpec(
156:                 remote_rank=rank,
157:                 local_block_ids=[
158:                     list(local_block_ids[g])
159:                     if rank in plan.source_ranks_per_group[g]
160:                     else []
161:                     for g in range(num_groups)
162:                 ],
163:                 remote_block_ids=[
164:                     list(remote_block_ids[g])
165:                     if rank in plan.source_ranks_per_group[g]
166:                     else []
167:                     for g in range(num_groups)
168:                 ],
169:             )
170:             for rank in plan.all_source_ranks
171:         ]
```
```text
240:         local_block_ids = read_spec.local_block_ids
241:         remote_block_ids = read_spec.remote_block_ids
242: 
243:         remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
244:         block_size_ratio = self.transfer_topo.block_size_ratio(
245:             remote_info.remote_block_size
246:         )
247:         if block_size_ratio > 1:
248:             local_block_ids, remote_block_ids = (
249:                 self._map_block_ids_for_block_size_ratio(
250:                     local_block_ids, remote_block_ids, block_size_ratio
251:                 )
252:             )
```
```text
287:         assert (
288:             len(remote_block_ids)
289:             == len(local_block_ids)
290:             == len(self.kv_cache_config.kv_cache_groups)
291:         )
292:         local_block_ids, remote_block_ids = self._apply_prefix_caching(
293:             decode_block_ids=local_block_ids,
294:             prefill_block_ids=remote_block_ids,
295:             decode_physical_per_logical=self._physical_blocks_per_logical_kv_block,
296:             prefill_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
297:         )
298: 
299:         # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
300:         # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
301:         # workers will issue xfers to parts of the P worker remote kv caches.
302: 
303:         # Get descs ids.
304:         remote_block_descs_ids = self._compute_desc_ids(
305:             block_ids=remote_block_ids,
306:             dst_num_blocks=self.dst_num_blocks[dst_engine_id],
307:             block_size_ratio=None,
308:             physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
309:         )
310:         local_block_descs_ids = self._compute_desc_ids(
311:             block_ids=local_block_ids,
312:             dst_num_blocks=self.dst_num_blocks[self.engine_id],
313:             block_size_ratio=block_size_ratio,
314:             physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
315:         )
316: 
317:         assert len(local_block_descs_ids) == len(remote_block_descs_ids)
318: 
319:         # Prepare transfer with Nixl.
320:         handle = None
321:         try:
322:             handle = self.nixl_wrapper.make_prepped_xfer(
323:                 "READ",
324:                 local_xfer_side_handle,
325:                 local_block_descs_ids,
326:                 remote_xfer_side_handle,
327:                 remote_block_descs_ids,
328:                 notif_msg=notif_id,
329:             )
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py
```text
75:     tp_rank = transfer_topology.tp_rank
76:     tp_size = transfer_topology.tp_size
77:     total_num_kv_heads = transfer_topology.total_num_kv_heads
78:     # --- Attention source ranks ---
79:     if transfer_topology.is_mla or tp_size >= remote_tp_size:
80:         # D (local TP) > P (remote TP): multiple local ranks read different chunks from
81:         # *one* remote rank, corresponding to different kv heads.
82:         # For MLA, we only need one remote since cache is duplicated. When P TP=k*TP k,
83:         # this will spread mla ranks to read from remote k*tp_rank.
84:         attn_ranks = [tp_rank * remote_tp_size // tp_size]
85:     else:
86:         # P (remote TP) > D (local TP): one local rank
87:         # reads from multiple remote ranks.
88:         # GQA dedup: when K < remote_tp_size, several remote ranks
89:         # hold the same KV head.  np.unique keeps only the first
90:         # rank per unique head so we don't issue redundant reads.
91:         abs_tp = remote_tp_size // tp_size
92:         start = tp_rank * abs_tp
93:         heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
94:         _, unique_idx = np.unique(heads, return_index=True)
95:         attn_ranks = (start + np.sort(unique_idx)).tolist()
96: 
97:     # --- SSM source ranks ---
98:     has_ssm = any(_is_ssm_spec(t) for t in group_spec_types)
99:     if has_ssm:
100:         if tp_size < remote_tp_size:
101:             abs_tp = remote_tp_size // tp_size
102:             ssm_ranks = list(range(tp_rank * abs_tp, (tp_rank + 1) * abs_tp))
103:         else:
104:             ssm_ranks = list(attn_ranks)
105:     else:
106:         ssm_ranks = []
107: 
108:     all_ranks = sorted(set(attn_ranks) | set(ssm_ranks))
109: 
110:     # --- Per-group ordered source ranks ---
111:     source_ranks_per_group = tuple(
112:         tuple(ssm_ranks) if _is_ssm_spec(t) else tuple(attn_ranks)
113:         for t in group_spec_types
114:     )
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/v1/core/kv_cache_manager.py
```text
99:     def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
100:         """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
101:         # Skip padding blocks.
102:         return [
103:             [
104:                 block.block_id
105:                 for block in group
106:                 if block.block_hash is None and not block.is_null
107:             ]
108:             for group in self.blocks
109:         ]
```
```text
449:             )
450: 
451:         if new_computed_blocks is not None:
452:             new_computed_block_list = new_computed_blocks.blocks
453:         else:
454:             new_computed_block_list = self.empty_kv_cache_blocks.blocks
455: 
456:         # The number of computed tokens is the number of computed tokens plus
457:         # the new prefix caching hits
458:         num_local_computed_tokens = (
459:             request.num_computed_tokens + num_new_computed_tokens
460:         )
461:         total_computed_tokens = min(
462:             num_local_computed_tokens + num_external_computed_tokens,
463:             self.max_model_len,
464:         )
```
```text
493:         num_tokens_main_model = total_computed_tokens + num_new_tokens
494:         num_tokens_need_slot = min(
495:             num_tokens_main_model + num_lookahead_tokens, self.max_model_len
496:         )
497: 
498:         # Free the blocks that are skipped during the attention computation
499:         # (e.g., tokens outside the sliding window).
500:         # We can do this even if we cannot schedule this request due to
501:         # insufficient free blocks.
502:         # Should call this function before allocating new blocks to reduce
503:         # the number of evicted blocks.
504:         # Free on the processed-token basis: in-flight steps' attention windows
505:         # still read blocks below the optimistic boundary, and rejected spec
506:         # tokens can roll it back.
507:         self.coordinator.remove_skipped_blocks(
508:             request.request_id,
509:             max(0, total_computed_tokens - request.num_in_flight_tokens),
510:             num_prompt_tokens=request.num_prompt_tokens,
511:         )
512: 
513:         num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
514:             request_id=request.request_id,
515:             num_tokens=num_tokens_need_slot,
516:             new_computed_blocks=new_computed_block_list,
517:             num_encoder_tokens=num_encoder_tokens,
518:             total_computed_tokens=num_local_computed_tokens
519:             + num_external_computed_tokens,
520:             num_local_computed_tokens=num_local_computed_tokens,
521:             num_tokens_main_model=num_tokens_main_model,
522:         )
523: 
524:         # Keep `reserved_blocks` free for other in-flight sequences, and an
525:         # additional watermark of headroom for waiting/preempted admissions.
526:         available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
527:         required_blocks = num_blocks_to_allocate + watermark_blocks
528:         if required_blocks > available_blocks:
529:             # Cannot allocate new blocks
530:             return None
531: 
532:         if (
533:             new_computed_block_list is not self.empty_kv_cache_blocks.blocks
534:             or num_external_computed_tokens > 0
535:         ):
536:             # Append the new computed blocks to the request blocks until now to
537:             # avoid the case where the new blocks cannot be allocated.
538:             self.coordinator.allocate_new_computed_blocks(
539:                 request_id=request.request_id,
540:                 new_computed_blocks=new_computed_block_list,
541:                 num_local_computed_tokens=num_local_computed_tokens,
542:                 num_external_computed_tokens=num_external_computed_tokens,
543:             )
544: 
545:         new_blocks = self.coordinator.allocate_new_blocks(
546:             request.request_id,
547:             num_tokens_need_slot,
548:             num_tokens_main_model,
549:             num_encoder_tokens,
550:         )
551: 
```
```text
783:         """Return a lookup-result view truncated at an aligned token endpoint.
784: 
785:         An external hit can supply the final Mamba state even when the local
786:         Mamba group ends before this endpoint. Other groups must cover it.
787:         Pure slicing: refcounts are untouched and ``blocks`` is not mutated.
788:         """
789:         truncated: list[list[KVCacheBlock]] = []
790:         for group_blocks, manager, group in zip(
791:             blocks.blocks,
792:             self.coordinator.single_type_managers,
793:             self.kv_cache_config.kv_cache_groups,
794:             strict=True,
795:         ):
796:             assert num_computed_tokens % manager.block_size == 0
797:             num_blocks = num_computed_tokens // manager.block_size
798:             if isinstance(group.kv_cache_spec, MambaSpec):
799:                 num_blocks = min(num_blocks, len(group_blocks))
800:             else:
801:                 assert num_blocks <= len(group_blocks)
802:             truncated.append(list(group_blocks[:num_blocks]))
803:         return self.create_kv_cache_blocks(tuple(truncated))
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/v1/core/single_type_kv_cache_manager.py
```text
250:         # The coordinator only calls this for first-time allocations (running
251:         # requests are short-circuited there), so the request has no blocks yet.
252:         req_blocks = self.req_to_blocks[request_id]
253:         assert len(req_blocks) == 0
254:         num_total_computed_tokens = (
255:             num_local_computed_tokens + num_external_computed_tokens
256:         )
257:         num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
258:         num_skipped_blocks = num_skipped_tokens // self.block_size
259:         if num_skipped_blocks > 0:
260:             # It is possible that all new computed blocks are skipped when
261:             # num_skipped_blocks > len(new_computed_blocks).
262:             new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
263: 
264:         # Touch the computed blocks to make sure they won't be evicted.
265:         if self.enable_caching:
266:             self.block_pool.touch(new_computed_blocks)
267:         else:
268:             assert not any(new_computed_blocks), (
269:                 "Computed blocks should be empty when prefix caching is disabled"
270:             )
271: 
272:         # Skip blocks are padded with null blocks.
273:         req_blocks.extend([self._null_block] * num_skipped_blocks)
274:         # Add the remaining computed blocks.
275:         req_blocks.extend(new_computed_blocks)
276:         # All cached hits (including skipped nulls) are already cached; mark
277:         # them so cache_blocks() will not try to re-cache blocks that already
278:         # have a block_hash set.
279:         self.num_cached_block[request_id] = len(req_blocks)
```
```text
306:         num_total_computed_tokens = (
307:             num_local_computed_tokens + num_external_computed_tokens
308:         )
309:         num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
310:         if num_skipped_tokens > 0:
311:             # Some external computed tokens may be skipped too.
312:             num_external_computed_tokens = min(
313:                 num_total_computed_tokens - num_skipped_tokens,
314:                 num_external_computed_tokens,
315:             )
316:         if num_external_computed_tokens <= 0:
317:             return
318: 
319:         req_blocks = self.req_to_blocks[request_id]
320:         allocated_blocks = self.block_pool.get_new_blocks(
321:             cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
322:         )
323:         req_blocks.extend(allocated_blocks)
324:         if self._record_new_block_ids:
325:             self.new_block_ids.extend(b.block_id for b in allocated_blocks)
```
```text
597:     def _remove_blocks_in_range(
598:         self,
599:         request_id: str,
600:         first_block: int,
601:         last_block: int,
602:     ) -> None:
603:         """Free blocks in ``[first_block, last_block)`` and replace with null_block.
604: 
605:         Iterates backward so newly-evictable tail blocks are reached even after
606:         earlier blocks in the range were nulled in a prior call.
607:         """
608:         if request_id not in self.req_to_blocks:
609:             return
610:         if first_block >= last_block:
611:             return
612:         blocks = self.req_to_blocks[request_id]
613:         last_block = min(last_block, len(blocks))
614: 
615:         freed: list[KVCacheBlock] = []
616:         for i in range(last_block - 1, first_block - 1, -1):
617:             if blocks[i] == self._null_block:
618:                 break
619:             freed.append(blocks[i])
620:             blocks[i] = self._null_block
621:         if freed:
622:             self.block_pool.free_blocks(freed)
623: 
624:     def remove_skipped_blocks(
625:         self,
626:         request_id: str,
627:         processed_computed_tokens: int,
628:         num_prompt_tokens: int | None = None,
629:     ) -> None:
630:         """
631:         Remove and free the blocks that are no longer needed for attention computation.
632:         The removed blocks should be replaced by null_block.
633: 
634:         This function depends on `get_num_skipped_tokens`, which need to be implemented
635:         differently for each attention type.
636: 
637:         Args:
638:             request_id: The request ID.
639:             processed_computed_tokens: Computed-token prefix length covering
640:                 fully processed and committed tokens only (safe to free).
641:             num_prompt_tokens: Optional prompt length for attention types (e.g.
642:                 R-SWA) that evict a middle gap rather than a head prefix. Ignored
643:                 by the default implementation.
644:         """
645:         del num_prompt_tokens
646:         # Remove the blocks that will be skipped during attention computation.
647:         num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
648:         if num_skipped_tokens <= 0:
649:             # This indicates that ALL tokens are inside attention window.
650:             # Thus we do not need to free any blocks outside attention window.
651:             # A typical case is full attention that we never free any token
652:             # before the request is finished.
653:             return
654:         blocks = self.req_to_blocks[request_id]
655:         num_skipped_blocks = num_skipped_tokens // self.block_size
656:         # `num_skipped_tokens` may include tokens that haven't been allocated yet
657:         # (e.g., when the attention window moves into the external computed tokens
658:         # range), so we must cap to the number of blocks that currently exist for
659:         # this request.
660:         num_skipped_blocks = min(num_skipped_blocks, len(blocks))
661:         self._remove_blocks_in_range(request_id, 0, num_skipped_blocks)
```
```text
1095:         return max(
1096:             0,
1097:             num_computed_tokens - self.sliding_window + 1 - self.extra_retained_tokens,
1098:         )
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/v1/core/sched/scheduler.py
```text
918:                 if load_kv_async:
919:                     # KVTransfer: loading remote KV, do not allocate for new work.
920:                     assert num_external_computed_tokens > 0
921:                     num_new_tokens = 0
```
```text
1008:                 limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
1009:                 effective_lookahead_tokens = (
1010:                     0 if limit_lookahead_tokens else self.num_lookahead_tokens
1011:                 )
1012: 
1013:                 # Determine if we need to allocate cross-attention blocks.
1014:                 num_encoder_tokens = 0
1015:                 if (
1016:                     self.is_encoder_decoder
1017:                     and request.has_encoder_inputs
1018:                     and encoder_inputs_to_schedule
1019:                 ):
1020:                     num_encoder_tokens = sum(
1021:                         request.get_num_encoder_embeds(i)
1022:                         for i in encoder_inputs_to_schedule
1023:                     )
1024: 
1025:                 reserved_blocks = 0
1026:                 if load_kv_async:
1027:                     # An async load holds its blocks for the whole transfer with
1028:                     # no forward progress and isn't preemptible here. Admit it
1029:                     # only if it fits in (free - other in-flight reservations), to
1030:                     # avoid deadlock and predictable preemptions.
1031:                     reserved_blocks = self._inflight_prefill_reserved_blocks()
1032: 
1033:                 new_blocks = self.kv_cache_manager.allocate_slots(
1034:                     request,
1035:                     num_new_tokens,
1036:                     num_new_computed_tokens=num_new_local_computed_tokens,
1037:                     new_computed_blocks=new_computed_blocks,
1038:                     num_lookahead_tokens=effective_lookahead_tokens,
1039:                     num_external_computed_tokens=num_external_computed_tokens,
1040:                     delay_cache_blocks=load_kv_async,
1041:                     num_encoder_tokens=num_encoder_tokens,
1042:                     full_sequence_must_fit=self.scheduler_reserve_full_isl,
1043:                     reserved_blocks=reserved_blocks,
```
```text
1050:                     # NOTE: we need to untouch the request from the encode cache
1051:                     # manager
1052:                     if request.has_encoder_inputs:
1053:                         self.encoder_cache_manager.free(request)
1054:                     break
1055: 
1056:                 # KVTransfer: the connector uses this info to determine
1057:                 # if a load is needed. Note that
1058:                 # This information is used to determine if a load is
1059:                 # needed for this request.
1060:                 if self.connector is not None:
1061:                     self.connector.update_state_after_alloc(
1062:                         request,
1063:                         self.kv_cache_manager.get_blocks(request_id),
1064:                         num_external_computed_tokens,
1065:                     )
1066:                     if (
1067:                         self.connector_prefix_cache_stats is not None
1068:                         and connector_prefix_cache_queries != 0
1069:                     ):
1070:                         self.connector_prefix_cache_stats.record(
1071:                             num_tokens=connector_prefix_cache_queries,
1072:                             num_hits=connector_prefix_cache_hits,
1073:                             preempted=request.num_preemptions > 0,
1074:                         )
1075: 
1076:                 # Record at admission so unscheduled lookups are not counted.
1077:                 if did_prefix_cache_lookup:
1078:                     self.kv_cache_manager.record_prefix_cache_stats(
1079:                         request, num_new_local_computed_tokens
1080:                     )
```
```text
2677:         # Free any out-of-window prefix blocks before we hand the block table to
2678:         # the connector, on the processed-token basis (see `allocate_slots`).
2679:         self.kv_cache_manager.remove_skipped_blocks(
2680:             request_id=request.request_id,
2681:             processed_computed_tokens=max(
2682:                 0, request.num_computed_tokens - request.num_in_flight_tokens
2683:             ),
2684:             num_prompt_tokens=request.num_prompt_tokens,
2685:         )
2686: 
2687:         block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
2688:             request_id=request.request_id,
2689:             num_computed_tokens=request.num_computed_tokens,
2690:         )
2691: 
2692:         if not isinstance(self.connector, SupportsHMA):
2693:             # NOTE(Kuntai): We should deprecate this code path after we enforce
2694:             # all connectors to support HMA.
2695:             # Hybrid memory allocator should be already turned off for this
2696:             # code path, but let's double-check here.
2697:             assert len(self.kv_cache_config.kv_cache_groups) == 1
2698:             return self.connector.request_finished(request, block_ids[0])
2699: 
2700:         return self.connector.request_finished_all_groups(request, block_ids)
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/v1/kv_cache_interface.py
```text
537: class SlidingWindowSpec(AttentionSpec):
538:     sliding_window: int
539:     # The trailing edge of the window is extended by ``extra_retained_tokens``
540:     # so that those extra trailing tokens' blocks are retained (but not
541:     # attended). This is needed for multi-module spec decoding which can
542:     # re-prefill the last num_spec_prefill_tokens - 1 tokens from the end
543:     # of the sequence, and thus needs to delay freeing/caching of blocks.
544:     extra_retained_tokens: int = 0
545: 
```
```text
595: class SlidingWindowMLASpec(SlidingWindowSpec):
596:     """Sliding window attention with MLA cache format."""
597: 
598:     cache_dtype_str: str | None = None
599:     # DeepseekV4-only: see MLAAttentionSpec.model_version.
600:     alignment: int | None = None  # Default to None for no padding.
601:     compress_ratio: int = 1
602:     model_version: str | None = None
603:     # MLA stores a single latent vector per state; there is no separate V.
604:     head_size_v: int = 0
605: 
606:     def __post_init__(self):
607:         assert self.model_version in (None, "deepseek_v4"), (
608:             f"Unsupported model version: {self.model_version}"
609:         )
610:         super().__post_init__()
611:         _apply_alignment_padding(self)
612: 
613:     @property
614:     def storage_block_size(self) -> int:
615:         return self.block_size // self.compress_ratio
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/models/deepseek_v4/compressor.py
```text
61: class CompressorBackend(AttentionBackend):
62:     def __init__(self):
63:         super().__init__()
64: 
65:     @staticmethod
66:     def get_name() -> str:
67:         return "CompressorBackend"
68: 
69:     @staticmethod
70:     def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
71:         return [MultipleOf(1)]
```
```text
173:         assert self.dtype == torch.float32
174:         assert compress_ratio in [4, 128]
175:         coff = 1 + (compress_ratio == 4)
176:         self.sliding_window = coff * compress_ratio
177:         # Block size is constrained by tensor sharing between compressor states
178:         # and KV blocks. Since compressor states share the same physical tensor
179:         # as KV blocks, they must use the same page size.
180:         # The KV block shape [256//4, head_dim] = [64, 584] determines:
181:         # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
182:         # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
183:         # TODO(yifan): make block size automatically determined and configurable.
184:         if compress_ratio == 4:
185:             self.block_size = 4
186:         elif compress_ratio == 128:
187:             self.block_size = 8
188:         else:
189:             raise ValueError(f"Invalid compress ratio: {compress_ratio}")
190: 
191:     def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
192:         # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
193:         # full-cache rows share state pages with contiguous KV pages, so padding
194:         # would break page matching.
195:         uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
196:         return SlidingWindowMLASpec(  # only has one vector instead of K + V
197:             block_size=self.block_size,
198:             num_kv_heads=1,
199:             head_size=self.state_dim,
200:             dtype=self.dtype,
201:             sliding_window=self.sliding_window,
202:             alignment=576 if uses_fp8_ds_mla_layout else 512,
203:         )
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/models/deepseek_v4/attention.py
```text
716:     def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
717:         if (
718:             self.compress_ratio <= 1
719:         ):  # SWA part. Allocated separately as DeepseekV4SWACache.
720:             return None
721:         # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
722:         # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
723:         # pages.
724:         uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
725:         return MLAAttentionSpec(
726:             block_size=vllm_config.cache_config.block_size,
727:             num_kv_heads=1,
728:             head_size=self.head_dim,
729:             dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
730:             compress_ratio=self.compress_ratio,
731:             cache_dtype_str=self.kv_cache_dtype,
732:             alignment=576 if uses_fp8_ds_mla_layout else 512,
733:             model_version="deepseek_v4",
734:             kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
735:             # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token;
736:             # head_size stays semantic (512).
737:             state_content_bytes=584 if uses_fp8_ds_mla_layout else None,
738:         )
```
```text
762:     def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
763:         # head_dim already carries the fp8 scale padding
764:         # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
765:         uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
766:         return MLAAttentionSpec(
767:             block_size=self.cache_config.block_size,
768:             num_kv_heads=1,
769:             head_size=self.head_dim,
770:             dtype=self.dtype,
771:             compress_ratio=self.compress_ratio,
772:             # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
773:             alignment=576 if uses_fp8_ds_mla_layout else 512,
774:         )
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/models/deepseek_v4/sparse_mla.py
```text
51: 
52:     @staticmethod
53:     def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
54:         return [256]
55: 
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/v1/attention/backends/mla/sparse_swa.py
```text
82:         # Block size is constrained by tensor sharing between SWA and C4A KV blocks.
83:         # Since both block types share the same physical tensor, they must use the
84:         # same page size. The C4A KV block shape [256//4, head_dim] = [64, head_dim]
85:         # determines the SWA block size of 64 tokens per block.
86:         # TODO(yifan): make SWA block size automatically determined and configurable.
87:         self.block_size = 64
88:         # uint8: fp8_ds_mla UE8M0 paged layout. bfloat16 / float8_e4m3fn:
89:         # contiguous full-cache layout.
90:         assert self.dtype in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn)
91: 
92:     def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
93:         # fp8_ds_mla's UE8M0 paged layout needs 576B alignment; contiguous
94:         # bf16/fp8 cache uses the natural element-size page.
95:         uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
96:         return SlidingWindowMLASpec(
97:             block_size=self.block_size,
98:             num_kv_heads=1,
99:             head_size=self.head_dim,
100:             dtype=self.dtype,
101:             sliding_window=self.window_size,
102:             cache_dtype_str=self.cache_config.cache_dtype,
103:             # DeepseekV4 fp8_ds_mla: 584B per token (448B NoPE + 128B RoPE + 8B scales)
104:             state_content_bytes=584 if uses_fp8_ds_mla_layout else None,
105:             # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
106:             alignment=576 if uses_fp8_ds_mla_layout else 512,
107:             model_version="deepseek_v4",
108:             kv_quant_mode=get_kv_quant_mode(self.cache_config.cache_dtype),
109:         )
110: 
```
```text
117: class DeepseekSparseSWABackend(AttentionBackend):
118:     @staticmethod
119:     def get_name() -> str:
120:         return "DEEPSEEK_SPARSE_SWA"
121: 
122:     @staticmethod
123:     def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
124:         return [MultipleOf(64)]
125: 
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/v1/attention/backends/mla/indexer.py
```text
202: class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):
203:     @staticmethod
204:     def get_name() -> str:
205:         return "DEEPSEEK_V4_INDEXER"
206: 
207:     @staticmethod
208:     def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
209:         return [256]
210: 
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/v1/worker/utils.py
```text
301: def select_common_block_size(
302:     kv_manager_block_size: int,
303:     backends: list[type[AttentionBackend]],
304: ) -> int:
305:     """
306:     Select a block size that is supported by all backends and is a factor of
307:     kv_manager_block_size.
308: 
309:     If kv_manager_block_size is supported by all backends, return it directly.
310:     Otherwise, return the max supported size.
```
```text
323:     def block_size_is_supported(
324:         backends: list[type[AttentionBackend]], block_size: int
325:     ) -> bool:
326:         """Check if the block size is supported by all backends."""
327:         for backend in backends:
328:             is_supported = False
329:             for supported_size in backend.get_supported_kernel_block_sizes():
330:                 if isinstance(supported_size, int):
331:                     if block_size == supported_size:
332:                         is_supported = True
333:                 elif isinstance(supported_size, MultipleOf):
334:                     if block_size % supported_size.base == 0:
335:                         is_supported = True
336:                 else:
337:                     raise ValueError(f"Unknown supported size: {supported_size}")
338:             if not is_supported:
339:                 return False
340:         return True
341: 
342:     # Case 1: if the block_size of kv cache manager is supported by all backends,
343:     # return it directly.
344:     if block_size_is_supported(backends, kv_manager_block_size):
345:         return kv_manager_block_size
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/config/vllm.py
```text
1717:         # - Explicit enable (--no-disable-kv-cache-manager): error if runtime
1718:         #   disables it
1719:         # - No preference: auto-disable for unsupported features or connector configs
1720:         # - Explicit disable (--disable-kv-cache-manager): always respect it
1721:         need_disable_hybrid_kv_cache_manager = False
1722:         # logger should only print warning message for hybrid models. As we
1723:         # can't know whether the model is hybrid or not now, so we don't log
1724:         # warning message here and will log it later.
1725:         if not current_platform.support_hybrid_kv_cache():
1726:             # Hybrid KV cache manager is not supported on non-GPU platforms.
1727:             need_disable_hybrid_kv_cache_manager = True
1728:         if (
1729:             self.model_config is not None
1730:             and self.model_config.attention_chunk_size is not None
1731:         ):
1732:             if (
1733:                 self.speculative_config is not None
1734:                 and self.speculative_config.use_eagle()
1735:             ):
1736:                 # Hybrid KV cache manager is not yet supported with chunked
1737:                 # local attention + eagle.
1738:                 need_disable_hybrid_kv_cache_manager = True
1739:             elif not envs.VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE:
1740:                 logger.warning(
1741:                     "There is a latency regression when using chunked local"
1742:                     " attention with the hybrid KV cache manager. Disabling"
1743:                     " it, by default. To enable it, set the environment "
1744:                     "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1."
1745:                 )
1746:                 # Hybrid KV cache manager is not yet supported with chunked
1747:                 # local attention.
1748:                 need_disable_hybrid_kv_cache_manager = True
1749: 
1750:         if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
1751:             # Auto-disable HMA only when the connector config does not support it.
1752:             if self.kv_transfer_config is not None:
1753:                 from vllm.distributed.kv_transfer.kv_connector.factory import (
1754:                     KVConnectorFactory,
1755:                 )
1756: 
1757:                 if not KVConnectorFactory.supports_hma_config(self.kv_transfer_config):
1758:                     need_disable_hybrid_kv_cache_manager = True
1759:                     logger.warning(
1760:                         "Turning off hybrid kv cache manager because "
1761:                         "`--kv-transfer-config` selects a KV connector that "
1762:                         "does not support it. Impact: hybrid SSM models "
1763:                         "(e.g. Jamba, Bamba) require HMA and will fail at "
1764:                         "startup without it; models with sliding window "
1765:                         "attention will run with reduced performance. "
1766:                         "To add HMA support to a KV connector, subclass "
1767:                         "`SupportsHMA` defined in kv_connector/v1/base.py "
1768:                         "(for MultiConnector, all child connectors must "
1769:                         "support HMA)."
1770:                     )
1771:             self.scheduler_config.disable_hybrid_kv_cache_manager = (
1772:                 need_disable_hybrid_kv_cache_manager
1773:             )
1774:         elif (
1775:             self.scheduler_config.disable_hybrid_kv_cache_manager is False
1776:             and need_disable_hybrid_kv_cache_manager
1777:         ):
1778:             raise ValueError(
1779:                 "Hybrid KV cache manager was explicitly enabled but is not "
1780:                 "supported in this configuration. Consider omitting the "
1781:                 "--no-disable-hybrid-kv-cache-manager flag to let vLLM decide"
1782:                 " automatically."
1783:             )
1784: 
1785:         if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
1786:             # Default to enable HMA if not explicitly disabled by user or logic above.
1787:             self.scheduler_config.disable_hybrid_kv_cache_manager = False
1788: 
1789:         if self.compilation_config.debug_dump_path:
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/config/model.py
```text
619:         if dict_overrides:
620:             self._apply_dict_overrides(hf_config, dict_overrides)
621:         self.hf_text_config = get_hf_text_config(self.hf_config)
622:         self.model_arch_config = self.get_model_arch_config()
623:         self.attention_chunk_size = getattr(
624:             self.hf_text_config, "attention_chunk_size", None
625:         )
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py
```text
79: class NixlBaseConnector(KVConnectorBase_V1, SupportsHMA):
80:     """Base connector with common logic shared by pull and push modes."""
81: 
82:     @property
83:     def supports_divergent_local_hybrid_hits(self) -> bool:
84:         return True
85: 
86:     @property
87:     def prefer_cross_layer_blocks(self) -> bool:
88:         if any(
89:             [
90:                 isinstance(group.kv_cache_spec, MambaSpec)
91:                 for group in self.kv_cache_config.kv_cache_groups
92:             ]
93:         ):
```

## /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/platforms/cuda.py
```text
660:     @classmethod
661:     def support_hybrid_kv_cache(cls) -> bool:
662:         return True
663: 
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py
```text
54:         source = source_groups[group.group_id]
55:         target = tuple(block for block in target_groups[group.group_id] if block is not None)
56:         if group.retention == "sliding_window":
57:             limit = ceil(group.window_tokens / group.logical_block_size) + 1
58:             source, target = source[-limit:], target[-limit:]
59:         selected = tuple(source[-len(target):])
60:         if any(block is None for block in selected):
61:             raise ValueError(f"NIXL group {group.group_id} selected a null source page")
62:         transferred[group.group_id] = selected
63:         payload_bytes += len(selected) * cache_capacity.get_group_size_bytes(group)
64:     return transferred, payload_bytes
65: 
66: 
67: def get_grouped_nixl_wire_bytes(cache_capacity, transferred):
68:     """Bytes one worker/lane reads for the selected pages of every group.
69: 
70:     Source: pinned NIXL base_worker._register_packed_kv_cache registers the
71:     packed cache as one region with one block_stride descriptor per physical
72:     block, so each selected page moves the full physical block.
73:     """
74:     return sum(map(len, transferred.values())) * cache_capacity.physical_block_size_bytes
```
```text
84: 
85:     def get_transfer_time(
86:         self,
87:         source_cluster_type: ClusterType,
88:         target_cluster_type: ClusterType,
89:         batch: "Batch",
90:         kv_cache_size_bytes: int,
91:     ) -> float:
92:         effective_size_bytes = kv_cache_size_bytes
93:         if self._config.enable_compression:
94:             effective_size_bytes = kv_cache_size_bytes / self._config.compression_ratio
95: 
96:         bandwidth_bytes_per_ms = (self._config.network_bandwidth_gbps * 1e9) / (8 * 1000)
97:         return self._config.network_latency_ms + (effective_size_bytes / bandwidth_bytes_per_ms)
```
```text
139:             families = {spec.family_id for spec in replica_config.model_config.get_layer_attention_specs()}
140:             if families != {cache_capacity.family_id}:
141:                 raise ValueError("V4 NIXL capacity does not match the model")
142:             transferred, payload_bytes = get_grouped_nixl_payload(
143:                 cache_capacity, request.num_prefill_tokens,
144:                 source_group_block_ids, target_group_block_ids,
145:             )
146:             return payload_bytes, self.get_transfer_time(
147:                 ClusterType.PREFILL, ClusterType.DECODE, None,
148:                 get_grouped_nixl_wire_bytes(cache_capacity, transferred),
149:             )
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/attention/memory.py
```text
211:     def get_group_size_bytes(self, group: AttentionKVCacheGroup) -> int:
212:         components = {
213:             (component.component_id, owner): component
214:             for component in self.components for owner in component.owner_layer_ids
215:         }
216:         return sum(
217:             components[identity].page_size_bytes for identity in group.component_owners
218:         )
219: 
220:     @property
221:     def physical_block_size_bytes(self) -> int:
222:         if self.groups:
223:             return max(self.get_group_size_bytes(group) for group in self.groups)
224:         return self.resident_bytes_per_common_block
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/events/kv_cache_transfer_start_event.py
```text
165:                 (
166:                     self._target_replica_id,
167:                     self._target_replica_local_id,
168:                 ) = target_cluster_scheduler.select_kv_transfer_target(
169:                     self.time, request
170:                 )
171:             target_replica_scheduler = target_cluster_scheduler.get_replica_scheduler(
172:                 self._target_replica_id,
173:                 self._target_replica_local_id,
174:             )
175:             has_older_pending = (
176:                 not self._has_target_reservation()
177:                 and target_replica_scheduler.has_pending_kv_transfer_starts()
178:             )
179:             if has_older_pending or not self.try_reserve_target(target_replica_scheduler):
180:                 target_replica_scheduler.defer_kv_transfer_start(self)
181:                 from frontier.events.replica_schedule_event import ReplicaScheduleEvent
182: 
183:                 return [
184:                     ReplicaScheduleEvent(
185:                         self.time,
186:                         self._target_replica_id,
```
```text
192:             try:
193:                 (
194:                     self._kv_cache_size_bytes,
195:                     self._transfer_time_ms,
196:                 ) = predictor.get_structural_nixl_transfer_info_for_request(
197:                     request=request,
198:                     replica_config=replica_config,
199:                     cache_capacity=source_capacity,
200:                     source_prompt_common_blocks=self._source_prompt_common_blocks,
201:                     target_reserved_common_blocks=(
202:                         self._target_reserved_common_blocks
203:                     ),
204:                     **(dict(
205:                         source_group_block_ids=self._source_group_block_ids,
206:                         target_group_block_ids=self._target_group_block_ids,
207:                     ) if source_capacity.is_grouped else {}),
208:                 )
209:                 if source_capacity.is_grouped:
210:                     from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import get_grouped_nixl_payload
211: 
212:                     self._transferred_group_block_ids, _ = get_grouped_nixl_payload(
213:                         source_capacity, request.num_prefill_tokens,
214:                         self._source_group_block_ids, self._target_group_block_ids,
215:                     )
```
```text
228:         transfer_info = KVCacheTransferInfo(
229:             batch=self._batch,
230:             source_cluster_type=self._source_cluster_type,
231:             target_cluster_type=self._target_cluster_type,
232:             source_replica_id=self._source_replica_id,
233:             source_replica_local_id=self._source_replica_local_id,
234:             source_batch_stage_id=self._source_batch_stage_id,
235:             kv_cache_size_bytes=self._kv_cache_size_bytes,
236:             transfer_time_ms=self._transfer_time_ms,
237:             transfer_start_time=self.time,
238:             target_replica_id=self._target_replica_id,
239:             target_replica_local_id=self._target_replica_local_id,
240:             kv_connector=self._kv_connector,
241:             payload_kind=(
242:                 self._payload_kind()
243:                 if self._kv_connector == "nixl"
244:                 else None
245:             ),
246:             source_prompt_common_blocks=self._source_prompt_common_blocks,
247:             target_reserved_common_blocks=self._target_reserved_common_blocks,
248:             source_group_block_ids=self._source_group_block_ids,
249:             target_group_block_ids=self._target_group_block_ids,
250:             transferred_group_block_ids=self._transferred_group_block_ids,
251:         )
252:         if self._kv_connector == "nixl":
253:             self._started = True
254: 
255:         metrics_store.on_kv_cache_transfer_start(
256:             self.time,
257:             self._source_replica_id,
258:             self._source_replica_local_id,
259:             self._target_cluster_type,
260:             self._kv_cache_size_bytes,
261:             transfer_info,
262:         )
263: 
264:         for request in self._batch.requests:
265:             request.on_kv_cache_transfer_start(self.time)
266: 
267:         if read_lane is not None:
268:             from frontier.events.replica_schedule_event import ReplicaScheduleEvent
269: 
270:             # The lane's next engine step posts the READ.
271:             read_lane.register_nixl_read(transfer_info)
272:             return [
273:                 ReplicaScheduleEvent(
274:                     self.time,
275:                     self._target_replica_id,
276:                     self._target_cluster_type,
277:                     self._target_replica_local_id,
278:                 )
279:             ]
280:         transfer_end_time = self.time + self._transfer_time_ms * 1e-3
281:         return [KVCacheTransferEndEvent(transfer_end_time, transfer_info)]
282: 
283:     def get_target_cluster(self) -> ClusterType:
284:         return self._target_cluster_type
285: 
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/events/kv_cache_transfer_end_event.py
```text
12: def report_nixl_reads(
13:     scheduler: "BaseGlobalScheduler",
14:     metrics_store: "MetricsStore",
15:     replica_scheduler,
16:     time: float,
17: ) -> List[BaseEvent]:
18:     """Complete the NIXL READs that a DECODE engine step's output reports (`finished_recving`).
19: 
20:     The transfer end runs inline at the output pop: at equal times the
21:     engine's next pass (REPLICA_SCHEDULE) runs before a KV_CACHE_TRANSFER_END
22:     event, and that pass must find the requests ready.
23:     """
24:     if not hasattr(replica_scheduler, "consume_found_nixl_reads"):
25:         return []
26:     events: List[BaseEvent] = []
27:     for transfer_info in replica_scheduler.consume_found_nixl_reads():
28:         events.extend(
29:             KVCacheTransferEndEvent(time, transfer_info).handle_event(scheduler, metrics_store)
30:         )
31:     return events
32: 
33: 
34: class KVCacheTransferEndEvent(BaseEvent):
35:     """Event emitted when a KV cache transfer completes."""
36: 
37:     def __init__(
38:         self,
39:         time: float,
40:         transfer_info: "KVCacheTransferInfo",
41:     ) -> None:
42:         super().__init__(time, EventType.KV_CACHE_TRANSFER_END)
43:         self._transfer_info = transfer_info
44:         self._transfer_info.transfer_end_time = time
```
```text
68:             target_replica_scheduler.validate_kv_transfer_target_completion(
69:                 request,
70:                 sum(block is not None for table in self._transfer_info.target_group_block_ids.values() for block in table)
71:                 if self._transfer_info.target_group_block_ids is not None
72:                 else self._transfer_info.target_reserved_common_blocks,
73:             )
74:             source_cluster_scheduler = scheduler.get_cluster_scheduler(
75:                 self._transfer_info.source_cluster_type
76:             )
77:             source_replica_scheduler = source_cluster_scheduler.get_replica_scheduler(
78:                 self._transfer_info.source_replica_id,
79:                 self._transfer_info.source_replica_local_id,
80:             )
81:             source_capacity = source_replica_scheduler._kv_cache_capacity
82:             if source_capacity.is_grouped:
83:                 from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import get_grouped_nixl_payload
84: 
85:                 source_groups = source_replica_scheduler.get_kv_transfer_source_group_block_ids(request)
86:                 target_groups = target_replica_scheduler._kv_cache_manager.get_group_block_ids(request.id)
87:                 if source_groups != self._transfer_info.source_group_block_ids or target_groups != self._transfer_info.target_group_block_ids:
88:                     raise RuntimeError("NIXL grouped source or destination ownership changed")
89:                 transferred, expected_payload_bytes = get_grouped_nixl_payload(
90:                     source_capacity, request.num_prefill_tokens, source_groups, target_groups,
91:                 )
92:                 if transferred != self._transfer_info.transferred_group_block_ids:
93:                     raise RuntimeError("NIXL grouped transferred pages changed")
94:             else:
95:                 source_blocks = source_replica_scheduler.get_kv_transfer_source_prompt_common_blocks(request)
96:                 if source_blocks != self._transfer_info.source_prompt_common_blocks:
97:                     raise RuntimeError(
98:                         f"Request {request.id} PREFILL ownership changed from "
99:                         f"{self._transfer_info.source_prompt_common_blocks} to "
100:                         f"{source_blocks} prompt blocks"
101:                     )
102:                 expected_payload_bytes = source_blocks * source_capacity.resident_bytes_per_common_block
103:             if self._transfer_info.kv_cache_size_bytes != expected_payload_bytes:
104:                 raise RuntimeError(
105:                     f"Request {request.id} NIXL payload changed from "
106:                     f"{expected_payload_bytes} to "
107:                     f"{self._transfer_info.kv_cache_size_bytes} bytes"
108:                 )
109:             source_replica_scheduler.validate_kv_transfer_source_release(request)
```
```text
111:         transfer_duration_s = self.time - self._transfer_info.transfer_start_time
112:         transfer_duration_ms = transfer_duration_s * 1e3
113:         metrics_store.on_kv_cache_transfer_end(
114:             self.time,
115:             transfer_duration_ms,
116:             self._transfer_info.kv_cache_size_bytes,
117:             self._transfer_info.target_cluster_type,
118:             self._transfer_info,
119:         )
120: 
121:         for request in batch.requests:
122:             request.on_kv_cache_transfer_complete(self.time, transfer_duration_s)
123: 
124:         if is_structural_nixl:
125:             from frontier.logger import get_cluster_logger
126:             from frontier.scheduler.utils.kv_arrival import (
127:                 handle_reserved_decode_arrival,
128:             )
129: 
130:             arrival_events = handle_reserved_decode_arrival(
131:                 target_replica_scheduler,
132:                 self.time,
133:                 batch,
134:                 self._transfer_info,
135:                 get_cluster_logger(__name__, ClusterType.DECODE.name),
136:             )
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py
```text
39:     def on_engine_step_start(self, time: float, step: Optional[Batch] = None) -> None:
40:         """Start one engine step: a batch, a dummy forward, or with no `step` a connector-only step.
41: 
42:         The step first finds the posted READs whose wire time has ended
43:         (`get_finished`) and keeps them until its output, then posts the READs
44:         it carries (`start_load_kv`), so the step that posts a READ never finds
45:         it. A connector-only step reports what it finds at once.
46:         """
47:         posted = []
48:         found = []
49:         for wire_end, transfer_info in self._nixl_reads_posted:
50:             if wire_end <= time:
51:                 found.append(transfer_info)
52:             else:
53:                 posted.append((wire_end, transfer_info))
54:         to_post = self._nixl_reads_unposted
55:         if step is None:
56:             self._nixl_reads_reported.extend(found)
57:         elif isinstance(step, DummyForwardBatch):
58:             self._nixl_reads_found_by_dummy_forward.extend(found)
59:         else:
60:             if found:
61:                 self._nixl_reads_found_by_batch[step.id] = found
62:             to_post = to_post + self._nixl_reads_claimed.pop(step.id, [])
63:         posted.extend(
64:             (time + transfer_info.transfer_time_ms * 1e-3, transfer_info)
65:             for transfer_info in to_post
66:         )
67:         self._nixl_reads_posted = posted
68:         self._nixl_reads_unposted = []
69: 
70:     def _report_nixl_reads_of_batch(self, batch: Batch) -> None:
71:         """The engine takes a batch's output, which reports the READs its start found."""
72:         self._nixl_reads_reported.extend(self._nixl_reads_found_by_batch.pop(batch.id, ()))
73: 
74:     def _report_nixl_reads_of_dummy_forward(self) -> None:
75:         """The engine's dummy forward ended; its step reports the READs it found."""
76:         self._nixl_reads_reported.extend(self._nixl_reads_found_by_dummy_forward)
77:         self._nixl_reads_found_by_dummy_forward = []
78: 
79:     def consume_found_nixl_reads(self) -> List:
80:         """The found READs whose step output the engine took, to complete now."""
81:         reported, self._nixl_reads_reported = self._nixl_reads_reported, []
82:         return reported
83: 
84:     @property
85:     def next_nixl_read_end(self) -> Optional[float]:
86:         """Earliest wire end among the posted READs."""
87:         return min((wire_end for wire_end, _ in self._nixl_reads_posted), default=None)
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py
```text
353:     def next_pass_start(self, time: float) -> float:
354:         """Start of the engine's next pass after its dummy forward ended at `time`.
355: 
356:         An execution path may price an idle forward at zero
357:         (`predict_step_cost_execution`), so a group whose forwards are all
358:         dummy forwards spins at no simulated cost, and such a dummy forward
359:         ends when it starts. While no engine of the group then has a request
360:         to schedule and some wait for posted NIXL READs, no pass changes
361:         anything until the earliest READ's wire ends; the pass that starts
362:         then finds it (`get_finished`), as the next step of an idle engine
363:         does. An engine that receives work earlier wakes the others through
364:         its forward's first room.
365:         """
366:         if self._dummy_pass_start is None or time > self._dummy_pass_start:
367:             return time
368:         group = self._dp_group()
369:         if self._runs_dp_forwards_ahead and not all(
370:             lane._last_zero_cost_dummy_end == time for lane in group
371:         ):
372:             # A host return at formation says nothing about GPU cost. Only
373:             # device ends at that same formation time witness a zero-cost group.
374:             return time
375:         if not all(lane._only_posted_nixl_reads_unfinished for lane in group):
376:             return time
377:         return min(
378:             (lane.next_nixl_read_end for lane in group if lane._nixl_reads_posted),
379:             default=time,
380:         )
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/entities/request.py
```text
329:     def e2e_time(self) -> float:
330:         return self._completed_at - self._arrived_at
331: 
332:     @property
333:     @check_scheduled
334:     def e2e_time_normalized(self) -> float:
335:         return self.e2e_time / self.num_decode_tokens
```
```text
953:     def ttft(self) -> float:
954:         """
955:         Time To First Token (TTFT).
956:         Measured from request arrival to prefill completion.
957:         Returns 0 if prefill hasn't completed yet.
958:         """
959:         if self._prefill_completed_at == 0:
960:             return 0
961:         return self._prefill_completed_at - self._arrived_at
962: 
963:     @property
964:     def tpot(self) -> float:
965:         """
966:         Time Per Output Token (TPOT).
967:         Average time per output token excluding the first token.
968:         Returns 0 if there's only one or no decode tokens.
969:         """
970:         if self._num_decode_tokens <= 1 or self._first_decode_token_completed_at == 0:
971:             return 0
972:         total_decode_time = self._completed_at - self._first_decode_token_completed_at
973:         return total_decode_time / (self._num_decode_tokens - 1)
974: 
975:     def on_kv_cache_transfer_start(self, transfer_start_time: float) -> None:
976:         """Record the earliest request-level KV transfer start timestamp."""
977:         start_time = float(transfer_start_time)
978:         if self._kv_cache_transfer_start_time is None:
979:             self._kv_cache_transfer_start_time = start_time
980:             return
981:         self._kv_cache_transfer_start_time = min(
982:             self._kv_cache_transfer_start_time,
983:             start_time,
984:         )
985: 
986:     def on_kv_cache_transfer_complete(
987:         self,
988:         transfer_end_time: float,
989:         transfer_time: float,
990:     ) -> None:
991:         """
992:         Record KV cache transfer time for this request.
993: 
994:         Args:
995:             transfer_end_time: End timestamp of the request-level transfer window in
996:                 seconds.
997:             transfer_time: Duration of the KV cache transfer in seconds
998:         """
999:         end_time = float(transfer_end_time)
1000:         if self._kv_cache_transfer_end_time is None:
1001:             self._kv_cache_transfer_end_time = end_time
1002:         else:
1003:             self._kv_cache_transfer_end_time = max(
1004:                 self._kv_cache_transfer_end_time,
1005:                 end_time,
1006:             )
1007:         self._kv_cache_transfer_time += transfer_time
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/metrics/metrics_store.py
```text
2424:         # 5. Throughput metrics
2425:         if self._completed_requests > 0:
2426:             # Get first request arrival and last request completion times
2427:             arrival_data = self._request_completion_metrics_time_series.get(
2428:                 RequestCompletionMetricsTimeSeries.REQUEST_ARRIVAL
2429:             )
2430:             completion_data = self._request_completion_metrics_time_series.get(
2431:                 RequestCompletionMetricsTimeSeries.REQUEST_COMPLETION
2432:             )
2433: 
2434:             if (
2435:                 arrival_data
2436:                 and completion_data
2437:                 and len(arrival_data) > 0
2438:                 and len(completion_data) > 0
2439:             ):
2440:                 first_arrival = min(x for x, _ in arrival_data._data_series)
2441:                 last_completion = max(x for x, _ in completion_data._data_series)
2442:                 # NOTE: Simulator time is in SECONDS, convert to milliseconds for reporting
2443:                 total_duration_s = last_completion - first_arrival
2444:                 total_duration_ms = total_duration_s * 1000.0  # Convert to milliseconds
2445: 
2446:                 if total_duration_ms > 0:
2447:                     # Calculate total tokens processed
2448:                     total_tokens = 0
2449:                     total_decode_tokens = 0
2450:                     decode_tokens_data = self._request_metrics_histogram.get(
2451:                         RequestMetricsHistogram.REQUEST_DECODE_TOKENS
2452:                     )
2453:                     num_tokens_data = self._request_metrics_histogram.get(
2454:                         RequestMetricsHistogram.REQUEST_NUM_TOKENS
2455:                     )
2456: 
2457:                     if decode_tokens_data and len(decode_tokens_data) > 0:
2458:                         total_decode_tokens = sum(
2459:                             y for _, y in decode_tokens_data._data_series
2460:                         )
2461:                     if num_tokens_data and len(num_tokens_data) > 0:
2462:                         total_tokens = sum(y for _, y in num_tokens_data._data_series)
2463: 
2464:                     system_metrics["throughput_metrics"] = {
2465:                         "total_duration_ms": total_duration_ms,
2466:                         "total_duration_seconds": total_duration_s,
2467:                         "requests_per_second": self._completed_requests
2468:                         / total_duration_s,
2469:                         "total_tokens_processed": int(total_tokens),
2470:                         "total_decode_tokens_generated": int(total_decode_tokens),
2471:                         "tokens_per_second": total_tokens / total_duration_s
2472:                         if total_tokens > 0
2473:                         else 0,
2474:                         "decode_tokens_per_second": total_decode_tokens
2475:                         / total_duration_s
2476:                         if total_decode_tokens > 0
2477:                         else 0,
2478:                     }
2479:                 else:
```
```text
2981:     def _get_ttft_export_value(self, request: Request) -> float:
2982:         return request.ttft
```
```text
3501:         # TTFT (Time To First Token) metrics
3502:         if request.is_prefill_complete:
3503:             ttft_value = self._get_ttft_export_value(request)
3504:             self._request_metrics_time_distributions[
3505:                 RequestMetricsTimeDistributions.TTFT
3506:             ].put(request.id, ttft_value)
3507: 
3508:             # TTFT breakdown components
3509:             if request.is_prefill_complete:
3510:                 # Prefill execution time
3511:                 prefill_exec_time = request.prefill_completed_at - request.scheduled_at
3512:                 self._request_metrics_time_distributions[
3513:                     RequestMetricsTimeDistributions.TTFT_PREFILL_ONLY
3514:                 ].put(request.id, prefill_exec_time)
3515: 
3516:                 # KV cache transfer time
3517:                 self._request_metrics_time_distributions[
3518:                     RequestMetricsTimeDistributions.TTFT_KV_TRANSFER
3519:                 ].put(request.id, request.kv_cache_transfer_time)
3520: 
3521:                 # Legacy TTFT residual kept for backward-compatible reporting.
3522:                 # This is no longer the canonical first pure decode latency.
3523:                 # Use decode_first_token_latency for the actual first pure decode
3524:                 # token duration after TTFT re-anchoring.
3525:                 ttft_decode_first = (
3526:                     ttft_value - prefill_exec_time - request.kv_cache_transfer_time
3527:                 )
3528:                 self._request_metrics_time_distributions[
3529:                     RequestMetricsTimeDistributions.TTFT_DECODE_FIRST_TOKEN
3530:                 ].put(request.id, max(0, ttft_decode_first))  # Ensure non-negative
3531: 
3532:         if request.num_decode_tokens > 1:
3533:             if request.first_decode_token_completed_at <= 0:
3534:                 raise ValueError(
3535:                     f"Missing first token completion timestamp for request_id={request.id}"
3536:                 )
3537:             if request.decode_first_token_completed_at <= 0:
3538:                 raise ValueError(
3539:                     f"Missing decode-first-token completion timestamp for request_id={request.id}"
3540:                 )
3541:             decode_first_token_latency = (
3542:                 request.decode_first_token_completed_at - request.first_decode_token_completed_at
3543:             )
3544:             if decode_first_token_latency < -1e-9:
3545:                 raise ValueError(
3546:                     "Negative decode-first-token latency for "
3547:                     f"request_id={request.id}: {decode_first_token_latency} "
3548:                     f"(first_decode_token_completed_at={request.first_decode_token_completed_at}, "
3549:                     f"decode_first_token_completed_at={request.decode_first_token_completed_at})"
3550:                 )
3551:             self._request_metrics_time_distributions[
3552:                 RequestMetricsTimeDistributions.DECODE_FIRST_TOKEN_LATENCY
3553:             ].put(request.id, max(0, decode_first_token_latency))
3554: 
3555:         # TPOT (Time Per Output Token) metrics
3556:         if (
3557:             request.num_decode_tokens > 1
3558:             and request.first_decode_token_completed_at > 0
3559:         ):
3560:             self._request_metrics_time_distributions[
3561:                 RequestMetricsTimeDistributions.TPOT
3562:             ].put(request.id, request.tpot)
```
```text
5468:         if self._trace_store and self._config.enable_op_level_tracing:
5469:             from frontier.metrics.trace_store import TraceEvent
5470: 
5471:             batch_id = transfer_info.batch.id if transfer_info.batch else None
5472:             request_ids = (
5473:                 [str(req.id) for req in transfer_info.batch.requests]
5474:                 if transfer_info.batch and transfer_info.batch.requests
5475:                 else []
5476:             )
5477: 
5478:             cluster_config = self._cluster_configs.get(transfer_info.source_cluster_type)
5479:             if cluster_config is None:
5480:                 raise ValueError(
5481:                     f"Cluster config not found for {transfer_info.source_cluster_type}"
5482:                 )
5483:             transfer_meta = build_kv_cache_transfer_meta(
5484:                 transfer_info.batch,
5485:                 cluster_config.replica_config,
5486:                 transfer_info.source_cluster_type,
5487:                 kv_cache_size_bytes,
5488:                 transfer_info=transfer_info,
5489:                 block_size=cluster_config.replica_scheduler_config.block_size,
5490:             )
5491:             total_tokens = transfer_meta["total_tokens"]
5492:             trace_context = OpTraceContext(
5493:                 cluster_type=transfer_info.source_cluster_type,
5494:                 model_config=cluster_config.replica_config.model_config,
5495:                 replica_config=cluster_config.replica_config,
5496:                 total_tokens=total_tokens,
5497:                 effective_tokens_compute=total_tokens,
5498:                 effective_tokens_transfer=total_tokens,
5499:                 effective_tokens_rounded=(total_tokens + 7) // 8 * 8,
5500:                 tokens_are_post_routing=False,
5501:             )
5502:             transfer_meta["parallel_context"] = build_parallel_context(trace_context)
5503:             transfer_meta["model_name"] = cluster_config.replica_config.model_name
5504:             transfer_meta["request_ids"] = request_ids
5505:             transfer_meta["source_replica_local_id"] = source_replica_local_id
5506: 
5507:             event = TraceEvent(
5508:                 type="TRANSFER",
5509:                 name="kv_cache_transfer",
5510:                 ts_start=time,
5511:                 duration_ms=transfer_info.transfer_time_ms,
5512:                 cluster=transfer_info.source_cluster_type.name,
5513:                 replica_id=source_replica_id,
5514:                 batch_id=batch_id,
5515:                 target_cluster=target_cluster_type.name,
5516:                 meta=transfer_meta,
5517:             )
5518:             self._trace_store.log_event(event)
```
```text
5520:         if not self._config.write_metrics:
5521:             return
5522: 
5523:         self._register_transfer_ledger_start(
5524:             transfer_info=transfer_info,
5525:             transfer_kind="kv_cache",
5526:             source_replica_id=source_replica_id,
5527:             source_replica_local_id=source_replica_local_id,
5528:             target_cluster_type=target_cluster_type,
5529:             byte_count=kv_cache_size_bytes,
5530:             start_time=time,
5531:         )
5532:         self._kv_cache_transfer_metrics["transfer_count"] += 1
```
```text
5534:     def on_kv_cache_transfer_end(
5535:         self,
5536:         time: float,
5537:         duration: float,
5538:         size_bytes: int,
5539:         target_cluster_type: ClusterType,
5540:         transfer_info: Any,
5541:     ) -> None:
5542:         if not self._config.write_metrics:
5543:             return
5544: 
5545:         self._kv_cache_transfer_metrics["total_transfer_time"] += duration
5546:         self._kv_cache_transfer_metrics["total_data_transferred"] += size_bytes
5547: 
5548:         request_info = ""
5549:         if transfer_info.batch and transfer_info.batch.requests:
5550:             request_ids = [str(req.id) for req in transfer_info.batch.requests]
5551:             request_info = f"_req_{'_'.join(request_ids)}"
5552: 
5553:         transfer_id = (
5554:             f"transfer_{self._kv_cache_transfer_metrics['transfer_count']}"
5555:             f"{request_info}"
5556:         )
5557:         self._kv_cache_transfer_metrics["transfer_times"].put(transfer_id, duration)
5558:         self._kv_cache_transfer_metrics["transfer_sizes"].put(transfer_id, size_bytes)
5559:         self._complete_transfer_ledger_end(
5560:             transfer_info=transfer_info,
5561:             end_time=time,
5562:             duration_ms=duration,
5563:         )
5564: 
5565:     def on_m2n_transfer_start(
5566:         self,
5567:         time: float,
5568:         source_replica_id: int,
5569:         source_cluster_type: ClusterType,
5570:         target_cluster_type: ClusterType,
5571:         activation_size_bytes: int,
5572:         transfer_info: Any,
5573:     ) -> None:
5574:         if self._trace_store and self._config.enable_op_level_tracing:
5575:             from frontier.metrics.trace_store import TraceEvent
5576: 
5577:             cluster_config = self._cluster_configs.get(source_cluster_type)
5578:             if cluster_config is None:
5579:                 raise ValueError(
5580:                     f"Cluster config not found for {source_cluster_type}"
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/metrics/op_trace_utils.py
```text
841:     transferred, payload_bytes = get_grouped_nixl_payload(
842:         capacity, request.num_prefill_tokens,
843:         transfer_info.source_group_block_ids, transfer_info.target_group_block_ids,
844:     )
845:     if (
846:         transferred != transfer_info.transferred_group_block_ids
847:         or payload_bytes != transfer_size_bytes
848:         or payload_bytes != transfer_info.kv_cache_size_bytes
849:     ):
850:         raise ValueError("Grouped NIXL trace snapshots or bytes do not match the payload")
```
```text
881:     return {
882:         "precision_op": "kv_cache_transfer", "dtype": "MIXED", "dtype_bytes": None,
883:         # Packed pages include mixed row formats and alignment, not dense tensors.
884:         "tensor_shape": {}, "tensor_size_bytes": tensor_size_bytes,
885:         "transfer_size_bytes": payload_bytes, "total_tokens": request.num_prefill_tokens,
886:         "num_layers": model_config.num_layers, "num_q_heads": model_config.num_q_heads,
887:         "num_heads": None, "num_kv_heads": None, "head_dim": None,
888:         "payload_kind": transfer_info.payload_kind,
889:         "byte_scope": "per_worker_lane_resident_pages", "block_size": capacity.block_size,
890:         "cache_groups": groups, "cache_components": components,
891:         "source_group_block_ids": {key: list(value) for key, value in transfer_info.source_group_block_ids.items()},
892:         "target_group_block_ids": {key: list(value) for key, value in transfer_info.target_group_block_ids.items()},
893:         "transferred_group_block_ids": {key: list(value) for key, value in transferred.items()},
894:         "wire_bytes": get_grouped_nixl_wire_bytes(capacity, transferred),
895:         "native_wire_bytes_known": True,
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/scheduler/utils/memory_planner.py
```text
253:         Frontier approximates non_kv_cache_memory with parameter memory
254:         plus an optional calibrated overhead term.
255:         """
256:         capacity = self._get_component_kv_cache_capacity(block_size)
257:         if capacity is None:
258:             page_size = self._get_kv_cache_memory_per_layer_per_block(block_size)
259:             num_full_attention_layers = self._get_num_full_attention_layers_per_device()
260:             common_block_bytes = page_size * num_full_attention_layers
261:             cache_details = {"page_size_bytes": int(page_size)}
262:         else:
263:             common_block_bytes = capacity.physical_block_size_bytes
264:             size_key = (
265:                 "physical_block_size_bytes"
266:                 if capacity.is_grouped else "resident_bytes_per_common_block"
267:             )
268:             cache_details = {
269:                 size_key: common_block_bytes,
270:                 "num_cache_components": len(capacity.components),
271:                 "num_cache_component_owners": sum(
272:                     len(component.owner_layer_ids)
273:                     for component in capacity.components
274:                 ),
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py
```text
414:         )
415:         batch._cluster_type = self._cluster_type
416:         if self._is_grouped_kv_allocation():
417:             batch.kv_block_table_writes = self._record_kv_block_table_writes(requests)
418:             block_ids = tuple(
419:                 block_id for group in self._kv_cache_capacity.groups
420:                 for block_id in self._pending_cache_zero_block_ids.get(group.group_id, ())
421:             )
422:             if block_ids:
423:                 batch.cache_block_zeroing = KVCacheBlockZeroing(
424:                     block_ids, self._kv_cache_capacity.physical_block_size_bytes,
425:                 )
426:             self._emit_schedule_decision_event(
427:                 event="grouped_cache_batch", decision_result=None, request_id=None,
428:                 token_budget=int(self._current_iteration_token_budget),
429:                 num_tokens=sum(num_tokens),
430:                 extra_fields={
431:                     "batch_id": batch.id, "replica_id": self._replica_id,
432:                     "replica_local_id": self._replica_local_id,
433:                     "zero_required_block_ids": list(block_ids),
434:                     "physical_block_size_bytes": self._kv_cache_capacity.physical_block_size_bytes,
435:                 },
436:             )
437:             self._pending_cache_zero_block_ids.clear()
438:         metadata = self._build_decode_cuda_graph_metadata(batch)
```

## /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/entities/kv_cache_transfer_info.py
```text
19:     source_batch_stage_id: int
20:     kv_cache_size_bytes: int
21:     transfer_time_ms: float
22:     transfer_start_time: float
23:     transfer_end_time: Optional[float] = None
24:     enable_compression: bool = False
25:     compression_ratio: float = 1.0
```
```text
46:         ):
47:             raise ValueError(
48:                 "source_batch_stage_id must be a non-negative int, "
49:                 f"got {self.source_batch_stage_id!r}"
50:             )
51:         if self.target_batch_stage_id is not None and (
52:             type(self.target_batch_stage_id) is not int
53:             or self.target_batch_stage_id < 0
54:         ):
55:             raise ValueError(
56:                 "target_batch_stage_id must be a non-negative int or None, "
57:                 f"got {self.target_batch_stage_id!r}"
58:             )
59:         if self.transfer_end_time is None:
60:             self.transfer_end_time = self.transfer_start_time + (self.transfer_time_ms * 1e-3)
61:         if self.kv_connector == "nixl":
62:             if self.payload_kind == "structural_grouped_prompt_pages_per_worker_lane":
63:                 if self.source_prompt_common_blocks is not None or self.target_reserved_common_blocks is not None:
64:                     raise ValueError("Grouped NIXL must not use scalar common blocks")
65:                 if not self.source_group_block_ids or not self.target_group_block_ids or not self.transferred_group_block_ids:
```
```text
96:     @property
97:     def effective_data_size_bytes(self) -> int:
98:         if self.enable_compression:
99:             return int(self.kv_cache_size_bytes / self.compression_ratio)
100:         return self.kv_cache_size_bytes
```

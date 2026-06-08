from frontier.entities.batch import Batch
from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from frontier.logger import get_cluster_logger
from frontier.types.cluster_type import ClusterType



class OrcaReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests = []
        self._num_running_batches = 0

        # Initialize micro-batch size for decode-attn (PD+AF) use case
        # Fallbacks:
        #  - prefer cluster-specific decode_attn_micro_batch_size from ClusterConfig
        #  - else use generic micro_batch_size if provided on config
        #  - else default to 1 (most conservative)
        if self._cluster_type == ClusterType.DECODE_ATTN:
            mbs = None
            try:
                if getattr(self, "_cluster_scheduler", None) is not None:
                    cfg = getattr(self._cluster_scheduler, "_config", None)
                    if cfg is not None:
                        mbs = getattr(cfg, "decode_attn_micro_batch_size", None)
            except Exception:
                mbs = None
            if mbs is None:
                mbs = 1
            self._micro_batch_size = int(mbs)
        else:
            # For other clusters, micro-batch sizing is not used by Orca; set to max batch size
            self._micro_batch_size = self._max_batch_size

    def on_batch_end(self, batch: Batch) -> None:
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.is_finished_for_cluster(self._cluster_type):
                # Request is done for this cluster.
                # In PREFILL, its KV cache will be transferred. We simply stop tracking it.
                # In DECODE or monolithic mode, it's fully done, so we free its memory.
                if self._cluster_type != ClusterType.PREFILL:
                    # Avoid double-free: only free if still tracked in allocation map
                    if request.id in self._allocation_map:
                        self.free(request.id)
                    else:
                        logger.debug(f"[ON_BATCH_END] Skip freeing request {request.id}: already freed (cluster={self._cluster_type})")

                # NOTE: for prefill cluster, we will free reqs in kvcache transfer end event
            else:
                # Handle different cases based on cluster type
                # exclude ClusterType.DECODE_ATTN,
                if self._cluster_type in [ClusterType.DECODE_FFN]:
                    # we donnot need to do anything for decode-ffn cluster
                    pass
                else:
                    # This case applies to monolithic mode where a request
                    # is not fully decoded and needs to be preempted.
                    # and we also consider decode-attn here, because re-batching is needed in decode-attn
                    self._preempted_requests.append(request)
                    if self._cluster_type == ClusterType.DECODE_ATTN:
                        logger.info(
                            f"[DECODE_ATTN ON_BATCH_END] Request {request.id} preempted -> preempted_queue; "
                            f"tok_idx={getattr(request,'current_decode_token_index',None)} "
                            f"layer={getattr(request,'completed_layer_count',None)} (cluster={self._cluster_type})"
                        )

    def _get_next_batch(self, is_micro_batch: bool) -> Batch:
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        """
        Build the next batch.
        - Non micro-batch: keep original behavior (preempted first, then main queue)
        - Micro-batch on DECODE_ATTN: group requests by completed_layer_count and
          pick the group that can best fill the micro-batch size. Requests in the
          same micro-batch must share the same current layer count; token indices
          may differ.
        """
        requests = []
        num_tokens = []
        batch_or_micro_batch_size = self._max_batch_size if not is_micro_batch else self._micro_batch_size

        # Fast path for non micro-batch or non-DECODE_ATTN clusters: original behavior
        if (not is_micro_batch) or (self._cluster_type != ClusterType.DECODE_ATTN):
            preempted_req_ids = [r.id for r in self._preempted_requests]
            if preempted_req_ids:
                logger.info(f"[{self._cluster_type.name}][Replica {self._replica_id}][DP {self._dp_id}] "
                            f"_get_next_batch: Found {len(preempted_req_ids)} requests in preempted queue: {preempted_req_ids}")

            # all preempted_requests will have prefill completed
            while self._preempted_requests:
                if len(requests) == batch_or_micro_batch_size:
                    break
                req = self._preempted_requests.pop(0)
                next_num_tokens = self._get_request_next_num_tokens(req)
                requests.append(req)
                num_tokens.append(next_num_tokens)

            request_queue_ids = [r.id for r in self._request_queue]
            if request_queue_ids:
                logger.info(f"[{self._cluster_type.name}][Replica {self._replica_id}][DP {self._dp_id}] "
                            f"_get_next_batch: Found {len(request_queue_ids)} requests in main queue: {request_queue_ids}")

            while self._request_queue:
                if len(requests) == batch_or_micro_batch_size:
                    break
                if not self.can_allocate(self._max_blocks_per_sequence):
                    break
                req = self._request_queue.pop(0)
                self.allocate(req.id, self._max_blocks_per_sequence)
                next_num_tokens = self._get_request_next_num_tokens(req)
                requests.append(req)
                num_tokens.append(next_num_tokens)

            if not requests:
                return

            new_batch = self._create_batch(requests, num_tokens)
            new_batch_req_ids = [r.id for r in new_batch.requests]
            logger.info(f"[{self._cluster_type.name}][Replica {self._replica_id}][DP {self._dp_id}] "
                        f"_get_next_batch: CREATED batch {new_batch.id} with requests {new_batch_req_ids}")
            return new_batch

        # Debug: micro-batch formation input state (IDs, token_idx, layer)
        try:
            pre_list = [f"id={r.id}|tok={getattr(r,'current_decode_token_index',None)}|layer={getattr(r,'completed_layer_count',None)}" for r in self._preempted_requests]
            main_list = [f"id={r.id}|tok={getattr(r,'current_decode_token_index',None)}|layer={getattr(r,'completed_layer_count',None)}" for r in self._request_queue]
            logger.info(
                f"[MB-FORMATION][PRE-STATE][Replica {self._replica_id}][DP {self._dp_id}] preempted={pre_list} main={main_list}"
            )
        except Exception as e:
            logger.debug(f"[MB-FORMATION][PRE-STATE] logging failed: {e}")

        # Micro-batch for DECODE_ATTN: layer-consistent grouping
        # 1) Build groups by completed_layer_count from both preempted and main queues (lookahead only)
        from collections import defaultdict
        group_preempted = defaultdict(list)  # layer -> list[(idx_in_preempted, req)]
        for idx, req in enumerate(self._preempted_requests):
            layer = req.completed_layer_count
            group_preempted[layer].append((idx, req))

        group_main = defaultdict(list)  # layer -> list[(idx_in_main, req)]
        for idx, req in enumerate(self._request_queue):
            layer = req.completed_layer_count
            group_main[layer].append((idx, req))

        # 2) Choose the layer that can best fill the micro-batch size
        best_layer = None
        best_total = -1
        best_preempted_count = -1
        for layer in set(list(group_preempted.keys()) + list(group_main.keys())):
            total = len(group_preempted[layer]) + len(group_main[layer])
            preempted_cnt = len(group_preempted[layer])
            # Pick the group with the most candidates; tie-breaker: more preempted
            if total > best_total or (total == best_total and preempted_cnt > best_preempted_count):
                best_total = total
                best_preempted_count = preempted_cnt
                best_layer = layer
        try:
            cand_pre = [req.id for (_i, req) in group_preempted.get(best_layer, [])]
            cand_main = [req.id for (_i, req) in group_main.get(best_layer, [])]
            logger.info(
                f"[MB-FORMATION][CAND] layer={best_layer} preempted={cand_pre} main={cand_main}"
            )
        except Exception as e:
            logger.debug(f"[MB-FORMATION][CAND] logging failed: {e}")


        if best_layer is None:
            # No candidates at all: nothing to schedule
            return

        logger.info(
            f"[DECODE_ATTN][Replica {self._replica_id}][DP {self._dp_id}] _get_next_batch(MB) "
            f"pick layer={best_layer} candidates total={best_total} (preempted={best_preempted_count}, main={best_total - best_preempted_count}) "
            f"mb_size={batch_or_micro_batch_size}"
        )

        # 3) Materialize the batch: take from preempted first (no extra allocation), then from main queue
        #    Do not destructively modify queues until we know what to pick
        target_size = batch_or_micro_batch_size

        # From preempted
        take_from_preempted = []  # list of indices in self._preempted_requests
        for idx, req in group_preempted.get(best_layer, []):
            if len(requests) == target_size:
                break
            take_from_preempted.append(idx)
            requests.append(req)
            num_tokens.append(self._get_request_next_num_tokens(req))

        # From main queue (respect allocation limits)
        take_from_main = []  # list of indices in self._request_queue
        for idx, req in group_main.get(best_layer, []):
            if len(requests) == target_size:
                break
            if not self.can_allocate(self._max_blocks_per_sequence):
                break
            take_from_main.append(idx)
            requests.append(req)
            num_tokens.append(self._get_request_next_num_tokens(req))

        if not requests:
            # Fallback: nothing could be formed due to allocation; do not modify queues
            return

        # 4) Apply destructive pops in reverse index order to keep indices valid
        for idx in sorted(take_from_preempted, reverse=True):
            self._preempted_requests.pop(idx)
        for idx in sorted(take_from_main, reverse=True):
            req = self._request_queue.pop(idx)
            self.allocate(req.id, self._max_blocks_per_sequence)

        new_batch = self._create_batch(requests, num_tokens)
        new_batch_req_ids = [r.id for r in new_batch.requests]
        logger.info(
            f"[DECODE_ATTN][Replica {self._replica_id}][DP {self._dp_id}] _get_next_batch(MB): CREATED batch {new_batch.id} "
            f"layer={best_layer} reqs={new_batch_req_ids}"
        )
        try:
            r0_in = 0 in new_batch_req_ids
            r1_in = 1 in new_batch_req_ids
            logger.info(
                f"[MB-FORMATION][INCLUDE] batch={new_batch.id} includes req0={r0_in} req1={r1_in} reqs={new_batch_req_ids}"
            )
        except Exception as e:
            logger.debug(f"[MB-FORMATION][INCLUDE] logging failed: {e}")

        return new_batch

"""Terminal release timing for monolithic pipeline parallelism.

With PP > 1 a finished request's sampled token reaches the scheduler only after
the pipeline drains, so its KV blocks stay allocated for extra iterations.
These methods count those iterations down and free the blocks when the
release becomes visible to the scheduler.  They apply to every request, with
or without speculative decoding.
"""

from typing import Dict, List

from frontier.logger import get_cluster_logger
from frontier.types import ClusterType


class PipelineTerminalRelease:
    """Deferred KV release of finished requests under MONOLITHIC PP."""

    def _get_monolithic_pp_pending_terminal_release_iters(self) -> Dict[int, int]:
        pending = getattr(
            self,
            "_monolithic_pp_pending_terminal_release_iters",
            None,
        )
        if pending is None:
            pending = {}
            self._monolithic_pp_pending_terminal_release_iters = pending
        return pending

    def _get_monolithic_pp_extra_terminal_release_iters(self) -> int:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return 0

        pp = int(getattr(self._replica_config, "num_pipeline_stages", 1))
        if pp <= 1:
            return 0

        # Frontier's last-stage batch-end already accounts for one terminal
        # drain iteration. Deeper PP still needs the sampled-token-return
        # boundary to reach the scheduler before blocks can be released.
        return max(pp // 2 - 1, 0)

    def _has_monolithic_pp_pending_terminal_release(self) -> bool:
        return bool(self._get_monolithic_pp_pending_terminal_release_iters())

    def _has_monolithic_pp_visible_waiting_requests(self) -> bool:
        return bool(self._request_queue or self._preempted_requests)

    def _get_monolithic_pp_iteration_start_release_threshold(self) -> int:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return 1

        pp = int(getattr(self._replica_config, "num_pipeline_stages", 1))
        if pp <= 4:
            return 1

        # Once terminal release is materialized at iteration_start, deeper
        # MONOLITHIC+PP pipelines expose the release boundary earlier than the
        # old end-of-iteration bookkeeping. The validated scheduler-visible
        # contracts are pp4->1 and pp8->2, so keep the threshold PP-depth
        # aware instead of assuming a single remaining hop for every PP size.
        return max(pp // 4, 1)

    def _advance_monolithic_pp_terminal_release_boundary(self) -> int:
        pending_release_iters = (
            self._get_monolithic_pp_pending_terminal_release_iters()
        )
        if not pending_release_iters:
            return 0

        release_visible_threshold = (
            self._get_monolithic_pp_iteration_start_release_threshold()
        )
        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )

        ready_request_ids: List[int] = []
        for request_id, remaining_iters in list(pending_release_iters.items()):
            if (
                remaining_iters <= release_visible_threshold
                and self._has_monolithic_pp_visible_waiting_requests()
                and request_id
                not in self._monolithic_pp_waiting_sensitive_release_extensions
            ):
                self._monolithic_pp_waiting_sensitive_release_extensions.add(request_id)
                pending_release_iters[request_id] = 1
                logger.debug(
                    "[VLLMv1Engine] Delaying MONOLITHIC+PP terminal release for "
                    "request %s by one extra empty iteration because waiting "
                    "requests are already visible",
                    request_id,
                )
                continue
            if remaining_iters <= 1:
                ready_request_ids.append(request_id)
                pending_release_iters.pop(request_id, None)
            else:
                pending_release_iters[request_id] = remaining_iters - 1

        if not ready_request_ids:
            if pending_release_iters:
                self._monolithic_pp_terminal_release_followup_poll_pending = True
                logger.debug(
                    "[VLLMv1Engine] Keeping MONOLITHIC+PP terminal release self-driven "
                    "with one follow-up schedule poll while pending state remains: %s",
                    dict(pending_release_iters),
                )
            return 0

        ready_request_id_set = set(ready_request_ids)
        for request_id in ready_request_ids:
            self._free_request_resources_by_id(request_id)
            self._scheduled_num_computed_tokens_by_request.pop(request_id, None)
            self._monolithic_pp_waiting_sensitive_release_extensions.discard(request_id)

        self._running_requests = [
            request
            for request in self._running_requests
            if request.id not in ready_request_id_set
        ]
        self._monolithic_pp_terminal_release_followup_poll_pending = bool(
            pending_release_iters
        ) or self._has_monolithic_pp_visible_waiting_requests()

        logger.debug(
            "[VLLMv1Engine] Released %s MONOLITHIC+PP terminal request(s) "
            "after sampled-token-return-equivalent boundary: %s",
            len(ready_request_ids),
            ready_request_ids,
        )
        return len(ready_request_ids)

    def _materialize_monolithic_pp_terminal_release_before_iteration_start(
        self,
    ) -> int:
        pending_release_iters = (
            self._get_monolithic_pp_pending_terminal_release_iters()
        )
        if not pending_release_iters:
            return 0
        if self._has_monolithic_pp_visible_waiting_requests():
            return 0

        release_visible_threshold = (
            self._get_monolithic_pp_iteration_start_release_threshold()
        )
        ready_request_ids = [
            request_id
            for request_id, remaining_iters in list(pending_release_iters.items())
            if remaining_iters <= release_visible_threshold
        ]
        if not ready_request_ids:
            return 0

        logger = get_cluster_logger(
            __name__, self._cluster_type.name if self._cluster_type else None
        )
        ready_request_id_set = set(ready_request_ids)
        for request_id in ready_request_ids:
            pending_release_iters.pop(request_id, None)
            self._free_request_resources_by_id(request_id)
            self._scheduled_num_computed_tokens_by_request.pop(request_id, None)
            self._monolithic_pp_waiting_sensitive_release_extensions.discard(
                request_id
            )

        self._running_requests = [
            request
            for request in self._running_requests
            if request.id not in ready_request_id_set
        ]
        logger.debug(
            "[VLLMv1Engine] Materialized %s MONOLITHIC+PP terminal release(s) "
            "before iteration_start because no waiting request is visible: %s",
            len(ready_request_ids),
            ready_request_ids,
        )
        return len(ready_request_ids)

    def consume_monolithic_pp_terminal_release_followup_poll(self) -> bool:
        pending = bool(
            getattr(
                self,
                "_monolithic_pp_terminal_release_followup_poll_pending",
                False,
            )
        )
        self._monolithic_pp_terminal_release_followup_poll_pending = False
        return pending

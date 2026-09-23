"""Output-wait policy for target-embedded MTP under monolithic pipeline parallelism.

A target-embedded MTP request can finish its draft tokens before the pipeline
has drained, so admitting the next request immediately would let the simulated
engine run ahead of what the real one can do.  These methods decide how long to
hold admission and terminal release, and how much visible budget to reserve.
"""

from typing import Dict, List, Optional, Tuple

from frontier.entities.batch import Batch, Request
from frontier.logger import get_cluster_logger
from frontier.spec_decode import compute_iteration_outcome, get_planned_draft_tokens
from frontier.types import ClusterType


class TargetEmbeddedMtpWaitPolicy:
    """Admission, output-wait and terminal-release timing for embedded MTP."""

    def _refresh_target_embedded_mtp_prefill_boundary_state(
        self, batch: Batch, request: Request
    ) -> None:
        metadata = batch.spec_decode_metadata
        if metadata is not None:
            for _, batch_request in enumerate(batch.requests):
                if batch_request is request:
                    # The metadata row is authoritative for this batch. A
                    # positive verify width was already recorded when metadata
                    # was built; a zero verify width means the request did not
                    # participate in this spec-decode iteration.
                    return
        if self._cluster_type != ClusterType.MONOLITHIC:
            return
        if not getattr(request, "spec_decode_enabled", False):
            return
        if not getattr(request, "spec_method_is_target_embedded_mtp", False):
            return
        if not getattr(request, "is_prefill_complete", False):
            return
        spec_decode_config = getattr(self, "_spec_decode_config", None)
        if spec_decode_config is None:
            raise ValueError("Speculative decoding config is not initialized")
        if int(getattr(request, "spec_total_iterations", 0)) == 0:
            planned_drafts = get_planned_draft_tokens(
                spec_decode_config,
                request.remaining_decode_tokens,
                iteration_index=0,
                request_id=str(request.id),
            )
            outcome = compute_iteration_outcome(
                spec_decode_config,
                request.remaining_decode_tokens,
                planned_draft_tokens=planned_drafts,
                iteration_index=0,
                request_id=str(request.id),
            )
            if planned_drafts == 0 and outcome.committed_tokens == 1:
                # Some vLLM target-embedded MTP requests emit a real prefill
                # commit row: one sampled token, no scheduled drafts. Frontier
                # already advanced that token at the prefill boundary, so only
                # the trace cursor and spec stats need to catch up here.
                request.record_spec_decode_iteration(
                    verify_tokens=outcome.verify_tokens,
                    accepted_drafts=outcome.accepted_draft_tokens,
                    rejected_drafts=outcome.rejected_draft_tokens,
                    committed_tokens=outcome.committed_tokens,
                )
            else:
                request.set_spec_next_planned_draft_tokens(planned_drafts)
                return
        if int(getattr(request, "spec_total_iterations", 0)) != 1:
            return
        request.set_spec_next_planned_draft_tokens(
            get_planned_draft_tokens(
                spec_decode_config,
                request.remaining_decode_tokens,
                iteration_index=request.spec_total_iterations,
                request_id=str(request.id),
            )
        )

    def _get_monolithic_pp_waiting_admission_delay_iters(self) -> Dict[int, int]:
        delay_iters = getattr(
            self,
            "_monolithic_pp_waiting_admission_delay_iters",
            None,
        )
        if delay_iters is None:
            delay_iters = {}
            self._monolithic_pp_waiting_admission_delay_iters = delay_iters
        return delay_iters

    def _is_target_embedded_mtp_request(self, request: Request) -> bool:
        return (
            bool(getattr(request, "spec_decode_enabled", False))
            and bool(getattr(request, "spec_method_is_target_embedded_mtp", False))
        )

    def _has_future_planned_draft_tokens(self, request: Request) -> bool:
        spec_config = getattr(
            getattr(self, "_replica_config", None),
            "speculative_decoding_config",
            None,
        )
        per_request_trace = getattr(
            spec_config,
            "_per_request_scheduled_draft_tokens_trace",
            None,
        )
        if per_request_trace is None:
            return False
        request_id = str(request.id)
        if request_id not in per_request_trace:
            raise ValueError(
                "per-request scheduled draft trace missing request_id="
                f"{request_id!r}"
            )
        next_iteration = int(getattr(request, "spec_total_iterations", 0)) + 1
        return any(int(tokens) > 0 for tokens in per_request_trace[request_id][next_iteration:])

    def _get_target_embedded_mtp_terminal_overshoot_rows(
        self,
        request: Request,
        *,
        start_iteration_index: int,
    ) -> List[Tuple[int, int, int, int, int]]:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return []
        if self._num_stages <= 1:
            return []
        if not getattr(request, "spec_decode_enabled", False):
            return []
        if not getattr(request, "spec_method_is_target_embedded_mtp", False):
            return []

        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            raise ValueError("Speculative decoding config is not initialized")
        per_request_planned_trace = getattr(
            spec_config,
            "_per_request_scheduled_draft_tokens_trace",
            None,
        )
        per_request_committed_trace = getattr(
            spec_config,
            "_per_request_committed_tokens_trace",
            None,
        )
        if per_request_planned_trace is None and per_request_committed_trace is None:
            return []
        if per_request_planned_trace is None or per_request_committed_trace is None:
            raise ValueError(
                "terminal target-embedded MTP overshoot modeling requires both "
                "per-request scheduled draft and committed token traces"
            )

        request_id = str(request.id)
        if request_id not in per_request_planned_trace:
            raise ValueError(
                "per-request scheduled draft trace missing request_id="
                f"{request_id!r}"
            )
        if request_id not in per_request_committed_trace:
            raise ValueError(
                "per-request acceptance trace missing request_id="
                f"{request_id!r}"
            )

        planned_trace = per_request_planned_trace[request_id]
        committed_trace = per_request_committed_trace[request_id]
        if len(planned_trace) != len(committed_trace):
            raise ValueError(
                "per-request MTP trace length mismatch: "
                f"request_id={request_id!r}, planned_len={len(planned_trace)}, "
                f"committed_len={len(committed_trace)}"
            )

        start_idx = int(start_iteration_index)
        if start_idx < 0:
            raise ValueError(
                f"start_iteration_index must be >= 0, got={start_idx}"
            )
        if start_idx >= len(planned_trace):
            return []

        terminal_rows: List[Tuple[int, int, int, int, int]] = []
        for idx in range(start_idx, len(planned_trace)):
            planned_drafts = int(planned_trace[idx])
            raw_committed = int(committed_trace[idx])
            if planned_drafts < 0:
                raise ValueError(
                    "terminal scheduled draft tokens must be >= 0, "
                    f"request_id={request_id!r}, iteration_index={idx}, "
                    f"got={planned_drafts}"
                )
            if raw_committed < 0:
                raise ValueError(
                    "terminal committed tokens must be >= 0, "
                    f"request_id={request_id!r}, iteration_index={idx}, "
                    f"got={raw_committed}"
                )
            trace_verify_tokens = 1 + planned_drafts
            if raw_committed > trace_verify_tokens:
                raise ValueError(
                    "terminal committed tokens cannot exceed verify window: "
                    f"request_id={request_id!r}, iteration_index={idx}, "
                    f"committed={raw_committed}, "
                    f"verify_tokens={trace_verify_tokens}"
                )
            if planned_drafts == 0 and raw_committed == 0:
                continue

            # Once the logical response is complete, vLLM's online scheduler no
            # longer replays the full forced-acceptance scheduled-draft window
            # for request latency. The diagnostic acceptance trace can still
            # contain scheduled_draft_tokens=32 for the next trace row, while
            # the clean scheduler batch log exposes only a one-token cleanup row
            # for that completed request. Model that terminal cleanup as one
            # target token and keep the raw committed count only for audit
            # provenance. Replaying the full trace window here over-extends
            # short decode-tail request latency and violates the clean online
            # metric scope.
            terminal_cleanup_verify_tokens = 1
            terminal_rows.append(
                (
                    0,
                    terminal_cleanup_verify_tokens,
                    0,
                    0,
                    raw_committed,
                )
            )
        return terminal_rows

    def _should_delay_monolithic_pp_waiting_admission_on_add(
        self, request: Request
    ) -> bool:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return False
        if self._num_stages <= 1:
            return False
        if not self._is_target_embedded_mtp_request(request):
            return False
        planned_drafts = int(getattr(request, "spec_next_planned_draft_tokens", 0))
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if planned_drafts <= 0:
            # A zero first scheduled-draft step still has PP lookahead admission
            # visibility cost when a long decode will enter later MTP draft steps.
            if num_speculative_tokens > 2:
                return False
            if int(getattr(request, "num_decode_tokens", 0)) < 128:
                return False
            if not self._has_future_planned_draft_tokens(request):
                return False
        else:
            block_size = int(getattr(self._config, "block_size", 16))
            if num_speculative_tokens >= block_size:
                # Full-block-or-wider target-embedded MTP request admission is
                # already protected by output-visible guards after it starts
                # running. Adding a pre-admission PP boundary here makes late
                # online arrivals miss the vLLM-visible scheduler slot and
                # under-batches wide verify traces relative to vLLM.
                return False
            if (
                num_speculative_tokens > 2
                and int(getattr(request, "num_prefill_tokens", 0))
                >= int(self._max_num_scheduled_tokens)
                + max(1, int(self._max_num_scheduled_tokens) // 2)
            ):
                # Long multi-chunk prefills have enough remaining prefill work
                # to expose the next PP boundary through the prefill chunks
                # themselves. Adding a separate half-block MTP admission delay
                # over-queues these arrivals and inflates p90 TTFT; keep the
                # delay for shorter prefills where r106 showed global
                # half-block skipping over-corrects TPOT tails.
                return False
        return (
            self._num_running_batches > 0
            or bool(self._running_requests)
            or bool(self._get_active_batch_request_counts())
        )

    def _add_monolithic_pp_waiting_admission_delay(
        self, request_id: int, *, wait_iters: Optional[int] = None
    ) -> None:
        resolved_wait_iters = int(
            wait_iters if wait_iters is not None else max(1, self._num_stages - 1)
        )
        if resolved_wait_iters <= 0:
            raise ValueError(
                f"wait_iters must be positive, got={resolved_wait_iters}"
            )
        delay_iters = self._get_monolithic_pp_waiting_admission_delay_iters()
        delay_iters[request_id] = max(
            int(delay_iters.get(request_id, 0)),
            resolved_wait_iters,
        )

    def _should_defer_monolithic_pp_waiting_admission(
        self, request: Request
    ) -> bool:
        delay_iters = self._get_monolithic_pp_waiting_admission_delay_iters()
        remaining = int(delay_iters.get(request.id, 0))
        if remaining <= 0:
            delay_iters.pop(request.id, None)
            return False
        if self._cluster_type != ClusterType.MONOLITHIC or self._num_stages <= 1:
            delay_iters.pop(request.id, None)
            return False
        if not self._is_target_embedded_mtp_request(request):
            delay_iters.pop(request.id, None)
            return False
        if (
            not self._running_requests
            and self._num_running_batches <= 0
            and not self._get_active_batch_request_counts()
        ):
            # No active PP work remains to provide a future output-visible
            # scheduler boundary; fail open to avoid deadlocking the queue.
            delay_iters.pop(request.id, None)
            return False
        remaining -= 1
        if remaining > 0:
            delay_iters[request.id] = remaining
        else:
            delay_iters.pop(request.id, None)
        return True

    def _get_monolithic_pp_mtp_near_full_prefill_request_ids(self) -> set[int]:
        request_ids = getattr(
            self,
            "_monolithic_pp_mtp_near_full_prefill_request_ids",
            None,
        )
        if request_ids is None:
            request_ids = set()
            self._monolithic_pp_mtp_near_full_prefill_request_ids = request_ids
        return request_ids

    def _get_monolithic_pp_mtp_single_output_wait_request_ids(self) -> set[int]:
        request_ids = getattr(
            self,
            "_monolithic_pp_mtp_single_output_wait_request_ids",
            None,
        )
        if request_ids is None:
            request_ids = set()
            self._monolithic_pp_mtp_single_output_wait_request_ids = request_ids
        return request_ids

    def _get_target_embedded_mtp_request_acceptance_ratio(
        self, request: Request
    ) -> Optional[float]:
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        if spec_config is None:
            return None
        committed_trace_map = getattr(
            spec_config,
            "_per_request_committed_tokens_trace",
            None,
        )
        scheduled_trace_map = getattr(
            spec_config,
            "_per_request_scheduled_draft_tokens_trace",
            None,
        )
        if committed_trace_map is None and scheduled_trace_map is None:
            return None
        if committed_trace_map is None or scheduled_trace_map is None:
            raise ValueError(
                "MTP request acceptance audit requires both per-request "
                "committed and scheduled-draft traces"
            )
        request_id = str(request.id)
        if request_id not in committed_trace_map:
            raise ValueError(
                "per-request acceptance trace missing request_id="
                f"{request_id!r}"
            )
        if request_id not in scheduled_trace_map:
            raise ValueError(
                "per-request scheduled draft trace missing request_id="
                f"{request_id!r}"
            )
        committed_trace = committed_trace_map[request_id]
        scheduled_trace = scheduled_trace_map[request_id]
        if len(committed_trace) != len(scheduled_trace):
            raise ValueError(
                "MTP request acceptance audit trace length mismatch: "
                f"request_id={request_id!r}, "
                f"committed_len={len(committed_trace)}, "
                f"scheduled_len={len(scheduled_trace)}"
            )
        accepted_drafts = 0
        scheduled_drafts = 0
        for committed_tokens, planned_drafts in zip(
            committed_trace,
            scheduled_trace,
        ):
            planned = int(planned_drafts)
            if planned <= 0:
                continue
            committed = int(committed_tokens)
            accepted_drafts += min(max(committed - 1, 0), planned)
            scheduled_drafts += planned
        if scheduled_drafts <= 0:
            return None
        return accepted_drafts / scheduled_drafts

    def _get_monolithic_pp_mtp_output_wait_prefill_threshold(self) -> int:
        max_scheduled_tokens = int(self._max_num_scheduled_tokens)
        block_size = int(getattr(self._config, "block_size", 16))
        headroom_tokens = 4 * block_size
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if num_speculative_tokens >= block_size:
            # Wide target-embedded MTP carries a larger PP lookahead payload than
            # the narrow-window cases that established the original four-block
            # threshold. Reserve window-proportional headroom so long chunked
            # prefill slices enter the same output-visible continuation lane
            # instead of being treated as ordinary prefill chunks.
            headroom_tokens = max(headroom_tokens, 8 * num_speculative_tokens)
        return max(1, max_scheduled_tokens - headroom_tokens)

    def _get_monolithic_pp_mtp_output_wait_iters(self) -> int:
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        if spec_config is None:
            return 2

        block_size = int(getattr(self._config, "block_size", 16))
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if num_speculative_tokens < block_size:
            return 2

        committed_trace_map = getattr(
            spec_config,
            "_per_request_committed_tokens_trace",
            None,
        )
        scheduled_trace_map = getattr(
            spec_config,
            "_per_request_scheduled_draft_tokens_trace",
            None,
        )
        accepted_drafts = 0
        scheduled_drafts = 0
        if committed_trace_map is not None or scheduled_trace_map is not None:
            if committed_trace_map is None or scheduled_trace_map is None:
                raise ValueError(
                    "MTP output-wait acceptance audit requires both "
                    "per-request committed and scheduled-draft traces"
                )
            if set(committed_trace_map.keys()) != set(scheduled_trace_map.keys()):
                raise ValueError(
                    "MTP output-wait acceptance audit trace keys mismatch"
                )
            for request_id, committed_trace in committed_trace_map.items():
                scheduled_trace = scheduled_trace_map[request_id]
                if len(committed_trace) != len(scheduled_trace):
                    raise ValueError(
                        "MTP output-wait acceptance audit trace length mismatch: "
                        f"request_id={request_id!r}, "
                        f"committed_len={len(committed_trace)}, "
                        f"scheduled_len={len(scheduled_trace)}"
                    )
                for committed_tokens, planned_drafts in zip(
                    committed_trace,
                    scheduled_trace,
                ):
                    planned = int(planned_drafts)
                    if planned <= 0:
                        continue
                    committed = int(committed_tokens)
                    accepted_drafts += min(max(committed - 1, 0), planned)
                    scheduled_drafts += planned
        else:
            committed_trace = getattr(
                spec_config,
                "_committed_tokens_trace",
                None,
            )
            scheduled_trace = getattr(
                spec_config,
                "_scheduled_draft_tokens_trace",
                None,
            )
            if committed_trace is None or scheduled_trace is None:
                return 2
            if len(committed_trace) != len(scheduled_trace):
                raise ValueError(
                    "MTP output-wait acceptance audit global trace length mismatch: "
                    f"committed_len={len(committed_trace)}, "
                    f"scheduled_len={len(scheduled_trace)}"
                )
            for committed_tokens, planned_drafts in zip(
                committed_trace,
                scheduled_trace,
            ):
                planned = int(planned_drafts)
                if planned <= 0:
                    continue
                committed = int(committed_tokens)
                accepted_drafts += min(max(committed - 1, 0), planned)
                scheduled_drafts += planned

        if scheduled_drafts <= 0:
            return 2
        acceptance_ratio = accepted_drafts / scheduled_drafts
        if acceptance_ratio >= 0.5:
            if self._has_monolithic_pp_visible_waiting_requests():
                # High-acceptance wide MTP should still preserve the extra
                # output-visible boundary while fresh prefill admissions are
                # visible. Otherwise decode continuation can consume the
                # online token budget before late-arriving prefills enter,
                # inflating TTFT tails. Once no waiting prefill/resume request
                # is visible, shorten the wait to protect short decode tails.
                return 2
            # High-acceptance wide target-embedded MTP has fewer decode
            # scheduler turns per request. A single output-visible turn is
            # enough to expose PP continuation without repeatedly holding
            # short tail requests behind terminal trace rows.
            return 1
        return 2

    def _get_monolithic_pp_mtp_output_wait_iters_for_request(
        self, request: Request
    ) -> int:
        wait_iters = self._get_monolithic_pp_mtp_output_wait_iters()
        if request.id in self._get_monolithic_pp_mtp_single_output_wait_request_ids():
            return min(wait_iters, 1)
        if self._should_apply_monolithic_pp_mtp_fractional_extra_output_wait(
            request
        ):
            counts = self._get_monolithic_pp_mtp_fractional_output_wait_counts()
            previous_count = int(counts.get(request.id, 0))
            counts[request.id] = previous_count + 1
            if previous_count == 0:
                return 0
            if previous_count % 2 == 1:
                return wait_iters
            return min(wait_iters, 1)
        if self._should_extend_monolithic_pp_mtp_long_decode_output_wait(request):
            return wait_iters + 1
        return wait_iters

    def _is_monolithic_pp_mtp_half_block_low_acceptance_request(
        self,
        request: Request,
        *,
        acceptance_ratio_limit: float,
    ) -> bool:
        block_size = int(getattr(self._config, "block_size", 16))
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if num_speculative_tokens < max(1, block_size // 2):
            return False
        if num_speculative_tokens >= block_size:
            return False

        acceptance_ratio = self._get_target_embedded_mtp_request_acceptance_ratio(
            request
        )
        if acceptance_ratio is None:
            return False
        return acceptance_ratio < acceptance_ratio_limit

    def _is_monolithic_pp_mtp_mid_prefill_request(
        self,
        request: Request,
    ) -> bool:
        block_size = int(getattr(self._config, "block_size", 16))
        num_prefill_tokens = int(getattr(request, "num_prefill_tokens", 0))
        max_scheduled_tokens = int(self._max_num_scheduled_tokens)
        if num_prefill_tokens < max(1, max_scheduled_tokens - 4 * block_size):
            return False
        if num_prefill_tokens > (
            max_scheduled_tokens + max(1, max_scheduled_tokens // 2)
        ):
            return False
        return True

    def _should_apply_monolithic_pp_mtp_fractional_extra_output_wait(
        self,
        request: Request,
    ) -> bool:
        if not self._is_target_embedded_mtp_request(request):
            return False
        if not self._is_monolithic_pp_mtp_mid_prefill_request(request):
            return False

        block_size = int(getattr(self._config, "block_size", 16))
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if num_speculative_tokens < max(1, block_size // 2):
            return False
        if num_speculative_tokens >= block_size:
            return False

        acceptance_ratio = self._get_target_embedded_mtp_request_acceptance_ratio(
            request
        )
        if acceptance_ratio is None:
            return False

        block_size = int(getattr(self._config, "block_size", 16))
        num_decode_tokens = int(getattr(request, "num_decode_tokens", 0))
        if num_decode_tokens < 16 * block_size:
            return False
        if num_decode_tokens < 24 * block_size:
            if acceptance_ratio >= 0.5:
                return False
        else:
            if acceptance_ratio >= 0.55:
                return False

        # The v8/a0.3 request-level RCA shows that medium-length decode tails
        # and long decode tails need about one and a half PP output-visible
        # wait turns after the first visible decode result. Skip the first
        # wait so TTFT remains a prefill/first-token metric, then alternate
        # the regular low-acceptance wait with a single-turn wait. This keeps
        # the correction as a scheduler visibility family rather than an
        # op-runtime calibration scale.
        return True

    def _should_extend_monolithic_pp_mtp_long_decode_output_wait(
        self, request: Request
    ) -> bool:
        if not self._is_monolithic_pp_mtp_half_block_low_acceptance_request(
            request,
            acceptance_ratio_limit=0.25,
        ):
            return False
        if not self._is_monolithic_pp_mtp_mid_prefill_request(request):
            return False

        block_size = int(getattr(self._config, "block_size", 16))
        num_decode_tokens = int(getattr(request, "num_decode_tokens", 0))
        if num_decode_tokens < 24 * block_size:
            return False

        # Low-acceptance half-block MTP keeps many long decode continuations
        # alive after a near-full prefill admission. vLLM exposes an additional
        # PP-visible output boundary for the very long decode tail; model that
        # boundary as one extra scheduler wait turn instead of
        # hiding the residual in compute calibration scale.
        return True

    def _record_monolithic_pp_mtp_near_full_prefill_slices(self, batch: Batch) -> None:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return
        if self._num_stages <= 1:
            return
        near_full_prefill_threshold = (
            self._get_monolithic_pp_mtp_output_wait_prefill_threshold()
        )
        for request, num_tokens in zip(batch.requests, batch.num_tokens):
            if getattr(request, "is_prefill_complete", False):
                continue
            if not getattr(request, "spec_decode_enabled", False):
                continue
            if not getattr(request, "spec_method_is_target_embedded_mtp", False):
                continue
            if int(num_tokens) < near_full_prefill_threshold:
                if not self._should_record_monolithic_pp_mtp_subthreshold_single_wait_prefill(
                    request,
                    num_tokens=int(num_tokens),
                    near_full_prefill_threshold=near_full_prefill_threshold,
                ):
                    if not self._should_record_monolithic_pp_mtp_subthreshold_long_decode_prefill(
                        request,
                        num_tokens=int(num_tokens),
                        near_full_prefill_threshold=near_full_prefill_threshold,
                    ):
                        continue
                else:
                    self._get_monolithic_pp_mtp_single_output_wait_request_ids().add(
                        request.id
                    )
            self._get_monolithic_pp_mtp_near_full_prefill_request_ids().add(
                request.id
            )

    def _should_record_monolithic_pp_mtp_subthreshold_single_wait_prefill(
        self,
        request: Request,
        *,
        num_tokens: int,
        near_full_prefill_threshold: int,
    ) -> bool:
        block_size = int(getattr(self._config, "block_size", 16))
        spec_config = getattr(self, "_spec_decode_config", None)
        if spec_config is None:
            spec_config = getattr(
                getattr(self, "_replica_config", None),
                "speculative_decoding_config",
                None,
            )
        num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0)
        )
        if num_speculative_tokens >= block_size:
            return False

        max_scheduled_tokens = int(self._max_num_scheduled_tokens)
        min_subthreshold_tokens = max(
            1,
            max_scheduled_tokens - 8 * block_size,
        )
        if int(num_tokens) < min_subthreshold_tokens:
            return False
        if int(num_tokens) >= int(near_full_prefill_threshold):
            return False
        if int(getattr(request, "num_prefill_tokens", 0)) < (
            max_scheduled_tokens + max(1, max_scheduled_tokens // 2)
        ):
            return False
        if int(getattr(request, "num_decode_tokens", 0)) > 4 * block_size:
            return False

        acceptance_ratio = self._get_target_embedded_mtp_request_acceptance_ratio(
            request
        )
        if acceptance_ratio is None:
            return False
        if acceptance_ratio >= 0.5:
            return False

        # Low-acceptance narrow MTP preserves more decode turns than the
        # high-acceptance case, so a two-turn output wait over-delays the
        # terminal request tail. A single PP-visible wait models the missing
        # output boundary without absorbing the residual into compute scale.
        return True

    def _should_record_monolithic_pp_mtp_subthreshold_long_decode_prefill(
        self,
        request: Request,
        *,
        num_tokens: int,
        near_full_prefill_threshold: int,
    ) -> bool:
        block_size = int(getattr(self._config, "block_size", 16))
        max_scheduled_tokens = int(self._max_num_scheduled_tokens)
        min_subthreshold_tokens = max(
            1,
            max_scheduled_tokens - 8 * block_size,
        )
        if int(num_tokens) < min_subthreshold_tokens:
            return False
        if int(num_tokens) >= int(near_full_prefill_threshold):
            return False
        if not (
            self._should_apply_monolithic_pp_mtp_fractional_extra_output_wait(
                request
            )
            or self._should_extend_monolithic_pp_mtp_long_decode_output_wait(
                request
            )
        ):
            return False

        # These subthreshold prefill slices are close enough to the online
        # max-token boundary to expose the same PP output-visible behavior as
        # near-full chunks, but only for the low-acceptance half-block
        # medium/long decode slices identified by the request-level TPOT RCA.
        return True

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

    def _advance_monolithic_pp_terminal_release_boundary(self) -> None:
        pending_release_iters = (
            self._get_monolithic_pp_pending_terminal_release_iters()
        )
        if not pending_release_iters:
            return

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
            return

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

    def _materialize_monolithic_pp_terminal_release_before_iteration_start(
        self,
    ) -> None:
        pending_release_iters = (
            self._get_monolithic_pp_pending_terminal_release_iters()
        )
        if not pending_release_iters:
            return
        if self._has_monolithic_pp_visible_waiting_requests():
            return

        release_visible_threshold = (
            self._get_monolithic_pp_iteration_start_release_threshold()
        )
        ready_request_ids = [
            request_id
            for request_id, remaining_iters in list(pending_release_iters.items())
            if remaining_iters <= release_visible_threshold
        ]
        if not ready_request_ids:
            return

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

    def _get_monolithic_pp_mtp_output_wait_request_ids(self) -> set[int]:
        request_ids = getattr(
            self,
            "_monolithic_pp_mtp_output_wait_request_ids",
            None,
        )
        if request_ids is None:
            request_ids = set()
            self._monolithic_pp_mtp_output_wait_request_ids = request_ids
        return request_ids

    def _get_monolithic_pp_mtp_output_wait_remaining_iters(self) -> Dict[int, int]:
        remaining_iters = getattr(
            self,
            "_monolithic_pp_mtp_output_wait_remaining_iters",
            None,
        )
        if remaining_iters is None:
            remaining_iters = {}
            self._monolithic_pp_mtp_output_wait_remaining_iters = remaining_iters
        return remaining_iters

    def _get_monolithic_pp_mtp_fractional_output_wait_counts(self) -> Dict[int, int]:
        counts = getattr(
            self,
            "_monolithic_pp_mtp_fractional_output_wait_counts",
            None,
        )
        if counts is None:
            counts = {}
            self._monolithic_pp_mtp_fractional_output_wait_counts = counts
        return counts

    def _add_monolithic_pp_mtp_output_wait(
        self, request_id: int, *, wait_iters: int = 2
    ) -> None:
        if wait_iters <= 0:
            raise ValueError(f"wait_iters must be positive, got={wait_iters}")
        self._get_monolithic_pp_mtp_output_wait_request_ids().add(request_id)
        remaining_iters = self._get_monolithic_pp_mtp_output_wait_remaining_iters()
        remaining_iters[request_id] = max(
            int(remaining_iters.get(request_id, 0)),
            int(wait_iters),
        )

    def _should_apply_monolithic_pp_mtp_output_wait(
        self, request: Request
    ) -> bool:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return False
        if self._num_stages <= 1:
            return False
        if not getattr(request, "spec_decode_enabled", False):
            return False
        if not getattr(request, "spec_method_is_target_embedded_mtp", False):
            return False
        if not getattr(request, "is_prefill_complete", False):
            return False
        if (
            request.id
            not in self._get_monolithic_pp_mtp_near_full_prefill_request_ids()
        ):
            return False
        processed_decode_tokens = int(
            getattr(request, "num_processed_decode_tokens", 0)
        )
        if processed_decode_tokens <= 0:
            return False
        if self._get_monolithic_pp_mtp_output_wait_iters() == 1:
            block_size = int(getattr(self._config, "block_size", 16))
            remaining_decode_tokens = (
                int(getattr(request, "num_decode_tokens", 0))
                - processed_decode_tokens
            )
            if remaining_decode_tokens <= block_size:
                # High-acceptance wide-MTP short tails have no useful future
                # prefill visibility to protect once the request is within the
                # final block-sized decode window. Skipping this idle turn
                # avoids a terminal PP output-wait residual without changing
                # CUDA op calibration.
                return False
        return True

    def _has_monolithic_pp_mtp_output_wait(self) -> bool:
        return bool(self._get_monolithic_pp_mtp_output_wait_request_ids())

    def _should_reserve_monolithic_pp_mtp_visible_budget(
        self, request: Request
    ) -> bool:
        if self._cluster_type != ClusterType.MONOLITHIC:
            return False
        if self._num_stages <= 1:
            return False
        if not self._is_target_embedded_mtp_request(request):
            return False
        if not getattr(request, "is_prefill_complete", False):
            return False
        block_size = int(getattr(self._config, "block_size", 16))
        active_verify_window_tokens = int(
            getattr(request, "spec_current_verify_tokens", 0)
        )
        if active_verify_window_tokens <= 0:
            spec_config = getattr(self, "_spec_decode_config", None)
            if spec_config is None:
                spec_config = getattr(
                    getattr(self, "_replica_config", None),
                    "speculative_decoding_config",
                    None,
                )
            active_verify_window_tokens = int(
                getattr(spec_config, "num_speculative_tokens", 0)
            )
        if active_verify_window_tokens < block_size:
            # Narrow target-embedded MTP verify windows do not consume a full
            # cache block of output-visible scheduler budget. Reserving a
            # synthetic block-sized budget for them under MONOLITHIC+PP
            # under-batches waiting prefill relative to vLLM clean traces.
            return False
        return self._has_monolithic_pp_visible_waiting_requests()

    def _get_monolithic_pp_mtp_visible_budget_reservation_tokens(
        self, request: Request, token_budget: int
    ) -> int:
        if not self._should_reserve_monolithic_pp_mtp_visible_budget(request):
            return 0
        reserved_tokens = max(
            int(getattr(request, "spec_current_verify_tokens", 1)),
            self._get_request_next_num_tokens(request),
        )
        return min(max(reserved_tokens, 0), token_budget)

    def _clear_monolithic_pp_mtp_output_wait(self) -> None:
        request_ids = self._get_monolithic_pp_mtp_output_wait_request_ids()
        remaining_iters = self._get_monolithic_pp_mtp_output_wait_remaining_iters()
        if not remaining_iters:
            request_ids.clear()
            return
        next_waiting_request_ids: set[int] = set()
        for request_id in list(request_ids):
            remaining = int(remaining_iters.get(request_id, 1)) - 1
            if remaining > 0:
                remaining_iters[request_id] = remaining
                next_waiting_request_ids.add(request_id)
            else:
                remaining_iters.pop(request_id, None)
        request_ids.clear()
        request_ids.update(next_waiting_request_ids)

    def consume_monolithic_pp_mtp_output_wait_followup_poll(self) -> bool:
        pending = bool(
            getattr(
                self,
                "_monolithic_pp_mtp_output_wait_followup_poll_pending",
                False,
            )
        )
        self._monolithic_pp_mtp_output_wait_followup_poll_pending = False
        return pending

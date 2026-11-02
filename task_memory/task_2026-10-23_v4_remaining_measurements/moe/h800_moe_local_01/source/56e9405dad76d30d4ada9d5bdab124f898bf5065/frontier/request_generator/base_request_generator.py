from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from frontier.config import BaseRequestGeneratorConfig
from frontier.entities import Request, RequestRoundPlan


class BaseRequestGenerator(ABC):

    def __init__(self, config: BaseRequestGeneratorConfig):
        self.config = config
        self._thinking_mode_enabled = False
        self._thinking_depth = 1
        self._tool_call_latency = 0.001
        self._thinking_round_prefill_tokens: Optional[List[int]] = None
        self._thinking_round_decode_tokens: Optional[List[int]] = None

    def configure_thinking_mode(
        self,
        *,
        enable_thinking_mode: bool,
        thinking_depth: int,
        tool_call_latency: float,
        thinking_round_prefill_tokens: Optional[Sequence[int]] = None,
        thinking_round_decode_tokens: Optional[Sequence[int]] = None,
    ) -> None:
        if thinking_depth < 1:
            raise ValueError(f"thinking_depth must be >= 1, got {thinking_depth}")
        if tool_call_latency < 0:
            raise ValueError(
                f"tool_call_latency must be >= 0, got {tool_call_latency}"
            )
        if not enable_thinking_mode and (
            thinking_depth > 1
            or thinking_round_prefill_tokens is not None
            or thinking_round_decode_tokens is not None
        ):
            raise ValueError(
                "Multi-round thinking requests require enable_thinking_mode=True."
            )

        has_explicit_hidden_rounds = (
            thinking_round_prefill_tokens is not None
            or thinking_round_decode_tokens is not None
        )
        if has_explicit_hidden_rounds:
            if (
                thinking_round_prefill_tokens is None
                or thinking_round_decode_tokens is None
            ):
                raise ValueError(
                    "thinking_round_prefill_tokens and thinking_round_decode_tokens "
                    "must be provided together."
                )
            expected_hidden_rounds = thinking_depth - 1
            if len(thinking_round_prefill_tokens) != expected_hidden_rounds:
                raise ValueError(
                    "thinking_round_prefill_tokens length must equal "
                    f"thinking_depth - 1 ({expected_hidden_rounds})."
                )
            if len(thinking_round_decode_tokens) != expected_hidden_rounds:
                raise ValueError(
                    "thinking_round_decode_tokens length must equal "
                    f"thinking_depth - 1 ({expected_hidden_rounds})."
                )

        self._thinking_mode_enabled = enable_thinking_mode
        self._thinking_depth = thinking_depth if enable_thinking_mode else 1
        self._tool_call_latency = tool_call_latency
        self._thinking_round_prefill_tokens = (
            [int(value) for value in thinking_round_prefill_tokens]
            if thinking_round_prefill_tokens is not None
            else None
        )
        self._thinking_round_decode_tokens = (
            [int(value) for value in thinking_round_decode_tokens]
            if thinking_round_decode_tokens is not None
            else None
        )

    def _generate_implicit_hidden_round_plans(
        self,
        final_prefill_tokens: int,
        final_decode_tokens: int,
    ) -> List[RequestRoundPlan]:
        return [
            RequestRoundPlan(final_prefill_tokens, final_decode_tokens)
            for _ in range(max(self._thinking_depth - 1, 0))
        ]

    def _build_hidden_round_plans(
        self,
        *,
        final_prefill_tokens: int,
        final_decode_tokens: int,
    ) -> List[RequestRoundPlan]:
        if not self._thinking_mode_enabled or self._thinking_depth == 1:
            return []

        if self._thinking_round_prefill_tokens is not None:
            return [
                RequestRoundPlan(prefill_tokens, decode_tokens)
                for prefill_tokens, decode_tokens in zip(
                    self._thinking_round_prefill_tokens,
                    self._thinking_round_decode_tokens,
                )
            ]

        return self._generate_implicit_hidden_round_plans(
            final_prefill_tokens=final_prefill_tokens,
            final_decode_tokens=final_decode_tokens,
        )

    def _build_request(
        self,
        *,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        priority: int = 0,
        block_hash_ids=None,
        session_id: Optional[int] = None,
        cohort: Optional[str] = None,
    ) -> Request:
        hidden_round_plans = self._build_hidden_round_plans(
            final_prefill_tokens=num_prefill_tokens,
            final_decode_tokens=num_decode_tokens,
        )
        thinking_round_plans = None
        if hidden_round_plans:
            thinking_round_plans = hidden_round_plans + [
                RequestRoundPlan(num_prefill_tokens, num_decode_tokens)
            ]

        return Request(
            arrived_at=arrived_at,
            num_prefill_tokens=int(num_prefill_tokens),
            num_decode_tokens=int(num_decode_tokens),
            priority=priority,
            block_hash_ids=block_hash_ids,
            session_id=session_id,
            cohort=cohort,
            thinking_depth=self._thinking_depth if self._thinking_mode_enabled else 1,
            tool_call_latency=self._tool_call_latency,
            thinking_round_plans=thinking_round_plans,
        )

    @abstractmethod
    def generate_requests(self) -> List[Request]:
        pass

    def generate(self) -> List[Request]:
        requests = self.generate_requests()
        self.config.num_decode_bound_requests = sum(
            1 for request in requests if request.num_decode_tokens > 0
        )
        return requests
